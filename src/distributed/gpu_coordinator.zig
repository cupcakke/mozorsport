const std = @import("std");
const nccl = @import("nccl_bindings.zig");
const Allocator = std.mem.Allocator;

fn constOpaquePtrFrom(value: anytype) !*const anyopaque {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .pointer => |ptr_info| switch (ptr_info.size) {
            .one => @ptrCast(value),
            .slice => blk: {
                if (value.len == 0) {
                    return error.EmptyBuffer;
                }
                break :blk @ptrCast(&value[0]);
            },
            .many, .c => @compileError("constOpaquePtrFrom: expected single pointer or slice, got unbounded pointer type " ++ @typeName(T)),
        },
        else => @compileError("constOpaquePtrFrom: expected pointer or slice, got " ++ @typeName(T)),
    };
}

fn opaquePtrFrom(value: anytype) !*anyopaque {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .pointer => |ptr_info| switch (ptr_info.size) {
            .one => blk: {
                if (ptr_info.is_const) {
                    @compileError("opaquePtrFrom: expected mutable pointer, got " ++ @typeName(T));
                }
                break :blk @ptrCast(value);
            },
            .slice => blk: {
                if (ptr_info.is_const) {
                    @compileError("opaquePtrFrom: expected mutable slice, got " ++ @typeName(T));
                }
                if (value.len == 0) {
                    return error.EmptyBuffer;
                }
                break :blk @ptrCast(&value[0]);
            },
            .many, .c => @compileError("opaquePtrFrom: expected single pointer or slice, got unbounded pointer type " ++ @typeName(T)),
        },
        else => @compileError("opaquePtrFrom: expected pointer or slice, got " ++ @typeName(T)),
    };
}

fn checkCuda(err: nccl.CudaError, comptime tag: []const u8, fail_error: anyerror) !void {
    if (err != .cudaSuccess) {
        const err_str = nccl.cudaGetErrorString(err);
        std.debug.print("CUDA error [{s}]: {s}\n", .{ tag, err_str });
        return fail_error;
    }
}

fn checkNccl(err: nccl.ncclResult_t, comptime tag: []const u8, fail_error: anyerror) !void {
    if (err != .ncclSuccess) {
        const err_str = nccl.ncclGetErrorString(err);
        std.debug.print("NCCL error [{s}]: {s}\n", .{ tag, err_str });
        return fail_error;
    }
}

fn logCudaFailure(err: nccl.CudaError, comptime tag: []const u8) void {
    if (err != .cudaSuccess) {
        const err_str = nccl.cudaGetErrorString(err);
        std.debug.print("CUDA cleanup error [{s}]: {s}\n", .{ tag, err_str });
    }
}

fn logNcclFailure(err: nccl.ncclResult_t, comptime tag: []const u8) void {
    if (err != .ncclSuccess) {
        const err_str = nccl.ncclGetErrorString(err);
        std.debug.print("NCCL cleanup error [{s}]: {s}\n", .{ tag, err_str });
    }
}

pub const GPUCoordinator = struct {
    allocator: Allocator,
    world_size: usize,
    rank: usize,
    device_id: i32,
    nccl_comm: ?*nccl.ncclComm,
    cuda_stream: ?*anyopaque,
    barrier_buffer: ?*anyopaque,
    device_allocations: std.AutoHashMap(usize, usize),
    current_device_allocation_bytes: usize,
    peak_device_allocation_bytes: usize,

    pub fn init(allocator: Allocator, world_size: usize, rank: usize, local_rank: usize, nccl_id: nccl.ncclUniqueId) !GPUCoordinator {
        if (world_size == 0) {
            return error.InvalidWorldSize;
        }
        if (rank >= world_size) {
            return error.InvalidRank;
        }
        if (world_size > @as(usize, @intCast(std.math.maxInt(c_int)))) {
            return error.WorldSizeTooLarge;
        }
        if (rank > @as(usize, @intCast(std.math.maxInt(c_int)))) {
            return error.RankTooLarge;
        }

        var device_count: c_int = 0;
        try checkCuda(nccl.cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error.CudaGetDeviceCountFailed);
        if (device_count <= 0) {
            return error.InsufficientGPUs;
        }

        const local_device_count: usize = @intCast(device_count);
        if (local_rank >= local_device_count) {
            return error.LocalRankExceedsDeviceCount;
        }
        const device_id_usize: usize = local_rank;
        if (device_id_usize > @as(usize, @intCast(std.math.maxInt(i32)))) {
            return error.DeviceIdOutOfRange;
        }
        const device_id: i32 = @intCast(device_id_usize);

        try checkCuda(nccl.cudaSetDevice(device_id), "cudaSetDevice", error.CudaSetDeviceFailed);

        var nccl_comm_local: *nccl.ncclComm = undefined;
        try checkNccl(
            nccl.ncclCommInitRank(&nccl_comm_local, @intCast(world_size), nccl_id, @intCast(rank)),
            "ncclCommInitRank",
            error.NCCLCommInitFailed,
        );
        errdefer logNcclFailure(nccl.ncclCommDestroy(nccl_comm_local), "ncclCommDestroy(init rollback)");

        var cuda_stream_local: *anyopaque = undefined;
        try checkCuda(nccl.cudaStreamCreate(&cuda_stream_local), "cudaStreamCreate", error.CudaStreamCreateFailed);
        errdefer logCudaFailure(nccl.cudaStreamDestroy(cuda_stream_local), "cudaStreamDestroy(init rollback)");

        var barrier_buf_local: ?*anyopaque = null;
        try checkCuda(nccl.cudaMalloc(&barrier_buf_local, @sizeOf(f32)), "cudaMalloc(barrier)", error.CudaMallocFailed);
        const barrier_buffer = barrier_buf_local orelse return error.CudaMallocFailed;
        errdefer logCudaFailure(nccl.cudaFree(barrier_buffer), "cudaFree(barrier init rollback)");

        try checkCuda(nccl.cudaMemset(barrier_buffer, 0, @sizeOf(f32)), "cudaMemset(barrier)", error.CudaMemsetFailed);
        try checkCuda(nccl.cudaDeviceSynchronize(), "cudaDeviceSynchronize(barrier init)", error.CudaSynchronizeFailed);

        return GPUCoordinator{
            .allocator = allocator,
            .world_size = world_size,
            .rank = rank,
            .device_id = device_id,
            .nccl_comm = nccl_comm_local,
            .cuda_stream = cuda_stream_local,
            .barrier_buffer = barrier_buffer,
            .device_allocations = std.AutoHashMap(usize, usize).init(allocator),
            .current_device_allocation_bytes = 0,
            .peak_device_allocation_bytes = 0,
        };
    }

    pub fn deinit(self: *GPUCoordinator) void {
        logCudaFailure(nccl.cudaSetDevice(self.device_id), "cudaSetDevice(deinit)");

        if (self.cuda_stream) |stream| {
            logCudaFailure(nccl.cudaStreamSynchronize(stream), "cudaStreamSynchronize(deinit)");
        }

        if (self.barrier_buffer) |buffer| {
            logCudaFailure(nccl.cudaFree(buffer), "cudaFree(barrier)");
            self.barrier_buffer = null;
        }

        if (self.nccl_comm) |comm| {
            logNcclFailure(nccl.ncclCommFinalize(comm), "ncclCommFinalize");
            logNcclFailure(nccl.ncclCommDestroy(comm), "ncclCommDestroy");
            self.nccl_comm = null;
        }

        if (self.cuda_stream) |stream| {
            logCudaFailure(nccl.cudaStreamDestroy(stream), "cudaStreamDestroy");
            self.cuda_stream = null;
        }
        self.device_allocations.deinit();
        self.current_device_allocation_bytes = 0;
    }

    fn setDevice(self: *GPUCoordinator) !void {
        try checkCuda(nccl.cudaSetDevice(self.device_id), "cudaSetDevice", error.CudaSetDeviceFailed);
    }

    fn requireComm(self: *GPUCoordinator) !*nccl.ncclComm {
        return self.nccl_comm orelse return error.CoordinatorNotInitialized;
    }

    fn requireStream(self: *GPUCoordinator) !*anyopaque {
        return self.cuda_stream orelse return error.CoordinatorNotInitialized;
    }

    pub fn allocDeviceMemory(self: *GPUCoordinator, size: usize) !*anyopaque {
        if (size == 0) {
            return error.InvalidAllocationSize;
        }

        try self.setDevice();

        var dev_ptr: ?*anyopaque = null;
        try checkCuda(nccl.cudaMalloc(&dev_ptr, size), "cudaMalloc", error.CudaMallocFailed);
        const ptr = dev_ptr orelse return error.CudaMallocFailed;
        const key = @intFromPtr(ptr);
        self.device_allocations.put(key, size) catch {
            logCudaFailure(nccl.cudaFree(ptr), "cudaFree(alloc tracking rollback)");
            return error.OutOfMemory;
        };
        self.current_device_allocation_bytes = std.math.add(usize, self.current_device_allocation_bytes, size) catch std.math.maxInt(usize);
        self.peak_device_allocation_bytes = @max(self.peak_device_allocation_bytes, self.current_device_allocation_bytes);
        return ptr;
    }

    pub fn freeDeviceMemory(self: *GPUCoordinator, ptr: ?*anyopaque) void {
        self.setDevice() catch |err| {
            std.debug.print("CUDA cleanup error [cudaSetDevice(freeDeviceMemory)]: {}\n", .{err});
            return;
        };
        if (ptr) |p| {
            const key = @intFromPtr(p);
            if (self.device_allocations.fetchRemove(key)) |entry| {
                if (self.current_device_allocation_bytes >= entry.value) {
                    self.current_device_allocation_bytes -= entry.value;
                } else {
                    self.current_device_allocation_bytes = 0;
                }
            }
            logCudaFailure(nccl.cudaFree(p), "cudaFree");
        }
    }

    fn doMemcpy(
        self: *GPUCoordinator,
        dst_ptr: *anyopaque,
        src_ptr: *const anyopaque,
        size: usize,
        kind: c_int,
        comptime tag: []const u8,
    ) !void {
        if (size == 0) {
            return;
        }

        _ = try self.requireStream();
        try self.setDevice();
        try checkCuda(nccl.cudaMemcpy(dst_ptr, src_ptr, size, kind), tag, error.CudaMemcpyFailed);
    }

    pub fn copyHostToDevice(self: *GPUCoordinator, dst: anytype, src: anytype, size: usize) !void {
        if (size == 0) {
            return;
        }

        const dst_ptr = try opaquePtrFrom(dst);
        const src_ptr = try constOpaquePtrFrom(src);
        try self.doMemcpy(dst_ptr, src_ptr, size, nccl.cudaMemcpyKind.cudaMemcpyHostToDevice, "cudaMemcpyHostToDevice");
    }

    pub fn copyDeviceToHost(self: *GPUCoordinator, dst: anytype, src: anytype, size: usize) !void {
        if (size == 0) {
            return;
        }

        const dst_ptr = try opaquePtrFrom(dst);
        const src_ptr = try constOpaquePtrFrom(src);
        try self.doMemcpy(dst_ptr, src_ptr, size, nccl.cudaMemcpyKind.cudaMemcpyDeviceToHost, "cudaMemcpyDeviceToHost");
    }

    pub fn copyDeviceToDevice(self: *GPUCoordinator, dst: anytype, src: anytype, size: usize) !void {
        if (size == 0) {
            return;
        }

        const dst_ptr = try opaquePtrFrom(dst);
        const src_ptr = try constOpaquePtrFrom(src);
        try self.doMemcpy(dst_ptr, src_ptr, size, nccl.cudaMemcpyKind.cudaMemcpyDeviceToDevice, "cudaMemcpyDeviceToDevice");
    }

    fn doAllReduce(
        self: *GPUCoordinator,
        send_buf: *const anyopaque,
        recv_buf: *anyopaque,
        count: usize,
        dtype: nccl.ncclDataType_t,
        op: nccl.ncclRedOp_t,
        comptime tag: []const u8,
    ) !void {
        if (count == 0) {
            return;
        }

        const comm = try self.requireComm();
        const stream = try self.requireStream();
        try self.setDevice();

        try checkNccl(
            nccl.ncclAllReduce(send_buf, recv_buf, count, dtype, op, comm, stream),
            tag,
            error.NCCLAllReduceFailed,
        );
    }

    pub fn allReduceFloat32(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclSum, "ncclAllReduceFloat32Sum");
    }

    pub fn allReduceFloat16(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat16, .ncclSum, "ncclAllReduceFloat16Sum");
    }

    /// In-place average across ranks (NCCL ncclAvg op, available since 2.10).
    /// The result on every rank equals (sum of inputs) / world_size, computed
    /// without leaving the GPU.
    pub fn allReduceFloat16Avg(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat16, .ncclAvg, "ncclAllReduceFloat16Avg");
    }

    pub fn allReduceFloat32Avg(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclAvg, "ncclAllReduceFloat32Avg");
    }

    pub fn allReduceFloat32Max(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclMax, "ncclAllReduceFloat32Max");
    }

    pub fn allReduceFloat32Min(self: *GPUCoordinator, send_buf: *const anyopaque, recv_buf: *anyopaque, count: usize) !void {
        try self.doAllReduce(send_buf, recv_buf, count, .ncclFloat32, .ncclMin, "ncclAllReduceFloat32Min");
    }

    fn doBroadcast(
        self: *GPUCoordinator,
        send_buf: *const anyopaque,
        recv_buf: *anyopaque,
        count: usize,
        dtype: nccl.ncclDataType_t,
        root: usize,
        comptime tag: []const u8,
    ) !void {
        if (count == 0) {
            return;
        }
        if (root >= self.world_size) {
            return error.InvalidRootRank;
        }
        if (root > @as(usize, @intCast(std.math.maxInt(c_int)))) {
            return error.InvalidRootRank;
        }

        const comm = try self.requireComm();
        const stream = try self.requireStream();
        try self.setDevice();

        try checkNccl(
            nccl.ncclBroadcast(send_buf, recv_buf, count, dtype, @intCast(root), comm, stream),
            tag,
            error.NCCLBroadcastFailed,
        );
    }

    pub fn broadcastFloat32(self: *GPUCoordinator, buf: *anyopaque, count: usize, root: usize) !void {
        try self.doBroadcast(buf, buf, count, .ncclFloat32, root, "ncclBroadcastFloat32");
    }

    pub fn broadcastFloat16(self: *GPUCoordinator, buf: *anyopaque, count: usize, root: usize) !void {
        try self.doBroadcast(buf, buf, count, .ncclFloat16, root, "ncclBroadcastFloat16");
    }

    pub fn allGatherFloat32(
        self: *GPUCoordinator,
        send_buf: *const anyopaque,
        recv_buf: *anyopaque,
        send_count: usize,
    ) !void {
        if (send_count == 0) {
            return;
        }

        const comm = try self.requireComm();
        const stream = try self.requireStream();
        try self.setDevice();

        try checkNccl(
            nccl.ncclAllGather(send_buf, recv_buf, send_count, .ncclFloat32, comm, stream),
            "ncclAllGatherFloat32",
            error.NCCLAllGatherFailed,
        );
    }

    pub fn reduceScatterFloat32(
        self: *GPUCoordinator,
        send_buf: *const anyopaque,
        recv_buf: *anyopaque,
        recv_count: usize,
    ) !void {
        if (recv_count == 0) {
            return;
        }

        const comm = try self.requireComm();
        const stream = try self.requireStream();
        try self.setDevice();

        try checkNccl(
            nccl.ncclReduceScatter(send_buf, recv_buf, recv_count, .ncclFloat32, .ncclSum, comm, stream),
            "ncclReduceScatterFloat32Sum",
            error.NCCLReduceScatterFailed,
        );
    }

    pub fn synchronize(self: *GPUCoordinator) !void {
        const stream = try self.requireStream();
        try self.setDevice();

        try checkCuda(nccl.cudaStreamSynchronize(stream), "cudaStreamSynchronize", error.CudaSynchronizeFailed);

        const pending_err = nccl.cudaGetLastError();
        try checkCuda(pending_err, "cudaGetLastError", error.CudaSynchronizeFailed);
    }

    pub fn deviceSynchronize(self: *GPUCoordinator) !void {
        try self.setDevice();
        try checkCuda(nccl.cudaDeviceSynchronize(), "cudaDeviceSynchronize", error.CudaSynchronizeFailed);

        const pending_err = nccl.cudaGetLastError();
        try checkCuda(pending_err, "cudaGetLastError(device)", error.CudaSynchronizeFailed);
    }

    pub fn barrier(self: *GPUCoordinator) !void {
        const buf = self.barrier_buffer orelse return error.CoordinatorNotInitialized;
        const comm = try self.requireComm();
        const stream = try self.requireStream();

        try self.setDevice();

        try checkNccl(
            nccl.ncclAllReduce(buf, buf, 1, .ncclFloat32, .ncclSum, comm, stream),
            "ncclAllReduceBarrier",
            error.NCCLAllReduceFailed,
        );
        try checkCuda(nccl.cudaStreamSynchronize(stream), "cudaStreamSynchronize(barrier)", error.CudaSynchronizeFailed);
    }

    pub fn currentTrackedDeviceBytes(self: *const GPUCoordinator) usize {
        return self.current_device_allocation_bytes;
    }

    pub fn peakTrackedDeviceBytes(self: *const GPUCoordinator) usize {
        return self.peak_device_allocation_bytes;
    }

    pub fn isRoot(self: *const GPUCoordinator) bool {
        return self.rank == 0;
    }
};

