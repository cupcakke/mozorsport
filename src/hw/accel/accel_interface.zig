const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const WeightKind = enum {
    weights_s,
    weights_t,
    velocity_s,
    velocity_t,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        if (gpu_enabled) {
            futhark.gpu_stub.set_device(cfg, "");
            futhark.gpu_stub.set_default_group_size(cfg, 256);
            futhark.gpu_stub.set_default_num_groups(cfg, 128);
            futhark.gpu_stub.set_default_tile_size(cfg, 32);
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkContextFailed;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx, .cfg = cfg };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            const err_str = futhark.futhark_context_get_error(ctx);
            if (err_str != null) {
                _ = futhark.futhark_context_clear_caches(ctx);
            }
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }

    pub fn getDataPointer1D(self: *Self, array: *FutharkArray1DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_1d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }
};

fn get1DDevicePtr(ctx: *FutharkContext, array: *FutharkArray1DF16) AccelError!*anyopaque {
    return ctx.getDataPointer1D(array);
}

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    fallback_slice: ?[]align(64) u8,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0, .fallback_slice = null };
        }

        if (gpu_enabled) {
            var ptr: ?*anyopaque = null;
            const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
            if (err != cuda.cudaSuccess) {
                return AccelError.CudaHostAllocFailed;
            }
            return Self{
                .ptr = ptr,
                .size = size,
                .fallback_slice = null,
            };
        }

        const slice = std.heap.page_allocator.alignedAlloc(u8, 64, size) catch return AccelError.CudaHostAllocFailed;
        return Self{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .fallback_slice = slice,
        };
    }

    pub fn free(self: *Self) void {
        if (self.fallback_slice) |slice| {
            std.heap.page_allocator.free(slice);
            self.fallback_slice = null;
            self.ptr = null;
            self.size = 0;
            return;
        }
        if (self.ptr) |p| {
            if (gpu_enabled) {
                _ = cuda.cudaFreeHost(p);
            }
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        const count = self.size / @sizeOf(T);
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != length) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn newZeros(ctx: *FutharkContext, length: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;

        const zeros = allocator.alloc(f16, length) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn values1D(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;

        const buf = allocator.alloc(f16, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);

        const result = futhark.futhark_values_f16_1d(ctx.ctx, self.arr, @ptrCast(buf.ptr));
        if (result != 0) {
            allocator.free(buf);
            return AccelError.FutharkValuesFailed;
        }

        const sync_result = futhark.futhark_context_sync(ctx.ctx);
        if (sync_result != 0) {
            allocator.free(buf);
            return AccelError.FutharkSyncFailed;
        }

        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        const rows = self.rows;
        const cols = self.cols;

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        if (flat.len != d0 * d1 * d2) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_3d(
            ctx.ctx,
            @ptrCast(flat.ptr),
            @intCast(d0),
            @intCast(d1),
            @intCast(d2),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const RSFLayer = struct {
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,

    pub fn free(self: *RSFLayer, ctx: *FutharkContext) void {
        self.velocity_t.free(ctx);
        self.velocity_s.free(ctx);
        self.weights_t.free(ctx);
        self.weights_s.free(ctx);
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    layers: []RSFLayer,
    layers_owner: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        const half: usize = model_dim / 2;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const base_seed: u64 = 0x4A41494445204E4F;
        const init_stddev: f32 = 0.02;

        var layers = allocator.alloc(RSFLayer, num_layers) catch return AccelError.AllocationFailed;
        errdefer allocator.free(layers);

        const total: usize = half * (half + 1);
        const ws_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(ws_buf);
        const wt_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(wt_buf);

        var layers_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < layers_built) : (idx += 1) {
                layers[idx].free(&ctx);
            }
        }

        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            const layer_seed: u64 = base_seed +% (@as(u64, @intCast(layer_idx)) *% 0x9E3779B97F4A7C15);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const rnd = rng.random();
            for (ws_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            for (wt_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            {
                var d: usize = 0;
                while (d < half) : (d += 1) {
                    ws_buf[d * (half + 1) + half] = @as(f16, 0.0);
                    wt_buf[d * (half + 1) + half] = @as(f16, 0.0);
                }
            }

            const weights_s = try FutharkArray2DF16.newFromFlat(&ctx, ws_buf, half, half + 1);
            const weights_t = try FutharkArray2DF16.newFromFlat(&ctx, wt_buf, half, half + 1);
            const velocity_s = try FutharkArray2DF16.newZeros(&ctx, half, half + 1, allocator);
            const velocity_t = try FutharkArray2DF16.newZeros(&ctx, half, half + 1, allocator);

            layers[layer_idx] = .{
                .weights_s = weights_s,
                .weights_t = weights_t,
                .velocity_s = velocity_s,
                .velocity_t = velocity_t,
            };
            layers_built += 1;
        }

        return Self{
            .ctx = ctx,
            .layers = layers,
            .layers_owner = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = @as(f16, -2.0),
            .clip_max = @as(f16, 2.0),
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            self.layers[i].free(&self.ctx);
        }
        self.layers_owner.free(self.layers);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var current_arr: ?*futhark.struct_futhark_f16_2d = input.arr;
        const rows = input.rows;
        const cols = input.cols;

        var li: usize = 0;
        while (li < self.layers.len) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var next_arr: ?*futhark.struct_futhark_f16_2d = null;
            const result = futhark.futhark_entry_rsf_forward(
                self.ctx.ctx,
                &next_arr,
                current_arr,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (result != 0) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.FutharkForwardFailed;
            }
            if (next_arr == null) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.NullPointer;
            }

            if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
            current_arr = next_arr;
        }

        return FutharkArray2DF16{ .arr = current_arr, .rows = rows, .cols = cols };
    }

    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const n_layers = self.layers.len;

        var current_act: ?*futhark.struct_futhark_f16_3d = inputs.arr;
        var current_act_owned: bool = false;
        var grad_out: ?*futhark.struct_futhark_f16_3d = null;

        errdefer {
            if (grad_out) |g| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
                grad_out = null;
            }
            if (current_act_owned) {
                if (current_act) |act| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, act);
                    current_act = null;
                }
                current_act_owned = false;
            }
        }

        var li: usize = 0;
        while (li < n_layers) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var rsf_out: ?*futhark.struct_futhark_f16_3d = null;
            const rc = futhark.futhark_entry_batch_forward(
                self.ctx.ctx,
                &rsf_out,
                current_act,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (rc != 0 or rsf_out == null) {
                const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
                if (err_str) |s| std.debug.print("[Futhark batch_forward L{d} error] {s}\n", .{ li, std.mem.span(s) });
                if (rsf_out) |out| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, out);
                }
                return AccelError.FutharkForwardFailed;
            }

            if (current_act_owned) {
                if (current_act) |act| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, act);
                }
            }
            current_act = rsf_out;
            current_act_owned = true;

            var oftb_out: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_rc = futhark.futhark_entry_batch_oftb_forward(
                self.ctx.ctx,
                &oftb_out,
                current_act,
            );
            if (current_act) |act| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, act);
            }
            current_act = null;
            current_act_owned = false;
            if (oftb_rc != 0 or oftb_out == null) {
                if (oftb_out) |out| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, out);
                }
                return AccelError.FutharkForwardFailed;
            }
            current_act = oftb_out;
            current_act_owned = true;
        }

        if (current_act == null) return AccelError.NullPointer;

        var loss_bits: u16 = 0;
        const loss_rc = futhark.futhark_entry_batch_compute_loss(
            self.ctx.ctx,
            &loss_bits,
            current_act,
            targets.arr,
        );
        if (loss_rc != 0) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark batch_compute_loss error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkComputeLossFailed;
        }
        const loss_f16: f16 = @bitCast(loss_bits);

        const gseed_rc = futhark.futhark_entry_compute_initial_grad_l2(
            self.ctx.ctx,
            &grad_out,
            current_act,
            targets.arr,
        );
        if (gseed_rc != 0 or grad_out == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark initial_grad_l2 error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkBackwardFailed;
        }

        var lb: usize = n_layers;
        while (lb > 0) {
            lb -= 1;
            const layer = &self.layers[lb];

            if (current_act == null or grad_out == null) return AccelError.NullPointer;
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var rsf_out_reconstructed: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_inv_rc = futhark.futhark_entry_batch_oftb_backward(
                self.ctx.ctx,
                &rsf_out_reconstructed,
                current_act,
            );
            if (oftb_inv_rc != 0 or rsf_out_reconstructed == null) {
                if (rsf_out_reconstructed) |out| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, out);
                }
                return AccelError.FutharkBackwardFailed;
            }

            if (current_act) |act| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, act);
            }
            current_act = null;
            current_act_owned = false;

            var layer_input_reconstructed: ?*futhark.struct_futhark_f16_3d = null;
            const rsf_inv_rc = futhark.futhark_entry_batch_rsf_inverse(
                self.ctx.ctx,
                &layer_input_reconstructed,
                rsf_out_reconstructed,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (rsf_out_reconstructed) |out| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, out);
            }
            rsf_out_reconstructed = null;
            if (rsf_inv_rc != 0 or layer_input_reconstructed == null) {
                const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
                if (err_str) |s| std.debug.print("[Futhark batch_rsf_inverse L{d} error] {s}\n", .{ lb, std.mem.span(s) });
                if (layer_input_reconstructed) |inp| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, inp);
                }
                return AccelError.FutharkBackwardFailed;
            }

            var oftb_grad: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_bwd_rc = futhark.futhark_entry_batch_oftb_backward(
                self.ctx.ctx,
                &oftb_grad,
                grad_out,
            );
            if (grad_out) |g| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
            }
            grad_out = null;
            if (oftb_bwd_rc != 0 or oftb_grad == null) {
                if (oftb_grad) |g| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
                }
                if (layer_input_reconstructed) |inp| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, inp);
                }
                return AccelError.FutharkBackwardFailed;
            }

            var grad_tup: ?*futhark.struct_futhark_opaque_tup3_grad_full = null;
            const bg_rc = futhark.futhark_entry_batch_gradients_full(
                self.ctx.ctx,
                &grad_tup,
                layer_input_reconstructed,
                oftb_grad,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (oftb_grad) |g| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
            }
            oftb_grad = null;

            if (bg_rc != 0 or grad_tup == null) {
                const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
                if (err_str) |s| std.debug.print("[Futhark batch_gradients_full L{d} error] {s}\n", .{ lb, std.mem.span(s) });
                if (layer_input_reconstructed) |inp| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, inp);
                }
                return AccelError.FutharkBackwardFailed;
            }

            var grad_ws: ?*futhark.struct_futhark_f16_2d = null;
            var grad_wt: ?*futhark.struct_futhark_f16_2d = null;
            var grad_in: ?*futhark.struct_futhark_f16_3d = null;

            const proj0 = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_0(self.ctx.ctx, &grad_ws, grad_tup);
            const proj1 = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_1(self.ctx.ctx, &grad_wt, grad_tup);
            const proj2 = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_2(self.ctx.ctx, &grad_in, grad_tup);
            _ = futhark.futhark_free_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16(self.ctx.ctx, grad_tup);

            if (proj0 != 0 or proj1 != 0 or proj2 != 0 or
                grad_ws == null or grad_wt == null or grad_in == null)
            {
                if (grad_ws != null) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_ws);
                if (grad_wt != null) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_wt);
                if (grad_in != null) _ = futhark.futhark_free_f16_3d(self.ctx.ctx, grad_in);
                if (layer_input_reconstructed) |inp| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, inp);
                }
                return AccelError.FutharkBackwardFailed;
            }

            errdefer {
                if (grad_ws) |g| {
                    _ = futhark.futhark_free_f16_2d(self.ctx.ctx, g);
                    grad_ws = null;
                }
                if (grad_wt) |g| {
                    _ = futhark.futhark_free_f16_2d(self.ctx.ctx, g);
                    grad_wt = null;
                }
                if (grad_in) |g| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
                    grad_in = null;
                }
                if (layer_input_reconstructed) |inp| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, inp);
                    layer_input_reconstructed = null;
                }
            }

            try sfdUpdateMat(self, &layer.weights_s, &layer.velocity_s, grad_ws, lr_bits, momentum_bits);
            try sfdUpdateMat(self, &layer.weights_t, &layer.velocity_t, grad_wt, lr_bits, momentum_bits);

            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_ws);
            grad_ws = null;
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_wt);
            grad_wt = null;

            grad_out = grad_in;
            grad_in = null;

            current_act = layer_input_reconstructed;
            current_act_owned = true;
            layer_input_reconstructed = null;
        }

        if (grad_out) |g| {
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, g);
            grad_out = null;
        }

        if (current_act_owned) {
            if (current_act) |act| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, act);
                current_act = null;
            }
            current_act_owned = false;
        }

        return loss_f16;
    }

    fn sfdUpdateMat(
        self: *Self,
        weights: *FutharkArray2DF16,
        velocity: *FutharkArray2DF16,
        gradients: ?*futhark.struct_futhark_f16_2d,
        lr_bits: u16,
        momentum_bits: u16,
    ) AccelError!void {
        if (weights.arr == null or velocity.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup2_2d = null;
        const rc = futhark.futhark_entry_sfd_update_mat(
            self.ctx.ctx,
            &out_tup,
            weights.arr,
            gradients,
            lr_bits,
            momentum_bits,
            velocity.arr,
        );
        if (rc != 0 or out_tup == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark sfd_update_mat error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_w: ?*futhark.struct_futhark_f16_2d = null;
        var new_v: ?*futhark.struct_futhark_f16_2d = null;
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_0(self.ctx.ctx, &new_w, out_tup);
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_1(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup2_arr2d_f16_arr2d_f16(self.ctx.ctx, out_tup);
        if (new_w == null or new_v == null) return AccelError.FutharkSFDUpdateFailed;
        const old_w = weights.arr;
        const old_v = velocity.arr;
        weights.arr = new_w;
        velocity.arr = new_v;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_w);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_v);
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (scale_factor == @as(f16, 0.0)) return AccelError.InvalidDimensions;

        const scale_bits: u16 = @bitCast(scale_factor);
        for (self.layers) |*layer| {
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var new_ws: ?*futhark.struct_futhark_f16_2d = null;
            const result_s = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_ws,
                layer.weights_s.arr,
                scale_bits,
            );
            if (result_s != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_ws != null) {
                const old = layer.weights_s.arr;
                layer.weights_s.arr = new_ws;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }

            var new_wt: ?*futhark.struct_futhark_f16_2d = null;
            const result_t = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_wt,
                layer.weights_t.arr,
                scale_bits,
            );
            if (result_t != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_wt != null) {
                const old = layer.weights_t.arr;
                layer.weights_t.arr = new_wt;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn layerPtr(self: *Self, layer_idx: usize) AccelError!*RSFLayer {
        if (!self.initialized) return AccelError.NullPointer;
        if (layer_idx >= self.layers.len) return AccelError.InvalidDimensions;
        return &self.layers[layer_idx];
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_s.free(&self.ctx);
        layer.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_t.free(&self.ctx);
        layer.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocityS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_s.free(&self.ctx);
        layer.velocity_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocityT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_t.free(&self.ctx);
        layer.velocity_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn readLayerWeightsFlat(self: *Self, layer_idx: usize, kind: WeightKind, allocator: std.mem.Allocator) AccelError![]f16 {
        const layer = try self.layerPtr(layer_idx);
        return switch (kind) {
            .weights_s => readMatFlat(self, &layer.weights_s, allocator),
            .weights_t => readMatFlat(self, &layer.weights_t, allocator),
            .velocity_s => readMatFlat(self, &layer.velocity_s, allocator),
            .velocity_t => readMatFlat(self, &layer.velocity_t, allocator),
        };
    }

    pub fn getLayerDevicePtr(
        self: *Self,
        layer_idx: usize,
        kind: WeightKind,
    ) AccelError!struct { ptr: *anyopaque, count: usize } {
        if (!self.initialized) return AccelError.NullPointer;
        const layer = try self.layerPtr(layer_idx);
        const half = self.model_dim / 2;
        return switch (kind) {
            .weights_s => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_s), .count = half * (half + 1) },
            .weights_t => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_t), .count = half * (half + 1) },
            .velocity_s => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_s), .count = half * (half + 1) },
            .velocity_t => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_t), .count = half * (half + 1) },
        };
    }

    fn readMatFlat(self: *Self, mat: *FutharkArray2DF16, allocator: std.mem.Allocator) AccelError![]f16 {
        const rows = try mat.values(&self.ctx, allocator);
        defer {
            for (rows) |row| allocator.free(row);
            allocator.free(rows);
        }
        const half = self.model_dim / 2;
        const cols = half + 1;
        if (rows.len != half) return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, half, cols) catch return AccelError.AllocationFailed;
        var flat = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        var idx: usize = 0;
        for (rows) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
            for (row) |v| {
                flat[idx] = v;
                idx += 1;
            }
        }
        return flat;
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (clip_min_val >= clip_max_val) return AccelError.InvalidDimensions;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, output.rows * output.cols) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const EmbeddingAccelerator = struct {
    ctx: *FutharkContext,
    weight: FutharkArray2DF16,
    grad_weight: FutharkArray2DF16,
    vocab_size: usize,
    dim: usize,
    initialized: bool,

    const Self = @This();

    pub fn init(ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;

        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const total = vocab_size * dim;
        const weight_data = std.heap.page_allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(weight_data);
        for (weight_data) |*v| {
            v.* = @floatCast((rnd.float(f32) - 0.5) * 0.02);
        }

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_data, vocab_size, dim);
        errdefer weight.free(ctx);
        const grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, std.heap.page_allocator);

        return Self{
            .ctx = ctx,
            .weight = weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.grad_weight.free(self.ctx);
        self.weight.free(self.ctx);
        self.initialized = false;
    }

    pub fn forward(self: *Self, tokens: []const u32) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var out: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_forward(
            self.ctx.ctx,
            &out,
            tok_arr.arr,
            self.weight.arr,
        );

        if (rc != 0 or out == null) return AccelError.FutharkForwardFailed;
        return FutharkArray2DF16{ .arr = out, .rows = tokens.len, .cols = self.dim };
    }

    pub fn backwardAndUpdate(self: *Self, tokens: []const u32, grad_output: *FutharkArray2DF16, lr: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var new_grad: ?*futhark.struct_futhark_f16_2d = null;
        const bwd_rc = futhark.futhark_entry_embedding_backward(
            self.ctx.ctx,
            &new_grad,
            tok_arr.arr,
            grad_output.arr,
            self.grad_weight.arr,
        );
        if (bwd_rc != 0 or new_grad == null) return AccelError.FutharkBackwardFailed;

        const old_grad = self.grad_weight.arr;
        self.grad_weight.arr = new_grad;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_grad);

        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        const lr_bits: u16 = @bitCast(lr);
        const upd_rc = futhark.futhark_entry_embedding_update(
            self.ctx.ctx,
            &new_weight,
            self.weight.arr,
            self.grad_weight.arr,
            lr_bits,
        );
        if (upd_rc != 0 or new_weight == null) return AccelError.FutharkSFDUpdateFailed;

        const old_weight = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_weight);

        const zeroed_grad = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, std.heap.page_allocator);
        const old_g = self.grad_weight.arr;
        self.grad_weight.arr = zeroed_grad.arr;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_g);
    }
};
