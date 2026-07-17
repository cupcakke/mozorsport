const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const ModelFormat = @import("model_io.zig").ModelFormat;
const ModelError = @import("model_io.zig").ModelError;
const RSF = @import("../processor/rsf.zig").RSF;
const Tensor = @import("tensor.zig").Tensor;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const LearnedEmbedding = @import("learned_embedding.zig").LearnedEmbedding;

pub const MAGIC: [8]u8 = .{ 'J', 'A', 'I', 'D', 'E', 'I', 'M', 0 };
pub const VERSION: u32 = 1;

const Reader = struct {
    file: std.fs.File,
    pos: u64,
    size: u64,
    hasher: std.crypto.hash.sha2.Sha256,

    fn readNoEof(self: *Reader, dst: []u8) !void {
        if (dst.len == 0) return;
        const len_u64: u64 = @intCast(dst.len);
        if (self.pos > self.size or len_u64 > self.size - self.pos) return ModelError.CorruptedData;
        try self.file.reader().readNoEof(dst);
        self.hasher.update(dst);
        self.pos += len_u64;
    }

    fn readRawNoHash(self: *Reader, dst: []u8) !void {
        if (dst.len == 0) return;
        const len_u64: u64 = @intCast(dst.len);
        if (self.pos > self.size or len_u64 > self.size - self.pos) return ModelError.CorruptedData;
        try self.file.reader().readNoEof(dst);
        self.pos += len_u64;
    }

    fn readInt(self: *Reader, comptime T: type) !T {
        var bytes: [@sizeOf(T)]u8 = undefined;
        try self.readNoEof(bytes[0..]);
        return std.mem.readInt(T, &bytes, .little);
    }

    fn readF32(self: *Reader) !f32 {
        const bits = try self.readInt(u32);
        return @as(f32, @bitCast(bits));
    }
};

fn openRead(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

fn createWrite(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .truncate = true });
    return std.fs.cwd().createFile(path, .{ .truncate = true });
}

fn renamePath(from: []const u8, to: []const u8) !void {
    if (std.fs.path.isAbsolute(from)) return std.fs.renameAbsolute(from, to);
    return std.fs.cwd().rename(from, to);
}

fn deletePath(path: []const u8) void {
    if (std.fs.path.isAbsolute(path)) {
        std.fs.deleteFileAbsolute(path) catch {};
    } else {
        std.fs.cwd().deleteFile(path) catch {};
    }
}

fn writeHash(writer: anytype, hasher: *std.crypto.hash.sha2.Sha256, bytes: []const u8) !void {
    try writer.writeAll(bytes);
    hasher.update(bytes);
}

fn writeInt(writer: anytype, hasher: *std.crypto.hash.sha2.Sha256, comptime T: type, value: T) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try writeHash(writer, hasher, bytes[0..]);
}

fn writeF32(writer: anytype, hasher: *std.crypto.hash.sha2.Sha256, value: f32) !void {
    if (!std.math.isFinite(value)) return ModelError.CorruptedData;
    try writeInt(writer, hasher, u32, @as(u32, @bitCast(value)));
}

fn loadTokenizerFromBytes(allocator: std.mem.Allocator, model_path: []const u8, bytes: []const u8) !*MGT {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tokenizer.{d}.tmp", .{ model_path, std.time.nanoTimestamp() });
    defer allocator.free(tmp_path);
    var committed = false;
    defer if (!committed) deletePath(tmp_path);
    {
        const file = try createWrite(tmp_path);
        var closed = false;
        defer if (!closed) file.close();
        try file.writeAll(bytes);
        try file.sync();
        file.close();
        closed = true;
    }
    const tokenizer = try allocator.create(MGT);
    errdefer allocator.destroy(tokenizer);
    tokenizer.* = try MGT.init(allocator, &.{}, &.{}, null, .english);
    errdefer tokenizer.deinit();
    try tokenizer.loadVocab(tmp_path);
    committed = true;
    deletePath(tmp_path);
    return tokenizer;
}

fn readTensor(allocator: std.mem.Allocator, reader: *Reader, rows: usize, cols: usize) !Tensor {
    const len_u64 = try reader.readInt(u64);
    const expected = try std.math.mul(usize, rows, cols);
    if (len_u64 != @as(u64, @intCast(expected))) return ModelError.CorruptedData;
    var tensor = try Tensor.init(allocator, &.{ rows, cols });
    errdefer tensor.deinit();
    for (tensor.data) |*value| {
        const v = try reader.readF32();
        if (!std.math.isFinite(v)) return ModelError.CorruptedData;
        value.* = v;
    }
    return tensor;
}

pub fn exportCheckpointModel(allocator: std.mem.Allocator, checkpoint_path: []const u8, output_path: []const u8) !void {
    var state = try checkpoint.loadCheckpointState(allocator, checkpoint_path);
    defer state.deinit();
    const tokenizer_bytes = try checkpoint.extractTokenizerData(allocator, checkpoint_path);
    defer allocator.free(tokenizer_bytes);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.{d}.tmp", .{ output_path, std.time.nanoTimestamp() });
    defer allocator.free(tmp_path);
    var committed = false;
    defer if (!committed) deletePath(tmp_path);
    {
        const file = try createWrite(tmp_path);
        var closed = false;
        defer if (!closed) file.close();
        var buffered = std.io.bufferedWriter(file.writer());
        const writer = buffered.writer();
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        try writeHash(writer, &hasher, MAGIC[0..]);
        try writeInt(writer, &hasher, u32, VERSION);
        try writeInt(writer, &hasher, u64, state.metadata.global_step);
        try writeInt(writer, &hasher, u64, @as(u64, @intCast(state.metadata.model_dim)));
        try writeInt(writer, &hasher, u64, @as(u64, @intCast(state.metadata.num_layers)));
        try writeInt(writer, &hasher, u64, @as(u64, @intCast(state.metadata.vocab_size)));
        try writeF32(writer, &hasher, state.metadata.clip_min);
        try writeF32(writer, &hasher, state.metadata.clip_max);
        try writeInt(writer, &hasher, u64, @as(u64, @intCast(tokenizer_bytes.len)));
        try writeHash(writer, &hasher, tokenizer_bytes);
        var layer: usize = 0;
        while (layer < state.metadata.num_layers) : (layer += 1) {
            const ws = try state.accelerator.readLayerWeightsFlat(layer, .weights_s, allocator);
            defer allocator.free(ws);
            try writeInt(writer, &hasher, u64, @as(u64, @intCast(ws.len)));
            for (ws) |v| try writeF32(writer, &hasher, @floatCast(v));
            const wt = try state.accelerator.readLayerWeightsFlat(layer, .weights_t, allocator);
            defer allocator.free(wt);
            try writeInt(writer, &hasher, u64, @as(u64, @intCast(wt.len)));
            for (wt) |v| try writeF32(writer, &hasher, @floatCast(v));
        }
        try writeInt(writer, &hasher, u64, @as(u64, @intCast(state.embedding.weight.data.len)));
        for (state.embedding.weight.data) |v| try writeF32(writer, &hasher, v);
        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        try writer.writeAll(digest[0..]);
        try buffered.flush();
        try file.sync();
        file.close();
        closed = true;
    }
    var loaded = try importInferenceModel(tmp_path, allocator);
    defer loaded.deinit();
    const loaded_embedding = loaded.embedding orelse return ModelError.MissingComponent;
    if (loaded_embedding.weight.data.len != state.embedding.weight.data.len) return ModelError.CorruptedData;
    for (loaded_embedding.weight.data, 0..) |value, i| {
        if (value != state.embedding.weight.data[i]) return ModelError.CorruptedData;
    }
    const loaded_rsf = loaded.rsf orelse return ModelError.MissingComponent;
    const loaded_ctrl = loaded_rsf.ctrl orelse return ModelError.MissingComponent;
    var verify_layer: usize = 0;
    while (verify_layer < state.metadata.num_layers) : (verify_layer += 1) {
        const ws = try state.accelerator.readLayerWeightsFlat(verify_layer, .weights_s, allocator);
        defer allocator.free(ws);
        const wt = try state.accelerator.readLayerWeightsFlat(verify_layer, .weights_t, allocator);
        defer allocator.free(wt);
        if (loaded_ctrl.layers[verify_layer].s_weight.data.len != ws.len or loaded_ctrl.layers[verify_layer].t_weight.data.len != wt.len) return ModelError.CorruptedData;
        for (ws, 0..) |value, i| {
            if (loaded_ctrl.layers[verify_layer].s_weight.data[i] != @as(f32, @floatCast(value))) return ModelError.CorruptedData;
        }
        for (wt, 0..) |value, i| {
            if (loaded_ctrl.layers[verify_layer].t_weight.data[i] != @as(f32, @floatCast(value))) return ModelError.CorruptedData;
        }
    }
    try renamePath(tmp_path, output_path);
    committed = true;
}

pub fn importInferenceModel(path: []const u8, allocator: std.mem.Allocator) !ModelFormat {
    var file = try openRead(path);
    defer file.close();
    const stat = try file.stat();
    var reader = Reader{ .file = file, .pos = 0, .size = stat.size, .hasher = std.crypto.hash.sha2.Sha256.init(.{}) };
    var magic: [8]u8 = undefined;
    try reader.readNoEof(magic[0..]);
    if (!std.mem.eql(u8, magic[0..], MAGIC[0..])) return ModelError.InvalidMagicHeader;
    const version = try reader.readInt(u32);
    if (version != VERSION) return ModelError.UnsupportedVersion;
    const global_step = try reader.readInt(u64);
    const model_dim_u64 = try reader.readInt(u64);
    const num_layers_u64 = try reader.readInt(u64);
    const vocab_size_u64 = try reader.readInt(u64);
    const clip_min = try reader.readF32();
    const clip_max = try reader.readF32();
    if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max) or !(clip_min < clip_max)) return ModelError.CorruptedData;
    if (model_dim_u64 == 0 or model_dim_u64 % 2 != 0 or num_layers_u64 == 0 or vocab_size_u64 == 0) return ModelError.CorruptedData;
    const model_dim = std.math.cast(usize, model_dim_u64) orelse return ModelError.CorruptedData;
    const half = model_dim / 2;
    const num_layers = std.math.cast(usize, num_layers_u64) orelse return ModelError.CorruptedData;
    const vocab_size = std.math.cast(usize, vocab_size_u64) orelse return ModelError.CorruptedData;
    const tokenizer_len_u64 = try reader.readInt(u64);
    const tokenizer_len = std.math.cast(usize, tokenizer_len_u64) orelse return ModelError.CorruptedData;
    if (tokenizer_len == 0 or tokenizer_len > 1024 * 1024 * 1024) return ModelError.CorruptedData;
    const tokenizer_bytes = try allocator.alloc(u8, tokenizer_len);
    defer allocator.free(tokenizer_bytes);
    try reader.readNoEof(tokenizer_bytes);
    var model = try ModelFormat.init(allocator, "jaide-exported-checkpoint", "inference-only model exported from checkpoint");
    errdefer model.deinit();
    _ = global_step;
    model.metadata.rsf_dim = half;
    model.metadata.rsf_layers = num_layers;
    model.metadata.mgt_vocab_size = vocab_size;
    const rsf_ptr = try allocator.create(RSF);
    errdefer allocator.destroy(rsf_ptr);
    rsf_ptr.* = try RSF.initWithConfig(allocator, half, num_layers, .{ .clip_min = clip_min, .clip_max = clip_max });
    errdefer rsf_ptr.deinit();
    const ctrl = rsf_ptr.ctrl orelse return ModelError.CorruptedData;
    var layer: usize = 0;
    while (layer < num_layers) : (layer += 1) {
        ctrl.layers[layer].s_weight.deinit();
        ctrl.layers[layer].s_weight = try readTensor(allocator, &reader, half, half + 1);
        ctrl.layers[layer].t_weight.deinit();
        ctrl.layers[layer].t_weight = try readTensor(allocator, &reader, half, half + 1);
    }
    ctrl.cpu_weight_version +%= 1;
    ctrl.gpu_available.store(0, .monotonic);
    const tokenizer = try loadTokenizerFromBytes(allocator, path, tokenizer_bytes);
    errdefer {
        tokenizer.deinit();
        allocator.destroy(tokenizer);
    }
    if (@as(usize, tokenizer.next_token_id) != vocab_size) return ModelError.CorruptedData;
    var embedding = try LearnedEmbedding.initZero(allocator, vocab_size, model_dim);
    errdefer embedding.deinit();
    const embedding_len = try reader.readInt(u64);
    if (embedding_len != @as(u64, @intCast(embedding.weight.data.len))) return ModelError.CorruptedData;
    for (embedding.weight.data) |*value| {
        const v = try reader.readF32();
        if (!std.math.isFinite(v)) return ModelError.CorruptedData;
        value.* = v;
    }
    @memset(embedding.grad.data, 0.0);
    @memset(embedding.velocity.data, 0.0);
    var expected: [32]u8 = undefined;
    reader.hasher.final(&expected);
    var stored: [32]u8 = undefined;
    try reader.readRawNoHash(stored[0..]);
    if (!std.mem.eql(u8, expected[0..], stored[0..])) return ModelError.ChecksumMismatch;
    if (reader.pos != reader.size) return ModelError.CorruptedData;
    model.rsf = rsf_ptr;
    model.mgt = tokenizer;
    model.embedding = embedding;
    return model;
}
