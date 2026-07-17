const std = @import("std");
const accel = @import("../hw/accel/accel_interface.zig");
const MGT = @import("../tokenizer/mgt.zig").MGT;
const LearnedEmbedding = @import("learned_embedding.zig").LearnedEmbedding;

pub const CHECKPOINT_MAGIC: [8]u8 = .{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
pub const CHECKPOINT_TRAILER: u32 = 0xDEADBEEF;
pub const CURRENT_CHECKPOINT_VERSION: u32 = 9;
pub const MAX_ID_LENGTH: usize = 1 << 20;
pub const MAX_NODE_DATA_LENGTH: usize = 1 << 24;
pub const MAX_EDGE_GROUP_COUNT: u32 = 1 << 24;
pub const MAX_TOKENIZER_BYTES: u64 = 1 << 34;
pub const MAX_DIMENSION: u64 = 1 << 30;
pub const MAX_LAYERS: u64 = 1 << 20;
pub const MAX_VOCAB: u64 = 1 << 32;

pub const CheckpointError = error{
    CheckpointMagicMismatch,
    UnsupportedCheckpointVersion,
    TruncatedCheckpoint,
    InvalidCheckpointDimension,
    InvalidCheckpointLength,
    InvalidCheckpointEmbeddingFlag,
    InvalidCheckpointTrailer,
    TrailingCheckpointData,
    InvalidCheckpointFloat,
    InvalidCheckpointGraph,
    InvalidCheckpointTokenizer,
    MissingCheckpointEmbedding,
    FileTooLarge,
};

pub const CheckpointMetadata = struct {
    magic: [8]u8,
    version: u32,
    global_step: u64,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    stored_batch_size: usize,
    learning_rate: f32,
    momentum: f32,
    has_embedding: bool,
    embedding_vocab_size: usize,
    embedding_dim: usize,
    embedding_weight_len: usize,
    embedding_velocity_len: usize,
    nsir_node_count: u32,
    nsir_edge_group_count: u32,
    nsir_edge_count: u64,
    tokenizer_size: u64,
    tokenizer_offset: u64,
    file_size: u64,
    rsf_tensor_elements_per_matrix: usize,
    rsf_weight_bytes: u64,
    rsf_velocity_bytes: u64,
    embedding_weight_bytes: u64,
    embedding_velocity_bytes: u64,
    checkpoint_payload_bytes: u64,
    clip_min: f32,
    clip_max: f32,
    integrity_ok: bool,
};

pub const CheckpointState = struct {
    allocator: std.mem.Allocator,
    metadata: CheckpointMetadata,
    tokenizer: MGT,
    accelerator: accel.RSFAccelerator,
    embedding: LearnedEmbedding,

    pub fn deinit(self: *CheckpointState) void {
        self.embedding.deinit();
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }
};

const Reader = struct {
    file: std.fs.File,
    pos: u64,
    size: u64,

    fn readNoEof(self: *Reader, dst: []u8) !void {
        if (dst.len == 0) return;
        const len_u64: u64 = @intCast(dst.len);
        if (self.pos > self.size or len_u64 > self.size - self.pos) return CheckpointError.TruncatedCheckpoint;
        try self.file.reader().readNoEof(dst);
        self.pos += len_u64;
    }

    fn readByte(self: *Reader) !u8 {
        var b: [1]u8 = undefined;
        try self.readNoEof(b[0..]);
        return b[0];
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

    fn readF64(self: *Reader) !f64 {
        const bits = try self.readInt(u64);
        return @as(f64, @bitCast(bits));
    }

    fn skip(self: *Reader, count: u64) !void {
        if (count == 0) return;
        if (self.pos > self.size or count > self.size - self.pos) return CheckpointError.TruncatedCheckpoint;
        try self.file.seekTo(self.pos + count);
        self.pos += count;
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

fn deletePath(path: []const u8) void {
    if (std.fs.path.isAbsolute(path)) {
        std.fs.deleteFileAbsolute(path) catch {};
    } else {
        std.fs.cwd().deleteFile(path) catch {};
    }
}

fn castU64ToUsize(value: u64) !usize {
    return std.math.cast(usize, value) orelse CheckpointError.FileTooLarge;
}

fn checkedMulU64(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return CheckpointError.InvalidCheckpointLength;
}

fn checkedAddU64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return CheckpointError.InvalidCheckpointLength;
}

fn checkedMulUsize(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return CheckpointError.InvalidCheckpointLength;
}

fn finiteF32(value: f32) !void {
    if (!std.math.isFinite(value)) return CheckpointError.InvalidCheckpointFloat;
}

fn finiteF64(value: f64) !void {
    if (!std.math.isFinite(value)) return CheckpointError.InvalidCheckpointFloat;
}

fn checkedF32ToF16(value: f32) !f16 {
    try finiteF32(value);
    if (value < -65504.0 or value > 65504.0) return CheckpointError.InvalidCheckpointFloat;
    const converted: f16 = @floatCast(value);
    if (!std.math.isFinite(converted)) return CheckpointError.InvalidCheckpointFloat;
    return converted;
}

fn validateHyperparameters(learning_rate: f32, momentum: f32) !void {
    try finiteF32(learning_rate);
    try finiteF32(momentum);
    if (learning_rate < 0.0 or learning_rate > 65504.0) return CheckpointError.InvalidCheckpointFloat;
    if (momentum < 0.0 or momentum >= 1.0) return CheckpointError.InvalidCheckpointFloat;
    _ = try checkedF32ToF16(learning_rate);
    _ = try checkedF32ToF16(momentum);
}

fn validateDimensions(model_dim_u64: u64, num_layers_u64: u64, vocab_size_u64: u64, batch_size_u64: u64) !void {
    if (model_dim_u64 == 0 or model_dim_u64 > MAX_DIMENSION or model_dim_u64 % 2 != 0) return CheckpointError.InvalidCheckpointDimension;
    if (num_layers_u64 == 0 or num_layers_u64 > MAX_LAYERS) return CheckpointError.InvalidCheckpointDimension;
    if (vocab_size_u64 == 0 or vocab_size_u64 > MAX_VOCAB) return CheckpointError.InvalidCheckpointDimension;
    if (batch_size_u64 == 0) return CheckpointError.InvalidCheckpointDimension;
}

fn readHeader(reader: *Reader) !CheckpointMetadata {
    var magic: [8]u8 = undefined;
    try reader.readNoEof(magic[0..]);
    if (!std.mem.eql(u8, magic[0..], CHECKPOINT_MAGIC[0..])) return CheckpointError.CheckpointMagicMismatch;
    const version = try reader.readInt(u32);
    if (version != CURRENT_CHECKPOINT_VERSION) return CheckpointError.UnsupportedCheckpointVersion;
    const global_step = try reader.readInt(u64);
    const model_dim_u64 = try reader.readInt(u64);
    const num_layers_u64 = try reader.readInt(u64);
    const vocab_size_u64 = try reader.readInt(u64);
    const batch_size_u64 = try reader.readInt(u64);
    const learning_rate = try reader.readF32();
    const momentum = try reader.readF32();
    try validateDimensions(model_dim_u64, num_layers_u64, vocab_size_u64, batch_size_u64);
    try validateHyperparameters(learning_rate, momentum);
    const model_dim = try castU64ToUsize(model_dim_u64);
    const num_layers = try castU64ToUsize(num_layers_u64);
    const vocab_size = try castU64ToUsize(vocab_size_u64);
    const batch_size = try castU64ToUsize(batch_size_u64);
    const half = model_dim / 2;
    const columns = try checkedMulUsize(half + 1, 1);
    const matrix_len = try checkedMulUsize(half, columns);
    return .{
        .magic = magic,
        .version = version,
        .global_step = global_step,
        .model_dim = model_dim,
        .num_layers = num_layers,
        .vocab_size = vocab_size,
        .stored_batch_size = batch_size,
        .learning_rate = learning_rate,
        .momentum = momentum,
        .has_embedding = false,
        .embedding_vocab_size = 0,
        .embedding_dim = 0,
        .embedding_weight_len = 0,
        .embedding_velocity_len = 0,
        .nsir_node_count = 0,
        .nsir_edge_group_count = 0,
        .nsir_edge_count = 0,
        .tokenizer_size = 0,
        .tokenizer_offset = 0,
        .file_size = reader.size,
        .rsf_tensor_elements_per_matrix = matrix_len,
        .rsf_weight_bytes = 0,
        .rsf_velocity_bytes = 0,
        .embedding_weight_bytes = 0,
        .embedding_velocity_bytes = 0,
        .checkpoint_payload_bytes = reader.size,
        .clip_min = 0.0,
        .clip_max = 0.0,
        .integrity_ok = false,
    };
}

fn scanF32Payload(reader: *Reader, count: usize, require_f16: bool) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const value = try reader.readF32();
        if (require_f16) {
            _ = try checkedF32ToF16(value);
        } else {
            try finiteF32(value);
        }
    }
}

fn readF16Payload(allocator: std.mem.Allocator, reader: *Reader, expected_len: usize) ![]f16 {
    const saved_len_u64 = try reader.readInt(u64);
    const saved_len = try castU64ToUsize(saved_len_u64);
    if (saved_len != expected_len) return CheckpointError.InvalidCheckpointLength;
    const values = try allocator.alloc(f16, saved_len);
    errdefer allocator.free(values);
    var i: usize = 0;
    while (i < saved_len) : (i += 1) values[i] = try checkedF32ToF16(try reader.readF32());
    return values;
}

fn skipSizedBytes(reader: *Reader, max_len: usize) !void {
    const len = try reader.readInt(u32);
    if (len > max_len) return CheckpointError.InvalidCheckpointLength;
    try reader.skip(len);
}

pub fn inspectCheckpoint(path: []const u8) !CheckpointMetadata {
    var file = try openRead(path);
    defer file.close();
    const stat = try file.stat();
    var reader = Reader{ .file = file, .pos = 0, .size = stat.size };
    var metadata = try readHeader(&reader);
    const matrix_len = metadata.rsf_tensor_elements_per_matrix;
    const matrix_bytes = try checkedMulU64(@intCast(matrix_len), @sizeOf(f32));
    var layer_index: usize = 0;
    while (layer_index < metadata.num_layers) : (layer_index += 1) {
        var matrix_index: usize = 0;
        while (matrix_index < 4) : (matrix_index += 1) {
            const saved_len_u64 = try reader.readInt(u64);
            const saved_len = try castU64ToUsize(saved_len_u64);
            if (saved_len != matrix_len) return CheckpointError.InvalidCheckpointLength;
            try scanF32Payload(&reader, saved_len, true);
            if (matrix_index < 2) {
                metadata.rsf_weight_bytes = try checkedAddU64(metadata.rsf_weight_bytes, matrix_bytes);
            } else {
                metadata.rsf_velocity_bytes = try checkedAddU64(metadata.rsf_velocity_bytes, matrix_bytes);
            }
        }
    }
    const clip_min = try reader.readF32();
    const clip_max = try reader.readF32();
    try finiteF32(clip_min);
    try finiteF32(clip_max);
    if (!(clip_min < clip_max)) return CheckpointError.InvalidCheckpointFloat;
    _ = try checkedF32ToF16(clip_min);
    _ = try checkedF32ToF16(clip_max);
    metadata.clip_min = clip_min;
    metadata.clip_max = clip_max;
    const embedding_flag = try reader.readByte();
    if (embedding_flag > 1) return CheckpointError.InvalidCheckpointEmbeddingFlag;
    metadata.has_embedding = embedding_flag == 1;
    if (metadata.has_embedding) {
        const embedding_vocab_u64 = try reader.readInt(u64);
        const embedding_dim_u64 = try reader.readInt(u64);
        if (embedding_vocab_u64 != @as(u64, @intCast(metadata.vocab_size)) or embedding_dim_u64 != @as(u64, @intCast(metadata.model_dim))) return CheckpointError.InvalidCheckpointDimension;
        metadata.embedding_vocab_size = try castU64ToUsize(embedding_vocab_u64);
        metadata.embedding_dim = try castU64ToUsize(embedding_dim_u64);
        const expected_len = try checkedMulUsize(metadata.embedding_vocab_size, metadata.embedding_dim);
        const weight_len_u64 = try reader.readInt(u64);
        const weight_len = try castU64ToUsize(weight_len_u64);
        if (weight_len != expected_len) return CheckpointError.InvalidCheckpointLength;
        metadata.embedding_weight_len = weight_len;
        try scanF32Payload(&reader, weight_len, false);
        metadata.embedding_weight_bytes = try checkedMulU64(@intCast(weight_len), @sizeOf(f32));
        const velocity_len_u64 = try reader.readInt(u64);
        const velocity_len = try castU64ToUsize(velocity_len_u64);
        if (velocity_len != expected_len) return CheckpointError.InvalidCheckpointLength;
        metadata.embedding_velocity_len = velocity_len;
        try scanF32Payload(&reader, velocity_len, false);
        metadata.embedding_velocity_bytes = try checkedMulU64(@intCast(velocity_len), @sizeOf(f32));
    }
    const node_count = try reader.readInt(u32);
    metadata.nsir_node_count = node_count;
    var node_index: u32 = 0;
    while (node_index < node_count) : (node_index += 1) {
        try skipSizedBytes(&reader, MAX_ID_LENGTH);
        try skipSizedBytes(&reader, MAX_NODE_DATA_LENGTH);
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
    }
    const edge_group_count = try reader.readInt(u32);
    if (edge_group_count > MAX_EDGE_GROUP_COUNT) return CheckpointError.InvalidCheckpointGraph;
    metadata.nsir_edge_group_count = edge_group_count;
    var group_index: u32 = 0;
    while (group_index < edge_group_count) : (group_index += 1) {
        try skipSizedBytes(&reader, MAX_ID_LENGTH);
        try skipSizedBytes(&reader, MAX_ID_LENGTH);
        const edge_count = try reader.readInt(u32);
        if (edge_count > MAX_EDGE_GROUP_COUNT) return CheckpointError.InvalidCheckpointGraph;
        metadata.nsir_edge_count = try checkedAddU64(metadata.nsir_edge_count, @as(u64, edge_count));
        var edge_index: u32 = 0;
        while (edge_index < edge_count) : (edge_index += 1) {
            try finiteF64(try reader.readF64());
            const quality = try reader.readByte();
            if (quality > 4) return CheckpointError.InvalidCheckpointGraph;
            try finiteF64(try reader.readF64());
            try finiteF64(try reader.readF64());
            try finiteF64(try reader.readF64());
        }
    }
    const tokenizer_len = try reader.readInt(u64);
    if (tokenizer_len == 0 or tokenizer_len > MAX_TOKENIZER_BYTES) return CheckpointError.InvalidCheckpointTokenizer;
    if (tokenizer_len > reader.size - reader.pos) return CheckpointError.TruncatedCheckpoint;
    metadata.tokenizer_size = tokenizer_len;
    metadata.tokenizer_offset = reader.pos;
    try reader.skip(tokenizer_len);
    const trailer = try reader.readInt(u32);
    if (trailer != CHECKPOINT_TRAILER) return CheckpointError.InvalidCheckpointTrailer;
    if (reader.pos != reader.size) return CheckpointError.TrailingCheckpointData;
    metadata.integrity_ok = true;
    return metadata;
}

fn skipF32Array(reader: *Reader, expected_len: usize, require_f16: bool) !void {
    const saved_len_u64 = try reader.readInt(u64);
    const saved_len = try castU64ToUsize(saved_len_u64);
    if (saved_len != expected_len) return CheckpointError.InvalidCheckpointLength;
    try scanF32Payload(reader, saved_len, require_f16);
}

fn skipGraph(reader: *Reader, metadata: *CheckpointMetadata) !void {
    const node_count = try reader.readInt(u32);
    metadata.nsir_node_count = node_count;
    var node_index: u32 = 0;
    while (node_index < node_count) : (node_index += 1) {
        try skipSizedBytes(reader, MAX_ID_LENGTH);
        try skipSizedBytes(reader, MAX_NODE_DATA_LENGTH);
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
        try finiteF64(try reader.readF64());
    }
    const edge_group_count = try reader.readInt(u32);
    if (edge_group_count > MAX_EDGE_GROUP_COUNT) return CheckpointError.InvalidCheckpointGraph;
    metadata.nsir_edge_group_count = edge_group_count;
    var group_index: u32 = 0;
    while (group_index < edge_group_count) : (group_index += 1) {
        try skipSizedBytes(reader, MAX_ID_LENGTH);
        try skipSizedBytes(reader, MAX_ID_LENGTH);
        const edge_count = try reader.readInt(u32);
        if (edge_count > MAX_EDGE_GROUP_COUNT) return CheckpointError.InvalidCheckpointGraph;
        metadata.nsir_edge_count = try checkedAddU64(metadata.nsir_edge_count, @as(u64, edge_count));
        var edge_index: u32 = 0;
        while (edge_index < edge_count) : (edge_index += 1) {
            try finiteF64(try reader.readF64());
            const quality = try reader.readByte();
            if (quality > 4) return CheckpointError.InvalidCheckpointGraph;
            try finiteF64(try reader.readF64());
            try finiteF64(try reader.readF64());
            try finiteF64(try reader.readF64());
        }
    }
}

fn loadTokenizerFromBytes(allocator: std.mem.Allocator, checkpoint_path: []const u8, bytes: []const u8) !MGT {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tokenizer.{d}.tmp", .{ checkpoint_path, std.time.nanoTimestamp() });
    defer allocator.free(tmp_path);
    var committed = false;
    defer if (!committed) deletePath(tmp_path);
    {
        const file = try createWrite(tmp_path);
        var closed = false;
        defer if (!closed) file.close();
        try file.writer().writeAll(bytes);
        try file.sync();
        file.close();
        closed = true;
    }
    var tokenizer = try MGT.init(allocator, &.{}, &.{}, null, .english);
    errdefer tokenizer.deinit();
    try tokenizer.loadVocab(tmp_path);
    committed = true;
    deletePath(tmp_path);
    return tokenizer;
}

pub fn loadCheckpointState(allocator: std.mem.Allocator, path: []const u8) !CheckpointState {
    var metadata = try inspectCheckpoint(path);
    var file = try openRead(path);
    defer file.close();
    const stat = try file.stat();
    var reader = Reader{ .file = file, .pos = 0, .size = stat.size };
    var header_metadata = try readHeader(&reader);
    metadata.global_step = header_metadata.global_step;
    var accelerator = try accel.RSFAccelerator.initMultiLayer(metadata.model_dim, metadata.num_layers, allocator);
    var accelerator_committed = false;
    errdefer if (!accelerator_committed) accelerator.deinit();
    const half = metadata.model_dim / 2;
    const columns = half + 1;
    const matrix_len = metadata.rsf_tensor_elements_per_matrix;
    var layer_index: usize = 0;
    while (layer_index < metadata.num_layers) : (layer_index += 1) {
        const ws = try readF16Payload(allocator, &reader, matrix_len);
        defer allocator.free(ws);
        const wt = try readF16Payload(allocator, &reader, matrix_len);
        defer allocator.free(wt);
        const vs = try readF16Payload(allocator, &reader, matrix_len);
        defer allocator.free(vs);
        const vt = try readF16Payload(allocator, &reader, matrix_len);
        defer allocator.free(vt);
        try accelerator.setLayerWeightsS(layer_index, ws, half, columns);
        try accelerator.setLayerWeightsT(layer_index, wt, half, columns);
        try accelerator.setLayerVelocityS(layer_index, vs, half, columns);
        try accelerator.setLayerVelocityT(layer_index, vt, half, columns);
    }
    const clip_min_f32 = try reader.readF32();
    const clip_max_f32 = try reader.readF32();
    const clip_min = try checkedF32ToF16(clip_min_f32);
    const clip_max = try checkedF32ToF16(clip_max_f32);
    try accelerator.setClipRange(clip_min, clip_max);
    try accelerator.sync();
    const embedding_flag = try reader.readByte();
    if (embedding_flag > 1) return CheckpointError.InvalidCheckpointEmbeddingFlag;
    if (embedding_flag == 0) return CheckpointError.MissingCheckpointEmbedding;
    const embedding_vocab = try castU64ToUsize(try reader.readInt(u64));
    const embedding_dim = try castU64ToUsize(try reader.readInt(u64));
    if (embedding_vocab != metadata.vocab_size or embedding_dim != metadata.model_dim) return CheckpointError.InvalidCheckpointDimension;
    var embedding = try LearnedEmbedding.initZero(allocator, embedding_vocab, embedding_dim);
    var embedding_committed = false;
    errdefer if (!embedding_committed) embedding.deinit();
    const weight_len = try castU64ToUsize(try reader.readInt(u64));
    if (weight_len != embedding.weight.data.len) return CheckpointError.InvalidCheckpointLength;
    for (embedding.weight.data) |*w| {
        const value = try reader.readF32();
        try finiteF32(value);
        w.* = value;
    }
    const velocity_len = try castU64ToUsize(try reader.readInt(u64));
    if (velocity_len != embedding.velocity.data.len) return CheckpointError.InvalidCheckpointLength;
    for (embedding.velocity.data) |*v| {
        const value = try reader.readF32();
        try finiteF32(value);
        v.* = value;
    }
    @memset(embedding.grad.data, 0.0);
    try skipGraph(&reader, &header_metadata);
    const tokenizer_len = try reader.readInt(u64);
    if (tokenizer_len == 0 or tokenizer_len > MAX_TOKENIZER_BYTES) return CheckpointError.InvalidCheckpointTokenizer;
    const tokenizer_len_usize = try castU64ToUsize(tokenizer_len);
    const tokenizer_bytes = try allocator.alloc(u8, tokenizer_len_usize);
    defer allocator.free(tokenizer_bytes);
    try reader.readNoEof(tokenizer_bytes);
    const trailer = try reader.readInt(u32);
    if (trailer != CHECKPOINT_TRAILER) return CheckpointError.InvalidCheckpointTrailer;
    if (reader.pos != reader.size) return CheckpointError.TrailingCheckpointData;
    var tokenizer = try loadTokenizerFromBytes(allocator, path, tokenizer_bytes);
    var tokenizer_committed = false;
    errdefer if (!tokenizer_committed) tokenizer.deinit();
    if (@as(usize, tokenizer.next_token_id) != metadata.vocab_size) return CheckpointError.InvalidCheckpointTokenizer;
    accelerator_committed = true;
    embedding_committed = true;
    tokenizer_committed = true;
    return .{
        .allocator = allocator,
        .metadata = metadata,
        .tokenizer = tokenizer,
        .accelerator = accelerator,
        .embedding = embedding,
    };
}

pub fn extractTokenizerData(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const metadata = try inspectCheckpoint(path);
    var file = try openRead(path);
    defer file.close();
    try file.seekTo(metadata.tokenizer_offset);
    const len = try castU64ToUsize(metadata.tokenizer_size);
    const bytes = try allocator.alloc(u8, len);
    errdefer allocator.free(bytes);
    try file.reader().readNoEof(bytes);
    return bytes;
}

pub fn sha256FileHex(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try openRead(path);
    defer file.close();
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [65536]u8 = undefined;
    while (true) {
        const n = try file.read(buffer[0..]);
        if (n == 0) break;
        hasher.update(buffer[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const hex = try allocator.alloc(u8, 64);
    const alphabet = "0123456789abcdef";
    for (digest, 0..) |byte, i| {
        hex[i * 2] = alphabet[byte >> 4];
        hex[i * 2 + 1] = alphabet[byte & 0x0f];
    }
    return hex;
}

pub fn writeMetadataJson(writer: anytype, metadata: CheckpointMetadata) !void {
    try writer.writeAll("{");
    try writer.print("\"magic\":\"{s}\",\"version\":{d},\"global_step\":{d},\"model_dim\":{d},\"num_layers\":{d},\"vocabulary_size\":{d},\"stored_batch_size\":{d},\"learning_rate\":{d},\"momentum\":{d},\"has_embedding\":{s},\"embedding_vocab_size\":{d},\"embedding_dim\":{d},\"nsir_node_count\":{d},\"nsir_edge_group_count\":{d},\"nsir_edge_count\":{d},\"tokenizer_size\":{d},\"file_size\":{d},\"clip_min\":{d},\"clip_max\":{d},\"integrity_ok\":{s}", .{
        metadata.magic[0..],
        metadata.version,
        metadata.global_step,
        metadata.model_dim,
        metadata.num_layers,
        metadata.vocab_size,
        metadata.stored_batch_size,
        metadata.learning_rate,
        metadata.momentum,
        if (metadata.has_embedding) "true" else "false",
        metadata.embedding_vocab_size,
        metadata.embedding_dim,
        metadata.nsir_node_count,
        metadata.nsir_edge_group_count,
        metadata.nsir_edge_count,
        metadata.tokenizer_size,
        metadata.file_size,
        metadata.clip_min,
        metadata.clip_max,
        if (metadata.integrity_ok) "true" else "false",
    });
    try writer.writeAll("}");
}
