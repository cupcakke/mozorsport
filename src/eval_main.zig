const std = @import("std");
const checkpoint = @import("core/checkpoint.zig");
const accel = @import("hw/accel/accel_interface.zig");
const LearnedEmbedding = @import("core/learned_embedding.zig").LearnedEmbedding;

const EvalError = error{
    MissingCheckpoint,
    MissingDataset,
    MissingOutput,
    InvalidArgument,
    InvalidDataset,
    InvalidTextField,
    EmptyCheckpointList,
    NumericalFailure,
    TokenizerMutationDetected,
    InvalidTokenId,
    ReproducibilityFailure,
    GenerationOutputRequired,
};

const DatasetRecord = struct {
    id: ?[]u8,
    category: ?[]u8,
    text: []u8,

    fn deinit(self: *DatasetRecord, allocator: std.mem.Allocator) void {
        if (self.id) |v| allocator.free(v);
        if (self.category) |v| allocator.free(v);
        allocator.free(self.text);
    }
};

const PromptRecord = struct {
    id: ?[]u8,
    category: ?[]u8,
    reference: ?[]u8,
    prompt: []u8,

    fn deinit(self: *PromptRecord, allocator: std.mem.Allocator) void {
        if (self.id) |v| allocator.free(v);
        if (self.category) |v| allocator.free(v);
        if (self.reference) |v| allocator.free(v);
        allocator.free(self.prompt);
    }
};

const TokenizedSample = struct {
    tokens: []u32,
    pred_len: usize,
    truncated: bool,

    fn deinit(self: *TokenizedSample, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
    }
};

const TimingStats = struct {
    values: std.ArrayList(f64),

    fn init(allocator: std.mem.Allocator) TimingStats {
        return .{ .values = std.ArrayList(f64).init(allocator) };
    }

    fn deinit(self: *TimingStats) void {
        self.values.deinit();
    }

    fn add(self: *TimingStats, value: f64) !void {
        if (!std.math.isFinite(value) or value < 0.0) return EvalError.NumericalFailure;
        try self.values.append(value);
    }

    fn percentile(sorted: []const f64, p: f64) ?f64 {
        if (sorted.len == 0) return null;
        if (sorted.len == 1) return sorted[0];
        const pos = p * @as(f64, @floatFromInt(sorted.len - 1));
        const lo_f = @floor(pos);
        const hi_f = @ceil(pos);
        const lo: usize = @intFromFloat(lo_f);
        const hi: usize = @intFromFloat(hi_f);
        if (lo == hi) return sorted[lo];
        const t = pos - lo_f;
        return sorted[lo] * (1.0 - t) + sorted[hi] * t;
    }

    fn writeJson(self: *const TimingStats, writer: anytype, allocator: std.mem.Allocator) !void {
        if (self.values.items.len == 0) {
            try writer.writeAll("null");
            return;
        }
        const copy = try allocator.dupe(f64, self.values.items);
        defer allocator.free(copy);
        std.mem.sort(f64, copy, {}, struct {
            fn lessThan(_: void, a: f64, b: f64) bool {
                return a < b;
            }
        }.lessThan);
        var sum: f64 = 0.0;
        const min_v: f64 = copy[0];
        const max_v: f64 = copy[copy.len - 1];
        for (copy) |v| sum += v;
        const mean = sum / @as(f64, @floatFromInt(copy.len));
        var var_sum: f64 = 0.0;
        for (copy) |v| {
            const d = v - mean;
            var_sum += d * d;
        }
        const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(copy.len)));
        try writer.print("{{\"count\":{d},\"mean_ms\":{d},\"median_ms\":{d},\"min_ms\":{d},\"max_ms\":{d},\"stddev_ms\":{d},\"p50_ms\":{d},\"p90_ms\":", .{ copy.len, mean, percentile(copy, 0.5).?, min_v, max_v, stddev, percentile(copy, 0.5).? });
        if (percentile(copy, 0.90)) |v| try writer.print("{d}", .{v}) else try writer.writeAll("null");
        try writer.writeAll(",\"p95_ms\":");
        if (percentile(copy, 0.95)) |v| try writer.print("{d}", .{v}) else try writer.writeAll("null");
        try writer.writeAll(",\"p99_ms\":");
        if (percentile(copy, 0.99)) |v| try writer.print("{d}", .{v}) else try writer.writeAll("null");
        try writer.writeAll("}");
    }
};

const EvalMetrics = struct {
    total_input_samples: u64 = 0,
    evaluated_samples: u64 = 0,
    skipped_samples: u64 = 0,
    skipped_empty_samples: u64 = 0,
    skipped_short_samples: u64 = 0,
    truncated_samples: u64 = 0,
    valid_target_tokens: u64 = 0,
    embedding_l2_loss_sum: f64 = 0.0,
    top1_count: u64 = 0,
    top5_count: u64 = 0,
    top10_count: u64 = 0,
    reciprocal_rank_sum: f64 = 0.0,
    rank_sum: f64 = 0.0,
    elapsed_seconds: f64 = 0.0,
    tokenizer_seconds: f64 = 0.0,
    forward_seconds: f64 = 0.0,
    ranking_seconds: f64 = 0.0,
    tokens_per_second: f64 = 0.0,
    samples_per_second: f64 = 0.0,
    activation_hash: [32]u8 = [_]u8{0} ** 32,
    tokenization_hash: [32]u8 = [_]u8{0} ** 32,
    ranking_hash: [32]u8 = [_]u8{0} ** 32,

    fn embeddingLoss(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return self.embedding_l2_loss_sum / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top1(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top1_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top5(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top5_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top10(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top10_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn mrr(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return self.reciprocal_rank_sum / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn meanRank(self: *const EvalMetrics) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return self.rank_sum / @as(f64, @floatFromInt(self.valid_target_tokens));
    }
};

const CollapseDiagnostics = struct {
    mean_norm: f64 = 0.0,
    stddev_norm: f64 = 0.0,
    min_norm: f64 = 0.0,
    max_norm: f64 = 0.0,
    zero_or_near_zero_vectors: u64 = 0,
    per_dimension_variance_mean: f64 = 0.0,
    per_dimension_variance_stddev: f64 = 0.0,
    per_dimension_variance_min: f64 = 0.0,
    per_dimension_variance_max: f64 = 0.0,
    sampled_pair_count: u64 = 0,
    sampled_mean_cosine_similarity: f64 = 0.0,
    sampled_cosine_similarity_stddev: f64 = 0.0,
    effective_rank: f64 = 0.0,
    near_duplicate_pair_rate: f64 = 0.0,
    predicted_token_entropy: f64 = 0.0,
    most_frequent_prediction_percentage: f64 = 0.0,
    unique_predicted_token_count: u64 = 0,
    suspicious_low_norm: bool = false,
    suspicious_low_effective_rank: bool = false,
    suspicious_prediction_collapse: bool = false,
};

const MemoryMetrics = struct {
    serialized_checkpoint_bytes: u64 = 0,
    checkpoint_payload_bytes: u64 = 0,
    rsf_weight_bytes: u64 = 0,
    rsf_velocity_bytes: u64 = 0,
    embedding_weight_bytes: u64 = 0,
    embedding_velocity_bytes: u64 = 0,
    embedding_gradient_bytes: u64 = 0,
    activation_peak_bytes: u64 = 0,
    ranking_buffer_peak_bytes: u64 = 0,
    host_model_state_bytes: u64 = 0,
    tracked_device_allocation_peak_bytes: ?u64 = null,
};

const ReproducibilityReport = struct {
    enabled: bool = false,
    runs: u32 = 1,
    passed: bool = true,
    max_absolute_metric_difference: f64 = 0.0,
    max_relative_metric_difference: f64 = 0.0,
    tokenization_hashes_match: bool = true,
    activation_hashes_match: bool = true,
    ranking_hashes_match: bool = true,
};

const CheckpointReport = struct {
    path: []const u8,
    digest: []u8,
    metadata: checkpoint.CheckpointMetadata,
    metrics: EvalMetrics,
    diagnostics: CollapseDiagnostics,
    memory: MemoryMetrics,
    reproducibility: ReproducibilityReport,
    load_seconds: f64,
    batch_latency: TimingStats,
    success: bool,
    error_name: ?[]const u8,

    fn deinit(self: *CheckpointReport, allocator: std.mem.Allocator) void {
        allocator.free(self.digest);
        self.batch_latency.deinit();
    }
};

const Args = struct {
    allocator: std.mem.Allocator,
    checkpoints: std.ArrayList([]const u8),
    dataset: ?[]const u8 = null,
    prompts: ?[]const u8 = null,
    generation_output: ?[]const u8 = null,
    output: ?[]const u8 = null,
    batch_size: usize = 32,
    max_sequence_length: usize = 512,
    reproducibility_runs: u32 = 1,
    seed: u64 = 42,
    ranking_chunk_size: usize = 4096,
    generation_max_tokens: usize = 32,
    abs_tolerance: f64 = 1e-6,
    rel_tolerance: f64 = 1e-6,
    selection_metric: []const u8 = "embedding_l2_loss",
    selection_metric_owned: bool = false,

    fn init(allocator: std.mem.Allocator) Args {
        return .{ .allocator = allocator, .checkpoints = std.ArrayList([]const u8).init(allocator) };
    }

    fn deinit(self: *Args) void {
        for (self.checkpoints.items) |p| self.allocator.free(p);
        self.checkpoints.deinit();
        if (self.dataset) |p| self.allocator.free(p);
        if (self.prompts) |p| self.allocator.free(p);
        if (self.generation_output) |p| self.allocator.free(p);
        if (self.output) |p| self.allocator.free(p);
        if (self.selection_metric_owned) self.allocator.free(self.selection_metric);
    }
};

fn jsonEscape(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0...7, 11, 12, 14...31 => try writer.print("\\u{x:0>4}", .{c}),
            else => try writer.writeByte(c),
        }
    }
    try writer.writeByte('"');
}

fn hashInt(comptime T: type, hasher: *std.crypto.hash.sha2.Sha256, value: T) void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    hasher.update(bytes[0..]);
}

fn hashFloat64(hasher: *std.crypto.hash.sha2.Sha256, value: f64) void {
    hashInt(u64, hasher, @as(u64, @bitCast(value)));
}

fn hexDigest(allocator: std.mem.Allocator, digest: [32]u8) ![]u8 {
    const out = try allocator.alloc(u8, 64);
    const alphabet = "0123456789abcdef";
    for (digest, 0..) |byte, i| {
        out[i * 2] = alphabet[byte >> 4];
        out[i * 2 + 1] = alphabet[byte & 0x0f];
    }
    return out;
}

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var parsed = Args.init(allocator);
    errdefer parsed.deinit();
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            try parsed.checkpoints.append(try allocator.dupe(u8, argv[i]));
        } else if (std.mem.eql(u8, arg, "--dataset")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.dataset = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--prompts")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.prompts = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--generation-output")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.generation_output = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--output")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.output = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.batch_size = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-sequence-length")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.max_sequence_length = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--reproducibility-runs")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.reproducibility_runs = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.seed = try std.fmt.parseInt(u64, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--ranking-chunk-size")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.ranking_chunk_size = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--generation-max-tokens")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.generation_max_tokens = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--abs-tolerance")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.abs_tolerance = try std.fmt.parseFloat(f64, argv[i]);
        } else if (std.mem.eql(u8, arg, "--rel-tolerance")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            parsed.rel_tolerance = try std.fmt.parseFloat(f64, argv[i]);
        } else if (std.mem.eql(u8, arg, "--selection-metric")) {
            if (i + 1 >= argv.len) return EvalError.InvalidArgument;
            i += 1;
            if (parsed.selection_metric_owned) allocator.free(parsed.selection_metric);
            parsed.selection_metric = try allocator.dupe(u8, argv[i]);
            parsed.selection_metric_owned = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("jaide-eval --checkpoint PATH [--checkpoint PATH] --dataset PATH --output PATH [--batch-size N] [--max-sequence-length N] [--reproducibility-runs N] [--prompts PATH --generation-output PATH]\n", .{});
            std.process.exit(0);
        } else {
            return EvalError.InvalidArgument;
        }
    }
    if (parsed.checkpoints.items.len == 0) return EvalError.EmptyCheckpointList;
    if (parsed.output == null) return EvalError.MissingOutput;
    if (parsed.dataset == null and parsed.prompts == null) return EvalError.MissingDataset;
    if (parsed.batch_size == 0 or parsed.max_sequence_length == 0 or parsed.reproducibility_runs == 0 or parsed.ranking_chunk_size == 0) return EvalError.InvalidArgument;
    if (!std.math.isFinite(parsed.abs_tolerance) or !std.math.isFinite(parsed.rel_tolerance) or parsed.abs_tolerance < 0.0 or parsed.rel_tolerance < 0.0) return EvalError.InvalidArgument;
    return parsed;
}

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

fn readLinesJsonRecords(allocator: std.mem.Allocator, path: []const u8) ![]DatasetRecord {
    var file = try openRead(path);
    defer file.close();
    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();
    var records = std.ArrayList(DatasetRecord).init(allocator);
    errdefer {
        for (records.items) |*r| r.deinit(allocator);
        records.deinit();
    }
    while (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |line_raw| {
        defer allocator.free(line_raw);
        const line = std.mem.trim(u8, line_raw, " \t\r\n");
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always }) catch return EvalError.InvalidDataset;
        defer parsed.deinit();
        if (parsed.value != .object) return EvalError.InvalidDataset;
        const obj = parsed.value.object;
        const text_value = obj.get("text") orelse return EvalError.InvalidTextField;
        if (text_value != .string) return EvalError.InvalidTextField;
        const id = if (obj.get("id")) |v| switch (v) { .string => |s| try allocator.dupe(u8, s), else => null } else null;
        errdefer if (id) |v| allocator.free(v);
        const category = if (obj.get("category")) |v| switch (v) { .string => |s| try allocator.dupe(u8, s), else => null } else null;
        errdefer if (category) |v| allocator.free(v);
        const text = try allocator.dupe(u8, text_value.string);
        try records.append(.{ .id = id, .category = category, .text = text });
    }
    return try records.toOwnedSlice();
}

fn readPromptRecords(allocator: std.mem.Allocator, path: []const u8) ![]PromptRecord {
    var file = try openRead(path);
    defer file.close();
    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();
    var records = std.ArrayList(PromptRecord).init(allocator);
    errdefer {
        for (records.items) |*r| r.deinit(allocator);
        records.deinit();
    }
    while (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |line_raw| {
        defer allocator.free(line_raw);
        const line = std.mem.trim(u8, line_raw, " \t\r\n");
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always }) catch return EvalError.InvalidDataset;
        defer parsed.deinit();
        if (parsed.value != .object) return EvalError.InvalidDataset;
        const obj = parsed.value.object;
        const prompt_value = obj.get("prompt") orelse return EvalError.InvalidTextField;
        if (prompt_value != .string) return EvalError.InvalidTextField;
        const id = if (obj.get("id")) |v| switch (v) { .string => |s| try allocator.dupe(u8, s), else => null } else null;
        errdefer if (id) |v| allocator.free(v);
        const category = if (obj.get("category")) |v| switch (v) { .string => |s| try allocator.dupe(u8, s), else => null } else null;
        errdefer if (category) |v| allocator.free(v);
        const reference = if (obj.get("reference")) |v| switch (v) { .string => |s| try allocator.dupe(u8, s), else => null } else null;
        errdefer if (reference) |v| allocator.free(v);
        const prompt = try allocator.dupe(u8, prompt_value.string);
        try records.append(.{ .id = id, .category = category, .reference = reference, .prompt = prompt });
    }
    return try records.toOwnedSlice();
}

fn datasetDigest(allocator: std.mem.Allocator, path_opt: ?[]const u8) !?[]u8 {
    if (path_opt == null) return null;
    return try checkpoint.sha256FileHex(allocator, path_opt.?);
}

fn fingerprintState(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState) ![32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashInt(u64, &hasher, state.metadata.global_step);
    hashInt(u32, &hasher, state.tokenizer.next_token_id);
    var layer: usize = 0;
    while (layer < state.metadata.num_layers) : (layer += 1) {
        const ws = try state.accelerator.readLayerWeightsFlat(layer, .weights_s, allocator);
        defer allocator.free(ws);
        for (ws) |v| hashInt(u16, &hasher, @as(u16, @bitCast(v)));
        const wt = try state.accelerator.readLayerWeightsFlat(layer, .weights_t, allocator);
        defer allocator.free(wt);
        for (wt) |v| hashInt(u16, &hasher, @as(u16, @bitCast(v)));
        const vs = try state.accelerator.readLayerWeightsFlat(layer, .velocity_s, allocator);
        defer allocator.free(vs);
        for (vs) |v| hashInt(u16, &hasher, @as(u16, @bitCast(v)));
        const vt = try state.accelerator.readLayerWeightsFlat(layer, .velocity_t, allocator);
        defer allocator.free(vt);
        for (vt) |v| hashInt(u16, &hasher, @as(u16, @bitCast(v)));
    }
    for (state.embedding.weight.data) |v| hashFloat64(&hasher, @as(f64, @floatCast(v)));
    for (state.embedding.velocity.data) |v| hashFloat64(&hasher, @as(f64, @floatCast(v)));
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

fn buildEmbeddingNorms(allocator: std.mem.Allocator, embedding: *const LearnedEmbedding) ![]f64 {
    const norms = try allocator.alloc(f64, embedding.vocab_size);
    errdefer allocator.free(norms);
    var token: usize = 0;
    while (token < embedding.vocab_size) : (token += 1) {
        var sum: f64 = 0.0;
        var d: usize = 0;
        while (d < embedding.dim) : (d += 1) {
            const value = embedding.weight.data[token * embedding.dim + d];
            if (!std.math.isFinite(value)) return EvalError.NumericalFailure;
            sum += @as(f64, value) * @as(f64, value);
        }
        norms[token] = @sqrt(sum);
        if (!std.math.isFinite(norms[token])) return EvalError.NumericalFailure;
    }
    return norms;
}

const RankResult = struct {
    rank: u64,
    best_token: u32,
};

fn cosineScore(pred: []const f32, pred_norm: f64, emb: []const f32, emb_norm: f64) !f64 {
    if (pred_norm <= 1e-12 or emb_norm <= 1e-12) return -std.math.inf(f64);
    var dot: f64 = 0.0;
    for (pred, 0..) |v, i| {
        const e = emb[i];
        if (!std.math.isFinite(v) or !std.math.isFinite(e)) return EvalError.NumericalFailure;
        dot += @as(f64, v) * @as(f64, e);
    }
    const score = dot / (pred_norm * emb_norm);
    if (std.math.isNan(score)) return EvalError.NumericalFailure;
    return score;
}

fn rankPrediction(pred: []const f32, embedding: *const LearnedEmbedding, norms: []const f64, target: u32, chunk_size: usize) !RankResult {
    if (@as(usize, target) >= embedding.vocab_size) return EvalError.InvalidTokenId;
    var pred_norm_sum: f64 = 0.0;
    for (pred) |v| {
        if (!std.math.isFinite(v)) return EvalError.NumericalFailure;
        pred_norm_sum += @as(f64, v) * @as(f64, v);
    }
    const pred_norm = @sqrt(pred_norm_sum);
    if (!std.math.isFinite(pred_norm)) return EvalError.NumericalFailure;
    const target_slice = embedding.weight.data[@as(usize, target) * embedding.dim .. @as(usize, target) * embedding.dim + embedding.dim];
    const target_score = try cosineScore(pred, pred_norm, target_slice, norms[@as(usize, target)]);
    var rank: u64 = 1;
    var best_score: f64 = -std.math.inf(f64);
    var best_token: u32 = 0;
    var start: usize = 0;
    while (start < embedding.vocab_size) : (start += chunk_size) {
        const end = @min(start + chunk_size, embedding.vocab_size);
        var token = start;
        while (token < end) : (token += 1) {
            const emb_slice = embedding.weight.data[token * embedding.dim .. token * embedding.dim + embedding.dim];
            const score = try cosineScore(pred, pred_norm, emb_slice, norms[token]);
            if (score > target_score or (score == target_score and token < @as(usize, target))) rank += 1;
            if (score > best_score or (score == best_score and token < @as(usize, best_token))) {
                best_score = score;
                best_token = @intCast(token);
            }
        }
    }
    return .{ .rank = rank, .best_token = best_token };
}

fn makeTokenizedBatch(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, records: []const DatasetRecord, max_sequence_length: usize, metrics: *EvalMetrics, token_hash: *std.crypto.hash.sha2.Sha256) ![]TokenizedSample {
    var out = std.ArrayList(TokenizedSample).init(allocator);
    errdefer {
        for (out.items) |*s| s.deinit(allocator);
        out.deinit();
    }
    for (records) |record| {
        metrics.total_input_samples += 1;
        if (record.text.len == 0) {
            metrics.skipped_samples += 1;
            metrics.skipped_empty_samples += 1;
            continue;
        }
        var timer = try std.time.Timer.start();
        const vocab_before = state.tokenizer.next_token_id;
        var tokens_list = std.ArrayList(u32).init(allocator);
        errdefer tokens_list.deinit();
        try state.tokenizer.encode(record.text, &tokens_list);
        if (state.tokenizer.next_token_id != vocab_before) return EvalError.TokenizerMutationDetected;
        metrics.tokenizer_seconds += @as(f64, @floatFromInt(timer.read())) / 1.0e9;
        if (tokens_list.items.len < 2) {
            metrics.skipped_samples += 1;
            metrics.skipped_short_samples += 1;
            tokens_list.deinit();
            continue;
        }
        var keep_len = tokens_list.items.len;
        var truncated = false;
        const max_tokens = max_sequence_length + 1;
        if (keep_len > max_tokens) {
            keep_len = max_tokens;
            truncated = true;
            metrics.truncated_samples += 1;
        }
        const owned = try allocator.dupe(u32, tokens_list.items[0..keep_len]);
        tokens_list.deinit();
        for (owned) |t| hashInt(u32, token_hash, t);
        try out.append(.{ .tokens = owned, .pred_len = keep_len - 1, .truncated = truncated });
        metrics.evaluated_samples += 1;
    }
    return try out.toOwnedSlice();
}

fn runForwardBatch(
    allocator: std.mem.Allocator,
    state: *checkpoint.CheckpointState,
    samples: []const TokenizedSample,
    embedding_norms: []const f64,
    ranking_chunk_size: usize,
    metrics: *EvalMetrics,
    pred_counts: *std.AutoHashMap(u32, u64),
    activation_hash: *std.crypto.hash.sha2.Sha256,
    ranking_hash: *std.crypto.hash.sha2.Sha256,
    batch_latency: *TimingStats,
    memory: *MemoryMetrics,
    collect: bool,
) !void {
    if (samples.len == 0) return;
    var max_pred_len: usize = 0;
    for (samples) |s| max_pred_len = @max(max_pred_len, s.pred_len);
    if (max_pred_len == 0) return;
    const model_dim = state.metadata.model_dim;
    const rows = try std.math.mul(usize, samples.len, max_pred_len);
    const elements = try std.math.mul(usize, rows, model_dim);
    const activation_bytes = try std.math.mul(usize, elements, @sizeOf(f16));
    memory.activation_peak_bytes = @max(memory.activation_peak_bytes, @as(u64, @intCast(activation_bytes)));
    const input_data = try allocator.alloc(f16, elements);
    defer allocator.free(input_data);
    @memset(input_data, @as(f16, 0.0));
    for (samples, 0..) |sample, batch_index| {
        var pos: usize = 0;
        while (pos < sample.pred_len) : (pos += 1) {
            const token = sample.tokens[pos];
            if (@as(usize, token) >= state.embedding.vocab_size) return EvalError.InvalidTokenId;
            const source = state.embedding.weight.data[@as(usize, token) * model_dim .. @as(usize, token) * model_dim + model_dim];
            const base = (batch_index * max_pred_len + pos) * model_dim;
            var d: usize = 0;
            while (d < model_dim) : (d += 1) {
                const value = source[d];
                if (!std.math.isFinite(value)) return EvalError.NumericalFailure;
                input_data[base + d] = @floatCast(value);
            }
        }
    }
    var batch_timer = try std.time.Timer.start();
    var inputs = try accel.FutharkArray3DF16.newFromFlat(&state.accelerator.ctx, input_data, samples.len, max_pred_len, model_dim);
    defer inputs.free(&state.accelerator.ctx);
    try state.accelerator.sync();
    var forward_timer = try std.time.Timer.start();
    var outputs = try state.accelerator.forwardBatch(&inputs);
    defer outputs.free(&state.accelerator.ctx);
    try state.accelerator.sync();
    const forward_ns = forward_timer.read();
    if (collect) metrics.forward_seconds += @as(f64, @floatFromInt(forward_ns)) / 1.0e9;
    const output_data_f16 = try outputs.valuesFlat(&state.accelerator.ctx, allocator);
    defer allocator.free(output_data_f16);
    const output_data = try allocator.alloc(f32, output_data_f16.len);
    defer allocator.free(output_data);
    for (output_data_f16, 0..) |v, i| {
        hashInt(u16, activation_hash, @as(u16, @bitCast(v)));
        const f: f32 = @floatCast(v);
        if (!std.math.isFinite(f)) return EvalError.NumericalFailure;
        output_data[i] = f;
    }
    var ranking_timer = try std.time.Timer.start();
    const ranking_bytes = try std.math.mul(usize, ranking_chunk_size, @sizeOf(f64));
    memory.ranking_buffer_peak_bytes = @max(memory.ranking_buffer_peak_bytes, @as(u64, @intCast(ranking_bytes)));
    for (samples, 0..) |sample, batch_index| {
        var pos: usize = 0;
        while (pos < sample.pred_len) : (pos += 1) {
            const target = sample.tokens[pos + 1];
            if (target == 0) continue;
            if (@as(usize, target) >= state.embedding.vocab_size) return EvalError.InvalidTokenId;
            const base = (batch_index * max_pred_len + pos) * model_dim;
            const pred = output_data[base .. base + model_dim];
            const target_vec = state.embedding.weight.data[@as(usize, target) * model_dim .. @as(usize, target) * model_dim + model_dim];
            var mse: f64 = 0.0;
            var d: usize = 0;
            while (d < model_dim) : (d += 1) {
                const diff = @as(f64, pred[d]) - @as(f64, target_vec[d]);
                if (!std.math.isFinite(diff)) return EvalError.NumericalFailure;
                mse += diff * diff;
            }
            mse /= @as(f64, @floatFromInt(model_dim));
            if (!std.math.isFinite(mse)) return EvalError.NumericalFailure;
            const rr = try rankPrediction(pred, &state.embedding, embedding_norms, target, ranking_chunk_size);
            if (collect) {
                metrics.valid_target_tokens += 1;
                metrics.embedding_l2_loss_sum += mse;
                if (rr.rank <= 1) metrics.top1_count += 1;
                if (rr.rank <= 5) metrics.top5_count += 1;
                if (rr.rank <= 10) metrics.top10_count += 1;
                metrics.reciprocal_rank_sum += 1.0 / @as(f64, @floatFromInt(rr.rank));
                metrics.rank_sum += @as(f64, @floatFromInt(rr.rank));
                const entry = try pred_counts.getOrPut(rr.best_token);
                if (entry.found_existing) entry.value_ptr.* += 1 else entry.value_ptr.* = 1;
                hashInt(u32, ranking_hash, rr.best_token);
                hashInt(u64, ranking_hash, rr.rank);
                hashFloat64(ranking_hash, mse);
            }
        }
    }
    const ranking_ns = ranking_timer.read();
    if (collect) metrics.ranking_seconds += @as(f64, @floatFromInt(ranking_ns)) / 1.0e9;
    const batch_ms = @as(f64, @floatFromInt(batch_timer.read())) / 1.0e6;
    if (collect) try batch_latency.add(batch_ms);
}

fn warmup(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, records: []const DatasetRecord, max_sequence_length: usize, ranking_chunk_size: usize, memory: *MemoryMetrics) !void {
    if (records.len == 0) return;
    var metrics = EvalMetrics{};
    var token_hash = std.crypto.hash.sha2.Sha256.init(.{});
    const end = @min(records.len, @as(usize, 1));
    const samples = try makeTokenizedBatch(allocator, state, records[0..end], max_sequence_length, &metrics, &token_hash);
    defer {
        for (samples) |*s| s.deinit(allocator);
        allocator.free(samples);
    }
    if (samples.len == 0) return;
    const norms = try buildEmbeddingNorms(allocator, &state.embedding);
    defer allocator.free(norms);
    var pred_counts = std.AutoHashMap(u32, u64).init(allocator);
    defer pred_counts.deinit();
    var activation_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var ranking_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var timing = TimingStats.init(allocator);
    defer timing.deinit();
    try runForwardBatch(allocator, state, samples, norms, ranking_chunk_size, &metrics, &pred_counts, &activation_hash, &ranking_hash, &timing, memory, false);
}

fn computeDiagnostics(allocator: std.mem.Allocator, embedding: *const LearnedEmbedding, norms: []const f64, pred_counts: *std.AutoHashMap(u32, u64), seed: u64) !CollapseDiagnostics {
    var diag = CollapseDiagnostics{};
    if (embedding.vocab_size == 0 or embedding.dim == 0) return diag;
    var sum: f64 = 0.0;
    var min_norm = norms[0];
    var max_norm = norms[0];
    for (norms) |n| {
        if (!std.math.isFinite(n)) return EvalError.NumericalFailure;
        sum += n;
        min_norm = @min(min_norm, n);
        max_norm = @max(max_norm, n);
        if (n <= 1e-8) diag.zero_or_near_zero_vectors += 1;
    }
    diag.mean_norm = sum / @as(f64, @floatFromInt(norms.len));
    var norm_var: f64 = 0.0;
    for (norms) |n| {
        const d = n - diag.mean_norm;
        norm_var += d * d;
    }
    diag.stddev_norm = @sqrt(norm_var / @as(f64, @floatFromInt(norms.len)));
    diag.min_norm = min_norm;
    diag.max_norm = max_norm;
    const means = try allocator.alloc(f64, embedding.dim);
    defer allocator.free(means);
    @memset(means, 0.0);
    var token: usize = 0;
    while (token < embedding.vocab_size) : (token += 1) {
        var d: usize = 0;
        while (d < embedding.dim) : (d += 1) means[d] += embedding.weight.data[token * embedding.dim + d];
    }
    var d: usize = 0;
    while (d < embedding.dim) : (d += 1) means[d] /= @as(f64, @floatFromInt(embedding.vocab_size));
    const variances = try allocator.alloc(f64, embedding.dim);
    defer allocator.free(variances);
    @memset(variances, 0.0);
    token = 0;
    while (token < embedding.vocab_size) : (token += 1) {
        d = 0;
        while (d < embedding.dim) : (d += 1) {
            const diff = @as(f64, embedding.weight.data[token * embedding.dim + d]) - means[d];
            variances[d] += diff * diff;
        }
    }
    d = 0;
    var variance_sum: f64 = 0.0;
    var variance_min: f64 = variances[0] / @as(f64, @floatFromInt(embedding.vocab_size));
    var variance_max: f64 = variance_min;
    while (d < embedding.dim) : (d += 1) {
        variances[d] /= @as(f64, @floatFromInt(embedding.vocab_size));
        variance_sum += variances[d];
        variance_min = @min(variance_min, variances[d]);
        variance_max = @max(variance_max, variances[d]);
    }
    diag.per_dimension_variance_mean = variance_sum / @as(f64, @floatFromInt(embedding.dim));
    diag.per_dimension_variance_min = variance_min;
    diag.per_dimension_variance_max = variance_max;
    var variance_std_sum: f64 = 0.0;
    for (variances) |v| {
        const dv = v - diag.per_dimension_variance_mean;
        variance_std_sum += dv * dv;
    }
    diag.per_dimension_variance_stddev = @sqrt(variance_std_sum / @as(f64, @floatFromInt(embedding.dim)));
    if (variance_sum > 0.0) {
        var entropy: f64 = 0.0;
        for (variances) |v| {
            if (v > 0.0) {
                const p = v / variance_sum;
                entropy -= p * @log(p);
            }
        }
        diag.effective_rank = @exp(entropy);
    }
    var rng = std.Random.DefaultPrng.init(seed);
    const rnd = rng.random();
    const pair_count: usize = @min(@as(usize, 4096), if (embedding.vocab_size > 1) embedding.vocab_size * 16 else 0);
    var cos_sum: f64 = 0.0;
    var cos_sq_sum: f64 = 0.0;
    var duplicates: u64 = 0;
    var pair_index: usize = 0;
    while (pair_index < pair_count) : (pair_index += 1) {
        const a = rnd.uintLessThan(usize, embedding.vocab_size);
        var b = rnd.uintLessThan(usize, embedding.vocab_size - 1);
        if (b >= a) b += 1;
        const av = embedding.weight.data[a * embedding.dim .. a * embedding.dim + embedding.dim];
        const bv = embedding.weight.data[b * embedding.dim .. b * embedding.dim + embedding.dim];
        const cosine = try cosineScore(av, norms[a], bv, norms[b]);
        if (std.math.isFinite(cosine)) {
            cos_sum += cosine;
            cos_sq_sum += cosine * cosine;
            if (cosine >= 0.999999) duplicates += 1;
        }
    }
    diag.sampled_pair_count = pair_count;
    if (pair_count > 0) {
        diag.sampled_mean_cosine_similarity = cos_sum / @as(f64, @floatFromInt(pair_count));
        const mean_sq = cos_sq_sum / @as(f64, @floatFromInt(pair_count));
        diag.sampled_cosine_similarity_stddev = @sqrt(@max(0.0, mean_sq - diag.sampled_mean_cosine_similarity * diag.sampled_mean_cosine_similarity));
        diag.near_duplicate_pair_rate = @as(f64, @floatFromInt(duplicates)) / @as(f64, @floatFromInt(pair_count));
    }
    var total_predictions: u64 = 0;
    var max_count: u64 = 0;
    var it = pred_counts.iterator();
    while (it.next()) |entry| {
        total_predictions += entry.value_ptr.*;
        max_count = @max(max_count, entry.value_ptr.*);
    }
    diag.unique_predicted_token_count = pred_counts.count();
    if (total_predictions > 0) {
        it = pred_counts.iterator();
        while (it.next()) |entry| {
            const p = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(total_predictions));
            if (p > 0.0) diag.predicted_token_entropy -= p * @log(p);
        }
        diag.most_frequent_prediction_percentage = 100.0 * @as(f64, @floatFromInt(max_count)) / @as(f64, @floatFromInt(total_predictions));
    }
    diag.suspicious_low_norm = diag.mean_norm <= 1e-6 or diag.zero_or_near_zero_vectors == embedding.vocab_size;
    diag.suspicious_low_effective_rank = diag.effective_rank > 0.0 and diag.effective_rank < @min(@as(f64, @floatFromInt(embedding.dim)) * 0.05, 2.0);
    diag.suspicious_prediction_collapse = total_predictions > 0 and diag.most_frequent_prediction_percentage >= 50.0;
    return diag;
}

fn evaluateOnce(allocator: std.mem.Allocator, path: []const u8, records: []const DatasetRecord, args: Args, load_seconds_out: *f64, memory_out: *MemoryMetrics, batch_latency: *TimingStats) !struct { metrics: EvalMetrics, diagnostics: CollapseDiagnostics } {
    var load_timer = try std.time.Timer.start();
    var state = try checkpoint.loadCheckpointState(allocator, path);
    defer state.deinit();
    load_seconds_out.* = @as(f64, @floatFromInt(load_timer.read())) / 1.0e9;
    memory_out.serialized_checkpoint_bytes = state.metadata.file_size;
    memory_out.checkpoint_payload_bytes = state.metadata.checkpoint_payload_bytes;
    memory_out.rsf_weight_bytes = state.metadata.rsf_weight_bytes;
    memory_out.rsf_velocity_bytes = state.metadata.rsf_velocity_bytes;
    memory_out.embedding_weight_bytes = state.metadata.embedding_weight_bytes;
    memory_out.embedding_velocity_bytes = state.metadata.embedding_velocity_bytes;
    memory_out.embedding_gradient_bytes = @as(u64, @intCast(state.embedding.grad.data.len * @sizeOf(f32)));
    memory_out.host_model_state_bytes = memory_out.rsf_weight_bytes + memory_out.rsf_velocity_bytes + memory_out.embedding_weight_bytes + memory_out.embedding_velocity_bytes + memory_out.embedding_gradient_bytes;
    const state_fingerprint_before = try fingerprintState(allocator, &state);
    try warmup(allocator, &state, records, args.max_sequence_length, args.ranking_chunk_size, memory_out);
    const norms = try buildEmbeddingNorms(allocator, &state.embedding);
    defer allocator.free(norms);
    var pred_counts = std.AutoHashMap(u32, u64).init(allocator);
    defer pred_counts.deinit();
    var metrics = EvalMetrics{};
    var token_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var activation_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var ranking_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var eval_timer = try std.time.Timer.start();
    var start: usize = 0;
    while (start < records.len) : (start += args.batch_size) {
        const end = @min(start + args.batch_size, records.len);
        const samples = try makeTokenizedBatch(allocator, &state, records[start..end], args.max_sequence_length, &metrics, &token_hash);
        defer {
            for (samples) |*s| s.deinit(allocator);
            allocator.free(samples);
        }
        try runForwardBatch(allocator, &state, samples, norms, args.ranking_chunk_size, &metrics, &pred_counts, &activation_hash, &ranking_hash, batch_latency, memory_out, true);
    }
    metrics.elapsed_seconds = @as(f64, @floatFromInt(eval_timer.read())) / 1.0e9;
    if (metrics.elapsed_seconds > 0.0) {
        metrics.tokens_per_second = @as(f64, @floatFromInt(metrics.valid_target_tokens)) / metrics.elapsed_seconds;
        metrics.samples_per_second = @as(f64, @floatFromInt(metrics.evaluated_samples)) / metrics.elapsed_seconds;
    }
    token_hash.final(&metrics.tokenization_hash);
    activation_hash.final(&metrics.activation_hash);
    ranking_hash.final(&metrics.ranking_hash);
    if (metrics.valid_target_tokens == 0 and records.len > 0) return EvalError.InvalidDataset;
    const diagnostics = try computeDiagnostics(allocator, &state.embedding, norms, &pred_counts, args.seed);
    const state_fingerprint_after = try fingerprintState(allocator, &state);
    if (!std.mem.eql(u8, state_fingerprint_before[0..], state_fingerprint_after[0..])) return EvalError.ReproducibilityFailure;
    return .{ .metrics = metrics, .diagnostics = diagnostics };
}

fn relativeDiff(a: f64, b: f64) f64 {
    const denom = @max(@abs(a), @abs(b));
    if (denom <= 1e-30) return @abs(a - b);
    return @abs(a - b) / denom;
}

fn compareRepro(base: EvalMetrics, other: EvalMetrics, args: Args, report: *ReproducibilityReport) void {
    const fields = [_]struct { a: ?f64, b: ?f64 }{
        .{ .a = base.embeddingLoss(), .b = other.embeddingLoss() },
        .{ .a = base.top1(), .b = other.top1() },
        .{ .a = base.top5(), .b = other.top5() },
        .{ .a = base.top10(), .b = other.top10() },
        .{ .a = base.mrr(), .b = other.mrr() },
    };
    for (fields) |f| {
        if (f.a == null or f.b == null) {
            report.passed = false;
            continue;
        }
        const av = f.a.?;
        const bv = f.b.?;
        const abs_d = @abs(av - bv);
        const rel_d = relativeDiff(av, bv);
        report.max_absolute_metric_difference = @max(report.max_absolute_metric_difference, abs_d);
        report.max_relative_metric_difference = @max(report.max_relative_metric_difference, rel_d);
        if (abs_d > args.abs_tolerance and rel_d > args.rel_tolerance) report.passed = false;
    }
    if (!std.mem.eql(u8, base.tokenization_hash[0..], other.tokenization_hash[0..])) {
        report.tokenization_hashes_match = false;
        report.passed = false;
    }
    if (!std.mem.eql(u8, base.activation_hash[0..], other.activation_hash[0..])) {
        report.activation_hashes_match = false;
        report.passed = false;
    }
    if (!std.mem.eql(u8, base.ranking_hash[0..], other.ranking_hash[0..])) {
        report.ranking_hashes_match = false;
        report.passed = false;
    }
}

fn allocateEmptyDigestOrExit(allocator: std.mem.Allocator) []u8 {
    return allocator.dupe(u8, "") catch {
        std.debug.print("fatal: out of memory while allocating digest storage\n", .{});
        std.process.exit(1);
    };
}

fn evaluateCheckpoint(allocator: std.mem.Allocator, path: []const u8, records: []const DatasetRecord, args: Args) CheckpointReport {
    var report = CheckpointReport{
        .path = path,
        .digest = checkpoint.sha256FileHex(allocator, path) catch allocateEmptyDigestOrExit(allocator),
        .metadata = checkpoint.inspectCheckpoint(path) catch checkpoint.CheckpointMetadata{
            .magic = checkpoint.CHECKPOINT_MAGIC,
            .version = 0,
            .global_step = 0,
            .model_dim = 0,
            .num_layers = 0,
            .vocab_size = 0,
            .stored_batch_size = 0,
            .learning_rate = 0,
            .momentum = 0,
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
            .file_size = 0,
            .rsf_tensor_elements_per_matrix = 0,
            .rsf_weight_bytes = 0,
            .rsf_velocity_bytes = 0,
            .embedding_weight_bytes = 0,
            .embedding_velocity_bytes = 0,
            .checkpoint_payload_bytes = 0,
            .clip_min = 0,
            .clip_max = 0,
            .integrity_ok = false,
        },
        .metrics = .{},
        .diagnostics = .{},
        .memory = .{},
        .reproducibility = .{ .enabled = args.reproducibility_runs > 1, .runs = args.reproducibility_runs },
        .load_seconds = 0.0,
        .batch_latency = TimingStats.init(allocator),
        .success = false,
        .error_name = null,
    };
    var run: u32 = 0;
    var base_metrics: ?EvalMetrics = null;
    while (run < args.reproducibility_runs) : (run += 1) {
        var run_load_seconds: f64 = 0.0;
        var run_memory = MemoryMetrics{};
        var run_timing = TimingStats.init(allocator);
        defer if (run != 0) run_timing.deinit();
        const result = evaluateOnce(allocator, path, records, args, &run_load_seconds, &run_memory, if (run == 0) &report.batch_latency else &run_timing) catch |err| {
            report.error_name = @errorName(err);
            report.success = false;
            report.reproducibility.passed = false;
            return report;
        };
        if (run == 0) {
            report.metrics = result.metrics;
            report.diagnostics = result.diagnostics;
            report.memory = run_memory;
            report.load_seconds = run_load_seconds;
            base_metrics = result.metrics;
        } else if (base_metrics) |base| {
            compareRepro(base, result.metrics, args, &report.reproducibility);
        }
    }
    report.success = report.reproducibility.passed;
    if (!report.success and report.error_name == null) report.error_name = "ReproducibilityFailure";
    return report;
}

fn metricValue(report: *const CheckpointReport, metric: []const u8) ?f64 {
    if (!report.success or !report.metadata.integrity_ok) return null;
    if (std.mem.eql(u8, metric, "embedding_l2_loss")) return report.metrics.embeddingLoss();
    if (std.mem.eql(u8, metric, "top1")) return report.metrics.top1();
    if (std.mem.eql(u8, metric, "top-1")) return report.metrics.top1();
    if (std.mem.eql(u8, metric, "top5")) return report.metrics.top5();
    if (std.mem.eql(u8, metric, "top-5")) return report.metrics.top5();
    if (std.mem.eql(u8, metric, "top10")) return report.metrics.top10();
    if (std.mem.eql(u8, metric, "top-10")) return report.metrics.top10();
    if (std.mem.eql(u8, metric, "mrr")) return report.metrics.mrr();
    return report.metrics.embeddingLoss();
}

fn betterReport(a: *const CheckpointReport, b: *const CheckpointReport, metric: []const u8) bool {
    const av = metricValue(a, metric) orelse return false;
    const bv = metricValue(b, metric) orelse return true;
    const lower_is_better = std.mem.eql(u8, metric, "embedding_l2_loss");
    if (av != bv) return if (lower_is_better) av < bv else av > bv;
    if (a.metadata.global_step != b.metadata.global_step) return a.metadata.global_step > b.metadata.global_step;
    return std.mem.lessThan(u8, a.path, b.path);
}

fn writeOptionalFloat(writer: anytype, value: ?f64) !void {
    if (value) |v| {
        if (!std.math.isFinite(v)) return EvalError.NumericalFailure;
        try writer.print("{d}", .{v});
    } else {
        try writer.writeAll("null");
    }
}

fn writeHex(writer: anytype, digest: [32]u8) !void {
    const alphabet = "0123456789abcdef";
    try writer.writeByte('"');
    for (digest) |byte| {
        try writer.writeByte(alphabet[byte >> 4]);
        try writer.writeByte(alphabet[byte & 0x0f]);
    }
    try writer.writeByte('"');
}

fn writeMetricsJson(writer: anytype, metrics: EvalMetrics) !void {
    try writer.print("{{\"total_input_samples\":{d},\"evaluated_samples\":{d},\"skipped_samples\":{d},\"skipped_empty_samples\":{d},\"skipped_short_samples\":{d},\"valid_target_tokens\":{d},\"truncated_samples\":{d},\"embedding_l2_loss\":", .{ metrics.total_input_samples, metrics.evaluated_samples, metrics.skipped_samples, metrics.skipped_empty_samples, metrics.skipped_short_samples, metrics.valid_target_tokens, metrics.truncated_samples });
    try writeOptionalFloat(writer, metrics.embeddingLoss());
    try writer.writeAll(",\"top1_accuracy\":");
    try writeOptionalFloat(writer, metrics.top1());
    try writer.writeAll(",\"top5_accuracy\":");
    try writeOptionalFloat(writer, metrics.top5());
    try writer.writeAll(",\"top10_accuracy\":");
    try writeOptionalFloat(writer, metrics.top10());
    try writer.writeAll(",\"mean_reciprocal_rank\":");
    try writeOptionalFloat(writer, metrics.mrr());
    try writer.writeAll(",\"mean_rank\":");
    try writeOptionalFloat(writer, metrics.meanRank());
    try writer.print(",\"elapsed_seconds\":{d},\"tokenizer_seconds\":{d},\"forward_seconds\":{d},\"ranking_seconds\":{d},\"tokens_per_second\":{d},\"samples_per_second\":{d},\"tokenization_hash\":", .{ metrics.elapsed_seconds, metrics.tokenizer_seconds, metrics.forward_seconds, metrics.ranking_seconds, metrics.tokens_per_second, metrics.samples_per_second });
    try writeHex(writer, metrics.tokenization_hash);
    try writer.writeAll(",\"activation_hash\":");
    try writeHex(writer, metrics.activation_hash);
    try writer.writeAll(",\"ranking_hash\":");
    try writeHex(writer, metrics.ranking_hash);
    try writer.writeAll("}");
}

fn writeDiagnosticsJson(writer: anytype, diag: CollapseDiagnostics) !void {
    try writer.writeAll("{");
    try writer.print("\"mean_embedding_vector_norm\":{d},\"stddev_embedding_vector_norm\":{d},\"min_embedding_vector_norm\":{d},\"max_embedding_vector_norm\":{d},\"zero_or_near_zero_vectors\":{d},", .{
        diag.mean_norm,
        diag.stddev_norm,
        diag.min_norm,
        diag.max_norm,
        diag.zero_or_near_zero_vectors,
    });
    try writer.writeAll("\"per_dimension_variance\":{");
    try writer.print("\"mean\":{d},\"stddev\":{d},\"min\":{d},\"max\":{d}", .{
        diag.per_dimension_variance_mean,
        diag.per_dimension_variance_stddev,
        diag.per_dimension_variance_min,
        diag.per_dimension_variance_max,
    });
    try writer.writeAll("},");
    try writer.print("\"sampled_pair_count\":{d},\"mean_sampled_cosine_similarity\":{d},\"sampled_cosine_similarity_stddev\":{d},\"effective_rank\":{d},\"near_duplicate_pair_rate\":{d},\"predicted_token_distribution_entropy\":{d},\"most_frequent_prediction_percentage\":{d},\"unique_predicted_token_count\":{d},\"suspicious_conditions\":[", .{
        diag.sampled_pair_count,
        diag.sampled_mean_cosine_similarity,
        diag.sampled_cosine_similarity_stddev,
        diag.effective_rank,
        diag.near_duplicate_pair_rate,
        diag.predicted_token_entropy,
        diag.most_frequent_prediction_percentage,
        diag.unique_predicted_token_count,
    });
    var first = true;
    if (diag.suspicious_low_norm) {
        try writer.writeAll("\"low_embedding_norm\"");
        first = false;
    }
    if (diag.suspicious_low_effective_rank) {
        if (!first) try writer.writeByte(',');
        try writer.writeAll("\"low_effective_rank\"");
        first = false;
    }
    if (diag.suspicious_prediction_collapse) {
        if (!first) try writer.writeByte(',');
        try writer.writeAll("\"prediction_distribution_collapse\"");
    }
    try writer.writeAll("]}");
}

fn writeMemoryJson(writer: anytype, memory: MemoryMetrics) !void {
    try writer.writeAll("{");
    try writer.writeAll("\"serialized_checkpoint_size\":{");
    try writer.print("\"bytes\":{d},\"decimal_mb\":{d},\"binary_mib\":{d}", .{
        memory.serialized_checkpoint_bytes,
        @as(f64, @floatFromInt(memory.serialized_checkpoint_bytes)) / 1.0e6,
        @as(f64, @floatFromInt(memory.serialized_checkpoint_bytes)) / 1048576.0,
    });
    try writer.writeAll("},");
    try writer.print("\"checkpoint_payload_bytes\":{d},\"host_resident_model_state_bytes\":{d},\"rsf_weight_bytes\":{d},\"rsf_velocity_bytes\":{d},\"embedding_weight_bytes\":{d},\"embedding_velocity_bytes\":{d},\"embedding_gradient_bytes\":{d},\"activation_peak_bytes\":{d},\"temporary_ranking_buffer_peak_bytes\":{d},\"tracked_device_allocation_peak_bytes\":", .{
        memory.checkpoint_payload_bytes,
        memory.host_model_state_bytes,
        memory.rsf_weight_bytes,
        memory.rsf_velocity_bytes,
        memory.embedding_weight_bytes,
        memory.embedding_velocity_bytes,
        memory.embedding_gradient_bytes,
        memory.activation_peak_bytes,
        memory.ranking_buffer_peak_bytes,
    });
    if (memory.tracked_device_allocation_peak_bytes) |v| try writer.print("{d}", .{v}) else try writer.writeAll("null");
    try writer.writeAll("}");
}

fn writeCheckpointReportJson(writer: anytype, report: *CheckpointReport, allocator: std.mem.Allocator) !void {
    try writer.writeAll("{");
    try writer.writeAll("\"path\":");
    try jsonEscape(writer, report.path);
    try writer.writeAll(",\"sha256\":");
    try jsonEscape(writer, report.digest);
    try writer.print(",\"success\":{s},\"error\":", .{if (report.success) "true" else "false"});
    if (report.error_name) |e| try jsonEscape(writer, e) else try writer.writeAll("null");
    try writer.writeAll(",\"checkpoint_metadata\":");
    try checkpoint.writeMetadataJson(writer, report.metadata);
    try writer.writeAll(",\"quality_metrics\":");
    try writeMetricsJson(writer, report.metrics);
    try writer.writeAll(",\"collapse_diagnostics\":");
    try writeDiagnosticsJson(writer, report.diagnostics);
    try writer.print(",\"performance\":{{\"model_load_seconds\":{d},\"batch_latency\":", .{report.load_seconds});
    try report.batch_latency.writeJson(writer, allocator);
    try writer.writeAll("}");
    try writer.writeAll(",\"memory\":");
    try writeMemoryJson(writer, report.memory);
    try writer.writeAll(",\"reproducibility\":{");
    try writer.print("\"enabled\":{s},\"runs\":{d},\"passed\":{s},\"max_absolute_metric_difference\":{d},\"max_relative_metric_difference\":{d},\"tokenization_hashes_match\":{s},\"activation_hashes_match\":{s},\"ranking_hashes_match\":{s}", .{
        if (report.reproducibility.enabled) "true" else "false",
        report.reproducibility.runs,
        if (report.reproducibility.passed) "true" else "false",
        report.reproducibility.max_absolute_metric_difference,
        report.reproducibility.max_relative_metric_difference,
        if (report.reproducibility.tokenization_hashes_match) "true" else "false",
        if (report.reproducibility.activation_hashes_match) "true" else "false",
        if (report.reproducibility.ranking_hashes_match) "true" else "false",
    });
    try writer.writeAll("}}");
}

fn writeAtomic(allocator: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.{d}.tmp", .{ path, std.time.nanoTimestamp() });
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
    try renamePath(tmp_path, path);
    committed = true;
}

fn writeReport(allocator: std.mem.Allocator, args: Args, reports: []CheckpointReport, dataset_digest: ?[]const u8) !void {
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    const writer = out.writer();
    var best_index: ?usize = null;
    for (reports, 0..) |*r, i| {
        if (metricValue(r, args.selection_metric) == null) continue;
        if (best_index == null or betterReport(r, &reports[best_index.?], args.selection_metric)) best_index = i;
    }
    try writer.writeAll("{\"schema_version\":1,\"command\":\"jaide-eval\",\"timestamp_unix_seconds\":");
    try writer.print("{d}", .{std.time.timestamp()});
    try writer.writeAll(",\"configuration\":{");
    try writer.writeAll("\"checkpoints\":[");
    for (args.checkpoints.items, 0..) |p, i| {
        if (i > 0) try writer.writeByte(',');
        try jsonEscape(writer, p);
    }
    try writer.writeAll("],\"dataset\":");
    if (args.dataset) |p| try jsonEscape(writer, p) else try writer.writeAll("null");
    try writer.print(",\"batch_size\":{d},\"max_sequence_length\":{d},\"reproducibility_runs\":{d},\"seed\":{d},\"ranking_chunk_size\":{d},\"selection_metric\":", .{ args.batch_size, args.max_sequence_length, args.reproducibility_runs, args.seed, args.ranking_chunk_size });
    try jsonEscape(writer, args.selection_metric);
    try writer.writeAll("},\"dataset_identity\":{");
    try writer.writeAll("\"sha256\":");
    if (dataset_digest) |d| try jsonEscape(writer, d) else try writer.writeAll("null");
    try writer.writeAll("},\"checkpoints\":[");
    for (reports, 0..) |*report, i| {
        if (i > 0) try writer.writeByte(',');
        try writeCheckpointReportJson(writer, report, allocator);
    }
    try writer.writeAll("],\"selected_best_checkpoint\":");
    if (best_index) |idx| {
        try writer.writeAll("{");
        try writer.writeAll("\"path\":");
        try jsonEscape(writer, reports[idx].path);
        try writer.writeAll(",\"criterion\":");
        try jsonEscape(writer, args.selection_metric);
        try writer.writeAll(",\"value\":");
        try writeOptionalFloat(writer, metricValue(&reports[idx], args.selection_metric));
        try writer.writeAll(",\"reason\":");
        try jsonEscape(writer, "selected by metric with deterministic tie-breaking on global_step then path");
        try writer.writeAll("}");
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"warnings":[\"embedding_l2_loss is a mean squared embedding-vector objective and is not perplexity\"],\"errors":[");
    var first_error = true;
    for (reports) |*r| {
        if (r.error_name) |e| {
            if (!first_error) try writer.writeByte(',');
            first_error = false;
            try writer.writeAll("{");
            try writer.writeAll("\"checkpoint\":");
            try jsonEscape(writer, r.path);
            try writer.writeAll(",\"error\":");
            try jsonEscape(writer, e);
            try writer.writeAll("}");
        }
    }
    try writer.writeAll("]}");
    try writeAtomic(allocator, args.output.?, out.items);
}

const GenerationResult = struct {
    generated_tokens: []u32,
    generated_text: []u8,
    stop_reason: []const u8,
    latency_seconds: f64,
    first_token_latency_seconds: ?f64,
    repeated_token_loop: bool,
    repeated_ngram: bool,
    empty_output: bool,
    immediate_termination: bool,
    invalid_token: bool,
    excessive_unknown_token_output: bool,

    fn deinit(self: *GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_tokens);
        allocator.free(self.generated_text);
    }
};

fn neuralNextToken(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, tokens: []const u32, max_sequence_length: usize, norms: []const f64, ranking_chunk_size: usize) !u32 {
    if (tokens.len == 0) return 3;
    const start = if (tokens.len > max_sequence_length) tokens.len - max_sequence_length else 0;
    const window = tokens[start..];
    const seq_len = window.len;
    const model_dim = state.metadata.model_dim;
    const input_data = try allocator.alloc(f16, seq_len * model_dim);
    defer allocator.free(input_data);
    for (window, 0..) |token, pos| {
        if (@as(usize, token) >= state.embedding.vocab_size) return EvalError.InvalidTokenId;
        const source = state.embedding.weight.data[@as(usize, token) * model_dim .. @as(usize, token) * model_dim + model_dim];
        var d: usize = 0;
        while (d < model_dim) : (d += 1) input_data[pos * model_dim + d] = @floatCast(source[d]);
    }
    var arr = try accel.FutharkArray3DF16.newFromFlat(&state.accelerator.ctx, input_data, 1, seq_len, model_dim);
    defer arr.free(&state.accelerator.ctx);
    var output = try state.accelerator.forwardBatch(&arr);
    defer output.free(&state.accelerator.ctx);
    const flat = try output.valuesFlat(&state.accelerator.ctx, allocator);
    defer allocator.free(flat);
    const pred = try allocator.alloc(f32, model_dim);
    defer allocator.free(pred);
    const base = (seq_len - 1) * model_dim;
    var d: usize = 0;
    while (d < model_dim) : (d += 1) pred[d] = @floatCast(flat[base + d]);
    const rr = try rankPrediction(pred, &state.embedding, norms, 0, ranking_chunk_size);
    return rr.best_token;
}

fn hasRepeatedNgram(tokens: []const u32, n: usize) bool {
    if (tokens.len < n * 2 or n == 0) return false;
    const a = tokens[tokens.len - n .. tokens.len];
    const b = tokens[tokens.len - n * 2 .. tokens.len - n];
    return std.mem.eql(u32, a, b);
}

fn generatePrompt(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, prompt: []const u8, args: Args, norms: []const f64) !struct { tokenized_prompt: []u32, result: GenerationResult } {
    var prompt_tokens_list = std.ArrayList(u32).init(allocator);
    errdefer prompt_tokens_list.deinit();
    const before = state.tokenizer.next_token_id;
    try state.tokenizer.encode(prompt, &prompt_tokens_list);
    if (state.tokenizer.next_token_id != before) return EvalError.TokenizerMutationDetected;
    const tokenized_prompt = try prompt_tokens_list.toOwnedSlice();
    errdefer allocator.free(tokenized_prompt);
    var generated = std.ArrayList(u32).init(allocator);
    errdefer generated.deinit();
    try generated.appendSlice(tokenized_prompt);
    var new_tokens = std.ArrayList(u32).init(allocator);
    errdefer new_tokens.deinit();
    var stop_reason: []const u8 = "max_length";
    var repeated_token_loop = false;
    var repeated_ngram = false;
    var invalid_token = false;
    var first_latency: ?f64 = null;
    var timer = try std.time.Timer.start();
    var step: usize = 0;
    while (step < args.generation_max_tokens) : (step += 1) {
        var token_timer = try std.time.Timer.start();
        const next = neuralNextToken(allocator, state, generated.items, args.max_sequence_length, norms, args.ranking_chunk_size) catch |err| {
            invalid_token = true;
            stop_reason = @errorName(err);
            break;
        };
        const token_latency = @as(f64, @floatFromInt(token_timer.read())) / 1.0e9;
        if (first_latency == null) first_latency = token_latency;
        if (@as(usize, next) >= state.embedding.vocab_size) {
            invalid_token = true;
            stop_reason = "invalid_token";
            break;
        }
        if (next == 0) {
            stop_reason = "pad";
            break;
        }
        if (next == 3) {
            stop_reason = "eos";
            break;
        }
        try generated.append(next);
        try new_tokens.append(next);
        if (new_tokens.items.len >= 8) {
            const last = new_tokens.items[new_tokens.items.len - 1];
            var same: usize = 1;
            var idx = new_tokens.items.len - 1;
            while (idx > 0 and new_tokens.items[idx - 1] == last) : (idx -= 1) same += 1;
            if (same >= 8) {
                repeated_token_loop = true;
                stop_reason = "repeated_token_loop";
                break;
            }
        }
        if (hasRepeatedNgram(new_tokens.items, 3)) repeated_ngram = true;
    }
    const latency = @as(f64, @floatFromInt(timer.read())) / 1.0e9;
    var text_buf = std.ArrayList(u8).init(allocator);
    errdefer text_buf.deinit();
    try state.tokenizer.decode(new_tokens.items, &text_buf);
    var unknowns: usize = 0;
    for (new_tokens.items) |t| if (t == 1) unknowns += 1;
    const excessive_unknown = new_tokens.items.len > 0 and unknowns * 2 >= new_tokens.items.len;
    const owned_tokens = try new_tokens.toOwnedSlice();
    const owned_text = try text_buf.toOwnedSlice();
    return .{
        .tokenized_prompt = tokenized_prompt,
        .result = .{
            .generated_tokens = owned_tokens,
            .generated_text = owned_text,
            .stop_reason = stop_reason,
            .latency_seconds = latency,
            .first_token_latency_seconds = first_latency,
            .repeated_token_loop = repeated_token_loop,
            .repeated_ngram = repeated_ngram,
            .empty_output = owned_tokens.len == 0,
            .immediate_termination = owned_tokens.len == 0 and std.mem.eql(u8, stop_reason, "eos"),
            .invalid_token = invalid_token,
            .excessive_unknown_token_output = excessive_unknown,
        },
    };
}

fn appendGenerations(allocator: std.mem.Allocator, args: Args, prompts: []const PromptRecord, checkpoint_path: []const u8, digest: []const u8) !void {
    if (args.generation_output == null) return EvalError.GenerationOutputRequired;
    var state = try checkpoint.loadCheckpointState(allocator, checkpoint_path);
    defer state.deinit();
    const norms = try buildEmbeddingNorms(allocator, &state.embedding);
    defer allocator.free(norms);
    const path = args.generation_output.?;
    const file = if (std.fs.path.isAbsolute(path)) try std.fs.openFileAbsolute(path, .{ .mode = .write_only }) else try std.fs.cwd().openFile(path, .{ .mode = .write_only });
    defer file.close();
    try file.seekFromEnd(0);
    var writer = file.writer();
    for (prompts, 0..) |prompt, idx| {
        var generated = try generatePrompt(allocator, &state, prompt.prompt, args, norms);
        defer {
            allocator.free(generated.tokenized_prompt);
            generated.result.deinit(allocator);
        }
        try writer.writeAll("{");
        try writer.writeAll("\"checkpoint\":");
        try jsonEscape(writer, checkpoint_path);
        try writer.writeAll(",\"checkpoint_sha256\":");
        try jsonEscape(writer, digest);
        try writer.writeAll(",\"prompt_id\":");
        if (prompt.id) |id| try jsonEscape(writer, id) else try writer.print("{d}", .{idx});
        try writer.writeAll(",\"category\":");
        if (prompt.category) |category| try jsonEscape(writer, category) else try writer.writeAll("null");
        try writer.writeAll(",\"prompt\":");
        try jsonEscape(writer, prompt.prompt);
        try writer.writeAll(",\"tokenized_prompt":[");
        for (generated.tokenized_prompt, 0..) |t, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.print("{d}", .{t});
        }
        try writer.writeAll("],\"generated_token_ids\":[");
        for (generated.result.generated_tokens, 0..) |t, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.print("{d}", .{t});
        }
        try writer.writeAll("],\"generated_text\":");
        try jsonEscape(writer, generated.result.generated_text);
        try writer.writeAll(",\"stop_reason\":");
        try jsonEscape(writer, generated.result.stop_reason);
        try writer.print(",\"generation_length\":{d},\"latency_seconds\":{d},\"first_token_latency_seconds\":", .{ generated.result.generated_tokens.len, generated.result.latency_seconds });
        if (generated.result.first_token_latency_seconds) |v| try writer.print("{d}", .{v}) else try writer.writeAll("null");
        try writer.writeAll(",\"per_token_latency_seconds\":");
        if (generated.result.generated_tokens.len > 0) try writer.print("{d}", .{generated.result.latency_seconds / @as(f64, @floatFromInt(generated.result.generated_tokens.len))}) else try writer.writeAll("null");
        try writer.print(",\"generation_seed\":{d},\"parameters\":", .{args.seed});
        try writer.writeAll("{");
        try writer.print("\"decoding\":\"greedy\",\"max_new_tokens\":{d},\"max_sequence_length\":{d}", .{ args.generation_max_tokens, args.max_sequence_length });
        try writer.writeAll("},\"detections\":");
        try writer.writeAll("{");
        try writer.print("\"repeated_token_loop\":{s},\"repeated_ngram\":{s},\"empty_output\":{s},\"immediate_termination\":{s},\"invalid_token_ids\":{s},\"excessive_unknown_token_output\":{s}", .{
            if (generated.result.repeated_token_loop) "true" else "false",
            if (generated.result.repeated_ngram) "true" else "false",
            if (generated.result.empty_output) "true" else "false",
            if (generated.result.immediate_termination) "true" else "false",
            if (generated.result.invalid_token) "true" else "false",
            if (generated.result.excessive_unknown_token_output) "true" else "false",
        });
        try writer.writeAll("}}\n");
    }
    try file.sync();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var args = try parseArgs(allocator);
    defer args.deinit();
    var records = try allocator.alloc(DatasetRecord, 0);
    defer {
        for (records) |*r| r.deinit(allocator);
        allocator.free(records);
    };
    if (args.dataset) |dataset_path| {
        const loaded_records = try readLinesJsonRecords(allocator, dataset_path);
        allocator.free(records);
        records = loaded_records;
    }
    var prompts = try allocator.alloc(PromptRecord, 0);
    defer {
        for (prompts) |*p| p.deinit(allocator);
        allocator.free(prompts);
    };
    if (args.prompts) |prompt_path| {
        const loaded_prompts = try readPromptRecords(allocator, prompt_path);
        allocator.free(prompts);
        prompts = loaded_prompts;
        if (args.generation_output == null) return EvalError.GenerationOutputRequired;
        try writeAtomic(allocator, args.generation_output.?, "");
    }
    const digest = try datasetDigest(allocator, args.dataset);
    defer if (digest) |d| allocator.free(d);
    var reports = std.ArrayList(CheckpointReport).init(allocator);
    defer {
        for (reports.items) |*r| r.deinit(allocator);
        reports.deinit();
    }
    var any_failure = false;
    for (args.checkpoints.items) |path| {
        std.debug.print("evaluating {s}\n", .{path});
        var report = evaluateCheckpoint(allocator, path, records, args);
        if (!report.success) any_failure = true;
        if (args.prompts != null and report.success) {
            appendGenerations(allocator, args, prompts, path, report.digest) catch |err| {
                report.success = false;
                report.error_name = @errorName(err);
                any_failure = true;
            };
        }
        try reports.append(report);
    }
    try writeReport(allocator, args, reports.items, if (digest) |d| d else null);
    if (any_failure) std.process.exit(1);
}

test "exact ranking deterministic tie breaking and reciprocal rank basis" {
    const allocator = std.testing.allocator;
    var embedding = try LearnedEmbedding.init(allocator, 4, 2, 1);
    defer embedding.deinit();
    embedding.weight.data[0] = 1.0;
    embedding.weight.data[1] = 0.0;
    embedding.weight.data[2] = 1.0;
    embedding.weight.data[3] = 0.0;
    embedding.weight.data[4] = 0.0;
    embedding.weight.data[5] = 1.0;
    embedding.weight.data[6] = -1.0;
    embedding.weight.data[7] = 0.0;
    const norms = try buildEmbeddingNorms(allocator, &embedding);
    defer allocator.free(norms);
    const pred = [_]f32{ 1.0, 0.0 };
    const r0 = try rankPrediction(pred[0..], &embedding, norms, 0, 2);
    try std.testing.expectEqual(@as(u64, 1), r0.rank);
    try std.testing.expectEqual(@as(u32, 0), r0.best_token);
    const r1 = try rankPrediction(pred[0..], &embedding, norms, 1, 2);
    try std.testing.expectEqual(@as(u64, 2), r1.rank);
    try std.testing.expectEqual(@as(u32, 0), r1.best_token);
}

test "metric weighted aggregation excludes skipped samples" {
    var metrics = EvalMetrics{};
    metrics.total_input_samples = 3;
    metrics.evaluated_samples = 1;
    metrics.skipped_samples = 2;
    metrics.valid_target_tokens = 2;
    metrics.embedding_l2_loss_sum = 4.0;
    metrics.top1_count = 1;
    metrics.top5_count = 2;
    metrics.top10_count = 2;
    metrics.reciprocal_rank_sum = 1.5;
    try std.testing.expectEqual(@as(f64, 2.0), metrics.embeddingLoss().?);
    try std.testing.expectEqual(@as(f64, 0.5), metrics.top1().?);
    try std.testing.expectEqual(@as(f64, 1.0), metrics.top5().?);
    try std.testing.expectEqual(@as(f64, 0.75), metrics.mrr().?);
}
