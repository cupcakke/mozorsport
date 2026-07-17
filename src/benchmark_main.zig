const std = @import("std");
const checkpoint = @import("core/checkpoint.zig");
const accel = @import("hw/accel/accel_interface.zig");
const LearnedEmbedding = @import("core/learned_embedding.zig").LearnedEmbedding;
const build_options = @import("build_options");

const BenchmarkError = error{
    MissingCheckpoint,
    MissingDataset,
    MissingOutput,
    InvalidArgument,
    InvalidDataset,
    InvalidTextField,
    EmptyUsableDataset,
    NumericalFailure,
    InvalidTokenId,
    TokenizerMutationDetected,
};

const Args = struct {
    allocator: std.mem.Allocator,
    checkpoint_path: ?[]const u8 = null,
    dataset_path: ?[]const u8 = null,
    prompts_path: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    batch_sizes: []const usize = &.{},
    sequence_lengths: []const usize = &.{},
    generation_lengths: []const usize = &.{},
    warmup_iterations: usize = 3,
    measured_iterations: usize = 10,
    ranking_chunk_size: usize = 4096,
    seed: u64 = 42,

    fn init(allocator: std.mem.Allocator) Args {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Args) void {
        if (self.checkpoint_path) |p| self.allocator.free(p);
        if (self.dataset_path) |p| self.allocator.free(p);
        if (self.prompts_path) |p| self.allocator.free(p);
        if (self.output_path) |p| self.allocator.free(p);
        if (self.batch_sizes.len > 0) self.allocator.free(self.batch_sizes);
        if (self.sequence_lengths.len > 0) self.allocator.free(self.sequence_lengths);
        if (self.generation_lengths.len > 0) self.allocator.free(self.generation_lengths);
    }
};

const TextRecord = struct {
    text: []u8,

    fn deinit(self: *TextRecord, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

const TokenSample = struct {
    tokens: []u32,

    fn deinit(self: *TokenSample, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
    }
};

const Stats = struct {
    values: std.ArrayList(f64),

    fn init(allocator: std.mem.Allocator) Stats {
        return .{ .values = std.ArrayList(f64).init(allocator) };
    }

    fn deinit(self: *Stats) void {
        self.values.deinit();
    }

    fn add(self: *Stats, value: f64) !void {
        if (!std.math.isFinite(value) or value < 0.0) return BenchmarkError.NumericalFailure;
        try self.values.append(value);
    }

    fn writeJson(self: *const Stats, writer: anytype, allocator: std.mem.Allocator, unit_name: []const u8) !void {
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
        for (copy) |v| sum += v;
        const mean = sum / @as(f64, @floatFromInt(copy.len));
        var var_sum: f64 = 0.0;
        for (copy) |v| {
            const d = v - mean;
            var_sum += d * d;
        }
        const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(copy.len)));
        try writer.writeAll("{");
        try jsonEscape(writer, unit_name);
        try writer.writeAll(":{");
        try writer.print("\"count\":{d},\"mean\":{d},\"median\":{d},\"min\":{d},\"max\":{d},\"stddev\":{d},\"p50\":{d},\"p90\":{d},\"p95\":{d},\"p99\":{d}", .{
            copy.len,
            mean,
            percentile(copy, 0.50),
            copy[0],
            copy[copy.len - 1],
            stddev,
            percentile(copy, 0.50),
            percentile(copy, 0.90),
            percentile(copy, 0.95),
            percentile(copy, 0.99),
        });
        try writer.writeAll("}}");
    }
};

const ForwardBenchmarkResult = struct {
    batch_size: usize,
    sequence_length: usize,
    measured_iterations: usize,
    warmup_iterations: usize,
    valid_target_tokens: u64 = 0,
    evaluated_samples: u64 = 0,
    embedding_l2_loss_sum: f64 = 0.0,
    top1_count: u64 = 0,
    top5_count: u64 = 0,
    top10_count: u64 = 0,
    reciprocal_rank_sum: f64 = 0.0,
    tokenizer_time: Stats,
    forward_time: Stats,
    ranking_time: Stats,
    batch_time: Stats,
    tokens_per_second: Stats,
    samples_per_second: Stats,
    activation_peak_bytes: u64 = 0,
    ranking_buffer_peak_bytes: u64 = 0,

    fn init(allocator: std.mem.Allocator, batch_size: usize, sequence_length: usize, warmup_iterations: usize, measured_iterations: usize) ForwardBenchmarkResult {
        return .{
            .batch_size = batch_size,
            .sequence_length = sequence_length,
            .measured_iterations = measured_iterations,
            .warmup_iterations = warmup_iterations,
            .tokenizer_time = Stats.init(allocator),
            .forward_time = Stats.init(allocator),
            .ranking_time = Stats.init(allocator),
            .batch_time = Stats.init(allocator),
            .tokens_per_second = Stats.init(allocator),
            .samples_per_second = Stats.init(allocator),
        };
    }

    fn deinit(self: *ForwardBenchmarkResult) void {
        self.tokenizer_time.deinit();
        self.forward_time.deinit();
        self.ranking_time.deinit();
        self.batch_time.deinit();
        self.tokens_per_second.deinit();
        self.samples_per_second.deinit();
    }

    fn loss(self: *const ForwardBenchmarkResult) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return self.embedding_l2_loss_sum / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top1(self: *const ForwardBenchmarkResult) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top1_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top5(self: *const ForwardBenchmarkResult) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top5_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn top10(self: *const ForwardBenchmarkResult) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return @as(f64, @floatFromInt(self.top10_count)) / @as(f64, @floatFromInt(self.valid_target_tokens));
    }

    fn mrr(self: *const ForwardBenchmarkResult) ?f64 {
        if (self.valid_target_tokens == 0) return null;
        return self.reciprocal_rank_sum / @as(f64, @floatFromInt(self.valid_target_tokens));
    }
};

const GenerationBenchmarkResult = struct {
    generation_length: usize,
    prompt_count: usize,
    warmup_iterations: usize,
    measured_generations: usize,
    first_token_latency: Stats,
    generated_token_latency: Stats,
    complete_generation_latency: Stats,
    generated_tokens_per_second: Stats,
    empty_outputs: u64 = 0,
    repeated_token_loops: u64 = 0,
    repeated_ngrams: u64 = 0,
    invalid_tokens: u64 = 0,
    unknown_token_outputs: u64 = 0,
    unique_generated_token_count: u64 = 0,
    generated_token_entropy: f64 = 0.0,

    fn init(allocator: std.mem.Allocator, generation_length: usize, warmup_iterations: usize) GenerationBenchmarkResult {
        return .{
            .generation_length = generation_length,
            .prompt_count = 0,
            .warmup_iterations = warmup_iterations,
            .measured_generations = 0,
            .first_token_latency = Stats.init(allocator),
            .generated_token_latency = Stats.init(allocator),
            .complete_generation_latency = Stats.init(allocator),
            .generated_tokens_per_second = Stats.init(allocator),
        };
    }

    fn deinit(self: *GenerationBenchmarkResult) void {
        self.first_token_latency.deinit();
        self.generated_token_latency.deinit();
        self.complete_generation_latency.deinit();
        self.generated_tokens_per_second.deinit();
    }
};

fn percentile(sorted: []const f64, p: f64) f64 {
    if (sorted.len == 0) return 0.0;
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

fn writeOptionalFloat(writer: anytype, value: ?f64) !void {
    if (value) |v| {
        if (!std.math.isFinite(v)) return BenchmarkError.NumericalFailure;
        try writer.print("{d}", .{v});
    } else {
        try writer.writeAll("null");
    }
}

fn parseList(allocator: std.mem.Allocator, input: []const u8) ![]usize {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();
    var split = std.mem.splitScalar(u8, input, ',');
    while (split.next()) |part_raw| {
        const part = std.mem.trim(u8, part_raw, " \t\r\n");
        if (part.len == 0) continue;
        const value = try std.fmt.parseInt(usize, part, 10);
        if (value == 0) return BenchmarkError.InvalidArgument;
        try result.append(value);
    }
    if (result.items.len == 0) return BenchmarkError.InvalidArgument;
    return try result.toOwnedSlice();
}

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);
    var args = Args.init(allocator);
    errdefer args.deinit();
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.checkpoint_path) |old| allocator.free(old);
            args.checkpoint_path = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--dataset")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.dataset_path) |old| allocator.free(old);
            args.dataset_path = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--prompts")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.prompts_path) |old| allocator.free(old);
            args.prompts_path = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--output")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.output_path) |old| allocator.free(old);
            args.output_path = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--batch-sizes")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.batch_sizes.len > 0) allocator.free(args.batch_sizes);
            args.batch_sizes = try parseList(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--sequence-lengths")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.sequence_lengths.len > 0) allocator.free(args.sequence_lengths);
            args.sequence_lengths = try parseList(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--generation-lengths")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            if (args.generation_lengths.len > 0) allocator.free(args.generation_lengths);
            args.generation_lengths = try parseList(allocator, argv[i]);
        } else if (std.mem.eql(u8, arg, "--warmup-iterations")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            args.warmup_iterations = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--measured-iterations")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            args.measured_iterations = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--ranking-chunk-size")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            args.ranking_chunk_size = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            if (i + 1 >= argv.len) return BenchmarkError.InvalidArgument;
            i += 1;
            args.seed = try std.fmt.parseInt(u64, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("jaide-benchmark --checkpoint /checkpoints/epoch_003/model.ckpt --dataset /data/dataset/validation.jsonl --output /reports/benchmark.json [--batch-sizes 1,8,32] [--sequence-lengths 32,128,512] [--prompts prompts.jsonl --generation-lengths 16,64] [--warmup-iterations 3] [--measured-iterations 10]\n", .{});
            std.process.exit(0);
        } else {
            return BenchmarkError.InvalidArgument;
        }
    }
    if (args.batch_sizes.len == 0) args.batch_sizes = try allocator.dupe(usize, &[_]usize{ 1, 8, 32 });
    if (args.sequence_lengths.len == 0) args.sequence_lengths = try allocator.dupe(usize, &[_]usize{ 32, 128, 512 });
    if (args.generation_lengths.len == 0) args.generation_lengths = try allocator.dupe(usize, &[_]usize{ 16, 64 });
    if (args.checkpoint_path == null) return BenchmarkError.MissingCheckpoint;
    if (args.dataset_path == null) return BenchmarkError.MissingDataset;
    if (args.output_path == null) return BenchmarkError.MissingOutput;
    if (args.warmup_iterations == 0 or args.measured_iterations == 0 or args.ranking_chunk_size == 0) return BenchmarkError.InvalidArgument;
    return args;
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

fn loadJsonTextRecords(allocator: std.mem.Allocator, path: []const u8, field_name: []const u8) ![]TextRecord {
    var file = try openRead(path);
    defer file.close();
    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();
    var records = std.ArrayList(TextRecord).init(allocator);
    errdefer {
        for (records.items) |*record| record.deinit(allocator);
        records.deinit();
    }
    while (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |line_raw| {
        defer allocator.free(line_raw);
        const line = std.mem.trim(u8, line_raw, " \t\r\n");
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always }) catch return BenchmarkError.InvalidDataset;
        defer parsed.deinit();
        if (parsed.value != .object) return BenchmarkError.InvalidDataset;
        const value = parsed.value.object.get(field_name) orelse return BenchmarkError.InvalidTextField;
        if (value != .string) return BenchmarkError.InvalidTextField;
        if (value.string.len == 0) return BenchmarkError.InvalidTextField;
        try records.append(.{ .text = try allocator.dupe(u8, value.string) });
    }
    if (records.items.len == 0) return BenchmarkError.InvalidDataset;
    return try records.toOwnedSlice();
}

fn tokenizeRecords(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, records: []const TextRecord, max_sequence_len: usize, tokenizer_seconds: *f64) ![]TokenSample {
    var samples = std.ArrayList(TokenSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| sample.deinit(allocator);
        samples.deinit();
    }
    for (records) |record| {
        const before = state.tokenizer.next_token_id;
        var tokens = std.ArrayList(u32).init(allocator);
        errdefer tokens.deinit();
        var timer = try std.time.Timer.start();
        try state.tokenizer.encode(record.text, &tokens);
        tokenizer_seconds.* += @as(f64, @floatFromInt(timer.read())) / 1.0e9;
        if (state.tokenizer.next_token_id != before) return BenchmarkError.TokenizerMutationDetected;
        if (tokens.items.len < 2) {
            tokens.deinit();
            continue;
        }
        const keep = @min(tokens.items.len, max_sequence_len + 1);
        const owned = try allocator.dupe(u32, tokens.items[0..keep]);
        tokens.deinit();
        if (owned.len >= 2) try samples.append(.{ .tokens = owned }) else allocator.free(owned);
    }
    if (samples.items.len == 0) return BenchmarkError.EmptyUsableDataset;
    return try samples.toOwnedSlice();
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
            if (!std.math.isFinite(value)) return BenchmarkError.NumericalFailure;
            sum += @as(f64, value) * @as(f64, value);
        }
        norms[token] = @sqrt(sum);
        if (!std.math.isFinite(norms[token])) return BenchmarkError.NumericalFailure;
    }
    return norms;
}

fn cosineScore(pred: []const f32, pred_norm: f64, emb: []const f32, emb_norm: f64) !f64 {
    if (pred_norm <= 1e-12 or emb_norm <= 1e-12) return -std.math.inf(f64);
    var dot: f64 = 0.0;
    for (pred, 0..) |v, i| {
        const e = emb[i];
        if (!std.math.isFinite(v) or !std.math.isFinite(e)) return BenchmarkError.NumericalFailure;
        dot += @as(f64, v) * @as(f64, e);
    }
    const score = dot / (pred_norm * emb_norm);
    if (std.math.isNan(score)) return BenchmarkError.NumericalFailure;
    return score;
}

const RankResult = struct {
    rank: u64,
    best_token: u32,
};

fn rankPrediction(pred: []const f32, embedding: *const LearnedEmbedding, norms: []const f64, target: u32, chunk_size: usize) !RankResult {
    if (@as(usize, target) >= embedding.vocab_size) return BenchmarkError.InvalidTokenId;
    var pred_norm_sq: f64 = 0.0;
    for (pred) |v| {
        if (!std.math.isFinite(v)) return BenchmarkError.NumericalFailure;
        pred_norm_sq += @as(f64, v) * @as(f64, v);
    }
    const pred_norm = @sqrt(pred_norm_sq);
    const target_vec = embedding.weight.data[@as(usize, target) * embedding.dim .. @as(usize, target) * embedding.dim + embedding.dim];
    const target_score = try cosineScore(pred, pred_norm, target_vec, norms[@as(usize, target)]);
    var rank: u64 = 1;
    var best_score: f64 = -std.math.inf(f64);
    var best_token: u32 = 0;
    var start: usize = 0;
    while (start < embedding.vocab_size) : (start += chunk_size) {
        const end = @min(start + chunk_size, embedding.vocab_size);
        var token = start;
        while (token < end) : (token += 1) {
            const emb_vec = embedding.weight.data[token * embedding.dim .. token * embedding.dim + embedding.dim];
            const score = try cosineScore(pred, pred_norm, emb_vec, norms[token]);
            if (score > target_score or (score == target_score and token < @as(usize, target))) rank += 1;
            if (score > best_score or (score == best_score and token < @as(usize, best_token))) {
                best_score = score;
                best_token = @intCast(token);
            }
        }
    }
    return .{ .rank = rank, .best_token = best_token };
}

fn fillBatchInputs(state: *checkpoint.CheckpointState, samples: []const TokenSample, batch_start: usize, batch_size: usize, sequence_length: usize, input_data: []f16, valid_counts: []usize) !void {
    const model_dim = state.metadata.model_dim;
    @memset(input_data, @as(f16, 0.0));
    @memset(valid_counts, 0);
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const sample = samples[(batch_start + b) % samples.len];
        const valid_len = @min(sequence_length, sample.tokens.len - 1);
        valid_counts[b] = valid_len;
        var pos: usize = 0;
        while (pos < valid_len) : (pos += 1) {
            const token = sample.tokens[pos];
            if (@as(usize, token) >= state.embedding.vocab_size) return BenchmarkError.InvalidTokenId;
            const source = state.embedding.weight.data[@as(usize, token) * model_dim .. @as(usize, token) * model_dim + model_dim];
            const base = (b * sequence_length + pos) * model_dim;
            var d: usize = 0;
            while (d < model_dim) : (d += 1) {
                const value = source[d];
                if (!std.math.isFinite(value)) return BenchmarkError.NumericalFailure;
                input_data[base + d] = @floatCast(value);
            }
        }
    }
}

fn scoreBatch(state: *checkpoint.CheckpointState, samples: []const TokenSample, batch_start: usize, batch_size: usize, sequence_length: usize, output_data: []const f32, valid_counts: []const usize, norms: []const f64, ranking_chunk_size: usize, result: *ForwardBenchmarkResult, collect: bool) !u64 {
    const model_dim = state.metadata.model_dim;
    var valid_tokens: u64 = 0;
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const sample = samples[(batch_start + b) % samples.len];
        var pos: usize = 0;
        while (pos < valid_counts[b]) : (pos += 1) {
            const target = sample.tokens[pos + 1];
            if (target == 0) continue;
            if (@as(usize, target) >= state.embedding.vocab_size) return BenchmarkError.InvalidTokenId;
            const base = (b * sequence_length + pos) * model_dim;
            const pred = output_data[base .. base + model_dim];
            const target_vec = state.embedding.weight.data[@as(usize, target) * model_dim .. @as(usize, target) * model_dim + model_dim];
            var mse: f64 = 0.0;
            var d: usize = 0;
            while (d < model_dim) : (d += 1) {
                const diff = @as(f64, pred[d]) - @as(f64, target_vec[d]);
                if (!std.math.isFinite(diff)) return BenchmarkError.NumericalFailure;
                mse += diff * diff;
            }
            mse /= @as(f64, @floatFromInt(model_dim));
            const rank = try rankPrediction(pred, &state.embedding, norms, target, ranking_chunk_size);
            valid_tokens += 1;
            if (collect) {
                result.embedding_l2_loss_sum += mse;
                result.valid_target_tokens += 1;
                if (rank.rank <= 1) result.top1_count += 1;
                if (rank.rank <= 5) result.top5_count += 1;
                if (rank.rank <= 10) result.top10_count += 1;
                result.reciprocal_rank_sum += 1.0 / @as(f64, @floatFromInt(rank.rank));
            }
        }
    }
    return valid_tokens;
}

fn runOneForwardIteration(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, samples: []const TokenSample, norms: []const f64, batch_start: usize, result: *ForwardBenchmarkResult, ranking_chunk_size: usize, collect: bool) !void {
    const batch_size = result.batch_size;
    const sequence_length = result.sequence_length;
    const model_dim = state.metadata.model_dim;
    const element_count = try std.math.mul(usize, try std.math.mul(usize, batch_size, sequence_length), model_dim);
    const input_data = try allocator.alloc(f16, element_count);
    defer allocator.free(input_data);
    const valid_counts = try allocator.alloc(usize, batch_size);
    defer allocator.free(valid_counts);
    var batch_timer = try std.time.Timer.start();
    try fillBatchInputs(state, samples, batch_start, batch_size, sequence_length, input_data, valid_counts);
    var inputs = try accel.FutharkArray3DF16.newFromFlat(&state.accelerator.ctx, input_data, batch_size, sequence_length, model_dim);
    defer inputs.free(&state.accelerator.ctx);
    try state.accelerator.sync();
    var forward_timer = try std.time.Timer.start();
    var outputs = try state.accelerator.forwardBatch(&inputs);
    defer outputs.free(&state.accelerator.ctx);
    try state.accelerator.sync();
    const forward_seconds = @as(f64, @floatFromInt(forward_timer.read())) / 1.0e9;
    const output_f16 = try outputs.valuesFlat(&state.accelerator.ctx, allocator);
    defer allocator.free(output_f16);
    const output_f32 = try allocator.alloc(f32, output_f16.len);
    defer allocator.free(output_f32);
    for (output_f16, 0..) |value, i| {
        const v: f32 = @floatCast(value);
        if (!std.math.isFinite(v)) return BenchmarkError.NumericalFailure;
        output_f32[i] = v;
    }
    var ranking_timer = try std.time.Timer.start();
    const valid_tokens = try scoreBatch(state, samples, batch_start, batch_size, sequence_length, output_f32, valid_counts, norms, ranking_chunk_size, result, collect);
    const ranking_seconds = @as(f64, @floatFromInt(ranking_timer.read())) / 1.0e9;
    const batch_seconds = @as(f64, @floatFromInt(batch_timer.read())) / 1.0e9;
    result.activation_peak_bytes = @max(result.activation_peak_bytes, @as(u64, @intCast(element_count * @sizeOf(f16))));
    result.ranking_buffer_peak_bytes = @max(result.ranking_buffer_peak_bytes, @as(u64, @intCast(ranking_chunk_size * @sizeOf(f64))));
    if (collect) {
        result.evaluated_samples += batch_size;
        try result.forward_time.add(forward_seconds * 1000.0);
        try result.ranking_time.add(ranking_seconds * 1000.0);
        try result.batch_time.add(batch_seconds * 1000.0);
        if (batch_seconds > 0.0) {
            try result.tokens_per_second.add(@as(f64, @floatFromInt(valid_tokens)) / batch_seconds);
            try result.samples_per_second.add(@as(f64, @floatFromInt(batch_size)) / batch_seconds);
        }
    }
}

fn runForwardBenchmark(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, records: []const TextRecord, batch_size: usize, sequence_length: usize, args: Args, norms: []const f64) !ForwardBenchmarkResult {
    var result = ForwardBenchmarkResult.init(allocator, batch_size, sequence_length, args.warmup_iterations, args.measured_iterations);
    errdefer result.deinit();
    var tokenizer_seconds: f64 = 0.0;
    const samples = try tokenizeRecords(allocator, state, records, sequence_length, &tokenizer_seconds);
    defer {
        for (samples) |*sample| sample.deinit(allocator);
        allocator.free(samples);
    }
    try result.tokenizer_time.add(tokenizer_seconds * 1000.0);
    var iter: usize = 0;
    while (iter < args.warmup_iterations) : (iter += 1) try runOneForwardIteration(allocator, state, samples, norms, iter * batch_size, &result, args.ranking_chunk_size, false);
    iter = 0;
    while (iter < args.measured_iterations) : (iter += 1) try runOneForwardIteration(allocator, state, samples, norms, (args.warmup_iterations + iter) * batch_size, &result, args.ranking_chunk_size, true);
    return result;
}

fn neuralNextToken(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, tokens: []const u32, sequence_length: usize, norms: []const f64, ranking_chunk_size: usize) !u32 {
    if (tokens.len == 0) return 3;
    const start = if (tokens.len > sequence_length) tokens.len - sequence_length else 0;
    const window = tokens[start..];
    const model_dim = state.metadata.model_dim;
    const input_data = try allocator.alloc(f16, window.len * model_dim);
    defer allocator.free(input_data);
    for (window, 0..) |token, pos| {
        if (@as(usize, token) >= state.embedding.vocab_size) return BenchmarkError.InvalidTokenId;
        const source = state.embedding.weight.data[@as(usize, token) * model_dim .. @as(usize, token) * model_dim + model_dim];
        var d: usize = 0;
        while (d < model_dim) : (d += 1) input_data[pos * model_dim + d] = @floatCast(source[d]);
    }
    var inputs = try accel.FutharkArray3DF16.newFromFlat(&state.accelerator.ctx, input_data, 1, window.len, model_dim);
    defer inputs.free(&state.accelerator.ctx);
    var outputs = try state.accelerator.forwardBatch(&inputs);
    defer outputs.free(&state.accelerator.ctx);
    const out_f16 = try outputs.valuesFlat(&state.accelerator.ctx, allocator);
    defer allocator.free(out_f16);
    const pred = try allocator.alloc(f32, model_dim);
    defer allocator.free(pred);
    const base = (window.len - 1) * model_dim;
    var d: usize = 0;
    while (d < model_dim) : (d += 1) pred[d] = @floatCast(out_f16[base + d]);
    return (try rankPrediction(pred, &state.embedding, norms, 0, ranking_chunk_size)).best_token;
}

fn hasRepeatedNgram(tokens: []const u32, n: usize) bool {
    if (tokens.len < n * 2 or n == 0) return false;
    return std.mem.eql(u32, tokens[tokens.len - n .. tokens.len], tokens[tokens.len - n * 2 .. tokens.len - n]);
}

fn runGenerationBenchmark(allocator: std.mem.Allocator, state: *checkpoint.CheckpointState, prompts: []const TextRecord, generation_length: usize, args: Args, norms: []const f64) !GenerationBenchmarkResult {
    var result = GenerationBenchmarkResult.init(allocator, generation_length, args.warmup_iterations);
    errdefer result.deinit();
    var pred_counts = std.AutoHashMap(u32, u64).init(allocator);
    defer pred_counts.deinit();
    var total_generated: u64 = 0;
    var prompt_index: usize = 0;
    while (prompt_index < prompts.len) : (prompt_index += 1) {
        const measured = prompt_index >= @min(args.warmup_iterations, prompts.len);
        const before = state.tokenizer.next_token_id;
        var prompt_tokens = std.ArrayList(u32).init(allocator);
        defer prompt_tokens.deinit();
        try state.tokenizer.encode(prompts[prompt_index].text, &prompt_tokens);
        if (state.tokenizer.next_token_id != before) return BenchmarkError.TokenizerMutationDetected;
        if (prompt_tokens.items.len == 0) continue;
        var generated = std.ArrayList(u32).init(allocator);
        defer generated.deinit();
        try generated.appendSlice(prompt_tokens.items);
        var new_tokens = std.ArrayList(u32).init(allocator);
        defer new_tokens.deinit();
        var first_token_seconds: ?f64 = null;
        var complete_timer = try std.time.Timer.start();
        var step: usize = 0;
        while (step < generation_length) : (step += 1) {
            var token_timer = try std.time.Timer.start();
            const next = neuralNextToken(allocator, state, generated.items, @max(@as(usize, 1), generation_length), norms, args.ranking_chunk_size) catch |err| {
                if (measured) result.invalid_tokens += 1;
                return err;
            };
            const token_seconds = @as(f64, @floatFromInt(token_timer.read())) / 1.0e9;
            if (first_token_seconds == null) first_token_seconds = token_seconds;
            if (@as(usize, next) >= state.embedding.vocab_size) {
                if (measured) result.invalid_tokens += 1;
                break;
            }
            if (next == 0 or next == 3) break;
            try generated.append(next);
            try new_tokens.append(next);
            if (measured) {
                try result.generated_token_latency.add(token_seconds * 1000.0);
                total_generated += 1;
                const entry = try pred_counts.getOrPut(next);
                if (entry.found_existing) entry.value_ptr.* += 1 else entry.value_ptr.* = 1;
            }
        }
        const complete_seconds = @as(f64, @floatFromInt(complete_timer.read())) / 1.0e9;
        if (measured) {
            result.prompt_count += 1;
            result.measured_generations += 1;
            if (first_token_seconds) |v| try result.first_token_latency.add(v * 1000.0);
            try result.complete_generation_latency.add(complete_seconds * 1000.0);
            if (complete_seconds > 0.0) try result.generated_tokens_per_second.add(@as(f64, @floatFromInt(new_tokens.items.len)) / complete_seconds);
            if (new_tokens.items.len == 0) result.empty_outputs += 1;
            if (hasRepeatedNgram(new_tokens.items, 3)) result.repeated_ngrams += 1;
            var unknowns: usize = 0;
            for (new_tokens.items) |token| if (token == 1) unknowns += 1;
            if (new_tokens.items.len > 0 and unknowns * 2 >= new_tokens.items.len) result.unknown_token_outputs += 1;
            if (new_tokens.items.len >= 8) {
                const last = new_tokens.items[new_tokens.items.len - 1];
                var same: usize = 1;
                var idx = new_tokens.items.len - 1;
                while (idx > 0 and new_tokens.items[idx - 1] == last) : (idx -= 1) same += 1;
                if (same >= 8) result.repeated_token_loops += 1;
            }
        }
    }
    result.unique_generated_token_count = pred_counts.count();
    if (total_generated > 0) {
        var it = pred_counts.iterator();
        while (it.next()) |entry| {
            const p = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(total_generated));
            if (p > 0.0) result.generated_token_entropy -= p * @log(p);
        }
    }
    return result;
}

fn writeForwardResult(writer: anytype, allocator: std.mem.Allocator, result: *const ForwardBenchmarkResult) !void {
    try writer.print("{{\"batch_size\":{d},\"sequence_length\":{d},\"warmup_iterations\":{d},\"measured_iterations\":{d},\"evaluated_samples\":{d},\"valid_target_tokens\":{d},\"quality\":{{\"embedding_l2_loss\":", .{ result.batch_size, result.sequence_length, result.warmup_iterations, result.measured_iterations, result.evaluated_samples, result.valid_target_tokens });
    try writeOptionalFloat(writer, result.loss());
    try writer.writeAll(",\"top1_accuracy\":");
    try writeOptionalFloat(writer, result.top1());
    try writer.writeAll(",\"top5_accuracy\":");
    try writeOptionalFloat(writer, result.top5());
    try writer.writeAll(",\"top10_accuracy\":");
    try writeOptionalFloat(writer, result.top10());
    try writer.writeAll(",\"mean_reciprocal_rank\":");
    try writeOptionalFloat(writer, result.mrr());
    try writer.writeAll("},\"timing\":{");
    try writer.writeAll("\"tokenizer_time\":");
    try result.tokenizer_time.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"host_to_device_transfer_time\":null,\"forward_pass_time\":");
    try result.forward_time.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"ranking_time\":");
    try result.ranking_time.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"complete_batch_latency\":");
    try result.batch_time.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"valid_tokens_per_second\":");
    try result.tokens_per_second.writeJson(writer, allocator, "tokens_per_second");
    try writer.writeAll(",\"samples_per_second\":");
    try result.samples_per_second.writeJson(writer, allocator, "samples_per_second");
    try writer.writeAll("},\"memory\":{");
    try writer.print("\"activation_peak_bytes\":{d},\"temporary_ranking_buffer_peak_bytes\":{d}", .{ result.activation_peak_bytes, result.ranking_buffer_peak_bytes });
    try writer.writeAll("}}");
}

fn writeGenerationResult(writer: anytype, allocator: std.mem.Allocator, result: *const GenerationBenchmarkResult) !void {
    try writer.print("{{\"generation_length\":{d},\"warmup_iterations\":{d},\"measured_generations\":{d},\"prompt_count\":{d},\"first_token_latency\":", .{ result.generation_length, result.warmup_iterations, result.measured_generations, result.prompt_count });
    try result.first_token_latency.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"generated_token_latency\":");
    try result.generated_token_latency.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"complete_generation_latency\":");
    try result.complete_generation_latency.writeJson(writer, allocator, "milliseconds");
    try writer.writeAll(",\"generated_tokens_per_second\":");
    try result.generated_tokens_per_second.writeJson(writer, allocator, "tokens_per_second");
    try writer.writeAll(",\"quality_detections\":{");
    try writer.print("\"empty_outputs\":{d},\"repeated_token_loops\":{d},\"repeated_ngrams\":{d},\"invalid_tokens\":{d},\"excessive_unknown_token_outputs\":{d},\"unique_generated_token_count\":{d},\"generated_token_entropy\":{d}", .{ result.empty_outputs, result.repeated_token_loops, result.repeated_ngrams, result.invalid_tokens, result.unknown_token_outputs, result.unique_generated_token_count, result.generated_token_entropy });
    try writer.writeAll("}}");
}

fn writeReport(allocator: std.mem.Allocator, args: Args, metadata: checkpoint.CheckpointMetadata, checkpoint_digest: []const u8, load_seconds: f64, forward_results: []ForwardBenchmarkResult, generation_results: []GenerationBenchmarkResult) !void {
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    const writer = out.writer();
    const cpu_count = std.Thread.getCpuCount() catch 0;
    try writer.writeAll("{\"schema_version\":1,\"command\":\"jaide-benchmark\",\"timestamp_unix_seconds\":");
    try writer.print("{d}", .{std.time.timestamp()});
    try writer.writeAll(",\"configuration\":{");
    try writer.writeAll("\"checkpoint\":");
    try jsonEscape(writer, args.checkpoint_path.?);
    try writer.writeAll(",\"dataset\":");
    try jsonEscape(writer, args.dataset_path.?);
    try writer.print(",\"warmup_iterations\":{d},\"measured_iterations\":{d},\"ranking_chunk_size\":{d},\"seed\":{d},\"batch_sizes\":[", .{ args.warmup_iterations, args.measured_iterations, args.ranking_chunk_size, args.seed });
    for (args.batch_sizes, 0..) |value, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("],\"sequence_lengths\":[");
    for (args.sequence_lengths, 0..) |value, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("],\"generation_lengths\":[");
    for (args.generation_lengths, 0..) |value, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("]},\"checkpoint\":{");
    try writer.writeAll("\"sha256\":");
    try jsonEscape(writer, checkpoint_digest);
    try writer.writeAll(",\"metadata\":");
    try checkpoint.writeMetadataJson(writer, metadata);
    try writer.writeAll("},\"hardware_software_context\":{");
    try writer.print("\"cpu_thread_count\":{d},\"accelerator_backend\":", .{cpu_count});
    try jsonEscape(writer, if (build_options.gpu_acceleration) "futhark-cuda" else "futhark-cpu");
    try writer.print(",\"world_size\":null,\"rank_count\":null,\"model_dim\":{d},\"precision\":\"f16_rsf_f32_embedding\",\"futhark_backend\":", .{metadata.model_dim});
    try jsonEscape(writer, if (build_options.gpu_acceleration) "cuda" else "c");
    try writer.writeAll(",\"gpu_device_identity\":null},\"model_load_time_seconds\":");
    try writer.print("{d}", .{load_seconds});
    try writer.writeAll(",\"forward_benchmarks\":[");
    for (forward_results, 0..) |*result, i| {
        if (i > 0) try writer.writeByte(',');
        try writeForwardResult(writer, allocator, result);
    }
    try writer.writeAll("],\"generation_benchmarks\":[");
    for (generation_results, 0..) |*result, i| {
        if (i > 0) try writer.writeByte(',');
        try writeGenerationResult(writer, allocator, result);
    }
    const host_state = metadata.rsf_weight_bytes + metadata.rsf_velocity_bytes + metadata.embedding_weight_bytes + metadata.embedding_velocity_bytes;
    try writer.print("],\"memory\":{{\"serialized_checkpoint_bytes\":{d},\"checkpoint_payload_bytes\":{d},\"host_resident_model_state_bytes\":{d},\"rsf_weight_bytes\":{d},\"rsf_velocity_bytes\":{d},\"embedding_weight_bytes\":{d},\"embedding_velocity_bytes\":{d},\"externally_reported_process_gpu_memory\":null}},\"warnings\":[\"host-to-device transfer timing is null when unavailable from the active Futhark backend\",\"embedding_l2_loss is not perplexity\"],\"errors\":[]}}", .{ metadata.file_size, metadata.checkpoint_payload_bytes, host_state, metadata.rsf_weight_bytes, metadata.rsf_velocity_bytes, metadata.embedding_weight_bytes, metadata.embedding_velocity_bytes });
    try writeAtomic(allocator, args.output_path.?, out.items);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var args = try parseArgs(allocator);
    defer args.deinit();
    var load_timer = try std.time.Timer.start();
    var state = try checkpoint.loadCheckpointState(allocator, args.checkpoint_path.?);
    defer state.deinit();
    const load_seconds = @as(f64, @floatFromInt(load_timer.read())) / 1.0e9;
    const checkpoint_digest = try checkpoint.sha256FileHex(allocator, args.checkpoint_path.?);
    defer allocator.free(checkpoint_digest);
    var records = try loadJsonTextRecords(allocator, args.dataset_path.?, "text");
    defer {
        for (records) |*record| record.deinit(allocator);
        allocator.free(records);
    }
    const norms = try buildEmbeddingNorms(allocator, &state.embedding);
    defer allocator.free(norms);
    var forward_results = std.ArrayList(ForwardBenchmarkResult).init(allocator);
    defer {
        for (forward_results.items) |*result| result.deinit();
        forward_results.deinit();
    }
    for (args.batch_sizes) |batch_size| {
        for (args.sequence_lengths) |sequence_length| {
            std.debug.print("benchmark forward checkpoint={s} batch_size={d} sequence_length={d}\n", .{ args.checkpoint_path.?, batch_size, sequence_length });
            try forward_results.append(try runForwardBenchmark(allocator, &state, records, batch_size, sequence_length, args, norms));
        }
    }
    var generation_results = std.ArrayList(GenerationBenchmarkResult).init(allocator);
    defer {
        for (generation_results.items) |*result| result.deinit();
        generation_results.deinit();
    }
    if (args.prompts_path) |prompts_path| {
        var prompts = try loadJsonTextRecords(allocator, prompts_path, "prompt");
        defer {
            for (prompts) |*prompt| prompt.deinit(allocator);
            allocator.free(prompts);
        }
        for (args.generation_lengths) |generation_length| {
            std.debug.print("benchmark generation checkpoint={s} generation_length={d}\n", .{ args.checkpoint_path.?, generation_length });
            try generation_results.append(try runGenerationBenchmark(allocator, &state, prompts, generation_length, args, norms));
        }
    }
    try writeReport(allocator, args, state.metadata, checkpoint_digest, load_seconds, forward_results.items, generation_results.items);
}
