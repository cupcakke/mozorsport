const std = @import("std");

const SplitError = error{
    MissingInput,
    MissingTrainOutput,
    MissingValidationOutput,
    InvalidArgument,
    InvalidJsonl,
    InvalidTextField,
    OutputExists,
    DuplicateAcrossSplits,
};

const Args = struct {
    allocator: std.mem.Allocator,
    input: ?[]const u8 = null,
    train_output: ?[]const u8 = null,
    validation_output: ?[]const u8 = null,
    validation_ratio: f64 = 0.05,
    seed: u64 = 42,
    group_field: ?[]const u8 = null,
    overwrite: bool = false,
    duplicate_policy: []const u8 = "fail",
    duplicate_policy_owned: bool = false,

    fn init(allocator: std.mem.Allocator) Args {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Args) void {
        if (self.input) |v| self.allocator.free(v);
        if (self.train_output) |v| self.allocator.free(v);
        if (self.validation_output) |v| self.allocator.free(v);
        if (self.group_field) |v| self.allocator.free(v);
        if (self.duplicate_policy_owned) self.allocator.free(self.duplicate_policy);
    }
};

const Record = struct {
    raw: []u8,
    text: []u8,
    group_key: []u8,
    text_hash: [32]u8,
    bytes: usize,
    valid: bool = true,

    fn deinit(self: *Record, allocator: std.mem.Allocator) void {
        allocator.free(self.raw);
        allocator.free(self.text);
        allocator.free(self.group_key);
    }
};

const Group = struct {
    key: []const u8,
    record_indices: std.ArrayList(usize),
    hash: [32]u8,
    validation: bool = false,

    fn deinit(self: *Group) void {
        self.record_indices.deinit();
    }
};

fn jsonEscape(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |c| {
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

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);
    var args = Args.init(allocator);
    errdefer args.deinit();
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--input")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.input = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--train-output")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.train_output = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--validation-output")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.validation_output = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--validation-ratio")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.validation_ratio = try std.fmt.parseFloat(f64, argv[i]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.seed = try std.fmt.parseInt(u64, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--group-field")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            args.group_field = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--overwrite")) {
            args.overwrite = true;
        } else if (std.mem.eql(u8, arg, "--duplicate-policy")) {
            if (i + 1 >= argv.len) return SplitError.InvalidArgument;
            i += 1;
            if (args.duplicate_policy_owned) allocator.free(args.duplicate_policy);
            args.duplicate_policy = try allocator.dupe(u8, argv[i]);
            args.duplicate_policy_owned = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("jaide-dataset-split --input all.jsonl --train-output train.jsonl --validation-output validation.jsonl --validation-ratio 0.05 --seed 42 [--group-field document_id] [--duplicate-policy fail|deduplicate] [--overwrite]\n", .{});
            std.process.exit(0);
        } else {
            return SplitError.InvalidArgument;
        }
    }
    if (args.input == null) return SplitError.MissingInput;
    if (args.train_output == null) return SplitError.MissingTrainOutput;
    if (args.validation_output == null) return SplitError.MissingValidationOutput;
    if (!std.math.isFinite(args.validation_ratio) or args.validation_ratio <= 0.0 or args.validation_ratio >= 1.0) return SplitError.InvalidArgument;
    if (std.mem.eql(u8, args.train_output.?, args.validation_output.?)) return SplitError.InvalidArgument;
    if (!std.mem.eql(u8, args.duplicate_policy, "fail") and !std.mem.eql(u8, args.duplicate_policy, "deduplicate")) return SplitError.InvalidArgument;
    return args;
}

fn openRead(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

fn pathExists(path: []const u8) bool {
    const file = openRead(path) catch return false;
    file.close();
    return true;
}

fn createExclusive(path: []const u8, overwrite: bool) !std.fs.File {
    if (!overwrite and pathExists(path)) return SplitError.OutputExists;
    if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .truncate = true });
    return std.fs.cwd().createFile(path, .{ .truncate = true });
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

fn hashBytes(bytes: []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn hashGroup(seed: u64, key: []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var seed_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &seed_bytes, seed, .little);
    h.update(seed_bytes[0..]);
    h.update(key);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn autoGroupField(obj: anytype) ?[]const u8 {
    const keys = [_][]const u8{ "document_id", "doc_id", "source_document_id", "source_id", "source", "id" };
    for (keys) |key| {
        if (obj.get(key)) |value| {
            if (value == .string and value.string.len > 0) return value.string;
        }
    }
    return null;
}

fn loadRecords(allocator: std.mem.Allocator, args: Args, skipped_invalid: *u64, input_bytes: *u64) ![]Record {
    var file = try openRead(args.input.?);
    defer file.close();
    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();
    var records = std.ArrayList(Record).init(allocator);
    errdefer {
        for (records.items) |*record| record.deinit(allocator);
        records.deinit();
    }
    while (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |line_raw| {
        defer allocator.free(line_raw);
        input_bytes.* += line_raw.len + 1;
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (std.mem.trim(u8, line, " \t\r\n").len == 0) {
            skipped_invalid.* += 1;
            continue;
        }
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always }) catch {
            skipped_invalid.* += 1;
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            skipped_invalid.* += 1;
            continue;
        }
        const obj = parsed.value.object;
        const text_value = obj.get("text") orelse {
            skipped_invalid.* += 1;
            continue;
        };
        if (text_value != .string) {
            skipped_invalid.* += 1;
            continue;
        }
        if (text_value.string.len == 0) {
            skipped_invalid.* += 1;
            continue;
        }
        const group_value = if (args.group_field) |field| blk: {
            if (obj.get(field)) |v| {
                if (v == .string and v.string.len > 0) break :blk v.string;
            }
            break :blk autoGroupField(obj) orelse text_value.string;
        } else autoGroupField(obj) orelse text_value.string;
        const raw_owned = try allocator.dupe(u8, line);
        errdefer allocator.free(raw_owned);
        const text_owned = try allocator.dupe(u8, text_value.string);
        errdefer allocator.free(text_owned);
        const group_owned = try allocator.dupe(u8, group_value);
        errdefer allocator.free(group_owned);
        try records.append(.{ .raw = raw_owned, .text = text_owned, .group_key = group_owned, .text_hash = hashBytes(text_owned), .bytes = raw_owned.len });
    }
    return try records.toOwnedSlice();
}

fn makeGroups(allocator: std.mem.Allocator, records: []const Record, seed: u64) ![]Group {
    var map = std.StringHashMap(usize).init(allocator);
    defer map.deinit();
    var groups = std.ArrayList(Group).init(allocator);
    errdefer {
        for (groups.items) |*g| g.deinit();
        groups.deinit();
    }
    for (records, 0..) |record, idx| {
        const entry = try map.getOrPut(record.group_key);
        if (!entry.found_existing) {
            entry.value_ptr.* = groups.items.len;
            try groups.append(.{ .key = record.group_key, .record_indices = std.ArrayList(usize).init(allocator), .hash = hashGroup(seed, record.group_key) });
        }
        try groups.items[entry.value_ptr.*].record_indices.append(idx);
    }
    return try groups.toOwnedSlice();
}

fn groupLessThan(_: void, a: Group, b: Group) bool {
    const order = std.mem.order(u8, a.hash[0..], b.hash[0..]);
    if (order != .eq) return order == .lt;
    return std.mem.lessThan(u8, a.key, b.key);
}

fn assignSplits(groups: []Group, total_records: usize, ratio: f64) void {
    std.mem.sort(Group, groups, {}, groupLessThan);
    const target = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(total_records)) * ratio))));
    var validation_count: usize = 0;
    for (groups) |*group| {
        if (validation_count < target) {
            group.validation = true;
            validation_count += group.record_indices.items.len;
        } else {
            group.validation = false;
        }
    }
}

fn writeOutputs(allocator: std.mem.Allocator, args: Args, records: []Record, groups: []const Group, train_count: *u64, validation_count: *u64, train_bytes: *u64, validation_bytes: *u64, duplicates: *u64) !void {
    const train_tmp = try std.fmt.allocPrint(allocator, "{s}.{d}.tmp", .{ args.train_output.?, std.time.nanoTimestamp() });
    defer allocator.free(train_tmp);
    const validation_tmp = try std.fmt.allocPrint(allocator, "{s}.{d}.tmp", .{ args.validation_output.?, std.time.nanoTimestamp() });
    defer allocator.free(validation_tmp);
    var train_committed = false;
    var validation_committed = false;
    defer if (!train_committed) deletePath(train_tmp);
    defer if (!validation_committed) deletePath(validation_tmp);
    {
        var probe = try createExclusive(args.train_output.?, args.overwrite);
        probe.close();
        deletePath(args.train_output.?);
    }
    {
        var probe = try createExclusive(args.validation_output.?, args.overwrite);
        probe.close();
        deletePath(args.validation_output.?);
    }
    var train_file = try createWrite(train_tmp);
    defer train_file.close();
    var validation_file = try createWrite(validation_tmp);
    defer validation_file.close();
    var seen = std.AutoHashMap([32]u8, u8).init(allocator);
    defer seen.deinit();
    for (groups) |group| {
        const split_id: u8 = if (group.validation) 2 else 1;
        for (group.record_indices.items) |idx| {
            if (!records[idx].valid) continue;
            const entry = try seen.getOrPut(records[idx].text_hash);
            if (entry.found_existing and entry.value_ptr.* != split_id) {
                duplicates.* += 1;
                if (std.mem.eql(u8, args.duplicate_policy, "fail")) return SplitError.DuplicateAcrossSplits;
                continue;
            } else if (!entry.found_existing) {
                entry.value_ptr.* = split_id;
            }
            if (group.validation) {
                try validation_file.writeAll(records[idx].raw);
                try validation_file.writeByte('\n');
                validation_count.* += 1;
                validation_bytes.* += records[idx].bytes + 1;
            } else {
                try train_file.writeAll(records[idx].raw);
                try train_file.writeByte('\n');
                train_count.* += 1;
                train_bytes.* += records[idx].bytes + 1;
            }
        }
    }
    try train_file.sync();
    try validation_file.sync();
    train_file.close();
    validation_file.close();
    try renamePath(train_tmp, args.train_output.?);
    train_committed = true;
    try renamePath(validation_tmp, args.validation_output.?);
    validation_committed = true;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var args = try parseArgs(allocator);
    defer args.deinit();
    var skipped_invalid: u64 = 0;
    var input_bytes: u64 = 0;
    var records = try loadRecords(allocator, args, &skipped_invalid, &input_bytes);
    defer {
        for (records) |*record| record.deinit(allocator);
        allocator.free(records);
    }
    var groups = try makeGroups(allocator, records, args.seed);
    defer {
        for (groups) |*group| group.deinit();
        allocator.free(groups);
    }
    assignSplits(groups, records.len, args.validation_ratio);
    var train_count: u64 = 0;
    var validation_count: u64 = 0;
    var train_bytes: u64 = 0;
    var validation_bytes: u64 = 0;
    var duplicates: u64 = 0;
    try writeOutputs(allocator, args, records, groups, &train_count, &validation_count, &train_bytes, &validation_bytes, &duplicates);
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("{\"schema_version\":1,\"command\":\"jaide-dataset-split\",\"input\":");
    try jsonEscape(stdout, args.input.?);
    try stdout.writeAll(",\"train_output\":");
    try jsonEscape(stdout, args.train_output.?);
    try stdout.writeAll(",\"validation_output\":");
    try jsonEscape(stdout, args.validation_output.?);
    try stdout.print(",\"validation_ratio\":{d},\"seed\":{d},\"records_loaded\":{d},\"groups\":{d},\"train_samples\":{d},\"validation_samples\":{d},\"input_bytes\":{d},\"train_bytes\":{d},\"validation_bytes\":{d},\"skipped_invalid_records\":{d},\"duplicate_text_cross_split_count\":{d},\"duplicate_policy\":", .{ args.validation_ratio, args.seed, records.len, groups.len, train_count, validation_count, input_bytes, train_bytes, validation_bytes, skipped_invalid, duplicates });
    try jsonEscape(stdout, args.duplicate_policy);
    try stdout.writeAll("}\n");
}
