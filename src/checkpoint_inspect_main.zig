const std = @import("std");
const checkpoint = @import("core/checkpoint.zig");

const InspectError = error{
    MissingCheckpoint,
    InvalidArgument,
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);
    var path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--checkpoint")) {
            if (i + 1 >= argv.len) return InspectError.InvalidArgument;
            i += 1;
            path = argv[i];
        } else if (std.mem.eql(u8, argv[i], "--help")) {
            std.debug.print("jaide-checkpoint-inspect --checkpoint /checkpoints/epoch_003/model.ckpt\n", .{});
            return;
        } else {
            return InspectError.InvalidArgument;
        }
    }
    const checkpoint_path = path orelse return InspectError.MissingCheckpoint;
    const metadata = try checkpoint.inspectCheckpoint(checkpoint_path);
    const digest = try checkpoint.sha256FileHex(allocator, checkpoint_path);
    defer allocator.free(digest);
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("{\"schema_version\":1,\"command\":\"jaide-checkpoint-inspect\",\"checkpoint\":");
    try jsonEscape(stdout, checkpoint_path);
    try stdout.writeAll(",\"sha256\":");
    try jsonEscape(stdout, digest);
    try stdout.writeAll(",\"metadata\":");
    try checkpoint.writeMetadataJson(stdout, metadata);
    try stdout.writeAll("}\n");
}
