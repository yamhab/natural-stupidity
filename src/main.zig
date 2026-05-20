const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var buffer: [4096]u8 = undefined;

    var writer = std.Io.File.stdout().writer(io, &buffer);
    var stdout = &writer.interface;

    try stdout.print("cock and ball torture\n", .{});
    try writer.flush();
}
