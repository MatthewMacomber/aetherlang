//! Aether Runtime Terminal User Interface (TUI) Subsystem
//! TrueColor 24-bit ANSI Canvas with Unicode Half-Blocks (2x vertical resolution),
//! terminal widgets, telemetry sparklines, and non-blocking input.

const std = @import("std");
const graphics = @import("runtime_graphics.zig");
const Color = graphics.Color;
const Framebuffer = graphics.Framebuffer;

pub const TuiCanvas = struct {
    width: usize,
    height: usize,
    fb: Framebuffer,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize) !TuiCanvas {
        return .{
            .width = width,
            .height = height,
            .fb = try Framebuffer.init(allocator, width, height),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TuiCanvas) void {
        self.fb.deinit();
    }

    /// Renders the internal framebuffer to an ANSI terminal using Unicode half-blocks (▀)
    /// Each terminal character cell renders two vertical pixels.
    pub fn render(self: *const TuiCanvas, writer: anytype) !void {
        var prev_fg: ?Color = null;
        var prev_bg: ?Color = null;

        var y: usize = 0;
        while (y < self.height) : (y += 2) {
            var x: usize = 0;
            while (x < self.width) : (x += 1) {
                const top = self.fb.getPixel(@intCast(x), @intCast(y));
                const bottom = if (y + 1 < self.height)
                    self.fb.getPixel(@intCast(x), @intCast(y + 1))
                else
                    Color.black;

                // ponytail: ANSI escape sequence compression; skip color emissions when unchanged
                if (prev_fg == null or prev_fg.?.r != top.r or prev_fg.?.g != top.g or prev_fg.?.b != top.b) {
                    try writer.print("\x1b[38;2;{d};{d};{d}m", .{ top.r, top.g, top.b });
                    prev_fg = top;
                }
                if (prev_bg == null or prev_bg.?.r != bottom.r or prev_bg.?.g != bottom.g or prev_bg.?.b != bottom.b) {
                    try writer.print("\x1b[48;2;{d};{d};{d}m", .{ bottom.r, bottom.g, bottom.b });
                    prev_bg = bottom;
                }

                // Unicode U+2580 UPPER HALF BLOCK
                try writer.writeAll("\xe2\x96\x80");
            }
            try writer.writeAll("\x1b[0m\n");
            prev_fg = null;
            prev_bg = null;
        }
    }
};

pub fn clearScreen(writer: anytype) !void {
    try writer.writeAll("\x1b[2J\x1b[H");
}

pub fn hideCursor(writer: anytype) !void {
    try writer.writeAll("\x1b[?25l");
}

pub fn showCursor(writer: anytype) !void {
    try writer.writeAll("\x1b[?25h");
}

pub fn renderSparkline(values: []const f64, max_width: usize, writer: anytype) !void {
    if (values.len == 0) return;
    const bars = [_][]const u8{ " ", "▂", "▃", "▄", "▅", "▆", "▇", "█" };

    var min_v = values[0];
    var max_v = values[0];
    for (values) |v| {
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const range = if (max_v > min_v) max_v - min_v else 1.0;

    const count = @min(values.len, max_width);
    const start_idx = values.len - count;
    for (values[start_idx..]) |v| {
        const norm = @max(0.0, @min(1.0, (v - min_v) / range));
        const bar_idx: usize = @min(7, @as(usize, @intFromFloat(norm * 7.99)));
        try writer.writeAll(bars[bar_idx]);
    }
}

pub fn renderProgressBar(label: []const u8, progress: f64, width: usize, writer: anytype) !void {
    const p = @max(0.0, @min(1.0, progress));
    const fill_chars: usize = @intFromFloat(p * @as(f64, @floatFromInt(width)));
    try writer.print("{s} [", .{label});
    var i: usize = 0;
    while (i < fill_chars) : (i += 1) {
        try writer.writeAll("█");
    }
    while (i < width) : (i += 1) {
        try writer.writeAll("░");
    }
    try writer.print("] {d:.1}%\n", .{p * 100.0});
}

// Windows Console Input Support
pub const WindowsConsoleInput = struct {
    pub fn isKeyDown(vk_code: u16) bool {
        if (@import("builtin").os.tag != .windows) return false;
        const user32 = struct {
            extern "user32" fn GetAsyncKeyState(vKey: c_int) callconv(.c) i16;
        };
        const state = user32.GetAsyncKeyState(@intCast(vk_code));
        return (state & @as(i16, -32768)) != 0;
    }
};

pub const BufferWriter = struct {
    buf: []u8,
    len: usize = 0,

    pub fn writeAll(self: *BufferWriter, bytes: []const u8) !void {
        if (self.len + bytes.len > self.buf.len) return error.NoSpaceLeft;
        @memcpy(self.buf[self.len .. self.len + bytes.len], bytes);
        self.len += bytes.len;
    }

    pub fn print(self: *BufferWriter, comptime fmt: []const u8, args: anytype) !void {
        const remaining = self.buf[self.len..];
        const res = try std.fmt.bufPrint(remaining, fmt, args);
        self.len += res.len;
    }
};

test "tui canvas creation and sparkline" {
    var canvas = try TuiCanvas.init(std.testing.allocator, 16, 8);
    defer canvas.deinit();

    canvas.fb.clear(Color.black);
    canvas.fb.drawRect(2, 2, 8, 4, Color.green, true);

    var buf: [4096]u8 = undefined;
    var bw = BufferWriter{ .buf = &buf };
    try canvas.render(&bw);
    try std.testing.expect(bw.len > 0);

    bw.len = 0;
    const data = [_]f64{ 1.0, 2.5, 4.0, 2.0, 5.0 };
    try renderSparkline(&data, 5, &bw);
    try std.testing.expect(bw.len > 0);
}
