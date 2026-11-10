//! Aether Runtime Immediate-Mode GUI Subsystem
//! Clean, responsive immediate-mode controls, container panels, and real-time tensor telemetry plots.

const std = @import("std");
const graphics = @import("runtime_graphics.zig");
const Color = graphics.Color;
const Framebuffer = graphics.Framebuffer;

pub const GuiContext = struct {
    mouse_x: isize = 0,
    mouse_y: isize = 0,
    mouse_down: bool = false,
    mouse_clicked: bool = false,
    hot_id: u64 = 0,
    active_id: u64 = 0,

    pub fn update(self: *GuiContext, mx: isize, my: isize, mdown: bool, mclicked: bool) void {
        self.mouse_x = mx;
        self.mouse_y = my;
        self.mouse_down = mdown;
        self.mouse_clicked = mclicked;
        if (!mdown) self.active_id = 0;
    }

    pub fn isMouseInRect(self: *const GuiContext, x: isize, y: isize, w: isize, h: isize) bool {
        return self.mouse_x >= x and self.mouse_x < x + w and self.mouse_y >= y and self.mouse_y < y + h;
    }

    pub fn button(self: *GuiContext, fb: *Framebuffer, id: u64, label: []const u8, x: isize, y: isize, w: isize, h: isize) bool {
        const hover = self.isMouseInRect(x, y, w, h);
        if (hover) self.hot_id = id;

        var clicked = false;
        if (hover and self.mouse_clicked) {
            self.active_id = id;
            clicked = true;
        }

        const bg = if (self.active_id == id)
            Color.hex(0x2A52BE)
        else if (hover)
            Color.hex(0x406080)
        else
            Color.hex(0x283040);

        fb.drawRect(x, y, w, h, bg, true);
        fb.drawRect(x, y, w, h, Color.hex(0x708090), false);

        // Center text label
        const text_len: isize = @intCast(label.len * 8);
        const text_x = x + @divTrunc(w - text_len, 2);
        const text_y = y + @divTrunc(h - 8, 2);
        fb.drawText(text_x, text_y, label, Color.white);

        return clicked;
    }

    pub fn slider(self: *GuiContext, fb: *Framebuffer, id: u64, label: []const u8, val: f32, min_val: f32, max_val: f32, x: isize, y: isize, w: isize, h: isize) f32 {
        const hover = self.isMouseInRect(x, y, w, h);
        if (hover) self.hot_id = id;
        if (hover and self.mouse_down) self.active_id = id;

        var cur_val = val;
        if (self.active_id == id) {
            const rel_x = @max(0, @min(w, self.mouse_x - x));
            const frac = @as(f32, @floatFromInt(rel_x)) / @as(f32, @floatFromInt(w));
            cur_val = min_val + frac * (max_val - min_val);
        }

        // Trough
        fb.drawRect(x, y + @divTrunc(h, 2) - 2, w, 4, Color.hex(0x202028), true);
        fb.drawRect(x, y + @divTrunc(h, 2) - 2, w, 4, Color.hex(0x505060), false);

        // Handle
        const norm = @max(0.0, @min(1.0, (cur_val - min_val) / (max_val - min_val)));
        const handle_x = x + @as(isize, @intFromFloat(norm * @as(f32, @floatFromInt(w - 10))));
        fb.drawRect(handle_x, y, 10, h, Color.hex(0x00A0E0), true);
        fb.drawRect(handle_x, y, 10, h, Color.white, false);

        // Text
        var buf: [32]u8 = undefined;
        const txt = std.fmt.bufPrint(&buf, "{s}: {d:.2}", .{ label, cur_val }) catch label;
        fb.drawText(x, y - 10, txt, Color.hex(0xD0D0E0));

        return cur_val;
    }

    pub fn panel(self: *GuiContext, fb: *Framebuffer, title: []const u8, x: isize, y: isize, w: isize, h: isize) void {
        _ = self;
        // Panel background & title bar
        fb.drawRect(x, y, w, h, Color.hex(0x181820), true);
        fb.drawRect(x, y, w, 20, Color.hex(0x282838), true);
        fb.drawRect(x, y, w, h, Color.hex(0x404050), false);
        fb.drawText(x + 6, y + 6, title, Color.yellow);
    }

    pub fn plotTensor(self: *GuiContext, fb: *Framebuffer, title: []const u8, data: []const f64, x: isize, y: isize, w: isize, h: isize) void {
        _ = self;
        if (data.len < 2 or w <= 0 or h <= 0) return;

        fb.drawRect(x, y, w, h, Color.hex(0x101018), true);
        fb.drawRect(x, y, w, h, Color.hex(0x303040), false);
        fb.drawText(x + 4, y + 4, title, Color.cyan);

        var min_v = data[0];
        var max_v = data[0];
        for (data) |v| {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        const range = if (max_v > min_v) max_v - min_v else 1.0;

        const inner_h = h - 20;
        const inner_y = y + 15;

        var i: usize = 0;
        while (i < data.len - 1) : (i += 1) {
            const x0 = x + @as(isize, @intCast((i * @as(usize, @intCast(w - 8))) / (data.len - 1))) + 4;
            const x1 = x + @as(isize, @intCast(((i + 1) * @as(usize, @intCast(w - 8))) / (data.len - 1))) + 4;

            const n0 = (data[i] - min_v) / range;
            const n1 = (data[i + 1] - min_v) / range;

            const y0 = inner_y + inner_h - @as(isize, @intFromFloat(n0 * @as(f64, @floatFromInt(inner_h - 4))));
            const y1 = inner_y + inner_h - @as(isize, @intFromFloat(n1 * @as(f64, @floatFromInt(inner_h - 4))));

            fb.drawLine(x0, y0, x1, y1, Color.green);
        }
    }
};

test "gui button and slider interaction" {
    var fb = try Framebuffer.init(std.testing.allocator, 100, 100);
    defer fb.deinit();

    var ctx = GuiContext{};
    ctx.update(25, 25, true, true);

    const clicked = ctx.button(&fb, 1, "Play", 10, 10, 50, 20);
    try std.testing.expect(clicked);

    const s_val = ctx.slider(&fb, 2, "Vol", 0.5, 0.0, 1.0, 10, 40, 80, 16);
    try std.testing.expect(s_val >= 0.0 and s_val <= 1.0);
}

