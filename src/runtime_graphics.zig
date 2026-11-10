//! Aether Runtime Graphics Subsystem
//! High-performance pure-Zig 2D/3D software rasterizer, framebuffer, tensor bridge, and image export.

const std = @import("std");

pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8 = 255,

    pub const black = Color{ .r = 0, .g = 0, .b = 0, .a = 255 };
    pub const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
    pub const red = Color{ .r = 255, .g = 30, .b = 30, .a = 255 };
    pub const green = Color{ .r = 30, .g = 220, .b = 30, .a = 255 };
    pub const blue = Color{ .r = 30, .g = 100, .b = 255, .a = 255 };
    pub const yellow = Color{ .r = 255, .g = 220, .b = 30, .a = 255 };
    pub const cyan = Color{ .r = 30, .g = 220, .b = 255, .a = 255 };
    pub const magenta = Color{ .r = 255, .g = 30, .b = 220, .a = 255 };
    pub const gray = Color{ .r = 120, .g = 120, .b = 120, .a = 255 };
    pub const dark_gray = Color{ .r = 40, .g = 40, .b = 40, .a = 255 };
    pub const transparent = Color{ .r = 0, .g = 0, .b = 0, .a = 0 };

    pub fn rgb(r: u8, g: u8, b: u8) Color {
        return .{ .r = r, .g = g, .b = b, .a = 255 };
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: u8) Color {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn hex(val: u32) Color {
        return .{
            .r = @intCast((val >> 16) & 0xFF),
            .g = @intCast((val >> 8) & 0xFF),
            .b = @intCast(val & 0xFF),
            .a = if (val > 0xFFFFFF) @intCast((val >> 24) & 0xFF) else 255,
        };
    }

    pub fn blend(dst: Color, src: Color) Color {
        if (src.a == 255) return src;
        if (src.a == 0) return dst;
        const sa = @as(u32, src.a);
        const da = 255 - sa;
        return .{
            .r = @intCast((@as(u32, src.r) * sa + @as(u32, dst.r) * da) / 255),
            .g = @intCast((@as(u32, src.g) * sa + @as(u32, dst.g) * da) / 255),
            .b = @intCast((@as(u32, src.b) * sa + @as(u32, dst.b) * da) / 255),
            .a = @intCast(@min(255, @as(u32, src.a) + dst.a)),
        };
    }
};

pub const Framebuffer = struct {
    width: usize,
    height: usize,
    pixels: []Color,
    depth: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize) !Framebuffer {
        const total = width * height;
        const px = try allocator.alloc(Color, total);
        const dp = try allocator.alloc(f32, total);
        @memset(px, Color.black);
        @memset(dp, 10000.0);
        return .{
            .width = width,
            .height = height,
            .pixels = px,
            .depth = dp,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Framebuffer) void {
        self.allocator.free(self.pixels);
        self.allocator.free(self.depth);
    }

    pub fn resize(self: *Framebuffer, width: usize, height: usize) !void {
        if (width == self.width and height == self.height) return;
        const total = width * height;
        const new_px = try self.allocator.alloc(Color, total);
        const new_dp = try self.allocator.alloc(f32, total);
        @memset(new_px, Color.black);
        @memset(new_dp, 10000.0);
        self.allocator.free(self.pixels);
        self.allocator.free(self.depth);
        self.width = width;
        self.height = height;
        self.pixels = new_px;
        self.depth = new_dp;
    }

    pub fn clear(self: *Framebuffer, c: Color) void {
        @memset(self.pixels, c);
        @memset(self.depth, 10000.0);
    }

    pub fn setPixel(self: *Framebuffer, x: isize, y: isize, c: Color) void {
        if (x < 0 or y < 0 or x >= @as(isize, @intCast(self.width)) or y >= @as(isize, @intCast(self.height))) return;
        const idx = @as(usize, @intCast(y)) * self.width + @as(usize, @intCast(x));
        self.pixels[idx] = Color.blend(self.pixels[idx], c);
    }

    pub fn getPixel(self: *const Framebuffer, x: isize, y: isize) Color {
        if (x < 0 or y < 0 or x >= @as(isize, @intCast(self.width)) or y >= @as(isize, @intCast(self.height))) return Color.black;
        return self.pixels[@as(usize, @intCast(y)) * self.width + @as(usize, @intCast(x))];
    }

    pub fn drawLine(self: *Framebuffer, x0_in: isize, y0_in: isize, x1_in: isize, y1_in: isize, c: Color) void {
        var x0 = x0_in;
        var y0 = y0_in;
        const x1 = x1_in;
        const y1 = y1_in;
        const dx = @abs(x1 - x0);
        const dy = -@as(isize, @intCast(@abs(y1 - y0)));
        const sx: isize = if (x0 < x1) 1 else -1;
        const sy: isize = if (y0 < y1) 1 else -1;
        var err: isize = @as(isize, @intCast(dx)) + dy;

        while (true) {
            self.setPixel(x0, y0, c);
            if (x0 == x1 and y0 == y1) break;
            const e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= @as(isize, @intCast(dx))) {
                err += @as(isize, @intCast(dx));
                y0 += sy;
            }
        }
    }

    pub fn drawRect(self: *Framebuffer, x: isize, y: isize, w: isize, h: isize, c: Color, filled: bool) void {
        if (w <= 0 or h <= 0) return;
        if (filled) {
            var j: isize = y;
            while (j < y + h) : (j += 1) {
                var i: isize = x;
                while (i < x + w) : (i += 1) {
                    self.setPixel(i, j, c);
                }
            }
        } else {
            self.drawLine(x, y, x + w - 1, y, c);
            self.drawLine(x, y + h - 1, x + w - 1, y + h - 1, c);
            self.drawLine(x, y, x, y + h - 1, c);
            self.drawLine(x + w - 1, y, x + w - 1, y + h - 1, c);
        }
    }

    pub fn drawCircle(self: *Framebuffer, cx: isize, cy: isize, r: isize, c: Color, filled: bool) void {
        if (r <= 0) return;
        var x: isize = 0;
        var y: isize = r;
        var d: isize = 3 - 2 * r;

        while (x <= y) {
            if (filled) {
                self.drawLine(cx - x, cy - y, cx + x, cy - y, c);
                self.drawLine(cx - x, cy + y, cx + x, cy + y, c);
                self.drawLine(cx - y, cy - x, cx + y, cy - x, c);
                self.drawLine(cx - y, cy + x, cx + y, cy + x, c);
            } else {
                self.setPixel(cx + x, cy + y, c);
                self.setPixel(cx - x, cy + y, c);
                self.setPixel(cx + x, cy - y, c);
                self.setPixel(cx - x, cy - y, c);
                self.setPixel(cx + y, cy + x, c);
                self.setPixel(cx - y, cy + x, c);
                self.setPixel(cx + y, cy - x, c);
                self.setPixel(cx - y, cy - x, c);
            }
            if (d < 0) {
                d += 4 * x + 6;
            } else {
                d += 4 * (x - y) + 10;
                y -= 1;
            }
            x += 1;
        }
    }

    pub fn drawTriangle(self: *Framebuffer, x0: isize, y0: isize, x1: isize, y1: isize, x2: isize, y2: isize, c: Color, filled: bool) void {
        if (!filled) {
            self.drawLine(x0, y0, x1, y1, c);
            self.drawLine(x1, y1, x2, y2, c);
            self.drawLine(x2, y2, x0, y0, c);
            return;
        }

        // Bounding box scanline rasterization
        const min_x = @max(0, @min(x0, @min(x1, x2)));
        const max_x = @min(@as(isize, @intCast(self.width)) - 1, @max(x0, @max(x1, x2)));
        const min_y = @max(0, @min(y0, @min(y1, y2)));
        const max_y = @min(@as(isize, @intCast(self.height)) - 1, @max(y0, @max(y1, y2)));

        const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (area == 0) return;

        var y = min_y;
        while (y <= max_y) : (y += 1) {
            var x = min_x;
            while (x <= max_x) : (x += 1) {
                const w0 = (x1 - x) * (y2 - y) - (y1 - y) * (x2 - x);
                const w1 = (x2 - x) * (y0 - y) - (y2 - y) * (x0 - x);
                const w2 = (x0 - x) * (y1 - y) - (y0 - y) * (x1 - x);
                if ((w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0)) {
                    self.setPixel(x, y, c);
                }
            }
        }
    }

    pub fn drawText(self: *Framebuffer, x_pos: isize, y_pos: isize, text: []const u8, c: Color) void {
        var cur_x = x_pos;
        for (text) |ch| {
            if (ch == '\n') {
                cur_x = x_pos;
                continue;
            }
            drawChar8x8(self, cur_x, y_pos, ch, c);
            cur_x += 8;
        }
    }

    pub fn drawSprite(
        self: *Framebuffer,
        sprite: *const Framebuffer,
        dst_x: isize,
        dst_y: isize,
        src_x: usize,
        src_y: usize,
        w: usize,
        h: usize,
        flip_x: bool,
    ) void {
        var sy: usize = 0;
        while (sy < h) : (sy += 1) {
            var sx: usize = 0;
            while (sx < w) : (sx += 1) {
                const sample_x = if (flip_x) (src_x + w - 1 - sx) else (src_x + sx);
                const sample_y = src_y + sy;
                const px = sprite.getPixel(@intCast(sample_x), @intCast(sample_y));
                if (px.a > 0) {
                    self.setPixel(dst_x + @as(isize, @intCast(sx)), dst_y + @as(isize, @intCast(sy)), px);
                }
            }
        }
    }

    pub fn exportBMP(self: *const Framebuffer, file_path: []const u8) !void {
        // ponytail: BMP 24-bit with 4-byte row padding requires no external image libs
        const row_stride = (self.width * 3 + 3) & ~@as(usize, 3);
        const image_size = row_stride * self.height;
        const file_size = 54 + image_size;

        const buf = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(buf);
        @memset(buf, 0);

        // Signature 'BM'
        buf[0] = 'B';
        buf[1] = 'M';
        std.mem.writeInt(u32, buf[2..6], @intCast(file_size), .little);
        std.mem.writeInt(u32, buf[10..14], 54, .little); // Data offset
        std.mem.writeInt(u32, buf[14..18], 40, .little); // Header size
        std.mem.writeInt(i32, buf[18..22], @intCast(self.width), .little);
        std.mem.writeInt(i32, buf[22..26], @intCast(self.height), .little); // bottom-up
        std.mem.writeInt(u16, buf[26..28], 1, .little); // Planes
        std.mem.writeInt(u16, buf[28..30], 24, .little); // Bits per pixel
        std.mem.writeInt(u32, buf[34..38], @intCast(image_size), .little);

        var y: usize = 0;
        var out_idx: usize = 54;
        while (y < self.height) : (y += 1) {
            // BMP bottom-to-top
            const src_y = self.height - 1 - y;
            var x: usize = 0;
            const row_start = out_idx;
            while (x < self.width) : (x += 1) {
                const px = self.pixels[src_y * self.width + x];
                buf[out_idx] = px.b;
                buf[out_idx + 1] = px.g;
                buf[out_idx + 2] = px.r;
                out_idx += 3;
            }
            out_idx = row_start + row_stride;
        }

        try writeFileDirect(file_path, buf);
    }
};

const win32_io = struct {
    extern "kernel32" fn CreateFileA(
        lpFileName: [*:0]const u8,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: ?*anyopaque,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: ?*anyopaque,
    ) callconv(.c) ?*anyopaque;
    extern "kernel32" fn WriteFile(
        hFile: ?*anyopaque,
        lpBuffer: [*]const u8,
        nNumberOfBytesToWrite: u32,
        lpNumberOfBytesWritten: ?*u32,
        lpOverlapped: ?*anyopaque,
    ) callconv(.c) i32;
    extern "kernel32" fn CloseHandle(hObject: ?*anyopaque) callconv(.c) i32;
};

pub fn writeFileDirect(file_path: []const u8, data: []const u8) !void {
    if (@import("builtin").os.tag == .windows) {
        var path_z: [512:0]u8 = undefined;
        if (file_path.len >= 511) return error.PathTooLong;
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;

        const GENERIC_WRITE = 0x40000000;
        const CREATE_ALWAYS = 2;
        const FILE_ATTRIBUTE_NORMAL = 0x80;

        const handle = win32_io.CreateFileA(
            &path_z,
            GENERIC_WRITE,
            0,
            null,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (handle == null or handle == @as(?*anyopaque, @ptrFromInt(std.math.maxInt(usize)))) {
            return error.CannotCreateFile;
        }
        defer _ = win32_io.CloseHandle(handle);

        var written: u32 = 0;
        _ = win32_io.WriteFile(handle, data.ptr, @intCast(data.len), &written, null);
    }
}

// 3D Math & Software Rasterizer
pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn new(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub fn scale(a: Vec3, s: f32) Vec3 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s };
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn norm(a: Vec3) f32 {
        return @sqrt(a.dot(a));
    }

    pub fn normalize(a: Vec3) Vec3 {
        const len = a.norm();
        if (len < 1e-6) return a;
        return a.scale(1.0 / len);
    }
};

pub const Camera3D = struct {
    position: Vec3 = .{ .x = 0, .y = 2, .z = -5 },
    target: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    up: Vec3 = .{ .x = 0, .y = 1, .z = 0 },
    fov: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,

    pub fn project(self: Camera3D, p_world: Vec3, screen_w: f32, screen_h: f32) ?struct { x: f32, y: f32, z: f32 } {
        const forward = self.target.sub(self.position).normalize();
        const right = forward.cross(self.up).normalize();
        const true_up = right.cross(forward).normalize();

        const rel = p_world.sub(self.position);
        const cam_x = rel.dot(right);
        const cam_y = rel.dot(true_up);
        const cam_z = rel.dot(forward);

        if (cam_z <= self.near or cam_z >= self.far) return null;

        const aspect = screen_w / screen_h;
        const rad = self.fov * std.math.pi / 180.0;
        const fov_scale = 1.0 / @tan(rad * 0.5);

        const proj_x = (cam_x / cam_z) * (fov_scale / aspect);
        const proj_y = (cam_y / cam_z) * fov_scale;

        const screen_x = (proj_x * 0.5 + 0.5) * screen_w;
        const screen_y = (1.0 - (proj_y * 0.5 + 0.5)) * screen_h;

        return .{ .x = screen_x, .y = screen_y, .z = cam_z };
    }
};

pub fn drawWireCube3D(fb: *Framebuffer, cam: Camera3D, center: Vec3, size: f32, c: Color) void {
    const h = size * 0.5;
    const verts = [_]Vec3{
        center.add(Vec3.new(-h, -h, -h)),
        center.add(Vec3.new(h, -h, -h)),
        center.add(Vec3.new(h, h, -h)),
        center.add(Vec3.new(-h, h, -h)),
        center.add(Vec3.new(-h, -h, h)),
        center.add(Vec3.new(h, -h, h)),
        center.add(Vec3.new(h, h, h)),
        center.add(Vec3.new(-h, h, h)),
    };

    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 },
        .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 },
        .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 },
    };

    const sw = @as(f32, @floatFromInt(fb.width));
    const sh = @as(f32, @floatFromInt(fb.height));

    for (edges) |edge| {
        const p1 = cam.project(verts[edge[0]], sw, sh);
        const p2 = cam.project(verts[edge[1]], sw, sh);
        if (p1 != null and p2 != null) {
            fb.drawLine(
                @intFromFloat(p1.?.x),
                @intFromFloat(p1.?.y),
                @intFromFloat(p2.?.x),
                @intFromFloat(p2.?.y),
                c,
            );
        }
    }
}

// 8x8 ASCII Bitmap Font
fn drawChar8x8(fb: *Framebuffer, x: isize, y: isize, ch: u8, c: Color) void {
    const glyph = getFontGlyph(ch);
    var row: usize = 0;
    while (row < 8) : (row += 1) {
        const bits = glyph[row];
        var col: usize = 0;
        while (col < 8) : (col += 1) {
            if ((bits & (@as(u8, 1) << @as(u3, @intCast(7 - col)))) != 0) {
                fb.setPixel(x + @as(isize, @intCast(col)), y + @as(isize, @intCast(row)), c);
            }
        }
    }
}

// Minimalist 8x8 font table for ASCII 32..126
fn getFontGlyph(ch: u8) [8]u8 {
    return switch (ch) {
        ' ' => [8]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
        '!' => [8]u8{ 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00 },
        '"' => [8]u8{ 0x66, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00 },
        '#' => [8]u8{ 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00 },
        '$' => [8]u8{ 0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00 },
        '%' => [8]u8{ 0x00, 0x63, 0x66, 0x0C, 0x18, 0x33, 0x63, 0x00 },
        '&' => [8]u8{ 0x1C, 0x36, 0x1C, 0x38, 0x6B, 0x66, 0x3B, 0x00 },
        '\'' => [8]u8{ 0x18, 0x18, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00 },
        '(' => [8]u8{ 0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00 },
        ')' => [8]u8{ 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00 },
        '*' => [8]u8{ 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00 },
        '+' => [8]u8{ 0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00 },
        ',' => [8]u8{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x08 },
        '-' => [8]u8{ 0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00 },
        '.' => [8]u8{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00 },
        '/' => [8]u8{ 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00 },
        '0' => [8]u8{ 0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00 },
        '1' => [8]u8{ 0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00 },
        '2' => [8]u8{ 0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00 },
        '3' => [8]u8{ 0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00 },
        '4' => [8]u8{ 0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00 },
        '5' => [8]u8{ 0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00 },
        '6' => [8]u8{ 0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00 },
        '7' => [8]u8{ 0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00 },
        '8' => [8]u8{ 0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00 },
        '9' => [8]u8{ 0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00 },
        ':' => [8]u8{ 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00 },
        ';' => [8]u8{ 0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x08, 0x00 },
        '<' => [8]u8{ 0x0E, 0x1C, 0x38, 0x70, 0x38, 0x1C, 0x0E, 0x00 },
        '=' => [8]u8{ 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00 },
        '>' => [8]u8{ 0x70, 0x38, 0x1C, 0x0E, 0x1C, 0x38, 0x70, 0x00 },
        '?' => [8]u8{ 0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00 },
        '@' => [8]u8{ 0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00 },
        'A' => [8]u8{ 0x18, 0x3C, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00 },
        'B' => [8]u8{ 0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00 },
        'C' => [8]u8{ 0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00 },
        'D' => [8]u8{ 0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00 },
        'E' => [8]u8{ 0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00 },
        'F' => [8]u8{ 0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00 },
        'G' => [8]u8{ 0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00 },
        'H' => [8]u8{ 0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00 },
        'I' => [8]u8{ 0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00 },
        'J' => [8]u8{ 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00 },
        'K' => [8]u8{ 0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00 },
        'L' => [8]u8{ 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00 },
        'M' => [8]u8{ 0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00 },
        'N' => [8]u8{ 0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00 },
        'O' => [8]u8{ 0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00 },
        'P' => [8]u8{ 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00 },
        'Q' => [8]u8{ 0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00 },
        'R' => [8]u8{ 0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x63, 0x00 },
        'S' => [8]u8{ 0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00 },
        'T' => [8]u8{ 0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00 },
        'U' => [8]u8{ 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00 },
        'V' => [8]u8{ 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00 },
        'W' => [8]u8{ 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00 },
        'X' => [8]u8{ 0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00 },
        'Y' => [8]u8{ 0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00 },
        'Z' => [8]u8{ 0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00 },
        '[' => [8]u8{ 0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00 },
        '\\' => [8]u8{ 0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00 },
        ']' => [8]u8{ 0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00 },
        '^' => [8]u8{ 0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00 },
        '_' => [8]u8{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00 },
        'a' => [8]u8{ 0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3B, 0x00 },
        'b' => [8]u8{ 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00 },
        'c' => [8]u8{ 0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00 },
        'd' => [8]u8{ 0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3B, 0x00 },
        'e' => [8]u8{ 0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00 },
        'f' => [8]u8{ 0x0E, 0x18, 0x7C, 0x18, 0x18, 0x18, 0x18, 0x00 },
        'g' => [8]u8{ 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x3C },
        'h' => [8]u8{ 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00 },
        'i' => [8]u8{ 0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00 },
        'j' => [8]u8{ 0x06, 0x00, 0x0E, 0x06, 0x06, 0x06, 0x66, 0x3C },
        'k' => [8]u8{ 0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00 },
        'l' => [8]u8{ 0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00 },
        'm' => [8]u8{ 0x00, 0x00, 0x66, 0x7F, 0x7F, 0x6B, 0x63, 0x00 },
        'n' => [8]u8{ 0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00 },
        'o' => [8]u8{ 0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00 },
        'p' => [8]u8{ 0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60 },
        'q' => [8]u8{ 0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06 },
        'r' => [8]u8{ 0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00 },
        's' => [8]u8{ 0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00 },
        't' => [8]u8{ 0x18, 0x18, 0x7E, 0x18, 0x18, 0x18, 0x0E, 0x00 },
        'u' => [8]u8{ 0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3B, 0x00 },
        'v' => [8]u8{ 0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00 },
        'w' => [8]u8{ 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00 },
        'x' => [8]u8{ 0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00 },
        'y' => [8]u8{ 0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C },
        'z' => [8]u8{ 0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00 },
        '{' => [8]u8{ 0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00 },
        '|' => [8]u8{ 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00 },
        '}' => [8]u8{ 0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00 },
        '~' => [8]u8{ 0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
        else => [8]u8{ 0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF, 0x00 },
    };
}

test "runtime graphics framebuffer primitives" {
    var fb = try Framebuffer.init(std.testing.allocator, 64, 64);
    defer fb.deinit();

    fb.clear(Color.black);
    fb.setPixel(10, 10, Color.red);
    try std.testing.expectEqual(Color.red, fb.getPixel(10, 10));

    fb.drawLine(0, 0, 10, 10, Color.blue);
    try std.testing.expectEqual(Color.blue, fb.getPixel(5, 5));

    fb.drawRect(20, 20, 10, 10, Color.green, true);
    try std.testing.expectEqual(Color.green, fb.getPixel(25, 25));

    fb.drawCircle(40, 40, 5, Color.yellow, true);
    try std.testing.expectEqual(Color.yellow, fb.getPixel(40, 40));
}
