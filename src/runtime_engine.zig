//! Aether Runtime Game & Simulation Engine Subsystem
//! Love2D/Processing-style game loops, 2D/3D physics collision solvers,
//! pluggable behaviors (Construct 3/GDevelop), particle emitters, and procedural audio integration.

const std = @import("std");
const graphics = @import("runtime_graphics.zig");
const Color = graphics.Color;
const Framebuffer = graphics.Framebuffer;
const platform = @import("runtime_platform.zig");
const Platform = platform.Platform;
const PlatformConfig = platform.PlatformConfig;
const Backend = platform.Backend;
const KeyCode = platform.KeyCode;
const gui = @import("runtime_gui.zig");
const GuiContext = gui.GuiContext;
const audio = @import("runtime_audio.zig");
const Synth = audio.Synth;
const PlatformAudio = audio.PlatformAudio;

// 2D & 3D Collision Primitives & Queries
pub const AABB = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    pub fn intersects(self: AABB, other: AABB) bool {
        return self.x < other.x + other.w and
            self.x + self.w > other.x and
            self.y < other.y + other.h and
            self.y + self.h > other.y;
    }
};

pub const Circle = struct {
    x: f32,
    y: f32,
    r: f32,

    pub fn intersects(self: Circle, other: Circle) bool {
        const dx = self.x - other.x;
        const dy = self.y - other.y;
        const dist_sq = dx * dx + dy * dy;
        const rad_sum = self.r + other.r;
        return dist_sq <= rad_sum * rad_sum;
    }

    pub fn intersectsAABB(self: Circle, box: AABB) bool {
        const closest_x = @max(box.x, @min(self.x, box.x + box.w));
        const closest_y = @max(box.y, @min(self.y, box.y + box.h));
        const dx = self.x - closest_x;
        const dy = self.y - closest_y;
        return (dx * dx + dy * dy) <= (self.r * self.r);
    }
};

pub fn raycastAABB(ox: f32, oy: f32, dx: f32, dy: f32, box: AABB) ?f32 {
    var tmin: f32 = -std.math.inf(f32);
    var tmax: f32 = std.math.inf(f32);

    if (@abs(dx) > 1e-6) {
        var t1 = (box.x - ox) / dx;
        var t2 = (box.x + box.w - ox) / dx;
        if (t1 > t2) std.mem.swap(f32, &t1, &t2);
        tmin = @max(tmin, t1);
        tmax = @min(tmax, t2);
    } else if (ox < box.x or ox > box.x + box.w) {
        return null;
    }

    if (@abs(dy) > 1e-6) {
        var t1 = (box.y - oy) / dy;
        var t2 = (box.y + box.h - oy) / dy;
        if (t1 > t2) std.mem.swap(f32, &t1, &t2);
        tmin = @max(tmin, t1);
        tmax = @min(tmax, t2);
    } else if (oy < box.y or oy > box.y + box.h) {
        return null;
    }

    if (tmax >= tmin and tmax >= 0.0) {
        return if (tmin >= 0.0) tmin else tmax;
    }
    return null;
}

// Pluggable Behaviors (Construct 3 / GDevelop)
pub const PlatformerState = struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    w: f32 = 16.0,
    h: f32 = 16.0,
    vx: f32 = 0.0,
    vy: f32 = 0.0,
    on_ground: bool = false,
    speed: f32 = 120.0,
    jump_force: f32 = 250.0,
    gravity: f32 = 600.0,

    pub fn update(self: *PlatformerState, left: bool, right: bool, jump: bool, colliders: []const AABB, dt: f32) void {
        self.vx = 0.0;
        if (left) self.vx -= self.speed;
        if (right) self.vx += self.speed;

        if (jump and self.on_ground) {
            self.vy = -self.jump_force;
            self.on_ground = false;
        }

        self.vy += self.gravity * dt;

        // Horizontal movement & collision
        self.x += self.vx * dt;
        var my_box = AABB{ .x = self.x, .y = self.y, .w = self.w, .h = self.h };
        for (colliders) |col| {
            if (my_box.intersects(col)) {
                if (self.vx > 0) self.x = col.x - self.w;
                if (self.vx < 0) self.x = col.x + col.w;
                my_box.x = self.x;
            }
        }

        // Vertical movement & collision
        self.y += self.vy * dt;
        self.on_ground = false;
        my_box.y = self.y;
        for (colliders) |col| {
            if (my_box.intersects(col)) {
                if (self.vy > 0) {
                    self.y = col.y - self.h;
                    self.vy = 0;
                    self.on_ground = true;
                } else if (self.vy < 0) {
                    self.y = col.y + col.h;
                    self.vy = 0;
                }
                my_box.y = self.y;
            }
        }
    }
};

pub const Particle = struct {
    x: f32 = 0,
    y: f32 = 0,
    vx: f32 = 0,
    vy: f32 = 0,
    life: f32 = 0,
    max_life: f32 = 1.0,
    color: Color = Color.white,
    active: bool = false,
};

pub const ParticleEmitter = struct {
    particles: [256]Particle = @splat(.{}),
    gravity: f32 = 150.0,
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(1234),

    pub fn burst(self: *ParticleEmitter, x: f32, y: f32, count: usize, c: Color, speed: f32) void {
        var spawned: usize = 0;
        for (&self.particles) |*p| {
            if (spawned >= count) break;
            if (!p.active) {
                const angle = self.prng.random().float(f32) * 2.0 * std.math.pi;
                const spd = speed * (0.5 + self.prng.random().float(f32) * 0.5);
                p.x = x;
                p.y = y;
                p.vx = @cos(angle) * spd;
                p.vy = @sin(angle) * spd;
                p.max_life = 0.3 + self.prng.random().float(f32) * 0.4;
                p.life = p.max_life;
                p.color = c;
                p.active = true;
                spawned += 1;
            }
        }
    }

    pub fn update(self: *ParticleEmitter, dt: f32) void {
        for (&self.particles) |*p| {
            if (!p.active) continue;
            p.life -= dt;
            if (p.life <= 0.0) {
                p.active = false;
                continue;
            }
            p.vy += self.gravity * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
        }
    }

    pub fn draw(self: *const ParticleEmitter, fb: *Framebuffer) void {
        for (&self.particles) |p| {
            if (!p.active) continue;
            const alpha_frac = p.life / p.max_life;
            var c = p.color;
            c.a = @intFromFloat(alpha_frac * 255.0);
            fb.setPixel(@intFromFloat(p.x), @intFromFloat(p.y), c);
            fb.setPixel(@intFromFloat(p.x + 1), @intFromFloat(p.y), c);
            fb.setPixel(@intFromFloat(p.x), @intFromFloat(p.y + 1), c);
            fb.setPixel(@intFromFloat(p.x + 1), @intFromFloat(p.y + 1), c);
        }
    }
};

// Unified Game & Simulation Engine
pub const Engine = struct {
    fb: Framebuffer,
    platform: Platform,
    gui: GuiContext = .{},
    emitter: ParticleEmitter = .{},
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: PlatformConfig) !Engine {
        return .{
            .fb = try Framebuffer.init(allocator, config.width, config.height),
            .platform = try Platform.init(allocator, config),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.fb.deinit();
        self.platform.deinit();
    }

    pub fn pollEvents(self: *Engine) void {
        self.platform.pollEvents();
        self.gui.update(
            @intCast(self.platform.mouse_x),
            @intCast(self.platform.mouse_y),
            self.platform.isMouseButtonDown(0),
            self.platform.isMouseButtonPressed(0),
        );
    }

    pub fn clear(self: *Engine, c: Color) void {
        self.fb.clear(c);
    }

    pub fn resize(self: *Engine, width: usize, height: usize) !void {
        try self.fb.resize(width, height);
        self.platform.config.width = width;
        self.platform.config.height = height;
    }

    pub fn present(self: *Engine) !void {
        self.emitter.update(@floatCast(self.platform.dt));
        self.emitter.draw(&self.fb);
        try self.platform.present(&self.fb);
    }

    pub fn shouldClose(self: *const Engine) bool {
        return self.platform.should_close;
    }

    pub fn dt(self: *const Engine) f64 {
        return self.platform.dt;
    }

    pub fn playLaser(self: *Engine) void {
        const samples = Synth.generateLaser(self.allocator) catch return;
        defer self.allocator.free(samples);
        PlatformAudio.playAsync(samples, 44100);
    }

    pub fn playExplosion(self: *Engine) void {
        const samples = Synth.generateExplosion(self.allocator) catch return;
        defer self.allocator.free(samples);
        PlatformAudio.playAsync(samples, 44100);
    }

    pub fn playJump(self: *Engine) void {
        const samples = Synth.generateJump(self.allocator) catch return;
        defer self.allocator.free(samples);
        PlatformAudio.playAsync(samples, 44100);
    }
};

test "engine collision and physics detection" {
    const b1 = AABB{ .x = 10, .y = 10, .w = 20, .h = 20 };
    const b2 = AABB{ .x = 25, .y = 25, .w = 20, .h = 20 };
    const b3 = AABB{ .x = 100, .y = 100, .w = 10, .h = 10 };

    try std.testing.expect(b1.intersects(b2));
    try std.testing.expect(!b1.intersects(b3));

    const c1 = Circle{ .x = 15, .y = 15, .r = 10 };
    try std.testing.expect(c1.intersectsAABB(b1));

    const ray_hit = raycastAABB(0, 20, 1, 0, b1);
    try std.testing.expect(ray_hit != null);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ray_hit.?, 1e-4);
}
