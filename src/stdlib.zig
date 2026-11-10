//! Aether Standard & Domain Libraries (`std/*`)
//! Core math, collections, simulation & game math routines.

const std = @import("std");
const runtime = @import("runtime.zig");
const Value = runtime.Value;

pub const Math = struct {
    pub fn sqrt(v: Value) Value {
        if (v == .number) return runtime.number(@sqrt(v.number.val));
        @panic("sqrt requires number");
    }

    pub fn sin(v: Value) Value {
        if (v == .number) return runtime.number(@sin(v.number.val));
        @panic("sin requires number");
    }

    pub fn cos(v: Value) Value {
        if (v == .number) return runtime.number(@cos(v.number.val));
        @panic("cos requires number");
    }

    pub fn exp(v: Value) Value {
        if (v == .number) return runtime.number(@exp(v.number.val));
        @panic("exp requires number");
    }

    pub fn log(v: Value) Value {
        if (v == .number) return runtime.number(@log(v.number.val));
        @panic("log requires number");
    }

    pub fn abs(v: Value) Value {
        if (v == .number) return runtime.number(@abs(v.number.val));
        @panic("abs requires number");
    }
};

pub const Sim = struct {
    pub const Vec3 = struct {
        x: f64,
        y: f64,
        z: f64,

        pub fn add(a: Vec3, b: Vec3) Vec3 {
            return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
        }

        pub fn dot(a: Vec3, b: Vec3) f64 {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        pub fn norm(a: Vec3) f64 {
            return @sqrt(a.dot(a));
        }

        pub fn cross(a: Vec3, b: Vec3) Vec3 {
            return .{
                .x = a.y * b.z - a.z * b.y,
                .y = a.z * b.x - a.x * b.z,
                .z = a.x * b.y - a.y * b.x,
            };
        }
    };
};

pub const Gfx = runtime.graphics;
pub const Tui = runtime.tui;
pub const Audio = runtime.audio;
pub const Platform = runtime.platform;
pub const Gui = runtime.gui;
pub const Engine = runtime.engine;

test "stdlib math and sim vector operations" {
    const n = runtime.number(16.0);
    const sq = Math.sqrt(n);
    try std.testing.expectEqual(@as(f64, 4.0), sq.number.val);

    const v1 = Sim.Vec3{ .x = 1.0, .y = 2.0, .z = 3.0 };
    const v2 = Sim.Vec3{ .x = 4.0, .y = 5.0, .z = 6.0 };
    const d = Sim.Vec3.dot(v1, v2);
    try std.testing.expectEqual(@as(f64, 32.0), d);

    const cr = Sim.Vec3.cross(v1, v2);
    try std.testing.expectEqual(@as(f64, -3.0), cr.x);
    try std.testing.expectEqual(@as(f64, 6.0), cr.y);
    try std.testing.expectEqual(@as(f64, -3.0), cr.z);
}
