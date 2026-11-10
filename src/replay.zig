//! Aether Deterministic Replay Runtime
//! Provides seed-tagged PRNG streams and snapshot/rewind for bit-identical
//! simulation replay, debugging, and game physics verification.

const std = @import("std");
const runtime = @import("runtime.zig");
const Value = runtime.Value;

pub const Snapshot = struct {
    tick: u64,
    state: Value,
};

pub const ReplayRuntime = struct {
    allocator: std.mem.Allocator,
    seed: u64,
    tick: u64 = 0,
    prng: std.Random.DefaultPrng,
    snapshots: std.ArrayList(Snapshot),

    pub fn init(allocator: std.mem.Allocator, seed: u64) ReplayRuntime {
        return .{
            .allocator = allocator,
            .seed = seed,
            .tick = 0,
            .prng = std.Random.DefaultPrng.init(seed),
            .snapshots = .empty,
        };
    }

    pub fn deinit(self: *ReplayRuntime) void {
        self.snapshots.deinit(self.allocator);
    }

    /// Advances the simulation clock by 1 tick
    pub fn step(self: *ReplayRuntime) void {
        self.tick += 1;
    }

    /// Generates a reproducible pseudo-random number in [min, max)
    pub fn randomFloat(self: *ReplayRuntime, min: f64, max: f64) f64 {
        const u = self.prng.random().float(f64);
        return min + u * (max - min);
    }

    /// Captures a deterministic state snapshot
    pub fn captureSnapshot(self: *ReplayRuntime, state: Value) !void {
        try self.snapshots.append(self.allocator, .{
            .tick = self.tick,
            .state = state,
        });
    }

    /// Rewinds execution to the most recent snapshot at or before `target_tick`
    pub fn rewind(self: *ReplayRuntime, target_tick: u64) ?Snapshot {
        var best: ?Snapshot = null;
        for (self.snapshots.items) |s| {
            if (s.tick <= target_tick) {
                if (best == null or s.tick > best.?.tick) {
                    best = s;
                }
            }
        }
        if (best) |b| {
            self.tick = b.tick;
            return b;
        }
        return null;
    }
};

test "deterministic replay produces identical PRNG sequences and rewinds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var sim1 = ReplayRuntime.init(arena.allocator(), 1337);
    defer sim1.deinit();

    var sim2 = ReplayRuntime.init(arena.allocator(), 1337);
    defer sim2.deinit();

    // Verify identical PRNG stream
    const r1_a = sim1.randomFloat(0.0, 100.0);
    const r1_b = sim1.randomFloat(0.0, 100.0);

    const r2_a = sim2.randomFloat(0.0, 100.0);
    const r2_b = sim2.randomFloat(0.0, 100.0);

    try std.testing.expectEqual(r1_a, r2_a);
    try std.testing.expectEqual(r1_b, r2_b);

    // Test snapshot and rewind
    sim1.step(); // tick 1
    try sim1.captureSnapshot(runtime.number(10.0));
    sim1.step(); // tick 2
    try sim1.captureSnapshot(runtime.number(25.0));
    sim1.step(); // tick 3

    const rewound = sim1.rewind(1);
    try std.testing.expect(rewound != null);
    try std.testing.expectEqual(@as(u64, 1), rewound.?.tick);
    try std.testing.expectEqual(@as(f64, 10.0), rewound.?.state.number.val);
}
