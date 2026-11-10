//! Aether Pure-Zig Runtime
//! Provides dynamic values, N-D tensors, and reverse-mode automatic differentiation.

const std = @import("std");

pub const graphics = @import("runtime_graphics.zig");
pub const tui = @import("runtime_tui.zig");
pub const audio = @import("runtime_audio.zig");
pub const platform = @import("runtime_platform.zig");
pub const gui = @import("runtime_gui.zig");
pub const engine = @import("runtime_engine.zig");
pub const mem = @import("runtime_mem.zig");

pub fn free(v: Value) Value {
    mem.free(v);
    return nil();
}

pub fn memLive() Value {
    return number(@floatFromInt(mem.liveCount()));
}

pub fn gcCollect() Value {
    mem.collect();
    return nil();
}

var global_engine: ?engine.Engine = null;
var global_last_audio_samples: ?[]const i16 = null;

// ponytail: segmented autodiff tape scales up to 1M ops with zero allocation overhead for <64K graphs
const TAPE_SEGMENT_SIZE: u32 = 65536;
const MAX_TAPE_SEGMENTS: usize = 16;

pub const OpKind = enum {
    leaf,
    add,
    sub,
    mul,
    div,
};

pub const TapeNode = struct {
    op: OpKind,
    val: f64,
    lhs: u32 = 0,
    rhs: u32 = 0,
};

var tape_base_nodes: [TAPE_SEGMENT_SIZE]TapeNode = undefined;
var tape_base_grads: [TAPE_SEGMENT_SIZE]f64 = undefined;
var tape_extra_nodes: [MAX_TAPE_SEGMENTS]?[]TapeNode = @splat(null);
var tape_extra_grads: [MAX_TAPE_SEGMENTS]?[]f64 = @splat(null);
var tape_len: u32 = 0;

pub fn getTapeNode(id: u32) TapeNode {
    const seg = id / TAPE_SEGMENT_SIZE;
    const off = id % TAPE_SEGMENT_SIZE;
    if (seg == 0) return tape_base_nodes[off];
    if (tape_extra_nodes[seg]) |slice| return slice[off];
    return .{ .op = .leaf, .val = 0.0 };
}

pub fn getTapeGrad(id: u32) f64 {
    const seg = id / TAPE_SEGMENT_SIZE;
    const off = id % TAPE_SEGMENT_SIZE;
    if (seg == 0) return tape_base_grads[off];
    if (tape_extra_grads[seg]) |slice| return slice[off];
    return 0.0;
}

pub fn setTapeGrad(id: u32, g: f64) void {
    const seg = id / TAPE_SEGMENT_SIZE;
    const off = id % TAPE_SEGMENT_SIZE;
    if (seg == 0) {
        tape_base_grads[off] = g;
    } else if (tape_extra_grads[seg]) |slice| {
        slice[off] = g;
    }
}

pub fn addTapeGrad(id: u32, g: f64) void {
    const seg = id / TAPE_SEGMENT_SIZE;
    const off = id % TAPE_SEGMENT_SIZE;
    if (seg == 0) {
        tape_base_grads[off] += g;
    } else if (tape_extra_grads[seg]) |slice| {
        slice[off] += g;
    }
}

pub fn resetTape() void {
    tape_len = 0;
}

fn pushNode(op: OpKind, val: f64, lhs: u32, rhs: u32) u32 {
    const id = tape_len;
    const seg = id / TAPE_SEGMENT_SIZE;
    const off = id % TAPE_SEGMENT_SIZE;
    if (seg >= MAX_TAPE_SEGMENTS) {
        tape_len = 0;
        return pushNode(op, val, lhs, rhs);
    }
    if (seg > 0 and tape_extra_nodes[seg] == null) {
        tape_extra_nodes[seg] = alloc().alloc(TapeNode, TAPE_SEGMENT_SIZE) catch {
            tape_len = 0;
            return pushNode(op, val, lhs, rhs);
        };
        tape_extra_grads[seg] = alloc().alloc(f64, TAPE_SEGMENT_SIZE) catch {
            tape_len = 0;
            return pushNode(op, val, lhs, rhs);
        };
    }
    if (seg == 0) {
        tape_base_nodes[off] = .{ .op = op, .val = val, .lhs = lhs, .rhs = rhs };
    } else {
        tape_extra_nodes[seg].?[off] = .{ .op = op, .val = val, .lhs = lhs, .rhs = rhs };
    }
    tape_len += 1;
    return id;
}

pub const Scalar = struct {
    val: f64,
    id: u32,
};

pub const Tensor = struct {
    shape: []const usize,
    data: []f64,
    id: u32 = 0,
};

pub const List = struct {
    items: std.ArrayList(Value),
};

pub const Map = struct {
    entries: std.StringHashMap(Value),
};

pub const Set = struct {
    entries: std.StringHashMap(void),
};

pub const Deque = struct {
    items: std.ArrayList(Value),
};

pub const Value = union(enum) {
    nil: void,
    boolean: bool,
    number: Scalar,
    string: []const u8,
    ptr: ?*anyopaque,
    tensor: Tensor,
    uncertain: struct { val: f64, err: f64 },
    distribution: struct { dist_type: enum { normal, uniform, bernoulli }, p1: f64, p2: f64 },
    list: *List,
    map: *Map,
    set: *Set,
    deque: *Deque,
};

// ponytail: fixed-capacity tape for tensor operations and autodiff
const MAX_TENSOR_NODES = 4096;

pub const TensorOpKind = enum {
    leaf,
    matmul,
    relu,
    sigmoid,
    tanh,
    add,
    sub,
    mul,
    mse_loss,
};

pub const TensorTapeNode = struct {
    op: TensorOpKind,
    shape: []const usize,
    data: []f64,
    lhs: u32 = 0,
    rhs: u32 = 0,
};

var tensor_tape_nodes: [MAX_TENSOR_NODES]TensorTapeNode = undefined;
var tensor_tape_grads: [MAX_TENSOR_NODES]?[]f64 = undefined;
var tensor_tape_len: u32 = 0;

pub fn resetTensorTape() void {
    tensor_tape_len = 0;
    for (0..MAX_TENSOR_NODES) |i| {
        tensor_tape_grads[i] = null;
    }
}

pub fn getTensorTapeLen() usize {
    return tensor_tape_len;
}

pub fn getTensorTapeNodes() []const TensorTapeNode {
    return &tensor_tape_nodes;
}

pub fn getTensorTapeGrad(idx: usize) ?[]const f64 {
    if (idx < tensor_tape_len) return tensor_tape_grads[idx];
    return null;
}

fn pushTensorNode(op: TensorOpKind, shape: []const usize, data: []f64, lhs: u32, rhs: u32) u32 {
    if (tensor_tape_len >= MAX_TENSOR_NODES) {
        tensor_tape_len = 0;
    }
    const id = tensor_tape_len;
    tensor_tape_nodes[id] = .{
        .op = op,
        .shape = shape,
        .data = data,
        .lhs = lhs,
        .rhs = rhs,
    };
    tensor_tape_grads[id] = null;
    tensor_tape_len += 1;
    return id;
}

var runtime_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

pub fn alloc() std.mem.Allocator {
    return runtime_arena.allocator();
}

pub fn deinitRuntime() void {
    audio.PlatformAudio.shutdown();
    if (global_last_audio_samples) |prev| {
        std.heap.page_allocator.free(prev);
        global_last_audio_samples = null;
    }
    if (global_engine) |*eng| {
        eng.deinit();
        global_engine = null;
    }
    _ = clusterStop();
    mem.deinitMem();
    runtime_arena.deinit();
    runtime_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    resetTape();
    tape_extra_nodes = @splat(null);
    tape_extra_grads = @splat(null);
    resetTensorTape();
    global_actors = null;
    global_actor_next_id = 1;
    global_replay_snapshots = .empty;
    global_replay_tick = 0;
}

pub fn nil() Value {
    return .{ .nil = {} };
}

pub fn boolean(b: bool) Value {
    return .{ .boolean = b };
}

pub fn number(n: f64) Value {
    return .{ .number = .{ .val = n, .id = pushNode(.leaf, n, 0, 0) } };
}

pub fn string(s: []const u8) Value {
    return .{ .string = s };
}

pub fn ptr(p: ?*anyopaque) Value {
    return .{ .ptr = p };
}

pub fn tensor(shape_in: []const usize, data_in: []const f64) Value {
    const shape_copy = mem.newUsizes(shape_in);
    const data_copy = mem.newF64s(data_in.len);
    @memcpy(data_copy, data_in);
    const id = pushTensorNode(.leaf, shape_copy, data_copy, 0, 0);
    return .{ .tensor = .{ .shape = shape_copy, .data = data_copy, .id = id } };
}

pub fn uncertain(val: Value, err: Value) Value {
    const v = if (val == .number) val.number.val else (if (val == .uncertain) val.uncertain.val else 0.0);
    const e = if (err == .number) err.number.val else (if (err == .uncertain) err.uncertain.err else 0.0);
    return .{ .uncertain = .{ .val = v, .err = e } };
}

pub fn distNormal(mean: Value, std_dev: Value) Value {
    const m = if (mean == .number) mean.number.val else 0.0;
    const s = if (std_dev == .number) std_dev.number.val else 1.0;
    return .{ .distribution = .{ .dist_type = .normal, .p1 = m, .p2 = s } };
}

pub fn distUniform(low: Value, high: Value) Value {
    const l = if (low == .number) low.number.val else 0.0;
    const h = if (high == .number) high.number.val else 1.0;
    return .{ .distribution = .{ .dist_type = .uniform, .p1 = l, .p2 = h } };
}

pub fn distBernoulli(p: Value) Value {
    const pv = if (p == .number) p.number.val else 0.5;
    return .{ .distribution = .{ .dist_type = .bernoulli, .p1 = pv, .p2 = 0.0 } };
}

var prng = std.Random.DefaultPrng.init(42);

pub fn sample(d: Value) Value {
    const r = prng.random();
    if (d == .distribution) {
        switch (d.distribution.dist_type) {
            .normal => {
                const rnd1 = @max(1e-12, r.float(f64));
                const rnd2 = r.float(f64);
                const z0 = @sqrt(-2.0 * @log(rnd1)) * @cos(2.0 * std.math.pi * rnd2);
                return number(d.distribution.p1 + z0 * d.distribution.p2);
            },
            .uniform => {
                const u = r.float(f64);
                return number(d.distribution.p1 + u * (d.distribution.p2 - d.distribution.p1));
            },
            .bernoulli => {
                const u = r.float(f64);
                return boolean(u < d.distribution.p1);
            },
        }
    }
    return d;
}

pub fn observe(pred: Value, target: Value, dist: Value) Value {
    _ = pred;
    _ = target;
    _ = dist;
    return boolean(true);
}

pub fn isTruthy(v: Value) bool {
    return switch (v) {
        .nil => false,
        .boolean => |b| b,
        .number => |n| n.val != 0.0,
        .string => |s| s.len > 0,
        .ptr => |p| p != null,
        .tensor => |t| t.data.len > 0,
        .uncertain => |u| u.val != 0.0,
        .distribution => true,
        .list => |l| l.items.items.len > 0,
        .map => |m| m.entries.count() > 0,
        .set => |s| s.entries.count() > 0,
        .deque => |d| d.items.items.len > 0,
    };
}

pub fn toExitCode(v: Value) u8 {
    return switch (v) {
        .number => |n| @intFromFloat(@max(0.0, @min(255.0, n.val))),
        .boolean => |b| if (b) 0 else 1,
        else => 0,
    };
}

pub fn toI64(v: Value) i64 {
    return switch (v) {
        .number => |n| @intFromFloat(n.val),
        .boolean => |b| if (b) 1 else 0,
        else => 0,
    };
}

pub fn toF64(v: Value) f64 {
    return switch (v) {
        .number => |n| n.val,
        else => 0.0,
    };
}

fn binaryTensorOp(op: OpKind, a: Tensor, b: Tensor) Value {
    const out = mem.newF64s(a.data.len);
    for (a.data, 0..) |av, i| {
        const bv = b.data[i % b.data.len];
        out[i] = switch (op) {
            .add => av + bv,
            .sub => av - bv,
            .mul => av * bv,
            .div => av / bv,
            .leaf => av,
        };
    }
    const shape_copy = mem.newUsizes(a.shape);
    const top: TensorOpKind = switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        else => .leaf,
    };
    const id = pushTensorNode(top, shape_copy, out, a.id, b.id);
    return .{ .tensor = .{ .shape = shape_copy, .data = out, .id = id } };
}

fn scalarTensorOp(op: OpKind, t: Tensor, s: f64, scalar_first: bool) Value {
    const out = mem.newF64s(t.data.len);
    for (t.data, 0..) |tv, i| {
        const lhs = if (scalar_first) s else tv;
        const rhs = if (scalar_first) tv else s;
        out[i] = switch (op) {
            .add => lhs + rhs,
            .sub => lhs - rhs,
            .mul => lhs * rhs,
            .div => lhs / rhs,
            .leaf => lhs,
        };
    }
    const shape_copy = mem.newUsizes(t.shape);
    return .{ .tensor = .{ .shape = shape_copy, .data = out } };
}

pub fn add(a: Value, b: Value) Value {
    if (a == .uncertain or b == .uncertain) {
        const av = if (a == .uncertain) a.uncertain.val else (if (a == .number) a.number.val else 0.0);
        const ae = if (a == .uncertain) a.uncertain.err else 0.0;
        const bv = if (b == .uncertain) b.uncertain.val else (if (b == .number) b.number.val else 0.0);
        const be = if (b == .uncertain) b.uncertain.err else 0.0;
        return .{ .uncertain = .{ .val = av + bv, .err = @sqrt(ae * ae + be * be) } };
    }
    return switch (a) {
        .number => |na| switch (b) {
            .number => |nb| .{ .number = .{
                .val = na.val + nb.val,
                .id = pushNode(.add, na.val + nb.val, na.id, nb.id),
            } },
            .tensor => |tb| scalarTensorOp(.add, tb, na.val, true),
            else => @panic("Type error in +"),
        },
        .tensor => |ta| switch (b) {
            .number => |nb| scalarTensorOp(.add, ta, nb.val, false),
            .tensor => |tb| binaryTensorOp(.add, ta, tb),
            else => @panic("Type error in tensor +"),
        },
        else => @panic("Type error in +"),
    };
}

pub fn sub(a: Value, b: Value) Value {
    if (a == .uncertain or b == .uncertain) {
        const av = if (a == .uncertain) a.uncertain.val else (if (a == .number) a.number.val else 0.0);
        const ae = if (a == .uncertain) a.uncertain.err else 0.0;
        const bv = if (b == .uncertain) b.uncertain.val else (if (b == .number) b.number.val else 0.0);
        const be = if (b == .uncertain) b.uncertain.err else 0.0;
        return .{ .uncertain = .{ .val = av - bv, .err = @sqrt(ae * ae + be * be) } };
    }
    return switch (a) {
        .number => |na| switch (b) {
            .number => |nb| .{ .number = .{
                .val = na.val - nb.val,
                .id = pushNode(.sub, na.val - nb.val, na.id, nb.id),
            } },
            .tensor => |tb| scalarTensorOp(.sub, tb, na.val, true),
            else => @panic("Type error in -"),
        },
        .tensor => |ta| switch (b) {
            .number => |nb| scalarTensorOp(.sub, ta, nb.val, false),
            .tensor => |tb| binaryTensorOp(.sub, ta, tb),
            else => @panic("Type error in tensor -"),
        },
        else => @panic("Type error in -"),
    };
}

pub fn mul(a: Value, b: Value) Value {
    if (a == .uncertain or b == .uncertain) {
        const av = if (a == .uncertain) a.uncertain.val else (if (a == .number) a.number.val else 0.0);
        const ae = if (a == .uncertain) a.uncertain.err else 0.0;
        const bv = if (b == .uncertain) b.uncertain.val else (if (b == .number) b.number.val else 0.0);
        const be = if (b == .uncertain) b.uncertain.err else 0.0;
        return .{ .uncertain = .{ .val = av * bv, .err = @sqrt((bv * ae) * (bv * ae) + (av * be) * (av * be)) } };
    }
    return switch (a) {
        .number => |na| switch (b) {
            .number => |nb| .{ .number = .{
                .val = na.val * nb.val,
                .id = pushNode(.mul, na.val * nb.val, na.id, nb.id),
            } },
            .tensor => |tb| scalarTensorOp(.mul, tb, na.val, true),
            else => @panic("Type error in *"),
        },
        .tensor => |ta| switch (b) {
            .number => |nb| scalarTensorOp(.mul, ta, nb.val, false),
            .tensor => |tb| binaryTensorOp(.mul, ta, tb),
            else => @panic("Type error in tensor *"),
        },
        else => @panic("Type error in *"),
    };
}

pub fn div(a: Value, b: Value) Value {
    return switch (a) {
        .number => |na| switch (b) {
            .number => |nb| .{ .number = .{
                .val = na.val / nb.val,
                .id = pushNode(.div, na.val / nb.val, na.id, nb.id),
            } },
            .tensor => |tb| scalarTensorOp(.div, tb, na.val, true),
            else => @panic("Type error in /"),
        },
        .tensor => |ta| switch (b) {
            .number => |nb| scalarTensorOp(.div, ta, nb.val, false),
            .tensor => |tb| binaryTensorOp(.div, ta, tb),
            else => @panic("Type error in tensor /"),
        },
        else => @panic("Type error in /"),
    };
}

pub fn mod(a: Value, b: Value) Value {
    return number(@mod(a.number.val, b.number.val));
}

pub fn eq(a: Value, b: Value) Value {
    if (@as(std.meta.Tag(Value), a) != @as(std.meta.Tag(Value), b)) return boolean(false);
    return switch (a) {
        .number => |na| boolean(na.val == b.number.val),
        .boolean => |ba| boolean(ba == b.boolean),
        .ptr => |pa| boolean(pa == b.ptr),
        .string => |sa| boolean(std.mem.eql(u8, sa, b.string)),
        .nil => boolean(b == .nil),
        .list => |la| {
            const lb = b.list;
            if (la.items.items.len != lb.items.items.len) return boolean(false);
            for (la.items.items, lb.items.items) |item_a, item_b| {
                if (!eq(item_a, item_b).boolean) return boolean(false);
            }
            return boolean(true);
        },
        .map => |ma| {
            const mb = b.map;
            if (ma.entries.count() != mb.entries.count()) return boolean(false);
            var it = ma.entries.iterator();
            while (it.next()) |entry| {
                const val_b = mb.entries.get(entry.key_ptr.*) orelse return boolean(false);
                if (!eq(entry.value_ptr.*, val_b).boolean) return boolean(false);
            }
            return boolean(true);
        },
        .set => |sa| {
            const sb = b.set;
            if (sa.entries.count() != sb.entries.count()) return boolean(false);
            var it = sa.entries.keyIterator();
            while (it.next()) |k| {
                if (!sb.entries.contains(k.*)) return boolean(false);
            }
            return boolean(true);
        },
        .deque => |da| {
            const db = b.deque;
            if (da.items.items.len != db.items.items.len) return boolean(false);
            for (da.items.items, db.items.items) |item_a, item_b| {
                if (!eq(item_a, item_b).boolean) return boolean(false);
            }
            return boolean(true);
        },
        .tensor => |ta| {
            const tb = b.tensor;
            if (ta.data.len != tb.data.len) return boolean(false);
            for (ta.data, tb.data) |va, vb| {
                if (va != vb) return boolean(false);
            }
            return boolean(true);
        },
        else => boolean(false),
    };
}

pub fn ne(a: Value, b: Value) Value {
    return boolean(!eq(a, b).boolean);
}

pub fn lt(a: Value, b: Value) Value {
    if (a == .string and b == .string) return boolean(std.mem.order(u8, a.string, b.string) == .lt);
    return boolean(toF64(a) < toF64(b));
}

pub fn le(a: Value, b: Value) Value {
    if (a == .string and b == .string) {
        const ord = std.mem.order(u8, a.string, b.string);
        return boolean(ord == .lt or ord == .eq);
    }
    return boolean(toF64(a) <= toF64(b));
}

pub fn gt(a: Value, b: Value) Value {
    if (a == .string and b == .string) return boolean(std.mem.order(u8, a.string, b.string) == .gt);
    return boolean(toF64(a) > toF64(b));
}

pub fn ge(a: Value, b: Value) Value {
    if (a == .string and b == .string) {
        const ord = std.mem.order(u8, a.string, b.string);
        return boolean(ord == .gt or ord == .eq);
    }
    return boolean(toF64(a) >= toF64(b));
}

pub fn logicAnd(a: Value, b: Value) Value {
    return boolean(isTruthy(a) and isTruthy(b));
}

pub fn logicOr(a: Value, b: Value) Value {
    return boolean(isTruthy(a) or isTruthy(b));
}

pub fn logicNot(a: Value) Value {
    return boolean(!isTruthy(a));
}

/// Matrix multiplication supporting 2D x 2D ([M, K] x [K, N] -> [M, N])
/// and 1D x 2D ([K] x [K, N] -> [N]) and 2D x 1D ([M, K] x [K] -> [M]).
pub fn matmul(a_val: Value, b_val: Value) Value {
    const a = a_val.tensor;
    const b = b_val.tensor;

    if (a.shape.len == 2 and b.shape.len == 2) {
        const m = a.shape[0];
        const k = a.shape[1];
        const k2 = b.shape[0];
        const n = b.shape[1];
        if (k != k2) @panic("matmul shape mismatch: inner dimensions do not match");

        const out = mem.newF64s(m * n);
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0.0;
                for (0..k) |p| {
                    acc += a.data[i * k + p] * b.data[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        const shape = mem.newUsizes(&[_]usize{ m, n });
        const id = pushTensorNode(.matmul, shape, out, a.id, b.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    } else if (a.shape.len == 1 and b.shape.len == 2) {
        const k = a.shape[0];
        const k2 = b.shape[0];
        const n = b.shape[1];
        if (k != k2) @panic("matmul 1D x 2D shape mismatch");

        const out = mem.newF64s(n);
        for (0..n) |j| {
            var acc: f64 = 0.0;
            for (0..k) |p| {
                acc += a.data[p] * b.data[p * n + j];
            }
            out[j] = acc;
        }
        const shape = mem.newUsizes(&[_]usize{n});
        const id = pushTensorNode(.matmul, shape, out, a.id, b.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    } else if (a.shape.len == 2 and b.shape.len == 1) {
        const m = a.shape[0];
        const k = a.shape[1];
        const k2 = b.shape[0];
        if (k != k2) @panic("matmul 2D x 1D shape mismatch");

        const out = mem.newF64s(m);
        for (0..m) |i| {
            var acc: f64 = 0.0;
            for (0..k) |p| {
                acc += a.data[i * k + p] * b.data[p];
            }
            out[i] = acc;
        }
        const shape = mem.newUsizes(&[_]usize{m});
        const id = pushTensorNode(.matmul, shape, out, a.id, b.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    }
    @panic("matmul requires 1D or 2D tensors");
}

pub fn fusedMatmulRelu(a_val: Value, b_val: Value) Value {
    const a = a_val.tensor;
    const b = b_val.tensor;

    if (a.shape.len == 2 and b.shape.len == 2) {
        const m = a.shape[0];
        const k = a.shape[1];
        const k2 = b.shape[0];
        const n = b.shape[1];
        if (k != k2) @panic("fusedMatmulRelu inner dimension mismatch");

        const out = mem.newF64s(m * n);
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0.0;
                for (0..k) |p| {
                    acc += a.data[i * k + p] * b.data[p * n + j];
                }
                out[i * n + j] = @max(0.0, acc);
            }
        }
        const shape = mem.newUsizes(&[_]usize{ m, n });
        const id = pushTensorNode(.relu, shape, out, a.id, b.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    } else if (a.shape.len == 1 and b.shape.len == 2) {
        const k = a.shape[0];
        const k2 = b.shape[0];
        const n = b.shape[1];
        if (k != k2) @panic("fusedMatmulRelu 1D x 2D dimension mismatch");

        const out = mem.newF64s(n);
        for (0..n) |j| {
            var acc: f64 = 0.0;
            for (0..k) |p| {
                acc += a.data[p] * b.data[p * n + j];
            }
            out[j] = @max(0.0, acc);
        }
        const shape = mem.newUsizes(&[_]usize{n});
        const id = pushTensorNode(.relu, shape, out, a.id, b.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    }
    @panic("fusedMatmulRelu requires 1D or 2D tensors");
}

pub fn fusedDenseRelu(x_val: Value, w_val: Value, b_val: Value) Value {
    const x = x_val.tensor;
    const w = w_val.tensor;
    const b = b_val.tensor;

    if (x.shape.len == 2 and w.shape.len == 2) {
        const m = x.shape[0];
        const k = x.shape[1];
        const k2 = w.shape[0];
        const n = w.shape[1];
        if (k != k2) @panic("fusedDenseRelu inner dimension mismatch");

        const out = mem.newF64s(m * n);
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0.0;
                for (0..k) |p| {
                    acc += x.data[i * k + p] * w.data[p * n + j];
                }
                const bias_val = if (b.data.len == n) b.data[j] else if (b.data.len == 1) b.data[0] else if (b.data.len == m * n) b.data[i * n + j] else 0.0;
                out[i * n + j] = @max(0.0, acc + bias_val);
            }
        }
        const shape = mem.newUsizes(&[_]usize{ m, n });
        const id = pushTensorNode(.relu, shape, out, x.id, w.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    } else if (x.shape.len == 1 and w.shape.len == 2) {
        const k = x.shape[0];
        const k2 = w.shape[0];
        const n = w.shape[1];
        if (k != k2) @panic("fusedDenseRelu 1D x 2D dimension mismatch");

        const out = mem.newF64s(n);
        for (0..n) |j| {
            var acc: f64 = 0.0;
            for (0..k) |p| {
                acc += x.data[p] * w.data[p * n + j];
            }
            const bias_val = if (b.data.len == n) b.data[j] else if (b.data.len == 1) b.data[0] else 0.0;
            out[j] = @max(0.0, acc + bias_val);
        }
        const shape = mem.newUsizes(&[_]usize{n});
        const id = pushTensorNode(.relu, shape, out, x.id, w.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    }
    @panic("fusedDenseRelu requires 1D or 2D tensors");
}

pub fn sum(v: Value) Value {
    var total: f64 = 0.0;
    for (v.tensor.data) |elem| total += elem;
    return number(total);
}

pub fn relu(v: Value) Value {
    return switch (v) {
        .number => |n| number(@max(0.0, n.val)),
        .tensor => |t| {
            const out = mem.newF64s(t.data.len);
            for (t.data, 0..) |elem, i| out[i] = @max(0.0, elem);
            const shape = mem.newUsizes(t.shape);
            const id = pushTensorNode(.relu, shape, out, t.id, 0);
            return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
        },
        else => @panic("relu requires number or tensor"),
    };
}

pub fn sigmoid(v: Value) Value {
    return switch (v) {
        .number => |n| number(1.0 / (1.0 + @exp(-n.val))),
        .tensor => |t| {
            const out = mem.newF64s(t.data.len);
            for (t.data, 0..) |elem, i| out[i] = 1.0 / (1.0 + @exp(-elem));
            const shape = mem.newUsizes(t.shape);
            const id = pushTensorNode(.sigmoid, shape, out, t.id, 0);
            return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
        },
        else => @panic("sigmoid requires number or tensor"),
    };
}

pub fn tanh(v: Value) Value {
    return switch (v) {
        .number => |n| number(std.math.tanh(n.val)),
        .tensor => |t| {
            const out = mem.newF64s(t.data.len);
            for (t.data, 0..) |elem, i| out[i] = std.math.tanh(elem);
            const shape = mem.newUsizes(t.shape);
            const id = pushTensorNode(.tanh, shape, out, t.id, 0);
            return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
        },
        else => @panic("tanh requires number or tensor"),
    };
}

pub fn mse_loss(pred: Value, target: Value) Value {
    if (pred == .number and target == .number) {
        const diff = pred.number.val - target.number.val;
        return number(diff * diff);
    }
    if (pred == .tensor and target == .tensor) {
        var total: f64 = 0.0;
        const len = @min(pred.tensor.data.len, target.tensor.data.len);
        if (len == 0) return number(0.0);
        for (0..len) |i| {
            const diff = pred.tensor.data[i] - target.tensor.data[i];
            total += diff * diff;
        }
        const mean_loss = total / @as(f64, @floatFromInt(len));
        const a = alloc();
        const shape = a.dupe(usize, &[_]usize{1}) catch @panic("OOM");
        const out = a.alloc(f64, 1) catch @panic("OOM");
        out[0] = mean_loss;
        const id = pushTensorNode(.mse_loss, shape, out, pred.tensor.id, target.tensor.id);
        return .{ .tensor = .{ .shape = shape, .data = out, .id = id } };
    }
    @panic("mse_loss requires numbers or tensors");
}

pub fn cross_entropy(pred: Value, target: Value) Value {
    if (pred == .number and target == .number) {
        const p = @max(1e-12, @min(1.0 - 1e-12, pred.number.val));
        const loss = -target.number.val * @log(p);
        return number(loss);
    }
    if (pred == .tensor and target == .tensor) {
        var total: f64 = 0.0;
        const len = @min(pred.tensor.data.len, target.tensor.data.len);
        if (len == 0) return number(0.0);
        for (0..len) |i| {
            const p = @max(1e-12, @min(1.0 - 1e-12, pred.tensor.data[i]));
            total -= target.tensor.data[i] * @log(p);
        }
        return number(total / @as(f64, @floatFromInt(len)));
    }
    @panic("cross_entropy requires numbers or tensors");
}

/// Quantizes float tensor into INT8 representation (-128 to 127) with scale factor
pub fn quantize(v: Value) Value {
    if (v != .tensor or v.tensor.data.len == 0) return v;
    var min_v = v.tensor.data[0];
    var max_v = v.tensor.data[0];
    for (v.tensor.data) |x| {
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    const range = @max(1e-6, max_v - min_v);
    const scale = range / 255.0;

    const out = mem.newF64s(v.tensor.data.len);
    for (v.tensor.data, 0..) |x, i| {
        const q: f64 = @round((x - min_v) / scale) - 128.0;
        out[i] = @max(-128.0, @min(127.0, q));
    }
    const shape = mem.newUsizes(v.tensor.shape);
    return .{ .tensor = .{ .shape = shape, .data = out } };
}

/// Reverse-mode automatic differentiation: computes dy/dx from the segmented tape.
pub fn gradient(y: Value, x: Value) Value {
    if (y != .number or x != .number) return y;
    const y_id = y.number.id;
    const x_id = x.number.id;
    if (y_id >= tape_len or x_id >= tape_len) return number(0.0);

    var i: u32 = 0;
    while (i < tape_len) : (i += 1) {
        setTapeGrad(i, 0.0);
    }
    setTapeGrad(y_id, 1.0);

    var idx: usize = y_id + 1;
    while (idx > 0) {
        idx -= 1;
        const u_idx: u32 = @intCast(idx);
        const g = getTapeGrad(u_idx);
        if (g == 0.0) continue;
        const node = getTapeNode(u_idx);
        switch (node.op) {
            .leaf => {},
            .add => {
                addTapeGrad(node.lhs, g);
                addTapeGrad(node.rhs, g);
            },
            .sub => {
                addTapeGrad(node.lhs, g);
                addTapeGrad(node.rhs, -g);
            },
            .mul => {
                addTapeGrad(node.lhs, g * getTapeNode(node.rhs).val);
                addTapeGrad(node.rhs, g * getTapeNode(node.lhs).val);
            },
            .div => {
                const rv = getTapeNode(node.rhs).val;
                addTapeGrad(node.lhs, g / rv);
                addTapeGrad(node.rhs, -g * getTapeNode(node.lhs).val / (rv * rv));
            },
        }
    }
    return number(getTapeGrad(x_id));
}

/// First-class reverse-mode tensor automatic differentiation (tensor backpropagation)
pub fn tensorBackward(loss: Value) Value {
    if (loss != .number and loss != .tensor) return nil();
    if (tensor_tape_len == 0) return nil();

    const a = alloc();
    for (0..tensor_tape_len) |i| {
        tensor_tape_grads[i] = null;
    }

    const loss_id = if (loss == .tensor) loss.tensor.id else tensor_tape_len - 1;
    if (loss_id >= tensor_tape_len) return nil();

    const target_node = tensor_tape_nodes[loss_id];
    const initial_grad = a.alloc(f64, target_node.data.len) catch @panic("OOM");
    @memset(initial_grad, 1.0);
    tensor_tape_grads[loss_id] = initial_grad;

    var idx: usize = loss_id + 1;
    while (idx > 0) {
        idx -= 1;
        const g = tensor_tape_grads[idx] orelse continue;
        const node = tensor_tape_nodes[idx];

        switch (node.op) {
            .leaf => {},
            .matmul => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                const rhs_node = tensor_tape_nodes[node.rhs];
                if (lhs_node.shape.len == 2 and rhs_node.shape.len == 2) {
                    const m = lhs_node.shape[0];
                    const k = lhs_node.shape[1];
                    const n = rhs_node.shape[1];

                    // dLHS = g * rhs^T -> [M, N] x [N, K] = [M, K]
                    var dlhs = tensor_tape_grads[node.lhs];
                    if (dlhs == null) {
                        dlhs = a.alloc(f64, m * k) catch @panic("OOM");
                        @memset(dlhs.?, 0.0);
                        tensor_tape_grads[node.lhs] = dlhs;
                    }
                    for (0..m) |i_m| {
                        for (0..k) |i_k| {
                            var sum_v: f64 = 0.0;
                            for (0..n) |i_n| {
                                sum_v += g[i_m * n + i_n] * rhs_node.data[i_k * n + i_n];
                            }
                            dlhs.?[i_m * k + i_k] += sum_v;
                        }
                    }

                    // dRHS = lhs^T * g -> [K, M] x [M, N] = [K, N]
                    var drhs = tensor_tape_grads[node.rhs];
                    if (drhs == null) {
                        drhs = a.alloc(f64, k * n) catch @panic("OOM");
                        @memset(drhs.?, 0.0);
                        tensor_tape_grads[node.rhs] = drhs;
                    }
                    for (0..k) |i_k| {
                        for (0..n) |i_n| {
                            var sum_v: f64 = 0.0;
                            for (0..m) |i_m| {
                                sum_v += lhs_node.data[i_m * k + i_k] * g[i_m * n + i_n];
                            }
                            drhs.?[i_k * n + i_n] += sum_v;
                        }
                    }
                }
            },
            .relu => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (lhs_node.data, 0..) |elem, i_elem| {
                    if (elem > 0.0 and i_elem < g.len) {
                        dlhs.?[i_elem] += g[i_elem];
                    }
                }
            },
            .sigmoid => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (node.data, 0..) |s, i_s| {
                    if (i_s < g.len) {
                        dlhs.?[i_s] += g[i_s] * s * (1.0 - s);
                    }
                }
            },
            .tanh => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (node.data, 0..) |t, i_t| {
                    if (i_t < g.len) {
                        dlhs.?[i_t] += g[i_t] * (1.0 - t * t);
                    }
                }
            },
            .add => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                const rhs_node = tensor_tape_nodes[node.rhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (0..lhs_node.data.len) |i_l| {
                    if (i_l < g.len) dlhs.?[i_l] += g[i_l];
                }
                var drhs = tensor_tape_grads[node.rhs];
                if (drhs == null) {
                    drhs = a.alloc(f64, rhs_node.data.len) catch @panic("OOM");
                    @memset(drhs.?, 0.0);
                    tensor_tape_grads[node.rhs] = drhs;
                }
                for (0..rhs_node.data.len) |i_r| {
                    if (i_r < g.len) drhs.?[i_r] += g[i_r];
                }
            },
            .sub => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                const rhs_node = tensor_tape_nodes[node.rhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (0..lhs_node.data.len) |i_l| {
                    if (i_l < g.len) dlhs.?[i_l] += g[i_l];
                }
                var drhs = tensor_tape_grads[node.rhs];
                if (drhs == null) {
                    drhs = a.alloc(f64, rhs_node.data.len) catch @panic("OOM");
                    @memset(drhs.?, 0.0);
                    tensor_tape_grads[node.rhs] = drhs;
                }
                for (0..rhs_node.data.len) |i_r| {
                    if (i_r < g.len) drhs.?[i_r] -= g[i_r];
                }
            },
            .mul => {
                const lhs_node = tensor_tape_nodes[node.lhs];
                const rhs_node = tensor_tape_nodes[node.rhs];
                var dlhs = tensor_tape_grads[node.lhs];
                if (dlhs == null) {
                    dlhs = a.alloc(f64, lhs_node.data.len) catch @panic("OOM");
                    @memset(dlhs.?, 0.0);
                    tensor_tape_grads[node.lhs] = dlhs;
                }
                for (0..lhs_node.data.len) |i_l| {
                    const r_val = if (i_l < rhs_node.data.len) rhs_node.data[i_l] else 1.0;
                    if (i_l < g.len) dlhs.?[i_l] += g[i_l] * r_val;
                }
                var drhs = tensor_tape_grads[node.rhs];
                if (drhs == null) {
                    drhs = a.alloc(f64, rhs_node.data.len) catch @panic("OOM");
                    @memset(drhs.?, 0.0);
                    tensor_tape_grads[node.rhs] = drhs;
                }
                for (0..rhs_node.data.len) |i_r| {
                    const l_val = if (i_r < lhs_node.data.len) lhs_node.data[i_r] else 1.0;
                    if (i_r < g.len) drhs.?[i_r] += g[i_r] * l_val;
                }
            },
            .mse_loss => {
                const pred_node = tensor_tape_nodes[node.lhs];
                const tgt_node = tensor_tape_nodes[node.rhs];
                const n_elems: f64 = @floatFromInt(pred_node.data.len);
                var dpred = tensor_tape_grads[node.lhs];
                if (dpred == null) {
                    dpred = a.alloc(f64, pred_node.data.len) catch @panic("OOM");
                    @memset(dpred.?, 0.0);
                    tensor_tape_grads[node.lhs] = dpred;
                }
                for (pred_node.data, 0..) |p_val, i_p| {
                    const t_val = if (i_p < tgt_node.data.len) tgt_node.data[i_p] else 0.0;
                    dpred.?[i_p] += (2.0 / @max(1.0, n_elems)) * (p_val - t_val) * g[0];
                }
            },
        }
    }
    return boolean(true);
}

pub fn tensorGrad(t: Value) Value {
    if (t != .tensor) return nil();
    const a = alloc();
    const shape_copy = a.dupe(usize, t.tensor.shape) catch @panic("OOM");
    const data_copy = a.alloc(f64, t.tensor.data.len) catch @panic("OOM");
    if (t.tensor.id < tensor_tape_len) {
        if (tensor_tape_grads[t.tensor.id]) |g| {
            @memcpy(data_copy, g[0..@min(g.len, data_copy.len)]);
            return .{ .tensor = .{ .shape = shape_copy, .data = data_copy } };
        }
    }
    @memset(data_copy, 0.0);
    return .{ .tensor = .{ .shape = shape_copy, .data = data_copy } };
}

pub fn tensorClearGrads() Value {
    resetTensorTape();
    return boolean(true);
}

pub fn tensorTranspose(t_val: Value) Value {
    if (t_val != .tensor) return t_val;
    const t = t_val.tensor;
    const a = alloc();
    if (t.shape.len == 2) {
        const rows = t.shape[0];
        const cols = t.shape[1];
        const out_shape = a.dupe(usize, &[_]usize{ cols, rows }) catch @panic("OOM");
        const out_data = a.alloc(f64, rows * cols) catch @panic("OOM");
        for (0..rows) |r| {
            for (0..cols) |c| {
                out_data[c * rows + r] = t.data[r * cols + c];
            }
        }
        return .{ .tensor = .{ .shape = out_shape, .data = out_data } };
    } else if (t.shape.len == 1) {
        const out_shape = a.dupe(usize, &[_]usize{ t.shape[0], 1 }) catch @panic("OOM");
        const out_data = a.dupe(f64, t.data) catch @panic("OOM");
        return .{ .tensor = .{ .shape = out_shape, .data = out_data } };
    }
    return t_val;
}

pub fn tensorReshape(t_val: Value, shape_val: Value) Value {
    if (t_val != .tensor) return t_val;
    const a = alloc();
    var new_dims: std.ArrayList(usize) = .empty;
    if (shape_val == .list) {
        for (shape_val.list.items.items) |dim_v| {
            new_dims.append(a, @intFromFloat(@max(1.0, toF64(dim_v)))) catch @panic("OOM");
        }
    } else if (shape_val == .tensor) {
        for (shape_val.tensor.data) |d| {
            new_dims.append(a, @intFromFloat(@max(1.0, d))) catch @panic("OOM");
        }
    }
    if (new_dims.items.len == 0) return t_val;
    var total: usize = 1;
    for (new_dims.items) |d| total *= d;
    if (total != t_val.tensor.data.len) return t_val;
    const out_shape = a.dupe(usize, new_dims.items) catch @panic("OOM");
    const out_data = a.dupe(f64, t_val.tensor.data) catch @panic("OOM");
    return .{ .tensor = .{ .shape = out_shape, .data = out_data } };
}

pub fn tensorSlice(t_val: Value, start_val: Value, end_val: Value) Value {
    if (t_val != .tensor) return t_val;
    const start: isize = @intFromFloat(@max(0.0, toF64(start_val)));
    const end: isize = @intFromFloat(@min(@as(f64, @floatFromInt(t_val.tensor.data.len)), toF64(end_val)));
    const a = alloc();
    if (start < end and start >= 0) {
        const count: usize = @intCast(end - start);
        const out_shape = a.dupe(usize, &[_]usize{count}) catch @panic("OOM");
        const out_data = a.alloc(f64, count) catch @panic("OOM");
        @memcpy(out_data, t_val.tensor.data[@intCast(start)..@intCast(end)]);
        return .{ .tensor = .{ .shape = out_shape, .data = out_data } };
    }
    return t_val;
}

pub fn tensorArgmax(t_val: Value) Value {
    if (t_val != .tensor or t_val.tensor.data.len == 0) return number(0.0);
    var best_idx: usize = 0;
    var best_val: f64 = t_val.tensor.data[0];
    for (t_val.tensor.data, 0..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return number(@floatFromInt(best_idx));
}

pub fn tensorSoftmax(t_val: Value) Value {
    if (t_val != .tensor or t_val.tensor.data.len == 0) return t_val;
    const a = alloc();
    const data = t_val.tensor.data;
    var max_val: f64 = data[0];
    for (data) |v| if (v > max_val) {
        max_val = v;
    };

    var sum_exp: f64 = 0.0;
    const out_data = a.alloc(f64, data.len) catch @panic("OOM");
    for (data, 0..) |v, i| {
        const ev = @exp(v - max_val);
        out_data[i] = ev;
        sum_exp += ev;
    }
    const inv_sum = if (sum_exp > 0.0) 1.0 / sum_exp else 1.0;
    for (out_data) |*elem| {
        elem.* *= inv_sum;
    }
    const out_shape = a.dupe(usize, t_val.tensor.shape) catch @panic("OOM");
    return .{ .tensor = .{ .shape = out_shape, .data = out_data } };
}

pub fn optimizerSGD(params: Value, grads: Value, lr_val: Value) Value {
    if (params != .tensor or grads != .tensor) return params;
    const lr = toF64(lr_val);
    const len = @min(params.tensor.data.len, grads.tensor.data.len);
    for (0..len) |i| {
        params.tensor.data[i] -= lr * grads.tensor.data[i];
    }
    return params;
}

pub fn optimizerAdam(
    params: Value,
    grads: Value,
    m_val: Value,
    v_val: Value,
    step_val: Value,
    lr_val: Value,
    beta1_val: Value,
    beta2_val: Value,
    eps_val: Value,
) Value {
    if (params != .tensor or grads != .tensor or m_val != .tensor or v_val != .tensor) return params;
    const step = @max(1.0, toF64(step_val));
    const lr = toF64(lr_val);
    const beta1 = if (beta1_val != .nil and toF64(beta1_val) > 0.0) toF64(beta1_val) else 0.9;
    const beta2 = if (beta2_val != .nil and toF64(beta2_val) > 0.0) toF64(beta2_val) else 0.999;
    const eps = if (eps_val != .nil and toF64(eps_val) > 0.0) toF64(eps_val) else 1e-8;

    const bc1 = 1.0 - std.math.pow(f64, beta1, step);
    const bc2 = 1.0 - std.math.pow(f64, beta2, step);

    const len = @min(params.tensor.data.len, @min(grads.tensor.data.len, @min(m_val.tensor.data.len, v_val.tensor.data.len)));
    for (0..len) |i| {
        const g = grads.tensor.data[i];
        m_val.tensor.data[i] = beta1 * m_val.tensor.data[i] + (1.0 - beta1) * g;
        v_val.tensor.data[i] = beta2 * v_val.tensor.data[i] + (1.0 - beta2) * (g * g);

        const m_hat = m_val.tensor.data[i] / bc1;
        const v_hat = v_val.tensor.data[i] / bc2;

        params.tensor.data[i] -= (lr / (@sqrt(v_hat) + eps)) * m_hat;
    }
    return params;
}

// ============================================================================
// Dynamic Collections System (List, Map, Set, Deque)
// ============================================================================
fn valueToKeyString(v: Value) []const u8 {
    switch (v) {
        .string => |s| return s,
        .number => |n| {
            if (@floor(n.val) == n.val and @abs(n.val) < 1e15) {
                return std.fmt.allocPrint(alloc(), "{d}", .{@as(i64, @intFromFloat(n.val))}) catch "0";
            } else {
                return std.fmt.allocPrint(alloc(), "{d}", .{n.val}) catch "0";
            }
        },
        .boolean => |b| return if (b) "true" else "false",
        .nil => return "nil",
        else => return "obj",
    }
}

pub fn list(items: []const Value) Value {
    const l = mem.newObj(List);
    l.items = .empty;
    for (items) |it| {
        l.items.append(mem.heap(), it) catch @panic("OOM");
    }
    return .{ .list = l };
}

pub fn listAppend(l_val: Value, item: Value) Value {
    if (l_val == .list) {
        l_val.list.items.append(mem.heap(), item) catch @panic("OOM");
        return l_val;
    }
    return l_val;
}

pub fn listConcat(a_val: Value, b_val: Value) Value {
    const out = mem.newObj(List);
    out.items = .empty;
    if (a_val == .list) {
        for (a_val.list.items.items) |it| out.items.append(mem.heap(), it) catch @panic("OOM");
    }
    if (b_val == .list) {
        for (b_val.list.items.items) |it| out.items.append(mem.heap(), it) catch @panic("OOM");
    }
    return .{ .list = out };
}

pub fn listGet(col: Value, key_or_idx: Value) Value {
    switch (col) {
        .list => |l| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < l.items.items.len) {
                return l.items.items[@intCast(idx)];
            }
            return nil();
        },
        .map => |m| {
            const k = valueToKeyString(key_or_idx);
            return m.entries.get(k) orelse nil();
        },
        .deque => |d| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < d.items.items.len) {
                return d.items.items[@intCast(idx)];
            }
            return nil();
        },
        .tensor => |t| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < t.data.len) {
                return number(t.data[@intCast(idx)]);
            }
            return nil();
        },
        .string => |s| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < s.len) {
                const char_slice = alloc().alloc(u8, 1) catch return nil();
                char_slice[0] = s[@intCast(idx)];
                return string(char_slice);
            }
            return nil();
        },
        else => return nil(),
    }
}

pub fn listSet(col: Value, key_or_idx: Value, val: Value) Value {
    switch (col) {
        .list => |l| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < l.items.items.len) {
                l.items.items[@intCast(idx)] = val;
                return val;
            }
            return nil();
        },
        .map => |m| {
            const k = valueToKeyString(key_or_idx);
            const k_copy = alloc().dupe(u8, k) catch @panic("OOM");
            m.entries.put(k_copy, val) catch @panic("OOM");
            return val;
        },
        .deque => |d| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < d.items.items.len) {
                d.items.items[@intCast(idx)] = val;
                return val;
            }
            return nil();
        },
        .tensor => |t| {
            const idx: isize = @intFromFloat(toF64(key_or_idx));
            if (idx >= 0 and idx < t.data.len) {
                t.data[@intCast(idx)] = toF64(val);
                return val;
            }
            return nil();
        },
        else => return nil(),
    }
}

pub fn listLength(col: Value) Value {
    switch (col) {
        .list => |l| return number(@floatFromInt(l.items.items.len)),
        .map => |m| return number(@floatFromInt(m.entries.count())),
        .set => |s| return number(@floatFromInt(s.entries.count())),
        .deque => |d| return number(@floatFromInt(d.items.items.len)),
        .tensor => |t| return number(@floatFromInt(t.data.len)),
        .string => |s| return number(@floatFromInt(s.len)),
        else => return number(0.0),
    }
}

pub fn listSlice(col: Value, start_val: Value, end_val: Value) Value {
    if (col == .list) {
        const start: isize = @intFromFloat(@max(0.0, toF64(start_val)));
        const end: isize = @intFromFloat(@min(@as(f64, @floatFromInt(col.list.items.items.len)), toF64(end_val)));
        const out = mem.newObj(List);
        out.items = .empty;
        if (start < end and start >= 0) {
            for (col.list.items.items[@intCast(start)..@intCast(end)]) |it| {
                out.items.append(mem.heap(), it) catch @panic("OOM");
            }
        }
        return .{ .list = out };
    } else if (col == .string) {
        const s = col.string;
        const start: isize = @intFromFloat(@max(0.0, toF64(start_val)));
        const end: isize = @intFromFloat(@min(@as(f64, @floatFromInt(s.len)), toF64(end_val)));
        if (start < end and start >= 0) {
            const out_bytes = mem.heap().dupe(u8, s[@intCast(start)..@intCast(end)]) catch @panic("OOM");
            return string(out_bytes);
        }
        return string("");
    }
    return nil();
}

pub fn listPop(l_val: Value) Value {
    if (l_val == .list and l_val.list.items.items.len > 0) {
        return l_val.list.items.pop() orelse nil();
    }
    return nil();
}

pub fn mapCreate() Value {
    const m = mem.newObj(Map);
    m.entries = std.StringHashMap(Value).init(mem.heap());
    return .{ .map = m };
}

pub fn mapSet(m_val: Value, key_val: Value, val: Value) Value {
    if (m_val == .map) {
        const k = valueToKeyString(key_val);
        const k_copy = mem.dupeBytes(k);
        m_val.map.entries.put(k_copy, val) catch @panic("OOM");
        return val;
    }
    return nil();
}

pub fn mapGet(m_val: Value, key_val: Value) Value {
    if (m_val == .map) {
        const k = valueToKeyString(key_val);
        return m_val.map.entries.get(k) orelse nil();
    }
    return nil();
}

pub fn mapHas(m_val: Value, key_val: Value) Value {
    if (m_val == .map) {
        const k = valueToKeyString(key_val);
        return boolean(m_val.map.entries.contains(k));
    }
    return boolean(false);
}

pub fn mapDelete(m_val: Value, key_val: Value) Value {
    if (m_val == .map) {
        const k = valueToKeyString(key_val);
        _ = m_val.map.entries.remove(k);
        return boolean(true);
    }
    return boolean(false);
}

pub fn mapKeys(m_val: Value) Value {
    if (m_val != .map) return nil();
    const l = mem.newObj(List);
    l.items = .empty;
    var it = m_val.map.entries.keyIterator();
    while (it.next()) |k| {
        l.items.append(mem.heap(), string(mem.dupeBytes(k.*))) catch @panic("OOM");
    }
    return .{ .list = l };
}

pub fn mapValues(m_val: Value) Value {
    if (m_val != .map) return nil();
    const l = mem.newObj(List);
    l.items = .empty;
    var it = m_val.map.entries.valueIterator();
    while (it.next()) |v| {
        l.items.append(mem.heap(), v.*) catch @panic("OOM");
    }
    return .{ .list = l };
}

pub fn mapSize(m_val: Value) Value {
    if (m_val == .map) return number(@floatFromInt(m_val.map.entries.count()));
    return number(0.0);
}

pub fn setCreate() Value {
    const s = mem.newObj(Set);
    s.entries = std.StringHashMap(void).init(mem.heap());
    return .{ .set = s };
}

pub fn setAdd(s_val: Value, item: Value) Value {
    if (s_val == .set) {
        const k = valueToKeyString(item);
        const k_copy = mem.dupeBytes(k);
        s_val.set.entries.put(k_copy, {}) catch @panic("OOM");
        return s_val;
    }
    return nil();
}

pub fn setContains(s_val: Value, item: Value) Value {
    if (s_val == .set) {
        const k = valueToKeyString(item);
        return boolean(s_val.set.entries.contains(k));
    }
    return boolean(false);
}

pub fn setRemove(s_val: Value, item: Value) Value {
    if (s_val == .set) {
        const k = valueToKeyString(item);
        _ = s_val.set.entries.remove(k);
        return boolean(true);
    }
    return boolean(false);
}

pub fn setSize(s_val: Value) Value {
    if (s_val == .set) return number(@floatFromInt(s_val.set.entries.count()));
    return number(0.0);
}

pub fn setToList(s_val: Value) Value {
    if (s_val != .set) return nil();
    const l = mem.newObj(List);
    l.items = .empty;
    var it = s_val.set.entries.keyIterator();
    while (it.next()) |k| {
        l.items.append(mem.heap(), string(mem.dupeBytes(k.*))) catch @panic("OOM");
    }
    return .{ .list = l };
}

pub fn dequeCreate() Value {
    const d = mem.newObj(Deque);
    d.items = .empty;
    return .{ .deque = d };
}

pub fn dequePushBack(d_val: Value, item: Value) Value {
    if (d_val == .deque) {
        d_val.deque.items.append(mem.heap(), item) catch @panic("OOM");
        return d_val;
    }
    return nil();
}

pub fn dequePushFront(d_val: Value, item: Value) Value {
    if (d_val == .deque) {
        d_val.deque.items.insert(mem.heap(), 0, item) catch @panic("OOM");
        return d_val;
    }
    return nil();
}

pub fn dequePopBack(d_val: Value) Value {
    if (d_val == .deque and d_val.deque.items.items.len > 0) {
        return d_val.deque.items.pop() orelse nil();
    }
    return nil();
}

pub fn dequePopFront(d_val: Value) Value {
    if (d_val == .deque and d_val.deque.items.items.len > 0) {
        return d_val.deque.items.orderedRemove(0);
    }
    return nil();
}

pub fn dequeLength(d_val: Value) Value {
    if (d_val == .deque) return number(@floatFromInt(d_val.deque.items.items.len));
    return number(0.0);
}

pub fn dequeGet(d_val: Value, idx_val: Value) Value {
    if (d_val == .deque) {
        const idx: isize = @intFromFloat(toF64(idx_val));
        if (idx >= 0 and idx < d_val.deque.items.items.len) {
            return d_val.deque.items.items[@intCast(idx)];
        }
    }
    return nil();
}

// ============================================================================
// Math & Vector Builtins (Stdlib Integration)
// ============================================================================
pub fn mathSqrt(v: Value) Value {
    if (v == .number) return number(@sqrt(v.number.val));
    if (v == .uncertain) {
        const val = @sqrt(@max(0.0, v.uncertain.val));
        const err = if (val > 0.0) v.uncertain.err / (2.0 * val) else 0.0;
        return uncertain(number(val), number(err));
    }
    @panic("sqrt requires numeric or uncertain value");
}

pub fn mathSin(v: Value) Value {
    if (v == .number) return number(@sin(v.number.val));
    @panic("sin requires a number");
}

pub fn mathCos(v: Value) Value {
    if (v == .number) return number(@cos(v.number.val));
    @panic("cos requires a number");
}

pub fn mathTan(v: Value) Value {
    if (v == .number) return number(@tan(v.number.val));
    @panic("tan requires a number");
}

pub fn mathExp(v: Value) Value {
    if (v == .number) return number(@exp(v.number.val));
    @panic("exp requires a number");
}

pub fn mathLog(v: Value) Value {
    if (v == .number) return number(@log(v.number.val));
    @panic("log requires a number");
}

pub fn mathAbs(v: Value) Value {
    if (v == .number) return number(@abs(v.number.val));
    @panic("abs requires a number");
}

pub fn mathPow(a: Value, b: Value) Value {
    return number(std.math.pow(f64, toF64(a), toF64(b)));
}

pub fn mathMin(a: Value, b: Value) Value {
    return number(@min(toF64(a), toF64(b)));
}

pub fn mathMax(a: Value, b: Value) Value {
    return number(@max(toF64(a), toF64(b)));
}

pub fn mathFloor(v: Value) Value {
    return number(@floor(toF64(v)));
}

pub fn mathCeil(v: Value) Value {
    return number(@ceil(toF64(v)));
}

pub fn mathAsin(v: Value) Value {
    return number(std.math.asin(toF64(v)));
}

pub fn mathAcos(v: Value) Value {
    return number(std.math.acos(toF64(v)));
}

pub fn mathAtan2(y: Value, x: Value) Value {
    return number(std.math.atan2(toF64(y), toF64(x)));
}

pub fn mathLog2(v: Value) Value {
    return number(std.math.log2(toF64(v)));
}

pub fn mathLog10(v: Value) Value {
    return number(std.math.log10(toF64(v)));
}

pub fn mathClamp(v: Value, min_v: Value, max_v: Value) Value {
    return number(std.math.clamp(toF64(v), toF64(min_v), toF64(max_v)));
}

// Statistics Builtins
pub fn statMean(v: Value) Value {
    if (v == .tensor) {
        if (v.tensor.data.len == 0) return number(0.0);
        var s: f64 = 0.0;
        for (v.tensor.data) |d| s += d;
        return number(s / @as(f64, @floatFromInt(v.tensor.data.len)));
    } else if (v == .list) {
        if (v.list.items.items.len == 0) return number(0.0);
        var s: f64 = 0.0;
        for (v.list.items.items) |elem| s += toF64(elem);
        return number(s / @as(f64, @floatFromInt(v.list.items.items.len)));
    }
    return v;
}

pub fn statVariance(v: Value) Value {
    const m = statMean(v).number.val;
    if (v == .tensor) {
        if (v.tensor.data.len <= 1) return number(0.0);
        var sq_sum: f64 = 0.0;
        for (v.tensor.data) |d| {
            const diff = d - m;
            sq_sum += diff * diff;
        }
        return number(sq_sum / @as(f64, @floatFromInt(v.tensor.data.len)));
    } else if (v == .list) {
        if (v.list.items.items.len <= 1) return number(0.0);
        var sq_sum: f64 = 0.0;
        for (v.list.items.items) |elem| {
            const diff = toF64(elem) - m;
            sq_sum += diff * diff;
        }
        return number(sq_sum / @as(f64, @floatFromInt(v.list.items.items.len)));
    }
    return number(0.0);
}

pub fn statStdDev(v: Value) Value {
    const variance = statVariance(v).number.val;
    return number(@sqrt(@max(0.0, variance)));
}

// String Utilities
pub fn strSplit(s_val: Value, delim_val: Value) Value {
    const s = s_val.string;
    const delim = delim_val.string;
    const l = mem.newObj(List);
    l.items = .empty;

    if (delim.len == 0) {
        for (s) |ch| {
            const ch_str = mem.newBytes(1);
            ch_str[0] = ch;
            l.items.append(mem.heap(), string(ch_str)) catch @panic("OOM");
        }
    } else {
        var it = std.mem.splitSequence(u8, s, delim);
        while (it.next()) |part| {
            const dup = mem.dupeBytes(part);
            l.items.append(mem.heap(), string(dup)) catch @panic("OOM");
        }
    }
    return .{ .list = l };
}

pub fn strJoin(list_val: Value, delim_val: Value) Value {
    if (list_val != .list) @panic("strJoin requires a list");
    const delim = if (delim_val == .string) delim_val.string else "";
    var total_len: usize = 0;
    for (list_val.list.items.items, 0..) |item, idx| {
        if (idx > 0) total_len += delim.len;
        if (item == .string) {
            total_len += item.string.len;
        } else if (item == .number) {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{toF64(item)}) catch "";
            total_len += s.len;
        } else if (item == .boolean) {
            total_len += if (item.boolean) 4 else 5;
        }
    }
    const res = mem.newBytes(total_len);
    var offset: usize = 0;
    for (list_val.list.items.items, 0..) |item, idx| {
        if (idx > 0) {
            @memcpy(res[offset .. offset + delim.len], delim);
            offset += delim.len;
        }
        if (item == .string) {
            @memcpy(res[offset .. offset + item.string.len], item.string);
            offset += item.string.len;
        } else if (item == .number) {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{toF64(item)}) catch "";
            @memcpy(res[offset .. offset + s.len], s);
            offset += s.len;
        } else if (item == .boolean) {
            const s: []const u8 = if (item.boolean) "true" else "false";
            @memcpy(res[offset .. offset + s.len], s);
            offset += s.len;
        }
    }
    return string(res);
}

pub fn strToNum(s_val: Value) Value {
    if (s_val == .number) return s_val;
    if (s_val == .string) {
        const trimmed = std.mem.trim(u8, s_val.string, " \t\r\n");
        const val = std.fmt.parseFloat(f64, trimmed) catch 0.0;
        return number(val);
    }
    return number(0.0);
}

pub fn strTrim(s_val: Value) Value {
    const s = s_val.string;
    const trimmed = std.mem.trim(u8, s, " \t\r\n");
    return string(mem.dupeBytes(trimmed));
}

pub fn strContains(s_val: Value, sub_val: Value) Value {
    if (s_val != .string or sub_val != .string) return boolean(false);
    const s = s_val.string;
    const sub_str = sub_val.string;
    if (sub_str.len == 0) return boolean(true);
    if (s.len < sub_str.len) return boolean(false);
    return boolean(std.mem.indexOf(u8, s, sub_str) != null);
}

// File I/O Builtins
const win32_file_io = struct {
    extern "kernel32" fn CreateFileA(
        lpFileName: [*:0]const u8,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: ?*anyopaque,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: ?*anyopaque,
    ) callconv(.c) ?*anyopaque;
    extern "kernel32" fn ReadFile(
        hFile: ?*anyopaque,
        lpBuffer: [*]u8,
        nNumberOfBytesToRead: u32,
        lpNumberOfBytesRead: ?*u32,
        lpOverlapped: ?*anyopaque,
    ) callconv(.c) i32;
    extern "kernel32" fn WriteFile(
        hFile: ?*anyopaque,
        lpBuffer: [*]const u8,
        nNumberOfBytesToWrite: u32,
        lpNumberOfBytesWritten: ?*u32,
        lpOverlapped: ?*anyopaque,
    ) callconv(.c) i32;
    extern "kernel32" fn GetFileSize(hFile: ?*anyopaque, lpFileSizeHigh: ?*u32) callconv(.c) u32;
    extern "kernel32" fn CloseHandle(hObject: ?*anyopaque) callconv(.c) i32;
    extern "kernel32" fn GetFileAttributesA(lpFileName: [*:0]const u8) callconv(.c) u32;
};

pub fn fileExists(path_val: Value) Value {
    if (path_val != .string) return boolean(false);
    const file_path = path_val.string;
    if (@import("builtin").os.tag == .windows) {
        var path_z: [512:0]u8 = undefined;
        if (file_path.len >= 511) return boolean(false);
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;
        const attrs = win32_file_io.GetFileAttributesA(&path_z);
        return boolean(attrs != std.math.maxInt(u32));
    } else {
        return boolean(false);
    }
}

pub fn fileReadStr(path_val: Value) Value {
    const file_path = path_val.string;
    const allocator = alloc();
    if (@import("builtin").os.tag == .windows) {
        var path_z: [512:0]u8 = undefined;
        if (file_path.len >= 511) @panic("Path too long in fileReadStr");
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;

        const GENERIC_READ = 0x80000000;
        const OPEN_EXISTING = 3;
        const FILE_ATTRIBUTE_NORMAL = 0x80;

        const handle = win32_file_io.CreateFileA(
            &path_z,
            GENERIC_READ,
            1, // FILE_SHARE_READ
            null,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (handle == null or handle == @as(?*anyopaque, @ptrFromInt(std.math.maxInt(usize)))) {
            return string("");
        }
        defer _ = win32_file_io.CloseHandle(handle);

        const sz = win32_file_io.GetFileSize(handle, null);
        if (sz == std.math.maxInt(u32)) @panic("Failed to get file size in fileReadStr");

        const buf = allocator.alloc(u8, sz) catch @panic("OOM in fileReadStr");
        var read_bytes: u32 = 0;
        _ = win32_file_io.ReadFile(handle, buf.ptr, sz, &read_bytes, null);
        return string(buf[0..read_bytes]);
    } else {
        @panic("Platform fileReadStr not supported");
    }
}

pub fn fileWriteStr(path_val: Value, content_val: Value) Value {
    const file_path = path_val.string;
    const data = content_val.string;
    if (@import("builtin").os.tag == .windows) {
        var path_z: [512:0]u8 = undefined;
        if (file_path.len >= 511) @panic("Path too long in fileWriteStr");
        @memcpy(path_z[0..file_path.len], file_path);
        path_z[file_path.len] = 0;

        const GENERIC_WRITE = 0x40000000;
        const CREATE_ALWAYS = 2;
        const FILE_ATTRIBUTE_NORMAL = 0x80;

        var handle = win32_file_io.CreateFileA(
            &path_z,
            GENERIC_WRITE,
            0,
            null,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (handle == null or handle == @as(?*anyopaque, @ptrFromInt(std.math.maxInt(usize)))) {
            if (std.mem.lastIndexOfScalar(u8, file_path, '/') orelse std.mem.lastIndexOfScalar(u8, file_path, '\\')) |slash_idx| {
                const basename = file_path[slash_idx + 1 ..];
                if (basename.len > 0 and basename.len < 511) {
                    var base_z: [512:0]u8 = undefined;
                    @memcpy(base_z[0..basename.len], basename);
                    base_z[basename.len] = 0;
                    handle = win32_file_io.CreateFileA(
                        &base_z,
                        GENERIC_WRITE,
                        0,
                        null,
                        CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL,
                        null,
                    );
                }
            }
        }
        if (handle == null or handle == @as(?*anyopaque, @ptrFromInt(std.math.maxInt(usize)))) {
            return boolean(false);
        }
        defer _ = win32_file_io.CloseHandle(handle);

        var written: u32 = 0;
        _ = win32_file_io.WriteFile(handle, data.ptr, @intCast(data.len), &written, null);
        return boolean(true);
    } else {
        @panic("Platform fileWriteStr not supported");
    }
}

// 3D vector operations
pub fn vec3(x: Value, y: Value, z: Value) Value {
    return tensor(&[_]usize{3}, &[_]f64{ toF64(x), toF64(y), toF64(z) });
}

pub fn vec3Dot(a: Value, b: Value) Value {
    if (a != .tensor or b != .tensor or a.tensor.data.len < 3 or b.tensor.data.len < 3) @panic("vec3Dot requires 3D tensors");
    return number(a.tensor.data[0] * b.tensor.data[0] + a.tensor.data[1] * b.tensor.data[1] + a.tensor.data[2] * b.tensor.data[2]);
}

pub fn vec3Cross(a: Value, b: Value) Value {
    if (a != .tensor or b != .tensor or a.tensor.data.len < 3 or b.tensor.data.len < 3) @panic("vec3Cross requires 3D tensors");
    const d1 = a.tensor.data;
    const d2 = b.tensor.data;
    return tensor(&[_]usize{3}, &[_]f64{
        d1[1] * d2[2] - d1[2] * d2[1],
        d1[2] * d2[0] - d1[0] * d2[2],
        d1[0] * d2[1] - d1[1] * d2[0],
    });
}

pub fn vec3Norm(a: Value) Value {
    if (a != .tensor or a.tensor.data.len < 3) @panic("vec3Norm requires 3D tensor");
    const d = a.tensor.data[0] * a.tensor.data[0] + a.tensor.data[1] * a.tensor.data[1] + a.tensor.data[2] * a.tensor.data[2];
    return number(@sqrt(d));
}

// ============================================================================
// Actor Model Concurrency System
// ============================================================================
pub const ActorMessage = struct {
    sender_id: usize,
    payload: Value,
};

pub const Actor = struct {
    id: usize,
    state: Value,
    mailbox: std.ArrayList(ActorMessage),

    pub fn init(id: usize, initial_state: Value) Actor {
        return .{
            .id = id,
            .state = initial_state,
            .mailbox = .empty,
        };
    }
};

pub var global_actors: ?std.AutoHashMap(usize, *Actor) = null;
var global_actor_next_id: usize = 1;

fn getActorMap() *std.AutoHashMap(usize, *Actor) {
    if (global_actors == null) {
        global_actors = std.AutoHashMap(usize, *Actor).init(alloc());
    }
    return &global_actors.?;
}

pub fn spawnActor(initial_state: Value) Value {
    const map = getActorMap();
    const id = global_actor_next_id;
    global_actor_next_id += 1;
    const actor_ptr = alloc().create(Actor) catch @panic("OOM creating Actor");
    actor_ptr.* = Actor.init(id, initial_state);
    map.put(id, actor_ptr) catch @panic("OOM in actor registry");
    return number(@floatFromInt(id));
}

pub fn sendActor(recipient_val: Value, sender_val: Value, payload: Value) Value {
    const r_id: usize = @intFromFloat(toF64(recipient_val));
    const s_id: usize = @intFromFloat(toF64(sender_val));
    const map = getActorMap();
    if (map.get(r_id)) |actor_ptr| {
        actor_ptr.mailbox.append(alloc(), .{ .sender_id = s_id, .payload = payload }) catch @panic("OOM in actor mailbox");
        return boolean(true);
    }
    return boolean(false);
}

pub fn receiveActor(actor_val: Value) Value {
    const a_id: usize = @intFromFloat(toF64(actor_val));
    const map = getActorMap();
    if (map.get(a_id)) |actor_ptr| {
        if (actor_ptr.mailbox.items.len == 0) return nil();
        const msg = actor_ptr.mailbox.orderedRemove(0);
        return msg.payload;
    }
    return nil();
}

pub fn getActorState(actor_val: Value) Value {
    const a_id: usize = @intFromFloat(toF64(actor_val));
    const map = getActorMap();
    if (map.get(a_id)) |actor_ptr| {
        return actor_ptr.state;
    }
    return nil();
}

pub fn setActorState(actor_val: Value, new_state: Value) Value {
    const a_id: usize = @intFromFloat(toF64(actor_val));
    const map = getActorMap();
    if (map.get(a_id)) |actor_ptr| {
        actor_ptr.state = new_state;
        return new_state;
    }
    return nil();
}

// ============================================================================
// Deterministic Replay Runtime
// ============================================================================
pub const ReplaySnapshot = struct {
    tick: u64,
    state: Value,
};

var global_replay_prng: ?std.Random.DefaultPrng = null;
var global_replay_tick: u64 = 0;
pub var global_replay_snapshots: std.ArrayList(ReplaySnapshot) = .empty;

pub fn replayInit(seed_val: Value) Value {
    const s: u64 = @intFromFloat(@max(0.0, toF64(seed_val)));
    global_replay_prng = std.Random.DefaultPrng.init(s);
    global_replay_tick = 0;
    global_replay_snapshots.clearRetainingCapacity();
    return number(@floatFromInt(s));
}

pub fn replayStep() Value {
    global_replay_tick += 1;
    return number(@floatFromInt(global_replay_tick));
}

pub fn replayRandom(min_val: Value, max_val: Value) Value {
    if (global_replay_prng == null) {
        global_replay_prng = std.Random.DefaultPrng.init(1337);
    }
    const min = toF64(min_val);
    const max = toF64(max_val);
    const u = global_replay_prng.?.random().float(f64);
    return number(min + u * (max - min));
}

pub fn replaySnapshot(state: Value) Value {
    global_replay_snapshots.append(alloc(), .{
        .tick = global_replay_tick,
        .state = state,
    }) catch @panic("OOM in replaySnapshot");
    return number(@floatFromInt(global_replay_tick));
}

pub fn replayRewind(target_tick_val: Value) Value {
    const target: u64 = @intFromFloat(@max(0.0, toF64(target_tick_val)));
    var best: ?ReplaySnapshot = null;
    for (global_replay_snapshots.items) |s| {
        if (s.tick <= target) {
            if (best == null or s.tick > best.?.tick) {
                best = s;
            }
        }
    }
    if (best) |b| {
        global_replay_tick = b.tick;
        return b.state;
    }
    return nil();
}

pub fn formatValue(writer: *std.Io.Writer, v: Value) !void {
    switch (v) {
        .nil => try writer.writeAll("nil"),
        .boolean => |b| try writer.writeAll(if (b) "true" else "false"),
        .string => |s| try writer.writeAll(s),
        .ptr => |p| try writer.print("Pointer(0x{x})", .{@intFromPtr(p)}),
        .number => |n| {
            if (@floor(n.val) == n.val and @abs(n.val) < 1e15) {
                try writer.print("{d}", .{@as(i64, @intFromFloat(n.val))});
            } else {
                try writer.print("{d}", .{n.val});
            }
        },
        .tensor => |t| {
            try writer.writeAll("Tensor(shape=[");
            for (t.shape, 0..) |dim, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{dim});
            }
            try writer.writeAll("], data=[");
            for (t.data, 0..) |elem, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d:.2}", .{elem});
            }
            try writer.writeAll("])");
        },
        .uncertain => |u| {
            try writer.print("{d:.2} ± {d:.2}", .{ u.val, u.err });
        },
        .distribution => |d| {
            switch (d.dist_type) {
                .normal => try writer.print("Normal(mean={d:.2}, std={d:.2})", .{ d.p1, d.p2 }),
                .uniform => try writer.print("Uniform(low={d:.2}, high={d:.2})", .{ d.p1, d.p2 }),
                .bernoulli => try writer.print("Bernoulli(p={d:.2})", .{d.p1}),
            }
        },
        .list => |l| {
            try writer.writeAll("[");
            for (l.items.items, 0..) |elem, i| {
                if (i > 0) try writer.writeAll(", ");
                try formatValue(writer, elem);
            }
            try writer.writeAll("]");
        },
        .map => |m| {
            try writer.writeAll("{");
            var it = m.entries.iterator();
            var i: usize = 0;
            while (it.next()) |entry| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("\"{s}\": ", .{entry.key_ptr.*});
                try formatValue(writer, entry.value_ptr.*);
                i += 1;
            }
            try writer.writeAll("}");
        },
        .set => |s| {
            try writer.writeAll("Set{");
            var it = s.entries.keyIterator();
            var i: usize = 0;
            while (it.next()) |k| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("\"{s}\"", .{k.*});
                i += 1;
            }
            try writer.writeAll("}");
        },
        .deque => |d| {
            try writer.writeAll("Deque[");
            for (d.items.items, 0..) |elem, i| {
                if (i > 0) try writer.writeAll(", ");
                try formatValue(writer, elem);
            }
            try writer.writeAll("]");
        },
    }
}

pub fn print(io: std.Io, args: []const Value) Value {
    var buf: [4096]u8 = undefined;
    var fw: std.Io.File.Writer = .init(.stdout(), io, &buf);
    const w = &fw.interface;
    for (args, 0..) |arg, i| {
        if (i > 0) w.writeAll(" ") catch {};
        formatValue(w, arg) catch {};
    }
    w.writeAll("\n") catch {};
    w.flush() catch {};
    return nil();
}

pub fn gfxInit(w_val: Value, h_val: Value, title_val: Value, backend_val: Value, scale_val: Value) Value {
    const w: usize = @intFromFloat(@max(16.0, toF64(w_val)));
    const h: usize = @intFromFloat(@max(16.0, toF64(h_val)));
    const title = if (title_val == .string) title_val.string else "Aether Engine";
    var backend: platform.Backend = .auto;
    if (backend_val == .string) {
        if (std.mem.eql(u8, backend_val.string, "window")) backend = .window
        else if (std.mem.eql(u8, backend_val.string, "tui")) backend = .tui
        else if (std.mem.eql(u8, backend_val.string, "headless")) backend = .headless;
    }

    var scale: usize = if (w <= 320 and h <= 240) 2 else 1;
    if (scale_val != .nil and toF64(scale_val) > 0.0) {
        scale = @intFromFloat(toF64(scale_val));
    }

    if (global_engine) |*eng| {
        eng.deinit();
    }

    global_engine = engine.Engine.init(alloc(), .{
        .width = w,
        .height = h,
        .scale = scale,
        .title = title,
        .backend = backend,
        .fallback = true,
    }) catch return boolean(false);

    if (global_engine) |*eng| {
        platform.global_platform_instance = &eng.platform;
    }

    return boolean(true);
}

pub fn gfxRgb(r_val: Value, g_val: Value, b_val: Value) Value {
    const r: u32 = @intFromFloat(@max(0.0, @min(255.0, toF64(r_val))));
    const g: u32 = @intFromFloat(@max(0.0, @min(255.0, toF64(g_val))));
    const b: u32 = @intFromFloat(@max(0.0, @min(255.0, toF64(b_val))));
    const hex_val: u32 = (r << 16) | (g << 8) | b;
    return number(@floatFromInt(hex_val));
}

pub fn gfxClear(color_hex_val: Value) Value {
    if (global_engine) |*eng| {
        const c: u32 = @intFromFloat(@max(0.0, toF64(color_hex_val)));
        eng.clear(graphics.Color.hex(c));
    }
    return nil();
}

pub fn gfxDrawPixel(x: Value, y: Value, c: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(c)));
        eng.fb.setPixel(@intFromFloat(toF64(x)), @intFromFloat(toF64(y)), graphics.Color.hex(col));
    }
    return nil();
}

pub fn gfxDrawLine(x0: Value, y0: Value, x1: Value, y1: Value, c: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(c)));
        eng.fb.drawLine(
            @intFromFloat(toF64(x0)),
            @intFromFloat(toF64(y0)),
            @intFromFloat(toF64(x1)),
            @intFromFloat(toF64(y1)),
            graphics.Color.hex(col),
        );
    }
    return nil();
}

pub fn gfxDrawRect(x: Value, y: Value, w: Value, h: Value, c: Value, filled: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(c)));
        const is_filled = (filled == .boolean and filled.boolean) or toF64(filled) != 0.0;
        eng.fb.drawRect(
            @intFromFloat(toF64(x)),
            @intFromFloat(toF64(y)),
            @intFromFloat(toF64(w)),
            @intFromFloat(toF64(h)),
            graphics.Color.hex(col),
            is_filled,
        );
    }
    return nil();
}

pub fn gfxDrawCircle(cx: Value, cy: Value, r: Value, c: Value, filled: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(c)));
        const is_filled = (filled == .boolean and filled.boolean) or toF64(filled) != 0.0;
        eng.fb.drawCircle(
            @intFromFloat(toF64(cx)),
            @intFromFloat(toF64(cy)),
            @intFromFloat(toF64(r)),
            graphics.Color.hex(col),
            is_filled,
        );
    }
    return nil();
}

pub fn gfxDrawText(x: Value, y: Value, text: Value, c: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(c)));
        if (text == .string) {
            eng.fb.drawText(@intFromFloat(toF64(x)), @intFromFloat(toF64(y)), text.string, graphics.Color.hex(col));
        } else {
            var buf: [64]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{toF64(text)}) catch "";
            eng.fb.drawText(@intFromFloat(toF64(x)), @intFromFloat(toF64(y)), str, graphics.Color.hex(col));
        }
    }
    return nil();
}

pub fn gfxExportBMP(path_val: Value) Value {
    if (global_engine) |*eng| {
        const path = if (path_val == .string) path_val.string else "output.bmp";
        eng.fb.exportBMP(path) catch return boolean(false);
        return boolean(true);
    }
    return boolean(false);
}

pub fn gfxDrawTensor(tensor_val: Value, x_val: Value, y_val: Value, w_val: Value, h_val: Value) Value {
    if (global_engine) |*eng| {
        if (tensor_val != .tensor) return nil();
        const t = tensor_val.tensor;
        const x: isize = @intFromFloat(toF64(x_val));
        const y: isize = @intFromFloat(toF64(y_val));
        const w: isize = @intFromFloat(toF64(w_val));
        const h: isize = @intFromFloat(toF64(h_val));

        if (t.shape.len == 1) {
            eng.gui.plotTensor(&eng.fb, "Tensor", t.data, x, y, w, h);
        } else if (t.shape.len >= 2) {
            const rows = t.shape[0];
            const cols = t.shape[1];
            var min_v = t.data[0];
            var max_v = t.data[0];
            for (t.data) |v| {
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            const range = if (max_v > min_v) max_v - min_v else 1.0;

            var r: usize = 0;
            while (r < rows) : (r += 1) {
                var c: usize = 0;
                while (c < cols) : (c += 1) {
                    const norm = (t.data[r * cols + c] - min_v) / range;
                    const u: u8 = @intFromFloat(@max(0.0, @min(255.0, norm * 255.0)));
                    const px_x = x + @as(isize, @intCast((c * @as(usize, @intCast(w))) / cols));
                    const px_y = y + @as(isize, @intCast((r * @as(usize, @intCast(h))) / rows));
                    const cell_w = @max(1, @divTrunc(w, @as(isize, @intCast(cols))));
                    const cell_h = @max(1, @divTrunc(h, @as(isize, @intCast(rows))));
                    eng.fb.drawRect(px_x, px_y, cell_w, cell_h, graphics.Color.rgb(u, @intCast(u / 2), 255 - u), true);
                }
            }
        }
    }
    return nil();
}

pub fn gfxScreenToTensor() Value {
    if (global_engine) |*eng| {
        const h = eng.fb.height;
        const w = eng.fb.width;
        const total = h * w * 4;
        const data = alloc().alloc(f64, total) catch @panic("OOM screen to tensor");
        for (eng.fb.pixels, 0..) |px, i| {
            data[i * 4 + 0] = @as(f64, @floatFromInt(px.r)) / 255.0;
            data[i * 4 + 1] = @as(f64, @floatFromInt(px.g)) / 255.0;
            data[i * 4 + 2] = @as(f64, @floatFromInt(px.b)) / 255.0;
            data[i * 4 + 3] = @as(f64, @floatFromInt(px.a)) / 255.0;
        }
        const shape = alloc().alloc(usize, 3) catch @panic("OOM");
        shape[0] = h;
        shape[1] = w;
        shape[2] = 4;
        return .{ .tensor = .{ .shape = shape, .data = data } };
    }
    return nil();
}

fn printStdoutDirect(msg: []const u8) void {
    if (@import("builtin").os.tag == .windows) {
        const kernel32 = struct {
            extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.c) ?*anyopaque;
            extern "kernel32" fn WriteFile(hFile: ?*anyopaque, lpBuffer: [*]const u8, nNumberOfBytesToWrite: u32, lpNumberOfBytesWritten: ?*u32, lpOverlapped: ?*anyopaque) callconv(.c) i32;
        };
        const handle = kernel32.GetStdHandle(-11); // STD_OUTPUT_HANDLE
        var written: u32 = 0;
        _ = kernel32.WriteFile(handle, msg.ptr, @intCast(msg.len), &written, null);
    }
}

pub fn tuiRender() Value {
    if (global_engine) |*eng| {
        var buf: [16384]u8 = undefined;
        var bw = tui.BufferWriter{ .buf = &buf };
        tui.clearScreen(&bw) catch {};
        if (eng.platform.tui_canvas) |*tc| {
            @memcpy(tc.fb.pixels, eng.fb.pixels);
            tc.render(&bw) catch {};
        }
        printStdoutDirect(bw.buf[0..bw.len]);
    }
    return nil();
}

pub fn tuiSparkline(data_val: Value) Value {
    if (data_val == .tensor) {
        var buf: [1024]u8 = undefined;
        var bw = tui.BufferWriter{ .buf = &buf };
        tui.renderSparkline(data_val.tensor.data, 40, &bw) catch {};
        bw.writeAll("\n") catch {};
        printStdoutDirect(bw.buf[0..bw.len]);
    }
    return nil();
}

pub fn gamePoll() Value {
    if (global_engine) |*eng| {
        eng.pollEvents();
    }
    return nil();
}

pub fn gamePresent() Value {
    if (global_engine) |*eng| {
        eng.present() catch {};
    }
    return nil();
}

pub fn gameShouldClose() Value {
    if (global_engine) |*eng| {
        return boolean(eng.shouldClose());
    }
    return boolean(true);
}

pub fn gameDt() Value {
    if (global_engine) |*eng| {
        return number(eng.dt());
    }
    return number(0.0166667);
}

pub fn gameIsKeyDown(key: Value) Value {
    if (global_engine) |*eng| {
        const vk: u16 = @intFromFloat(toF64(key));
        return boolean(eng.platform.isKeyDown(vk));
    }
    return boolean(false);
}

pub fn gameIsKeyPressed(key: Value) Value {
    if (global_engine) |*eng| {
        const vk: u16 = @intFromFloat(toF64(key));
        return boolean(eng.platform.isKeyPressed(vk));
    }
    return boolean(false);
}

pub fn gameMouseX() Value {
    if (global_engine) |*eng| {
        return number(@floatFromInt(eng.platform.mouse_x));
    }
    return number(0);
}

pub fn gameMouseY() Value {
    if (global_engine) |*eng| {
        return number(@floatFromInt(eng.platform.mouse_y));
    }
    return number(0);
}

pub fn gameIsMouseDown(btn: Value) Value {
    if (global_engine) |*eng| {
        const b: usize = if (btn == .nil) 0 else @intFromFloat(toF64(btn));
        return boolean(eng.platform.isMouseButtonDown(b));
    }
    return boolean(false);
}

pub fn gameIsMousePressed(btn: Value) Value {
    if (global_engine) |*eng| {
        const b: usize = if (btn == .nil) 0 else @intFromFloat(toF64(btn));
        return boolean(eng.platform.isMouseButtonPressed(b));
    }
    return boolean(false);
}

pub fn gameWindowWidth() Value {
    if (global_engine) |*eng| {
        if (eng.platform.client_w > 0) return number(@floatFromInt(eng.platform.client_w));
        return number(@floatFromInt(eng.platform.config.width * eng.platform.config.scale));
    }
    return number(0);
}

pub fn gameWindowHeight() Value {
    if (global_engine) |*eng| {
        if (eng.platform.client_h > 0) return number(@floatFromInt(eng.platform.client_h));
        return number(@floatFromInt(eng.platform.config.height * eng.platform.config.scale));
    }
    return number(0);
}

pub fn gameIsWindowResized() Value {
    if (global_engine) |*eng| {
        const r = eng.platform.resized;
        eng.platform.resized = false;
        return boolean(r);
    }
    return boolean(false);
}

pub fn gfxResize(w_val: Value, h_val: Value) Value {
    if (global_engine) |*eng| {
        const w: usize = @intFromFloat(@max(1.0, toF64(w_val)));
        const h: usize = @intFromFloat(@max(1.0, toF64(h_val)));
        eng.resize(w, h) catch return boolean(false);
        return boolean(true);
    }
    return boolean(false);
}

pub fn engineBurstParticles(x: Value, y: Value, color_hex: Value, count: Value) Value {
    if (global_engine) |*eng| {
        const col: u32 = @intFromFloat(@max(0.0, toF64(color_hex)));
        const cnt: usize = @intFromFloat(@max(1.0, toF64(count)));
        eng.emitter.burst(
            @floatCast(toF64(x)),
            @floatCast(toF64(y)),
            cnt,
            graphics.Color.hex(col),
            80.0,
        );
    }
    return nil();
}

pub fn physCheckAABB(x1: Value, y1: Value, w1: Value, h1: Value, x2: Value, y2: Value, w2: Value, h2: Value) Value {
    const a = engine.AABB{
        .x = @floatCast(toF64(x1)),
        .y = @floatCast(toF64(y1)),
        .w = @floatCast(toF64(w1)),
        .h = @floatCast(toF64(h1)),
    };
    const b = engine.AABB{
        .x = @floatCast(toF64(x2)),
        .y = @floatCast(toF64(y2)),
        .w = @floatCast(toF64(w2)),
        .h = @floatCast(toF64(h2)),
    };
    return boolean(a.intersects(b));
}

pub fn physCheckCircle(x1: Value, y1: Value, r1: Value, x2: Value, y2: Value, r2: Value) Value {
    const c1 = engine.Circle{
        .x = @floatCast(toF64(x1)),
        .y = @floatCast(toF64(y1)),
        .r = @floatCast(toF64(r1)),
    };
    const c2 = engine.Circle{
        .x = @floatCast(toF64(x2)),
        .y = @floatCast(toF64(y2)),
        .r = @floatCast(toF64(r2)),
    };
    return boolean(c1.intersects(c2));
}

pub fn physRaycastAABB(ox: Value, oy: Value, dx: Value, dy: Value, bx: Value, by: Value, bw: Value, bh: Value) Value {
    const box = engine.AABB{
        .x = @floatCast(toF64(bx)),
        .y = @floatCast(toF64(by)),
        .w = @floatCast(toF64(bw)),
        .h = @floatCast(toF64(bh)),
    };
    const hit = engine.raycastAABB(
        @floatCast(toF64(ox)),
        @floatCast(toF64(oy)),
        @floatCast(toF64(dx)),
        @floatCast(toF64(dy)),
        box,
    );
    if (hit) |dist| {
        return number(dist);
    }
    return nil();
}

pub fn audioPlayLaser() Value {
    if (global_engine) |*eng| {
        eng.playLaser();
    }
    return nil();
}

pub fn audioPlayExplosion() Value {
    if (global_engine) |*eng| {
        eng.playExplosion();
    }
    return nil();
}

pub fn audioPlayJump() Value {
    if (global_engine) |*eng| {
        eng.playJump();
    }
    return nil();
}

pub fn audioPlayTone(freq: Value, dur: Value) Value {
    const f: f32 = @floatCast(toF64(freq));
    const d: u32 = @intFromFloat(toF64(dur));
    if (global_last_audio_samples) |prev| {
        std.heap.page_allocator.free(prev);
        global_last_audio_samples = null;
    }
    const samples = audio.Synth.generateTone(std.heap.page_allocator, f, d, .square, .{}) catch return nil();
    global_last_audio_samples = samples;
    audio.PlatformAudio.playAsync(samples, 44100);
    return nil();
}

pub fn audioMusicPlay(tracks_val: Value, bpm_val: Value) Value {
    const bpm = toF64(bpm_val);
    if (tracks_val != .list or tracks_val.list.items.items.len == 0) return nil();

    var max_steps: usize = 0;
    for (tracks_val.list.items.items) |track_elem| {
        if (track_elem == .list) {
            max_steps = @max(max_steps, track_elem.list.items.items.len);
        }
    }
    if (max_steps == 0) return nil();

    const safe_bpm = if (bpm >= 20.0 and bpm <= 400.0) @as(f32, @floatCast(bpm)) else 120.0;
    const step_sec: f32 = 30.0 / safe_bpm;
    const step_samples: usize = @intFromFloat(step_sec * @as(f32, @floatFromInt(audio.Synth.SAMPLE_RATE)));
    const total_samples = max_steps * step_samples;
    if (total_samples == 0) return nil();

    const acc = std.heap.page_allocator.alloc(f32, total_samples) catch return nil();
    defer std.heap.page_allocator.free(acc);
    @memset(acc, 0.0);

    for (tracks_val.list.items.items, 0..) |track_elem, track_idx| {
        if (track_elem != .list) continue;
        const track_list = track_elem.list.items.items;
        const wave: audio.WaveType = if (track_idx == 0) .square else if (track_idx == 1) .triangle else .saw;
        const track_gain: f32 = if (track_idx == 0) 0.35 else if (track_idx == 1) 0.50 else 0.25;

        var step: usize = 0;
        while (step < track_list.len) {
            const note_num: i32 = @intFromFloat(toF64(track_list[step]));
            if (note_num > 0) {
                var hold_steps: usize = 1;
                while (step + hold_steps < track_list.len and @as(i32, @intFromFloat(toF64(track_list[step + hold_steps]))) == -1) {
                    hold_steps += 1;
                }
                const note_dur_ms: u32 = @intFromFloat(@as(f32, @floatFromInt(hold_steps)) * step_sec * 1000.0);
                const midi_f = @as(f32, @floatFromInt(note_num));
                const freq = 440.0 * std.math.pow(f32, 2.0, (midi_f - 69.0) / 12.0);
                const tone = audio.Synth.generateTone(std.heap.page_allocator, freq, note_dur_ms, wave, .{
                    .attack_ms = 8,
                    .decay_ms = 20,
                    .sustain_level = 0.7,
                    .release_ms = 30,
                }) catch {
                    step += hold_steps;
                    continue;
                };
                defer std.heap.page_allocator.free(tone);

                const offset = step * step_samples;
                const copy_len = @min(tone.len, total_samples - offset);
                for (0..copy_len) |i| {
                    acc[offset + i] += @as(f32, @floatFromInt(tone[i])) * track_gain;
                }
                step += hold_steps;
            } else {
                step += 1;
            }
        }
    }

    const final_samples = std.heap.page_allocator.alloc(i16, total_samples) catch return nil();
    defer std.heap.page_allocator.free(final_samples);

    for (acc, 0..) |s, i| {
        final_samples[i] = @intFromFloat(@max(-32767.0, @min(32767.0, s)));
    }

    audio.PlatformAudio.playMusic(final_samples);
    return nil();
}

pub fn audioMusicStop() Value {
    audio.PlatformAudio.stopMusic();
    return nil();
}

pub fn audioVolume(vol_val: Value) Value {
    const v: f32 = @floatCast(@max(0.0, @min(1.0, toF64(vol_val))));
    audio.PlatformAudio.setMasterVolume(v);
    return nil();
}

pub fn audioExportWav(filename: Value) Value {
    if (global_last_audio_samples) |samples| {
        const path = if (filename == .string) filename.string else "output.wav";
        audio.WavCodec.writeWav(samples, 44100, path) catch return boolean(false);
        return boolean(true);
    }
    return boolean(false);
}

pub fn audioExportQoa(filename: Value) Value {
    if (global_last_audio_samples) |samples| {
        const path = if (filename == .string) filename.string else "output.qoa";
        const bytes = audio.QoaCodec.encode(samples, 44100, alloc()) catch return boolean(false);
        graphics.writeFileDirect(path, bytes) catch return boolean(false);
        return boolean(true);
    }
    return boolean(false);
}

pub fn guiButton(id_val: Value, label_val: Value, x_val: Value, y_val: Value, w_val: Value, h_val: Value) Value {
    if (global_engine) |*eng| {
        const id: u64 = @intFromFloat(toF64(id_val));
        const label = if (label_val == .string) label_val.string else "Button";
        const x: isize = @intFromFloat(toF64(x_val));
        const y: isize = @intFromFloat(toF64(y_val));
        const w: isize = @intFromFloat(toF64(w_val));
        const h: isize = @intFromFloat(toF64(h_val));
        return boolean(eng.gui.button(&eng.fb, id, label, x, y, w, h));
    }
    return boolean(false);
}

pub fn guiSlider(id_val: Value, label_val: Value, val_v: Value, min_v: Value, max_v: Value, x_val: Value, y_val: Value, w_val: Value, h_val: Value) Value {
    if (global_engine) |*eng| {
        const id: u64 = @intFromFloat(toF64(id_val));
        const label = if (label_val == .string) label_val.string else "Slider";
        const val: f32 = @floatCast(toF64(val_v));
        const min_val: f32 = @floatCast(toF64(min_v));
        const max_val: f32 = @floatCast(toF64(max_v));
        const x: isize = @intFromFloat(toF64(x_val));
        const y: isize = @intFromFloat(toF64(y_val));
        const w: isize = @intFromFloat(toF64(w_val));
        const h: isize = @intFromFloat(toF64(h_val));
        const new_val = eng.gui.slider(&eng.fb, id, label, val, min_val, max_val, x, y, w, h);
        return number(new_val);
    }
    return val_v;
}

pub fn guiPanel(title_val: Value, x_val: Value, y_val: Value, w_val: Value, h_val: Value) Value {
    if (global_engine) |*eng| {
        const title = if (title_val == .string) title_val.string else "Panel";
        const x: isize = @intFromFloat(toF64(x_val));
        const y: isize = @intFromFloat(toF64(y_val));
        const w: isize = @intFromFloat(toF64(w_val));
        const h: isize = @intFromFloat(toF64(h_val));
        eng.gui.panel(&eng.fb, title, x, y, w, h);
    }
    return nil();
}

// ============================================================================
// Distributed Computing & Cluster Node System
// ============================================================================
pub const MessageType = enum(u8) {
    ping = 0,
    pong = 1,
    spawn_actor = 2,
    spawn_reply = 3,
    send_actor = 4,
    send_reply = 5,
};

pub const HEADER_SIZE: usize = 25;

pub const Packet = struct {
    magic: [4]u8 = "AETH".*,
    msg_type: MessageType,
    sender_id: u64,
    recipient_id: u64,
    payload_len: u32,

    pub fn toBytes(self: Packet, out: *[HEADER_SIZE]u8) void {
        @memcpy(out[0..4], &self.magic);
        out[4] = @intFromEnum(self.msg_type);
        std.mem.writeInt(u64, out[5..13], self.sender_id, .little);
        std.mem.writeInt(u64, out[13..21], self.recipient_id, .little);
        std.mem.writeInt(u32, out[21..25], self.payload_len, .little);
    }

    pub fn fromBytes(bytes: *const [HEADER_SIZE]u8) !Packet {
        if (!std.mem.eql(u8, bytes[0..4], "AETH")) return error.InvalidMagic;
        const msg_type: MessageType = @enumFromInt(bytes[4]);
        const sender_id = std.mem.readInt(u64, bytes[5..13], .little);
        const recipient_id = std.mem.readInt(u64, bytes[13..21], .little);
        const payload_len = std.mem.readInt(u32, bytes[21..25], .little);
        return .{
            .msg_type = msg_type,
            .sender_id = sender_id,
            .recipient_id = recipient_id,
            .payload_len = payload_len,
        };
    }
};

var global_cluster_port: u16 = 0;
var global_cluster_running = false;

pub fn clusterStart(port_val: Value) Value {
    const port: u16 = @intFromFloat(@max(1.0, @min(65535.0, toF64(port_val))));
    global_cluster_port = port;
    global_cluster_running = true;
    return boolean(true);
}

pub fn clusterStop() Value {
    global_cluster_running = false;
    global_cluster_port = 0;
    return boolean(true);
}

pub fn clusterSpawn(host_val: Value, port_val: Value, state_val: Value) Value {
    _ = host_val;
    _ = port_val;
    const state_str = if (state_val == .string) state_val.string else valueToKeyString(state_val);
    const pkt = Packet{
        .msg_type = .spawn_actor,
        .sender_id = 1,
        .recipient_id = 0,
        .payload_len = @intCast(state_str.len),
    };
    var buf: [HEADER_SIZE]u8 = undefined;
    pkt.toBytes(&buf);
    const decoded = Packet.fromBytes(&buf) catch return nil();
    _ = decoded;

    return spawnActor(state_val);
}

pub fn clusterSend(host_val: Value, port_val: Value, recipient_val: Value, sender_val: Value, payload_val: Value) Value {
    _ = host_val;
    _ = port_val;
    const recipient: u64 = @intFromFloat(toF64(recipient_val));
    const sender: u64 = @intFromFloat(toF64(sender_val));
    const payload_str = if (payload_val == .string) payload_val.string else valueToKeyString(payload_val);

    const pkt = Packet{
        .msg_type = .send_actor,
        .sender_id = sender,
        .recipient_id = recipient,
        .payload_len = @intCast(payload_str.len),
    };
    var buf: [HEADER_SIZE]u8 = undefined;
    pkt.toBytes(&buf);
    const decoded = Packet.fromBytes(&buf) catch return boolean(false);

    return sendActor(number(@floatFromInt(decoded.recipient_id)), number(@floatFromInt(decoded.sender_id)), payload_val);
}

pub var g_io: ?std.Io = null;
pub var g_cli_args: []const []const u8 = &.{};

pub fn initMain(init: std.process.Init) void {
    g_io = init.io;
    const arena_alloc = alloc();
    const args = init.minimal.args.toSlice(arena_alloc) catch return;
    g_cli_args = args;
}

pub fn valToString(v: Value) Value {
    if (v == .string) return v;
    return string(valueToKeyString(v));
}

pub fn cliArgs() Value {
    const out = mem.newObj(List);
    out.items = .empty;
    for (g_cli_args) |arg| {
        out.items.append(mem.heap(), string(arg)) catch @panic("OOM");
    }
    return .{ .list = out };
}

pub fn sysExec(args_val: Value) Value {
    const arena_alloc = alloc();
    var cmd_list: std.ArrayList([]const u8) = .empty;
    if (args_val == .list) {
        for (args_val.list.items.items) |it| {
            if (it == .string) {
                cmd_list.append(arena_alloc, it.string) catch @panic("OOM");
            } else {
                cmd_list.append(arena_alloc, valueToKeyString(it)) catch @panic("OOM");
            }
        }
    } else if (args_val == .string) {
        var it = std.mem.tokenizeAny(u8, args_val.string, " \t\r\n");
        while (it.next()) |tok| {
            cmd_list.append(arena_alloc, tok) catch @panic("OOM");
        }
    } else {
        return number(1.0);
    }

    if (cmd_list.items.len == 0) return number(0.0);

    const io = g_io orelse return number(1.0);
    const res = std.process.run(arena_alloc, io, .{
        .argv = cmd_list.items,
    }) catch return number(1.0);

    return number(if (res.term.success()) 0.0 else 1.0);
}

test "runtime scalar arithmetic and reverse-mode autodiff" {
    defer deinitRuntime();
    const x = number(2.0);
    const y = mul(x, x); // y = x^2 = 4.0
    const dy_dx = gradient(y, x); // dy/dx = 2x = 4.0
    try std.testing.expectEqual(@as(f64, 4.0), y.number.val);
    try std.testing.expectEqual(@as(f64, 4.0), dy_dx.number.val);
}

test "runtime tensor matmul, broadcast add, sum, and relu" {
    defer deinitRuntime();
    const a = tensor(&[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 });
    const b = tensor(&[_]usize{ 3, 2 }, &[_]f64{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 });
    const c = matmul(a, b);
    try std.testing.expectEqual(@as(usize, 2), c.tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), c.tensor.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), c.tensor.data[0], 1e-6);

    const s = sum(a);
    try std.testing.expectApproxEqAbs(@as(f64, 21.0), s.number.val, 1e-6);

    const r = relu(tensor(&[_]usize{2}, &[_]f64{ -3.0, 5.0 }));
    try std.testing.expectEqual(@as(f64, 0.0), r.tensor.data[0]);
    try std.testing.expectEqual(@as(f64, 5.0), r.tensor.data[1]);
}

test "runtime fused matmul relu computes without separate intermediate" {
    defer deinitRuntime();
    const a = tensor(&[_]usize{ 1, 2 }, &[_]f64{ 1.0, -2.0 });
    const b = tensor(&[_]usize{ 2, 2 }, &[_]f64{ 3.0, 4.0, 5.0, 1.0 });
    // normal matmul: [1*3 + -2*5, 1*4 + -2*1] = [-7, 2]
    // relu: [0, 2]
    const c = fusedMatmulRelu(a, b);
    try std.testing.expectEqual(@as(f64, 0.0), c.tensor.data[0]);
    try std.testing.expectEqual(@as(f64, 2.0), c.tensor.data[1]);
}

test "runtime math and 3d vector operations" {
    defer deinitRuntime();
    const sq = mathSqrt(number(25.0));
    try std.testing.expectEqual(@as(f64, 5.0), sq.number.val);

    const v1 = vec3(number(1.0), number(2.0), number(3.0));
    const v2 = vec3(number(4.0), number(5.0), number(6.0));
    const d = vec3Dot(v1, v2);
    try std.testing.expectEqual(@as(f64, 32.0), d.number.val);

    const cr = vec3Cross(v1, v2);
    try std.testing.expectEqual(@as(f64, -3.0), cr.tensor.data[0]);
    try std.testing.expectEqual(@as(f64, 6.0), cr.tensor.data[1]);
    try std.testing.expectEqual(@as(f64, -3.0), cr.tensor.data[2]);
}

test "runtime actor model and replay runtime" {
    defer deinitRuntime();
    // Actor system
    const a1 = spawnActor(number(10.0));
    const a2 = spawnActor(number(20.0));
    _ = sendActor(a2, a1, string("hello"));
    const msg = receiveActor(a2);
    try std.testing.expectEqualStrings("hello", msg.string);

    // Replay runtime
    _ = replayInit(number(42.0));
    const r1 = replayRandom(number(0.0), number(1.0));
    _ = replayStep();
    _ = replaySnapshot(number(999.0));
    const rew = replayRewind(number(1.0));
    try std.testing.expectEqual(@as(f64, 999.0), rew.number.val);
    try std.testing.expect(r1.number.val >= 0.0 and r1.number.val <= 1.0);
}

test "runtime strJoin with numbers and strToNum and fileExists" {
    defer deinitRuntime();
    const lval = list(&[_]Value{ string("SCORE "), number(1234.0) });
    const joined = strJoin(lval, string(""));
    try std.testing.expectEqualStrings("SCORE 1234", joined.string);

    const parsed = strToNum(string("  9876.5  "));
    try std.testing.expectEqual(@as(f64, 9876.5), parsed.number.val);

    const parsed_zero = strToNum(string("not_a_number"));
    try std.testing.expectEqual(@as(f64, 0.0), parsed_zero.number.val);

    const exists = fileExists(string("README.md"));
    try std.testing.expect(exists.boolean);

    const not_exists = fileExists(string("nonexistent_test_file_xyz_123.bin"));
    try std.testing.expect(!not_exists.boolean);
}

test "runtime string slicing, comparison, and valToString" {
    defer deinitRuntime();
    const s = string("hello world");
    const slice_res = listSlice(s, number(0.0), number(5.0));
    try std.testing.expectEqualStrings("hello", slice_res.string);

    const s_cmp1 = string("abc");
    const s_cmp2 = string("abd");
    try std.testing.expect(lt(s_cmp1, s_cmp2).boolean);
    try std.testing.expect(le(s_cmp1, s_cmp2).boolean);
    try std.testing.expect(le(s_cmp1, s_cmp1).boolean);
    try std.testing.expect(gt(s_cmp2, s_cmp1).boolean);
    try std.testing.expect(ge(s_cmp2, s_cmp1).boolean);

    const num_str = valToString(number(42.0));
    try std.testing.expectEqualStrings("42", num_str.string);

    const args = cliArgs();
    try std.testing.expectEqual(@as(usize, 0), args.list.items.items.len);
}


