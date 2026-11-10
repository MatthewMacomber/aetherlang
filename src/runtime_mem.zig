//! Aether Runtime Memory Management System
//! Implements Arena, Manual, Tracing GC, and Autofree strategies.

const std = @import("std");
const builtin = @import("builtin");
const runtime = @import("runtime.zig");

pub const Value = runtime.Value;
pub const Tensor = runtime.Tensor;
pub const List = runtime.List;
pub const Map = runtime.Map;
pub const Set = runtime.Set;
pub const Deque = runtime.Deque;

pub const MemMode = enum {
    autofree,
    arena,
    manual,
    gc,
    gc_stress,
};

pub var mode: MemMode = .autofree;

// ponytail: default to smp_allocator on native OS, wasm_allocator on wasm, testing allocator in tests
pub var backing: std.mem.Allocator = if (builtin.cpu.arch.isWasm())
    std.heap.wasm_allocator
else
    std.heap.smp_allocator;

var payload_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

// Scoped arena blocks (thread-local for concurrency)
pub threadlocal var regions: [64]std.heap.ArenaAllocator = undefined;
pub threadlocal var region_depth: usize = 0;

pub fn heap() std.mem.Allocator {
    if (mode == .arena) {
        if (region_depth > 0) {
            return regions[region_depth - 1].allocator();
        }
        return payload_arena.allocator();
    }
    return backing;
}

// ============================================================================
// Allocation Registry
// ============================================================================

pub const PayloadKind = enum {
    bytes,
    f64s,
    usizes,
    list,
    map,
    set,
    deque,
    tensor,
};

pub const Header = struct {
    len: usize,
    kind: PayloadKind,
    marked: bool = false,
};

pub const Mutex = struct {
    state: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn lock(self: *Mutex) void {
        while (self.state.swap(true, .acquire)) {
            std.Thread.yield() catch {};
        }
    }

    pub fn unlock(self: *Mutex) void {
        self.state.store(false, .release);
    }
};

var registry: std.AutoHashMapUnmanaged(usize, Header) = .empty;
var registry_mutex: Mutex = .{};
var total_live_bytes: usize = 0;
var bytes_since_last_gc: usize = 0;
const GC_THRESHOLD_FLOOR: usize = 1024 * 1024; // 1 MiB floor

pub fn liveCount() usize {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    return registry.count();
}

pub fn liveBytes() usize {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    return total_live_bytes;
}

fn registerPayload(addr: usize, len: usize, kind: PayloadKind) void {
    if (mode == .arena) return;
    registry_mutex.lock();
    defer registry_mutex.unlock();

    registry.put(backing, addr, .{ .len = len, .kind = kind, .marked = false }) catch @panic("OOM in memory registry");
    total_live_bytes += len;
    bytes_since_last_gc += len;
}

fn unregisterPayloadLocked(addr: usize) ?Header {
    if (registry.fetchRemove(addr)) |kv| {
        total_live_bytes -|= kv.value.len;
        return kv.value;
    }
    return null;
}

fn unregisterPayload(addr: usize) ?Header {
    if (mode == .arena) return null;
    registry_mutex.lock();
    defer registry_mutex.unlock();
    return unregisterPayloadLocked(addr);
}

// ============================================================================
// Payload Constructors
// ============================================================================

pub fn newBytes(n: usize) []u8 {
    const h = heap();
    const slice = h.alloc(u8, n) catch @panic("OOM in newBytes");
    registerPayload(@intFromPtr(slice.ptr), n, .bytes);
    return slice;
}

pub fn dupeBytes(s: []const u8) []u8 {
    const slice = newBytes(s.len);
    @memcpy(slice, s);
    return slice;
}

pub fn newF64s(n: usize) []f64 {
    const h = heap();
    const slice = h.alloc(f64, n) catch @panic("OOM in newF64s");
    registerPayload(@intFromPtr(slice.ptr), n * @sizeOf(f64), .f64s);
    return slice;
}

pub fn newUsizes(src: []const usize) []usize {
    const h = heap();
    const slice = h.alloc(usize, src.len) catch @panic("OOM in newUsizes");
    @memcpy(slice, src);
    registerPayload(@intFromPtr(slice.ptr), src.len * @sizeOf(usize), .usizes);
    return slice;
}

pub fn newObj(comptime T: type) *T {
    const h = heap();
    const ptr = h.create(T) catch @panic("OOM in newObj");
    const kind: PayloadKind = if (T == List)
        .list
    else if (T == Map)
        .map
    else if (T == Set)
        .set
    else if (T == Deque)
        .deque
    else
        .bytes;
    registerPayload(@intFromPtr(ptr), @sizeOf(T), kind);
    return ptr;
}

// ============================================================================
// Deallocation & Finalization
// ============================================================================

fn finalizePayloadLocked(addr: usize, h: Header) void {
    const alloc_inst = backing;
    switch (h.kind) {
        .bytes => {
            const ptr: [*]u8 = @ptrFromInt(addr);
            alloc_inst.free(ptr[0..h.len]);
        },
        .f64s => {
            const ptr: [*]f64 = @ptrFromInt(addr);
            const count = h.len / @sizeOf(f64);
            alloc_inst.free(ptr[0..count]);
        },
        .usizes => {
            const ptr: [*]usize = @ptrFromInt(addr);
            const count = h.len / @sizeOf(usize);
            alloc_inst.free(ptr[0..count]);
        },
        .list => {
            const list_ptr: *List = @ptrFromInt(addr);
            list_ptr.items.deinit(alloc_inst);
            alloc_inst.destroy(list_ptr);
        },
        .map => {
            const map_ptr: *Map = @ptrFromInt(addr);
            var it = map_ptr.entries.keyIterator();
            while (it.next()) |k| {
                _ = unregisterPayloadLocked(@intFromPtr(k.*.ptr));
                alloc_inst.free(k.*);
            }
            map_ptr.entries.deinit();
            alloc_inst.destroy(map_ptr);
        },
        .set => {
            const set_ptr: *Set = @ptrFromInt(addr);
            var it = set_ptr.entries.keyIterator();
            while (it.next()) |k| {
                _ = unregisterPayloadLocked(@intFromPtr(k.*.ptr));
                alloc_inst.free(k.*);
            }
            set_ptr.entries.deinit();
            alloc_inst.destroy(set_ptr);
        },
        .deque => {
            const deque_ptr: *Deque = @ptrFromInt(addr);
            deque_ptr.items.deinit(alloc_inst);
            alloc_inst.destroy(deque_ptr);
        },
        .tensor => {
            // Handled via components (shape and data)
        },
    }
}

pub fn free(v: Value) void {
    if (mode == .arena) return;
    registry_mutex.lock();
    defer registry_mutex.unlock();
    freeLocked(v);
}

fn freeLocked(v: Value) void {
    switch (v) {
        .string => |s| {
            if (unregisterPayloadLocked(@intFromPtr(s.ptr))) |h| {
                finalizePayloadLocked(@intFromPtr(s.ptr), h);
            }
        },
        .tensor => |t| {
            if (unregisterPayloadLocked(@intFromPtr(t.data.ptr))) |hd| {
                finalizePayloadLocked(@intFromPtr(t.data.ptr), hd);
            }
            if (unregisterPayloadLocked(@intFromPtr(t.shape.ptr))) |hs| {
                finalizePayloadLocked(@intFromPtr(t.shape.ptr), hs);
            }
        },
        .list => |l| {
            if (unregisterPayloadLocked(@intFromPtr(l))) |h| {
                finalizePayloadLocked(@intFromPtr(l), h);
            }
        },
        .map => |m| {
            if (unregisterPayloadLocked(@intFromPtr(m))) |h| {
                finalizePayloadLocked(@intFromPtr(m), h);
            }
        },
        .set => |s| {
            if (unregisterPayloadLocked(@intFromPtr(s))) |h| {
                finalizePayloadLocked(@intFromPtr(s), h);
            }
        },
        .deque => |d| {
            if (unregisterPayloadLocked(@intFromPtr(d))) |h| {
                finalizePayloadLocked(@intFromPtr(d), h);
            }
        },
        else => {},
    }
}

pub fn autoFree(v: Value) void {
    free(v);
}

// ============================================================================
// Multi-Thread Shadow Stack Roots & Handle Scope Pins
// ============================================================================

pub const Root = struct {
    ptr: *anyopaque,
    trace_fn: *const fn (*anyopaque, *Collector) void,
};

pub threadlocal var thread_roots: std.ArrayListUnmanaged(Root) = .empty;
pub threadlocal var thread_pins: std.ArrayListUnmanaged(usize) = .empty;
pub threadlocal var thread_is_registered: bool = false;

pub const ThreadDescriptor = struct {
    roots: *std.ArrayListUnmanaged(Root),
    pins: *std.ArrayListUnmanaged(usize),
    thread_id: std.Thread.Id,
};

var global_threads: std.ArrayListUnmanaged(ThreadDescriptor) = .empty;
var global_threads_mutex: Mutex = .{};

fn ensureThreadRegistered() void {
    if (thread_is_registered) return;
    global_threads_mutex.lock();
    defer global_threads_mutex.unlock();
    const tid = std.Thread.getCurrentId();
    global_threads.append(backing, .{
        .roots = &thread_roots,
        .pins = &thread_pins,
        .thread_id = tid,
    }) catch @panic("OOM registering thread in GC");
    thread_is_registered = true;
}

pub fn gcRoot(p: anytype) void {
    if (mode == .arena or mode == .manual) return;
    ensureThreadRegistered();

    const PtrType = @TypeOf(p);
    const ChildType = @typeInfo(PtrType).pointer.child;

    const Wrapper = struct {
        fn trace(raw_ptr: *anyopaque, col: *Collector) void {
            const typed_ptr: *ChildType = @ptrCast(@alignCast(raw_ptr));
            col.traceAny(typed_ptr.*);
        }
    };

    thread_roots.append(backing, .{
        .ptr = p,
        .trace_fn = Wrapper.trace,
    }) catch @panic("OOM pushing GC root");
}

pub fn gcUnroot(p: anytype) void {
    if (mode == .arena or mode == .manual) return;
    _ = p;
    if (thread_roots.items.len > 0) {
        _ = thread_roots.pop();
    }
}

pub fn gcPin(v: anytype) @TypeOf(v) {
    if (mode == .arena or mode == .manual) return v;
    ensureThreadRegistered();

    const T = @TypeOf(v);
    if (T == Value) {
        switch (v) {
            .string => |s| thread_pins.append(backing, @intFromPtr(s.ptr)) catch {},
            .tensor => |t| {
                thread_pins.append(backing, @intFromPtr(t.data.ptr)) catch {};
                thread_pins.append(backing, @intFromPtr(t.shape.ptr)) catch {};
            },
            .list => |l| thread_pins.append(backing, @intFromPtr(l)) catch {},
            .map => |m| thread_pins.append(backing, @intFromPtr(m)) catch {},
            .set => |s| thread_pins.append(backing, @intFromPtr(s)) catch {},
            .deque => |d| thread_pins.append(backing, @intFromPtr(d)) catch {},
            else => {},
        }
    }
    return v;
}

pub fn gcMark() usize {
    return thread_pins.items.len;
}

pub fn gcSafepoint(mark: usize) void {
    if (mode == .arena or mode == .manual) return;
    if (mark <= thread_pins.items.len) {
        thread_pins.shrinkRetainingCapacity(mark);
    }
    const threshold = @max(GC_THRESHOLD_FLOOR, total_live_bytes * 2);
    if (mode == .gc_stress or bytes_since_last_gc > threshold) {
        collect();
    }
}

// ============================================================================
// Tracing Collector
// ============================================================================

var gc_mutex: Mutex = .{};

pub const Collector = struct {
    worklist: std.ArrayListUnmanaged(usize) = .empty,

    pub fn traceAddress(self: *Collector, addr: usize) void {
        if (addr == 0) return;
        if (registry.getPtr(addr)) |h| {
            if (!h.marked) {
                h.marked = true;
                self.worklist.append(backing, addr) catch @panic("OOM in GC worklist");
            }
        }
    }

    pub fn traceValue(self: *Collector, v: Value) void {
        switch (v) {
            .string => |s| self.traceAddress(@intFromPtr(s.ptr)),
            .tensor => |t| {
                self.traceAddress(@intFromPtr(t.data.ptr));
                self.traceAddress(@intFromPtr(t.shape.ptr));
            },
            .list => |l| self.traceAddress(@intFromPtr(l)),
            .map => |m| self.traceAddress(@intFromPtr(m)),
            .set => |s| self.traceAddress(@intFromPtr(s)),
            .deque => |d| self.traceAddress(@intFromPtr(d)),
            else => {},
        }
    }

    pub fn traceAny(self: *Collector, val: anytype) void {
        const T = @TypeOf(val);
        if (T == Value) {
            self.traceValue(val);
        } else if (@typeInfo(T) == .@"struct") {
            inline for (std.meta.fields(T)) |f| {
                self.traceAny(@field(val, f.name));
            }
        }
    }

    pub fn drain(self: *Collector) void {
        while (self.worklist.pop()) |addr| {
            const h = registry.get(addr) orelse continue;
            switch (h.kind) {
                .list => {
                    const l: *List = @ptrFromInt(addr);
                    for (l.items.items) |item| {
                        self.traceValue(item);
                    }
                },
                .map => {
                    const m: *Map = @ptrFromInt(addr);
                    var it = m.entries.iterator();
                    while (it.next()) |entry| {
                        self.traceAddress(@intFromPtr(entry.key_ptr.*.ptr));
                        self.traceValue(entry.value_ptr.*);
                    }
                },
                .set => {
                    const s: *Set = @ptrFromInt(addr);
                    var it = s.entries.keyIterator();
                    while (it.next()) |k| {
                        self.traceAddress(@intFromPtr(k.*.ptr));
                    }
                },
                .deque => {
                    const d: *Deque = @ptrFromInt(addr);
                    for (d.items.items) |item| {
                        self.traceValue(item);
                    }
                },
                else => {},
            }
        }
    }
};

pub fn collect() void {
    gc_mutex.lock();
    defer gc_mutex.unlock();

    registry_mutex.lock();
    defer registry_mutex.unlock();

    var col: Collector = .{};
    defer col.worklist.deinit(backing);

    // 1. Mark roots from all registered threads
    global_threads_mutex.lock();
    for (global_threads.items) |t| {
        for (t.roots.items) |r| {
            r.trace_fn(r.ptr, &col);
        }
        for (t.pins.items) |pin_addr| {
            col.traceAddress(pin_addr);
        }
    }
    global_threads_mutex.unlock();

    // 2. Mark runtime globals (autodiff tape, actors, etc.)
    markRuntimeGlobals(&col);

    // 3. Drain worklist iteratively
    col.drain();

    // 4. Sweep unmarked registry entries
    var sweep_list: std.ArrayListUnmanaged(struct { addr: usize, h: Header }) = .empty;
    defer sweep_list.deinit(backing);

    var it = registry.iterator();
    while (it.next()) |entry| {
        if (!entry.value_ptr.marked) {
            sweep_list.append(backing, .{ .addr = entry.key_ptr.*, .h = entry.value_ptr.* }) catch {};
        } else {
            entry.value_ptr.marked = false; // Reset mark bit for next GC cycle
        }
    }

    for (sweep_list.items) |item| {
        _ = registry.remove(item.addr);
        total_live_bytes -|= item.h.len;
        finalizePayloadLocked(item.addr, item.h);
    }

    bytes_since_last_gc = 0;
}

fn markRuntimeGlobals(col: *Collector) void {
    // Autodiff tape nodes and gradients
    const tape_len = runtime.getTensorTapeLen();
    const tape_nodes = runtime.getTensorTapeNodes();
    for (0..tape_len) |i| {
        const node = tape_nodes[i];
        col.traceAddress(@intFromPtr(node.data.ptr));
        col.traceAddress(@intFromPtr(node.shape.ptr));
        if (runtime.getTensorTapeGrad(i)) |grad| {
            col.traceAddress(@intFromPtr(grad.ptr));
        }
    }

    // Actors state & mailboxes
    if (runtime.global_actors) |*actors| {
        var a_it = actors.valueIterator();
        while (a_it.next()) |actor_ptr| {
            col.traceValue(actor_ptr.*.state);
            for (actor_ptr.*.mailbox.items) |msg| {
                col.traceValue(msg.payload);
            }
        }
    }

    // Replay snapshots
    for (runtime.global_replay_snapshots.items) |snap| {
        col.traceValue(snap.state);
    }
}

// ============================================================================
// Scoped Arena Blocks
// ============================================================================

pub fn regionEnter() void {
    if (mode != .arena) return;
    if (region_depth >= 64) @panic("Maximum arena nesting depth (64) exceeded");
    regions[region_depth] = std.heap.ArenaAllocator.init(backing);
    region_depth += 1;
}

pub fn regionExit(result: Value) Value {
    if (mode != .arena) return result;
    if (region_depth == 0) return result;

    region_depth -= 1;
    var child_arena = regions[region_depth];
    defer child_arena.deinit();

    // Deep-copy result out to parent allocator
    const parent_alloc = heap();
    return deepCopyValue(result, parent_alloc);
}

fn deepCopyValue(v: Value, parent_alloc: std.mem.Allocator) Value {
    switch (v) {
        .string => |s| {
            const dup = parent_alloc.alloc(u8, s.len) catch @panic("OOM copying string from arena");
            @memcpy(dup, s);
            return runtime.string(dup);
        },
        .tensor => |t| {
            const shp = parent_alloc.alloc(usize, t.shape.len) catch @panic("OOM");
            @memcpy(shp, t.shape);
            const dt = parent_alloc.alloc(f64, t.data.len) catch @panic("OOM");
            @memcpy(dt, t.data);
            return .{ .tensor = .{ .shape = shp, .data = dt, .id = t.id } };
        },
        .list => |l| {
            const new_l = parent_alloc.create(List) catch @panic("OOM");
            new_l.items = .empty;
            for (l.items.items) |item| {
                new_l.items.append(parent_alloc, deepCopyValue(item, parent_alloc)) catch @panic("OOM");
            }
            return .{ .list = new_l };
        },
        .map => |m| {
            const new_m = parent_alloc.create(Map) catch @panic("OOM");
            new_m.entries = std.StringHashMap(Value).init(parent_alloc);
            var it = m.entries.iterator();
            while (it.next()) |entry| {
                const k_dup = parent_alloc.alloc(u8, entry.key_ptr.*.len) catch @panic("OOM");
                @memcpy(k_dup, entry.key_ptr.*);
                new_m.entries.put(k_dup, deepCopyValue(entry.value_ptr.*, parent_alloc)) catch @panic("OOM");
            }
            return .{ .map = new_m };
        },
        .set => |s| {
            const new_s = parent_alloc.create(Set) catch @panic("OOM");
            new_s.entries = std.StringHashMap(void).init(parent_alloc);
            var it = s.entries.keyIterator();
            while (it.next()) |k| {
                const k_dup = parent_alloc.alloc(u8, k.*.len) catch @panic("OOM");
                @memcpy(k_dup, k.*);
                new_s.entries.put(k_dup, {}) catch @panic("OOM");
            }
            return .{ .set = new_s };
        },
        .deque => |d| {
            const new_d = parent_alloc.create(Deque) catch @panic("OOM");
            new_d.items = .empty;
            for (d.items.items) |item| {
                new_d.items.append(parent_alloc, deepCopyValue(item, parent_alloc)) catch @panic("OOM");
            }
            return .{ .deque = new_d };
        },
        else => return v,
    }
}

// ============================================================================
// Runtime Shutdown & Cleanup
// ============================================================================

pub fn deinitMem() void {
    if (mode == .manual) {
        registry_mutex.lock();
        const live_cnt = registry.count();
        const live_bytes_amt = total_live_bytes;
        registry_mutex.unlock();

        if (live_cnt > 0) {
            std.debug.print("aether (manual memory): {d} allocations ({d} bytes) still live at exit\n", .{ live_cnt, live_bytes_amt });
        }
    }

    registry_mutex.lock();
    var it = registry.iterator();
    while (it.next()) |entry| {
        finalizePayloadLocked(entry.key_ptr.*, entry.value_ptr.*);
    }
    registry.deinit(backing);
    registry = .empty;
    total_live_bytes = 0;
    bytes_since_last_gc = 0;
    registry_mutex.unlock();

    global_threads_mutex.lock();
    for (global_threads.items) |t| {
        t.roots.deinit(backing);
        t.pins.deinit(backing);
    }
    global_threads.deinit(backing);
    global_threads = .empty;
    thread_is_registered = false;
    global_threads_mutex.unlock();

    payload_arena.deinit();
    payload_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    region_depth = 0;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "runtime_mem: registry and manual free" {
    mode = .manual;
    backing = std.testing.allocator;
    defer deinitMem();

    const bytes = newBytes(64);
    try std.testing.expectEqual(@as(usize, 1), liveCount());
    try std.testing.expectEqual(@as(usize, 64), liveBytes());

    free(runtime.string(bytes));
    try std.testing.expectEqual(@as(usize, 0), liveCount());
    try std.testing.expectEqual(@as(usize, 0), liveBytes());
}

test "runtime_mem: tracing gc collects unrooted objects" {
    mode = .gc;
    backing = std.testing.allocator;
    defer deinitMem();

    var kept = runtime.list(&[_]Value{});
    gcRoot(&kept);
    defer gcUnroot(&kept);

    // Create garbage list not rooted
    _ = runtime.list(&[_]Value{runtime.number(1.0), runtime.number(2.0)});

    try std.testing.expect(liveCount() >= 2);
    collect();
    // Kept list should remain, garbage list collected
    try std.testing.expectEqual(@as(usize, 1), liveCount());
}

test "runtime_mem: arena region deep-copies out" {
    mode = .arena;
    backing = std.testing.allocator;
    defer deinitMem();

    regionEnter();
    const str = runtime.string("hello scoped arena");
    const out = regionExit(str);

    try std.testing.expectEqualStrings("hello scoped arena", out.string);
}
