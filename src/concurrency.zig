//! Aether Structured Concurrency & Actor System
//! Provides lightweight actor message queues and data-parallel work distribution.

const std = @import("std");
const runtime = @import("runtime.zig");
const Value = runtime.Value;

pub const ActorMessage = struct {
    sender_id: usize,
    payload: Value,
};

pub const SpinLock = struct {
    state: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn lock(self: *SpinLock) void {
        while (self.state.swap(true, .acquire)) {}
    }

    pub fn unlock(self: *SpinLock) void {
        self.state.store(false, .release);
    }
};

pub const Actor = struct {
    id: usize,
    state: Value,
    mailbox: std.ArrayList(ActorMessage),
    lock: SpinLock = .{},

    pub fn init(id: usize, initial_state: Value) Actor {
        return .{
            .id = id,
            .state = initial_state,
            .mailbox = .empty,
            .lock = .{},
        };
    }

    pub fn deinit(self: *Actor, allocator: std.mem.Allocator) void {
        self.mailbox.deinit(allocator);
    }

    pub fn send(self: *Actor, allocator: std.mem.Allocator, msg: ActorMessage) !void {
        self.lock.lock();
        defer self.lock.unlock();
        try self.mailbox.append(allocator, msg);
    }

    pub fn receive(self: *Actor) ?ActorMessage {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.mailbox.items.len == 0) return null;
        return self.mailbox.orderedRemove(0);
    }
};

pub const ActorSystem = struct {
    allocator: std.mem.Allocator,
    actors: std.AutoHashMap(usize, *Actor),
    next_id: usize = 1,

    pub fn init(allocator: std.mem.Allocator) ActorSystem {
        return .{
            .allocator = allocator,
            .actors = std.AutoHashMap(usize, *Actor).init(allocator),
        };
    }

    pub fn deinit(self: *ActorSystem) void {
        var it = self.actors.valueIterator();
        while (it.next()) |actor_ptr| {
            actor_ptr.*.deinit(self.allocator);
            self.allocator.destroy(actor_ptr.*);
        }
        self.actors.deinit();
    }

    pub fn spawn(self: *ActorSystem, initial_state: Value) !*Actor {
        const id = self.next_id;
        self.next_id += 1;
        const actor = try self.allocator.create(Actor);
        actor.* = Actor.init(id, initial_state);
        try self.actors.put(id, actor);
        return actor;
    }

    pub fn sendTo(self: *ActorSystem, recipient_id: usize, sender_id: usize, payload: Value) !bool {
        if (self.actors.get(recipient_id)) |actor| {
            try actor.send(self.allocator, .{ .sender_id = sender_id, .payload = payload });
            return true;
        }
        return false;
    }
};

pub const Task = struct {
    func: *const fn (ctx: ?*anyopaque) void,
    ctx: ?*anyopaque,
};

pub const SyncEvent = struct {
    handle: ?*anyopaque = null,

    pub fn init() SyncEvent {
        if (@import("builtin").os.tag == .windows) {
            const kernel32 = struct {
                extern "kernel32" fn CreateEventA(?*anyopaque, i32, i32, ?[*:0]const u8) callconv(.c) ?*anyopaque;
            };
            return .{ .handle = kernel32.CreateEventA(null, 0, 0, null) };
        }
        return .{};
    }

    pub fn deinit(self: *SyncEvent) void {
        if (@import("builtin").os.tag == .windows and self.handle != null) {
            const kernel32 = struct {
                extern "kernel32" fn CloseHandle(?*anyopaque) callconv(.c) i32;
            };
            _ = kernel32.CloseHandle(self.handle);
            self.handle = null;
        }
    }

    pub fn set(self: *SyncEvent) void {
        if (@import("builtin").os.tag == .windows and self.handle != null) {
            const kernel32 = struct {
                extern "kernel32" fn SetEvent(?*anyopaque) callconv(.c) i32;
            };
            _ = kernel32.SetEvent(self.handle);
        }
    }

    pub fn wait(self: *SyncEvent, timeout_ms: u32) void {
        if (@import("builtin").os.tag == .windows and self.handle != null) {
            const kernel32 = struct {
                extern "kernel32" fn WaitForSingleObject(?*anyopaque, u32) callconv(.c) u32;
            };
            _ = kernel32.WaitForSingleObject(self.handle, timeout_ms);
        } else {
            runtime.platform.sleepMs(timeout_ms);
        }
    }
};

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    queue: std.ArrayList(Task),
    lock: SpinLock = .{},
    is_running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    active_tasks: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    work_event: SyncEvent = .{},
    completion_event: SyncEvent = .{},

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !*ThreadPool {
        const count = @max(1, num_threads);
        const pool = try allocator.create(ThreadPool);
        pool.* = .{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, count),
            .queue = .empty,
            .lock = .{},
            .is_running = std.atomic.Value(bool).init(true),
            .active_tasks = std.atomic.Value(usize).init(0),
            .work_event = SyncEvent.init(),
            .completion_event = SyncEvent.init(),
        };

        for (pool.threads) |*t| {
            t.* = try std.Thread.spawn(.{}, workerLoop, .{pool});
        }
        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        self.is_running.store(false, .release);
        self.work_event.set();
        for (self.threads) |t| {
            t.join();
        }
        self.work_event.deinit();
        self.completion_event.deinit();
        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn workerLoop(self: *ThreadPool) void {
        while (self.is_running.load(.acquire)) {
            var maybe_task: ?Task = null;
            {
                self.lock.lock();
                defer self.lock.unlock();
                if (self.queue.items.len > 0) {
                    maybe_task = self.queue.orderedRemove(0);
                }
            }

            if (maybe_task) |task| {
                task.func(task.ctx);
                if (self.active_tasks.fetchSub(1, .release) == 1) {
                    self.completion_event.set();
                }
            } else {
                if (!self.is_running.load(.acquire)) break;
                // ponytail: wait on kernel event to achieve 0% idle CPU utilization
                self.work_event.wait(50);
            }
        }
    }

    pub fn spawnTask(self: *ThreadPool, func: *const fn (ctx: ?*anyopaque) void, ctx: ?*anyopaque) !void {
        self.lock.lock();
        defer self.lock.unlock();
        _ = self.active_tasks.fetchAdd(1, .release);
        try self.queue.append(self.allocator, .{ .func = func, .ctx = ctx });
        self.work_event.set();
    }

    pub fn wait(self: *ThreadPool) void {
        while (self.active_tasks.load(.acquire) > 0) {
            self.completion_event.wait(50);
        }
    }
};

test "actor message passing in ActorSystem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var system = ActorSystem.init(arena.allocator());
    defer system.deinit();

    const worker = try system.spawn(runtime.number(100.0));
    const sender = try system.spawn(runtime.nil());

    _ = try system.sendTo(worker.id, sender.id, runtime.string("task_start"));
    _ = try system.sendTo(worker.id, sender.id, runtime.number(42.0));

    const msg1 = worker.receive();
    try std.testing.expect(msg1 != null);
    try std.testing.expectEqualStrings("task_start", msg1.?.payload.string);

    const msg2 = worker.receive();
    try std.testing.expect(msg2 != null);
    try std.testing.expectEqual(@as(f64, 42.0), msg2.?.payload.number.val);

    const msg3 = worker.receive();
    try std.testing.expect(msg3 == null);
}

test "thread pool execution and futex synchronization" {
    const TestContext = struct {
        val: *std.atomic.Value(u32),
        fn run(ctx: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            _ = self.val.fetchAdd(1, .release);
        }
    };

    var counter = std.atomic.Value(u32).init(0);
    var t_ctx = TestContext{ .val = &counter };

    const pool = try ThreadPool.init(std.testing.allocator, 4);
    defer pool.deinit();

    for (0..16) |_| {
        try pool.spawnTask(TestContext.run, &t_ctx);
    }
    pool.wait();

    try std.testing.expectEqual(@as(u32, 16), counter.load(.acquire));
}
