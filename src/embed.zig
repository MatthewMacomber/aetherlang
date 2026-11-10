//! Aether Embeddable Scripting Engine
//! Provides an in-process VM and C ABI bindings for game engines and host applications.

const std = @import("std");
const parser = @import("parser.zig");
const sweet = @import("sweet.zig");
const macros = @import("macros.zig");
const air = @import("air.zig");
const repl = @import("repl.zig");
pub const runtime = @import("runtime.zig");
pub const Value = runtime.Value;

pub const HostFn = repl.HostFn;

// ============================================================================
// Zig High-Level Embedding API
// ============================================================================

pub const VM = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    env: repl.REPLEnvironment,
    last_error_buf: [512]u8 = undefined,
    last_error_len: usize = 0,

    pub fn init(allocator: std.mem.Allocator) !*VM {
        const vm = try allocator.create(VM);
        vm.* = .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .env = repl.REPLEnvironment.init(allocator),
            .last_error_len = 0,
        };
        return vm;
    }

    pub fn deinit(self: *VM) void {
        self.env.deinit();
        self.arena.deinit();
        const alloc = self.allocator;
        alloc.destroy(self);
    }

    pub fn setLastError(self: *VM, msg: []const u8) void {
        const copy_len = @min(msg.len, self.last_error_buf.len - 1);
        @memcpy(self.last_error_buf[0..copy_len], msg[0..copy_len]);
        self.last_error_buf[copy_len] = 0;
        self.last_error_len = copy_len;
    }

    pub fn getLastError(self: *const VM) ?[:0]const u8 {
        if (self.last_error_len == 0) return null;
        return self.last_error_buf[0..self.last_error_len :0];
    }

    pub fn clearLastError(self: *VM) void {
        self.last_error_len = 0;
        self.last_error_buf[0] = 0;
    }

    /// Evaluates Aether source code (detects Sweet or Canonical S-expression syntax).
    pub fn eval(self: *VM, source: []const u8) !Value {
        self.clearLastError();
        const arena_alloc = self.arena.allocator();

        var canonical_source = source;
        if (sweet.isSweetSyntax(source)) {
            var transpiler = sweet.SweetTranspiler.init(arena_alloc);
            canonical_source = transpiler.toCanonical(source) catch |err| {
                self.setLastError(@errorName(err));
                return err;
            };
        }

        var p = parser.Parser.init(canonical_source, arena_alloc) catch |err| {
            self.setLastError(@errorName(err));
            return err;
        };

        const raw_nodes = p.parseAll() catch |err| {
            self.setLastError(@errorName(err));
            return err;
        };

        var expander = macros.MacroExpander.init(arena_alloc);
        defer expander.deinit();
        const expanded_nodes = expander.expandProgram(raw_nodes) catch |err| {
            self.setLastError(@errorName(err));
            return err;
        };

        var opt = air.AIROptimizer.init(arena_alloc);
        const opt_nodes = opt.optimizeAst(expanded_nodes) catch expanded_nodes;

        var last_val = runtime.nil();
        for (opt_nodes) |node| {
            last_val = self.env.eval(node) catch |err| {
                self.setLastError(@errorName(err));
                return err;
            };
        }
        return last_val;
    }

    /// Loads and evaluates an Aether source script file from disk.
    pub fn loadFile(self: *VM, path: []const u8) !Value {
        if (!runtime.fileExists(runtime.string(path)).boolean) {
            self.setLastError("FileNotFound");
            return error.FileNotFound;
        }
        const s_val = runtime.fileReadStr(runtime.string(path));
        return self.eval(s_val.string);
    }

    /// Calls a script-defined or host-registered function by name.
    pub fn call(self: *VM, fn_name: []const u8, args: []const Value) !Value {
        self.clearLastError();
        return self.env.callFunction(fn_name, args) catch |err| {
            self.setLastError(@errorName(err));
            return err;
        };
    }

    /// Registers a host Zig callback into the Aether scripting environment.
    pub fn registerFn(self: *VM, name: []const u8, func: HostFn, userdata: ?*anyopaque) !void {
        try self.env.registerHostFn(name, func, userdata);
    }

    /// Sets a global variable in the scripting environment.
    pub fn setGlobal(self: *VM, name: []const u8, val: Value) !void {
        self.env.set(name, val);
    }

    /// Retrieves a global variable from the scripting environment.
    pub fn getGlobal(self: *const VM, name: []const u8) ?Value {
        return self.env.get(name);
    }
};

// ============================================================================
// C ABI Types & Value Conversions
// ============================================================================

pub const AeType = enum(i32) {
    nil = 0,
    boolean = 1,
    number = 2,
    string = 3,
    ptr = 4,
    tensor = 5,
    object = 6,
};

pub const AeValue = extern struct {
    type: i32 = 0,
    _pad: i32 = 0,
    num: f64 = 0.0,
    ptr: ?*const anyopaque = null,
    len: usize = 0,
};

pub const AeHostFn = *const fn (argc: usize, argv: ?[*]const AeValue, userdata: ?*anyopaque) callconv(.c) AeValue;

pub fn valueToAe(v: Value) AeValue {
    switch (v) {
        .nil => return .{ .type = @intFromEnum(AeType.nil) },
        .boolean => |b| return .{
            .type = @intFromEnum(AeType.boolean),
            .num = if (b) 1.0 else 0.0,
        },
        .number => |n| return .{
            .type = @intFromEnum(AeType.number),
            .num = n.val,
        },
        .string => |s| return .{
            .type = @intFromEnum(AeType.string),
            .ptr = s.ptr,
            .len = s.len,
        },
        .ptr => |p| return .{
            .type = @intFromEnum(AeType.ptr),
            .ptr = p,
        },
        .tensor => |t| return .{
            .type = @intFromEnum(AeType.tensor),
            .ptr = t.data.ptr,
            .len = t.data.len,
        },
        else => return .{
            .type = @intFromEnum(AeType.object),
        },
    }
}

pub fn aeToValue(v: AeValue) Value {
    const t: AeType = @enumFromInt(v.type);
    switch (t) {
        .nil => return runtime.nil(),
        .boolean => return runtime.boolean(v.num != 0.0),
        .number => return runtime.number(v.num),
        .string => {
            if (v.ptr) |p| {
                const s = @as([*]const u8, @ptrCast(p))[0..v.len];
                return runtime.string(s);
            }
            return runtime.string("");
        },
        .ptr => return runtime.ptr(@constCast(v.ptr)),
        .tensor => {
            if (v.ptr) |p| {
                const slice = @as([*]const f64, @ptrCast(@alignCast(p)))[0..v.len];
                return runtime.tensor(&[_]usize{v.len}, slice);
            }
            return runtime.nil();
        },
        .object => return runtime.nil(),
    }
}

// Trampoline wrapper for C host functions
const CCallbackClosure = struct {
    c_fn: AeHostFn,
    userdata: ?*anyopaque,

    fn invoke(args: []const Value, ud: ?*anyopaque) Value {
        const closure: *CCallbackClosure = @ptrCast(@alignCast(ud.?));
        var static_args: [16]AeValue = undefined;
        var dyn_args: ?[]AeValue = null;
        const slice = if (args.len <= 16)
            static_args[0..args.len]
        else blk: {
            dyn_args = runtime.alloc().alloc(AeValue, args.len) catch return runtime.nil();
            break :blk dyn_args.?;
        };
        defer if (dyn_args) |d| runtime.alloc().free(d);

        for (args, 0..) |a, i| {
            slice[i] = valueToAe(a);
        }

        const res = closure.c_fn(args.len, slice.ptr, closure.userdata);
        return aeToValue(res);
    }
};

// ============================================================================
// Exported C ABI Functions
// ============================================================================

export fn ae_vm_create() callconv(.c) ?*VM {
    return VM.init(std.heap.c_allocator) catch null;
}

export fn ae_vm_destroy(vm: ?*VM) callconv(.c) void {
    if (vm) |v| v.deinit();
}

export fn ae_last_error(vm: ?*const VM) callconv(.c) ?[*:0]const u8 {
    if (vm) |v| {
        if (v.getLastError()) |err| return err.ptr;
    }
    return null;
}

export fn ae_eval(vm: ?*VM, source: ?[*:0]const u8) callconv(.c) AeValue {
    if (vm == null or source == null) return valueToAe(runtime.nil());
    const src_slice = std.mem.span(source.?);
    const val = vm.?.eval(src_slice) catch runtime.nil();
    return valueToAe(val);
}

export fn ae_load_file(vm: ?*VM, path: ?[*:0]const u8) callconv(.c) AeValue {
    if (vm == null or path == null) return valueToAe(runtime.nil());
    const path_slice = std.mem.span(path.?);
    const val = vm.?.loadFile(path_slice) catch runtime.nil();
    return valueToAe(val);
}

export fn ae_call(vm: ?*VM, fn_name: ?[*:0]const u8, argc: usize, argv: ?[*]const AeValue) callconv(.c) AeValue {
    if (vm == null or fn_name == null) return valueToAe(runtime.nil());
    const name_slice = std.mem.span(fn_name.?);

    var static_args: [16]Value = undefined;
    var dyn_args: ?[]Value = null;
    const args_slice = if (argc <= 16)
        static_args[0..argc]
    else blk: {
        dyn_args = vm.?.arena.allocator().alloc(Value, argc) catch return valueToAe(runtime.nil());
        break :blk dyn_args.?;
    };

    if (argv) |arr| {
        for (0..argc) |i| {
            args_slice[i] = aeToValue(arr[i]);
        }
    }

    const val = vm.?.call(name_slice, args_slice) catch runtime.nil();
    return valueToAe(val);
}

export fn ae_set_global(vm: ?*VM, name: ?[*:0]const u8, val: AeValue) callconv(.c) void {
    if (vm == null or name == null) return;
    const name_slice = std.mem.span(name.?);
    vm.?.setGlobal(name_slice, aeToValue(val)) catch {};
}

export fn ae_get_global(vm: ?*VM, name: ?[*:0]const u8) callconv(.c) AeValue {
    if (vm == null or name == null) return valueToAe(runtime.nil());
    const name_slice = std.mem.span(name.?);
    if (vm.?.getGlobal(name_slice)) |val| {
        return valueToAe(val);
    }
    return valueToAe(runtime.nil());
}

export fn ae_register_fn(vm: ?*VM, name: ?[*:0]const u8, func: ?AeHostFn, userdata: ?*anyopaque) callconv(.c) void {
    if (vm == null or name == null or func == null) return;
    const name_slice = std.mem.span(name.?);

    const closure = vm.?.arena.allocator().create(CCallbackClosure) catch return;
    closure.* = .{
        .c_fn = func.?,
        .userdata = userdata,
    };
    vm.?.registerFn(name_slice, CCallbackClosure.invoke, closure) catch {};
}

export fn ae_tensor_create_1d(vm: ?*VM, data: ?[*]const f64, len: usize) callconv(.c) AeValue {
    _ = vm;
    if (data == null or len == 0) return valueToAe(runtime.nil());
    const slice = data.?[0..len];
    const t = runtime.tensor(&[_]usize{len}, slice);
    return valueToAe(t);
}

export fn ae_gc_collect(vm: ?*VM) callconv(.c) void {
    _ = vm;
    runtime.mem.collect();
}

// ============================================================================
// Unit Tests
// ============================================================================

test "embed: Zig VM lifecycle and evaluation" {
    const vm = try VM.init(std.testing.allocator);
    defer vm.deinit();

    // 1. Canonical S-expression eval
    const res1 = try vm.eval("(+ 10 32)");
    try std.testing.expectEqual(@as(f64, 42.0), res1.number.val);

    // 2. Sweet syntax eval
    const sweet_code =
        \\fn get_bonus(score: i64) {
        \\  let bonus = score * 2;
        \\  return bonus;
        \\}
    ;
    _ = try vm.eval(sweet_code);

    // 3. Call script function
    const res2 = try vm.call("get_bonus", &[_]Value{runtime.number(25.0)});
    try std.testing.expectEqual(@as(f64, 50.0), res2.number.val);

    // 4. Set and get globals
    try vm.setGlobal("player_hp", runtime.number(100.0));
    const hp = vm.getGlobal("player_hp");
    try std.testing.expect(hp != null);
    try std.testing.expectEqual(@as(f64, 100.0), hp.?.number.val);

    // 5. Host callback with userdata
    var counter: usize = 0;
    const hostTick = struct {
        fn cb(args: []const Value, ud: ?*anyopaque) Value {
            const ptr: *usize = @ptrCast(@alignCast(ud.?));
            if (args.len >= 1 and args[0] == .number) {
                ptr.* += @intFromFloat(args[0].number.val);
            }
            return runtime.number(@floatFromInt(ptr.*));
        }
    }.cb;

    try vm.registerFn("host_tick", hostTick, &counter);
    const tick_res = try vm.eval("(host_tick 5)");
    try std.testing.expectEqual(@as(f64, 5.0), tick_res.number.val);
    try std.testing.expectEqual(@as(usize, 5), counter);
}

test "embed: C ABI export wrappers" {
    const vm = ae_vm_create() orelse return error.VmCreationFailed;
    defer ae_vm_destroy(vm);

    // Eval C string
    const res = ae_eval(vm, "(let a 15) (+ a 27)");
    try std.testing.expectEqual(@intFromEnum(AeType.number), res.type);
    try std.testing.expectEqual(@as(f64, 42.0), res.num);

    // Set global
    ae_set_global(vm, "gravity", .{ .type = @intFromEnum(AeType.number), .num = 9.81 });
    const g = ae_get_global(vm, "gravity");
    try std.testing.expectEqual(@as(f64, 9.81), g.num);

    // C callback registration
    const cCallback = struct {
        fn cb(argc: usize, argv: ?[*]const AeValue, ud: ?*anyopaque) callconv(.c) AeValue {
            _ = ud;
            if (argc >= 2 and argv != null) {
                const a = argv.?[0].num;
                const b = argv.?[1].num;
                return .{ .type = @intFromEnum(AeType.number), .num = a * b };
            }
            return .{ .type = @intFromEnum(AeType.nil) };
        }
    }.cb;

    ae_register_fn(vm, "multiply", cCallback, null);
    const call_res = ae_eval(vm, "(multiply 6 7)");
    try std.testing.expectEqual(@intFromEnum(AeType.number), call_res.type);
    try std.testing.expectEqual(@as(f64, 42.0), call_res.num);
}
