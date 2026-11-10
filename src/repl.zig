//! Aether Interactive REPL & Embedded AST Evaluator
//! Provides immediate S-expression evaluation, collections, functions,
//! control flow, and host application callback integration.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;
const runtime = @import("runtime.zig");
const Value = runtime.Value;

pub const HostFn = *const fn (args: []const Value, userdata: ?*anyopaque) Value;

pub const HostCallback = struct {
    func: HostFn,
    userdata: ?*anyopaque = null,
};

pub const REPLEnvironment = struct {
    allocator: std.mem.Allocator,
    parent: ?*REPLEnvironment = null,
    bindings: std.StringHashMap(Value),
    functions: std.StringHashMap(*Node),
    host_functions: std.StringHashMap(HostCallback),
    returned: bool = false,
    return_val: ?Value = null,

    pub fn init(allocator: std.mem.Allocator) REPLEnvironment {
        return .{
            .allocator = allocator,
            .parent = null,
            .bindings = std.StringHashMap(Value).init(allocator),
            .functions = std.StringHashMap(*Node).init(allocator),
            .host_functions = std.StringHashMap(HostCallback).init(allocator),
        };
    }

    pub fn initChild(parent: *REPLEnvironment) REPLEnvironment {
        return .{
            .allocator = parent.allocator,
            .parent = parent,
            .bindings = std.StringHashMap(Value).init(parent.allocator),
            .functions = std.StringHashMap(*Node).init(parent.allocator),
            .host_functions = std.StringHashMap(HostCallback).init(parent.allocator),
        };
    }

    pub fn deinit(self: *REPLEnvironment) void {
        self.bindings.deinit();
        self.functions.deinit();
        self.host_functions.deinit();
    }

    pub fn get(self: *const REPLEnvironment, name: []const u8) ?Value {
        if (self.bindings.get(name)) |v| return v;
        if (self.parent) |p| return p.get(name);
        return null;
    }

    pub fn set(self: *REPLEnvironment, name: []const u8, val: Value) void {
        if (self.bindings.contains(name)) {
            self.bindings.put(name, val) catch {};
            return;
        }
        if (self.parent) |p| {
            if (p.hasBinding(name)) {
                p.set(name, val);
                return;
            }
        }
        self.bindings.put(name, val) catch {};
    }

    pub fn hasBinding(self: *const REPLEnvironment, name: []const u8) bool {
        if (self.bindings.contains(name)) return true;
        if (self.parent) |p| return p.hasBinding(name);
        return false;
    }

    pub fn getFunction(self: *const REPLEnvironment, name: []const u8) ?*Node {
        if (self.functions.get(name)) |f| return f;
        if (self.parent) |p| return p.getFunction(name);
        return null;
    }

    pub fn getHostFunction(self: *const REPLEnvironment, name: []const u8) ?HostCallback {
        if (self.host_functions.get(name)) |cb| return cb;
        if (self.parent) |p| return p.getHostFunction(name);
        return null;
    }

    pub fn registerHostFn(self: *REPLEnvironment, name: []const u8, func: HostFn, userdata: ?*anyopaque) !void {
        try self.host_functions.put(name, .{ .func = func, .userdata = userdata });
    }

    pub fn callFunction(self: *REPLEnvironment, fn_name: []const u8, args: []const Value) anyerror!Value {
        if (self.getFunction(fn_name)) |fn_node| {
            if (fn_node.data != .list or fn_node.data.list.len < 3) return runtime.nil();
            const params_node = fn_node.data.list[2];
            const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

            var sub_env = REPLEnvironment.initChild(self);
            defer sub_env.deinit();

            const num_args = @min(params.len, args.len);
            for (0..num_args) |idx| {
                const p = params[idx];
                const p_name = if (p.data == .atom and p.data.atom == .symbol)
                    p.data.atom.symbol
                else if (p.data == .list and p.data.list.len > 0 and p.data.list[0].data == .atom and p.data.list[0].data.atom == .symbol)
                    p.data.list[0].data.atom.symbol
                else
                    continue;
                try sub_env.bindings.put(p_name, args[idx]);
            }

            var body_start: usize = 3;
            if (fn_node.data.list.len > 3 and fn_node.data.list[3].data == .atom and fn_node.data.list[3].data.atom == .symbol) {
                const s = fn_node.data.list[3].data.atom.symbol;
                if (std.mem.eql(u8, s, "i32") or std.mem.eql(u8, s, "i64") or std.mem.eql(u8, s, "int") or
                    std.mem.eql(u8, s, "f32") or std.mem.eql(u8, s, "f64") or std.mem.eql(u8, s, "float") or
                    std.mem.eql(u8, s, "bool") or std.mem.eql(u8, s, "string") or std.mem.eql(u8, s, "void") or
                    std.mem.eql(u8, s, "tensor"))
                {
                    body_start = 4;
                }
            }

            var result = runtime.nil();
            for (fn_node.data.list[body_start..]) |stmt| {
                result = try sub_env.eval(stmt);
                if (sub_env.returned) {
                    return sub_env.return_val orelse runtime.nil();
                }
            }
            return result;
        }

        if (self.getHostFunction(fn_name)) |cb| {
            return cb.func(args, cb.userdata);
        }

        return runtime.nil();
    }

    pub fn eval(self: *REPLEnvironment, node: *Node) anyerror!Value {
        switch (node.data) {
            .atom => |a| switch (a) {
                .nil => return runtime.nil(),
                .boolean => |b| return runtime.boolean(b),
                .number => |n| return runtime.number(n),
                .string => |s| return runtime.string(s),
                .symbol => |s| {
                    if (self.get(s)) |v| return v;
                    return runtime.nil();
                },
            },
            .bracket_list => |children| {
                var elems: std.ArrayList(Value) = .empty;
                defer elems.deinit(self.allocator);
                for (children) |c| {
                    try elems.append(self.allocator, try self.eval(c));
                }
                return runtime.list(elems.items);
            },
            .list => |children| {
                if (children.len == 0) return runtime.nil();
                const head = children[0];
                if (head.data != .atom or head.data.atom != .symbol) return runtime.nil();
                const op = head.data.atom.symbol;

                // Function definition: (func name (params...) [ret_type] body...)
                if (std.mem.eql(u8, op, "func") and children.len >= 3) {
                    if (children[1].data == .atom and children[1].data.atom == .symbol) {
                        const fn_name = children[1].data.atom.symbol;
                        // Store in root environment if we are a child
                        var target_env = self;
                        while (target_env.parent) |p| target_env = p;
                        try target_env.functions.put(fn_name, node);
                        return runtime.string(fn_name);
                    }
                }

                // Variable declaration: (let name expr)
                if (std.mem.eql(u8, op, "let") and children.len >= 3) {
                    var var_name: []const u8 = "";
                    if (children[1].data == .atom and children[1].data.atom == .symbol) {
                        var_name = children[1].data.atom.symbol;
                    } else if (children[1].data == .list and children[1].data.list.len > 0 and children[1].data.list[0].data == .atom and children[1].data.list[0].data.atom == .symbol) {
                        var_name = children[1].data.list[0].data.atom.symbol;
                    }
                    const val = try self.eval(children[2]);
                    if (var_name.len > 0) try self.bindings.put(var_name, val);
                    return val;
                }

                // Assignment: (set name expr) or (set col idx val)
                if ((std.mem.eql(u8, op, "set") or std.mem.eql(u8, op, "set!"))) {
                    if (children.len >= 4) {
                        // (set col idx val) -> listSet / mapSet
                        const col = try self.eval(children[1]);
                        const idx = try self.eval(children[2]);
                        const val = try self.eval(children[3]);
                        return runtime.listSet(col, idx, val);
                    } else if (children.len >= 3) {
                        var var_name: []const u8 = "";
                        if (children[1].data == .atom and children[1].data.atom == .symbol) {
                            var_name = children[1].data.atom.symbol;
                        }
                        const val = try self.eval(children[2]);
                        if (var_name.len > 0) self.set(var_name, val);
                        return val;
                    }
                }

                // Return statement: (return [expr]) or (ret [expr])
                if (std.mem.eql(u8, op, "return") or std.mem.eql(u8, op, "ret")) {
                    const val = if (children.len >= 2) try self.eval(children[1]) else runtime.nil();
                    self.returned = true;
                    self.return_val = val;
                    return val;
                }

                // Sequential block: (do ...), (block ...), (begin ...)
                if (std.mem.eql(u8, op, "do") or std.mem.eql(u8, op, "block") or std.mem.eql(u8, op, "begin") or std.mem.eql(u8, op, "arena")) {
                    var last_val = runtime.nil();
                    for (children[1..]) |stmt| {
                        last_val = try self.eval(stmt);
                        if (self.returned) return self.return_val.?;
                    }
                    return last_val;
                }

                // Conditional: (if cond then [else])
                if (std.mem.eql(u8, op, "if") and children.len >= 3) {
                    const cond_val = try self.eval(children[1]);
                    if (runtime.isTruthy(cond_val)) {
                        return try self.eval(children[2]);
                    } else if (children.len >= 4) {
                        return try self.eval(children[3]);
                    }
                    return runtime.nil();
                }

                // While loop: (while cond body...)
                if (std.mem.eql(u8, op, "while") and children.len >= 2) {
                    var last_val = runtime.nil();
                    while (runtime.isTruthy(try self.eval(children[1]))) {
                        for (children[2..]) |stmt| {
                            last_val = try self.eval(stmt);
                            if (self.returned) return self.return_val.?;
                        }
                    }
                    return last_val;
                }

                // For loop: (for var (range ...) body...) or (for var limit body...)
                if (std.mem.eql(u8, op, "for") and children.len >= 4) {
                    const var_node = children[1];
                    const var_name = if (var_node.data == .atom and var_node.data.atom == .symbol) var_node.data.atom.symbol else "";
                    const iter_node = children[2];

                    var start: f64 = 0.0;
                    var limit: f64 = 0.0;
                    var step: f64 = 1.0;

                    if (iter_node.data == .list and iter_node.data.list.len >= 2 and iter_node.data.list[0].data == .atom and std.mem.eql(u8, iter_node.data.list[0].data.atom.symbol, "range")) {
                        const r_args = iter_node.data.list[1..];
                        if (r_args.len == 1) {
                            limit = runtime.toF64(try self.eval(r_args[0]));
                        } else if (r_args.len >= 2) {
                            start = runtime.toF64(try self.eval(r_args[0]));
                            limit = runtime.toF64(try self.eval(r_args[1]));
                            if (r_args.len >= 3) {
                                step = runtime.toF64(try self.eval(r_args[2]));
                            }
                        }
                    } else {
                        limit = runtime.toF64(try self.eval(iter_node));
                    }

                    var cur = start;
                    var last_val = runtime.nil();
                    while (cur < limit) : (cur += step) {
                        if (var_name.len > 0) {
                            try self.bindings.put(var_name, runtime.number(cur));
                        }
                        for (children[3..]) |stmt| {
                            last_val = try self.eval(stmt);
                            if (self.returned) return self.return_val.?;
                        }
                    }
                    return last_val;
                }

                // Pattern matching: (match target (pat1 body...) (pat2 body...) ...)
                if (std.mem.eql(u8, op, "match") and children.len >= 3) {
                    const target_val = try self.eval(children[1]);
                    for (children[2..]) |case_node| {
                        if (case_node.data != .list or case_node.data.list.len < 2) continue;
                        const case_parts = case_node.data.list;
                        const pat = case_parts[0];
                        const is_wildcard = (pat.data == .atom and pat.data.atom == .symbol and std.mem.eql(u8, pat.data.atom.symbol, "_"));
                        var matched = is_wildcard;
                        if (!matched) {
                            const pat_val = try self.eval(pat);
                            matched = runtime.eq(target_val, pat_val).boolean;
                        }
                        if (matched) {
                            var case_res = runtime.nil();
                            for (case_parts[1..]) |stmt| {
                                case_res = try self.eval(stmt);
                                if (self.returned) return self.return_val.?;
                            }
                            return case_res;
                        }
                    }
                    return runtime.nil();
                }

                // Field access: (.field obj)
                if (std.mem.startsWith(u8, op, ".") and op.len > 1 and children.len >= 2) {
                    const field_name = op[1..];
                    const obj = try self.eval(children[1]);
                    return runtime.listGet(obj, runtime.string(field_name));
                }

                // Arithmetic & Comparison Builtins
                if (std.mem.eql(u8, op, "+") and children.len >= 3) {
                    return runtime.add(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "-") and children.len >= 3) {
                    return runtime.sub(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "*") and children.len >= 3) {
                    return runtime.mul(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "/") and children.len >= 3) {
                    return runtime.div(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "%") or std.mem.eql(u8, op, "mod")) and children.len >= 3) {
                    return runtime.mod(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "<") and children.len >= 3) {
                    return runtime.lt(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "<=") and children.len >= 3) {
                    return runtime.le(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, ">") and children.len >= 3) {
                    return runtime.gt(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, ">=") and children.len >= 3) {
                    return runtime.ge(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "==") or std.mem.eql(u8, op, "=")) and children.len >= 3) {
                    return runtime.eq(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "!=") and children.len >= 3) {
                    return runtime.ne(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "and") and children.len >= 3) {
                    return runtime.logicAnd(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "or") and children.len >= 3) {
                    return runtime.logicOr(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "not") and children.len >= 2) {
                    return runtime.logicNot(try self.eval(children[1]));
                }

                // Math & Trigonometry Builtins
                if (std.mem.eql(u8, op, "sqrt") and children.len >= 2) {
                    return runtime.mathSqrt(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "abs") and children.len >= 2) {
                    return runtime.mathAbs(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "sin") and children.len >= 2) {
                    return runtime.mathSin(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "cos") and children.len >= 2) {
                    return runtime.mathCos(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "tan") and children.len >= 2) {
                    return runtime.mathTan(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "asin") and children.len >= 2) {
                    return runtime.mathAsin(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "acos") and children.len >= 2) {
                    return runtime.mathAcos(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "atan2") and children.len >= 3) {
                    return runtime.mathAtan2(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "exp") and children.len >= 2) {
                    return runtime.mathExp(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "log") and children.len >= 2) {
                    return runtime.mathLog(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "log2") and children.len >= 2) {
                    return runtime.mathLog2(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "log10") and children.len >= 2) {
                    return runtime.mathLog10(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "pow") and children.len >= 3) {
                    return runtime.mathPow(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "min") and children.len >= 3) {
                    return runtime.mathMin(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "max") and children.len >= 3) {
                    return runtime.mathMax(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "clamp") and children.len >= 4) {
                    return runtime.mathClamp(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                } else if (std.mem.eql(u8, op, "floor") and children.len >= 2) {
                    return runtime.mathFloor(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "ceil") and children.len >= 2) {
                    return runtime.mathCeil(try self.eval(children[1]));
                }

                // 3D Vectors & Geometry Builtins
                if (std.mem.eql(u8, op, "vec3") and children.len >= 4) {
                    return runtime.vec3(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                } else if (std.mem.eql(u8, op, "dot") and children.len >= 3) {
                    return runtime.vec3Dot(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "cross") and children.len >= 3) {
                    return runtime.vec3Cross(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "norm") and children.len >= 2) {
                    return runtime.vec3Norm(try self.eval(children[1]));
                }

                // Physics Collision Queries
                if ((std.mem.eql(u8, op, "phys_check_aabb") or std.mem.eql(u8, op, "phys.check_aabb")) and children.len >= 9) {
                    return runtime.physCheckAABB(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]), try self.eval(children[4]), try self.eval(children[5]), try self.eval(children[6]), try self.eval(children[7]), try self.eval(children[8]));
                } else if ((std.mem.eql(u8, op, "phys_check_circle") or std.mem.eql(u8, op, "phys.check_circle")) and children.len >= 7) {
                    return runtime.physCheckCircle(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]), try self.eval(children[4]), try self.eval(children[5]), try self.eval(children[6]));
                } else if ((std.mem.eql(u8, op, "phys_raycast_aabb") or std.mem.eql(u8, op, "phys.raycast_aabb")) and children.len >= 9) {
                    return runtime.physRaycastAABB(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]), try self.eval(children[4]), try self.eval(children[5]), try self.eval(children[6]), try self.eval(children[7]), try self.eval(children[8]));
                }

                // Dynamic Collections Builtins
                if (std.mem.eql(u8, op, "list")) {
                    var items: std.ArrayList(Value) = .empty;
                    defer items.deinit(self.allocator);
                    for (children[1..]) |arg| {
                        try items.append(self.allocator, try self.eval(arg));
                    }
                    return runtime.list(items.items);
                } else if (std.mem.eql(u8, op, "append") and children.len >= 3) {
                    return runtime.listAppend(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "concat") or std.mem.eql(u8, op, "list_concat")) and children.len >= 3) {
                    return runtime.listConcat(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "get") or std.mem.eql(u8, op, "index")) and children.len >= 3) {
                    return runtime.listGet(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "set_at") or std.mem.eql(u8, op, "list_set")) and children.len >= 4) {
                    return runtime.listSet(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                } else if ((std.mem.eql(u8, op, "len") or std.mem.eql(u8, op, "length")) and children.len >= 2) {
                    return runtime.listLength(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "slice") and children.len >= 4) {
                    return runtime.listSlice(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                } else if (std.mem.eql(u8, op, "pop") and children.len >= 2) {
                    return runtime.listPop(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "map") or std.mem.eql(u8, op, "dict")) {
                    return runtime.mapCreate();
                } else if (std.mem.eql(u8, op, "map_set") and children.len >= 4) {
                    return runtime.mapSet(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                } else if (std.mem.eql(u8, op, "map_get") and children.len >= 3) {
                    return runtime.mapGet(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "map_has") and children.len >= 3) {
                    return runtime.mapHas(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "map_delete") and children.len >= 3) {
                    return runtime.mapDelete(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "keys") or std.mem.eql(u8, op, "map_keys")) and children.len >= 2) {
                    return runtime.mapKeys(try self.eval(children[1]));
                } else if ((std.mem.eql(u8, op, "values") or std.mem.eql(u8, op, "map_values")) and children.len >= 2) {
                    return runtime.mapValues(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "set") and children.len == 1) {
                    return runtime.setCreate();
                } else if (std.mem.eql(u8, op, "set_add") and children.len >= 3) {
                    return runtime.setAdd(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "set_contains") and children.len >= 3) {
                    return runtime.setContains(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "deque") and children.len == 1) {
                    return runtime.dequeCreate();
                }

                // String Utilities
                if (std.mem.eql(u8, op, "str_split") and children.len >= 3) {
                    return runtime.strSplit(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "str_join") and children.len >= 3) {
                    return runtime.strJoin(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "str_trim") and children.len >= 2) {
                    return runtime.strTrim(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "str_contains") and children.len >= 3) {
                    return runtime.strContains(try self.eval(children[1]), try self.eval(children[2]));
                } else if ((std.mem.eql(u8, op, "str") or std.mem.eql(u8, op, "to_str") or std.mem.eql(u8, op, "to_string")) and children.len >= 2) {
                    return runtime.valToString(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "str_to_num") and children.len >= 2) {
                    return runtime.strToNum(try self.eval(children[1]));
                }

                // Tensor & AI Builtins
                if (std.mem.eql(u8, op, "matmul") and children.len >= 3) {
                    return runtime.matmul(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "relu") and children.len >= 2) {
                    return runtime.relu(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "sigmoid") and children.len >= 2) {
                    return runtime.sigmoid(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "tanh") and children.len >= 2) {
                    return runtime.tanh(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "softmax") and children.len >= 2) {
                    return runtime.tensorSoftmax(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "argmax") and children.len >= 2) {
                    return runtime.tensorArgmax(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "transpose") and children.len >= 2) {
                    return runtime.tensorTranspose(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "sum") and children.len >= 2) {
                    return runtime.sum(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "mse_loss") and children.len >= 3) {
                    return runtime.mse_loss(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "cross_entropy") and children.len >= 3) {
                    return runtime.cross_entropy(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "backward") and children.len >= 2) {
                    return runtime.tensorBackward(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "grad") and children.len >= 2) {
                    return runtime.tensorGrad(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "clear_grads")) {
                    return runtime.tensorClearGrads();
                } else if (std.mem.eql(u8, op, "sgd_step") and children.len >= 4) {
                    return runtime.optimizerSGD(try self.eval(children[1]), try self.eval(children[2]), try self.eval(children[3]));
                }

                // Probabilistic Builtins
                if (std.mem.eql(u8, op, "uncertain") and children.len >= 3) {
                    return runtime.uncertain(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "Normal") and children.len >= 3) {
                    return runtime.distNormal(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "Uniform") and children.len >= 3) {
                    return runtime.distUniform(try self.eval(children[1]), try self.eval(children[2]));
                } else if (std.mem.eql(u8, op, "Bernoulli") and children.len >= 2) {
                    return runtime.distBernoulli(try self.eval(children[1]));
                } else if (std.mem.eql(u8, op, "sample") and children.len >= 2) {
                    return runtime.sample(try self.eval(children[1]));
                }

                // Output Builtin: print
                if (std.mem.eql(u8, op, "print") or (std.mem.eql(u8, op, "call") and children.len >= 2 and children[1].data == .atom and std.mem.eql(u8, children[1].data.atom.symbol, "print"))) {
                    const arg_start: usize = if (std.mem.eql(u8, op, "print")) 1 else 2;
                    var printed_args: std.ArrayList(Value) = .empty;
                    defer printed_args.deinit(self.allocator);
                    for (children[arg_start..]) |arg_node| {
                        try printed_args.append(self.allocator, try self.eval(arg_node));
                    }
                    if (runtime.g_io) |io| {
                        return runtime.print(io, printed_args.items);
                    } else {
                        return runtime.nil();
                    }
                }

                // Explicit Host or User Function Call: (call fn_name args...) or (fn_name args...)
                var target_fn: ?[]const u8 = null;
                var call_args_start: usize = 1;
                if (std.mem.eql(u8, op, "call") and children.len >= 2) {
                    if (children[1].data == .atom and children[1].data.atom == .symbol) {
                        target_fn = children[1].data.atom.symbol;
                        call_args_start = 2;
                    }
                } else if (self.getFunction(op) != null or self.getHostFunction(op) != null) {
                    target_fn = op;
                    call_args_start = 1;
                }

                if (target_fn) |fn_name| {
                    var eval_args: std.ArrayList(Value) = .empty;
                    defer eval_args.deinit(self.allocator);
                    for (children[call_args_start..]) |arg_node| {
                        try eval_args.append(self.allocator, try self.eval(arg_node));
                    }
                    return try self.callFunction(fn_name, eval_args.items);
                }

                return runtime.nil();
            },
        }
    }
};

test "REPL evaluator calculates expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var env = REPLEnvironment.init(arena.allocator());
    defer env.deinit();

    var p1 = try parser.Parser.init("(let x (* 6 7))", arena.allocator());
    const n1 = try p1.parseAll();
    const v1 = try env.eval(n1[0]);
    try std.testing.expectEqual(@as(f64, 42.0), v1.number.val);

    var p2 = try parser.Parser.init("(+ x 8)", arena.allocator());
    const n2 = try p2.parseAll();
    const v2 = try env.eval(n2[0]);
    try std.testing.expectEqual(@as(f64, 50.0), v2.number.val);

    var p3 = try parser.Parser.init("(/ x 6)", arena.allocator());
    const n3 = try p3.parseAll();
    const v3 = try env.eval(n3[0]);
    try std.testing.expectEqual(@as(f64, 7.0), v3.number.val);

    var p4 = try parser.Parser.init("(sqrt 49)", arena.allocator());
    const n4 = try p4.parseAll();
    const v4 = try env.eval(n4[0]);
    try std.testing.expectEqual(@as(f64, 7.0), v4.number.val);

    var p5 = try parser.Parser.init("(< 5 10)", arena.allocator());
    const n5 = try p5.parseAll();
    const v5 = try env.eval(n5[0]);
    try std.testing.expect(v5.boolean);
}

test "REPL evaluator supports collections and user functions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var env = REPLEnvironment.init(arena.allocator());
    defer env.deinit();

    // 1. List operations
    var p1 = try parser.Parser.init("(let xs (list 10 20 30))", arena.allocator());
    const n1 = try p1.parseAll();
    _ = try env.eval(n1[0]);

    var p2 = try parser.Parser.init("(len xs)", arena.allocator());
    const n2 = try p2.parseAll();
    const v2 = try env.eval(n2[0]);
    try std.testing.expectEqual(@as(f64, 3.0), v2.number.val);

    // 2. User-defined function
    var p3 = try parser.Parser.init("(func square (n) (return (* n n)))", arena.allocator());
    const n3 = try p3.parseAll();
    _ = try env.eval(n3[0]);

    var p4 = try parser.Parser.init("(call square 9)", arena.allocator());
    const n4 = try p4.parseAll();
    const v4 = try env.eval(n4[0]);
    try std.testing.expectEqual(@as(f64, 81.0), v4.number.val);

    // 3. String utility
    var p5 = try parser.Parser.init("(str_contains \"hello world\" \"world\")", arena.allocator());
    const n5 = try p5.parseAll();
    const v5 = try env.eval(n5[0]);
    try std.testing.expect(v5.boolean);
}

test "REPL evaluator control flow: if, while, for, match, set" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var env = REPLEnvironment.init(arena.allocator());
    defer env.deinit();

    // While loop and set
    var p1 = try parser.Parser.init(
        \\(do
        \\  (let sum 0)
        \\  (let i 1)
        \\  (while (<= i 5)
        \\    (set sum (+ sum i))
        \\    (set i (+ i 1)))
        \\  sum)
    , arena.allocator());
    const n1 = try p1.parseAll();
    const v1 = try env.eval(n1[0]);
    try std.testing.expectEqual(@as(f64, 15.0), v1.number.val);

    // Match statement
    var p2 = try parser.Parser.init(
        \\(match 2
        \\  (1 "one")
        \\  (2 "two")
        \\  (_ "other"))
    , arena.allocator());
    const n2 = try p2.parseAll();
    const v2 = try env.eval(n2[0]);
    try std.testing.expectEqualStrings("two", v2.string);

    // Host function callback
    const hostAdd = struct {
        fn cb(args: []const Value, ud: ?*anyopaque) Value {
            _ = ud;
            if (args.len >= 2 and args[0] == .number and args[1] == .number) {
                return runtime.number(args[0].number.val + args[1].number.val * 2.0);
            }
            return runtime.nil();
        }
    }.cb;

    try env.registerHostFn("host_calc", hostAdd, null);
    var p3 = try parser.Parser.init("(host_calc 10 20)", arena.allocator());
    const n3 = try p3.parseAll();
    const v3 = try env.eval(n3[0]);
    try std.testing.expectEqual(@as(f64, 50.0), v3.number.val);
}
