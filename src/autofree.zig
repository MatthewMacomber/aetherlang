//! Aether Static Interprocedural Escape Analysis for Autofree
//! Identifies local allocations that can be safely and deterministically deallocated at scope exit.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const AutoFreeResult = struct {
    allocator: std.mem.Allocator,
    lets_to_free: std.AutoHashMapUnmanaged(*Node, void) = .empty,
    has_autodiff_backprop: bool = false,

    pub fn deinit(self: *AutoFreeResult) void {
        self.lets_to_free.deinit(self.allocator);
    }

    pub fn shouldFree(self: *const AutoFreeResult, node: *Node) bool {
        return self.lets_to_free.contains(node);
    }
};

const FRESH_BUILTINS = [_][]const u8{
    "list",
    "map",
    "set",
    "deque",
    "concat",
    "slice",
    "str_split",
    "strSplit",
    "str_join",
    "strJoin",
    "str_trim",
    "strTrim",
    "to_str",
    "file_read_str",
    "fileReadStr",
    "map_keys",
    "mapKeys",
    "map_values",
    "mapValues",
    "set_to_list",
    "setToList",
    "transpose",
    "tensor_transpose",
};

const TENSOR_FRESH_OPS = [_][]const u8{
    "tensor",
    "matmul",
    "relu",
    "sigmoid",
    "tanh",
    "mse_loss",
    "cross_entropy",
    "quantize",
    "fused_matmul_relu",
    "fused_dense_relu",
};

const BORROW_BUILTINS = [_][]const u8{
    "print",
    "length",
    "get",
    "map_get",
    "mapGet",
    "map_has",
    "mapHas",
    "map_size",
    "mapSize",
    "set_contains",
    "setContains",
    "set_size",
    "setSize",
    "deque_len",
    "deque_get",
    "str_contains",
    "strContains",
    "str_join",
    "strJoin",
    "str_split",
    "strSplit",
    "concat",
    "slice",
    "to_str",
    "toStr",
    "dot",
    "cross",
    "norm",
    "file_write_str",
    "fileWriteStr",
    "sum",
    "mean",
    "variance",
    "stddev",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "and",
    "or",
    "not",
    "typeof",
    "is_truthy",
    "assert",
};

const MUTATE_BUILTINS = [_][]const u8{
    "append",
    "list_append",
    "set_at",
    "map_set",
    "mapSet",
    "map_delete",
    "mapDelete",
    "set_add",
    "setAdd",
    "set_remove",
    "setRemove",
    "push_back",
    "push_front",
    "pop",
    "pop_back",
    "pop_front",
};

fn isBuiltin(name: []const u8, list: []const []const u8) bool {
    for (list) |b| {
        if (std.mem.eql(u8, name, b)) return true;
    }
    return false;
}

pub fn analyze(allocator: std.mem.Allocator, ast_nodes: []const *Node) !AutoFreeResult {
    var res = AutoFreeResult{ .allocator = allocator };

    // Pass 1: Check if program calls backward or grad (preserves tensor tape nodes for GC)
    for (ast_nodes) |node| {
        if (hasBackpropCall(node)) {
            res.has_autodiff_backprop = true;
            break;
        }
    }

    // Pass 2: Collect function return-freshness and parameter-escape summaries
    var fn_names = std.StringHashMap(usize).init(allocator);
    defer fn_names.deinit();

    var fn_returns_fresh: std.ArrayList(bool) = .empty;
    defer fn_returns_fresh.deinit(allocator);

    for (ast_nodes) |node| {
        if (node.data == .list and node.data.list.len >= 3) {
            const sym = getSymbol(node.data.list[0]) orelse continue;
            if (std.mem.eql(u8, sym, "func")) {
                const fname = getSymbol(node.data.list[1]) orelse continue;
                const idx = fn_returns_fresh.items.len;
                try fn_names.put(fname, idx);
                try fn_returns_fresh.append(allocator, false);
            }
        }
    }

    // Iterative fixpoint for returns_fresh
    var changed = true;
    while (changed) {
        changed = false;
        for (ast_nodes) |node| {
            if (node.data == .list and node.data.list.len >= 3) {
                const sym = getSymbol(node.data.list[0]) orelse continue;
                if (std.mem.eql(u8, sym, "func")) {
                    const fname = getSymbol(node.data.list[1]) orelse continue;
                    const idx = fn_names.get(fname) orelse continue;
                    const prev = fn_returns_fresh.items[idx];
                    const now = computeReturnsFresh(node.data.list, &fn_names, fn_returns_fresh.items, res.has_autodiff_backprop);
                    if (now != prev) {
                        fn_returns_fresh.items[idx] = now;
                        changed = true;
                    }
                }
            }
        }
    }

    // Pass 3: For each function and scope, analyze lets
    for (ast_nodes) |node| {
        if (node.data == .list and node.data.list.len >= 3) {
            const sym = getSymbol(node.data.list[0]) orelse continue;
            if (std.mem.eql(u8, sym, "func")) {
                try analyzeFunctionBody(allocator, node.data.list, &res, &fn_names, fn_returns_fresh.items);
            }
        }
    }

    return res;
}

fn hasBackpropCall(node: *Node) bool {
    switch (node.data) {
        .atom => |a| {
            if (a == .symbol) {
                const s = a.symbol;
                if (std.mem.eql(u8, s, "backward") or std.mem.eql(u8, s, "tensorBackward") or
                    std.mem.eql(u8, s, "grad") or std.mem.eql(u8, s, "tensorGrad"))
                {
                    return true;
                }
            }
            return false;
        },
        .list => |children| {
            for (children) |c| {
                if (hasBackpropCall(c)) return true;
            }
            return false;
        },
        .bracket_list => |children| {
            for (children) |c| {
                if (hasBackpropCall(c)) return true;
            }
            return false;
        },
    }
}

fn getSymbol(node: *Node) ?[]const u8 {
    if (node.data == .atom and node.data.atom == .symbol) {
        return node.data.atom.symbol;
    }
    return null;
}

fn isFreshExpr(node: *Node, fn_names: *const std.StringHashMap(usize), returns_fresh: []const bool, has_backprop: bool) bool {
    if (node.data != .list or node.data.list.len == 0) return false;
    const parts = node.data.list;

    // Check (call target args...)
    var call_target: ?[]const u8 = null;
    if (parts[0].data == .atom and parts[0].data.atom == .symbol) {
        const s = parts[0].data.atom.symbol;
        if (std.mem.eql(u8, s, "call") and parts.len >= 2) {
            call_target = getSymbol(parts[1]);
        } else {
            call_target = s;
        }
    }

    if (call_target) |tgt| {
        if (isBuiltin(tgt, &FRESH_BUILTINS)) return true;
        if (!has_backprop and isBuiltin(tgt, &TENSOR_FRESH_OPS)) return true;
        if (fn_names.get(tgt)) |idx| {
            return returns_fresh[idx];
        }
    }
    return false;
}

fn computeReturnsFresh(func_parts: []const *Node, fn_names: *const std.StringHashMap(usize), returns_fresh: []const bool, has_backprop: bool) bool {
    // func name params [ret] body...
    var body_start: usize = 3;
    if (func_parts.len > 3 and getSymbol(func_parts[3]) != null) {
        const sym = getSymbol(func_parts[3]).?;
        if (std.mem.eql(u8, sym, "void") or std.mem.eql(u8, sym, "i32") or std.mem.eql(u8, sym, "i64") or
            std.mem.eql(u8, sym, "f32") or std.mem.eql(u8, sym, "f64") or std.mem.eql(u8, sym, "bool") or
            std.mem.eql(u8, sym, "string"))
        {
            body_start = 4;
        }
    }

    // Find all return expressions
    var found_return = false;
    var all_fresh = true;

    for (func_parts[body_start..]) |stmt| {
        checkReturnFreshness(stmt, &found_return, &all_fresh, fn_names, returns_fresh, has_backprop);
    }

    return found_return and all_fresh;
}

fn checkReturnFreshness(node: *Node, found_return: *bool, all_fresh: *bool, fn_names: *const std.StringHashMap(usize), returns_fresh: []const bool, has_backprop: bool) void {
    if (node.data != .list or node.data.list.len == 0) return;
    const parts = node.data.list;
    const op = getSymbol(parts[0]) orelse return;

    if (std.mem.eql(u8, op, "return")) {
        found_return.* = true;
        if (parts.len >= 2) {
            if (!isFreshExpr(parts[1], fn_names, returns_fresh, has_backprop)) {
                all_fresh.* = false;
            }
        } else {
            all_fresh.* = false;
        }
        return;
    }

    for (parts[1..]) |child| {
        checkReturnFreshness(child, found_return, all_fresh, fn_names, returns_fresh, has_backprop);
    }
}

fn analyzeFunctionBody(
    allocator: std.mem.Allocator,
    func_parts: []const *Node,
    res: *AutoFreeResult,
    fn_names: *const std.StringHashMap(usize),
    returns_fresh: []const bool,
) !void {
    var body_start: usize = 3;
    if (func_parts.len > 3 and getSymbol(func_parts[3]) != null) {
        const sym = getSymbol(func_parts[3]).?;
        if (std.mem.eql(u8, sym, "void") or std.mem.eql(u8, sym, "i32") or std.mem.eql(u8, sym, "i64") or
            std.mem.eql(u8, sym, "f32") or std.mem.eql(u8, sym, "f64") or std.mem.eql(u8, sym, "bool") or
            std.mem.eql(u8, sym, "string"))
        {
            body_start = 4;
        }
    }

    const stmts = func_parts[body_start..];
    for (stmts) |stmt| {
        try analyzeStatement(allocator, stmt, stmts, res, fn_names, returns_fresh);
    }
}

fn analyzeStatement(
    allocator: std.mem.Allocator,
    stmt: *Node,
    scope_stmts: []const *Node,
    res: *AutoFreeResult,
    fn_names: *const std.StringHashMap(usize),
    returns_fresh: []const bool,
) !void {
    if (stmt.data != .list or stmt.data.list.len == 0) return;
    const parts = stmt.data.list;
    const op = getSymbol(parts[0]) orelse return;

    if (std.mem.eql(u8, op, "let") and parts.len >= 3) {
        const var_name = getSymbol(parts[1]) orelse return;
        const init_expr = parts[2];

        // Is initialization fresh?
        if (isFreshExpr(init_expr, fn_names, returns_fresh, res.has_autodiff_backprop)) {
            // Check if var_name escapes anywhere in scope_stmts
            if (!escapesInScope(var_name, scope_stmts)) {
                try res.lets_to_free.put(allocator, stmt, {});
            }
        }
    }

    if (std.mem.eql(u8, op, "block") or std.mem.eql(u8, op, "begin") or std.mem.eql(u8, op, "do")) {
        for (parts[1..]) |s| {
            try analyzeStatement(allocator, s, parts[1..], res, fn_names, returns_fresh);
        }
    } else if (std.mem.eql(u8, op, "while") and parts.len >= 3) {
        for (parts[2..]) |s| {
            try analyzeStatement(allocator, s, parts[2..], res, fn_names, returns_fresh);
        }
    } else if (std.mem.eql(u8, op, "for") and parts.len >= 4) {
        for (parts[3..]) |s| {
            try analyzeStatement(allocator, s, parts[3..], res, fn_names, returns_fresh);
        }
    }
}

fn escapesInScope(var_name: []const u8, scope_stmts: []const *Node) bool {
    for (scope_stmts) |stmt| {
        if (checkEscapes(var_name, stmt, false)) {
            return true;
        }
    }
    return false;
}

fn checkEscapes(var_name: []const u8, node: *Node, in_escaping_context: bool) bool {
    switch (node.data) {
        .atom => |a| {
            if (a == .symbol and std.mem.eql(u8, a.symbol, var_name)) {
                return in_escaping_context;
            }
            return false;
        },
        .list => |parts| {
            if (parts.len == 0) return false;
            const op = getSymbol(parts[0]) orelse "";

            if (std.mem.eql(u8, op, "return")) {
                // Return of var_name -> escapes!
                for (parts[1..]) |arg| {
                    if (containsSymbol(arg, var_name)) return true;
                }
                return false;
            }

            if (std.mem.eql(u8, op, "set") or std.mem.eql(u8, op, "set!")) {
                if (parts.len >= 2 and getSymbol(parts[1]) != null and std.mem.eql(u8, getSymbol(parts[1]).?, var_name)) {
                    // Reassignment of the variable itself
                    return true;
                }
                if (parts.len >= 3) {
                    if (parts[2].data == .atom and parts[2].data.atom == .symbol and std.mem.eql(u8, parts[2].data.atom.symbol, var_name)) {
                        return true;
                    }
                    return checkEscapes(var_name, parts[2], in_escaping_context);
                }
                return false;
            }

            if (std.mem.eql(u8, op, "let")) {
                if (parts.len >= 3) {
                    // If this is the declaration of var_name itself, its init expression does not escape var_name
                    if (getSymbol(parts[1])) |decl_sym| {
                        if (std.mem.eql(u8, decl_sym, var_name)) {
                            return false;
                        }
                    }
                    if (parts[2].data == .atom and parts[2].data.atom == .symbol and std.mem.eql(u8, parts[2].data.atom.symbol, var_name)) {
                        return true;
                    }
                    return checkEscapes(var_name, parts[2], in_escaping_context);
                }
                return false;
            }

            if (std.mem.eql(u8, op, "lambda")) {
                if (containsSymbol(node, var_name)) return true; // Captured by closure
                return false;
            }

            // Check calls
            var call_target: []const u8 = op;
            var args = parts[1..];
            if (std.mem.eql(u8, op, "call") and parts.len >= 2) {
                call_target = getSymbol(parts[1]) orelse "";
                args = parts[2..];
            }

            if (isBuiltin(call_target, &BORROW_BUILTINS)) {
                // Safe borrow arguments
                for (args) |arg| {
                    if (checkEscapes(var_name, arg, false)) return true;
                }
                return false;
            }

            if (isBuiltin(call_target, &MUTATE_BUILTINS)) {
                if (args.len >= 1) {
                    // Arg 0 is mutated in place -> non-escaping for var_name
                    if (checkEscapes(var_name, args[0], false)) return true;
                    // Any subsequent arguments (items stored into container) -> ESCAPES!
                    for (args[1..]) |arg| {
                        if (containsSymbol(arg, var_name)) return true;
                    }
                }
                return false;
            }

            // Other calls: default conservative (assume arguments might escape)
            for (args) |arg| {
                if (containsSymbol(arg, var_name)) return true;
            }
            return false;
        },
        .bracket_list => |parts| {
            for (parts) |c| {
                if (checkEscapes(var_name, c, in_escaping_context)) return true;
            }
            return false;
        },
    }
}

fn containsSymbol(node: *Node, sym: []const u8) bool {
    switch (node.data) {
        .atom => |a| return a == .symbol and std.mem.eql(u8, a.symbol, sym),
        .list => |children| {
            for (children) |c| {
                if (containsSymbol(c, sym)) return true;
            }
            return false;
        },
        .bracket_list => |children| {
            for (children) |c| {
                if (containsSymbol(c, sym)) return true;
            }
            return false;
        },
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "autofree: identifies non-escaping fresh list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func test_fn ()
        \\  (let l (list 1 2 3))
        \\  (call print l)
        \\  (return 0))
    ;
    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var analysis = try analyze(arena.allocator(), nodes);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(usize, 1), analysis.lets_to_free.count());
}

test "autofree: does not free escaping returned list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func test_fn ()
        \\  (let l (list 1 2 3))
        \\  (return l))
    ;
    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var analysis = try analyze(arena.allocator(), nodes);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(usize, 0), analysis.lets_to_free.count());
}

test "autofree: does not free list stored into outer collection" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func test_fn (outer)
        \\  (let inner (list 42))
        \\  (append outer inner)
        \\  (return 0))
    ;
    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var analysis = try analyze(arena.allocator(), nodes);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(usize, 0), analysis.lets_to_free.count());
}
