//! Aether Intermediate Representation (AIR) & Optimization Pipeline
//! Typed SSA IR performing operator fusion (matmul+relu -> fused_matmul_relu, matmul+bias+relu -> fused_dense_relu),
//! dead code elimination (DCE), and compile-time constant folding.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const Op = enum {
    constant,
    add,
    sub,
    mul,
    div,
    matmul,
    relu,
    fused_matmul_relu,
    fused_dense_relu,
    call,
    ret,
};

pub const ValueRef = usize;

pub const Instruction = struct {
    op: Op,
    dest: ValueRef,
    args: [3]ValueRef = .{ 0, 0, 0 },
    const_num: f64 = 0.0,
    symbol_name: []const u8 = "",
    is_dead: bool = false,
};

pub const AIRFunction = struct {
    name: []const u8,
    params: []const []const u8,
    instructions: std.ArrayList(Instruction),
};

pub const AIROptimizer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) AIROptimizer {
        return .{ .allocator = allocator };
    }

    /// Operator fusion pass: fuses matmul+relu and matmul+bias+relu into single instructions
    pub fn passOperatorFusion(self: *AIROptimizer, func: *AIRFunction) void {
        _ = self;
        var i: usize = 0;
        while (i < func.instructions.items.len) : (i += 1) {
            const inst = &func.instructions.items[i];
            if (inst.op == .relu) {
                const src_ref = inst.args[0];
                var fused_3way = false;

                // Check if predecessor produced src_ref via add (bias addition)
                for (func.instructions.items[0..i]) |*prev_add| {
                    if (prev_add.dest == src_ref and prev_add.op == .add and !prev_add.is_dead) {
                        for (func.instructions.items[0..i]) |*prev_mm| {
                            if (!prev_mm.is_dead and prev_mm.op == .matmul) {
                                if (prev_mm.dest == prev_add.args[0]) {
                                    // Matmul is arg 0 of add, bias is arg 1
                                    inst.op = .fused_dense_relu;
                                    inst.args[0] = prev_mm.args[0];
                                    inst.args[1] = prev_mm.args[1];
                                    inst.args[2] = prev_add.args[1];
                                    prev_mm.is_dead = true;
                                    prev_add.is_dead = true;
                                    fused_3way = true;
                                    break;
                                } else if (prev_mm.dest == prev_add.args[1]) {
                                    // Matmul is arg 1 of add, bias is arg 0
                                    inst.op = .fused_dense_relu;
                                    inst.args[0] = prev_mm.args[0];
                                    inst.args[1] = prev_mm.args[1];
                                    inst.args[2] = prev_add.args[0];
                                    prev_mm.is_dead = true;
                                    prev_add.is_dead = true;
                                    fused_3way = true;
                                    break;
                                }
                            }
                        }
                        if (fused_3way) break;
                    }
                }

                if (!fused_3way) {
                    // Check if predecessor produced src_ref directly via matmul
                    for (func.instructions.items[0..i]) |*prev| {
                        if (prev.dest == src_ref and prev.op == .matmul and !prev.is_dead) {
                            // Fuse matmul + relu!
                            inst.op = .fused_matmul_relu;
                            inst.args[0] = prev.args[0];
                            inst.args[1] = prev.args[1];
                            prev.is_dead = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Dead Code Elimination (DCE): marks unused instructions as dead
    pub fn passDeadCodeElimination(self: *AIROptimizer, func: *AIRFunction) !void {
        var used = std.AutoHashMap(ValueRef, void).init(self.allocator);
        defer used.deinit();

        // Mark roots (ret, call, etc.)
        for (func.instructions.items) |inst| {
            if (inst.op == .ret or inst.op == .call) {
                try used.put(inst.args[0], {});
                try used.put(inst.args[1], {});
                try used.put(inst.args[2], {});
            }
        }

        // Propagate backwards
        var changed = true;
        while (changed) {
            changed = false;
            for (func.instructions.items) |inst| {
                if (used.contains(inst.dest) and !inst.is_dead) {
                    for (inst.args) |arg| {
                        if (!used.contains(arg)) {
                            try used.put(arg, {});
                            changed = true;
                        }
                    }
                }
            }
        }

        // Eliminate dead instructions
        for (func.instructions.items) |*inst| {
            if (inst.op != .ret and inst.op != .call and !used.contains(inst.dest)) {
                inst.is_dead = true;
            }
        }
    }

    /// Constant folding: evaluates arithmetic on constants
    pub fn passConstantFolding(self: *AIROptimizer, func: *AIRFunction) void {
        var const_vals = std.AutoHashMap(ValueRef, f64).init(self.allocator);
        defer const_vals.deinit();

        for (func.instructions.items) |*inst| {
            if (inst.is_dead) continue;
            if (inst.op == .constant) {
                const_vals.put(inst.dest, inst.const_num) catch {};
                continue;
            }

            const c1 = const_vals.get(inst.args[0]);
            const c2 = const_vals.get(inst.args[1]);
            if (c1 != null and c2 != null) {
                const v1 = c1.?;
                const v2 = c2.?;
                const res = switch (inst.op) {
                    .add => v1 + v2,
                    .sub => v1 - v2,
                    .mul => v1 * v2,
                    .div => if (v2 != 0) v1 / v2 else v1,
                    else => null,
                };
                if (res) |folded| {
                    inst.op = .constant;
                    inst.const_num = folded;
                    const_vals.put(inst.dest, folded) catch {};
                }
            }
        }
    }

    /// AST-level optimization pass: fuses matmul+relu and constant-folds arithmetic
    pub fn optimizeAst(self: *AIROptimizer, nodes: []const *Node) anyerror![]const *Node {
        var result: std.ArrayList(*Node) = .empty;
        for (nodes) |node| {
            try result.append(self.allocator, try self.optimizeNode(node));
        }
        return try result.toOwnedSlice(self.allocator);
    }

    fn optimizeNode(self: *AIROptimizer, node: *Node) anyerror!*Node {
        switch (node.data) {
            .atom => return node,
            .bracket_list => |children| {
                var new_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try new_children.append(self.allocator, try self.optimizeNode(child));
                }
                const n = try self.allocator.create(Node);
                n.* = Node{
                    .loc = node.loc,
                    .data = .{ .bracket_list = try new_children.toOwnedSlice(self.allocator) },
                };
                return n;
            },
            .list => |children| {
                if (children.len == 0) return node;

                var opt_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try opt_children.append(self.allocator, try self.optimizeNode(child));
                }
                const new_children = try opt_children.toOwnedSlice(self.allocator);

                // 1. Operator fusion
                if (new_children.len >= 2 and new_children[0].data == .atom and new_children[0].data.atom == .symbol and std.mem.eql(u8, new_children[0].data.atom.symbol, "relu")) {
                    const arg = new_children[1];

                    // 1a. 3-way fusion: (relu (+ (matmul a b) bias)) -> (fused_dense_relu a b bias)
                    if (arg.data == .list and arg.data.list.len >= 3 and arg.data.list[0].data == .atom and arg.data.list[0].data.atom == .symbol and std.mem.eql(u8, arg.data.list[0].data.atom.symbol, "+")) {
                        const op1 = arg.data.list[1];
                        const op2 = arg.data.list[2];
                        var mm_node: ?*Node = null;
                        var bias_node: ?*Node = null;
                        if (op1.data == .list and op1.data.list.len >= 3 and op1.data.list[0].data == .atom and std.mem.eql(u8, op1.data.list[0].data.atom.symbol, "matmul")) {
                            mm_node = op1;
                            bias_node = op2;
                        } else if (op2.data == .list and op2.data.list.len >= 3 and op2.data.list[0].data == .atom and std.mem.eql(u8, op2.data.list[0].data.atom.symbol, "matmul")) {
                            mm_node = op2;
                            bias_node = op1;
                        }
                        if (mm_node != null and bias_node != null) {
                            const fused_sym = try self.allocator.create(Node);
                            fused_sym.* = Node{
                                .loc = node.loc,
                                .data = .{ .atom = .{ .symbol = "fused_dense_relu" } },
                            };
                            var fused_list = try self.allocator.alloc(*Node, 4);
                            fused_list[0] = fused_sym;
                            fused_list[1] = mm_node.?.data.list[1];
                            fused_list[2] = mm_node.?.data.list[2];
                            fused_list[3] = bias_node.?;
                            const fused_node = try self.allocator.create(Node);
                            fused_node.* = Node{
                                .loc = node.loc,
                                .data = .{ .list = fused_list },
                            };
                            return fused_node;
                        }
                    }

                    // 1b. 2-way fusion: (relu (matmul a b)) -> (fused_matmul_relu a b)
                    if (arg.data == .list and arg.data.list.len >= 3 and arg.data.list[0].data == .atom and arg.data.list[0].data.atom == .symbol and std.mem.eql(u8, arg.data.list[0].data.atom.symbol, "matmul")) {
                        const fused_sym = try self.allocator.create(Node);
                        fused_sym.* = Node{
                            .loc = node.loc,
                            .data = .{ .atom = .{ .symbol = "fused_matmul_relu" } },
                        };
                        var fused_list = try self.allocator.alloc(*Node, 3);
                        fused_list[0] = fused_sym;
                        fused_list[1] = arg.data.list[1];
                        fused_list[2] = arg.data.list[2];

                        const fused_node = try self.allocator.create(Node);
                        fused_node.* = Node{
                            .loc = node.loc,
                            .data = .{ .list = fused_list },
                        };
                        return fused_node;
                    }
                }

                // 2. Constant folding: (+ 10 20) -> 30, (* 2 3) -> 6
                if (new_children.len >= 3 and new_children[0].data == .atom and new_children[0].data.atom == .symbol) {
                    const op = new_children[0].data.atom.symbol;
                    const a1 = new_children[1];
                    const a2 = new_children[2];
                    if (a1.data == .atom and a1.data.atom == .number and a2.data == .atom and a2.data.atom == .number) {
                        const n1 = a1.data.atom.number;
                        const n2 = a2.data.atom.number;
                        var folded: ?f64 = null;
                        if (std.mem.eql(u8, op, "+")) folded = n1 + n2
                        else if (std.mem.eql(u8, op, "-")) folded = n1 - n2
                        else if (std.mem.eql(u8, op, "*")) folded = n1 * n2
                        else if (std.mem.eql(u8, op, "/") and n2 != 0.0) folded = n1 / n2;

                        if (folded) |val| {
                            const const_node = try self.allocator.create(Node);
                            const const_node_val = Node{
                                .loc = node.loc,
                                .data = .{ .atom = .{ .number = val } },
                            };
                            const_node.* = const_node_val;
                            return const_node;
                        }
                    }
                }

                const n = try self.allocator.create(Node);
                n.* = Node{
                    .loc = node.loc,
                    .data = .{ .list = new_children },
                };
                return n;
            },
        }
    }
};

/// Human-readable SSA AIR printer
pub fn printAIR(func: *const AIRFunction, allocator: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(allocator, "function @");
    try out.appendSlice(allocator, func.name);
    try out.appendSlice(allocator, "(");
    for (func.params, 0..) |p, idx| {
        if (idx > 0) try out.appendSlice(allocator, ", ");
        try out.appendSlice(allocator, "%");
        try out.appendSlice(allocator, p);
    }
    try out.appendSlice(allocator, ") {\n");
    for (func.instructions.items) |inst| {
        if (inst.is_dead) {
            const line = try std.fmt.allocPrint(allocator, "  ; dead %{d} = {s}\n", .{ inst.dest, @tagName(inst.op) });
            try out.appendSlice(allocator, line);
            continue;
        }
        const dest_prefix = try std.fmt.allocPrint(allocator, "  %{d} = ", .{inst.dest});
        try out.appendSlice(allocator, dest_prefix);
        switch (inst.op) {
            .constant => {
                const s = try std.fmt.allocPrint(allocator, "const {d}\n", .{inst.const_num});
                try out.appendSlice(allocator, s);
            },
            .add => {
                const s = try std.fmt.allocPrint(allocator, "add %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .sub => {
                const s = try std.fmt.allocPrint(allocator, "sub %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .mul => {
                const s = try std.fmt.allocPrint(allocator, "mul %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .div => {
                const s = try std.fmt.allocPrint(allocator, "div %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .matmul => {
                const s = try std.fmt.allocPrint(allocator, "matmul %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .relu => {
                const s = try std.fmt.allocPrint(allocator, "relu %{d}\n", .{inst.args[0]});
                try out.appendSlice(allocator, s);
            },
            .fused_matmul_relu => {
                const s = try std.fmt.allocPrint(allocator, "fused_matmul_relu %{d}, %{d}\n", .{ inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .fused_dense_relu => {
                const s = try std.fmt.allocPrint(allocator, "fused_dense_relu %{d}, %{d}, %{d}\n", .{ inst.args[0], inst.args[1], inst.args[2] });
                try out.appendSlice(allocator, s);
            },
            .call => {
                const s = try std.fmt.allocPrint(allocator, "call @{s}(%{d}, %{d})\n", .{ inst.symbol_name, inst.args[0], inst.args[1] });
                try out.appendSlice(allocator, s);
            },
            .ret => {
                const s = try std.fmt.allocPrint(allocator, "ret %{d}\n", .{inst.args[0]});
                try out.appendSlice(allocator, s);
            },
        }
    }
    try out.appendSlice(allocator, "}\n");
}

pub fn formatAIR(func: *const AIRFunction, allocator: std.mem.Allocator) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try printAIR(func, allocator, &out);
    return try out.toOwnedSlice(allocator);
}

/// Helper to lower AST expression into AIR SSA instructions
fn emitNodeAIR(node: *Node, allocator: std.mem.Allocator, insts: *std.ArrayList(Instruction), syms: *std.StringHashMap(ValueRef), next_ref: *ValueRef) anyerror!ValueRef {
    switch (node.data) {
        .atom => |a| switch (a) {
            .number => |n| {
                const dest = next_ref.*;
                next_ref.* += 1;
                try insts.append(allocator, .{ .op = .constant, .dest = dest, .const_num = n });
                return dest;
            },
            .symbol => |s| {
                if (syms.get(s)) |ref| return ref;
                const dest = next_ref.*;
                next_ref.* += 1;
                try syms.put(s, dest);
                return dest;
            },
            else => {
                const dest = next_ref.*;
                next_ref.* += 1;
                try insts.append(allocator, .{ .op = .constant, .dest = dest, .const_num = 0.0 });
                return dest;
            },
        },
        .bracket_list => {
            const dest = next_ref.*;
            next_ref.* += 1;
            try insts.append(allocator, .{ .op = .constant, .dest = dest, .const_num = 0.0 });
            return dest;
        },
        .list => |children| {
            if (children.len == 0) return 0;
            const head = children[0];
            if (head.data != .atom or head.data.atom != .symbol) return 0;
            const op_sym = head.data.atom.symbol;

            if (std.mem.eql(u8, op_sym, "let") and children.len >= 3) {
                var var_name: []const u8 = "";
                if (children[1].data == .atom and children[1].data.atom == .symbol) {
                    var_name = children[1].data.atom.symbol;
                } else if (children[1].data == .list and children[1].data.list.len > 0 and children[1].data.list[0].data == .atom) {
                    var_name = children[1].data.list[0].data.atom.symbol;
                }
                const val_ref = try emitNodeAIR(children[2], allocator, insts, syms, next_ref);
                if (var_name.len > 0) try syms.put(var_name, val_ref);
                return val_ref;
            } else if (std.mem.eql(u8, op_sym, "return") or std.mem.eql(u8, op_sym, "ret")) {
                const arg = if (children.len >= 2) try emitNodeAIR(children[1], allocator, insts, syms, next_ref) else 0;
                const dest = next_ref.*;
                next_ref.* += 1;
                try insts.append(allocator, .{ .op = .ret, .dest = dest, .args = .{ arg, 0, 0 } });
                return dest;
            } else if (std.mem.eql(u8, op_sym, "+") or std.mem.eql(u8, op_sym, "-") or std.mem.eql(u8, op_sym, "*") or std.mem.eql(u8, op_sym, "/") or std.mem.eql(u8, op_sym, "matmul")) {
                const a1 = if (children.len >= 2) try emitNodeAIR(children[1], allocator, insts, syms, next_ref) else 0;
                const a2 = if (children.len >= 3) try emitNodeAIR(children[2], allocator, insts, syms, next_ref) else 0;
                const dest = next_ref.*;
                next_ref.* += 1;
                const air_op: Op = if (std.mem.eql(u8, op_sym, "+"))
                    .add
                else if (std.mem.eql(u8, op_sym, "-"))
                    .sub
                else if (std.mem.eql(u8, op_sym, "*"))
                    .mul
                else if (std.mem.eql(u8, op_sym, "/"))
                    .div
                else
                    .matmul;
                try insts.append(allocator, .{ .op = air_op, .dest = dest, .args = .{ a1, a2, 0 } });
                return dest;
            } else if (std.mem.eql(u8, op_sym, "relu") and children.len >= 2) {
                const a1 = try emitNodeAIR(children[1], allocator, insts, syms, next_ref);
                const dest = next_ref.*;
                next_ref.* += 1;
                try insts.append(allocator, .{ .op = .relu, .dest = dest, .args = .{ a1, 0, 0 } });
                return dest;
            } else if (std.mem.eql(u8, op_sym, "call") and children.len >= 2) {
                const target = if (children[1].data == .atom and children[1].data.atom == .symbol) children[1].data.atom.symbol else "func";
                const a1 = if (children.len >= 3) try emitNodeAIR(children[2], allocator, insts, syms, next_ref) else 0;
                const a2 = if (children.len >= 4) try emitNodeAIR(children[3], allocator, insts, syms, next_ref) else 0;
                const dest = next_ref.*;
                next_ref.* += 1;
                try insts.append(allocator, .{ .op = .call, .dest = dest, .args = .{ a1, a2, 0 }, .symbol_name = target });
                return dest;
            } else {
                var last_ref: ValueRef = 0;
                for (children[1..]) |child| {
                    last_ref = try emitNodeAIR(child, allocator, insts, syms, next_ref);
                }
                return last_ref;
            }
        },
    }
}

/// Builds an AIRFunction from an AST `(func ...)` node
pub fn buildAIR(node: *Node, allocator: std.mem.Allocator) !?AIRFunction {
    if (node.data != .list or node.data.list.len < 3) return null;
    const head = node.data.list[0];
    if (head.data != .atom or head.data.atom != .symbol or !std.mem.eql(u8, head.data.atom.symbol, "func")) return null;

    const fn_name = node.data.list[1].data.atom.symbol;
    const params_node = node.data.list[2];
    const params_list = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

    var param_names: std.ArrayList([]const u8) = .empty;
    var symbol_map = std.StringHashMap(ValueRef).init(allocator);
    defer symbol_map.deinit();

    var next_ref: ValueRef = 0;
    for (params_list) |p| {
        const p_name = if (p.data == .atom and p.data.atom == .symbol)
            p.data.atom.symbol
        else if (p.data == .list and p.data.list.len > 0 and p.data.list[0].data == .atom and p.data.list[0].data.atom == .symbol)
            p.data.list[0].data.atom.symbol
        else
            continue;
        try param_names.append(allocator, p_name);
        try symbol_map.put(p_name, next_ref);
        next_ref += 1;
    }

    var instructions: std.ArrayList(Instruction) = .empty;
    var body_idx: usize = 3;
    if (node.data.list.len > 3 and node.data.list[3].data == .atom and node.data.list[3].data.atom == .symbol) {
        const s = node.data.list[3].data.atom.symbol;
        if (std.mem.eql(u8, s, "i32") or std.mem.eql(u8, s, "i64") or std.mem.eql(u8, s, "f64") or std.mem.eql(u8, s, "void") or std.mem.eql(u8, s, "tensor") or std.mem.eql(u8, s, "bool") or std.mem.eql(u8, s, "string")) {
            body_idx = 4;
        }
    }

    var last_ref: ValueRef = 0;
    for (node.data.list[body_idx..]) |stmt| {
        last_ref = try emitNodeAIR(stmt, allocator, &instructions, &symbol_map, &next_ref);
    }

    // If last instruction wasn't a ret, append ret
    if (instructions.items.len == 0 or instructions.items[instructions.items.len - 1].op != .ret) {
        const ret_ref = next_ref;
        next_ref += 1;
        try instructions.append(allocator, .{ .op = .ret, .dest = ret_ref, .args = .{ last_ref, 0, 0 } });
    }

    return AIRFunction{
        .name = fn_name,
        .params = try param_names.toOwnedSlice(allocator),
        .instructions = instructions,
    };
}

/// Builds AIR functions for all functions in the program
pub fn buildProgramAIR(nodes: []const *Node, allocator: std.mem.Allocator) !std.ArrayList(AIRFunction) {
    var funcs: std.ArrayList(AIRFunction) = .empty;
    var top_level_nodes: std.ArrayList(*Node) = .empty;

    for (nodes) |node| {
        if (try buildAIR(node, allocator)) |f| {
            try funcs.append(allocator, f);
        } else {
            try top_level_nodes.append(allocator, node);
        }
    }

    // If there were top-level script statements, wrap them into a synthetic @main function
    if (top_level_nodes.items.len > 0) {
        var symbol_map = std.StringHashMap(ValueRef).init(allocator);
        defer symbol_map.deinit();
        var instructions: std.ArrayList(Instruction) = .empty;
        var next_ref: ValueRef = 0;
        var last_ref: ValueRef = 0;
        for (top_level_nodes.items) |stmt| {
            last_ref = try emitNodeAIR(stmt, allocator, &instructions, &symbol_map, &next_ref);
        }
        if (instructions.items.len == 0 or instructions.items[instructions.items.len - 1].op != .ret) {
            const ret_ref = next_ref;
            next_ref += 1;
            try instructions.append(allocator, .{ .op = .ret, .dest = ret_ref, .args = .{ last_ref, 0, 0 } });
        }
        try funcs.append(allocator, .{
            .name = "main",
            .params = &[_][]const u8{},
            .instructions = instructions,
        });
    }

    return funcs;
}

test "AIR operator fusion fuses matmul and relu" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var func = AIRFunction{
        .name = "forward_layer",
        .params = &[_][]const u8{"x"},
        .instructions = .empty,
    };

    // %0 = const W
    try func.instructions.append(arena.allocator(), .{ .op = .constant, .dest = 0, .const_num = 1.0 });
    // %1 = matmul x, %0
    try func.instructions.append(arena.allocator(), .{ .op = .matmul, .dest = 1, .args = .{ 99, 0, 0 } });
    // %2 = relu %1
    try func.instructions.append(arena.allocator(), .{ .op = .relu, .dest = 2, .args = .{ 1, 0, 0 } });
    // ret %2
    try func.instructions.append(arena.allocator(), .{ .op = .ret, .dest = 3, .args = .{ 2, 0, 0 } });

    var opt = AIROptimizer.init(arena.allocator());
    opt.passOperatorFusion(&func);

    // After fusion:
    try std.testing.expect(func.instructions.items[1].is_dead);
    try std.testing.expectEqual(Op.fused_matmul_relu, func.instructions.items[2].op);
    try std.testing.expectEqual(@as(usize, 99), func.instructions.items[2].args[0]);
    try std.testing.expectEqual(@as(usize, 0), func.instructions.items[2].args[1]);
}

test "AIR 3-way operator fusion fuses matmul + bias add + relu" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var func = AIRFunction{
        .name = "dense_layer",
        .params = &[_][]const u8{ "x", "w", "b" },
        .instructions = .empty,
    };

    // %0 = matmul x, w
    try func.instructions.append(arena.allocator(), .{ .op = .matmul, .dest = 3, .args = .{ 0, 1, 0 } });
    // %1 = add %0, b
    try func.instructions.append(arena.allocator(), .{ .op = .add, .dest = 4, .args = .{ 3, 2, 0 } });
    // %2 = relu %1
    try func.instructions.append(arena.allocator(), .{ .op = .relu, .dest = 5, .args = .{ 4, 0, 0 } });
    // ret %2
    try func.instructions.append(arena.allocator(), .{ .op = .ret, .dest = 6, .args = .{ 5, 0, 0 } });

    var opt = AIROptimizer.init(arena.allocator());
    opt.passOperatorFusion(&func);

    // After 3-way fusion:
    try std.testing.expect(func.instructions.items[0].is_dead);
    try std.testing.expect(func.instructions.items[1].is_dead);
    try std.testing.expectEqual(Op.fused_dense_relu, func.instructions.items[2].op);
    try std.testing.expectEqual(@as(usize, 0), func.instructions.items[2].args[0]); // x
    try std.testing.expectEqual(@as(usize, 1), func.instructions.items[2].args[1]); // w
    try std.testing.expectEqual(@as(usize, 2), func.instructions.items[2].args[2]); // b
}

test "AIR constant folding evaluates at compile-time" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var func = AIRFunction{
        .name = "calc",
        .params = &[_][]const u8{},
        .instructions = .empty,
    };

    // %0 = 10.0
    try func.instructions.append(arena.allocator(), .{ .op = .constant, .dest = 0, .const_num = 10.0 });
    // %1 = 20.0
    try func.instructions.append(arena.allocator(), .{ .op = .constant, .dest = 1, .const_num = 20.0 });
    // %2 = %0 + %1
    try func.instructions.append(arena.allocator(), .{ .op = .add, .dest = 2, .args = .{ 0, 1, 0 } });
    // ret %2
    try func.instructions.append(arena.allocator(), .{ .op = .ret, .dest = 3, .args = .{ 2, 0, 0 } });

    var opt = AIROptimizer.init(arena.allocator());
    opt.passConstantFolding(&func);

    try std.testing.expectEqual(Op.constant, func.instructions.items[2].op);
    try std.testing.expectEqual(@as(f64, 30.0), func.instructions.items[2].const_num);
}

test "AIR AST optimization fuses matmul and relu and folds constants" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(let h (relu (+ (matmul x w) b)))
        \\(let c (+ 15 25))
    ;
    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var opt = AIROptimizer.init(arena.allocator());
    const opt_nodes = try opt.optimizeAst(nodes);

    // Node 0: (let h (fused_dense_relu x w b))
    const let_h = opt_nodes[0].data.list;
    try std.testing.expectEqualStrings("let", let_h[0].data.atom.symbol);
    const fused_expr = let_h[2].data.list;
    try std.testing.expectEqualStrings("fused_dense_relu", fused_expr[0].data.atom.symbol);

    // Node 1: (let c 40)
    const let_c = opt_nodes[1].data.list;
    try std.testing.expectEqualStrings("let", let_c[0].data.atom.symbol);
    try std.testing.expectEqual(@as(f64, 40.0), let_c[2].data.atom.number);
}
