//! Aether Hygienic Macro Expander & Metaprogramming Engine
//! Transforms homoiconic AST lists via compile-time macro expansion,
//! parameter substitution, and quote preservation.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const MacroDef = struct {
    name: []const u8,
    params: []const []const u8,
    body: []const *Node,
};

pub const MacroExpander = struct {
    allocator: std.mem.Allocator,
    macros: std.StringHashMap(MacroDef),

    pub fn init(allocator: std.mem.Allocator) MacroExpander {
        return .{
            .allocator = allocator,
            .macros = std.StringHashMap(MacroDef).init(allocator),
        };
    }

    pub fn deinit(self: *MacroExpander) void {
        self.macros.deinit();
    }

    pub fn registerIfMacro(self: *MacroExpander, node: *Node) !bool {
        if (node.data != .list or node.data.list.len < 4) return false;
        const children = node.data.list;
        if (children[0].data != .atom or children[0].data.atom != .symbol) return false;
        if (!std.mem.eql(u8, children[0].data.atom.symbol, "macro")) return false;

        if (children[1].data != .atom or children[1].data.atom != .symbol) return false;
        const macro_name = children[1].data.atom.symbol;

        // Parse params
        var param_names: std.ArrayList([]const u8) = .empty;
        if (children[2].data == .list or children[2].data == .bracket_list) {
            const plist = if (children[2].data == .list) children[2].data.list else children[2].data.bracket_list;
            for (plist) |p| {
                if (p.data == .atom and p.data.atom == .symbol) {
                    try param_names.append(self.allocator, p.data.atom.symbol);
                } else if (p.data == .list and p.data.list.len >= 1 and p.data.list[0].data == .atom and p.data.list[0].data.atom == .symbol) {
                    // Typed param: (name type)
                    try param_names.append(self.allocator, p.data.list[0].data.atom.symbol);
                }
            }
        }

        try self.macros.put(macro_name, .{
            .name = macro_name,
            .params = try param_names.toOwnedSlice(self.allocator),
            .body = children[3..],
        });
        return true;
    }

    pub fn expandProgram(self: *MacroExpander, nodes: []const *Node) ![]const *Node {
        var non_macro_nodes: std.ArrayList(*Node) = .empty;
        // First pass: register all macros
        for (nodes) |node| {
            const is_macro = try self.registerIfMacro(node);
            if (!is_macro) {
                try non_macro_nodes.append(self.allocator, node);
            }
        }

        // Second pass: recursively expand macros across all ASTs
        var expanded_nodes: std.ArrayList(*Node) = .empty;
        for (non_macro_nodes.items) |node| {
            const expanded = try self.expandNode(node);
            try expanded_nodes.append(self.allocator, expanded);
        }

        return try expanded_nodes.toOwnedSlice(self.allocator);
    }

    pub fn expandNode(self: *MacroExpander, node: *Node) anyerror!*Node {
        switch (node.data) {
            .atom => return node,
            .bracket_list => |children| {
                var new_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try new_children.append(self.allocator, try self.expandNode(child));
                }
                const new_node = try self.allocator.create(Node);
                new_node.loc = node.loc;
                new_node.data = .{ .bracket_list = try new_children.toOwnedSlice(self.allocator) };
                return new_node;
            },
            .list => |children| {
                if (children.len == 0) return node;

                // Check quote: (quote x)
                if (children[0].data == .atom and children[0].data.atom == .symbol and std.mem.eql(u8, children[0].data.atom.symbol, "quote")) {
                    return node; // quoted expressions are not macro-expanded
                }

                // Check macro call
                if (children[0].data == .atom and children[0].data.atom == .symbol) {
                    const sym = children[0].data.atom.symbol;
                    if (self.macros.get(sym)) |m| {
                        const args = children[1..];
                        var env = std.StringHashMap(*Node).init(self.allocator);
                        defer env.deinit();

                        for (m.params, 0..) |param_name, i| {
                            if (i < args.len) {
                                try env.put(param_name, args[i]);
                            }
                        }

                        // Substitute into macro body
                        var substituted_stmts: std.ArrayList(*Node) = .empty;
                        for (m.body) |b| {
                            const sub = try self.substitute(b, &env);
                            try substituted_stmts.append(self.allocator, sub);
                        }

                        if (substituted_stmts.items.len == 1) {
                            return try self.expandNode(substituted_stmts.items[0]);
                        } else {
                            // Wrap multi-statement expansion into a do-block
                            const do_sym = try self.allocator.create(Node);
                            do_sym.loc = node.loc;
                            do_sym.data = .{ .atom = .{ .symbol = "do" } };

                            var do_items: std.ArrayList(*Node) = .empty;
                            try do_items.append(self.allocator, do_sym);
                            for (substituted_stmts.items) |stmt| {
                                try do_items.append(self.allocator, stmt);
                            }

                            const do_node = try self.allocator.create(Node);
                            do_node.loc = node.loc;
                            do_node.data = .{ .list = try do_items.toOwnedSlice(self.allocator) };
                            return try self.expandNode(do_node);
                        }
                    }
                }

                // Normal list: expand all children recursively
                var new_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try new_children.append(self.allocator, try self.expandNode(child));
                }
                const new_node = try self.allocator.create(Node);
                new_node.loc = node.loc;
                new_node.data = .{ .list = try new_children.toOwnedSlice(self.allocator) };
                return new_node;
            },
        }
    }

    fn substitute(self: *MacroExpander, node: *Node, env: *std.StringHashMap(*Node)) anyerror!*Node {
        switch (node.data) {
            .atom => |a| switch (a) {
                .symbol => |s| {
                    if (env.get(s)) |arg_val| {
                        return arg_val;
                    }
                    return node;
                },
                else => return node,
            },
            .bracket_list => |children| {
                var new_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try new_children.append(self.allocator, try self.substitute(child, env));
                }
                const new_node = try self.allocator.create(Node);
                new_node.loc = node.loc;
                new_node.data = .{ .bracket_list = try new_children.toOwnedSlice(self.allocator) };
                return new_node;
            },
            .list => |children| {
                var new_children: std.ArrayList(*Node) = .empty;
                for (children) |child| {
                    try new_children.append(self.allocator, try self.substitute(child, env));
                }
                const new_node = try self.allocator.create(Node);
                new_node.loc = node.loc;
                new_node.data = .{ .list = try new_children.toOwnedSlice(self.allocator) };
                return new_node;
            },
        }
    }
};

test "macro expansion substitutes arguments" {
    const src =
        \\(macro square (x) (* x x))
        \\(func main ()
        \\  (let result (square 5))
        \\  (return result))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var expander = MacroExpander.init(arena.allocator());
    defer expander.deinit();

    const expanded = try expander.expandProgram(nodes);
    try std.testing.expectEqual(@as(usize, 1), expanded.len); // macro definition stripped

    // Expanded function body should have (* 5 5)
    const fn_node = expanded[0];
    const fn_body = fn_node.data.list;
    const let_stmt = fn_body[3].data.list;
    const expr = let_stmt[2].data.list;
    try std.testing.expectEqualStrings("*", expr[0].data.atom.symbol);
    try std.testing.expectEqual(@as(f64, 5.0), expr[1].data.atom.number);
    try std.testing.expectEqual(@as(f64, 5.0), expr[2].data.atom.number);
}
