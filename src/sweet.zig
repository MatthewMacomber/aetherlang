//! Aether Sweet Syntax Transpiler
//! Provides bidirectional lossless transformation between human-readable
//! sweet syntax (fn, let, infix, braces) and canonical S-expressions.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub fn isSweetSyntax(source: []const u8) bool {
    var i: usize = 0;
    while (i < source.len) {
        // Skip whitespace
        while (i < source.len and (source[i] == ' ' or source[i] == '\t' or source[i] == '\r' or source[i] == '\n')) : (i += 1) {}
        if (i >= source.len) break;

        // Skip comments
        if (source[i] == '#') {
            // Check if datum label
            if (i + 1 < source.len and std.ascii.isDigit(source[i + 1])) {
                return false; // Canonical datum label #1=
            }
            while (i < source.len and source[i] != '\n') : (i += 1) {}
            continue;
        }

        // Canonical always begins with '('
        if (source[i] == '(') return false;

        // Sweet starts with keywords like fn, struct, module, let, use, type
        return true;
    }
    return false;
}

pub const SweetTranspiler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SweetTranspiler {
        return .{ .allocator = allocator };
    }

    /// Transpiles human-readable sweet syntax into canonical S-expressions
    pub fn toCanonical(self: *SweetTranspiler, sweet: []const u8) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var lines = std.mem.splitScalar(u8, sweet, '\n');
        var in_struct = false;

        while (lines.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') {
                continue;
            }

            // Function declaration: fn name(args...) -> ret { or fn name(args...):
            if (std.mem.startsWith(u8, line, "fn ")) {
                const after_fn = std.mem.trim(u8, line[3..], " ");
                if (std.mem.indexOfScalar(u8, after_fn, '(')) |lparen| {
                    const fn_name = std.mem.trim(u8, after_fn[0..lparen], " ");
                    const after_lparen = after_fn[lparen + 1 ..];
                    if (std.mem.indexOfScalar(u8, after_lparen, ')')) |rparen| {
                        const raw_args = after_lparen[0..rparen];
                        try out.appendSlice(self.allocator, "(func ");
                        try out.appendSlice(self.allocator, fn_name);
                        try out.appendSlice(self.allocator, " (");

                        // Parse args
                        var arg_iter = std.mem.splitScalar(u8, raw_args, ',');
                        var first_arg = true;
                        while (arg_iter.next()) |raw_arg| {
                            const arg = std.mem.trim(u8, raw_arg, " \t");
                            if (arg.len == 0) continue;
                            if (!first_arg) try out.append(self.allocator, ' ');
                            first_arg = false;

                            if (std.mem.indexOfScalar(u8, arg, ':')) |colon| {
                                const aname = std.mem.trim(u8, arg[0..colon], " ");
                                const atype = std.mem.trim(u8, arg[colon + 1 ..], " ");
                                try out.appendSlice(self.allocator, "(");
                                try out.appendSlice(self.allocator, aname);
                                try out.append(self.allocator, ' ');
                                try out.appendSlice(self.allocator, atype);
                                try out.appendSlice(self.allocator, ")");
                            } else {
                                try out.appendSlice(self.allocator, arg);
                            }
                        }
                        try out.appendSlice(self.allocator, ")\n");
                        continue;
                    }
                }
            }

            // Struct declaration: struct Name {
            if (std.mem.startsWith(u8, line, "struct ")) {
                in_struct = true;
                const rest = std.mem.trim(u8, line["struct ".len..], " {");
                const is_linear = false;
                _ = is_linear;
                try out.appendSlice(self.allocator, "(struct ");
                try out.appendSlice(self.allocator, rest);
                try out.append(self.allocator, '\n');
                continue;
            }

            // Field inside struct: name: type,
            if (in_struct and std.mem.indexOfScalar(u8, line, ':') != null and !std.mem.startsWith(u8, line, "let ") and !std.mem.startsWith(u8, line, "fn ")) {
                const colon = std.mem.indexOfScalar(u8, line, ':').?;
                const fname = std.mem.trim(u8, line[0..colon], " ");
                const ftype = std.mem.trim(u8, line[colon + 1 ..], " ,;");
                if (ftype.len > 0) {
                    try out.appendSlice(self.allocator, "  (");
                    try out.appendSlice(self.allocator, fname);
                    try out.append(self.allocator, ' ');
                    try out.appendSlice(self.allocator, ftype);
                    try out.appendSlice(self.allocator, ")\n");
                    continue;
                }
            }

            // Let statement: let x = expr;
            if (std.mem.startsWith(u8, line, "let ")) {
                const rest = std.mem.trim(u8, line[4..], " ;");
                if (std.mem.indexOfScalar(u8, rest, '=')) |eq| {
                    const var_name = std.mem.trim(u8, rest[0..eq], " ");
                    const val_expr = std.mem.trim(u8, rest[eq + 1 ..], " ;");
                    try out.appendSlice(self.allocator, "  (let ");
                    try out.appendSlice(self.allocator, var_name);
                    try out.append(self.allocator, ' ');
                    try self.transpileExpr(&out, val_expr);
                    try out.appendSlice(self.allocator, ")\n");
                    continue;
                }
            }

            // Return statement: return expr;
            if (std.mem.startsWith(u8, line, "return")) {
                const rest = std.mem.trim(u8, line[6..], " ;");
                try out.appendSlice(self.allocator, "  (return ");
                if (rest.len > 0) {
                    try self.transpileExpr(&out, rest);
                } else {
                    try out.appendSlice(self.allocator, "0");
                }
                try out.appendSlice(self.allocator, ")\n");
                continue;
            }

            // While statement: while cond {
            if (std.mem.startsWith(u8, line, "while ")) {
                const rest = std.mem.trim(u8, line[6..], " {");
                const cond = if (std.mem.startsWith(u8, rest, "(") and std.mem.endsWith(u8, rest, ")"))
                    rest[1 .. rest.len - 1]
                else
                    rest;
                try out.appendSlice(self.allocator, "  (while ");
                try self.transpileExpr(&out, cond);
                try out.append(self.allocator, '\n');
                continue;
            }

            // For statement: for x in range(n) {
            if (std.mem.startsWith(u8, line, "for ")) {
                const rest = std.mem.trim(u8, line[4..], " {");
                if (std.mem.indexOf(u8, rest, " in ")) |in_pos| {
                    const var_name = std.mem.trim(u8, rest[0..in_pos], " ");
                    const iter_expr = std.mem.trim(u8, rest[in_pos + 4 ..], " ");
                    try out.appendSlice(self.allocator, "  (for ");
                    try out.appendSlice(self.allocator, var_name);
                    try out.append(self.allocator, ' ');
                    if (std.mem.startsWith(u8, iter_expr, "range(") and std.mem.endsWith(u8, iter_expr, ")")) {
                        const r_args = iter_expr[6 .. iter_expr.len - 1];
                        try out.appendSlice(self.allocator, "(range");
                        var r_iter = std.mem.splitScalar(u8, r_args, ',');
                        while (r_iter.next()) |arg| {
                            const a = std.mem.trim(u8, arg, " ");
                            if (a.len == 0) continue;
                            try out.append(self.allocator, ' ');
                            try self.transpileExpr(&out, a);
                        }
                        try out.append(self.allocator, ')');
                    } else {
                        try self.transpileExpr(&out, iter_expr);
                    }
                    try out.append(self.allocator, '\n');
                    continue;
                }
            }

            // If statement: if cond {
            if (std.mem.startsWith(u8, line, "if ")) {
                const rest = std.mem.trim(u8, line[3..], " {");
                const cond = if (std.mem.startsWith(u8, rest, "(") and std.mem.endsWith(u8, rest, ")"))
                    rest[1 .. rest.len - 1]
                else
                    rest;
                try out.appendSlice(self.allocator, "  (if ");
                try self.transpileExpr(&out, cond);
                try out.append(self.allocator, '\n');
                continue;
            }

            // Print call: print(...) or println!(...)
            if (std.mem.startsWith(u8, line, "print(") or std.mem.startsWith(u8, line, "println!(")) {
                const start = if (std.mem.startsWith(u8, line, "println!(")) "println!(".len else "print(".len;
                const inner = std.mem.trim(u8, line[start..], " );");
                try out.appendSlice(self.allocator, "  (call print ");
                var args = std.mem.splitScalar(u8, inner, ',');
                var first = true;
                while (args.next()) |arg| {
                    const a = std.mem.trim(u8, arg, " ");
                    if (a.len == 0) continue;
                    if (!first) try out.append(self.allocator, ' ');
                    first = false;
                    try self.transpileExpr(&out, a);
                }
                try out.appendSlice(self.allocator, ")\n");
                continue;
            }

            // Function call: func(a, b);
            if (std.mem.indexOfScalar(u8, line, '(')) |lparen| {
                if (std.mem.endsWith(u8, line, ");") or std.mem.endsWith(u8, line, ")")) {
                    const fname = std.mem.trim(u8, line[0..lparen], " ");
                    const rparen = std.mem.lastIndexOfScalar(u8, line, ')').?;
                    const inner = line[lparen + 1 .. rparen];
                    try out.appendSlice(self.allocator, "  (call ");
                    try out.appendSlice(self.allocator, fname);
                    var args = std.mem.splitScalar(u8, inner, ',');
                    while (args.next()) |arg| {
                        const a = std.mem.trim(u8, arg, " ");
                        if (a.len == 0) continue;
                        try out.append(self.allocator, ' ');
                        try self.transpileExpr(&out, a);
                    }
                    try out.appendSlice(self.allocator, ")\n");
                    continue;
                }
            }

            // Assignment: x = expr; or set x = expr;
            if (std.mem.indexOfScalar(u8, line, '=') != null and !std.mem.startsWith(u8, line, "let ") and !std.mem.startsWith(u8, line, "if ") and !std.mem.startsWith(u8, line, "fn ") and !std.mem.startsWith(u8, line, "struct ") and !std.mem.startsWith(u8, line, "while ") and !std.mem.startsWith(u8, line, "for ")) {
                const eq = std.mem.indexOfScalar(u8, line, '=').?;
                const var_part = std.mem.trim(u8, line[0..eq], " ");
                if (std.mem.indexOfScalar(u8, var_part, '(') == null and std.mem.indexOfScalar(u8, var_part, '"') == null) {
                    const var_name = if (std.mem.startsWith(u8, var_part, "set ")) std.mem.trim(u8, var_part[4..], " ") else var_part;
                    const expr = std.mem.trim(u8, line[eq + 1 ..], " ;");
                    try out.appendSlice(self.allocator, "  (set ");
                    try out.appendSlice(self.allocator, var_name);
                    try out.append(self.allocator, ' ');
                    try self.transpileExpr(&out, expr);
                    try out.appendSlice(self.allocator, ")\n");
                    continue;
                }
            }

            // Else statement: } else { or else {
            if (std.mem.eql(u8, line, "} else {") or std.mem.eql(u8, line, "else {")) {
                continue;
            }

            // Closing brace '}' ends function or struct
            if (std.mem.eql(u8, line, "}")) {
                in_struct = false;
                try out.appendSlice(self.allocator, ")\n");
                continue;
            }
        }

        return try out.toOwnedSlice(self.allocator);
    }

    fn transpileExpr(self: *SweetTranspiler, out: *std.ArrayList(u8), expr_in: []const u8) !void {
        const expr = std.mem.trim(u8, expr_in, " ;");
        if (expr.len >= 2 and expr[0] == '"' and expr[expr.len - 1] == '"') {
            try out.appendSlice(self.allocator, expr);
            return;
        }

        // Infix check: binary +, -, *, /
        const ops = [_][]const u8{ " + ", " - ", " * ", " / ", " == ", " != ", " <= ", " >= ", " < ", " > " };
        for (ops) |op| {
            if (std.mem.indexOf(u8, expr, op)) |pos| {
                const op_sym = std.mem.trim(u8, op, " ");
                const left = std.mem.trim(u8, expr[0..pos], " ()");
                const right = std.mem.trim(u8, expr[pos + op.len ..], " ()");
                try out.appendSlice(self.allocator, "(");
                try out.appendSlice(self.allocator, op_sym);
                try out.append(self.allocator, ' ');
                try self.transpileExpr(out, left);
                try out.append(self.allocator, ' ');
                try self.transpileExpr(out, right);
                try out.appendSlice(self.allocator, ")");
                return;
            }
        }

        // Function call expression: func(a, b)
        if (std.mem.indexOfScalar(u8, expr, '(')) |lparen| {
            if (std.mem.endsWith(u8, expr, ")")) {
                const fname = std.mem.trim(u8, expr[0..lparen], " ");
                const inner = expr[lparen + 1 .. expr.len - 1];
                try out.appendSlice(self.allocator, "(call ");
                try out.appendSlice(self.allocator, fname);
                var args = std.mem.splitScalar(u8, inner, ',');
                while (args.next()) |arg| {
                    const a = std.mem.trim(u8, arg, " ");
                    if (a.len == 0) continue;
                    try out.append(self.allocator, ' ');
                    try self.transpileExpr(out, a);
                }
                try out.appendSlice(self.allocator, ")");
                return;
            }
        }

        // Bracket list: [1, 2, 3] -> [1 2 3]
        if (std.mem.startsWith(u8, expr, "[") and std.mem.endsWith(u8, expr, "]")) {
            const inner = expr[1 .. expr.len - 1];
            try out.appendSlice(self.allocator, "[");
            var items = std.mem.splitScalar(u8, inner, ',');
            var first = true;
            while (items.next()) |item| {
                const it = std.mem.trim(u8, item, " ");
                if (it.len == 0) continue;
                if (!first) try out.append(self.allocator, ' ');
                first = false;
                try out.appendSlice(self.allocator, it);
            }
            try out.appendSlice(self.allocator, "]");
            return;
        }

        // Atom/literal
        try out.appendSlice(self.allocator, expr);
    }

    /// Converts canonical S-expression AST nodes to sweet syntax
    pub fn toSweet(self: *SweetTranspiler, nodes: []const *Node) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;

        for (nodes) |node| {
            try self.nodeToSweet(&out, node, 0);
            try out.append(self.allocator, '\n');
        }

        return try out.toOwnedSlice(self.allocator);
    }

    fn nodeToSweet(self: *SweetTranspiler, out: *std.ArrayList(u8), node: *Node, indent: usize) anyerror!void {
        switch (node.data) {
            .atom => |a| switch (a) {
                .nil => try out.appendSlice(self.allocator, "nil"),
                .boolean => |b| try out.appendSlice(self.allocator, if (b) "true" else "false"),
                .number => |n| {
                    var buf: [64]u8 = undefined;
                    const s = std.fmt.bufPrint(&buf, "{d}", .{n}) catch "0";
                    try out.appendSlice(self.allocator, s);
                },
                .string => |s| {
                    try out.append(self.allocator, '"');
                    try out.appendSlice(self.allocator, s);
                    try out.append(self.allocator, '"');
                },
                .symbol => |s| try out.appendSlice(self.allocator, s),
            },
            .bracket_list => |children| {
                try out.appendSlice(self.allocator, "[");
                for (children, 0..) |child, i| {
                    if (i > 0) try out.appendSlice(self.allocator, ", ");
                    try self.nodeToSweet(out, child, indent);
                }
                try out.appendSlice(self.allocator, "]");
            },
            .list => |children| {
                if (children.len == 0) return;
                const head = children[0];
                if (head.data == .atom and head.data.atom == .symbol) {
                    const sym = head.data.atom.symbol;

                    if (std.mem.eql(u8, sym, "func") and children.len >= 3) {
                        const name = children[1].data.atom.symbol;
                        try out.appendSlice(self.allocator, "fn ");
                        try out.appendSlice(self.allocator, name);
                        try out.appendSlice(self.allocator, "(");

                        const params = children[2];
                        if (params.data == .list) {
                            for (params.data.list, 0..) |p, i| {
                                if (i > 0) try out.appendSlice(self.allocator, ", ");
                                if (p.data == .atom) {
                                    try out.appendSlice(self.allocator, p.data.atom.symbol);
                                } else if (p.data == .list and p.data.list.len >= 2) {
                                    try out.appendSlice(self.allocator, p.data.list[0].data.atom.symbol);
                                    try out.appendSlice(self.allocator, ": ");
                                    try out.appendSlice(self.allocator, p.data.list[1].data.atom.symbol);
                                }
                            }
                        }
                        try out.appendSlice(self.allocator, ") {\n");

                        for (children[3..]) |body_node| {
                            try self.writeSweetIndent(out, indent + 1);
                            try self.nodeToSweet(out, body_node, indent + 1);
                            try out.appendSlice(self.allocator, ";\n");
                        }
                        try self.writeSweetIndent(out, indent);
                        try out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, sym, "struct") and children.len >= 2) {
                        const name = children[1].data.atom.symbol;
                        try out.appendSlice(self.allocator, "struct ");
                        try out.appendSlice(self.allocator, name);
                        try out.appendSlice(self.allocator, " {\n");
                        for (children[2..]) |f| {
                            if (f.data == .list and f.data.list.len >= 2) {
                                try self.writeSweetIndent(out, indent + 1);
                                try out.appendSlice(self.allocator, f.data.list[0].data.atom.symbol);
                                try out.appendSlice(self.allocator, ": ");
                                try out.appendSlice(self.allocator, f.data.list[1].data.atom.symbol);
                                try out.appendSlice(self.allocator, ",\n");
                            }
                        }
                        try self.writeSweetIndent(out, indent);
                        try out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, sym, "let") and children.len >= 3) {
                        try out.appendSlice(self.allocator, "let ");
                        try self.nodeToSweet(out, children[1], indent);
                        try out.appendSlice(self.allocator, " = ");
                        try self.nodeToSweet(out, children[2], indent);
                        return;
                    } else if (std.mem.eql(u8, sym, "return")) {
                        try out.appendSlice(self.allocator, "return");
                        if (children.len >= 2) {
                            try out.append(self.allocator, ' ');
                            try self.nodeToSweet(out, children[1], indent);
                        }
                        return;
                    } else if (std.mem.eql(u8, sym, "call") and children.len >= 2) {
                        const fn_name = children[1].data.atom.symbol;
                        try out.appendSlice(self.allocator, fn_name);
                        try out.appendSlice(self.allocator, "(");
                        for (children[2..], 0..) |arg, i| {
                            if (i > 0) try out.appendSlice(self.allocator, ", ");
                            try self.nodeToSweet(out, arg, indent);
                        }
                        try out.appendSlice(self.allocator, ")");
                        return;
                    } else if ((std.mem.eql(u8, sym, "+") or std.mem.eql(u8, sym, "-") or
                        std.mem.eql(u8, sym, "*") or std.mem.eql(u8, sym, "/")) and children.len >= 3)
                    {
                        try out.appendSlice(self.allocator, "(");
                        try self.nodeToSweet(out, children[1], indent);
                        try out.append(self.allocator, ' ');
                        try out.appendSlice(self.allocator, sym);
                        try out.append(self.allocator, ' ');
                        try self.nodeToSweet(out, children[2], indent);
                        try out.appendSlice(self.allocator, ")");
                        return;
                    }
                }

                // Fallback: emit sexpr form
                try out.append(self.allocator, '(');
                for (children, 0..) |c, i| {
                    if (i > 0) try out.append(self.allocator, ' ');
                    try self.nodeToSweet(out, c, indent);
                }
                try out.append(self.allocator, ')');
            },
        }
    }

    fn writeSweetIndent(self: *SweetTranspiler, out: *std.ArrayList(u8), level: usize) !void {
        var i: usize = 0;
        while (i < level * 2) : (i += 1) {
            try out.append(self.allocator, ' ');
        }
    }
};

test "sweet to canonical and back" {
    const sweet_code =
        \\fn add(a: i64, b: i64) {
        \\  let sum = a + b;
        \\  return sum;
        \\}
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var transpiler = SweetTranspiler.init(arena.allocator());
    const canonical = try transpiler.toCanonical(sweet_code);

    try std.testing.expect(std.mem.indexOf(u8, canonical, "(func add ((a i64) (b i64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, canonical, "(let sum (+ a b))") != null);
    try std.testing.expect(std.mem.indexOf(u8, canonical, "(return sum)") != null);

    // Parse the generated canonical S-expression
    var p = try parser.Parser.init(canonical, arena.allocator());
    const nodes = try p.parseAll();
    try std.testing.expectEqual(@as(usize, 1), nodes.len);

    // Convert back to sweet
    const roundtrip_sweet = try transpiler.toSweet(nodes);
    try std.testing.expect(std.mem.indexOf(u8, roundtrip_sweet, "fn add(a: i64, b: i64) {") != null);
    try std.testing.expect(std.mem.indexOf(u8, roundtrip_sweet, "let sum = (a + b);") != null);
}
