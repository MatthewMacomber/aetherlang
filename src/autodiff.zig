//! Aether Static Whole-Graph Automatic Differentiation (Source-to-Source)
//! Symbolically differentiates AST expressions at compile time,
//! generating optimal adjoint code without runtime tape overhead.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const StaticDiff = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) StaticDiff {
        return .{ .allocator = allocator };
    }

    /// Symbolically differentiates expression `expr` with respect to variable `wrt`
    pub fn differentiate(self: *StaticDiff, expr: *Node, wrt: []const u8) anyerror!*Node {
        const d = try self.diffRaw(expr, wrt);
        return try self.simplify(d);
    }

    fn makeNumber(self: *StaticDiff, val: f64) !*Node {
        const n = try self.allocator.create(Node);
        n.loc = .{ .line = 1, .col = 1 };
        n.data = .{ .atom = .{ .number = val } };
        return n;
    }

    fn makeSymbol(self: *StaticDiff, sym: []const u8) !*Node {
        const n = try self.allocator.create(Node);
        n.loc = .{ .line = 1, .col = 1 };
        n.data = .{ .atom = .{ .symbol = sym } };
        return n;
    }

    fn makeBinary(self: *StaticDiff, op: []const u8, left: *Node, right: *Node) !*Node {
        const op_node = try self.makeSymbol(op);
        const list = try self.allocator.alloc(*Node, 3);
        list[0] = op_node;
        list[1] = left;
        list[2] = right;
        const n = try self.allocator.create(Node);
        n.loc = .{ .line = 1, .col = 1 };
        n.data = .{ .list = list };
        return n;
    }

    fn makeUnary(self: *StaticDiff, op: []const u8, child: *Node) !*Node {
        const op_node = try self.makeSymbol(op);
        const list = try self.allocator.alloc(*Node, 2);
        list[0] = op_node;
        list[1] = child;
        const n = try self.allocator.create(Node);
        n.loc = .{ .line = 1, .col = 1 };
        n.data = .{ .list = list };
        return n;
    }

    fn diffRaw(self: *StaticDiff, node: *Node, wrt: []const u8) anyerror!*Node {
        switch (node.data) {
            .atom => |a| switch (a) {
                .number, .nil, .boolean, .string => return try self.makeNumber(0.0),
                .symbol => |s| {
                    if (std.mem.eql(u8, s, wrt)) {
                        return try self.makeNumber(1.0); // d(x)/dx = 1
                    } else {
                        return try self.makeNumber(0.0); // d(c)/dx = 0
                    }
                },
            },
            .bracket_list => return try self.makeNumber(0.0),
            .list => |children| {
                if (children.len == 0) return try self.makeNumber(0.0);
                const head = children[0];
                if (head.data != .atom or head.data.atom != .symbol) return try self.makeNumber(0.0);
                const op = head.data.atom.symbol;

                if (std.mem.eql(u8, op, "+") and children.len >= 3) {
                    // d(u + v) = du + dv
                    const du = try self.diffRaw(children[1], wrt);
                    const dv = try self.diffRaw(children[2], wrt);
                    return try self.makeBinary("+", du, dv);
                } else if (std.mem.eql(u8, op, "-") and children.len >= 3) {
                    // d(u - v) = du - dv
                    const du = try self.diffRaw(children[1], wrt);
                    const dv = try self.diffRaw(children[2], wrt);
                    return try self.makeBinary("-", du, dv);
                } else if (std.mem.eql(u8, op, "*") and children.len >= 3) {
                    // Product rule: d(u * v) = du * v + u * dv
                    const u = children[1];
                    const v = children[2];
                    const du = try self.diffRaw(u, wrt);
                    const dv = try self.diffRaw(v, wrt);

                    const term1 = try self.makeBinary("*", du, v);
                    const term2 = try self.makeBinary("*", u, dv);
                    return try self.makeBinary("+", term1, term2);
                } else if (std.mem.eql(u8, op, "/") and children.len >= 3) {
                    // Quotient rule: d(u / v) = (du * v - u * dv) / (v * v)
                    const u = children[1];
                    const v = children[2];
                    const du = try self.diffRaw(u, wrt);
                    const dv = try self.diffRaw(v, wrt);

                    const num = try self.makeBinary("-", try self.makeBinary("*", du, v), try self.makeBinary("*", u, dv));
                    const denom = try self.makeBinary("*", v, v);
                    return try self.makeBinary("/", num, denom);
                } else if (std.mem.eql(u8, op, "relu") and children.len >= 2) {
                    // d(relu(u)) = (if (> u 0) du 0)
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    const cond = try self.makeBinary(">", u, try self.makeNumber(0.0));
                    const if_sym = try self.makeSymbol("if");
                    const list = try self.allocator.alloc(*Node, 4);
                    list[0] = if_sym;
                    list[1] = cond;
                    list[2] = du;
                    list[3] = try self.makeNumber(0.0);
                    const if_node = try self.allocator.create(Node);
                    if_node.loc = node.loc;
                    if_node.data = .{ .list = list };
                    return if_node;
                } else if (std.mem.eql(u8, op, "matmul") and children.len >= 3) {
                    // Symbolic matmul derivative
                    const a = children[1];
                    const b = children[2];
                    const da = try self.diffRaw(a, wrt);
                    const db = try self.diffRaw(b, wrt);

                    const term1 = try self.makeBinary("matmul", da, b);
                    const term2 = try self.makeBinary("matmul", a, db);
                    return try self.makeBinary("+", term1, term2);
                } else if (std.mem.eql(u8, op, "sigmoid") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    const s = try self.makeUnary("sigmoid", u);
                    const one_minus_s = try self.makeBinary("-", try self.makeNumber(1.0), s);
                    return try self.makeBinary("*", try self.makeBinary("*", s, one_minus_s), du);
                } else if (std.mem.eql(u8, op, "tanh") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    const t = try self.makeUnary("tanh", u);
                    const one_minus_t2 = try self.makeBinary("-", try self.makeNumber(1.0), try self.makeBinary("*", t, t));
                    return try self.makeBinary("*", one_minus_t2, du);
                } else if (std.mem.eql(u8, op, "sin") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    return try self.makeBinary("*", try self.makeUnary("cos", u), du);
                } else if (std.mem.eql(u8, op, "cos") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    const neg_sin = try self.makeBinary("-", try self.makeNumber(0.0), try self.makeUnary("sin", u));
                    return try self.makeBinary("*", neg_sin, du);
                } else if (std.mem.eql(u8, op, "exp") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    return try self.makeBinary("*", try self.makeUnary("exp", u), du);
                } else if (std.mem.eql(u8, op, "log") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    return try self.makeBinary("/", du, u);
                } else if (std.mem.eql(u8, op, "sqrt") and children.len >= 2) {
                    const u = children[1];
                    const du = try self.diffRaw(u, wrt);
                    const two_sqrt = try self.makeBinary("*", try self.makeNumber(2.0), try self.makeUnary("sqrt", u));
                    return try self.makeBinary("/", du, two_sqrt);
                }
                return try self.makeNumber(0.0);
            },
        }
    }

    /// Algebraic simplification pass (0 * x -> 0, 1 * x -> x, 0 + x -> x, etc.)
    pub fn simplify(self: *StaticDiff, node: *Node) anyerror!*Node {
        switch (node.data) {
            .atom => return node,
            .bracket_list => return node,
            .list => |children| {
                if (children.len == 0) return node;
                const head = children[0];
                if (head.data != .atom or head.data.atom != .symbol) return node;
                const op = head.data.atom.symbol;

                if ((std.mem.eql(u8, op, "+") or std.mem.eql(u8, op, "-") or std.mem.eql(u8, op, "*") or std.mem.eql(u8, op, "/")) and children.len >= 3) {
                    const l = try self.simplify(children[1]);
                    const r = try self.simplify(children[2]);

                    const l_num = if (l.data == .atom and l.data.atom == .number) l.data.atom.number else null;
                    const r_num = if (r.data == .atom and r.data.atom == .number) r.data.atom.number else null;

                    // Constant evaluation
                    if (l_num != null and r_num != null) {
                        if (std.mem.eql(u8, op, "+")) return try self.makeNumber(l_num.? + r_num.?);
                        if (std.mem.eql(u8, op, "-")) return try self.makeNumber(l_num.? - r_num.?);
                        if (std.mem.eql(u8, op, "*")) return try self.makeNumber(l_num.? * r_num.?);
                        if (std.mem.eql(u8, op, "/") and r_num.? != 0) return try self.makeNumber(l_num.? / r_num.?);
                    }

                    // Addition identity
                    if (std.mem.eql(u8, op, "+")) {
                        if (l_num != null and l_num.? == 0.0) return r;
                        if (r_num != null and r_num.? == 0.0) return l;
                    }

                    // Multiplication identities
                    if (std.mem.eql(u8, op, "*")) {
                        if ((l_num != null and l_num.? == 0.0) or (r_num != null and r_num.? == 0.0)) {
                            return try self.makeNumber(0.0);
                        }
                        if (l_num != null and l_num.? == 1.0) return r;
                        if (r_num != null and r_num.? == 1.0) return l;
                    }

                    return try self.makeBinary(op, l, r);
                }
                return node;
            },
        }
    }
};

test "static differentiation of polynomial x^2 + 3*x + 5" {
    // (* x x) + (* 3 x) + 5
    // Derivative w.r.t x is: (+ (* 2 x) 3) or (+ (+ x x) 3)
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src = "(+ (+ (* x x) (* 3.0 x)) 5.0)";
    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();
    try std.testing.expectEqual(@as(usize, 1), nodes.len);

    var sd = StaticDiff.init(arena.allocator());
    const derivative = try sd.differentiate(nodes[0], "x");

    // Print derivative structure
    try std.testing.expect(derivative.data == .list);
    // Evaluates symbolically: d/dx(x*x + 3x + 5) = (+ (+ x x) 3)
    const top = derivative.data.list;
    try std.testing.expectEqualStrings("+", top[0].data.atom.symbol);
}
