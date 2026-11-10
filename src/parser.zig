//! Aether S-Expression Parser & Lexer
//! Tokenizes and parses Aether S-expressions into an arena-backed AST.

const std = @import("std");

pub const SourceLoc = struct {
    line: u32,
    col: u32,
};

pub const Atom = union(enum) {
    nil: void,
    boolean: bool,
    number: f64,
    string: []const u8,
    symbol: []const u8,
};

pub const Node = struct {
    loc: SourceLoc,
    data: Data,

    pub const Data = union(enum) {
        atom: Atom,
        list: []const *Node,
        bracket_list: []const *Node,
    };
};

pub const Token = struct {
    loc: SourceLoc,
    kind: Kind,

    pub const Kind = union(enum) {
        lparen,
        rparen,
        lbracket,
        rbracket,
        quote,
        datum_def: usize,
        datum_ref: usize,
        atom: Atom,
        eof,
    };
};

pub const Lexer = struct {
    source: []const u8,
    index: usize = 0,
    line: u32 = 1,
    col: u32 = 1,

    pub fn init(source: []const u8) Lexer {
        return .{ .source = source };
    }

    fn peek(self: *Lexer) ?u8 {
        if (self.index >= self.source.len) return null;
        return self.source[self.index];
    }

    fn advance(self: *Lexer) ?u8 {
        if (self.index >= self.source.len) return null;
        const c = self.source[self.index];
        self.index += 1;
        if (c == '\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        return c;
    }

    fn isDatumLabel(self: *Lexer) bool {
        if (self.index >= self.source.len or self.source[self.index] != '#') return false;
        var j = self.index + 1;
        while (j < self.source.len and std.ascii.isDigit(self.source[j])) : (j += 1) {}
        if (j > self.index + 1 and j < self.source.len) {
            return self.source[j] == '=' or self.source[j] == '#';
        }
        return false;
    }

    fn skipWhitespaceAndComments(self: *Lexer) void {
        while (self.index < self.source.len) {
            const c = self.source[self.index];
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                _ = self.advance();
            } else if (c == '#') {
                if (self.isDatumLabel()) break;
                // Comment until end of line
                while (self.index < self.source.len and self.source[self.index] != '\n') {
                    _ = self.advance();
                }
            } else {
                break;
            }
        }
    }

    pub fn nextToken(self: *Lexer, allocator: std.mem.Allocator) !Token {
        self.skipWhitespaceAndComments();

        const loc = SourceLoc{ .line = self.line, .col = self.col };
        if (self.index >= self.source.len) {
            return Token{ .loc = loc, .kind = .eof };
        }

        const c = self.source[self.index];
        if (c == '\'') {
            _ = self.advance();
            return Token{ .loc = loc, .kind = .quote };
        }
        if (c == '#') {
            var j = self.index + 1;
            while (j < self.source.len and std.ascii.isDigit(self.source[j])) : (j += 1) {}
            if (j > self.index + 1 and j < self.source.len) {
                if (self.source[j] == '=') {
                    const num_str = self.source[self.index + 1 .. j];
                    const id = std.fmt.parseInt(usize, num_str, 10) catch 0;
                    self.index = j + 1;
                    return Token{ .loc = loc, .kind = .{ .datum_def = id } };
                } else if (self.source[j] == '#') {
                    const num_str = self.source[self.index + 1 .. j];
                    const id = std.fmt.parseInt(usize, num_str, 10) catch 0;
                    self.index = j + 1;
                    return Token{ .loc = loc, .kind = .{ .datum_ref = id } };
                }
            }
        }
        switch (c) {
            '(' => {
                _ = self.advance();
                return Token{ .loc = loc, .kind = .lparen };
            },
            ')' => {
                _ = self.advance();
                return Token{ .loc = loc, .kind = .rparen };
            },
            '[' => {
                _ = self.advance();
                return Token{ .loc = loc, .kind = .lbracket };
            },
            ']' => {
                _ = self.advance();
                return Token{ .loc = loc, .kind = .rbracket };
            },
            '"' => {
                _ = self.advance(); // consume opening quote
                const start = self.index;
                var has_escapes = false;

                while (self.index < self.source.len and self.source[self.index] != '"') {
                    if (self.source[self.index] == '\\') {
                        has_escapes = true;
                        _ = self.advance(); // skip backslash
                        if (self.index < self.source.len) _ = self.advance();
                    } else {
                        _ = self.advance();
                    }
                }

                if (self.index >= self.source.len) {
                    return error.UnterminatedString;
                }

                const raw = self.source[start..self.index];
                _ = self.advance(); // consume closing quote

                if (!has_escapes) {
                    return Token{ .loc = loc, .kind = .{ .atom = .{ .string = raw } } };
                }

                // Unescape
                var buf: std.ArrayList(u8) = .empty;
                var i: usize = 0;
                while (i < raw.len) : (i += 1) {
                    if (raw[i] == '\\' and i + 1 < raw.len) {
                        i += 1;
                        switch (raw[i]) {
                            'n' => try buf.append(allocator, '\n'),
                            't' => try buf.append(allocator, '\t'),
                            'r' => try buf.append(allocator, '\r'),
                            '\\' => try buf.append(allocator, '\\'),
                            '"' => try buf.append(allocator, '"'),
                            else => |esc| try buf.append(allocator, esc),
                        }
                    } else {
                        try buf.append(allocator, raw[i]);
                    }
                }
                const unescaped = try buf.toOwnedSlice(allocator);
                return Token{ .loc = loc, .kind = .{ .atom = .{ .string = unescaped } } };
            },
            else => {
                // Identifiers, numbers, operators
                const start = self.index;
                while (self.index < self.source.len) {
                    const ch = self.source[self.index];
                    if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n' or
                        ch == '(' or ch == ')' or ch == '[' or ch == ']' or
                        ch == '"' or ch == '#')
                    {
                        break;
                    }
                    _ = self.advance();
                }

                const text = self.source[start..self.index];

                // Check boolean
                if (std.mem.eql(u8, text, "true")) {
                    return Token{ .loc = loc, .kind = .{ .atom = .{ .boolean = true } } };
                } else if (std.mem.eql(u8, text, "false")) {
                    return Token{ .loc = loc, .kind = .{ .atom = .{ .boolean = false } } };
                } else if (std.mem.eql(u8, text, "nil")) {
                    return Token{ .loc = loc, .kind = .{ .atom = .nil } };
                }

                // Check number
                if (std.fmt.parseFloat(f64, text)) |num| {
                    return Token{ .loc = loc, .kind = .{ .atom = .{ .number = num } } };
                } else |_| {}

                // Otherwise symbol
                return Token{ .loc = loc, .kind = .{ .atom = .{ .symbol = text } } };
            },
        }
    }
};

pub const Parser = struct {
    lexer: Lexer,
    allocator: std.mem.Allocator,
    current: Token,
    labels: std.AutoHashMap(usize, *Node),

    pub fn init(source: []const u8, allocator: std.mem.Allocator) !Parser {
        var lexer = Lexer.init(source);
        const tok = try lexer.nextToken(allocator);
        return .{
            .lexer = lexer,
            .allocator = allocator,
            .current = tok,
            .labels = std.AutoHashMap(usize, *Node).init(allocator),
        };
    }

    fn advance(self: *Parser) !void {
        self.current = try self.lexer.nextToken(self.allocator);
    }

    pub fn parseExpression(self: *Parser) anyerror!?*Node {
        switch (self.current.kind) {
            .eof => return null,
            .datum_def => |id| {
                const loc = self.current.loc;
                try self.advance(); // consume #N=
                const placeholder = try self.allocator.create(Node);
                placeholder.loc = loc;
                placeholder.data = .{ .atom = .nil };
                try self.labels.put(id, placeholder);

                if (try self.parseExpression()) |inner| {
                    placeholder.* = inner.*;
                    return placeholder;
                } else {
                    return error.ExpectedDatumAfterLabel;
                }
            },
            .datum_ref => |id| {
                if (self.labels.get(id)) |existing| {
                    try self.advance(); // consume #N#
                    return existing;
                } else {
                    return error.UndefinedDatumLabel;
                }
            },
            .quote => {
                const loc = self.current.loc;
                try self.advance(); // consume '
                if (try self.parseExpression()) |quoted| {
                    const quote_sym = try self.allocator.create(Node);
                    quote_sym.loc = loc;
                    quote_sym.data = .{ .atom = .{ .symbol = "quote" } };
                    const quote_list = try self.allocator.alloc(*Node, 2);
                    quote_list[0] = quote_sym;
                    quote_list[1] = quoted;
                    const node = try self.allocator.create(Node);
                    node.loc = loc;
                    node.data = .{ .list = quote_list };
                    return node;
                } else {
                    return error.ExpectedExpressionAfterQuote;
                }
            },
            .atom => |a| {
                const node = try self.allocator.create(Node);
                node.loc = self.current.loc;
                node.data = .{ .atom = a };
                try self.advance();
                return node;
            },
            .lparen => {
                const loc = self.current.loc;
                try self.advance(); // consume '('
                var items: std.ArrayList(*Node) = .empty;

                while (self.current.kind != .rparen and self.current.kind != .eof) {
                    if (try self.parseExpression()) |child| {
                        try items.append(self.allocator, child);
                    }
                }

                if (self.current.kind != .rparen) {
                    return error.MissingClosingParen;
                }
                try self.advance(); // consume ')'

                const node = try self.allocator.create(Node);
                node.loc = loc;
                node.data = .{ .list = try items.toOwnedSlice(self.allocator) };
                return node;
            },
            .lbracket => {
                const loc = self.current.loc;
                try self.advance(); // consume '['
                var items: std.ArrayList(*Node) = .empty;

                while (self.current.kind != .rbracket and self.current.kind != .eof) {
                    if (try self.parseExpression()) |child| {
                        try items.append(self.allocator, child);
                    }
                }

                if (self.current.kind != .rbracket) {
                    return error.MissingClosingBracket;
                }
                try self.advance(); // consume ']'

                const node = try self.allocator.create(Node);
                node.loc = loc;
                node.data = .{ .bracket_list = try items.toOwnedSlice(self.allocator) };
                return node;
            },
            .rparen, .rbracket => {
                return error.UnexpectedClosingDelimiter;
            },
        }
    }

    pub fn parseAll(self: *Parser) ![]const *Node {
        var nodes: std.ArrayList(*Node) = .empty;
        while (try self.parseExpression()) |node| {
            try nodes.append(self.allocator, node);
        }
        return try nodes.toOwnedSlice(self.allocator);
    }
};

test "parse basic sexprs and brackets" {
    const src =
        \\# Hello World
        \\(func main ()
        \\  (call print "Hello" 42 [2 3])
        \\  (return 0))
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var parser = try Parser.init(src, arena.allocator());
    const nodes = try parser.parseAll();
    try std.testing.expectEqual(@as(usize, 1), nodes.len);

    const fn_node = nodes[0];
    switch (fn_node.data) {
        .list => |children| {
            try std.testing.expectEqual(@as(usize, 5), children.len);
            try std.testing.expectEqualStrings("func", children[0].data.atom.symbol);
            try std.testing.expectEqualStrings("main", children[1].data.atom.symbol);
        },
        else => try std.testing.expect(false),
    }
}

pub fn isCyclic(root: *const Node, allocator: std.mem.Allocator) !bool {
    var visited = std.AutoHashMap(*const Node, void).init(allocator);
    defer visited.deinit();
    return detectCycle(root, &visited);
}

fn detectCycle(node: *const Node, visited: *std.AutoHashMap(*const Node, void)) !bool {
    if (visited.contains(node)) return true;
    try visited.put(node, {});
    defer _ = visited.remove(node);

    switch (node.data) {
        .list => |children| {
            for (children) |child| {
                if (try detectCycle(child, visited)) return true;
            }
        },
        .bracket_list => |children| {
            for (children) |child| {
                if (try detectCycle(child, visited)) return true;
            }
        },
        .atom => {},
    }
    return false;
}

test "parse cyclic datum labels and detect cycles" {
    const src = "#1=(node \"a\" (edge #1#))";
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();
    try std.testing.expectEqual(@as(usize, 1), nodes.len);

    const root = nodes[0];
    try std.testing.expect(try isCyclic(root, arena.allocator()));

    // Verify self-referential edge points to root itself
    const children = root.data.list;
    try std.testing.expectEqual(@as(usize, 3), children.len);
    const edge_node = children[2];
    try std.testing.expectEqual(@as(usize, 2), edge_node.data.list.len);
    try std.testing.expectEqual(root, edge_node.data.list[1]);
}
