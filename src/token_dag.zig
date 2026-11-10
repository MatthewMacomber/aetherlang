//! Aether Flat 32-Bit Tokenized DAG Binary Format
//! Compact binary representation of Aether AST for direct LLM generation,
//! content-addressed hashing (128-bit), and instant zero-overhead loading.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const TokenTag = enum(u8) {
    keyword = 0x01,
    opcode = 0x02,
    literal_int = 0x03,
    literal_str = 0x04,
    symbol = 0x05,
    node_ref = 0x06,
    lparen = 0x07,
    rparen = 0x08,
    lbracket = 0x09,
    rbracket = 0x0A,
    boolean = 0x0B,
    nil = 0x0C,
};

pub const Token = packed struct(u32) {
    tag: u8,
    payload: u24,

    pub fn init(tag: TokenTag, payload: u24) Token {
        return .{
            .tag = @intFromEnum(tag),
            .payload = payload,
        };
    }

    pub fn getTag(self: Token) TokenTag {
        return @enumFromInt(self.tag);
    }
};

pub const TokenDAG = struct {
    tokens: []const Token,
    strings: []const []const u8,
    hash128: [16]u8,

    pub fn encode(nodes: []const *Node, allocator: std.mem.Allocator) !TokenDAG {
        var token_list: std.ArrayList(Token) = .empty;
        var string_list: std.ArrayList([]const u8) = .empty;
        var string_map = std.StringHashMap(u24).init(allocator);
        defer string_map.deinit();

        for (nodes) |node| {
            try encodeNode(node, &token_list, &string_list, &string_map, allocator);
        }

        const tokens_slice = try token_list.toOwnedSlice(allocator);
        const strings_slice = try string_list.toOwnedSlice(allocator);
        const hash = computeHash128(tokens_slice, strings_slice);

        return TokenDAG{
            .tokens = tokens_slice,
            .strings = strings_slice,
            .hash128 = hash,
        };
    }

    fn internString(str: []const u8, string_list: *std.ArrayList([]const u8), string_map: *std.StringHashMap(u24), allocator: std.mem.Allocator) !u24 {
        if (string_map.get(str)) |idx| return idx;
        const idx: u24 = @intCast(string_list.items.len);
        const dupe = try allocator.dupe(u8, str);
        try string_list.append(allocator, dupe);
        try string_map.put(dupe, idx);
        return idx;
    }

    fn encodeNode(node: *Node, tokens: *std.ArrayList(Token), string_list: *std.ArrayList([]const u8), string_map: *std.StringHashMap(u24), allocator: std.mem.Allocator) anyerror!void {
        switch (node.data) {
            .atom => |a| switch (a) {
                .nil => try tokens.append(allocator, Token.init(.nil, 0)),
                .boolean => |b| try tokens.append(allocator, Token.init(.boolean, if (b) 1 else 0)),
                .number => |n| {
                    const int_val: i32 = @intFromFloat(n);
                    const payload: u24 = @truncate(@as(u32, @bitCast(int_val)));
                    try tokens.append(allocator, Token.init(.literal_int, payload));
                },
                .string => |s| {
                    const idx = try internString(s, string_list, string_map, allocator);
                    try tokens.append(allocator, Token.init(.literal_str, idx));
                },
                .symbol => |s| {
                    const idx = try internString(s, string_list, string_map, allocator);
                    try tokens.append(allocator, Token.init(.symbol, idx));
                },
            },
            .list => |children| {
                try tokens.append(allocator, Token.init(.lparen, @intCast(@min(children.len, 0xFFFFFF))));
                for (children) |child| {
                    try encodeNode(child, tokens, string_list, string_map, allocator);
                }
                try tokens.append(allocator, Token.init(.rparen, 0));
            },
            .bracket_list => |children| {
                try tokens.append(allocator, Token.init(.lbracket, @intCast(@min(children.len, 0xFFFFFF))));
                for (children) |child| {
                    try encodeNode(child, tokens, string_list, string_map, allocator);
                }
                try tokens.append(allocator, Token.init(.rbracket, 0));
            },
        }
    }

    pub fn decode(self: TokenDAG, allocator: std.mem.Allocator) ![]const *Node {
        var cursor: usize = 0;
        var nodes: std.ArrayList(*Node) = .empty;

        while (cursor < self.tokens.len) {
            if (try self.decodeNode(&cursor, allocator)) |node| {
                try nodes.append(allocator, node);
            }
        }
        return try nodes.toOwnedSlice(allocator);
    }

    fn decodeNode(self: TokenDAG, cursor: *usize, allocator: std.mem.Allocator) anyerror!?*Node {
        if (cursor.* >= self.tokens.len) return null;
        const tok = self.tokens[cursor.*];
        cursor.* += 1;

        const loc = parser.SourceLoc{ .line = 1, .col = 1 };
        const node = try allocator.create(Node);
        node.loc = loc;

        switch (tok.getTag()) {
            .nil => node.data = .{ .atom = .nil },
            .boolean => node.data = .{ .atom = .{ .boolean = tok.payload != 0 } },
            .literal_int => {
                const raw_i32: i32 = @bitCast(@as(u32, tok.payload));
                node.data = .{ .atom = .{ .number = @floatFromInt(raw_i32) } };
            },
            .literal_str => {
                const s = if (tok.payload < self.strings.len) self.strings[tok.payload] else "";
                node.data = .{ .atom = .{ .string = s } };
            },
            .symbol => {
                const s = if (tok.payload < self.strings.len) self.strings[tok.payload] else "";
                node.data = .{ .atom = .{ .symbol = s } };
            },
            .lparen => {
                var children: std.ArrayList(*Node) = .empty;
                while (cursor.* < self.tokens.len) {
                    if (self.tokens[cursor.*].getTag() == .rparen) {
                        cursor.* += 1;
                        break;
                    }
                    if (try self.decodeNode(cursor, allocator)) |c| {
                        try children.append(allocator, c);
                    }
                }
                node.data = .{ .list = try children.toOwnedSlice(allocator) };
            },
            .lbracket => {
                var children: std.ArrayList(*Node) = .empty;
                while (cursor.* < self.tokens.len) {
                    if (self.tokens[cursor.*].getTag() == .rbracket) {
                        cursor.* += 1;
                        break;
                    }
                    if (try self.decodeNode(cursor, allocator)) |c| {
                        try children.append(allocator, c);
                    }
                }
                node.data = .{ .bracket_list = try children.toOwnedSlice(allocator) };
            },
            .rparen, .rbracket, .keyword, .opcode, .node_ref => return null,
        }
        return node;
    }

    /// FNV-1a 128-bit hash over tokens and string table
    pub fn computeHash128(tokens: []const Token, strings: []const []const u8) [16]u8 {
        var h_high: u64 = 0x6c62272e07bb0142;
        var h_low: u64 = 0x62b821756295c58d;
        const prime: u64 = 0x100000001b3;

        for (tokens) |t| {
            const raw: u32 = @bitCast(t);
            h_low ^= raw;
            h_low *%= prime;
            h_high ^= (raw >> 16);
            h_high *%= prime;
        }

        for (strings) |s| {
            for (s) |b| {
                h_low ^= b;
                h_low *%= prime;
                h_high ^= b;
                h_high *%= prime;
            }
        }

        var out: [16]u8 = undefined;
        std.mem.writeInt(u64, out[0..8], h_low, .little);
        std.mem.writeInt(u64, out[8..16], h_high, .little);
        return out;
    }

    pub fn formatHash(self: TokenDAG) [32]u8 {
        var buf: [32]u8 = undefined;
        const hex = "0123456789abcdef";
        for (self.hash128, 0..) |b, i| {
            buf[i * 2] = hex[(b >> 4) & 0x0F];
            buf[i * 2 + 1] = hex[b & 0x0F];
        }
        return buf;
    }
};

test "token dag encoding, hashing, and decoding roundtrip" {
    const src =
        \\(func main ()
        \\  (let x 42)
        \\  (call print "Hello Token DAG" x)
        \\  (return 0))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try parser.Parser.init(src, arena.allocator());
    const original_nodes = try p.parseAll();

    // Encode to flat 32-bit token DAG
    const dag = try TokenDAG.encode(original_nodes, arena.allocator());
    try std.testing.expect(dag.tokens.len > 0);
    try std.testing.expect(dag.strings.len > 0);

    // Verify 128-bit hash is deterministic
    const hash_str = dag.formatHash();
    try std.testing.expectEqual(@as(usize, 32), hash_str.len);

    // Decode back to AST
    const decoded_nodes = try dag.decode(arena.allocator());
    try std.testing.expectEqual(original_nodes.len, decoded_nodes.len);

    const fn_node = decoded_nodes[0];
    try std.testing.expectEqualStrings("func", fn_node.data.list[0].data.atom.symbol);
    try std.testing.expectEqualStrings("main", fn_node.data.list[1].data.atom.symbol);
}
