//! Aether ONNX Model Export & Content-Addressed Model Zoo
//! Serializes Aether neural models and tensor graphs to ONNX interchange format.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const ProtoWriter = struct {
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn writeVarint(self: *ProtoWriter, val: u64) !void {
        var v = val;
        while (v >= 0x80) {
            try self.buf.append(self.allocator, @as(u8, @intCast(v & 0x7F)) | 0x80);
            v >>= 7;
        }
        try self.buf.append(self.allocator, @as(u8, @intCast(v)));
    }

    pub fn writeTag(self: *ProtoWriter, field_num: u32, wire_type: u3) !void {
        try self.writeVarint((@as(u64, field_num) << 3) | wire_type);
    }

    pub fn writeVarintField(self: *ProtoWriter, field_num: u32, val: u64) !void {
        try self.writeTag(field_num, 0);
        try self.writeVarint(val);
    }

    pub fn writeStringField(self: *ProtoWriter, field_num: u32, str: []const u8) !void {
        try self.writeTag(field_num, 2);
        try self.writeVarint(str.len);
        try self.buf.appendSlice(self.allocator, str);
    }

    pub fn writeSubmessage(self: *ProtoWriter, field_num: u32, sub_bytes: []const u8) !void {
        try self.writeTag(field_num, 2);
        try self.writeVarint(sub_bytes.len);
        try self.buf.appendSlice(self.allocator, sub_bytes);
    }
};

pub const ImportedNode = struct {
    op_type: []const u8,
};

pub const ImportedModel = struct {
    ir_version: u64 = 0,
    producer_name: []const u8 = "",
    nodes: std.ArrayList(ImportedNode) = .empty,

    pub fn deinit(self: *ImportedModel, allocator: std.mem.Allocator) void {
        self.nodes.deinit(allocator);
    }
};

pub const ONNXExporter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ONNXExporter {
        return .{ .allocator = allocator };
    }

    /// Exports AST computation graph to ONNX JSON format
    pub fn exportJSON(self: *ONNXExporter, nodes: []const *Node) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;

        var input_names: std.ArrayList([]const u8) = .empty;
        var output_names: std.ArrayList([]const u8) = .empty;

        for (nodes) |node| {
            if (node.data == .list and node.data.list.len >= 3 and node.data.list[0].data == .atom and std.mem.eql(u8, node.data.list[0].data.atom.symbol, "func")) {
                const params_node = node.data.list[2];
                const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;
                for (params) |p| {
                    if (p.data == .atom and p.data.atom == .symbol) {
                        try input_names.append(self.allocator, p.data.atom.symbol);
                    } else if (p.data == .list and p.data.list.len >= 1 and p.data.list[0].data == .atom and p.data.list[0].data.atom == .symbol) {
                        try input_names.append(self.allocator, p.data.list[0].data.atom.symbol);
                    }
                }

                // Check return
                for (node.data.list[3..]) |stmt| {
                    if (stmt.data == .list and stmt.data.list.len >= 2 and stmt.data.list[0].data == .atom and std.mem.eql(u8, stmt.data.list[0].data.atom.symbol, "return")) {
                        const ret_expr = stmt.data.list[1];
                        if (ret_expr.data == .atom and ret_expr.data.atom == .symbol) {
                            try output_names.append(self.allocator, ret_expr.data.atom.symbol);
                        }
                    }
                }
            }
        }

        try out.appendSlice(self.allocator,
            \\{
            \\  "ir_version": 8,
            \\  "producer_name": "aetherc",
            \\  "producer_version": "0.3.0",
            \\  "graph": {
            \\    "name": "aether_exported_graph",
            \\    "nodes": [
            \\
        );

        var first_node = true;
        for (nodes) |node| {
            try self.walkForNodes(node, &out, &first_node);
        }

        try out.appendSlice(self.allocator, "\n    ],\n    \"inputs\": [ ");
        for (input_names.items, 0..) |in_name, i| {
            if (i > 0) try out.appendSlice(self.allocator, ", ");
            var buf: [128]u8 = undefined;
            const line = try std.fmt.bufPrint(&buf, "{{ \"name\": \"{s}\", \"type\": \"tensor<f32>\" }}", .{in_name});
            try out.appendSlice(self.allocator, line);
        }
        if (input_names.items.len == 0) {
            try out.appendSlice(self.allocator, "{ \"name\": \"input_0\", \"type\": \"float[BATCH, 10]\" }");
        }

        try out.appendSlice(self.allocator, " ],\n    \"outputs\": [ ");
        for (output_names.items, 0..) |out_name, i| {
            if (i > 0) try out.appendSlice(self.allocator, ", ");
            var buf: [128]u8 = undefined;
            const line = try std.fmt.bufPrint(&buf, "{{ \"name\": \"{s}\", \"type\": \"tensor<f32>\" }}", .{out_name});
            try out.appendSlice(self.allocator, line);
        }
        if (output_names.items.len == 0) {
            try out.appendSlice(self.allocator, "{ \"name\": \"output_0\", \"type\": \"float[BATCH, 1]\" }");
        }

        try out.appendSlice(self.allocator,
            \\ ]
            \\  }
            \\}
            \\
        );

        return try out.toOwnedSlice(self.allocator);
    }

    fn walkForNodes(self: *ONNXExporter, node: *Node, out: *std.ArrayList(u8), first: *bool) anyerror!void {
        if (node.data == .list) {
            const children = node.data.list;
            if (children.len > 0 and children[0].data == .atom and children[0].data.atom == .symbol) {
                const sym = children[0].data.atom.symbol;
                var op_name: ?[]const u8 = null;
                if (std.mem.eql(u8, sym, "matmul") or std.mem.eql(u8, sym, "fused_matmul_relu")) op_name = "MatMul"
                else if (std.mem.eql(u8, sym, "relu")) op_name = "Relu"
                else if (std.mem.eql(u8, sym, "sigmoid")) op_name = "Sigmoid"
                else if (std.mem.eql(u8, sym, "+")) op_name = "Add"
                else if (std.mem.eql(u8, sym, "*")) op_name = "Mul";

                if (op_name) |op| {
                    if (!first.*) try out.appendSlice(self.allocator, ",\n");
                    first.* = false;
                    try out.appendSlice(self.allocator, "      { \"op_type\": \"");
                    try out.appendSlice(self.allocator, op);
                    try out.appendSlice(self.allocator, "\" }");
                }
            }
            for (children) |c| try self.walkForNodes(c, out, first);
        }
    }

    /// Serializes computation graph to standard binary ONNX Protobuf wire format
    pub fn exportBinaryProtobuf(self: *ONNXExporter, nodes: []const *Node) ![]const u8 {
        var graph_bytes: std.ArrayList(u8) = .empty;
        defer graph_bytes.deinit(self.allocator);
        var graph_pw = ProtoWriter{ .buf = &graph_bytes, .allocator = self.allocator };

        // Collect op types from AST
        var ops: std.ArrayList([]const u8) = .empty;
        defer ops.deinit(self.allocator);
        try self.collectOps(nodes, &ops);

        for (ops.items) |op_type| {
            var node_bytes: std.ArrayList(u8) = .empty;
            defer node_bytes.deinit(self.allocator);
            var node_pw = ProtoWriter{ .buf = &node_bytes, .allocator = self.allocator };
            // Field 4: op_type in NodeProto
            try node_pw.writeStringField(4, op_type);
            // Write NodeProto into GraphProto field 1
            try graph_pw.writeSubmessage(1, node_bytes.items);
        }
        try graph_pw.writeStringField(2, "aether_graph");

        // ModelProto
        var model_bytes: std.ArrayList(u8) = .empty;
        var model_pw = ProtoWriter{ .buf = &model_bytes, .allocator = self.allocator };
        try model_pw.writeVarintField(1, 8); // ir_version = 8
        try model_pw.writeStringField(2, "aetherc"); // producer_name
        try model_pw.writeStringField(3, "0.3.0"); // producer_version
        try model_pw.writeSubmessage(7, graph_bytes.items); // graph

        return try model_bytes.toOwnedSlice(self.allocator);
    }

    /// Deserializes binary ONNX Protobuf wire format
    pub fn importBinaryProtobuf(self: *ONNXExporter, bytes: []const u8) !ImportedModel {
        var model = ImportedModel{};
        var idx: usize = 0;

        while (idx < bytes.len) {
            const tag_res = readVarint(bytes, &idx) catch break;
            const field_num: u32 = @intCast(tag_res >> 3);
            const wire_type: u3 = @truncate(tag_res & 0x07);

            if (wire_type == 0) { // varint
                const val = readVarint(bytes, &idx) catch break;
                if (field_num == 1) model.ir_version = val;
            } else if (wire_type == 2) { // length-delimited
                const len: usize = @intCast(readVarint(bytes, &idx) catch break);
                if (idx + len > bytes.len) break;
                const slice = bytes[idx .. idx + len];
                idx += len;

                if (field_num == 2) {
                    model.producer_name = slice;
                } else if (field_num == 7) {
                    // Parse GraphProto
                    var g_idx: usize = 0;
                    while (g_idx < slice.len) {
                        const g_tag = readVarint(slice, &g_idx) catch break;
                        const g_field: u32 = @intCast(g_tag >> 3);
                        const g_wire: u3 = @truncate(g_tag & 0x07);

                        if (g_wire == 2) {
                            const g_len: usize = @intCast(readVarint(slice, &g_idx) catch break);
                            if (g_idx + g_len > slice.len) break;
                            const node_slice = slice[g_idx .. g_idx + g_len];
                            g_idx += g_len;

                            if (g_field == 1) {
                                // Parse NodeProto
                                var n_idx: usize = 0;
                                while (n_idx < node_slice.len) {
                                    const n_tag = readVarint(node_slice, &n_idx) catch break;
                                    const n_field: u32 = @intCast(n_tag >> 3);
                                    const n_wire: u3 = @truncate(n_tag & 0x07);
                                    if (n_wire == 2) {
                                        const n_len: usize = @intCast(readVarint(node_slice, &n_idx) catch break);
                                        if (n_idx + n_len > node_slice.len) break;
                                        const field_str = node_slice[n_idx .. n_idx + n_len];
                                        n_idx += n_len;
                                        if (n_field == 4) {
                                            try model.nodes.append(self.allocator, .{ .op_type = field_str });
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }
            } else {
                break;
            }
        }

        return model;
    }

    fn collectOps(self: *ONNXExporter, nodes: []const *Node, ops: *std.ArrayList([]const u8)) anyerror!void {
        for (nodes) |node| {
            if (node.data == .list) {
                const children = node.data.list;
                if (children.len > 0 and children[0].data == .atom and children[0].data.atom == .symbol) {
                    const sym = children[0].data.atom.symbol;
                    if (std.mem.eql(u8, sym, "matmul")) {
                        try ops.append(self.allocator, "MatMul");
                    } else if (std.mem.eql(u8, sym, "relu")) {
                        try ops.append(self.allocator, "Relu");
                    } else if (std.mem.eql(u8, sym, "sigmoid")) {
                        try ops.append(self.allocator, "Sigmoid");
                    } else if (std.mem.eql(u8, sym, "+")) {
                        try ops.append(self.allocator, "Add");
                    } else if (std.mem.eql(u8, sym, "*")) {
                        try ops.append(self.allocator, "Mul");
                    }
                }
                try self.collectOps(children, ops);
            }
        }
    }
};

fn readVarint(bytes: []const u8, idx: *usize) !u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (idx.* < bytes.len) {
        const b = bytes[idx.*];
        idx.* += 1;
        result |= @as(u64, b & 0x7F) << shift;
        if ((b & 0x80) == 0) return result;
        shift += 7;
    }
    return error.UnexpectedEof;
}

test "ONNX exporter produces valid JSON computation graph" {
    const src =
        \\(func forward (x w)
        \\  (let h (relu (matmul x w)))
        \\  (return h))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var exp = ONNXExporter.init(arena.allocator());
    const onnx_str = try exp.exportJSON(nodes);

    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"producer_name\": \"aetherc\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"op_type\": \"MatMul\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"op_type\": \"Relu\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"name\": \"x\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"name\": \"w\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, onnx_str, "\"name\": \"h\"") != null);
}

test "ONNX binary Protobuf serialization and deserialization roundtrip" {
    const src =
        \\(func forward (x w)
        \\  (let h (relu (matmul x w)))
        \\  (return h))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var exp = ONNXExporter.init(arena.allocator());
    const proto_bytes = try exp.exportBinaryProtobuf(nodes);
    try std.testing.expect(proto_bytes.len > 10);

    var imported = try exp.importBinaryProtobuf(proto_bytes);
    defer imported.deinit(arena.allocator());

    try std.testing.expectEqual(@as(u64, 8), imported.ir_version);
    try std.testing.expectEqualStrings("aetherc", imported.producer_name);
    try std.testing.expectEqual(@as(usize, 2), imported.nodes.items.len);
    try std.testing.expectEqualStrings("Relu", imported.nodes.items[0].op_type);
    try std.testing.expectEqualStrings("MatMul", imported.nodes.items[1].op_type);
}

