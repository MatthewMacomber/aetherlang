//! Aether Static Analyzer & Explainable AI (XAI) Hooks
//! Inspects AST computation graphs to report tensor shapes, FLOP counts,
//! memory footprints, and JSON-structured diagnostics for LLM agents.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const XAITrace = struct {
    op: []const u8,
    details: []const u8,
};

pub const AnalysisReport = struct {
    functions_count: usize = 0,
    structs_count: usize = 0,
    models_count: usize = 0,
    tensor_allocations: usize = 0,
    estimated_tensor_bytes: usize = 0,
    estimated_flops: usize = 0,
    traces: std.ArrayList(XAITrace),

    pub fn deinit(self: *AnalysisReport, allocator: std.mem.Allocator) void {
        self.traces.deinit(allocator);
    }
};

pub const StaticAnalyzer = struct {
    allocator: std.mem.Allocator,
    known_shapes: std.StringHashMap([]const usize),

    pub fn init(allocator: std.mem.Allocator) StaticAnalyzer {
        return .{
            .allocator = allocator,
            .known_shapes = std.StringHashMap([]const usize).init(allocator),
        };
    }

    pub fn deinit(self: *StaticAnalyzer) void {
        self.known_shapes.deinit();
    }

    pub fn analyze(self: *StaticAnalyzer, nodes: []const *Node) !AnalysisReport {
        var report = AnalysisReport{
            .traces = .empty,
        };

        for (nodes) |node| {
            try self.walkNode(node, &report);
        }

        return report;
    }

    fn walkNode(self: *StaticAnalyzer, node: *Node, report: *AnalysisReport) anyerror!void {
        switch (node.data) {
            .atom => {},
            .bracket_list => |children| {
                for (children) |c| try self.walkNode(c, report);
            },
            .list => |children| {
                if (children.len == 0) return;
                const head = children[0];
                if (head.data == .atom and head.data.atom == .symbol) {
                    const sym = head.data.atom.symbol;
                    if (std.mem.eql(u8, sym, "func")) {
                        report.functions_count += 1;
                    } else if (std.mem.eql(u8, sym, "struct") or std.mem.eql(u8, sym, "linear-struct")) {
                        report.structs_count += 1;
                    } else if (std.mem.eql(u8, sym, "model")) {
                        report.models_count += 1;
                    } else if (std.mem.eql(u8, sym, "let") and children.len >= 3 and children[1].data == .atom and children[1].data.atom == .symbol) {
                        const var_name = children[1].data.atom.symbol;
                        const expr = children[2];
                        if (expr.data == .list and expr.data.list.len >= 2 and expr.data.list[0].data == .atom and std.mem.eql(u8, expr.data.list[0].data.atom.symbol, "tensor")) {
                            const shape_node = expr.data.list[1];
                            const dims = if (shape_node.data == .bracket_list) shape_node.data.bracket_list else shape_node.data.list;
                            var dim_list: std.ArrayList(usize) = .empty;
                            for (dims) |d| {
                                if (d.data == .atom and d.data.atom == .number) {
                                    try dim_list.append(self.allocator, @intFromFloat(d.data.atom.number));
                                }
                            }
                            try self.known_shapes.put(var_name, try dim_list.toOwnedSlice(self.allocator));
                        }
                    } else if (std.mem.eql(u8, sym, "tensor") and children.len >= 3) {
                        report.tensor_allocations += 1;
                        const shape_node = children[1];
                        const dims = if (shape_node.data == .bracket_list) shape_node.data.bracket_list else shape_node.data.list;
                        var elems: usize = 1;
                        for (dims) |d| {
                            if (d.data == .atom and d.data.atom == .number) {
                                elems *= @as(usize, @intFromFloat(d.data.atom.number));
                            }
                        }
                        report.estimated_tensor_bytes += elems * @sizeOf(f64);
                        try report.traces.append(self.allocator, .{
                            .op = "tensor_alloc",
                            .details = "Allocated tensor on heap",
                        });
                    } else if (std.mem.eql(u8, sym, "matmul") or std.mem.eql(u8, sym, "fused_matmul_relu")) {
                        var flops: usize = 1000;
                        if (children.len >= 3 and children[1].data == .atom and children[1].data.atom == .symbol and children[2].data == .atom and children[2].data.atom == .symbol) {
                            const s1 = self.known_shapes.get(children[1].data.atom.symbol);
                            const s2 = self.known_shapes.get(children[2].data.atom.symbol);
                            if (s1 != null and s2 != null and s1.?.len == 2 and s2.?.len == 2) {
                                const m = s1.?[0];
                                const k = s1.?[1];
                                const n = s2.?[1];
                                flops = 2 * m * n * k;
                            }
                        }
                        report.estimated_flops += flops;
                        try report.traces.append(self.allocator, .{
                            .op = "matmul",
                            .details = "Matrix multiplication operator",
                        });
                    }
                }

                for (children) |c| try self.walkNode(c, report);
            },
        }
    }
};

test "static analyzer calculates allocations and reports" {
    const src =
        \\(struct Point (x f64) (y f64))
        \\(func main ()
        \\  (let a (tensor [10 20] [1.0]))
        \\  (let b (tensor [20 30] [1.0]))
        \\  (let c (matmul a b)))
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var sa = StaticAnalyzer.init(arena.allocator());
    defer sa.deinit();
    var rep = try sa.analyze(nodes);
    defer rep.deinit(arena.allocator());

    try std.testing.expectEqual(@as(usize, 1), rep.structs_count);
    try std.testing.expectEqual(@as(usize, 1), rep.functions_count);
    try std.testing.expectEqual(@as(usize, 2), rep.tensor_allocations);
    // 2 * 10 * 30 * 20 = 12000 FLOPs
    try std.testing.expectEqual(@as(usize, 12000), rep.estimated_flops);
}
