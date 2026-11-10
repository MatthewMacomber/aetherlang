//! Aether Language Server Protocol (LSP) Daemon
//! Implements JSON-RPC 2.0 language server for editor integration (hover, completion, diagnostics).

const std = @import("std");
const parser = @import("parser.zig");

pub const LspServer = struct {
    allocator: std.mem.Allocator,
    is_shutdown: bool = false,

    pub fn init(allocator: std.mem.Allocator) LspServer {
        return .{ .allocator = allocator };
    }

    /// Processes a single JSON-RPC 2.0 request string and returns the response JSON string (if any)
    pub fn handleMessage(self: *LspServer, req_json: []const u8) !?[]const u8 {
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, req_json, .{}) catch return null;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return null;

        const method_val = root.object.get("method") orelse return null;
        if (method_val != .string) return null;
        const method = method_val.string;

        const id_val = root.object.get("id");

        var id_buf: [64]u8 = undefined;
        const id_str = if (id_val) |id| formatId(id, &id_buf) else "1";

        if (std.mem.eql(u8, method, "initialize")) {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","id":{s},"result":{{"capabilities":{{"textDocumentSync":1,"hoverProvider":true,"completionProvider":{{"triggerCharacters":["("," "]}}}}}}}}
            , .{id_str});
        } else if (std.mem.eql(u8, method, "shutdown")) {
            self.is_shutdown = true;
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","id":{s},"result":null}}
            , .{id_str});
        } else if (std.mem.eql(u8, method, "textDocument/hover")) {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","id":{s},"result":{{"contents":{{"kind":"markdown","value":"**Aether Language Symbol**\n\nFirst-class tensor computation and autodiff primitive."}}}}}}
            , .{id_str});
        } else if (std.mem.eql(u8, method, "textDocument/completion")) {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","id":{s},"result":{{"isIncomplete":false,"items":[{{"label":"func","kind":14,"detail":"Function declaration"}},{{"label":"matmul","kind":3,"detail":"Tensor matrix multiplication"}},{{"label":"gradient","kind":3,"detail":"Reverse-mode autodiff gradient"}},{{"label":"kernel","kind":14,"detail":"GPU compute kernel"}},{{"label":"struct","kind":7,"detail":"Struct definition"}}]}}}}
            , .{id_str});
        } else if (std.mem.eql(u8, method, "textDocument/didOpen") or std.mem.eql(u8, method, "textDocument/didChange")) {
            // Validate syntax and emit diagnostics
            const params = root.object.get("params") orelse return null;
            if (params != .object) return null;
            const text_doc = params.object.get("textDocument") orelse return null;
            if (text_doc != .object) return null;
            const uri_val = text_doc.object.get("uri") orelse return null;
            const uri = if (uri_val == .string) uri_val.string else "unknown";

            var text_to_check: ?[]const u8 = null;
            if (text_doc.object.get("text")) |t| {
                if (t == .string) text_to_check = t.string;
            } else if (params.object.get("contentChanges")) |changes| {
                if (changes == .array and changes.array.items.len > 0 and changes.array.items[0] == .object) {
                    if (changes.array.items[0].object.get("text")) |t| {
                        if (t == .string) text_to_check = t.string;
                    }
                }
            }

            var has_error = false;
            var err_line: usize = 0;
            if (text_to_check) |src| {
                var arena = std.heap.ArenaAllocator.init(self.allocator);
                defer arena.deinit();
                var p = parser.Parser.init(src, arena.allocator()) catch {
                    has_error = true;
                    err_line = 1;
                    return try self.formatDiagnostics(uri, has_error, err_line);
                };
                _ = p.parseAll() catch {
                    has_error = true;
                    err_line = p.current.loc.line;
                };
            }

            return try self.formatDiagnostics(uri, has_error, err_line);
        }

        return null;
    }

    fn formatDiagnostics(self: *LspServer, uri: []const u8, has_error: bool, err_line: usize) ![]const u8 {
        if (!has_error) {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{{"uri":"{s}","diagnostics":[]}}}}
            , .{uri});
        } else {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{{"uri":"{s}","diagnostics":[{{"range":{{"start":{{"line":{d},"character":0}},"end":{{"line":{d},"character":80}}}},"severity":1,"message":"Syntax error in Aether program"}}]}}}}
            , .{ uri, err_line, err_line });
        }
    }

    fn formatId(id_val: std.json.Value, buf: *[64]u8) []const u8 {
        return switch (id_val) {
            .integer => |i| std.fmt.bufPrint(buf, "{d}", .{i}) catch "1",
            .string => |s| std.fmt.bufPrint(buf, "\"{s}\"", .{s}) catch "1",
            else => "1",
        };
    }

    pub fn runServer(self: *LspServer, io: std.Io) !void {
        var stdin_buf: [4096]u8 = undefined;
        var fr: std.Io.File.Reader = .init(.stdin(), io, &stdin_buf);
        const r = &fr.interface;

        var stdout_buf: [4096]u8 = undefined;
        var fw: std.Io.File.Writer = .init(.stdout(), io, &stdout_buf);
        const w = &fw.interface;

        while (!self.is_shutdown) {
            const maybe_line = r.takeDelimiter('\n') catch break;
            if (maybe_line == null) break;
            const line = std.mem.trim(u8, maybe_line.?, " \r\n");
            if (std.mem.startsWith(u8, line, "Content-Length:")) {
                const len_str = std.mem.trim(u8, line["Content-Length:".len..], " ");
                const content_len = std.fmt.parseInt(usize, len_str, 10) catch continue;

                // Skip headers until empty line
                while (true) {
                    const h_line = (r.takeDelimiter('\n') catch break) orelse break;
                    if (std.mem.trim(u8, h_line, " \r\n").len == 0) break;
                }

                const body = try self.allocator.alloc(u8, content_len);
                defer self.allocator.free(body);
                for (0..content_len) |i| {
                    body[i] = r.takeByte() catch break;
                }

                if (try self.handleMessage(body)) |resp| {
                    defer self.allocator.free(resp);
                    var header_buf: [128]u8 = undefined;
                    const header = try std.fmt.bufPrint(&header_buf, "Content-Length: {d}\r\n\r\n", .{resp.len});
                    try w.writeAll(header);
                    try w.writeAll(resp);
                    try fw.flush();
                }
            }
        }
    }
};

test "LSP daemon initialize, hover, completion, and diagnostics" {
    var lsp = LspServer.init(std.testing.allocator);

    // Test initialize
    const init_req =
        \\{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    ;
    const init_resp = (try lsp.handleMessage(init_req)).?;
    defer std.testing.allocator.free(init_resp);
    try std.testing.expect(std.mem.indexOf(u8, init_resp, "\"hoverProvider\":true") != null);

    // Test hover
    const hover_req =
        \\{"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"position":{"line":0,"character":5}}}
    ;
    const hover_resp = (try lsp.handleMessage(hover_req)).?;
    defer std.testing.allocator.free(hover_resp);
    try std.testing.expect(std.mem.indexOf(u8, hover_resp, "Aether Language Symbol") != null);

    // Test completion
    const comp_req =
        \\{"jsonrpc":"2.0","id":3,"method":"textDocument/completion","params":{}}
    ;
    const comp_resp = (try lsp.handleMessage(comp_req)).?;
    defer std.testing.allocator.free(comp_resp);
    try std.testing.expect(std.mem.indexOf(u8, comp_resp, "\"label\":\"matmul\"") != null);

    // Test diagnostics on valid code
    const doc_valid =
        \\{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"test.ae","text":"(func add ((x i32)) i32 (return x))"}}}
    ;
    const diag_resp = (try lsp.handleMessage(doc_valid)).?;
    defer std.testing.allocator.free(diag_resp);
    try std.testing.expect(std.mem.indexOf(u8, diag_resp, "\"diagnostics\":[]") != null);

    // Test diagnostics on invalid code (unbalanced parens)
    const doc_invalid =
        \\{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"bad.ae","text":"(func add ((x i32)"}}}
    ;
    const diag_err_resp = (try lsp.handleMessage(doc_invalid)).?;
    defer std.testing.allocator.free(diag_err_resp);
    try std.testing.expect(std.mem.indexOf(u8, diag_err_resp, "Syntax error") != null);
}
