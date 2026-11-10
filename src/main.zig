//! Aether Compiler (aetherc) CLI Driver
//! Compiles .ae Aether programs to native executables via Zig.

const std = @import("std");
const parser = @import("parser.zig");
const codegen = @import("codegen.zig");
const types = @import("types.zig");
const macros = @import("macros.zig");
const sweet = @import("sweet.zig");
const token_dag = @import("token_dag.zig");
const ffi = @import("ffi.zig");
const gpu = @import("gpu.zig");
const analyze = @import("analyze.zig");
const onnx = @import("onnx.zig");
const repl = @import("repl.zig");
const air = @import("air.zig");
const lsp = @import("lsp.zig");
const runtime = @import("runtime.zig");

const RUNTIME_SOURCE = @embedFile("runtime.zig");
const RUNTIME_GRAPHICS_SOURCE = @embedFile("runtime_graphics.zig");
const RUNTIME_TUI_SOURCE = @embedFile("runtime_tui.zig");
const RUNTIME_AUDIO_SOURCE = @embedFile("runtime_audio.zig");
const RUNTIME_PLATFORM_SOURCE = @embedFile("runtime_platform.zig");
const RUNTIME_GUI_SOURCE = @embedFile("runtime_gui.zig");
const RUNTIME_ENGINE_SOURCE = @embedFile("runtime_engine.zig");
const RUNTIME_MEM_SOURCE = @embedFile("runtime_mem.zig");
const autofree = @import("autofree.zig");

const VERSION = "0.3.0";

const Usage =
    \\Aether Compiler (aetherc) v
++ VERSION ++
    \\
    \\Usage:
    \\  aetherc build <file.ae> [-o <out>] [--release] [--console|--no-console] [--emit-zig] [--emit-air] [-O0|-O1|-O2|-O3] [--mem <autofree|arena|manual|gc>] [--target <wasm32|gpu|cuda|rocm|spirv>]
    \\  aetherc run <file.ae> [-O0|-O1|-O2|-O3] [--mem <autofree|arena|manual|gc>]
    \\  aetherc check <file.ae>
    \\  aetherc fmt --to-canonical|--to-sweet <file.ae>
    \\  aetherc hash <file.ae>
    \\  aetherc bind [--language c|cpp] <file.ae>
    \\  aetherc analyze <file.ae>
    \\  aetherc export [--format onnx] <file.ae>
    \\  aetherc repl
    \\  aetherc node [--port <port>]
    \\  aetherc lsp
    \\  aetherc pkg publish <file.ae> | fetch <hash> | list
    \\  aetherc -v, --version
    \\  aetherc -h, --help
    \\
    \\Memory Management (--mem):
    \\  autofree  (default) Hybrid compile-time deterministic scope frees + fallback GC
    \\  arena     Global arena + scoped (arena ...) blocks
    \\  manual    Explicit (free x) with leak reporting on exit
    \\  gc        Precise mark-and-sweep tracing garbage collector
    \\
;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena_allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena_allocator);

    if (args.len < 2) {
        try printStdout(io, Usage);
        std.process.exit(1);
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "--help")) {
        try printStdout(io, Usage);
        return;
    } else if (std.mem.eql(u8, cmd, "-v") or std.mem.eql(u8, cmd, "--version")) {
        try printStdout(io, "aetherc v" ++ VERSION ++ "\n");
        return;
    } else if (std.mem.eql(u8, cmd, "fmt")) {
        if (args.len < 4) {
            try printStderr(io, "Usage: aetherc fmt --to-canonical|--to-sweet <file.ae>\n");
            std.process.exit(1);
        }
        const mode = args[2];
        const fmt_file = args[3];

        var in_buf: [1024 * 1024]u8 = undefined;
        const file_bytes = std.Io.Dir.cwd().readFile(io, fmt_file, &in_buf) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ fmt_file, @errorName(err) }) catch "Read error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };

        var transpiler = sweet.SweetTranspiler.init(arena_allocator);
        if (std.mem.eql(u8, mode, "--to-canonical")) {
            const canonical = try transpiler.toCanonical(file_bytes);
            try printStdout(io, canonical);
            return;
        } else if (std.mem.eql(u8, mode, "--to-sweet")) {
            var p = try parser.Parser.init(file_bytes, arena_allocator);
            const nodes = try p.parseAll();
            const sweet_text = try transpiler.toSweet(nodes);
            try printStdout(io, sweet_text);
            return;
        } else {
            try printStderr(io, "Error: Unknown format flag. Use --to-canonical or --to-sweet\n");
            std.process.exit(1);
        }
    } else if (std.mem.eql(u8, cmd, "hash")) {
        if (args.len < 3) {
            try printStderr(io, "Usage: aetherc hash <file.ae>\n");
            std.process.exit(1);
        }
        const hash_file = args[2];
        var in_buf: [1024 * 1024]u8 = undefined;
        const file_bytes = std.Io.Dir.cwd().readFile(io, hash_file, &in_buf) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ hash_file, @errorName(err) }) catch "Read error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };

        var source_bytes: []const u8 = file_bytes;
        if (sweet.isSweetSyntax(source_bytes)) {
            var transpiler = sweet.SweetTranspiler.init(arena_allocator);
            source_bytes = try transpiler.toCanonical(source_bytes);
        }

        var p = try parser.Parser.init(source_bytes, arena_allocator);
        const nodes = try p.parseAll();
        const dag = try token_dag.TokenDAG.encode(nodes, arena_allocator);
        const hash_str = dag.formatHash();

        try printStdout(io, "#0x");
        try printStdout(io, &hash_str);
        try printStdout(io, "\n");
        return;
    } else if (std.mem.eql(u8, cmd, "bind")) {
        if (args.len < 3) {
            try printStderr(io, "Usage: aetherc bind [--language c|cpp] <file.ae>\n");
            std.process.exit(1);
        }
        var bind_file: ?[]const u8 = null;
        var is_cpp = false;
        var j: usize = 2;
        while (j < args.len) : (j += 1) {
            if (std.mem.eql(u8, args[j], "--language") and j + 1 < args.len) {
                j += 1;
                is_cpp = std.mem.indexOf(u8, args[j], "cpp") != null;
            } else if (!std.mem.startsWith(u8, args[j], "-")) {
                bind_file = args[j];
            }
        }

        if (bind_file == null) {
            try printStderr(io, "Error: No file specified for bind\n");
            std.process.exit(1);
        }

        var in_buf: [1024 * 1024]u8 = undefined;
        const file_bytes = std.Io.Dir.cwd().readFile(io, bind_file.?, &in_buf) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ bind_file.?, @errorName(err) }) catch "Read error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };

        var source_bytes: []const u8 = file_bytes;
        if (sweet.isSweetSyntax(source_bytes)) {
            var transpiler = sweet.SweetTranspiler.init(arena_allocator);
            source_bytes = try transpiler.toCanonical(source_bytes);
        }

        var p = try parser.Parser.init(source_bytes, arena_allocator);
        const nodes = try p.parseAll();

        var gen = ffi.FfiBindingGen.init(arena_allocator);
        const header = try gen.generateHeader(nodes, is_cpp);
        try printStdout(io, header);
        return;
    } else if (std.mem.eql(u8, cmd, "analyze")) {
        if (args.len < 3) {
            try printStderr(io, "Usage: aetherc analyze <file.ae>\n");
            std.process.exit(1);
        }
        const an_file = args[2];
        var in_buf: [1024 * 1024]u8 = undefined;
        const file_bytes = std.Io.Dir.cwd().readFile(io, an_file, &in_buf) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ an_file, @errorName(err) }) catch "Read error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };

        var source_bytes: []const u8 = file_bytes;
        if (sweet.isSweetSyntax(source_bytes)) {
            var transpiler = sweet.SweetTranspiler.init(arena_allocator);
            source_bytes = try transpiler.toCanonical(source_bytes);
        }

        var p = try parser.Parser.init(source_bytes, arena_allocator);
        const nodes = try p.parseAll();

        var sa = analyze.StaticAnalyzer.init(arena_allocator);
        var rep = try sa.analyze(nodes);
        defer rep.deinit(arena_allocator);

        var buf: [1024]u8 = undefined;
        const out_msg = try std.fmt.bufPrint(&buf,
            \\=== Aether Static Analysis & XAI Report ===
            \\Functions: {d}
            \\Structs:   {d}
            \\Models:    {d}
            \\Tensor Allocations: {d}
            \\Estimated Tensor Memory: {d} bytes
            \\Estimated FLOPs: {d}
            \\Computation Traces: {d}
            \\
        , .{
            rep.functions_count,
            rep.structs_count,
            rep.models_count,
            rep.tensor_allocations,
            rep.estimated_tensor_bytes,
            rep.estimated_flops,
            rep.traces.items.len,
        });
        try printStdout(io, out_msg);
        return;
    } else if (std.mem.eql(u8, cmd, "export")) {
        if (args.len < 3) {
            try printStderr(io, "Usage: aetherc export [--format onnx] <file.ae>\n");
            std.process.exit(1);
        }
        var exp_file: ?[]const u8 = null;
        for (args[2..]) |arg| {
            if (!std.mem.startsWith(u8, arg, "-")) exp_file = arg;
        }
        if (exp_file == null) {
            try printStderr(io, "Error: No file specified for export\n");
            std.process.exit(1);
        }

        var in_buf: [1024 * 1024]u8 = undefined;
        const file_bytes = std.Io.Dir.cwd().readFile(io, exp_file.?, &in_buf) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ exp_file.?, @errorName(err) }) catch "Read error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };

        var source_bytes: []const u8 = file_bytes;
        if (sweet.isSweetSyntax(source_bytes)) {
            var transpiler = sweet.SweetTranspiler.init(arena_allocator);
            source_bytes = try transpiler.toCanonical(source_bytes);
        }

        var p = try parser.Parser.init(source_bytes, arena_allocator);
        const nodes = try p.parseAll();

        var exporter = onnx.ONNXExporter.init(arena_allocator);
        const onnx_json = try exporter.exportJSON(nodes);
        try printStdout(io, onnx_json);
        return;
    } else if (std.mem.eql(u8, cmd, "repl")) {
        try printStdout(io, "Aether Interactive REPL (v" ++ VERSION ++ ")\nType an expression e.g. (+ 10 20) or (exit) to quit.\n");
        var env = repl.REPLEnvironment.init(arena_allocator);
        defer env.deinit();

        var stdin_buf: [1024]u8 = undefined;
        var fr: std.Io.File.Reader = .init(.stdin(), io, &stdin_buf);
        const r = &fr.interface;

        while (true) {
            try printStdout(io, "ae> ");
            const maybe_line = r.takeDelimiter('\n') catch break;
            if (maybe_line == null) break;
            const line = maybe_line.?;
            if (line.len == 0) continue;
            const trimmed = std.mem.trim(u8, line, " \r\n");
            if (trimmed.len == 0) continue;
            if (std.mem.eql(u8, trimmed, "(exit)") or std.mem.eql(u8, trimmed, "quit")) break;

            var p = parser.Parser.init(trimmed, arena_allocator) catch {
                try printStderr(io, "Parse error\n");
                continue;
            };
            const nodes = p.parseAll() catch {
                try printStderr(io, "Syntax error\n");
                continue;
            };

            for (nodes) |node| {
                const val = env.eval(node) catch |err| {
                    var err_buf: [128]u8 = undefined;
                    const msg = std.fmt.bufPrint(&err_buf, "Eval error: {s}\n", .{@errorName(err)}) catch "Eval error\n";
                    try printStderr(io, msg);
                    continue;
                };
                _ = runtime.print(io, &[_]runtime.Value{val});
            }
        }
        return;
    } else if (std.mem.eql(u8, cmd, "node")) {
        var port: u16 = 9000;
        var p_idx: usize = 2;
        while (p_idx < args.len) : (p_idx += 1) {
            if (std.mem.eql(u8, args[p_idx], "--port") and p_idx + 1 < args.len) {
                p_idx += 1;
                port = std.fmt.parseInt(u16, args[p_idx], 10) catch 9000;
            }
        }
        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Starting Aether Distributed Cluster Node on 127.0.0.1:{d}...\n", .{port}) catch "Starting node...\n";
        try printStdout(io, msg);
        _ = runtime.clusterStart(runtime.number(@floatFromInt(port)));
        var count: usize = 0;
        while (count < 1) {
            runtime.platform.sleepMs(100);
            count += 1;
        }
        return;
    } else if (std.mem.eql(u8, cmd, "lsp")) {
        var server = lsp.LspServer.init(arena_allocator);
        try server.runServer(io);
        return;
    } else if (std.mem.eql(u8, cmd, "pkg")) {
        if (args.len < 3) {
            try printStderr(io, "Usage: aetherc pkg publish <file.ae> | fetch <hash> | list\n");
            std.process.exit(1);
        }
        const sub_cmd = args[2];
        try std.Io.Dir.cwd().createDirPath(io, ".aepkg/cache");

        if (std.mem.eql(u8, sub_cmd, "publish")) {
            if (args.len < 4) {
                try printStderr(io, "Error: 'pkg publish' requires a file\n");
                std.process.exit(1);
            }
            const pkg_file = args[3];
            var in_buf: [1024 * 1024]u8 = undefined;
            const pkg_src = std.Io.Dir.cwd().readFile(io, pkg_file, &in_buf) catch |err| {
                var err_buf: [256]u8 = undefined;
                const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ pkg_file, @errorName(err) }) catch "Read error\n";
                try printStderr(io, msg);
                std.process.exit(1);
            };

            var p_pkg = try parser.Parser.init(pkg_src, arena_allocator);
            const p_nodes = try p_pkg.parseAll();
            const dag = try token_dag.TokenDAG.encode(p_nodes, arena_allocator);
            const hash_hex = dag.formatHash();

            var cache_path_buf: [256]u8 = undefined;
            const cache_path = try std.fmt.bufPrint(&cache_path_buf, ".aepkg/cache/{s}.ae", .{hash_hex});
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cache_path, .data = pkg_src });

            var out_buf: [256]u8 = undefined;
            const out_msg = try std.fmt.bufPrint(&out_buf, "Published package: #0x{s} ({s})\n", .{ hash_hex, pkg_file });
            try printStdout(io, out_msg);
            return;
        } else if (std.mem.eql(u8, sub_cmd, "fetch")) {
            if (args.len < 4) {
                try printStderr(io, "Error: 'pkg fetch' requires a hash\n");
                std.process.exit(1);
            }
            const target_hash = args[3];
            const clean_hash = if (std.mem.startsWith(u8, target_hash, "#0x")) target_hash[3..] else if (std.mem.startsWith(u8, target_hash, "0x")) target_hash[2..] else target_hash;

            try std.Io.Dir.cwd().createDirPath(io, "pkg");
            var cache_path_buf: [256]u8 = undefined;
            const cache_path = try std.fmt.bufPrint(&cache_path_buf, ".aepkg/cache/{s}.ae", .{clean_hash});
            var in_buf: [1024 * 1024]u8 = undefined;
            const cached_data = std.Io.Dir.cwd().readFile(io, cache_path, &in_buf) catch {
                try printStderr(io, "Error: Package hash not found in registry cache\n");
                std.process.exit(1);
            };

            var out_path_buf: [256]u8 = undefined;
            const out_path = try std.fmt.bufPrint(&out_path_buf, "pkg/{s}.ae", .{clean_hash});
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = cached_data });

            var out_buf: [256]u8 = undefined;
            const out_msg = try std.fmt.bufPrint(&out_buf, "Fetched package #0x{s} to {s}\n", .{ clean_hash, out_path });
            try printStdout(io, out_msg);
            return;
        } else if (std.mem.eql(u8, sub_cmd, "list")) {
            try printStdout(io, "=== Local Package Cache (.aepkg/cache) ===\n");
            var dir = std.Io.Dir.cwd().openDir(io, ".aepkg/cache", .{ .iterate = true }) catch {
                try printStdout(io, "No packages published yet.\n");
                return;
            };
            defer dir.close(io);
            var it = dir.iterate();
            var count: usize = 0;
            while (try it.next(io)) |entry| {
                if (std.mem.endsWith(u8, entry.name, ".ae")) {
                    count += 1;
                    var buf: [256]u8 = undefined;
                    const line = try std.fmt.bufPrint(&buf, "  - #0x{s}\n", .{entry.name[0 .. entry.name.len - 3]});
                    try printStdout(io, line);
                }
            }
            if (count == 0) try printStdout(io, "No packages found.\n");
            return;
        } else {
            try printStderr(io, "Error: Unknown pkg subcommand. Expected publish, fetch, or list.\n");
            std.process.exit(1);
        }
    }

    var is_run = false;
    var is_check = false;
    var emit_air = false;
    var opt_level: u8 = 2; // Default -O2
    var mem_mode: runtime.mem.MemMode = .autofree; // Default: autofree
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var emit_zig = false;
    var target_triple: ?[]const u8 = null;
    var is_release = false;
    var console_override: ?bool = null;

    if (std.mem.eql(u8, cmd, "check")) {
        is_check = true;
        if (args.len < 3) {
            try printStderr(io, "Error: 'check' requires an input .ae file\n");
            std.process.exit(1);
        }
        input_path = args[2];
    } else if (std.mem.eql(u8, cmd, "run")) {
        is_run = true;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "-O0")) {
                opt_level = 0;
            } else if (std.mem.eql(u8, arg, "-O1")) {
                opt_level = 1;
            } else if (std.mem.eql(u8, arg, "-O2")) {
                opt_level = 2;
            } else if (std.mem.eql(u8, arg, "-O3")) {
                opt_level = 3;
            } else if (std.mem.eql(u8, arg, "--mem") and i + 1 < args.len) {
                i += 1;
                const m = args[i];
                if (std.mem.eql(u8, m, "autofree")) {
                    mem_mode = .autofree;
                } else if (std.mem.eql(u8, m, "arena")) {
                    mem_mode = .arena;
                } else if (std.mem.eql(u8, m, "manual")) {
                    mem_mode = .manual;
                } else if (std.mem.eql(u8, m, "gc")) {
                    mem_mode = .gc;
                } else if (std.mem.eql(u8, m, "gc-stress") or std.mem.eql(u8, m, "gc_stress")) {
                    mem_mode = .gc_stress;
                } else {
                    var err_buf: [256]u8 = undefined;
                    const msg = std.fmt.bufPrint(&err_buf, "Error: Unknown memory mode '{s}'. Expected autofree, arena, manual, or gc.\n", .{m}) catch "Error: Unknown memory mode\n";
                    try printStderr(io, msg);
                    std.process.exit(1);
                }
            } else if (input_path == null and !std.mem.startsWith(u8, arg, "-")) {
                input_path = arg;
            }
        }
    } else if (std.mem.eql(u8, cmd, "build")) {
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "-o") and i + 1 < args.len) {
                i += 1;
                output_path = args[i];
            } else if (std.mem.eql(u8, arg, "--release")) {
                is_release = true;
            } else if (std.mem.eql(u8, arg, "--console")) {
                console_override = true;
            } else if (std.mem.eql(u8, arg, "--no-console")) {
                console_override = false;
            } else if (std.mem.eql(u8, arg, "--emit-zig")) {
                emit_zig = true;
            } else if (std.mem.eql(u8, arg, "--emit-air")) {
                emit_air = true;
            } else if (std.mem.eql(u8, arg, "--check")) {
                is_check = true;
            } else if (std.mem.eql(u8, arg, "-O0")) {
                opt_level = 0;
            } else if (std.mem.eql(u8, arg, "-O1")) {
                opt_level = 1;
            } else if (std.mem.eql(u8, arg, "-O2")) {
                opt_level = 2;
            } else if (std.mem.eql(u8, arg, "-O3")) {
                opt_level = 3;
            } else if (std.mem.eql(u8, arg, "--mem") and i + 1 < args.len) {
                i += 1;
                const m = args[i];
                if (std.mem.eql(u8, m, "autofree")) {
                    mem_mode = .autofree;
                } else if (std.mem.eql(u8, m, "arena")) {
                    mem_mode = .arena;
                } else if (std.mem.eql(u8, m, "manual")) {
                    mem_mode = .manual;
                } else if (std.mem.eql(u8, m, "gc")) {
                    mem_mode = .gc;
                } else if (std.mem.eql(u8, m, "gc-stress") or std.mem.eql(u8, m, "gc_stress")) {
                    mem_mode = .gc_stress;
                } else {
                    var err_buf: [256]u8 = undefined;
                    const msg = std.fmt.bufPrint(&err_buf, "Error: Unknown memory mode '{s}'. Expected autofree, arena, manual, or gc.\n", .{m}) catch "Error: Unknown memory mode\n";
                    try printStderr(io, msg);
                    std.process.exit(1);
                }
            } else if (std.mem.eql(u8, arg, "--target") and i + 1 < args.len) {
                i += 1;
                target_triple = args[i];
            } else if (input_path == null and !std.mem.startsWith(u8, arg, "-")) {
                input_path = arg;
            }
        }
    } else if (!std.mem.startsWith(u8, cmd, "-")) {
        // Direct file argument defaults to run
        is_run = true;
        input_path = cmd;
    } else {
        try printStderr(io, "Unknown command or flag\n");
        std.process.exit(1);
    }

    if (input_path == null) {
        try printStderr(io, "Error: No input file specified\n");
        std.process.exit(1);
    }

    // Read input file
    var in_buf: [1024 * 1024]u8 = undefined;
    const raw_source = std.Io.Dir.cwd().readFile(io, input_path.?, &in_buf) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Error reading file '{s}': {s}\n", .{ input_path.?, @errorName(err) }) catch "Read error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    var source: []const u8 = raw_source;
    if (sweet.isSweetSyntax(source)) {
        var transpiler = sweet.SweetTranspiler.init(arena_allocator);
        source = transpiler.toCanonical(source) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Sweet syntax transpilation error: {s}\n", .{@errorName(err)}) catch "Transpilation error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };
    }

    // Parse S-expression AST
    var p = parser.Parser.init(source, arena_allocator) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Parse initialization error: {s}\n", .{@errorName(err)}) catch "Parse error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    const ast_nodes = p.parseAll() catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Syntax error at line {d}, col {d}: {s}\n", .{ p.current.loc.line, p.current.loc.col, @errorName(err) }) catch "Syntax error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    // Resolve content-addressed package imports e.g. (import "#0x...")
    var resolved_nodes: std.ArrayList(*parser.Node) = .empty;
    for (ast_nodes) |node| {
        if (node.data == .list and node.data.list.len >= 2 and node.data.list[0].data == .atom and std.mem.eql(u8, node.data.list[0].data.atom.symbol, "import")) {
            const imp_target = node.data.list[1];
            const imp_str = if (imp_target.data == .atom and imp_target.data.atom == .string) imp_target.data.atom.string else if (imp_target.data == .atom and imp_target.data.atom == .symbol) imp_target.data.atom.symbol else "";
            if (std.mem.startsWith(u8, imp_str, "#0x") or std.mem.startsWith(u8, imp_str, "0x")) {
                const hash_hex = if (std.mem.startsWith(u8, imp_str, "#0x")) imp_str[3..] else imp_str[2..];
                var cache_path_buf: [256]u8 = undefined;
                const cache_path = try std.fmt.bufPrint(&cache_path_buf, ".aepkg/cache/{s}.ae", .{hash_hex});
                var file_b: [1024 * 1024]u8 = undefined;
                if (std.Io.Dir.cwd().readFile(io, cache_path, &file_b)) |imp_src| {
                    var imp_p = try parser.Parser.init(imp_src, arena_allocator);
                    const imp_nodes = try imp_p.parseAll();
                    try resolved_nodes.appendSlice(arena_allocator, imp_nodes);
                    continue;
                } else |_| {}
            }
        }
        try resolved_nodes.append(arena_allocator, node);
    }

    // Compile-time macro expansion
    var expander = macros.MacroExpander.init(arena_allocator);
    defer expander.deinit();
    const expanded_nodes = expander.expandProgram(resolved_nodes.items) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Macro expansion error: {s}\n", .{@errorName(err)}) catch "Macro expansion error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    // Semantic & Type checking (gradual typing, dependent shapes, linear tracking)
    var tc = types.TypeChecker.init(arena_allocator);
    defer tc.deinit();
    tc.checkProgram(expanded_nodes) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Type error: {s}\n", .{@errorName(err)}) catch "Type error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    if (is_check) {
        try printStdout(io, "Check passed: 0 errors found.\n");
        return;
    }

    // AIR SSA Graph & AST Optimization pass (operator fusion, constant folding)
    var air_opt = air.AIROptimizer.init(arena_allocator);
    const optimized_nodes = if (opt_level == 0) expanded_nodes else air_opt.optimizeAst(expanded_nodes) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "AIR optimization error: {s}\n", .{@errorName(err)}) catch "Optimization error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    // If emit-air requested
    if (emit_air) {
        const air_funcs = air.buildProgramAIR(optimized_nodes, arena_allocator) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "AIR generation error: {s}\n", .{@errorName(err)}) catch "AIR error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };
        for (air_funcs.items) |*f| {
            if (opt_level >= 2) air_opt.passOperatorFusion(f);
            if (opt_level >= 1) air_opt.passConstantFolding(f);
            if (opt_level >= 3) try air_opt.passDeadCodeElimination(f);
        }
        const out_air_path = output_path orelse blk: {
            const base = std.fs.path.basename(input_path.?);
            const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
            break :blk try std.fmt.allocPrint(arena_allocator, "{s}.air", .{stem});
        };
        var air_out: std.ArrayList(u8) = .empty;
        defer air_out.deinit(arena_allocator);
        for (air_funcs.items) |f| {
            try air.printAIR(&f, arena_allocator, &air_out);
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_air_path, .data = air_out.items });
        try printStdout(io, "Emitted AIR SSA IR to ");
        try printStdout(io, out_air_path);
        try printStdout(io, "\n");
        return;
    }

    // Codegen to Zig
    var cg = codegen.Codegen.init(arena_allocator);
    defer cg.deinit();
    cg.mem_mode = mem_mode;

    if (mem_mode == .autofree) {
        cg.autofree_result = autofree.analyze(arena_allocator, optimized_nodes) catch |err| {
            var err_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_buf, "Autofree analysis error: {s}\n", .{@errorName(err)}) catch "Analysis error\n";
            try printStderr(io, msg);
            std.process.exit(1);
        };
    }

    const zig_code = cg.generate(optimized_nodes) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Codegen error: {s}\n", .{@errorName(err)}) catch "Codegen error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    // If emit-zig requested
    if (emit_zig) {
        const out_zig_path = output_path orelse "out.zig";
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_zig_path, .data = zig_code });
        try printStdout(io, "Emitted Zig source to ");
        try printStdout(io, out_zig_path);
        try printStdout(io, "\n");
        return;
    }

    // If GPU target requested (CUDA, ROCm/HIP, Vulkan SPIR-V, WebGPU WGSL)
    if (target_triple != null) {
        const tgt = target_triple.?;
        const is_cuda = std.mem.eql(u8, tgt, "cuda") or std.mem.eql(u8, tgt, "nvptx") or std.mem.eql(u8, tgt, "nvidia");
        const is_hip = std.mem.eql(u8, tgt, "rocm") or std.mem.eql(u8, tgt, "hip") or std.mem.eql(u8, tgt, "amd");
        const is_spirv = std.mem.eql(u8, tgt, "spirv") or std.mem.eql(u8, tgt, "vulkan");
        const is_wgsl = std.mem.eql(u8, tgt, "gpu") or std.mem.eql(u8, tgt, "wgsl") or std.mem.eql(u8, tgt, "webgpu");

        if (is_cuda or is_hip or is_spirv or is_wgsl) {
            var gpu_be = gpu.GPUBackend.init(arena_allocator);
            for (expanded_nodes) |node| {
                if (node.data == .list and node.data.list.len >= 2 and node.data.list[0].data == .atom and std.mem.eql(u8, node.data.list[0].data.atom.symbol, "kernel")) {
                    if (is_cuda) {
                        const code = try gpu_be.lowerToCUDA(node);
                        const out_p = output_path orelse "kernel.cu";
                        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_p, .data = code });
                        try printStdout(io, "Generated native NVIDIA CUDA compute kernel to ");
                        try printStdout(io, out_p);
                        try printStdout(io, "\n");
                        return;
                    } else if (is_hip) {
                        const code = try gpu_be.lowerToHIP(node);
                        const out_p = output_path orelse "kernel.hip";
                        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_p, .data = code });
                        try printStdout(io, "Generated native AMD ROCm/HIP compute kernel to ");
                        try printStdout(io, out_p);
                        try printStdout(io, "\n");
                        return;
                    } else if (is_spirv) {
                        const code = try gpu_be.lowerToSPIRV(node);
                        const out_p = output_path orelse "kernel.spv";
                        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_p, .data = code });
                        try printStdout(io, "Generated Vulkan SPIR-V compute shader to ");
                        try printStdout(io, out_p);
                        try printStdout(io, "\n");
                        return;
                    } else if (is_wgsl) {
                        const code = try gpu_be.lowerToWGSL(node);
                        const out_p = output_path orelse "kernel.wgsl";
                        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_p, .data = code });
                        try printStdout(io, "Generated GPU WebGPU compute shader to ");
                        try printStdout(io, out_p);
                        try printStdout(io, "\n");
                        return;
                    }
                }
            }
            try printStderr(io, "Error: No (kernel ...) found for GPU target\n");
            std.process.exit(1);
        }
    }

    // Determine output executable name
    const is_wasm = target_triple != null and std.mem.startsWith(u8, target_triple.?, "wasm32");
    const default_exe_name = blk: {
        const base = std.fs.path.basename(input_path.?);
        const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
        if (is_run) {
            break :blk try std.fmt.allocPrint(arena_allocator, ".aether_build/{s}.exe", .{stem});
        }
        if (is_wasm) {
            break :blk try std.fmt.allocPrint(arena_allocator, "{s}.wasm", .{stem});
        }
        break :blk try std.fmt.allocPrint(arena_allocator, "{s}.exe", .{stem});
    };
    const final_exe = output_path orelse default_exe_name;

    // Build directory: .aether_build
    try std.Io.Dir.cwd().createDirPath(io, ".aether_build");

    // Write runtime.zig and generated.zig into .aether_build
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime.zig", .data = RUNTIME_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_mem.zig", .data = RUNTIME_MEM_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_graphics.zig", .data = RUNTIME_GRAPHICS_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_tui.zig", .data = RUNTIME_TUI_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_audio.zig", .data = RUNTIME_AUDIO_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_platform.zig", .data = RUNTIME_PLATFORM_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_gui.zig", .data = RUNTIME_GUI_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/runtime_engine.zig", .data = RUNTIME_ENGINE_SOURCE });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".aether_build/generated.zig", .data = zig_code });

    // Invoke `zig build-exe`
    const abs_out = try std.fs.path.resolve(arena_allocator, &[_][]const u8{final_exe});
    const emit_bin_arg = try std.fmt.allocPrint(arena_allocator, "-femit-bin={s}", .{abs_out});
    
    var cmd_list: std.ArrayList([]const u8) = .empty;
    try cmd_list.append(arena_allocator, "zig");
    try cmd_list.append(arena_allocator, "build-exe");
    try cmd_list.append(arena_allocator, ".aether_build/generated.zig");
    try cmd_list.append(arena_allocator, "-O");
    try cmd_list.append(arena_allocator, "ReleaseFast");
    try cmd_list.append(arena_allocator, emit_bin_arg);
    if (@import("builtin").os.tag == .windows and !is_wasm) {
        try cmd_list.append(arena_allocator, "-luser32");
        try cmd_list.append(arena_allocator, "-lgdi32");
        try cmd_list.append(arena_allocator, "-lwinmm");
        const want_console = console_override orelse (!is_release or is_run);
        if (!want_console) {
            try cmd_list.append(arena_allocator, "--subsystem");
            try cmd_list.append(arena_allocator, "windows");
        }
    }
    if (is_release) {
        try cmd_list.append(arena_allocator, "-fstrip");
    }
    if (is_wasm) {
        try cmd_list.append(arena_allocator, "-target");
        try cmd_list.append(arena_allocator, if (std.mem.eql(u8, target_triple.?, "wasm32")) "wasm32-wasi" else target_triple.?);
    }
    const build_cmd = try cmd_list.toOwnedSlice(arena_allocator);

    // ponytail: compile with fast release mode for runtime efficiency
    const run_res = std.process.run(arena_allocator, io, .{
        .argv = build_cmd,
    }) catch |err| {
        var err_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&err_buf, "Failed to invoke zig build-exe: {s}\n", .{@errorName(err)}) catch "Build error\n";
        try printStderr(io, msg);
        std.process.exit(1);
    };

    if (!run_res.term.success()) {
        try printStderr(io, "Compilation failed:\n");
        try printStderr(io, run_res.stderr);
        std.process.exit(1);
    }

    if (is_run) {
        // Run the executable
        const exec_cmd = &[_][]const u8{abs_out};
        var child = try std.process.spawn(io, .{
            .argv = exec_cmd,
            .stdin = .inherit,
            .stdout = .inherit,
            .stderr = .inherit,
        });
        const term = try child.wait(io);
        std.process.exit(switch (term) {
            .exited => |c| c,
            else => 1,
        });
    } else {
        if (is_wasm) {
            const base = std.fs.path.basename(input_path.?);
            const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
            const html_name = try std.fmt.allocPrint(arena_allocator, "{s}.html", .{stem});
            const html_pre =
                \\<!DOCTYPE html>
                \\<html>
                \\<head>
                \\<meta charset="utf-8">
                \\<title>Aether Application</title>
                \\<style>
                \\  body { margin: 0; background: #0c0d12; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden; color: #fff; font-family: monospace; }
                \\  canvas { image-rendering: pixelated; border: 2px solid #334; box-shadow: 0 0 20px rgba(0,0,0,0.8); }
                \\</style>
                \\</head>
                \\<body>
                \\<canvas id="aether-canvas" width="640" height="480"></canvas>
                \\<script>
                \\  async function runAether() {
                \\    const canvas = document.getElementById('aether-canvas');
                \\    const ctx = canvas.getContext('2d');
                \\    const wasmResp = await fetch('
            ;
            const html_post =
                \\.wasm');
                \\    const wasmBytes = await wasmResp.arrayBuffer();
                \\    const imports = {
                \\      env: {
                \\        now: () => performance.now(),
                \\        render: (ptr, w, h) => {
                \\          const img = ctx.createImageData(w, h);
                \\          img.data.set(new Uint8ClampedArray(instance.exports.memory.buffer, ptr, w * h * 4));
                \\          ctx.putImageData(img, 0, 0);
                \\        }
                \\      }
                \\    };
                \\    const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
                \\    function frame(ts) {
                \\      if (instance.exports.tick) instance.exports.tick(ts);
                \\      requestAnimationFrame(frame);
                \\    }
                \\    requestAnimationFrame(frame);
                \\  }
                \\  runAether().catch(console.error);
                \\</script>
                \\</body>
                \\</html>
            ;
            const html_content = try std.mem.concat(arena_allocator, u8, &[_][]const u8{ html_pre, stem, html_post });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = html_name, .data = html_content });
            try printStdout(io, "Generated WebAssembly HTML harness: ");
            try printStdout(io, html_name);
            try printStdout(io, "\n");
        }
        try printStdout(io, "Built executable: ");
        try printStdout(io, final_exe);
        try printStdout(io, "\n");
    }
}

fn printStdout(io: std.Io, msg: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var fw: std.Io.File.Writer = .init(.stdout(), io, &buf);
    try fw.interface.writeAll(msg);
    try fw.flush();
}

fn printStderr(io: std.Io, msg: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var fw: std.Io.File.Writer = .init(.stderr(), io, &buf);
    try fw.interface.writeAll(msg);
    try fw.flush();
}
