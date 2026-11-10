const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main compiler binary: aetherc
    const exe = b.addExecutable(.{
        .name = "aetherc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Exported Zig module for embedding: @import("aether")
    _ = b.addModule("aether", .{
        .root_source_file = b.path("src/embed.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static embeddable library: aether.lib / libaether.a
    const lib_static = b.addLibrary(.{
        .linkage = .static,
        .name = "aether",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/embed.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(lib_static);

    // Dynamic embeddable shared library: aether.dll / libaether.so
    const lib_shared = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "aether",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/embed.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(lib_shared);

    // Install C / C++ header
    b.installFile("include/aether.h", "include/aether.h");

    // Run step for aetherc
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the Aether compiler (aetherc)");
    run_step.dependOn(&run_cmd.step);

    // Unit test suite
    const test_step = b.step("test", "Run all unit tests");

    const runtime_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_runtime_tests = b.addRunArtifact(runtime_tests);
    test_step.dependOn(&run_runtime_tests.step);

    const runtime_mem_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_mem.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_runtime_mem_tests = b.addRunArtifact(runtime_mem_tests);
    test_step.dependOn(&run_runtime_mem_tests.step);

    const autofree_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/autofree.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_autofree_tests = b.addRunArtifact(autofree_tests);
    test_step.dependOn(&run_autofree_tests.step);

    const parser_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_parser_tests = b.addRunArtifact(parser_tests);
    test_step.dependOn(&run_parser_tests.step);

    const codegen_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_codegen_tests = b.addRunArtifact(codegen_tests);
    test_step.dependOn(&run_codegen_tests.step);

    const types_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/types.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_types_tests = b.addRunArtifact(types_tests);
    test_step.dependOn(&run_types_tests.step);

    const macros_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/macros.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_macros_tests = b.addRunArtifact(macros_tests);
    test_step.dependOn(&run_macros_tests.step);

    const sweet_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/sweet.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_sweet_tests = b.addRunArtifact(sweet_tests);
    test_step.dependOn(&run_sweet_tests.step);

    const token_dag_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/token_dag.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_token_dag_tests = b.addRunArtifact(token_dag_tests);
    test_step.dependOn(&run_token_dag_tests.step);

    const autodiff_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/autodiff.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_autodiff_tests = b.addRunArtifact(autodiff_tests);
    test_step.dependOn(&run_autodiff_tests.step);

    const air_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/air.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_air_tests = b.addRunArtifact(air_tests);
    test_step.dependOn(&run_air_tests.step);

    const ffi_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ffi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ffi_tests = b.addRunArtifact(ffi_tests);
    test_step.dependOn(&run_ffi_tests.step);

    const gpu_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    test_step.dependOn(&run_gpu_tests.step);

    const concurrency_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/concurrency.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_concurrency_tests = b.addRunArtifact(concurrency_tests);
    test_step.dependOn(&run_concurrency_tests.step);

    const replay_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/replay.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_replay_tests = b.addRunArtifact(replay_tests);
    test_step.dependOn(&run_replay_tests.step);

    const stdlib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/stdlib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_stdlib_tests = b.addRunArtifact(stdlib_tests);
    test_step.dependOn(&run_stdlib_tests.step);

    const analyze_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/analyze.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_analyze_tests = b.addRunArtifact(analyze_tests);
    test_step.dependOn(&run_analyze_tests.step);

    const onnx_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/onnx.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_onnx_tests = b.addRunArtifact(onnx_tests);
    test_step.dependOn(&run_onnx_tests.step);

    const repl_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/repl.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_repl_tests = b.addRunArtifact(repl_tests);
    test_step.dependOn(&run_repl_tests.step);

    const lsp_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lsp.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lsp_tests = b.addRunArtifact(lsp_tests);
    test_step.dependOn(&run_lsp_tests.step);

    const graphics_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_graphics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_graphics_tests = b.addRunArtifact(graphics_tests);
    test_step.dependOn(&run_graphics_tests.step);

    const tui_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_tui.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tui_tests = b.addRunArtifact(tui_tests);
    test_step.dependOn(&run_tui_tests.step);

    const audio_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_audio.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_audio_tests = b.addRunArtifact(audio_tests);
    test_step.dependOn(&run_audio_tests.step);

    const gui_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_gui.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_gui_tests = b.addRunArtifact(gui_tests);
    test_step.dependOn(&run_gui_tests.step);

    const platform_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_platform.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_platform_tests = b.addRunArtifact(platform_tests);
    test_step.dependOn(&run_platform_tests.step);

    const engine_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runtime_engine.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_engine_tests = b.addRunArtifact(engine_tests);
    test_step.dependOn(&run_engine_tests.step);

    const embed_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/embed.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const run_embed_tests = b.addRunArtifact(embed_tests);
    test_step.dependOn(&run_embed_tests.step);

    const c_test_exe = b.addExecutable(.{
        .name = "test_embed_c",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    c_test_exe.root_module.addCSourceFile(.{
        .file = b.path("tests/test_embed_c.c"),
        .flags = &.{"-std=c99"},
    });
    c_test_exe.root_module.addIncludePath(b.path("include"));
    c_test_exe.root_module.linkLibrary(lib_static);
    const run_c_test = b.addRunArtifact(c_test_exe);
    test_step.dependOn(&run_c_test.step);
}
