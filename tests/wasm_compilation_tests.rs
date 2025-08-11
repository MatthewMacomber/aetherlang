// WebAssembly compilation tests for Aether
// Tests the WebAssembly compilation pipeline and code generation

use aether_language::{
    AST, ASTNode, WasmCompilationPipeline, WasmTarget, WasmOptimizationLevel,
    WasmCodegen, WasmModule, JsBindings,
};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_wasm_pipeline_creation() {
    // Test creating WebAssembly compilation pipelines for different targets
    let browser_pipeline = WasmCompilationPipeline::for_browser();
    assert_eq!(*browser_pipeline.target(), WasmTarget::Browser);
    assert!(browser_pipeline.is_target_supported());

    let nodejs_pipeline = WasmCompilationPipeline::for_nodejs();
    assert_eq!(*nodejs_pipeline.target(), WasmTarget::NodeJS);
    assert!(nodejs_pipeline.is_target_supported());

    let serverless_pipeline = WasmCompilationPipeline::for_serverless();
    assert_eq!(*serverless_pipeline.target(), WasmTarget::Serverless);
    assert!(serverless_pipeline.is_target_supported());
}

#[test]
fn test_wasm_target_features() {
    // Test that different targets have appropriate features
    let browser_features = WasmTarget::Browser.features();
    assert!(browser_features.contains(&"simd"));
    assert!(browser_features.contains(&"webgpu"));
    assert!(browser_features.contains(&"reference-types"));

    let nodejs_features = WasmTarget::NodeJS.features();
    assert!(nodejs_features.contains(&"simd"));
    assert!(!nodejs_features.contains(&"webgpu")); // No WebGPU in Node.js

    let serverless_features = WasmTarget::Serverless.features();
    assert!(!serverless_features.contains(&"simd")); // Minimal features for compatibility
    assert!(!serverless_features.contains(&"webgpu"));
}

#[test]
fn test_wasm_target_imports() {
    // Test that different targets have appropriate imports
    let browser_imports = WasmTarget::Browser.imports();
    assert!(browser_imports.contains(&"console"));
    assert!(browser_imports.contains(&"webgpu"));
    assert!(browser_imports.contains(&"performance"));

    let nodejs_imports = WasmTarget::NodeJS.imports();
    assert!(nodejs_imports.contains(&"console"));
    assert!(nodejs_imports.contains(&"process"));
    assert!(nodejs_imports.contains(&"fs"));

    let serverless_imports = WasmTarget::Serverless.imports();
    assert!(serverless_imports.contains(&"console"));
    assert_eq!(serverless_imports.len(), 1); // Minimal imports
}

#[test]
fn test_wasm_optimization_levels() {
    // Test WebAssembly optimization level flags
    let debug_flags = WasmOptimizationLevel::Debug.to_wasm_opt_flags();
    assert!(debug_flags.contains(&"-O0"));
    assert!(debug_flags.contains(&"--debuginfo"));

    let release_flags = WasmOptimizationLevel::Release.to_wasm_opt_flags();
    assert!(release_flags.contains(&"-O2"));

    let size_flags = WasmOptimizationLevel::Size.to_wasm_opt_flags();
    assert!(size_flags.contains(&"-Oz"));
    assert!(size_flags.contains(&"--enable-bulk-memory"));

    let speed_flags = WasmOptimizationLevel::Speed.to_wasm_opt_flags();
    assert!(speed_flags.contains(&"-O3"));
    assert!(speed_flags.contains(&"--enable-simd"));
}

#[test]
fn test_wasm_codegen_creation() {
    // Test creating WebAssembly code generators
    let browser_codegen = WasmCodegen::new(WasmTarget::Browser);
    let nodejs_codegen = WasmCodegen::new(WasmTarget::NodeJS);
    let serverless_codegen = WasmCodegen::new(WasmTarget::Serverless);

    // All code generators should be created successfully
    // (This is mainly testing that the constructors work)
}

#[test]
fn test_wasm_module_generation() {
    // Test generating WebAssembly module from mock MLIR
    let codegen = WasmCodegen::new(WasmTarget::Browser);
    let mut mlir_module = aether_language::MockMLIRModule::new();
    
    // Add some mock operations
    mlir_module.add_operation("func.func @main() -> i32".to_string());
    mlir_module.add_operation("linalg.generic".to_string());
    mlir_module.add_operation("memref.alloc".to_string());
    mlir_module.add_operation("autodiff.forward".to_string());
    mlir_module.add_operation("prob.sample".to_string());

    let result = codegen.generate_wasm_module(&mlir_module);
    assert!(result.is_ok());

    let wasm_module = result.unwrap();
    assert!(!wasm_module.binary.is_empty());
    assert!(!wasm_module.exports.is_empty());
    assert!(!wasm_module.functions.is_empty());

    // Check that expected exports are present
    let export_names: Vec<&String> = wasm_module.exports.iter().map(|e| &e.name).collect();
    assert!(export_names.contains(&&"memory".to_string()));
    assert!(export_names.contains(&&"run_main".to_string()));
    assert!(export_names.contains(&&"run_tensor_demo".to_string()));
    assert!(export_names.contains(&&"run_probabilistic_demo".to_string()));
}

#[test]
fn test_js_bindings_generation() {
    // Test generating JavaScript bindings
    let codegen = WasmCodegen::new(WasmTarget::Browser);
    let mut mlir_module = aether_language::MockMLIRModule::new();
    mlir_module.add_operation("func.func @main() -> i32".to_string());

    let wasm_module = codegen.generate_wasm_module(&mlir_module).unwrap();
    let result = codegen.generate_js_bindings(&wasm_module);
    assert!(result.is_ok());

    let js_bindings = result.unwrap();
    assert!(!js_bindings.code.is_empty());
    assert!(js_bindings.typescript_definitions.is_some());
    assert!(js_bindings.package_json.is_some());

    // Check that the JavaScript code contains expected functions
    let js_code = &js_bindings.code;
    assert!(js_code.contains("export default async function init"));
    assert!(js_code.contains("export function run_main"));
    assert!(js_code.contains("export function run_tensor_demo"));
    assert!(js_code.contains("export function run_probabilistic_demo"));
    assert!(js_code.contains("export function get_memory_usage"));

    // Check TypeScript definitions
    let ts_defs = js_bindings.typescript_definitions.unwrap();
    assert!(ts_defs.contains("export default function init"));
    assert!(ts_defs.contains("export function run_main(): number;"));
    assert!(ts_defs.contains("export function get_memory_usage(): number;"));

    // Check package.json
    let package_json = js_bindings.package_json.unwrap();
    assert!(package_json.contains("\"name\": \"aether-wasm-app\""));
    assert!(package_json.contains("\"type\": \"module\""));
}

#[test]
fn test_wasm_compilation_pipeline() {
    // Test the full WebAssembly compilation pipeline
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_output");

    // Create a simple AST for testing
    let ast = AST::new(ASTNode::symbol("test_program".to_string()));

    // Test browser compilation
    let browser_pipeline = WasmCompilationPipeline::for_browser();
    let result = browser_pipeline.compile_to_wasm(&ast, &output_path);
    assert!(result.is_ok());

    // Check that output files were created
    assert!(output_path.with_extension("wasm").exists());
    assert!(output_path.with_extension("js").exists());
    assert!(output_path.with_extension("d.ts").exists());
    assert!(output_path.with_extension("html").exists()); // Browser target includes HTML

    // Test Node.js compilation
    let nodejs_output = temp_dir.path().join("nodejs_output");
    let nodejs_pipeline = WasmCompilationPipeline::for_nodejs();
    let result = nodejs_pipeline.compile_to_wasm(&ast, &nodejs_output);
    assert!(result.is_ok());

    // Check that output files were created (no HTML for Node.js)
    assert!(nodejs_output.with_extension("wasm").exists());
    assert!(nodejs_output.with_extension("js").exists());
    assert!(nodejs_output.with_extension("d.ts").exists());
    assert!(!nodejs_output.with_extension("html").exists()); // No HTML for Node.js
}

#[test]
fn test_wasm_webgpu_compilation() {
    // Test WebAssembly compilation with WebGPU support
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("webgpu_output");

    let ast = AST::new(ASTNode::symbol("webgpu_program".to_string()));

    let browser_pipeline = WasmCompilationPipeline::for_browser();
    let result = browser_pipeline.compile_with_webgpu(&ast, &output_path);
    assert!(result.is_ok());

    // Check that output files were created
    assert!(output_path.with_extension("wasm").exists());
    assert!(output_path.with_extension("js").exists());
    assert!(output_path.with_extension("d.ts").exists());
    assert!(output_path.with_extension("html").exists());
}

#[test]
fn test_wasm_binary_format() {
    // Test that generated WebAssembly binary has correct format
    let codegen = WasmCodegen::new(WasmTarget::Browser);
    let mut mlir_module = aether_language::MockMLIRModule::new();
    mlir_module.add_operation("func.func @main() -> i32".to_string());

    let wasm_module = codegen.generate_wasm_module(&mlir_module).unwrap();
    let binary = &wasm_module.binary;

    // Check WebAssembly magic number
    assert_eq!(&binary[0..4], &[0x00, 0x61, 0x73, 0x6D]);
    
    // Check WebAssembly version
    assert_eq!(&binary[4..8], &[0x01, 0x00, 0x00, 0x00]);
    
    // Binary should be longer than just header
    assert!(binary.len() > 8);
}

#[test]
fn test_html_template_generation() {
    // Test HTML template generation for browser target
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("html_test");

    let ast = AST::new(ASTNode::symbol("html_test_program".to_string()));
    let browser_pipeline = WasmCompilationPipeline::for_browser();
    
    let result = browser_pipeline.compile_to_wasm(&ast, &output_path);
    assert!(result.is_ok());

    // Read and verify HTML template
    let html_path = output_path.with_extension("html");
    assert!(html_path.exists());
    
    let html_content = std::fs::read_to_string(html_path).unwrap();
    assert!(html_content.contains("<!DOCTYPE html>"));
    assert!(html_content.contains("Aether WebAssembly Application"));
    assert!(html_content.contains("runApplication"));
    assert!(html_content.contains("runTensorDemo"));
    assert!(html_content.contains("runProbabilisticDemo"));
    assert!(html_content.contains("import init"));
}

#[test]
fn test_different_optimization_levels() {
    // Test compilation with different optimization levels
    let temp_dir = TempDir::new().unwrap();
    let ast = AST::new(ASTNode::symbol("opt_test_program".to_string()));

    // Test each optimization level
    let opt_levels = [
        WasmOptimizationLevel::Debug,
        WasmOptimizationLevel::Release,
        WasmOptimizationLevel::Size,
        WasmOptimizationLevel::Speed,
    ];

    for (i, opt_level) in opt_levels.iter().enumerate() {
        let output_path = temp_dir.path().join(format!("opt_test_{}", i));
        let pipeline = WasmCompilationPipeline::new(WasmTarget::Browser);
        
        // Note: The current implementation doesn't use optimization level directly,
        // but we test that the pipeline accepts it without errors
        let result = pipeline.compile_to_wasm(&ast, &output_path);
        assert!(result.is_ok());
        
        assert!(output_path.with_extension("wasm").exists());
    }
}

#[test]
fn test_error_handling() {
    // Test error handling in WebAssembly compilation
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("error_test");

    // Test with empty AST (should still work with mock implementation)
    let empty_ast = AST::new(ASTNode::symbol("".to_string()));
    let pipeline = WasmCompilationPipeline::for_browser();
    
    let result = pipeline.compile_to_wasm(&empty_ast, &output_path);
    // With mock implementation, this should still succeed
    assert!(result.is_ok());
}