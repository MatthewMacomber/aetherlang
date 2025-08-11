// Tests for Aether compiler interface and source compilation system
// Validates syntax validation, AST generation, Windows executable generation, and diagnostics

use aether_language::build_system::{
    BuildSystemManager, AetherCompiler, AetherCompilationConfig, CompilationTarget,
    aether_compiler_helpers
};
use aether_language::compiler::{AST, ASTNode, AtomValue};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_aether_compiler_creation() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    
    // Create a mock compiler binary
    fs::write(&compiler_path, "mock aether compiler").unwrap();
    
    let compiler = AetherCompiler::new(compiler_path.clone());
    assert!(compiler.is_ok());
    
    let compiler = compiler.unwrap();
    assert_eq!(compiler.binary_path(), compiler_path.as_path());
}

#[test]
fn test_aether_compiler_creation_missing_binary() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("nonexistent.exe");
    
    let compiler = AetherCompiler::new(compiler_path);
    assert!(compiler.is_err());
}

#[test]
fn test_syntax_validation_valid_hello_world() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("hello.ae");
    fs::write(&source_path, "(func main () (call print \"Hello, World!\") (return 0))").unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
    assert!(validation.errors.is_empty());
    
    // Check AST structure
    let ast = validation.ast.unwrap();
    match &ast.root {
        ASTNode::List(children) => {
            assert_eq!(children.len(), 4); // func, main, (), body
            
            // Check first element is 'func'
            if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&children[0]) {
                assert_eq!(name, "func");
            } else {
                panic!("Expected 'func' symbol");
            }
        }
        _ => panic!("Expected list node for function definition"),
    }
}

#[test]
fn test_syntax_validation_invalid_syntax() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("invalid.ae");
    fs::write(&source_path, "(func main () (call print \"unterminated string").unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(!validation.is_valid);
    assert!(validation.ast.is_none());
    assert!(!validation.errors.is_empty());
}

#[test]
fn test_syntax_validation_empty_file() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("empty.ae");
    fs::write(&source_path, "").unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(!validation.is_valid);
    assert!(!validation.errors.is_empty());
}

#[test]
fn test_syntax_validation_multiple_expressions() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("multi.ae");
    fs::write(&source_path, r#"
        (func add (x y) (+ x y))
        (func main () 
          (let result (add 2 3))
          (call print result)
          (return 0))
    "#).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
    
    // Check that we have a program with multiple expressions
    let ast = validation.ast.unwrap();
    match &ast.root {
        ASTNode::List(children) => {
            assert_eq!(children.len(), 2); // Two function definitions
        }
        _ => panic!("Expected list node for program"),
    }
}

#[test]
fn test_compilation_config_validation() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let compiler = AetherCompiler::new(compiler_path).unwrap();
    
    // Valid config
    let valid_config = AetherCompilationConfig::default();
    assert!(compiler.validate_config(&valid_config).is_ok());
    
    // Invalid optimization level
    let mut invalid_config = AetherCompilationConfig::default();
    invalid_config.optimization_level = 5;
    assert!(compiler.validate_config(&invalid_config).is_err());
    
    // Empty compiler flag
    let mut invalid_config = AetherCompilationConfig::default();
    invalid_config.compiler_flags.push("".to_string());
    assert!(compiler.validate_config(&invalid_config).is_err());
}

#[test]
fn test_compilation_target_variants() {
    let targets = vec![
        CompilationTarget::WindowsNative,
        CompilationTarget::WebAssembly,
        CompilationTarget::GPU,
        CompilationTarget::Mobile,
    ];
    
    // Test that all targets are supported
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let compiler = AetherCompiler::new(compiler_path).unwrap();
    let supported = compiler.supported_targets();
    
    for target in targets {
        assert!(supported.contains(&target));
    }
}

#[test]
fn test_helper_functions() {
    use aether_compiler_helpers::*;
    
    let output_dir = Path::new("output").to_path_buf();
    
    // Test config creation helpers
    let windows_config = windows_native_config(output_dir.clone());
    assert_eq!(windows_config.target, CompilationTarget::WindowsNative);
    assert_eq!(windows_config.optimization_level, 2);
    
    let debug_config = debug_config(output_dir.clone());
    assert_eq!(debug_config.optimization_level, 0);
    assert!(debug_config.debug_info);
    assert!(debug_config.verbose);
    
    let release_config = release_config(output_dir.clone());
    assert_eq!(release_config.optimization_level, 3);
    assert!(!release_config.debug_info);
    assert!(!release_config.verbose);
    
    // Test executable name determination
    let source_path = Path::new("test.ae");
    
    let exe_name = determine_executable_name(source_path, &CompilationTarget::WindowsNative);
    assert_eq!(exe_name, Path::new("test.exe"));
    
    let wasm_name = determine_executable_name(source_path, &CompilationTarget::WebAssembly);
    assert_eq!(wasm_name, Path::new("test.wasm"));
    
    let gpu_name = determine_executable_name(source_path, &CompilationTarget::GPU);
    assert_eq!(gpu_name, Path::new("test.ptx"));
    
    let mobile_name = determine_executable_name(source_path, &CompilationTarget::Mobile);
    assert_eq!(mobile_name, Path::new("test.aether"));
}

#[test]
fn test_build_system_manager_aether_integration() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create mock Rust compiler binary (needed for compile_aether_compiler)
    let rust_target_dir = temp_dir.path().join("target/debug");
    fs::create_dir_all(&rust_target_dir).unwrap();
    let rust_binary = rust_target_dir.join("aetherc.exe");
    fs::write(&rust_binary, "mock rust compiled aether compiler").unwrap();
    
    // Create Aether source file
    let source_path = temp_dir.path().join("test.ae");
    fs::write(&source_path, "(func main () (call print \"Hello from Aether!\") (return 0))").unwrap();
    
    let mut manager = BuildSystemManager::new();
    
    // Test syntax validation through manager
    // Note: This will fail in practice because we don't have a real compiler,
    // but we can test the interface
    let validation_result = manager.validate_aether_syntax(&source_path);
    // We expect this to fail because we don't have a real compiler binary
    assert!(validation_result.is_err());
}

#[test]
fn test_ast_generation_from_complex_program() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("complex.ae");
    fs::write(&source_path, r#"
        (func factorial (n)
          (if (<= n 1)
              1
              (* n (factorial (- n 1)))))
        
        (func main ()
          (let result (factorial 5))
          (call print "Factorial of 5 is:" result)
          (return 0))
    "#).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
    
    let ast = validation.ast.unwrap();
    
    // Verify AST structure for complex program
    match &ast.root {
        ASTNode::List(children) => {
            assert_eq!(children.len(), 2); // factorial and main functions
            
            // Check first function is factorial
            if let Some(ASTNode::List(func_children)) = ast.resolve_ref(&children[0]) {
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&func_children[0]) {
                    assert_eq!(name, "func");
                }
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&func_children[1]) {
                    assert_eq!(name, "factorial");
                }
            }
        }
        _ => panic!("Expected program list"),
    }
}

#[test]
fn test_diagnostic_generation() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("error.ae");
    fs::write(&source_path, "(func main () (call print \"unterminated").unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(!validation.is_valid);
    assert!(!validation.errors.is_empty());
    
    // Test diagnostic report generation
    let error = aether_language::build_system::AetherCompilerError::SyntaxError(validation.errors);
    let report = compiler.generate_diagnostics(&error);
    
    assert!(report.error_count > 0);
    assert!(!report.formatted_output.is_empty());
    assert!(!report.diagnostics.is_empty());
}

#[test]
fn test_windows_executable_generation_interface() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("hello.ae");
    fs::write(&source_path, "(func main () (call print \"Hello, Windows!\") (return 0))").unwrap();
    
    let output_path = temp_dir.path().join("hello.exe");
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    
    // This will fail because we don't have a real compiler, but we test the interface
    let result = compiler.compile_to_executable(&source_path, &output_path);
    
    // We expect this to fail with an I/O error because the mock compiler can't actually compile
    assert!(result.is_err());
}

#[test]
fn test_compilation_config_updates() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    
    // Test initial config
    let initial_config = compiler.config();
    assert_eq!(initial_config.target, CompilationTarget::WindowsNative);
    assert_eq!(initial_config.optimization_level, 2);
    
    // Update config
    let mut new_config = AetherCompilationConfig::default();
    new_config.optimization_level = 3;
    new_config.target = CompilationTarget::WebAssembly;
    new_config.verbose = true;
    
    compiler.update_config(new_config.clone());
    
    // Verify config was updated
    let updated_config = compiler.config();
    assert_eq!(updated_config.optimization_level, 3);
    assert_eq!(updated_config.target, CompilationTarget::WebAssembly);
    assert!(updated_config.verbose);
}

#[test]
fn test_ffi_example_syntax_validation() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("ffi_test.ae");
    fs::write(&source_path, r#"
        (import ffi)
        
        (ffi.bind_c_library 
          (name "math_lib")
          (header "math.h")
          (functions 
            ((c_name "sqrt") (aether_name "square_root") 
             (params ((x Float64))) (returns Float64))))
        
        (func main ()
          (let result (square_root 16.0))
          (call print "Square root of 16 is:" result)
          (return 0))
    "#).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
}

#[test]
fn test_game_example_syntax_validation() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("game_test.ae");
    fs::write(&source_path, r#"
        (use stdlib.ecs as ecs)
        (use stdlib.math3d as math)
        
        (component PlayerController
          (speed f32)
          (jump_force f32)
          (grounded bool))
        
        (struct GameState
          (world ecs.World)
          (camera Camera))
        
        (func main ()
          (let game_state (initialize_game))
          (game_loop game_state)
          (return 0))
    "#).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
}

#[test]
fn test_error_recovery_and_reporting() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    let source_path = temp_dir.path().join("errors.ae");
    fs::write(&source_path, r#"
        (func broken_function (
          (let x 42)
          (call undefined_function x)
          (return x)
    "#).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    
    assert!(!validation.is_valid);
    assert!(!validation.errors.is_empty());
    
    // Check that we get meaningful error messages
    let first_error = &validation.errors[0];
    assert!(!first_error.message.is_empty());
    assert_eq!(first_error.severity, aether_language::compiler::DiagnosticSeverity::Error);
}

#[test]
fn test_performance_with_large_program() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();
    
    // Generate a large program
    let mut large_program = String::new();
    for i in 0..100 {
        large_program.push_str(&format!(
            "(func func_{} (x) (+ x {}))\n",
            i, i
        ));
    }
    large_program.push_str("(func main () (return 0))");
    
    let source_path = temp_dir.path().join("large.ae");
    fs::write(&source_path, large_program).unwrap();
    
    let mut compiler = AetherCompiler::new(compiler_path).unwrap();
    
    let start = std::time::Instant::now();
    let validation = compiler.validate_syntax(&source_path).unwrap();
    let duration = start.elapsed();
    
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
    
    // Ensure parsing doesn't take too long (should be under 1 second for this size)
    assert!(duration.as_secs() < 1);
}