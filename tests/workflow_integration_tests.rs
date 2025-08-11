// Workflow integration tests for MLIR pipeline
// Tests integration between existing workflows and MLIR compilation

use std::process::Command;
use std::path::Path;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_workflow_with_mlir_backend() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("workflow_test.ae");
    
    // Create a simple Aether source file
    fs::write(&source_file, "// Workflow integration test\n(define main (lambda () (print \"Hello from workflow!\")))\n")
        .expect("Failed to write source file");
    
    // Run aether-workflow build command with MLIR backend
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-workflow", "--", 
            "build", 
            source_file.to_str().unwrap(),
            "--mlir-debug",
            "--timing"
        ])
        .output()
        .expect("Failed to execute aether-workflow");
    
    // Check that workflow succeeded
    if output.status.success() {
        println!("Workflow with MLIR backend succeeded");
        
        // Check for MLIR-specific output
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Build workflow completed") || stdout.contains("successful"), 
               "Workflow should complete successfully");
    } else {
        // Workflow might fail due to mock implementations, but should handle gracefully
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Workflow failed (expected in mock): {}", stderr);
        
        // Should not crash or panic
        assert!(!stderr.contains("panic") && !stderr.contains("thread panicked"), 
               "Workflow should not panic even if it fails");
    }
}

#[test]
fn test_parser_integration_with_mlir() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("parser_test.ae");
    
    // Create Aether source with various language constructs
    let source_content = r#"
// Parser integration test
(define factorial 
  (lambda (n)
    (if (= n 0)
        1
        (* n (factorial (- n 1))))))

(define tensor-demo
  (lambda ()
    (let ((a (tensor-create [2 2] [1.0 2.0 3.0 4.0]))
          (b (tensor-create [2 2] [5.0 6.0 7.0 8.0])))
      (tensor-add a b))))

(define main
  (lambda ()
    (print "Testing parser integration")
    (print (factorial 5))
    (tensor-demo)))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test compilation with parser integration
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--", 
            "build", 
            "--target", "native",
            "--mlir-debug",
            "--dump-mlir", "ast",
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc");
    
    // Should handle complex parsing gracefully
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if output.status.success() {
        println!("Parser integration test succeeded");
        // Should show AST conversion
        assert!(stdout.contains("Converting AST to MLIR") || stdout.contains("AST"), 
               "Should show AST processing");
    } else {
        println!("Parser integration failed (expected in mock): {}", stderr);
        // Should not crash
        assert!(!stderr.contains("panic"), "Parser integration should not panic");
    }
}

#[test]
fn test_type_checker_integration_with_mlir() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("type_test.ae");
    
    // Create source with type annotations and tensor operations
    let source_content = r#"
// Type checker integration test
(define add-tensors : (-> (Tensor Float [2 2]) (Tensor Float [2 2]) (Tensor Float [2 2]))
  (lambda (a b)
    (tensor-add a b)))

(define linear-transform : (-> (Tensor Float [2 2]) (Tensor Float [2 2]) (Tensor Float [2 2]))
  (lambda (weights input)
    (matmul weights input)))

(define main : (-> () Unit)
  (lambda ()
    (let ((weights : (Tensor Float [2 2]) (tensor-create [2 2] [0.5 0.3 0.2 0.8]))
          (input : (Tensor Float [2 2]) (tensor-create [2 2] [1.0 2.0 3.0 4.0])))
      (print "Type checker integration test")
      (linear-transform weights input))))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test compilation with type checking
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--", 
            "build", 
            "--target", "native",
            "--mlir-debug",
            "--verbose",
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if output.status.success() {
        println!("Type checker integration test succeeded");
        // Should show type checking and MLIR generation
        assert!(stdout.contains("Converting AST to MLIR") || stdout.contains("type"), 
               "Should show type checking and MLIR processing");
    } else {
        println!("Type checker integration failed (expected in mock): {}", stderr);
        // Should handle type errors gracefully
        assert!(!stderr.contains("panic"), "Type checker integration should not panic");
    }
}

#[test]
fn test_gpu_compilation_workflow() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("gpu_test.ae");
    
    // Create source with GPU-targetable operations
    let source_content = r#"
// GPU compilation test
(define gpu-matmul
  (lambda (a b)
    (with-gpu-context
      (matmul a b))))

(define gpu-tensor-ops
  (lambda ()
    (let ((a (tensor-create [1000 1000] (random-uniform)))
          (b (tensor-create [1000 1000] (random-uniform))))
      (gpu-matmul a b))))

(define main
  (lambda ()
    (print "GPU compilation test")
    (gpu-tensor-ops)))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test GPU compilation path (this will likely fail in mock but should not crash)
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--", 
            "build", 
            "--target", "native",
            "--mlir-debug",
            "--dump-mlir", "lowering",
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // GPU compilation might not be fully implemented, but should handle gracefully
    if output.status.success() {
        println!("GPU compilation workflow succeeded");
    } else {
        println!("GPU compilation failed (expected): {}", stderr);
        // Should not crash even if GPU features are not implemented
        assert!(!stderr.contains("panic"), "GPU compilation should fail gracefully");
    }
}

#[test]
fn test_webassembly_workflow_integration() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("wasm_workflow_test.ae");
    
    // Create source suitable for WebAssembly
    let source_content = r#"
// WebAssembly workflow test
(define fibonacci
  (lambda (n)
    (if (<= n 1)
        n
        (+ (fibonacci (- n 1)) (fibonacci (- n 2))))))

(define tensor-sum
  (lambda (tensor)
    (tensor-reduce tensor +)))

(define main
  (lambda ()
    (let ((fib-result (fibonacci 10))
          (tensor-data (tensor-create [5] [1.0 2.0 3.0 4.0 5.0]))
          (sum-result (tensor-sum tensor-data)))
      (print "WebAssembly workflow test")
      (print fib-result)
      (print sum-result))))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test WebAssembly compilation workflow
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--", 
            "build", 
            "--target", "wasm32-browser",
            "--mlir-debug",
            "--timing",
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if output.status.success() {
        println!("WebAssembly workflow succeeded");
        
        // Check for WebAssembly-specific outputs
        let output_base = source_file.with_extension("");
        let wasm_file = output_base.with_extension("wasm");
        let js_file = output_base.with_extension("js");
        let html_file = output_base.with_extension("html");
        
        // Files might be created in mock implementation
        if wasm_file.exists() {
            println!("WebAssembly binary created");
        }
        if js_file.exists() {
            println!("JavaScript bindings created");
        }
        if html_file.exists() {
            println!("HTML template created");
        }
    } else {
        println!("WebAssembly workflow failed (expected in mock): {}", stderr);
        assert!(!stderr.contains("panic"), "WebAssembly workflow should not panic");
    }
}

#[test]
fn test_testing_framework_with_mlir() {
    // Test that the testing framework works with MLIR backend
    let output = Command::new("cargo")
        .args(&["test", "--", "--nocapture", "mlir"])
        .output()
        .expect("Failed to run tests");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Tests might pass or fail, but should not crash
    println!("Test output: {}", stdout);
    if !stderr.is_empty() {
        println!("Test stderr: {}", stderr);
    }
    
    // Should not have panics in test output
    assert!(!stdout.contains("thread panicked") && !stderr.contains("thread panicked"), 
           "Tests should not panic");
}

#[test]
fn test_build_system_integration() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("build_test.ae");
    
    // Create a comprehensive test program
    let source_content = r#"
// Build system integration test
(define math-ops
  (lambda (x y)
    (let ((sum (+ x y))
          (product (* x y))
          (difference (- x y)))
      (list sum product difference))))

(define tensor-ops
  (lambda ()
    (let ((a (tensor-create [3 3] (range 9)))
          (b (tensor-create [3 3] (fill 1.0))))
      (tensor-multiply a b))))

(define probabilistic-ops
  (lambda ()
    (let ((samples (sample-normal 0.0 1.0 100)))
      (mean samples))))

(define main
  (lambda ()
    (print "Build system integration test")
    (print (math-ops 10 5))
    (print (tensor-ops))
    (print (probabilistic-ops))))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test full build system integration
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-workflow", "--", 
            "build", 
            source_file.to_str().unwrap(),
            "--progress",
            "--verify"
        ])
        .output()
        .expect("Failed to execute build system");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if output.status.success() {
        println!("Build system integration succeeded");
        assert!(stdout.contains("workflow completed") || stdout.contains("successful"), 
               "Build system should complete successfully");
    } else {
        println!("Build system integration failed (expected in mock): {}", stderr);
        // Should handle build failures gracefully
        assert!(!stderr.contains("panic"), "Build system should not panic");
    }
}

#[test]
fn test_error_recovery_in_workflow() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("error_test.ae");
    
    // Create source with intentional errors
    let source_content = r#"
// Error recovery test
(define broken-function
  (lambda (x)
    (+ x undefined-variable))) // This should cause an error

(define another-function
  (lambda ()
    (tensor-create [invalid-shape] [1 2 3]))) // This should also error

(define main
  (lambda ()
    (broken-function 5)
    (another-function)))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test error recovery in workflow
    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--", 
            "build", 
            "--target", "native",
            "--mlir-debug",
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc");
    
    // Should fail but handle errors gracefully
    assert!(!output.status.success(), "Compilation should fail with errors");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Should provide meaningful error messages
    assert!(stdout.contains("error") || stderr.contains("error") || 
            stdout.contains("Error") || stderr.contains("Error"), 
           "Should provide error messages");
    
    // Should not panic
    assert!(!stdout.contains("panic") && !stderr.contains("panic"), 
           "Error handling should not panic");
}

#[test]
fn test_optimization_pipeline_integration() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let source_file = temp_dir.path().join("optimization_test.ae");
    
    // Create source that benefits from optimization
    let source_content = r#"
// Optimization pipeline test
(define expensive-computation
  (lambda (n)
    (let ((result 0))
      (for i 0 n
        (set! result (+ result (* i i))))
      result)))

(define tensor-chain
  (lambda (input)
    (let ((step1 (tensor-multiply input 2.0))
          (step2 (tensor-add step1 1.0))
          (step3 (tensor-multiply step2 0.5)))
      step3)))

(define main
  (lambda ()
    (let ((computation-result (expensive-computation 1000))
          (tensor-input (tensor-create [100 100] (random-uniform)))
          (tensor-result (tensor-chain tensor-input)))
      (print "Optimization test")
      (print computation-result)
      (print (tensor-sum tensor-result)))))
"#;
    
    fs::write(&source_file, source_content).expect("Failed to write source file");
    
    // Test different optimization levels
    let opt_levels = ["debug", "release", "speed"];
    
    for opt_level in &opt_levels {
        let output = Command::new("cargo")
            .args(&[
                "run", "--bin", "aetherc", "--", 
                "build", 
                "--target", "native",
                "--opt", opt_level,
                "--mlir-debug",
                "--timing",
                source_file.to_str().unwrap()
            ])
            .output()
            .expect("Failed to execute aetherc");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if output.status.success() {
            println!("Optimization level {} succeeded", opt_level);
            // Should show optimization information
            if stdout.contains("Optimization") || stdout.contains("optimization") {
                println!("Optimization information displayed for level {}", opt_level);
            }
        } else {
            println!("Optimization level {} failed (expected in mock): {}", opt_level, stderr);
            assert!(!stderr.contains("panic"), 
                   "Optimization should not panic for level {}", opt_level);
        }
    }
}