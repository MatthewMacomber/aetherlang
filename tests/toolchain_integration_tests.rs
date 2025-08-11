// Toolchain Integration Tests
// Tests for aetherc, aetherfmt, aether-analyze, and aether-playground

use std::process::Command;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_aetherc_version() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "aetherc", "--", "--version"])
        .output()
        .expect("Failed to execute aetherc");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Aether Compiler v0.1.0"));
}

#[test]
fn test_aetherc_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "aetherc", "--", "--help"])
        .output()
        .expect("Failed to execute aetherc");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: aetherc"));
    assert!(stdout.contains("build"));
    assert!(stdout.contains("demo"));
}

#[test]
fn test_aetherc_demo() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "aetherc", "--", "demo"])
        .output()
        .expect("Failed to execute aetherc demo");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Token System Demo"));
    assert!(stdout.contains("Symbol Table Demo"));
    assert!(stdout.contains("Sweet Syntax Transpiler Demo"));
    assert!(stdout.contains("Native Compilation Demo"));
}

#[test]
fn test_aetherc_native_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");
    let output_file = temp_dir.path().join("test_output");

    // Create a simple test file
    fs::write(&input_file, "let x = 42").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--",
            "build",
            "--target", "native",
            "--output", output_file.to_str().unwrap(),
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc build");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Compiling"));
    assert!(stdout.contains("native executable"));
}

#[test]
fn test_aetherc_wasm_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");
    let output_file = temp_dir.path().join("test_wasm");

    // Create a simple test file
    fs::write(&input_file, "let x = 42").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--",
            "build",
            "--target", "wasm32-browser",
            "--output", output_file.to_str().unwrap(),
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherc wasm build");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Compiling"));
    assert!(stdout.contains("WebAssembly"));
}

#[test]
fn test_aetherfmt_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "aetherfmt", "--", "--help"])
        .output()
        .expect("Failed to execute aetherfmt");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Aether Format Tool"));
    assert!(stdout.contains("--to-sweet"));
    assert!(stdout.contains("--to-canonical"));
}

#[test]
fn test_aetherfmt_sweet_to_canonical() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("input.ae");
    let output_file = temp_dir.path().join("output.ae");

    // Create test input
    fs::write(&input_file, "hello").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherfmt", "--",
            "--to-canonical",
            "--output", output_file.to_str().unwrap(),
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherfmt");

    assert!(output.status.success());
    assert!(output_file.exists());
}

#[test]
fn test_aetherfmt_canonical_to_sweet() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("input.ae");
    let output_file = temp_dir.path().join("output.ae");

    // Create test input (S-expression)
    fs::write(&input_file, "(hello world)").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherfmt", "--",
            "--to-sweet",
            "--output", output_file.to_str().unwrap(),
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherfmt");

    assert!(output.status.success());
    assert!(output_file.exists());
}

#[test]
fn test_aetherfmt_pretty_printing() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("input.ae");

    // Create test input
    fs::write(&input_file, "{x + y * z}").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherfmt", "--",
            "--to-canonical",
            "--pretty",
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aetherfmt with pretty printing");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.is_empty());
}

#[test]
fn test_aether_analyze_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "aether-analyze", "--", "--help"])
        .output()
        .expect("Failed to execute aether-analyze");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Aether Static Analyzer"));
    assert!(stdout.contains("--memory"));
    assert!(stdout.contains("--performance"));
    assert!(stdout.contains("--tensors"));
    assert!(stdout.contains("--gpu"));
}

#[test]
fn test_aether_analyze_memory() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");

    // Create test input
    fs::write(&input_file, "let x = 42\nlet y = {x * 2}").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--memory",
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aether-analyze memory");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Memory Analysis"));
}

#[test]
fn test_aether_analyze_performance() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");

    // Create test input
    fs::write(&input_file, "let x = 42\nlet y = {x * 2}").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--performance",
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aether-analyze performance");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Performance Analysis"));
}

#[test]
fn test_aether_analyze_all() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");

    // Create test input
    fs::write(&input_file, "let x = 42\nlet y = {x * 2}").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--all",
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aether-analyze all");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Memory Analysis"));
    assert!(stdout.contains("Performance Analysis"));
    assert!(stdout.contains("Tensor Analysis"));
    assert!(stdout.contains("GPU Analysis"));
}

#[test]
fn test_aether_analyze_json_output() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");
    let output_file = temp_dir.path().join("analysis.json");

    // Create test input
    fs::write(&input_file, "let x = 42").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--all",
            "--format", "json",
            "--output", output_file.to_str().unwrap(),
            input_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute aether-analyze json");

    assert!(output.status.success());
    assert!(output_file.exists());

    // Verify JSON format
    let json_content = fs::read_to_string(&output_file).unwrap();
    let _: serde_json::Value = serde_json::from_str(&json_content)
        .expect("Output should be valid JSON");
}

#[test]
fn test_aether_analyze_html_output() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("test.ae");

    // Create test input
    fs::write(&input_file, "let x = 42").unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--all",
            "--format", "html",
            input_file.to_str().unwrap()
        ])
        .current_dir(&temp_dir)
        .output()
        .expect("Failed to execute aether-analyze html");

    assert!(output.status.success());
    
    let html_file = temp_dir.path().join("analysis_report.html");
    assert!(html_file.exists());

    // Verify HTML content
    let html_content = fs::read_to_string(&html_file).unwrap();
    assert!(html_content.contains("<!DOCTYPE html>"));
    assert!(html_content.contains("Aether Static Analysis Report"));
}

#[test]
fn test_toolchain_integration_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let source_file = temp_dir.path().join("example.ae");
    let formatted_file = temp_dir.path().join("formatted.ae");
    let analysis_file = temp_dir.path().join("analysis.json");
    let executable_file = temp_dir.path().join("example");

    // Step 1: Create source file
    fs::write(&source_file, r#"
let x = 42
let y = {x * 2 + 1}
fn factorial(n) {
    if {n <= 1} {
        1
    } else {
        {n * factorial({n - 1})}
    }
}
factorial(5)
"#).unwrap();

    // Step 2: Format the code
    let format_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherfmt", "--",
            "--to-canonical",
            "--output", formatted_file.to_str().unwrap(),
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to format code");

    assert!(format_output.status.success());
    assert!(formatted_file.exists());

    // Step 3: Analyze the code
    let analyze_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--all",
            "--format", "json",
            "--output", analysis_file.to_str().unwrap(),
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to analyze code");

    assert!(analyze_output.status.success());
    assert!(analysis_file.exists());

    // Step 4: Compile the code
    let compile_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--",
            "build",
            "--target", "native",
            "--output", executable_file.to_str().unwrap(),
            source_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to compile code");

    assert!(compile_output.status.success());

    // Verify all outputs exist and contain expected content
    let formatted_content = fs::read_to_string(&formatted_file).unwrap();
    assert!(!formatted_content.is_empty());

    let analysis_content = fs::read_to_string(&analysis_file).unwrap();
    let analysis_json: serde_json::Value = serde_json::from_str(&analysis_content).unwrap();
    assert!(analysis_json.is_object());

    println!("✅ Complete toolchain integration test passed!");
}

#[test]
fn test_error_handling_and_diagnostics() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_file = temp_dir.path().join("invalid.ae");

    // Create invalid syntax file
    fs::write(&invalid_file, "invalid syntax here }{").unwrap();

    // Test that tools handle errors gracefully
    let format_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherfmt", "--",
            "--to-canonical",
            invalid_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run aetherfmt on invalid file");

    // Should fail but not crash
    assert!(!format_output.status.success());
    let stderr = String::from_utf8(format_output.stderr).unwrap();
    assert!(stderr.contains("Error") || stderr.contains("error"));

    let analyze_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aether-analyze", "--",
            "--memory",
            invalid_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run aether-analyze on invalid file");

    // Should handle gracefully
    assert!(analyze_output.status.success());

    let compile_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "aetherc", "--",
            "build",
            "--target", "native",
            invalid_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run aetherc on invalid file");

    // Should complete but indicate compilation issues
    assert!(compile_output.status.success());
}

#[test]
fn test_user_experience_validation() {
    // Test that help messages are informative
    let tools = ["aetherc", "aetherfmt", "aether-analyze"];
    
    for tool in &tools {
        let output = Command::new("cargo")
            .args(&["run", "--bin", tool, "--", "--help"])
            .output()
            .expect(&format!("Failed to get help for {}", tool));

        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        
        // Check that help contains essential information
        assert!(stdout.contains("Usage:") || stdout.contains("USAGE:"));
        assert!(stdout.contains("--help") || stdout.contains("-h"));
        assert!(!stdout.is_empty());
        
        println!("✅ {} help message is informative", tool);
    }

    // Test that version information is available
    for tool in &tools {
        let output = Command::new("cargo")
            .args(&["run", "--bin", tool, "--", "--version"])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8(output.stdout).unwrap();
                assert!(stdout.contains("0.1.0") || stdout.contains("version"));
                println!("✅ {} version information available", tool);
            }
        }
    }
}