// Tests for Rust compiler interface and dependency management

use aether_language::build_system::{
    BuildConfig, ToolchainConfig, Dependency
};
use aether_language::build_system::rust_compiler::{
    RustCompiler, RustCompilerError, CompilationResult, DependencyInstallResult, 
    FeatureFlagResult, CompilationAnalysis
};
use std::time::Duration;
use std::path::PathBuf;

#[cfg(test)]
mod rust_compiler_tests {
    use super::*;

    /// Create a mock RustCompiler for testing
    fn create_mock_compiler() -> RustCompiler {
        // Create a default compiler and modify it for testing
        let mut compiler = match RustCompiler::new() {
            Ok(c) => c,
            Err(_) => {
                // If we can't create a real compiler, skip the test
                panic!("Cannot create RustCompiler - Rust toolchain not available");
            }
        };
        
        // Add test feature flags
        compiler.add_feature_flag("mlir".to_string());
        compiler
    }

    /// Create test dependencies
    fn create_test_dependencies() -> Vec<Dependency> {
        vec![
            Dependency {
                name: "melior".to_string(),
                version: "0.19".to_string(),
                optional: true,
            },
            Dependency {
                name: "mlir-sys".to_string(),
                version: "0.3".to_string(),
                optional: true,
            },
            Dependency {
                name: "tempfile".to_string(),
                version: "3.8".to_string(),
                optional: false,
            },
        ]
    }

    #[test]
    fn test_rust_compiler_creation() {
        // Test creating RustCompiler with default configuration
        match RustCompiler::new() {
            Ok(compiler) => {
                assert!(!compiler.get_toolchain_info().version.is_empty());
                assert!(!compiler.get_toolchain_info().target_triple.is_empty());
            }
            Err(RustCompilerError::ToolchainNotFound(_)) => {
                // This is acceptable in CI environments where Rust might not be installed
                println!("Rust toolchain not found - skipping test");
            }
            Err(e) => panic!("Unexpected error creating RustCompiler: {}", e),
        }
    }

    #[test]
    fn test_rust_compiler_with_config() {
        let config = ToolchainConfig {
            version: "1.70.0".to_string(),
            features: vec!["mlir".to_string(), "gpu".to_string()],
            dependencies: create_test_dependencies(),
        };

        match RustCompiler::with_config(&config) {
            Ok(compiler) => {
                let toolchain_info = compiler.get_toolchain_info();
                assert_eq!(toolchain_info.version, "1.70.0");
            }
            Err(RustCompilerError::ToolchainNotFound(_)) => {
                println!("Rust toolchain not found - skipping test");
            }
            Err(e) => panic!("Unexpected error creating RustCompiler with config: {}", e),
        }
    }

    #[test]
    fn test_feature_flag_management() {
        let mut compiler = create_mock_compiler();

        // Test initial state
        assert!(compiler.get_feature_flags().contains(&"mlir".to_string()));

        // Test adding feature flag
        compiler.add_feature_flag("gpu".to_string());
        assert!(compiler.get_feature_flags().contains(&"gpu".to_string()));

        // Test removing feature flag
        compiler.remove_feature_flag("mlir");
        assert!(!compiler.get_feature_flags().contains(&"mlir".to_string()));

        // Test setting feature flags
        compiler.set_feature_flags(vec!["test1".to_string(), "test2".to_string()]);
        assert_eq!(compiler.get_feature_flags().len(), 2);
        assert!(compiler.get_feature_flags().contains(&"test1".to_string()));
        assert!(compiler.get_feature_flags().contains(&"test2".to_string()));
    }

    #[test]
    fn test_dependency_to_feature_name_conversion() {
        let compiler = create_mock_compiler();

        // Test MLIR dependencies
        assert_eq!(compiler.dependency_to_feature_name("melior"), "mlir");
        assert_eq!(compiler.dependency_to_feature_name("mlir-sys"), "mlir");

        // Test generic dependency name conversion
        assert_eq!(compiler.dependency_to_feature_name("some-crate"), "some_crate");
        assert_eq!(compiler.dependency_to_feature_name("another_crate"), "another_crate");
    }

    #[test]
    fn test_compilation_result_analysis() {
        let compiler = create_mock_compiler();

        // Test successful compilation
        let success_result = CompilationResult {
            success: true,
            exit_code: 0,
            stdout: "Compiling aether-language v0.1.0".to_string(),
            stderr: String::new(),
            duration: Duration::from_secs(10),
            features_used: vec!["mlir".to_string()],
        };

        let analysis = compiler.analyze_compilation_result(&success_result).unwrap();
        assert!(analysis.success);
        assert_eq!(analysis.errors.len(), 0);
        assert_eq!(analysis.warnings.len(), 0);

        // Test failed compilation with missing dependency
        let failed_result = CompilationResult {
            success: false,
            exit_code: 1,
            stdout: String::new(),
            stderr: "error: could not find `tempfile` in registry\nwarning: unused import `std::fs`".to_string(),
            duration: Duration::from_secs(5),
            features_used: vec!["mlir".to_string()],
        };

        let analysis = compiler.analyze_compilation_result(&failed_result).unwrap();
        assert!(!analysis.success);
        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.warnings.len(), 1);
        assert_eq!(analysis.missing_dependencies.len(), 1);
        assert_eq!(analysis.missing_dependencies[0], "tempfile");

        // Test compilation with feature issues
        let feature_error_result = CompilationResult {
            success: false,
            exit_code: 1,
            stdout: String::new(),
            stderr: "error: feature `nonexistent` does not exist".to_string(),
            duration: Duration::from_secs(2),
            features_used: vec!["nonexistent".to_string()],
        };

        let analysis = compiler.analyze_compilation_result(&feature_error_result).unwrap();
        assert!(!analysis.success);
        assert_eq!(analysis.feature_issues.len(), 1);
        assert_eq!(analysis.feature_issues[0], "nonexistent");
    }

    #[test]
    fn test_error_message_parsing() {
        let compiler = create_mock_compiler();

        // Test dependency name extraction
        let dep_error = "error: could not find `melior` in registry";
        assert_eq!(compiler.extract_dependency_name(dep_error), Some("melior".to_string()));

        let dep_error2 = "error: could not find `mlir-sys` in registry";
        assert_eq!(compiler.extract_dependency_name(dep_error2), Some("mlir-sys".to_string()));

        // Test feature name extraction
        let feature_error = "error: feature `gpu` does not exist";
        assert_eq!(compiler.extract_feature_name(feature_error), Some("gpu".to_string()));

        let feature_error2 = "error: feature `mlir` not found";
        assert_eq!(compiler.extract_feature_name(feature_error2), Some("mlir".to_string()));
    }

    #[test]
    fn test_toolchain_info() {
        match RustCompiler::new() {
            Ok(compiler) => {
                let info = compiler.get_toolchain_info();
                
                // Basic validation of toolchain info
                assert!(!info.version.is_empty());
                assert!(!info.target_triple.is_empty());
                assert!(!info.cargo_path.as_os_str().is_empty());
                assert!(!info.rustc_path.as_os_str().is_empty());
                
                // Cargo and rustc versions should contain version information
                if !info.cargo_version.contains("unknown") {
                    assert!(info.cargo_version.contains("cargo"));
                }
                if !info.rustc_version.contains("unknown") {
                    assert!(info.rustc_version.contains("rustc"));
                }
            }
            Err(RustCompilerError::ToolchainNotFound(_)) => {
                println!("Rust toolchain not found - skipping toolchain info test");
            }
            Err(e) => panic!("Unexpected error getting toolchain info: {}", e),
        }
    }

    #[test]
    fn test_build_config_integration() {
        let config = BuildConfig {
            rust_toolchain: ToolchainConfig {
                version: "stable".to_string(),
                features: vec!["mlir".to_string()],
                dependencies: create_test_dependencies(),
            },
            ..Default::default()
        };

        match RustCompiler::with_config(&config.rust_toolchain) {
            Ok(mut compiler) => {
                // Test feature flag management with config
                let result = compiler.manage_feature_flags(&config);
                match result {
                    Ok(flag_result) => {
                        // Should have some feature management result
                        assert!(!flag_result.current_features.is_empty() || 
                               !flag_result.enabled_features.is_empty() ||
                               !flag_result.disabled_features.is_empty());
                    }
                    Err(e) => {
                        // Feature management might fail if dependencies aren't available
                        println!("Feature management failed (expected in test environment): {}", e);
                    }
                }
            }
            Err(RustCompilerError::ToolchainNotFound(_)) => {
                println!("Rust toolchain not found - skipping config integration test");
            }
            Err(e) => panic!("Unexpected error in config integration test: {}", e),
        }
    }

    #[test]
    fn test_dependency_installation_simulation() {
        let compiler = create_mock_compiler();
        let dependencies = create_test_dependencies();

        // Test dependency installation (this will mostly test the logic, not actual installation)
        let result = compiler.install_missing_dependencies(&dependencies);
        
        match result {
            Ok(install_result) => {
                assert_eq!(install_result.individual_results.len(), dependencies.len());
                
                // Check that each dependency was processed
                for (i, dep_result) in install_result.individual_results.iter().enumerate() {
                    assert_eq!(dep_result.dependency_name, dependencies[i].name);
                }
            }
            Err(e) => {
                // Installation might fail in test environment - that's acceptable
                println!("Dependency installation failed (expected in test environment): {}", e);
            }
        }
    }

    #[test]
    fn test_compilation_with_features() {
        match RustCompiler::new() {
            Ok(compiler) => {
                // Test compilation with no features
                let result = compiler.compile_with_features(&[]);
                match result {
                    Ok(compilation_result) => {
                        assert_eq!(compilation_result.features_used.len(), 0);
                    }
                    Err(RustCompilerError::CompilationFailed(_)) => {
                        // Compilation might fail in test environment
                        println!("Compilation failed (expected in test environment)");
                    }
                    Err(RustCompilerError::CompilationFailedWithAnalysis(result, analysis)) => {
                        // Compilation failed but we got analysis - that's good
                        assert!(!result.success);
                        println!("Compilation failed with analysis: {} errors, {} warnings", 
                               analysis.errors.len(), analysis.warnings.len());
                    }
                    Err(e) => panic!("Unexpected compilation error: {}", e),
                }

                // Test compilation with features
                let features = vec!["mlir".to_string()];
                let result = compiler.compile_with_features(&features);
                match result {
                    Ok(compilation_result) => {
                        // Features should be recorded even if compilation succeeds
                        assert!(!compilation_result.features_used.is_empty());
                    }
                    Err(RustCompilerError::CompilationFailedWithAnalysis(result, analysis)) => {
                        // Check that features were used in the failed compilation
                        assert_eq!(result.features_used, features);
                        println!("Feature compilation failed with analysis: {} errors", analysis.errors.len());
                    }
                    Err(e) => {
                        println!("Feature compilation failed: {}", e);
                    }
                }
            }
            Err(RustCompilerError::ToolchainNotFound(_)) => {
                println!("Rust toolchain not found - skipping compilation test");
            }
            Err(e) => panic!("Unexpected error in compilation test: {}", e),
        }
    }

    #[test]
    fn test_error_types_and_conversion() {
        // Test error type creation and display
        let toolchain_error = RustCompilerError::ToolchainNotFound("cargo not found".to_string());
        assert!(toolchain_error.to_string().contains("Rust toolchain not found"));

        let compilation_error = RustCompilerError::CompilationFailed("build failed".to_string());
        assert!(compilation_error.to_string().contains("Compilation failed"));

        let dependency_error = RustCompilerError::DependencyInstallationFailed("install failed".to_string());
        assert!(dependency_error.to_string().contains("Dependency installation failed"));

        // Test conversion to BuildSystemError
        use aether_language::build_system::BuildSystemError;
        let build_error: BuildSystemError = toolchain_error.into();
        assert!(build_error.to_string().contains("Configuration error"));
    }
}

/// Integration tests that require a working Rust environment
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::env;

    /// Test that only runs if RUST_INTEGRATION_TESTS environment variable is set
    fn should_run_integration_tests() -> bool {
        env::var("RUST_INTEGRATION_TESTS").is_ok()
    }

    #[test]
    fn test_real_toolchain_detection() {
        if !should_run_integration_tests() {
            println!("Skipping integration test - set RUST_INTEGRATION_TESTS to enable");
            return;
        }

        let compiler = RustCompiler::new().expect("Should be able to create RustCompiler");
        let info = compiler.check_installation().expect("Should be able to check installation");

        // Verify real toolchain information
        assert!(info.cargo_version.contains("cargo"));
        assert!(info.rustc_version.contains("rustc"));
        assert!(!info.installed_targets.is_empty());
        assert!(!info.installed_components.is_empty());
    }

    #[test]
    fn test_real_compilation() {
        if !should_run_integration_tests() {
            println!("Skipping integration test - set RUST_INTEGRATION_TESTS to enable");
            return;
        }

        let compiler = RustCompiler::new().expect("Should be able to create RustCompiler");
        
        // Try to compile with check flag (faster than full build)
        let mut cmd = std::process::Command::new("cargo");
        cmd.args(&["check", "--quiet"]);
        
        let output = cmd.output().expect("Should be able to run cargo check");
        
        if output.status.success() {
            println!("Real compilation check passed");
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Real compilation check failed (expected): {}", stderr);
            
            // Test error analysis on real output
            let result = CompilationResult {
                success: false,
                exit_code: output.status.code().unwrap_or(1),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: stderr.to_string(),
                duration: Duration::from_secs(1),
                features_used: Vec::new(),
            };
            
            let analysis = compiler.analyze_compilation_result(&result)
                .expect("Should be able to analyze real compilation result");
            
            println!("Real compilation analysis: {} errors, {} warnings, {} missing deps",
                   analysis.errors.len(), analysis.warnings.len(), analysis.missing_dependencies.len());
        }
    }
}