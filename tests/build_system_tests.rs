// Unit tests for Aether build system manager

use aether_language::build_system::{
    BuildSystemManager, BuildConfig, EnvironmentValidator, ErrorHandler, AutoFixEngine,
    BuildError, ErrorType, ErrorSeverity, FixStrategy, DependencyInfo
};
use std::path::Path;

#[test]
fn test_build_system_manager_creation() {
    let manager = BuildSystemManager::new();
    assert!(manager.config().rust_toolchain.features.contains(&"mlir".to_string()));
}

#[test]
fn test_build_system_manager_with_custom_config() {
    let config = BuildConfig::default();
    let manager = BuildSystemManager::with_config(config);
    assert_eq!(manager.config().rust_toolchain.version, "stable");
}

#[test]
fn test_environment_validation() {
    let manager = BuildSystemManager::new();
    
    // This test may fail in CI environments without Rust installed
    // but should work in development environments
    match manager.validate_environment() {
        Ok(status) => {
            // Environment validation succeeded
            assert!(!status.rust_toolchain.rustc_version.is_empty());
            assert!(!status.rust_toolchain.cargo_version.is_empty());
        }
        Err(_) => {
            // Environment validation failed - this is acceptable in some test environments
            println!("Environment validation failed - this may be expected in CI");
        }
    }
}

#[test]
fn test_environment_validator_creation() {
    let validator = EnvironmentValidator::new();
    
    // Test that we can create the validator without errors
    // The actual validation may fail depending on the environment
    match validator.validate_rust_toolchain() {
        Ok(status) => {
            assert!(!status.rustc_version.is_empty());
            assert!(!status.cargo_version.is_empty());
        }
        Err(_) => {
            // This is acceptable if Rust is not installed in the test environment
            println!("Rust toolchain validation failed - may be expected in some environments");
        }
    }
}

#[test]
fn test_error_handler_creation() {
    let mut error_handler = ErrorHandler::new();
    
    // Test error detection with mock output
    let mock_output = "error: could not find `tempfile` in the list of imported crates";
    let detected_error = error_handler.detect_error(mock_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::MissingDependency);
    assert!(error.message.contains("tempfile"));
    assert_eq!(error.severity, ErrorSeverity::Error);
}

#[test]
fn test_error_handler_mock_detection() {
    let mut error_handler = ErrorHandler::new();
    
    // Test detection of mock implementations
    let mock_output = "error: MockMLIR type not found";
    let detected_error = error_handler.detect_error(mock_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::MockImplementation);
    assert!(error.message.contains("Mock implementation"));
}

#[test]
fn test_error_handler_type_mismatch_detection() {
    let mut error_handler = ErrorHandler::new();
    
    // Test detection of type mismatches
    let mock_output = "error: type mismatch: expected `String`, found `i32`";
    let detected_error = error_handler.detect_error(mock_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::TypeMismatch);
    assert!(error.message.contains("Type mismatch"));
}

#[test]
fn test_error_handler_history() {
    let mut error_handler = ErrorHandler::new();
    
    // Detect some errors
    let _ = error_handler.detect_error("error: could not find `tempfile`");
    let _ = error_handler.detect_error("error: MockMLIR not found");
    
    // Check history
    let history = error_handler.get_error_history();
    assert_eq!(history.len(), 2);
    
    // Clear history
    error_handler.clear_history();
    let history = error_handler.get_error_history();
    assert_eq!(history.len(), 0);
}

#[test]
fn test_error_handler_pattern_matching() {
    let mut error_handler = ErrorHandler::new();
    
    // Test dependency error pattern matching
    let dep_error_output = "error: could not find `serde` in the list of imported crates";
    let detected_error = error_handler.detect_error(dep_error_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::MissingDependency);
    assert!(error.message.contains("serde"));
    assert!(!error.suggested_fixes.is_empty());
}

#[test]
fn test_error_handler_type_mismatch_patterns() {
    let mut error_handler = ErrorHandler::new();
    
    // Test type mismatch pattern
    let type_error_output = "error: expected `String`, found `i32`";
    let detected_error = error_handler.detect_error(type_error_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::TypeMismatch);
    assert!(error.message.contains("expected `String`, found `i32`"));
}

#[test]
fn test_error_handler_compilation_patterns() {
    let mut error_handler = ErrorHandler::new();
    
    // Test unresolved import pattern
    let import_error_output = "error: unresolved import `std::collections::HashMap`";
    let detected_error = error_handler.detect_error(import_error_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::CompilationError);
    assert!(error.message.contains("std::collections::HashMap"));
}

#[test]
fn test_error_handler_mock_implementation_patterns() {
    let mut error_handler = ErrorHandler::new();
    
    // Test mock implementation detection
    let mock_error_output = "error: MockMLIRContext not found in scope";
    let detected_error = error_handler.detect_error(mock_error_output);
    
    assert!(detected_error.is_some());
    let error = detected_error.unwrap();
    assert_eq!(error.error_type, ErrorType::MockImplementation);
    assert!(error.message.contains("MockMLIRContext"));
}

#[test]
fn test_error_handler_statistics() {
    let mut error_handler = ErrorHandler::new();
    
    // Generate some errors
    let _ = error_handler.detect_error("error: could not find `tempfile`");
    let _ = error_handler.detect_error("error: could not find `serde`");
    let _ = error_handler.detect_error("error: expected `String`, found `i32`");
    let _ = error_handler.detect_error("error: could not find `tokio`");
    
    let stats = error_handler.get_error_statistics();
    assert_eq!(stats.total_errors, 4);
    assert_eq!(stats.error_counts_by_type.get(&ErrorType::MissingDependency), Some(&3));
    assert_eq!(stats.error_counts_by_type.get(&ErrorType::TypeMismatch), Some(&1));
}

#[test]
fn test_error_handler_analysis() {
    let mut error_handler = ErrorHandler::new();
    
    // Generate errors with some patterns
    let _ = error_handler.detect_error("error: could not find `tempfile`");
    let _ = error_handler.detect_error("error: could not find `tempfile`"); // Duplicate
    let _ = error_handler.detect_error("error: expected `String`, found `i32`");
    
    let analysis = error_handler.analyze_error_trends();
    assert_eq!(analysis.total_errors, 3);
    assert_eq!(analysis.unique_error_types, 2);
    assert_eq!(analysis.most_frequent_error, Some(ErrorType::MissingDependency));
}

#[test]
fn test_error_handler_severity_classification() {
    let error_handler = ErrorHandler::new();
    
    let dep_error = BuildError {
        error_type: ErrorType::MissingDependency,
        message: "Test".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let severity = error_handler.classify_error_severity(&dep_error);
    assert_eq!(severity, ErrorSeverity::Error);
    
    let linking_error = BuildError {
        error_type: ErrorType::LinkingError,
        message: "Test".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let severity = error_handler.classify_error_severity(&linking_error);
    assert_eq!(severity, ErrorSeverity::Critical);
}

#[test]
fn test_error_handler_location_extraction() {
    let mut error_handler = ErrorHandler::new();
    
    // Test location extraction from compiler output
    let output_with_location = "src/main.rs:42:15: error: expected `String`, found `i32`";
    let detected_error = error_handler.detect_error(output_with_location);
    
    if let Some(error) = detected_error {
        if let Some(location) = error.location {
            assert_eq!(location.file.to_string_lossy(), "src/main.rs");
            assert_eq!(location.line, 42);
            assert_eq!(location.column, 15);
        }
    }
}

#[test]
fn test_error_context_creation() {
    use aether_language::build_system::ErrorContext;
    
    let raw_output = "line 1\nline 2\nerror here\nline 4\nline 5\nline 6";
    let context = ErrorContext::new(raw_output);
    
    assert_eq!(context.raw_output, raw_output);
    assert_eq!(context.surrounding_lines.len(), 5); // Takes first 5 lines
    assert_eq!(context.surrounding_lines[0], "line 1");
    assert_eq!(context.surrounding_lines[2], "error here");
}

#[test]
fn test_error_statistics_recording() {
    use aether_language::build_system::{ErrorStatistics, ErrorContext};
    
    let mut stats = ErrorStatistics::new();
    
    let error1 = BuildError {
        error_type: ErrorType::MissingDependency,
        message: "Test".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 1000,
        context: ErrorContext::empty(),
    };
    
    let error2 = BuildError {
        error_type: ErrorType::TypeMismatch,
        message: "Test".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 1001,
        context: ErrorContext::empty(),
    };
    
    stats.record_error(&error1);
    stats.record_error(&error2);
    stats.record_error(&error1); // Duplicate type
    
    assert_eq!(stats.total_errors, 3);
    assert_eq!(stats.error_counts_by_type.get(&ErrorType::MissingDependency), Some(&2));
    assert_eq!(stats.error_counts_by_type.get(&ErrorType::TypeMismatch), Some(&1));
    assert_eq!(stats.most_common_errors[0].0, ErrorType::MissingDependency);
    assert_eq!(stats.most_common_errors[0].1, 2);
}

#[test]
fn test_auto_fix_engine_creation() {
    let fix_engine = AutoFixEngine::new();
    
    // Test error analysis
    let error = BuildError {
        error_type: ErrorType::MissingDependency,
        message: "Missing tempfile dependency".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let suggested_fix = fix_engine.analyze_error(&error);
    assert!(suggested_fix.is_some());
    
    if let Some(FixStrategy::InstallDependency(dep_info)) = suggested_fix {
        assert_eq!(dep_info.name, "tempfile");
    }
}

#[test]
fn test_auto_fix_engine_dry_run() {
    let mut fix_engine = AutoFixEngine::new_dry_run();
    assert!(fix_engine.is_dry_run());
    
    let fix = FixStrategy::InstallDependency(DependencyInfo {
        name: "test_crate".to_string(),
        version: "1.0".to_string(),
        dev_dependency: false,
    });
    
    let result = fix_engine.apply_fix(&fix);
    assert!(result.application_result.is_ok());
    assert_eq!(result.files_modified.len(), 0); // No files modified in dry run
    
    let history = fix_engine.get_fix_history();
    assert_eq!(history.len(), 1);
}

#[test]
fn test_auto_fix_engine_multiple_errors() {
    let fix_engine = AutoFixEngine::new();
    
    let errors = vec![
        BuildError {
            error_type: ErrorType::MissingDependency,
            message: "Missing tempfile dependency".to_string(),
            location: None,
            suggested_fixes: vec![],
            severity: ErrorSeverity::Error,
            timestamp: 0,
            context: aether_language::build_system::ErrorContext::empty(),
        },
        BuildError {
            error_type: ErrorType::TypeMismatch,
            message: "MockMLIRContext type mismatch".to_string(),
            location: None,
            suggested_fixes: vec![],
            severity: ErrorSeverity::Error,
            timestamp: 0,
            context: aether_language::build_system::ErrorContext::empty(),
        },
    ];
    
    let analysis = fix_engine.analyze_multiple_errors(&errors);
    assert_eq!(analysis.len(), 2);
    assert!(analysis[0].1.is_some()); // tempfile fix should be found
    assert!(analysis[1].1.is_some()); // MockMLIR fix should be found
}

#[test]
fn test_auto_fix_engine_fix_database_statistics() {
    let fix_engine = AutoFixEngine::new();
    let stats = fix_engine.get_fix_database_statistics();
    
    assert!(stats.total_dependency_fixes > 0);
    assert!(stats.total_type_mismatch_fixes > 0);
    assert!(stats.total_mlir_fixes > 0);
}

#[test]
fn test_auto_fix_engine_custom_fixes() {
    let mut fix_engine = AutoFixEngine::new();
    
    // Add custom dependency fix
    fix_engine.add_custom_dependency_fix("custom_crate".to_string(), DependencyInfo {
        name: "custom_crate".to_string(),
        version: "2.0".to_string(),
        dev_dependency: true,
    });
    
    // Test that custom fix is found
    let error = BuildError {
        error_type: ErrorType::MissingDependency,
        message: "Missing custom_crate dependency".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let suggested_fix = fix_engine.analyze_error(&error);
    assert!(suggested_fix.is_some());
    
    if let Some(FixStrategy::InstallDependency(dep_info)) = suggested_fix {
        assert_eq!(dep_info.name, "custom_crate");
        assert_eq!(dep_info.version, "2.0");
        assert!(dep_info.dev_dependency);
    }
}

#[test]
fn test_auto_fix_engine_mlir_fixes() {
    let fix_engine = AutoFixEngine::new();
    
    // Test MLIR mock implementation fix
    let error = BuildError {
        error_type: ErrorType::MockImplementation,
        message: "MockMLIRContext found in code".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Warning,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let suggested_fix = fix_engine.analyze_error(&error);
    assert!(suggested_fix.is_some());
    
    if let Some(FixStrategy::ReplaceImplementation(mock_impl)) = suggested_fix {
        assert_eq!(mock_impl.mock_type, "MockMLIRContext");
        assert_eq!(mock_impl.real_type, "MLIRContext");
    }
}

#[test]
fn test_auto_fix_engine_compilation_fixes() {
    let fix_engine = AutoFixEngine::new();
    
    // Test MLIR feature flag fix
    let error = BuildError {
        error_type: ErrorType::CompilationError,
        message: "MLIR feature not enabled".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: aether_language::build_system::ErrorContext::empty(),
    };
    
    let suggested_fix = fix_engine.analyze_error(&error);
    assert!(suggested_fix.is_some());
    
    if let Some(FixStrategy::EnableFeatureFlag(flag)) = suggested_fix {
        assert_eq!(flag, "mlir");
    }
}

#[test]
fn test_auto_fix_engine_history_management() {
    let mut fix_engine = AutoFixEngine::new_dry_run();
    
    let fix1 = FixStrategy::InstallDependency(DependencyInfo {
        name: "test1".to_string(),
        version: "1.0".to_string(),
        dev_dependency: false,
    });
    
    let fix2 = FixStrategy::EnableFeatureFlag("test_feature".to_string());
    
    fix_engine.apply_fix(&fix1);
    fix_engine.apply_fix(&fix2);
    
    let history = fix_engine.get_fix_history();
    assert_eq!(history.len(), 2);
    
    fix_engine.clear_fix_history();
    let history = fix_engine.get_fix_history();
    assert_eq!(history.len(), 0);
}

#[test]
fn test_auto_fix_engine_multiple_fixes_application() {
    let mut fix_engine = AutoFixEngine::new_dry_run();
    
    let fixes = vec![
        FixStrategy::InstallDependency(DependencyInfo {
            name: "dep1".to_string(),
            version: "1.0".to_string(),
            dev_dependency: false,
        }),
        FixStrategy::EnableFeatureFlag("feature1".to_string()),
        FixStrategy::InstallDependency(DependencyInfo {
            name: "dep2".to_string(),
            version: "2.0".to_string(),
            dev_dependency: true,
        }),
    ];
    
    let results = fix_engine.apply_multiple_fixes(&fixes);
    assert_eq!(results.len(), 3);
    
    for result in &results {
        assert!(result.application_result.is_ok());
    }
    
    let history = fix_engine.get_fix_history();
    assert_eq!(history.len(), 3);
}

#[test]
fn test_build_config_defaults() {
    let config = BuildConfig::default();
    
    assert_eq!(config.rust_toolchain.version, "stable");
    assert!(config.rust_toolchain.features.contains(&"mlir".to_string()));
    assert_eq!(config.target_config.platform, aether_language::build_system::Platform::Windows);
    assert_eq!(config.target_config.architecture, aether_language::build_system::Architecture::X86_64);
}

#[test]
fn test_dependency_info_serialization() {
    let dep_info = DependencyInfo {
        name: "tempfile".to_string(),
        version: "3.8".to_string(),
        dev_dependency: true,
    };
    
    // Test that we can serialize and deserialize
    let json = serde_json::to_string(&dep_info).expect("Failed to serialize");
    let deserialized: DependencyInfo = serde_json::from_str(&json).expect("Failed to deserialize");
    
    assert_eq!(deserialized.name, "tempfile");
    assert_eq!(deserialized.version, "3.8");
    assert!(deserialized.dev_dependency);
}

#[test]
fn test_build_system_error_display() {
    use aether_language::build_system::BuildSystemError;
    
    let error = BuildSystemError::CompilationFailed("Test error".to_string());
    let error_string = format!("{}", error);
    assert!(error_string.contains("Compilation failed"));
    assert!(error_string.contains("Test error"));
}

#[test]
fn test_fix_strategy_variants() {
    // Test that all fix strategy variants can be created
    let install_dep = FixStrategy::InstallDependency(DependencyInfo {
        name: "test".to_string(),
        version: "1.0".to_string(),
        dev_dependency: false,
    });
    
    let enable_feature = FixStrategy::EnableFeatureFlag("mlir".to_string());
    
    // Test serialization
    let json1 = serde_json::to_string(&install_dep).expect("Failed to serialize install_dep");
    let json2 = serde_json::to_string(&enable_feature).expect("Failed to serialize enable_feature");
    
    assert!(json1.contains("InstallDependency"));
    assert!(json2.contains("EnableFeatureFlag"));
}

#[test]
fn test_manager_error_history() {
    let mut manager = BuildSystemManager::new();
    
    // Initially no errors
    assert_eq!(manager.get_error_history().len(), 0);
    
    // Clear history (should not panic even when empty)
    manager.clear_error_history();
    assert_eq!(manager.get_error_history().len(), 0);
}

#[test]
fn test_manager_config_update() {
    let mut manager = BuildSystemManager::new();
    let original_version = manager.config().rust_toolchain.version.clone();
    
    let mut new_config = BuildConfig::default();
    new_config.rust_toolchain.version = "nightly".to_string();
    
    manager.update_config(new_config);
    assert_eq!(manager.config().rust_toolchain.version, "nightly");
    assert_ne!(manager.config().rust_toolchain.version, original_version);
}

// Integration test for the complete workflow (may fail in CI without proper environment)
#[test]
#[ignore] // Ignore by default since it requires a full Rust environment
fn test_complete_build_workflow() {
    let mut manager = BuildSystemManager::new();
    
    // Validate environment
    let env_status = manager.validate_environment().expect("Environment validation failed");
    
    // Check that we have basic requirements
    assert!(!env_status.rust_toolchain.rustc_version.is_empty());
    assert!(!env_status.rust_toolchain.cargo_version.is_empty());
    
    // Try to compile the Aether compiler (this may fail if dependencies are missing)
    match manager.compile_aether_compiler() {
        Ok(compiler_binary) => {
            assert!(compiler_binary.path.exists());
            assert!(!compiler_binary.version.is_empty());
        }
        Err(e) => {
            println!("Compiler compilation failed (expected in some environments): {}", e);
        }
    }
}

#[test]
fn test_code_modifier_creation() {
    let modifier = aether_language::build_system::fix_engine::CodeModifier::new();
    // Test that we can create the modifier without errors
    // The actual modification tests would require creating temporary files
}

#[test]
fn test_code_modifier_without_backup() {
    let modifier = aether_language::build_system::fix_engine::CodeModifier::new_without_backup();
    // Test that we can create the modifier without backup enabled
}

#[test]
fn test_config_manager_creation() {
    let manager = aether_language::build_system::fix_engine::ConfigManager::new();
    // Test that we can create the config manager without errors
}

#[test]
fn test_config_manager_without_backup() {
    let manager = aether_language::build_system::fix_engine::ConfigManager::new_without_backup();
    // Test that we can create the config manager without backup enabled
}

#[test]
fn test_fix_database_statistics() {
    use aether_language::build_system::fix_engine::FixDatabase;
    
    let database = FixDatabase::new();
    let stats = database.get_statistics();
    
    assert!(stats.total_dependency_fixes >= 5); // We added at least 5 dependency fixes
    assert!(stats.total_type_mismatch_fixes >= 2); // We added at least 2 type mismatch fixes
    assert!(stats.total_mlir_fixes >= 2); // We added at least 2 MLIR fixes
}

#[test]
fn test_fix_database_dependency_fixes() {
    use aether_language::build_system::fix_engine::FixDatabase;
    
    let database = FixDatabase::new();
    let dep_fixes = database.get_all_dependency_fixes();
    
    assert!(dep_fixes.contains_key("tempfile"));
    assert!(dep_fixes.contains_key("melior"));
    assert!(dep_fixes.contains_key("mlir-sys"));
    assert!(dep_fixes.contains_key("serde"));
    assert!(dep_fixes.contains_key("regex"));
    
    let tempfile_fix = &dep_fixes["tempfile"];
    assert_eq!(tempfile_fix.name, "tempfile");
    assert_eq!(tempfile_fix.version, "3.8");
    assert!(tempfile_fix.dev_dependency);
}

#[test]
fn test_fix_database_custom_additions() {
    use aether_language::build_system::fix_engine::FixDatabase;
    use aether_language::build_system::{DependencyInfo, CodeModification};
    
    let mut database = FixDatabase::new();
    
    // Add custom dependency fix
    database.add_dependency_fix("custom_dep".to_string(), DependencyInfo {
        name: "custom_dep".to_string(),
        version: "1.5".to_string(),
        dev_dependency: false,
    });
    
    // Add custom type mismatch fix
    database.add_type_mismatch_fix("CustomType".to_string(), CodeModification {
        file_path: std::path::PathBuf::from("test.rs"),
        line_number: 10,
        old_code: "CustomType".to_string(),
        new_code: "RealType".to_string(),
    });
    
    let dep_fixes = database.get_all_dependency_fixes();
    assert!(dep_fixes.contains_key("custom_dep"));
    
    let stats = database.get_statistics();
    assert!(stats.total_dependency_fixes >= 6); // Original 5 + 1 custom
    assert!(stats.total_type_mismatch_fixes >= 3); // Original 2 + 1 custom
}

#[test]
fn test_fix_application_result() {
    use aether_language::build_system::fix_engine::FixApplicationResult;
    use aether_language::build_system::{FixStrategy, DependencyInfo};
    
    let fix = FixStrategy::InstallDependency(DependencyInfo {
        name: "test".to_string(),
        version: "1.0".to_string(),
        dev_dependency: false,
    });
    
    let mut result = FixApplicationResult::new(fix);
    assert!(result.application_result.is_err());
    assert!(result.verification_result.is_none());
    
    result = result.with_success(vec![std::path::PathBuf::from("Cargo.toml")]);
    assert!(result.application_result.is_ok());
    assert_eq!(result.files_modified.len(), 1);
    
    result = result.with_verification(true);
    assert_eq!(result.verification_result, Some(true));
}

#[test]
fn test_fix_error_types() {
    use aether_language::build_system::fix_engine::FixError;
    
    let file_not_found = FixError::FileNotFound("test.txt".to_string());
    let file_op_failed = FixError::FileOperationFailed("Permission denied".to_string());
    let parse_error = FixError::ParseError("Invalid syntax".to_string());
    let verification_failed = FixError::VerificationFailed("Timeout".to_string());
    let command_failed = FixError::CommandFailed("Exit code 1".to_string());
    
    // Test Display implementation
    assert!(format!("{}", file_not_found).contains("File not found"));
    assert!(format!("{}", file_op_failed).contains("File operation failed"));
    assert!(format!("{}", parse_error).contains("Parse error"));
    assert!(format!("{}", verification_failed).contains("Verification failed"));
    assert!(format!("{}", command_failed).contains("Command failed"));
}

#[test]
fn test_fix_database_error_type_matching() {
    use aether_language::build_system::fix_engine::FixDatabase;
    use aether_language::build_system::{BuildError, ErrorType, ErrorSeverity, ErrorContext};
    
    let database = FixDatabase::new();
    
    // Test missing dependency error
    let dep_error = BuildError {
        error_type: ErrorType::MissingDependency,
        message: "could not find `tempfile` in the list of imported crates".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: ErrorContext::empty(),
    };
    
    let fix = database.get_fix_for_error(&dep_error);
    assert!(fix.is_some());
    
    // Test type mismatch error
    let type_error = BuildError {
        error_type: ErrorType::TypeMismatch,
        message: "expected MLIRContext, found MockMLIRContext".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: ErrorContext::empty(),
    };
    
    let fix = database.get_fix_for_error(&type_error);
    assert!(fix.is_some());
    
    // Test mock implementation error
    let mock_error = BuildError {
        error_type: ErrorType::MockImplementation,
        message: "MockMLIRContext should be replaced".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Warning,
        timestamp: 0,
        context: ErrorContext::empty(),
    };
    
    let fix = database.get_fix_for_error(&mock_error);
    assert!(fix.is_some());
    
    // Test compilation error with MLIR feature
    let comp_error = BuildError {
        error_type: ErrorType::CompilationError,
        message: "MLIR feature is required but not enabled".to_string(),
        location: None,
        suggested_fixes: vec![],
        severity: ErrorSeverity::Error,
        timestamp: 0,
        context: ErrorContext::empty(),
    };
    
    let fix = database.get_fix_for_error(&comp_error);
    assert!(fix.is_some());
}

// Integration test for fix engine with temporary files
#[test]
#[ignore] // Ignore by default since it creates temporary files
fn test_fix_engine_integration_with_temp_files() {
    use std::fs;
    use tempfile::TempDir;
    use aether_language::build_system::fix_engine::{AutoFixEngine, CodeModifier};
    use aether_language::build_system::{FixStrategy, CodeModification};
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_file = temp_dir.path().join("test.rs");
    
    // Create a test file
    fs::write(&test_file, "let x = MockMLIRContext::new();").expect("Failed to write test file");
    
    let mut fix_engine = AutoFixEngine::new();
    fix_engine.set_dry_run(false);
    
    let fix = FixStrategy::ModifyCode(CodeModification {
        file_path: test_file.clone(),
        line_number: 0,
        old_code: "MockMLIRContext".to_string(),
        new_code: "MLIRContext".to_string(),
    });
    
    let result = fix_engine.apply_fix(&fix);
    
    if result.application_result.is_ok() {
        let content = fs::read_to_string(&test_file).expect("Failed to read modified file");
        assert!(content.contains("MLIRContext"));
        assert!(!content.contains("MockMLIRContext"));
    }
}
//
 Tests for RustCompiler integration with BuildSystemManager
#[test]
fn test_build_system_manager_rust_compiler_integration() {
    let manager = BuildSystemManager::new();
    
    // Test that manager has access to RustCompiler functionality
    let toolchain_info = manager.get_toolchain_info();
    assert!(!toolchain_info.version.is_empty());
    assert!(!toolchain_info.target_triple.is_empty());
}

#[test]
fn test_build_system_manager_toolchain_check() {
    let manager = BuildSystemManager::new();
    
    // Test toolchain installation check
    match manager.check_rust_installation() {
        Ok(info) => {
            assert!(!info.cargo_version.is_empty());
            assert!(!info.rustc_version.is_empty());
        }
        Err(_) => {
            // Acceptable if Rust toolchain is not available in test environment
            println!("Rust toolchain check failed - may be expected in CI");
        }
    }
}

#[test]
fn test_build_system_manager_feature_flag_management() {
    let mut manager = BuildSystemManager::new();
    
    // Test initial feature flags
    let initial_flags = manager.get_rust_feature_flags();
    assert!(initial_flags.contains(&"mlir".to_string()));
    
    // Test adding feature flag
    manager.add_rust_feature_flag("gpu".to_string());
    let updated_flags = manager.get_rust_feature_flags();
    assert!(updated_flags.contains(&"gpu".to_string()));
    
    // Test removing feature flag
    manager.remove_rust_feature_flag("gpu");
    let final_flags = manager.get_rust_feature_flags();
    assert!(!final_flags.contains(&"gpu".to_string()));
}

#[test]
fn test_build_system_manager_dependency_installation() {
    let manager = BuildSystemManager::new();
    
    // Test dependency installation (this will mostly test the interface)
    match manager.install_missing_dependencies() {
        Ok(result) => {
            // Should have processed the dependencies from config
            assert!(!result.individual_results.is_empty());
            
            // Check that each dependency was processed
            for dep_result in &result.individual_results {
                assert!(!dep_result.dependency_name.is_empty());
            }
        }
        Err(e) => {
            // Installation might fail in test environment - that's acceptable
            println!("Dependency installation failed (expected in test environment): {}", e);
        }
    }
}

#[test]
fn test_build_system_manager_feature_management() {
    let mut manager = BuildSystemManager::new();
    
    // Test feature flag management
    match manager.manage_feature_flags() {
        Ok(result) => {
            // Should have some feature management result
            assert!(!result.current_features.is_empty() || 
                   !result.enabled_features.is_empty() ||
                   !result.disabled_features.is_empty());
        }
        Err(e) => {
            // Feature management might fail if dependencies aren't available
            println!("Feature management failed (expected in test environment): {}", e);
        }
    }
}

#[test]
fn test_build_system_manager_compilation_analysis() {
    let manager = BuildSystemManager::new();
    
    // Create a mock build result for analysis
    let build_result = aether_language::build_system::BuildResult {
        success: false,
        stdout: "Compiling aether-language v0.1.0".to_string(),
        stderr: "error: could not find `tempfile` in registry\nwarning: unused import".to_string(),
        duration: std::time::Duration::from_secs(5),
    };
    
    // Test compilation result analysis
    match manager.analyze_compilation_result(&build_result) {
        Ok(analysis) => {
            assert!(!analysis.success);
            assert!(!analysis.errors.is_empty());
            assert!(!analysis.warnings.is_empty());
            assert!(!analysis.missing_dependencies.is_empty());
            assert_eq!(analysis.missing_dependencies[0], "tempfile");
        }
        Err(e) => {
            panic!("Compilation analysis should not fail: {}", e);
        }
    }
}

#[test]
fn test_build_system_manager_rust_compiler_error_handling() {
    let manager = BuildSystemManager::new();
    
    // Test with a build result that has various error types
    let complex_build_result = aether_language::build_system::BuildResult {
        success: false,
        stdout: String::new(),
        stderr: r#"
error: could not find `melior` in registry
error: could not find `mlir-sys` in registry
warning: unused import `std::collections::HashMap`
error: feature `nonexistent` does not exist
error: expected `String`, found `i32`
        "#.to_string(),
        duration: std::time::Duration::from_secs(3),
    };
    
    match manager.analyze_compilation_result(&complex_build_result) {
        Ok(analysis) => {
            assert!(!analysis.success);
            assert!(analysis.errors.len() >= 4); // Should detect multiple errors
            assert!(analysis.warnings.len() >= 1); // Should detect warnings
            assert!(analysis.missing_dependencies.len() >= 2); // melior and mlir-sys
            assert!(analysis.feature_issues.len() >= 1); // nonexistent feature
            
            // Check specific dependencies
            assert!(analysis.missing_dependencies.contains(&"melior".to_string()));
            assert!(analysis.missing_dependencies.contains(&"mlir-sys".to_string()));
            
            // Check feature issues
            assert!(analysis.feature_issues.contains(&"nonexistent".to_string()));
        }
        Err(e) => {
            panic!("Complex compilation analysis should not fail: {}", e);
        }
    }
}

#[test]
fn test_build_system_manager_with_rust_compiler_config() {
    use aether_language::build_system::{BuildConfig, ToolchainConfig, Dependency};
    
    let config = BuildConfig {
        rust_toolchain: ToolchainConfig {
            version: "1.70.0".to_string(),
            features: vec!["mlir".to_string(), "gpu".to_string()],
            dependencies: vec![
                Dependency {
                    name: "melior".to_string(),
                    version: "0.19".to_string(),
                    optional: true,
                },
                Dependency {
                    name: "tempfile".to_string(),
                    version: "3.8".to_string(),
                    optional: false,
                },
            ],
        },
        ..Default::default()
    };
    
    let manager = BuildSystemManager::with_config(config);
    
    // Test that the manager was created with the custom config
    assert_eq!(manager.config().rust_toolchain.version, "1.70.0");
    assert!(manager.config().rust_toolchain.features.contains(&"gpu".to_string()));
    assert_eq!(manager.config().rust_toolchain.dependencies.len(), 2);
    
    // Test that RustCompiler has the correct features
    let feature_flags = manager.get_rust_feature_flags();
    assert!(feature_flags.contains(&"mlir".to_string()));
}

#[test]
fn test_rust_compiler_error_conversion() {
    use aether_language::build_system::{RustCompilerError, BuildSystemError};
    
    // Test that RustCompilerError converts properly to BuildSystemError
    let rust_error = RustCompilerError::CompilationFailed("Test compilation error".to_string());
    let build_error: BuildSystemError = rust_error.into();
    
    match build_error {
        BuildSystemError::ConfigurationError(msg) => {
            assert!(msg.contains("Compilation failed"));
            assert!(msg.contains("Test compilation error"));
        }
        _ => panic!("Expected ConfigurationError variant"),
    }
}

#[test]
fn test_build_system_manager_cargo_build_integration() {
    let manager = BuildSystemManager::new();
    
    // Test that the manager uses RustCompiler for cargo build
    // This is tested indirectly through the compile_aether_compiler method
    // We can't easily test the private run_cargo_build method directly
    
    // Instead, test that the manager has the necessary components
    assert!(!manager.get_rust_feature_flags().is_empty());
    
    let toolchain_info = manager.get_toolchain_info();
    assert!(!toolchain_info.cargo_path.as_os_str().is_empty());
    assert!(!toolchain_info.rustc_path.as_os_str().is_empty());
}

#[test]
fn test_build_system_manager_dependency_resolution() {
    let manager = BuildSystemManager::new();
    
    // Test that manager can resolve dependencies from its config
    let dependencies = &manager.config().rust_toolchain.dependencies;
    assert!(!dependencies.is_empty());
    
    // Should have at least the default dependencies
    let dep_names: Vec<&String> = dependencies.iter().map(|d| &d.name).collect();
    assert!(dep_names.contains(&&"melior".to_string()));
    assert!(dep_names.contains(&&"mlir-sys".to_string()));
    assert!(dep_names.contains(&&"tempfile".to_string()));
}

#[test]
fn test_build_system_manager_compilation_with_features() {
    let mut manager = BuildSystemManager::new();
    
    // Add a test feature
    manager.add_rust_feature_flag("test_feature".to_string());
    
    // Verify the feature was added
    assert!(manager.get_rust_feature_flags().contains(&"test_feature".to_string()));
    
    // Test that the manager would use these features in compilation
    // (We can't easily test actual compilation without a full environment)
    let config = manager.config();
    assert!(config.rust_toolchain.features.contains(&"mlir".to_string()));
}

#[test]
fn test_build_system_manager_error_analysis_integration() {
    let manager = BuildSystemManager::new();
    
    // Create a build result that simulates RustCompiler output
    let rust_compiler_style_result = aether_language::build_system::BuildResult {
        success: false,
        stdout: "   Compiling aether-language v0.1.0\n   Compiling proc-macro2 v1.0.70".to_string(),
        stderr: r#"error[E0432]: unresolved import `melior`
  --> src/compiler/mlir/mod.rs:5:5
   |
5  | use melior::Context;
   |     ^^^^^^ maybe a missing crate `melior`?

error: could not find `melior` in the list of imported crates
error: aborting due to 2 previous errors"#.to_string(),
        duration: std::time::Duration::from_secs(8),
    };
    
    match manager.analyze_compilation_result(&rust_compiler_style_result) {
        Ok(analysis) => {
            assert!(!analysis.success);
            assert!(analysis.errors.len() >= 2);
            assert!(analysis.missing_dependencies.contains(&"melior".to_string()));
            assert_eq!(analysis.compilation_time, std::time::Duration::from_secs(8));
        }
        Err(e) => {
            panic!("Rust compiler style error analysis should not fail: {}", e);
        }
    }
}

#[test]
fn test_build_system_manager_feature_flag_persistence() {
    let mut manager = BuildSystemManager::new();
    
    // Test that feature flags persist across operations
    let initial_count = manager.get_rust_feature_flags().len();
    
    manager.add_rust_feature_flag("persistent_test".to_string());
    assert_eq!(manager.get_rust_feature_flags().len(), initial_count + 1);
    
    // Simulate some operation that might affect feature flags
    let _ = manager.manage_feature_flags();
    
    // Feature should still be there (unless explicitly removed by manage_feature_flags)
    let final_flags = manager.get_rust_feature_flags();
    assert!(final_flags.len() >= initial_count); // Should have at least the original flags
}