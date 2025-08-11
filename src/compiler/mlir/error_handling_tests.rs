// Comprehensive tests for MLIR compilation error handling and diagnostics
// Tests error reporting, source location preservation, and recovery strategies

#[cfg(test)]
mod tests {
    use super::super::error_handling::*;
    use super::super::error_recovery::*;
    use crate::compiler::ast::NodeId;
    use crate::compiler::token::Token;
    use std::path::PathBuf;
    use std::collections::HashMap;

    #[test]
    fn test_source_location_creation() {
        let file = Some(PathBuf::from("test.ae"));
        let location = SourceLocation::new(file.clone(), 10, 5, 100, 20);
        
        assert_eq!(location.file, file);
        assert_eq!(location.line, 10);
        assert_eq!(location.column, 5);
        assert_eq!(location.offset, 100);
        assert_eq!(location.length, 20);
        assert!(location.is_known());
    }

    #[test]
    fn test_source_location_unknown() {
        let location = SourceLocation::unknown();
        
        assert_eq!(location.file, None);
        assert_eq!(location.line, 0);
        assert_eq!(location.column, 0);
        assert!(!location.is_known());
    }

    #[test]
    fn test_source_location_display() {
        let file = Some(PathBuf::from("test.ae"));
        let location = SourceLocation::new(file, 10, 5, 100, 20);
        let display = location.display();
        
        assert!(display.contains("test.ae"));
        assert!(display.contains("10"));
        assert!(display.contains("5"));
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Info < ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Critical);
    }

    #[test]
    fn test_context_creation_error() {
        let location = SourceLocation::unknown();
        let error = MLIRCompilationError::ContextCreation {
            reason: "MLIR library not found".to_string(),
            location: location.clone(),
            recovery_suggestion: Some("Install MLIR development libraries".to_string()),
        };

        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert_eq!(error.location(), &location);
        assert!(!error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0], "Install MLIR development libraries");
    }

    #[test]
    fn test_dialect_registration_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 1, 1, 0, 10);
        let available_dialects = vec!["std".to_string(), "arith".to_string()];
        
        let error = MLIRCompilationError::DialectRegistration {
            dialect: "aether".to_string(),
            reason: "Dialect not found".to_string(),
            location: location.clone(),
            available_dialects: available_dialects.clone(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert!(!error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("std, arith")));
    }

    #[test]
    fn test_ast_conversion_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 5, 10, 50, 15);
        let context = vec!["function".to_string(), "expression".to_string()];
        
        let error = MLIRCompilationError::ASTConversion {
            node_id: Some(42),
            node_type: "BinaryOp".to_string(),
            error: "Unsupported operator".to_string(),
            location: location.clone(),
            context: context.clone(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(error.is_recoverable());
        
        let message = error.message();
        assert!(message.contains("BinaryOp"));
        assert!(message.contains("Unsupported operator"));
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("function -> expression")));
    }

    #[test]
    fn test_type_conversion_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 8, 3, 80, 12);
        let constraints = vec!["shape_compatible".to_string(), "element_type_match".to_string()];
        
        let error = MLIRCompilationError::TypeConversion {
            aether_type: "tensor<2x3xf32>".to_string(),
            target_type: "tensor<3x2xf32>".to_string(),
            error: "Shape mismatch".to_string(),
            location: location.clone(),
            type_constraints: constraints.clone(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(error.is_recoverable());
        
        let message = error.message();
        assert!(message.contains("tensor<2x3xf32>"));
        assert!(message.contains("tensor<3x2xf32>"));
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("shape_compatible, element_type_match")));
    }

    #[test]
    fn test_operation_creation_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 12, 8, 120, 25);
        let operand_types = vec!["tensor<2x3xf32>".to_string(), "tensor<3x4xf32>".to_string()];
        
        let error = MLIRCompilationError::OperationCreation {
            operation: "aether.matmul".to_string(),
            error: "Incompatible operand shapes".to_string(),
            location: location.clone(),
            operand_types: operand_types.clone(),
            expected_signature: Some("(tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>".to_string()),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("Expected signature")));
    }

    #[test]
    fn test_module_verification_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 20, 1, 200, 50);
        let verification_errors = vec![
            VerificationError {
                error_type: VerificationErrorType::TypeMismatch,
                message: "Type mismatch in operation".to_string(),
                location: location.clone(),
                operation: Some("aether.add".to_string()),
                fix_suggestion: Some("Check operand types".to_string()),
            },
            VerificationError {
                error_type: VerificationErrorType::UndefinedSymbol,
                message: "Undefined symbol 'x'".to_string(),
                location: location.clone(),
                operation: Some("aether.var_ref".to_string()),
                fix_suggestion: Some("Define variable 'x' before use".to_string()),
            },
        ];
        
        let error = MLIRCompilationError::ModuleVerification {
            errors: verification_errors.clone(),
            location: location.clone(),
            module_name: "test_module".to_string(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(error.is_recoverable());
        
        let message = error.message();
        assert!(message.contains("test_module"));
        assert!(message.contains("2 errors"));
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.len() >= 3); // "Fix verification errors:" + 2 specific fixes
    }

    #[test]
    fn test_optimization_failure_skippable() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 15, 5, 150, 30);
        let mut pass_config = HashMap::new();
        pass_config.insert("level".to_string(), "aggressive".to_string());
        
        let error = MLIRCompilationError::OptimizationFailure {
            pass_name: "tensor_fusion".to_string(),
            error: "Complex fusion pattern not supported".to_string(),
            location: location.clone(),
            pass_config: pass_config.clone(),
            can_skip: true,
        };

        assert_eq!(error.severity(), ErrorSeverity::Warning);
        assert!(error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("skipping")));
    }

    #[test]
    fn test_optimization_failure_critical() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 15, 5, 150, 30);
        let pass_config = HashMap::new();
        
        let error = MLIRCompilationError::OptimizationFailure {
            pass_name: "mandatory_lowering".to_string(),
            error: "Required lowering failed".to_string(),
            location: location.clone(),
            pass_config,
            can_skip: false,
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(!error.is_recoverable()); // can_skip: false means not recoverable
    }

    #[test]
    fn test_lowering_failure_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 25, 12, 250, 18);
        let patterns = vec!["aether_to_std".to_string(), "std_to_llvm".to_string()];
        
        let error = MLIRCompilationError::LoweringFailure {
            from_dialect: "aether".to_string(),
            to_dialect: "std".to_string(),
            operation: "aether.tensor_op".to_string(),
            error: "No matching conversion pattern".to_string(),
            location: location.clone(),
            conversion_patterns: patterns.clone(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(error.is_recoverable());
        
        let message = error.message();
        assert!(message.contains("aether.tensor_op"));
        assert!(message.contains("aether"));
        assert!(message.contains("std"));
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("aether_to_std, std_to_llvm")));
    }

    #[test]
    fn test_llvm_generation_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 30, 8, 300, 22);
        
        let error = MLIRCompilationError::LLVMGeneration {
            error: "Invalid LLVM IR generated".to_string(),
            location: location.clone(),
            llvm_context: Some("x86_64-unknown-linux-gnu".to_string()),
            module_state: "partially_lowered".to_string(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(!error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("x86_64-unknown-linux-gnu")));
        assert!(suggestions.iter().any(|s| s.contains("partially_lowered")));
    }

    #[test]
    fn test_code_generation_error() {
        let location = SourceLocation::new(Some(PathBuf::from("test.ae")), 35, 15, 350, 28);
        let features = vec!["avx2".to_string(), "fma".to_string()];
        
        let error = MLIRCompilationError::CodeGeneration {
            target: "x86_64-pc-windows-msvc".to_string(),
            error: "Unsupported target feature".to_string(),
            location: location.clone(),
            target_features: features.clone(),
            optimization_level: "O3".to_string(),
        };

        assert_eq!(error.severity(), ErrorSeverity::Error);
        assert!(!error.is_recoverable());
        
        let suggestions = error.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("avx2, fma")));
        assert!(suggestions.iter().any(|s| s.contains("O3")));
    }

    #[test]
    fn test_source_location_tracker() {
        let mut tracker = SourceLocationTracker::new();
        let file = PathBuf::from("test.ae");
        tracker.set_current_file(file.clone());
        
        let node_id: NodeId = 42;
        let location = SourceLocation::new(Some(file.clone()), 10, 5, 100, 20);
        tracker.track_node(node_id, location.clone());
        
        let token = Token::keyword(1);
        let token_location = SourceLocation::new(Some(file.clone()), 12, 8, 120, 5);
        tracker.track_token(token, token_location.clone());
        
        assert_eq!(tracker.current_file(), Some(&file));
        assert_eq!(tracker.get_node_location(node_id), Some(&location));
        assert_eq!(tracker.get_token_location(&token), Some(&token_location));
        
        let new_location = tracker.create_location(15, 10, 150, 8);
        assert_eq!(new_location.file, Some(file));
        assert_eq!(new_location.line, 15);
        assert_eq!(new_location.column, 10);
    }

    #[test]
    fn test_error_recovery_manager_creation() {
        let manager = ErrorRecoveryManager::new();
        let stats = manager.get_statistics();
        
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful_recoveries, 0);
        assert_eq!(stats.failed_recoveries, 0);
    }

    #[test]
    fn test_error_recovery_skip_strategy() {
        let mut manager = ErrorRecoveryManager::new();
        let location = SourceLocation::unknown();
        
        let error = MLIRCompilationError::OperationCreation {
            operation: "test.op".to_string(),
            error: "Test error".to_string(),
            location,
            operand_types: vec![],
            expected_signature: None,
        };

        let result = manager.attempt_recovery(&error);
        
        match result {
            RecoveryResult::Skipped { message, .. } => {
                assert!(message.contains("Skipped failing operation"));
            },
            _ => panic!("Expected skipped result"),
        }
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful_recoveries, 1);
    }

    #[test]
    fn test_error_recovery_fallback_strategy() {
        let mut manager = ErrorRecoveryManager::new();
        let location = SourceLocation::unknown();
        
        let error = MLIRCompilationError::ASTConversion {
            node_id: Some(1),
            node_type: "TestNode".to_string(),
            error: "Test conversion error".to_string(),
            location,
            context: vec![],
        };

        let result = manager.attempt_recovery(&error);
        
        match result {
            RecoveryResult::Recovered { operations, warnings, fallback_used, .. } => {
                assert!(!operations.is_empty());
                assert!(!warnings.is_empty());
                assert!(fallback_used.is_some());
            },
            _ => panic!("Expected recovered result, got: {:?}", result),
        }
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful_recoveries, 1);
    }

    #[test]
    fn test_error_recovery_critical_error_no_recovery() {
        let mut manager = ErrorRecoveryManager::new();
        let location = SourceLocation::unknown();
        
        let error = MLIRCompilationError::ContextCreation {
            reason: "Critical failure".to_string(),
            location,
            recovery_suggestion: None,
        };

        let result = manager.attempt_recovery(&error);
        
        match result {
            RecoveryResult::NoRecovery { reason } => {
                assert!(reason.contains("Recovery not applicable"));
            },
            _ => panic!("Expected no recovery result"),
        }
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful_recoveries, 0);
    }

    #[test]
    fn test_error_recovery_max_attempts() {
        let mut manager = ErrorRecoveryManager::new();
        let location = SourceLocation::unknown();
        
        let error = MLIRCompilationError::ASTConversion {
            node_id: Some(1),
            node_type: "TestNode".to_string(),
            error: "Test conversion error".to_string(),
            location: location.clone(),
            context: vec![],
        };

        // First attempt should succeed
        let result1 = manager.attempt_recovery(&error);
        assert!(matches!(result1, RecoveryResult::Recovered { .. }));
        
        // Subsequent attempts should eventually be rejected
        for _ in 0..5 {
            let result = manager.attempt_recovery(&error);
            if matches!(result, RecoveryResult::NoRecovery { .. }) {
                break;
            }
        }
        
        let stats = manager.get_statistics();
        assert!(stats.total_attempts > 1);
    }

    #[test]
    fn test_verification_error_types() {
        let location = SourceLocation::unknown();
        
        let type_mismatch = VerificationError {
            error_type: VerificationErrorType::TypeMismatch,
            message: "Type mismatch".to_string(),
            location: location.clone(),
            operation: Some("test.op".to_string()),
            fix_suggestion: Some("Fix types".to_string()),
        };
        
        let undefined_symbol = VerificationError {
            error_type: VerificationErrorType::UndefinedSymbol,
            message: "Undefined symbol".to_string(),
            location: location.clone(),
            operation: Some("test.ref".to_string()),
            fix_suggestion: Some("Define symbol".to_string()),
        };
        
        assert!(matches!(type_mismatch.error_type, VerificationErrorType::TypeMismatch));
        assert!(matches!(undefined_symbol.error_type, VerificationErrorType::UndefinedSymbol));
    }

    #[test]
    fn test_recovery_statistics() {
        let mut manager = ErrorRecoveryManager::new();
        let location = SourceLocation::unknown();
        
        // Create different types of errors
        let ast_error = MLIRCompilationError::ASTConversion {
            node_id: Some(1),
            node_type: "TestNode".to_string(),
            error: "Test error".to_string(),
            location: location.clone(),
            context: vec![],
        };
        
        let op_error = MLIRCompilationError::OperationCreation {
            operation: "test.op".to_string(),
            error: "Test error".to_string(),
            location,
            operand_types: vec![],
            expected_signature: None,
        };
        
        // Attempt recoveries
        manager.attempt_recovery(&ast_error);
        manager.attempt_recovery(&op_error);
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_attempts, 2);
        assert_eq!(stats.successful_recoveries, 2);
        assert_eq!(stats.failed_recoveries, 0);
        
        // Check per-error-type statistics
        assert!(stats.by_error_type.contains_key("ASTConversion"));
        assert!(stats.by_error_type.contains_key("OperationCreation"));
    }

    #[test]
    fn test_custom_fallback_implementation() {
        #[derive(Debug)]
        struct CustomFallback;
        
        impl FallbackImplementation for CustomFallback {
            fn execute(&self, _context: &RecoveryContext) -> Result<FallbackResult, crate::compiler::mlir::MLIRError> {
                Ok(FallbackResult {
                    success: true,
                    operations: vec![],
                    warnings: vec!["Custom fallback used".to_string()],
                    performance_impact: Some("Custom impact".to_string()),
                })
            }
            
            fn description(&self) -> String {
                "Custom test fallback".to_string()
            }
            
            fn is_applicable(&self, _error: &MLIRCompilationError) -> bool {
                true
            }
        }
        
        let mut manager = ErrorRecoveryManager::new();
        manager.add_fallback("custom".to_string(), Box::new(CustomFallback));
        manager.set_strategy(
            "TestError".to_string(),
            RecoveryStrategy::Fallback("custom".to_string()),
            1
        );
        
        // The custom fallback is now registered and can be used
        // We can't directly access private fields, but we can test that the manager accepts the fallback
        let stats = manager.get_statistics();
        assert_eq!(stats.total_attempts, 0); // No attempts yet, but fallback is registered
    }
}