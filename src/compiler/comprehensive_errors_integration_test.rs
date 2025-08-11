// Integration tests for comprehensive error handling system
// Demonstrates how the new error types integrate with existing compiler components

#[cfg(test)]
mod integration_tests {
    use super::super::comprehensive_errors::*;
    use super::super::parser::{Parser, ParseError};
    use super::super::ast::*;
    use super::super::symbol_table::SymbolTable;

    #[test]
    fn test_compiler_error_from_parse_error() {
        let location = SourceLocation::new("test.ae".to_string(), 1, 5);
        
        // Create a syntax error that might come from the parser
        let error = CompilerError::SyntaxError {
            location: location.clone(),
            message: "unexpected token ')'".to_string(),
            suggestions: vec!["try adding a matching '(' before this".to_string()],
        };

        assert_eq!(error.location(), &location);
        assert!(error.message().contains("unexpected token"));
        assert!(error.is_recoverable());
        assert_eq!(error.suggestions().len(), 1);
    }

    #[test]
    fn test_diagnostic_collector_with_multiple_error_types() {
        let mut collector = DiagnosticCollector::new();
        
        // Add source file for context
        collector.add_source_file(
            "test.ae".to_string(),
            "let x: i32 = \"hello\"\nlet y = x + z\nprint(y)".to_string(),
        );

        // Type mismatch error
        let type_error = CompilerError::TypeMismatch {
            location: SourceLocation::new("test.ae".to_string(), 1, 14),
            expected: "i32".to_string(),
            found: "string".to_string(),
            context: Some("variable assignment".to_string()),
        };
        collector.report_error(type_error);

        // Undefined symbol error
        let undefined_error = CompilerError::UndefinedSymbol {
            location: SourceLocation::new("test.ae".to_string(), 2, 13),
            symbol: "z".to_string(),
            suggestions: vec!["did you mean `x`?".to_string()],
        };
        collector.report_error(undefined_error);

        // MLIR error
        let mlir_error = CompilerError::MLIRError {
            location: SourceLocation::new("test.ae".to_string(), 3, 1),
            operation: "tensor.add".to_string(),
            reason: "operand type mismatch".to_string(),
            recovery_hint: Some("ensure operands have compatible types".to_string()),
        };
        collector.report_error(mlir_error);

        assert_eq!(collector.error_count(), 3);
        assert!(collector.has_errors());
        
        let diagnostics = collector.diagnostics();
        assert_eq!(diagnostics.len(), 3);

        // Check that formatting includes source context
        let formatted = collector.format_diagnostics();
        assert!(formatted.contains("type mismatch"));
        assert!(formatted.contains("undefined symbol"));
        assert!(formatted.contains("MLIR operation"));
        assert!(formatted.contains("let x: i32 = \"hello\""));
        assert!(formatted.contains("let y = x + z"));
    }

    #[test]
    fn test_shape_mismatch_error_with_tensor_operations() {
        let mut collector = DiagnosticCollector::new();
        
        let shape_error = CompilerError::ShapeMismatch {
            location: SourceLocation::new("tensor_ops.ae".to_string(), 5, 20),
            expected: vec![3, 4],
            found: vec![4, 3],
            operation: "matrix multiplication".to_string(),
        };

        collector.report_error(shape_error);
        
        let diagnostics = collector.diagnostics();
        assert_eq!(diagnostics.len(), 1);
        
        let diagnostic = &diagnostics[0];
        assert!(diagnostic.message.contains("tensor shape mismatch"));
        assert!(diagnostic.message.contains("matrix multiplication"));
        assert!(diagnostic.message.contains("[3, 4]"));
        assert!(diagnostic.message.contains("[4, 3]"));
        
        // Check suggestions
        assert!(!diagnostic.suggestions.is_empty());
        assert!(diagnostic.suggestions[0].contains("ensure tensor has shape"));
    }

    #[test]
    fn test_mlir_context_error_handling() {
        let mut collector = DiagnosticCollector::new();
        
        let context_error = CompilerError::MLIRContextError {
            location: SourceLocation::new("compiler.ae".to_string(), 1, 1),
            reason: "failed to initialize MLIR context".to_string(),
        };

        collector.report_error(context_error);
        
        assert_eq!(collector.error_count(), 1);
        
        let diagnostics = collector.diagnostics();
        let diagnostic = &diagnostics[0];
        
        assert!(diagnostic.message.contains("MLIR context error"));
        assert!(diagnostic.message.contains("failed to initialize"));
    }

    #[test]
    fn test_memory_safety_error() {
        let mut collector = DiagnosticCollector::new();
        
        let memory_error = CompilerError::MemorySafetyError {
            location: SourceLocation::new("unsafe.ae".to_string(), 10, 5),
            violation_type: "use after free".to_string(),
            description: "variable 'ptr' used after being freed".to_string(),
        };

        collector.report_error(memory_error);
        
        let diagnostics = collector.diagnostics();
        let diagnostic = &diagnostics[0];
        
        assert!(diagnostic.message.contains("memory safety violation"));
        assert!(diagnostic.message.contains("use after free"));
        assert!(diagnostic.message.contains("variable 'ptr' used after being freed"));
        
        // Memory safety errors should not be recoverable
        let error = CompilerError::MemorySafetyError {
            location: SourceLocation::new("unsafe.ae".to_string(), 10, 5),
            violation_type: "use after free".to_string(),
            description: "variable 'ptr' used after being freed".to_string(),
        };
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_linear_type_violation() {
        let mut collector = DiagnosticCollector::new();
        
        let linear_error = CompilerError::LinearTypeViolation {
            location: SourceLocation::new("linear.ae".to_string(), 7, 12),
            variable: "resource".to_string(),
            violation: "used more than once".to_string(),
        };

        collector.report_error(linear_error);
        
        let diagnostics = collector.diagnostics();
        let diagnostic = &diagnostics[0];
        
        assert!(diagnostic.message.contains("linear type violation"));
        assert!(diagnostic.message.contains("resource"));
        assert!(diagnostic.message.contains("used more than once"));
        
        // Check that suggestions include linear type guidance
        assert!(!diagnostic.suggestions.is_empty());
        assert!(diagnostic.suggestions[0].contains("exactly once"));
    }

    #[test]
    fn test_arity_mismatch_error() {
        let mut collector = DiagnosticCollector::new();
        
        let arity_error = CompilerError::ArityMismatch {
            location: SourceLocation::new("functions.ae".to_string(), 3, 8),
            function: "add".to_string(),
            expected: 2,
            found: 3,
        };

        collector.report_error(arity_error);
        
        let diagnostics = collector.diagnostics();
        let diagnostic = &diagnostics[0];
        
        assert!(diagnostic.message.contains("function `add` expects 2 arguments, found 3"));
        
        // Check suggestions
        assert!(!diagnostic.suggestions.is_empty());
        assert!(diagnostic.suggestions[0].contains("provide exactly 2 arguments"));
    }

    #[test]
    fn test_internal_compiler_error() {
        let mut collector = DiagnosticCollector::new();
        
        let mut debug_info = std::collections::HashMap::new();
        debug_info.insert("phase".to_string(), "type_checking".to_string());
        debug_info.insert("node_id".to_string(), "42".to_string());
        
        let internal_error = CompilerError::InternalError {
            location: SourceLocation::new("compiler_bug.ae".to_string(), 15, 3),
            message: "unexpected null pointer in type checker".to_string(),
            debug_info,
        };

        collector.report_error(internal_error);
        
        let diagnostics = collector.diagnostics();
        let diagnostic = &diagnostics[0];
        
        assert!(diagnostic.message.contains("internal compiler error"));
        assert!(diagnostic.message.contains("unexpected null pointer"));
        
        // Internal errors should not be recoverable
        let error = CompilerError::InternalError {
            location: SourceLocation::new("test.ae".to_string(), 1, 1),
            message: "test".to_string(),
            debug_info: std::collections::HashMap::new(),
        };
        assert!(!error.is_recoverable());
        
        // Should suggest reporting the bug
        assert!(!diagnostic.suggestions.is_empty());
        assert!(diagnostic.suggestions[0].contains("compiler bug"));
    }

    #[test]
    fn test_diagnostic_summary() {
        let mut collector = DiagnosticCollector::new();
        
        // Add various types of diagnostics
        collector.report_error(CompilerError::SyntaxError {
            location: SourceLocation::new("test.ae".to_string(), 1, 1),
            message: "syntax error".to_string(),
            suggestions: vec![],
        });
        
        collector.report_warning(
            SourceLocation::new("test.ae".to_string(), 2, 1),
            "unused variable".to_string(),
        );
        
        collector.report_info(
            SourceLocation::new("test.ae".to_string(), 3, 1),
            "optimization applied".to_string(),
        );

        let summary = collector.summary();
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.info_count, 1);
        assert_eq!(summary.total_diagnostics, 3);
        assert!(!summary.max_errors_reached);
        
        let summary_str = summary.to_string();
        assert!(summary_str.contains("1 error(s)"));
        assert!(summary_str.contains("1 warning(s)"));
        assert!(summary_str.contains("1 info"));
    }

    #[test]
    fn test_max_errors_behavior() {
        let mut collector = DiagnosticCollector::with_max_errors(2);
        
        // Try to add 3 errors, but only 2 should be collected
        for i in 0..3 {
            collector.report_error(CompilerError::SyntaxError {
                location: SourceLocation::new("test.ae".to_string(), i + 1, 1),
                message: format!("error {}", i),
                suggestions: vec![],
            });
        }

        assert_eq!(collector.error_count(), 2);
        assert!(collector.max_errors_reached());
        
        let summary = collector.summary();
        assert!(summary.max_errors_reached);
    }
}