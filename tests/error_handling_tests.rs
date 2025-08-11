// Comprehensive Error Handling Tests
// Tests for compile-time and runtime error handling with diagnostics

use aether_language::compiler::{
    diagnostics::*,
    error_handling::*,
    parser::{parse_sexpr, ParseError},
    type_checker::{TypeChecker, TypeCheckError},
    types::Type,
};
use aether_language::runtime::{
    error_handling::*,
    native_runtime::RuntimeError,
};
use tempfile::TempDir;
use std::fs;

#[cfg(test)]
mod compile_time_error_tests {
    use super::*;

    #[test]
    fn test_diagnostic_engine_basic() {
        let mut engine = DiagnosticEngine::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        let diagnostic = Diagnostic::error(
            "E0001".to_string(),
            "test error message".to_string(),
            span,
        );
        
        engine.emit(diagnostic);
        
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.warning_count(), 0);
        assert!(engine.has_errors());
        
        let diagnostics = engine.diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, DiagnosticSeverity::Error);
    }

    #[test]
    fn test_diagnostic_with_source_context() {
        let mut engine = DiagnosticEngine::new();
        
        let source_code = "let x = 42\nlet y = x + z\nprint(y)";
        engine.add_source_file("test.ae".to_string(), source_code.to_string());
        
        let pos = SourcePosition::new("test.ae".to_string(), 2, 13, 22);
        let span = SourceSpan::single(pos);
        
        let diagnostic = Diagnostic::error(
            "E0003".to_string(),
            "undefined symbol `z`".to_string(),
            span,
        );
        
        engine.emit(diagnostic);
        
        let formatted = engine.format_diagnostics();
        assert!(formatted.contains("let y = x + z"));
        assert!(formatted.contains("^"));
        assert!(formatted.contains("undefined symbol `z`"));
    }

    #[test]
    fn test_diagnostic_with_fixes() {
        let mut engine = DiagnosticEngine::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 1, 10, 9);
        let span = SourceSpan::single(pos);
        
        let fix = DiagnosticFix::new(
            "add type annotation".to_string(),
            span.clone(),
            ": i32".to_string(),
        );
        
        let diagnostic = Diagnostic::error(
            "E0002".to_string(),
            "type annotation required".to_string(),
            span,
        ).with_fix(fix);
        
        engine.emit(diagnostic);
        
        let formatted = engine.format_diagnostics();
        assert!(formatted.contains("help: add type annotation"));
    }

    #[test]
    fn test_parse_error_handling() {
        let mut handler = AetherErrorHandler::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        let parse_error = ParseError::UnterminatedString(4);
        let aether_error = handler.handle_parse_error(parse_error, span);
        
        assert!(handler.has_errors());
        
        match aether_error {
            AetherError::Parse { error, .. } => {
                match error {
                    ParseError::UnterminatedString(_) => {},
                    _ => panic!("Expected unterminated string error"),
                }
            }
            _ => panic!("Expected parse error"),
        }
        
        let formatted = handler.format_all_diagnostics();
        assert!(formatted.contains("unterminated string literal"));
        assert!(formatted.contains("help: add closing quote"));
    }

    #[test]
    fn test_type_error_handling() {
        let mut handler = AetherErrorHandler::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 2, 8, 15);
        let span = SourceSpan::single(pos);
        
        let type_error = TypeCheckError::TypeMismatch {
            expected: Type::primitive(aether_language::compiler::types::PrimitiveType::Int32),
            actual: Type::primitive(aether_language::compiler::types::PrimitiveType::String),
            location: "assignment".to_string(),
        };
        
        let aether_error = handler.handle_type_error(type_error, span);
        
        assert!(handler.has_errors());
        
        match aether_error {
            AetherError::TypeCheck { error, .. } => {
                match error {
                    TypeCheckError::TypeMismatch { .. } => {},
                    _ => panic!("Expected type mismatch error"),
                }
            }
            _ => panic!("Expected type check error"),
        }
        
        let formatted = handler.format_all_diagnostics();
        assert!(formatted.contains("type mismatch"));
    }

    #[test]
    fn test_performance_analyzer() {
        let mut analyzer = PerformanceAnalyzer::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 5, 10, 45);
        let span = SourceSpan::single(pos);
        
        // Test large tensor warning
        analyzer.analyze_tensor_operation(&[10000, 10000], "matmul", span.clone());
        
        let warnings = analyzer.get_warnings();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].category, PerformanceCategory::Tensor);
        assert!(warnings[0].message.contains("Large tensor operation"));
        
        // Test memory allocation warning
        analyzer.analyze_memory_allocation(200_000_000, "heap", span.clone());
        
        let warnings = analyzer.get_warnings();
        assert_eq!(warnings.len(), 2);
        
        let memory_warnings: Vec<_> = warnings.iter()
            .filter(|w| w.category == PerformanceCategory::Memory)
            .collect();
        assert_eq!(memory_warnings.len(), 1);
        
        // Test GPU usage warning
        analyzer.analyze_gpu_usage("vector_add", 100, span);
        
        let warnings = analyzer.get_warnings();
        assert_eq!(warnings.len(), 3);
        
        let gpu_warnings: Vec<_> = warnings.iter()
            .filter(|w| w.category == PerformanceCategory::GPU)
            .collect();
        assert_eq!(gpu_warnings.len(), 1);
        assert!(gpu_warnings[0].message.contains("small data"));
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds {
            max_tensor_size: 1000,
            max_memory_allocation: 10000,
            max_loop_depth: 3,
            max_function_complexity: 10,
        };
        
        let mut analyzer = PerformanceAnalyzer::with_thresholds(thresholds);
        
        let pos = SourcePosition::new("test.ae".to_string(), 1, 1, 0);
        let span = SourceSpan::single(pos);
        
        // Should trigger warning with custom threshold
        analyzer.analyze_tensor_operation(&[100, 100], "matmul", span);
        
        let warnings = analyzer.get_warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("10000 elements"));
    }

    #[test]
    fn test_error_recovery() {
        let recovery = ErrorRecovery::new();
        
        // Test default strategies
        let strategy = recovery.get_strategy("missing_paren");
        assert!(strategy.is_some());
        
        match strategy.unwrap() {
            RecoveryStrategy::InsertToken(token) => assert_eq!(token, ")"),
            _ => panic!("Expected InsertToken strategy"),
        }
        
        let strategy = recovery.get_strategy("unexpected_token");
        assert!(strategy.is_some());
        
        match strategy.unwrap() {
            RecoveryStrategy::SkipToken => {},
            _ => panic!("Expected SkipToken strategy"),
        }
    }

    #[test]
    fn test_comprehensive_error_handling() {
        let mut handler = AetherErrorHandler::new();
        
        // Add source file
        let source = "let x: i32 = \"hello\"\nlet y = x + z";
        handler.add_source_file("test.ae".to_string(), source.to_string());
        
        // Handle multiple errors
        let pos1 = SourcePosition::new("test.ae".to_string(), 1, 14, 13);
        let span1 = SourceSpan::single(pos1);
        
        let type_error = TypeCheckError::TypeMismatch {
            expected: Type::primitive(aether_language::compiler::types::PrimitiveType::Int32),
            actual: Type::primitive(aether_language::compiler::types::PrimitiveType::String),
            location: "assignment".to_string(),
        };
        
        handler.handle_type_error(type_error, span1);
        
        let pos2 = SourcePosition::new("test.ae".to_string(), 2, 13, 26);
        let span2 = SourceSpan::single(pos2);
        
        let undefined_error = TypeCheckError::UndefinedSymbol {
            name: "z".to_string(),
            location: "expression".to_string(),
        };
        
        handler.handle_type_error(undefined_error, span2);
        
        // Add performance warnings
        let pos3 = SourcePosition::new("test.ae".to_string(), 3, 1, 30);
        let span3 = SourceSpan::single(pos3);
        
        handler.performance_analyzer().analyze_tensor_operation(&[5000, 5000], "matmul", span3);
        
        // Check results
        assert!(handler.has_errors());
        assert_eq!(handler.diagnostic_engine().error_count(), 2);
        
        let formatted = handler.format_all_diagnostics();
        assert!(formatted.contains("type mismatch"));
        assert!(formatted.contains("undefined symbol"));
        assert!(formatted.contains("Performance Warnings"));
        assert!(formatted.contains("Large tensor operation"));
    }
}

#[cfg(test)]
mod runtime_error_tests {
    use super::*;

    #[test]
    fn test_runtime_error_handler_creation() {
        let handler = RuntimeErrorHandler::new();
        
        // Test that default handlers are registered
        let error = RuntimeError::MemoryError("out of memory".to_string());
        let action = handler.handle_error(error);
        
        match action {
            RecoveryAction::Custom(msg) => {
                assert!(msg.contains("garbage collection"));
            }
            _ => panic!("Expected custom recovery action"),
        }
    }

    #[test]
    fn test_resource_tracking() {
        let handler = RuntimeErrorHandler::new();
        
        // Track various resources
        handler.track_allocation(0x1000, 1024, AllocationType::Heap);
        handler.track_allocation(0x2000, 2048, AllocationType::Gpu);
        handler.track_file(1, "test.txt".to_string(), "r".to_string());
        handler.track_gpu_resource(100, GpuResourceType::Buffer, 4096);
        handler.track_tensor(1, vec![32, 64], "f32".to_string(), "gpu".to_string());
        
        // Verify tracking
        {
            let resource_state = handler.resource_tracker.lock().unwrap();
            assert_eq!(resource_state.allocated_memory.len(), 2);
            assert_eq!(resource_state.open_files.len(), 1);
            assert_eq!(resource_state.gpu_resources.len(), 1);
            assert_eq!(resource_state.tensor_references.len(), 1);
        }
        
        // Test untracking
        handler.untrack_allocation(0x1000);
        handler.untrack_file(1);
        
        {
            let resource_state = handler.resource_tracker.lock().unwrap();
            assert_eq!(resource_state.allocated_memory.len(), 1);
            assert_eq!(resource_state.open_files.len(), 0);
        }
    }

    #[test]
    fn test_stack_trace_collection() {
        let collector = StackTraceCollector::new();
        let trace = collector.collect_stack_trace();
        
        assert!(!trace.is_empty());
        
        // Check that frames have expected structure
        for frame in &trace {
            assert!(!frame.function_name.is_empty());
            assert!(!frame.file.is_empty());
            assert!(frame.line > 0);
        }
        
        // Test simple trace
        let simple_trace = collector.collect_simple_trace();
        assert!(!simple_trace.is_empty());
        assert!(simple_trace.contains(&"aether_alloc".to_string()));
    }

    #[test]
    fn test_runtime_stack_frame() {
        let frame = RuntimeStackFrame::new(
            "test_function".to_string(),
            "test.ae".to_string(),
            10,
            5,
        )
        .with_variable("x".to_string(), "42".to_string())
        .with_variable("y".to_string(), "3.14".to_string())
        .with_tensor_shape("input".to_string(), vec![32, 64, 128])
        .with_memory_usage(2048);
        
        assert_eq!(frame.function_name, "test_function");
        assert_eq!(frame.file, "test.ae");
        assert_eq!(frame.line, 10);
        assert_eq!(frame.column, 5);
        assert_eq!(frame.local_variables.len(), 2);
        assert_eq!(frame.tensor_shapes.len(), 1);
        assert_eq!(frame.memory_usage, 2048);
        
        // Test display formatting
        let formatted = format!("{}", frame);
        assert!(formatted.contains("test_function"));
        assert!(formatted.contains("test.ae:10:5"));
        assert!(formatted.contains("locals: "));
        assert!(formatted.contains("tensors: "));
        assert!(formatted.contains("memory: 2048 bytes"));
    }

    #[test]
    fn test_error_classification() {
        let handler = RuntimeErrorHandler::new();
        
        let memory_error = RuntimeError::MemoryError("test".to_string());
        let tensor_error = RuntimeError::TensorError("test".to_string());
        let autodiff_error = RuntimeError::AutodiffError("test".to_string());
        
        // Test that different error types get different recovery actions
        let memory_action = handler.handle_error(memory_error);
        let tensor_action = handler.handle_error(tensor_error);
        let autodiff_action = handler.handle_error(autodiff_error);
        
        // Memory errors should attempt recovery
        match memory_action {
            RecoveryAction::Terminate | RecoveryAction::Custom(_) => {},
            _ => panic!("Unexpected memory error action"),
        }
        
        // Tensor errors should skip
        match tensor_action {
            RecoveryAction::Skip | RecoveryAction::Terminate => {},
            _ => panic!("Unexpected tensor error action"),
        }
        
        // Autodiff errors should terminate
        match autodiff_action {
            RecoveryAction::Terminate => {},
            _ => panic!("Unexpected autodiff error action"),
        }
    }

    #[test]
    fn test_custom_error_handlers() {
        let mut handler = RuntimeErrorHandler::new();
        
        // Register custom handler
        handler.register_error_handler("custom".to_string(), Box::new(|_ctx| {
            RecoveryAction::Retry
        }));
        
        // This won't actually trigger the custom handler since we don't have
        // a "custom" error type, but we can test the registration
        let error = RuntimeError::SystemError("custom error".to_string());
        let action = handler.handle_error(error);
        
        // Should use default handling for system errors
        match action {
            RecoveryAction::Terminate => {},
            _ => panic!("Expected terminate action for unhandled error"),
        }
    }

    #[test]
    fn test_cleanup_handlers() {
        let mut handler = RuntimeErrorHandler::new();
        
        let cleanup_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleanup_called_clone = cleanup_called.clone();
        
        handler.register_cleanup_handler(Box::new(move |_resource_state| {
            cleanup_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        
        // Trigger cleanup by handling a terminating error
        let error = RuntimeError::SystemError("fatal error".to_string());
        let action = handler.handle_error(error);
        
        match action {
            RecoveryAction::Terminate => {
                // Cleanup should have been called
                assert!(cleanup_called.load(std::sync::atomic::Ordering::SeqCst));
            }
            _ => panic!("Expected terminate action"),
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_error_handling() {
        // Create a temporary file with Aether code containing errors
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.ae");
        
        let source_code = r#"
let x: i32 = "hello"  // Type mismatch
let y = x + z         // Undefined symbol
let big_tensor = tensor([10000, 10000])  // Performance warning
"#;
        
        fs::write(&file_path, source_code).unwrap();
        
        // Set up error handler
        let mut handler = AetherErrorHandler::new();
        handler.add_source_file(
            file_path.to_string_lossy().to_string(),
            source_code.to_string(),
        );
        
        // Simulate parsing and type checking errors
        let pos1 = SourcePosition::new(file_path.to_string_lossy().to_string(), 2, 14, 27);
        let span1 = SourceSpan::single(pos1);
        
        let type_error = TypeCheckError::TypeMismatch {
            expected: Type::primitive(aether_language::compiler::types::PrimitiveType::Int32),
            actual: Type::primitive(aether_language::compiler::types::PrimitiveType::String),
            location: "variable initialization".to_string(),
        };
        
        handler.handle_type_error(type_error, span1);
        
        let pos2 = SourcePosition::new(file_path.to_string_lossy().to_string(), 3, 13, 54);
        let span2 = SourceSpan::single(pos2);
        
        let undefined_error = TypeCheckError::UndefinedSymbol {
            name: "z".to_string(),
            location: "binary operation".to_string(),
        };
        
        handler.handle_type_error(undefined_error, span2);
        
        // Add performance warning
        let pos3 = SourcePosition::new(file_path.to_string_lossy().to_string(), 4, 18, 85);
        let span3 = SourceSpan::single(pos3);
        
        handler.performance_analyzer().analyze_tensor_operation(&[10000, 10000], "tensor", span3);
        
        // Verify comprehensive error reporting
        assert!(handler.has_errors());
        assert_eq!(handler.diagnostic_engine().error_count(), 2);
        
        let formatted = handler.format_all_diagnostics();
        
        // Check that all errors and warnings are reported
        assert!(formatted.contains("type mismatch"));
        assert!(formatted.contains("undefined symbol"));
        assert!(formatted.contains("Performance Warnings"));
        assert!(formatted.contains("Large tensor operation"));
        
        // Check that source context is included
        assert!(formatted.contains("let x: i32 = \"hello\""));
        assert!(formatted.contains("let y = x + z"));
        
        // Check that suggestions are included
        assert!(formatted.contains("help:") || formatted.contains("suggestion:"));
        
        println!("Formatted diagnostics:\n{}", formatted);
    }

    #[test]
    fn test_runtime_error_with_cleanup() {
        let mut runtime_handler = RuntimeErrorHandler::new();
        
        // Track some resources
        runtime_handler.track_allocation(0x1000, 1024, AllocationType::Heap);
        runtime_handler.track_allocation(0x2000, 2048, AllocationType::Gpu);
        runtime_handler.track_file(1, "test.txt".to_string(), "w".to_string());
        runtime_handler.track_gpu_resource(100, GpuResourceType::Buffer, 4096);
        runtime_handler.track_tensor(1, vec![32, 64], "f32".to_string(), "gpu".to_string());
        
        // Simulate a fatal runtime error
        let error = RuntimeError::MemoryError("segmentation fault".to_string());
        let action = runtime_handler.handle_error(error);
        
        // Should terminate and perform cleanup
        match action {
            RecoveryAction::Terminate => {
                // Cleanup should have been performed
                // In a real implementation, we'd verify that resources were actually freed
            }
            _ => panic!("Expected terminate action for fatal error"),
        }
    }

    #[test]
    fn test_error_callback_system() {
        let mut handler = AetherErrorHandler::new();
        
        let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();
        
        handler.register_error_callback(move |error| {
            callback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            match error {
                AetherError::Parse { .. } => {
                    // Handle parse error
                }
                AetherError::TypeCheck { .. } => {
                    // Handle type error
                }
                _ => {}
            }
        });
        
        // Trigger an error
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        let parse_error = ParseError::UnexpectedChar('!', 4);
        
        handler.handle_parse_error(parse_error, span);
        
        // Verify callback was called
        assert!(callback_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}