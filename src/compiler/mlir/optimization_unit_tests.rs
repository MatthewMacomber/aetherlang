// Comprehensive unit tests for optimization passes
// Tests individual optimization passes and the optimization pipeline

use super::optimization::*;
use super::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use std::collections::HashMap;

// ===== MOCK OPTIMIZATION PASSES FOR TESTING =====

#[derive(Debug)]
struct MockSimplePass {
    name: String,
    should_change: bool,
    dependencies: Vec<String>,
    conflicts: Vec<String>,
}

impl MockSimplePass {
    fn new(name: &str) -> Self {
        MockSimplePass {
            name: name.to_string(),
            should_change: true,
            dependencies: Vec::new(),
            conflicts: Vec::new(),
        }
    }
    
    fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }
    
    fn with_conflicts(mut self, conflicts: Vec<String>) -> Self {
        self.conflicts = conflicts;
        self
    }
    
    fn no_change(mut self) -> Self {
        self.should_change = false;
        self
    }
}

impl OptimizationPass for MockSimplePass {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        "Mock optimization pass for testing"
    }
    
    fn run(&self, module: &mut MLIRModule, _context: &MLIRContext) -> Result<PassResult, crate::compiler::mlir::MLIRError> {
        if self.should_change {
            // Add a mock attribute to the first operation to simulate a change
            if let Some(op) = module.operations_mut().first_mut() {
                op.add_attribute(
                    format!("optimized_by_{}", self.name),
                    MLIRAttribute::Boolean(true)
                );
                Ok(PassResult::changed(1, 0, 0))
            } else {
                Ok(PassResult::no_change())
            }
        } else {
            Ok(PassResult::no_change())
        }
    }
    
    fn dependencies(&self) -> Vec<String> {
        self.dependencies.clone()
    }
    
    fn conflicts(&self) -> Vec<String> {
        self.conflicts.clone()
    }
}

#[derive(Debug)]
struct MockFailingPass {
    name: String,
}

impl MockFailingPass {
    fn new(name: &str) -> Self {
        MockFailingPass {
            name: name.to_string(),
        }
    }
}

impl OptimizationPass for MockFailingPass {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        "Mock pass that always fails"
    }
    
    fn run(&self, _module: &mut MLIRModule, _context: &MLIRContext) -> Result<PassResult, crate::compiler::mlir::MLIRError> {
        Err(crate::compiler::mlir::MLIRError::OptimizationError(
            format!("Mock failure in pass {}", self.name)
        ))
    }
}

// ===== UNIT TESTS =====

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    fn create_test_module_with_operations(context: &MLIRContext, count: usize) -> MLIRModule {
        let mut module = context.create_module("test").unwrap();
        
        for i in 0..count {
            let mut op = MLIROperation::new(format!("test.op_{}", i));
            op.add_attribute("id".to_string(), MLIRAttribute::Integer(i as i64));
            module.add_operation(op).unwrap();
        }
        
        module
    }

    #[test]
    fn test_pass_result_creation() {
        // Test no change result
        let no_change = PassResult::no_change();
        assert!(!no_change.changed);
        assert_eq!(no_change.operations_modified, 0);
        assert_eq!(no_change.operations_added, 0);
        assert_eq!(no_change.operations_removed, 0);
        assert!(no_change.metrics.is_empty());
        assert!(no_change.diagnostics.is_empty());

        // Test changed result
        let changed = PassResult::changed(2, 1, 0);
        assert!(changed.changed);
        assert_eq!(changed.operations_modified, 2);
        assert_eq!(changed.operations_added, 1);
        assert_eq!(changed.operations_removed, 0);

        // Test metrics and diagnostics
        let mut result = PassResult::no_change();
        result.add_metric("test_metric".to_string(), 42.0);
        result.add_diagnostic("test diagnostic".to_string());
        
        assert_eq!(result.metrics.get("test_metric"), Some(&42.0));
        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0], "test diagnostic");
    }

    #[test]
    fn test_pass_config_default() {
        let config = PassConfig::default();
        assert!(!config.debug);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.optimization_level, 2);
        assert_eq!(config.target_arch, "generic");
        assert!(config.custom_config.is_empty());
    }

    #[test]
    fn test_optimization_pipeline_creation() {
        let pipeline = OptimizationPipeline::new();
        assert_eq!(pipeline.get_pass_names().len(), 0);
        assert!(!pipeline.has_pass("nonexistent"));

        let config = PassConfig {
            debug: true,
            optimization_level: 3,
            ..Default::default()
        };
        let pipeline_with_config = OptimizationPipeline::with_config(config);
        assert_eq!(pipeline_with_config.get_pass_names().len(), 0);
    }

    #[test]
    fn test_pipeline_pass_management() {
        let mut pipeline = OptimizationPipeline::new();
        
        // Add passes
        pipeline.add_pass(MockSimplePass::new("pass1"));
        pipeline.add_pass(MockSimplePass::new("pass2"));
        
        assert_eq!(pipeline.get_pass_names().len(), 2);
        assert!(pipeline.has_pass("pass1"));
        assert!(pipeline.has_pass("pass2"));
        assert!(!pipeline.has_pass("pass3"));
        
        // Remove pass
        assert!(pipeline.remove_pass("pass1"));
        assert_eq!(pipeline.get_pass_names().len(), 1);
        assert!(!pipeline.has_pass("pass1"));
        assert!(pipeline.has_pass("pass2"));
        
        // Try to remove non-existent pass
        assert!(!pipeline.remove_pass("nonexistent"));
        assert_eq!(pipeline.get_pass_names().len(), 1);
    }

    #[test]
    fn test_simple_pass_execution() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("test_pass"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.total_changed);
        assert_eq!(pipeline_result.passes_run, 1);
        assert_eq!(pipeline_result.total_operations_modified, 1);
        assert_eq!(pipeline_result.total_operations_added, 0);
        assert_eq!(pipeline_result.total_operations_removed, 0);
        assert!(pipeline_result.execution_time_ms > 0.0);
        
        // Check that the operation was actually modified
        let operations = module.operations();
        assert!(operations[0].attributes.contains_key("optimized_by_test_pass"));
    }

    #[test]
    fn test_multiple_pass_execution() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("pass1"));
        pipeline.add_pass(MockSimplePass::new("pass2"));
        pipeline.add_pass(MockSimplePass::new("pass3"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.total_changed);
        assert_eq!(pipeline_result.passes_run, 3);
        assert_eq!(pipeline_result.total_operations_modified, 3); // Each pass modifies one operation
        
        // Check that all passes ran
        assert!(pipeline_result.pass_results.contains_key("pass1"));
        assert!(pipeline_result.pass_results.contains_key("pass2"));
        assert!(pipeline_result.pass_results.contains_key("pass3"));
        
        // Check that the first operation has attributes from all passes
        let operations = module.operations();
        assert!(operations[0].attributes.contains_key("optimized_by_pass1"));
        assert!(operations[0].attributes.contains_key("optimized_by_pass2"));
        assert!(operations[0].attributes.contains_key("optimized_by_pass3"));
    }

    #[test]
    fn test_pass_with_no_changes() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("changing_pass"));
        pipeline.add_pass(MockSimplePass::new("no_change_pass").no_change());
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.total_changed); // One pass made changes
        assert_eq!(pipeline_result.passes_run, 2);
        assert_eq!(pipeline_result.total_operations_modified, 1); // Only one pass made changes
        
        // Check individual pass results
        let changing_result = pipeline_result.pass_results.get("changing_pass").unwrap();
        assert!(changing_result.changed);
        
        let no_change_result = pipeline_result.pass_results.get("no_change_pass").unwrap();
        assert!(!no_change_result.changed);
    }

    #[test]
    fn test_pass_dependencies() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        // Add passes with dependencies: pass2 depends on pass1
        pipeline.add_pass(MockSimplePass::new("pass2").with_dependencies(vec!["pass1".to_string()]));
        pipeline.add_pass(MockSimplePass::new("pass1")); // Add in reverse order
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert_eq!(pipeline_result.passes_run, 2);
        
        // Both passes should have run successfully despite being added in reverse order
        assert!(pipeline_result.pass_results.contains_key("pass1"));
        assert!(pipeline_result.pass_results.contains_key("pass2"));
    }

    #[test]
    fn test_missing_dependency_error() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        // Add pass with missing dependency
        pipeline.add_pass(MockSimplePass::new("pass1").with_dependencies(vec!["missing_pass".to_string()]));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            crate::compiler::mlir::MLIRError::OptimizationError(msg) => {
                assert!(msg.contains("depends on"));
                assert!(msg.contains("missing_pass"));
            }
            _ => panic!("Expected OptimizationError"),
        }
    }

    #[test]
    fn test_pass_conflicts() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        // Add conflicting passes
        pipeline.add_pass(MockSimplePass::new("pass1").with_conflicts(vec!["pass2".to_string()]));
        pipeline.add_pass(MockSimplePass::new("pass2"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            crate::compiler::mlir::MLIRError::OptimizationError(msg) => {
                assert!(msg.contains("conflicts with"));
            }
            _ => panic!("Expected OptimizationError"),
        }
    }

    #[test]
    fn test_failing_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("good_pass"));
        pipeline.add_pass(MockFailingPass::new("failing_pass"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            crate::compiler::mlir::MLIRError::OptimizationError(msg) => {
                assert!(msg.contains("Mock failure"));
            }
            _ => panic!("Expected OptimizationError"),
        }
    }

    #[test]
    fn test_empty_module() {
        let context = create_test_context();
        let mut module = context.create_module("empty").unwrap();
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("test_pass"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert!(!pipeline_result.total_changed); // No operations to modify
        assert_eq!(pipeline_result.passes_run, 0); // Pass should be skipped
    }

    #[test]
    fn test_pipeline_statistics() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        pipeline.add_pass(MockSimplePass::new("pass1"));
        pipeline.add_pass(MockSimplePass::new("pass2").no_change());
        pipeline.add_pass(MockSimplePass::new("pass3"));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        
        // Check overall statistics
        assert_eq!(pipeline_result.passes_run, 3);
        assert_eq!(pipeline_result.total_operations_modified, 2); // pass1 and pass3 made changes
        assert!(pipeline_result.execution_time_ms > 0.0);
        
        // Check individual pass results
        assert_eq!(pipeline_result.pass_results.len(), 3);
        
        let pass1_result = pipeline_result.pass_results.get("pass1").unwrap();
        assert!(pass1_result.changed);
        assert_eq!(pass1_result.operations_modified, 1);
        
        let pass2_result = pipeline_result.pass_results.get("pass2").unwrap();
        assert!(!pass2_result.changed);
        assert_eq!(pass2_result.operations_modified, 0);
        
        let pass3_result = pipeline_result.pass_results.get("pass3").unwrap();
        assert!(pass3_result.changed);
        assert_eq!(pass3_result.operations_modified, 1);
    }

    #[test]
    fn test_debug_mode() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 2);
        
        let config = PassConfig {
            debug: true,
            ..Default::default()
        };
        let mut pipeline = OptimizationPipeline::with_config(config);
        
        pipeline.add_pass(MockSimplePass::new("debug_pass"));
        
        // In debug mode, the pipeline should still work but with debug output
        // (We can't easily test the debug output in unit tests, but we can verify it doesn't break)
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert!(pipeline_result.total_changed);
        assert_eq!(pipeline_result.passes_run, 1);
    }

    #[test]
    fn test_pass_can_run_check() {
        let context = create_test_context();
        
        // Test with empty module
        let mut empty_module = context.create_module("empty").unwrap();
        let pass = MockSimplePass::new("test_pass");
        assert!(!pass.can_run(&empty_module)); // Default implementation checks for non-empty operations
        
        // Test with non-empty module
        let mut module_with_ops = create_test_module_with_operations(&context, 1);
        assert!(pass.can_run(&module_with_ops));
    }

    #[test]
    fn test_complex_dependency_chain() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 5);
        let mut pipeline = OptimizationPipeline::new();
        
        // Create a complex dependency chain: pass3 -> pass2 -> pass1
        pipeline.add_pass(MockSimplePass::new("pass3").with_dependencies(vec!["pass2".to_string()]));
        pipeline.add_pass(MockSimplePass::new("pass1")); // No dependencies
        pipeline.add_pass(MockSimplePass::new("pass2").with_dependencies(vec!["pass1".to_string()]));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        assert_eq!(pipeline_result.passes_run, 3);
        
        // All passes should have run successfully
        assert!(pipeline_result.pass_results.contains_key("pass1"));
        assert!(pipeline_result.pass_results.contains_key("pass2"));
        assert!(pipeline_result.pass_results.contains_key("pass3"));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 3);
        let mut pipeline = OptimizationPipeline::new();
        
        // Create circular dependency: pass1 -> pass2 -> pass1
        pipeline.add_pass(MockSimplePass::new("pass1").with_dependencies(vec!["pass2".to_string()]));
        pipeline.add_pass(MockSimplePass::new("pass2").with_dependencies(vec!["pass1".to_string()]));
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_err());
        
        // Should detect circular dependency
        match result.unwrap_err() {
            crate::compiler::mlir::MLIRError::OptimizationError(msg) => {
                assert!(msg.contains("circular") || msg.contains("cycle") || msg.contains("dependency"));
            }
            _ => panic!("Expected OptimizationError for circular dependency"),
        }
    }

    #[test]
    fn test_pass_metrics_and_diagnostics() {
        let context = create_test_context();
        let mut module = create_test_module_with_operations(&context, 2);
        
        // Create a custom pass that adds metrics and diagnostics
        #[derive(Debug)]
        struct MetricsPass;
        
        impl OptimizationPass for MetricsPass {
            fn name(&self) -> &str { "metrics_pass" }
            fn description(&self) -> &str { "Pass that adds metrics and diagnostics" }
            
            fn run(&self, module: &mut MLIRModule, _context: &MLIRContext) -> Result<PassResult, crate::compiler::mlir::MLIRError> {
                let mut result = PassResult::changed(1, 0, 0);
                result.add_metric("operations_analyzed".to_string(), module.operations().len() as f64);
                result.add_metric("complexity_score".to_string(), 42.5);
                result.add_diagnostic("Analyzed module structure".to_string());
                result.add_diagnostic("Found optimization opportunities".to_string());
                
                // Actually modify an operation
                if let Some(op) = module.operations_mut().first_mut() {
                    op.add_attribute("analyzed".to_string(), MLIRAttribute::Boolean(true));
                }
                
                Ok(result)
            }
        }
        
        let mut pipeline = OptimizationPipeline::new();
        pipeline.add_pass(MetricsPass);
        
        let result = pipeline.run(&mut module, &context);
        assert!(result.is_ok());
        
        let pipeline_result = result.unwrap();
        let pass_result = pipeline_result.pass_results.get("metrics_pass").unwrap();
        
        // Check metrics
        assert_eq!(pass_result.metrics.get("operations_analyzed"), Some(&2.0));
        assert_eq!(pass_result.metrics.get("complexity_score"), Some(&42.5));
        
        // Check diagnostics
        assert_eq!(pass_result.diagnostics.len(), 2);
        assert!(pass_result.diagnostics.contains(&"Analyzed module structure".to_string()));
        assert!(pass_result.diagnostics.contains(&"Found optimization opportunities".to_string()));
    }
}