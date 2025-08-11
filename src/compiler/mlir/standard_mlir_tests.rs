// Tests for standard MLIR optimization passes
// Validates the new standard MLIR passes integration

use super::optimization::*;
use crate::compiler::mlir::{MLIRError};
use crate::compiler::mlir::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    fn create_test_module_with_loops(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("loop_test_module").unwrap();
        
        // Add loop operations
        let mut loop_op = MLIROperation::new("scf.for".to_string());
        loop_op.add_attribute("lower_bound".to_string(), MLIRAttribute::Integer(0));
        loop_op.add_attribute("upper_bound".to_string(), MLIRAttribute::Integer(4));
        loop_op.add_operand(MLIRValue::new("tensor1".to_string(), 
            MLIRType::Tensor { 
                element_type: Box::new(MLIRType::Float { width: 32 }), 
                shape: vec![100, 100] 
            }));
        module.add_operation(loop_op).unwrap();
        
        module
    }

    fn create_test_module_with_vectorizable_ops(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("vectorization_test_module").unwrap();
        
        // Add vectorizable operations
        let mut add_op = MLIROperation::new("arith.addf".to_string());
        add_op.add_operand(MLIRValue::new("a".to_string(), MLIRType::Float { width: 32 }));
        add_op.add_operand(MLIRValue::new("b".to_string(), MLIRType::Float { width: 32 }));
        module.add_operation(add_op).unwrap();
        
        module
    }

    #[test]
    fn test_loop_optimization_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_loops(&context);
        let pass = LoopOptimizationPass::new();
        
        assert_eq!(pass.name(), "loop-optimization");
        assert!(pass.description().contains("loop optimizations"));
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("loops_unrolled"));
        assert!(result.metrics.contains_key("loops_tiled"));
        
        // Check that loop optimizations were applied
        assert!(module.attributes().contains_key("loop_optimizations_applied"));
    }

    #[test]
    fn test_vectorization_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_vectorizable_ops(&context);
        let pass = VectorizationPass::new();
        
        assert_eq!(pass.name(), "vectorization");
        assert!(pass.description().contains("vector operations"));
        assert_eq!(pass.dependencies(), vec!["loop-optimization"]);
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("operations_vectorized"));
        
        // Check that vectorization was applied
        assert!(module.attributes().contains_key("vectorization_applied"));
    }

    #[test]
    fn test_memory_layout_optimization_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_memory_ops(&context);
        let pass = MemoryLayoutOptimizationPass::new();
        
        assert_eq!(pass.name(), "memory-layout-optimization");
        assert!(pass.description().contains("memory layouts"));
        assert_eq!(pass.dependencies(), vec!["vectorization"]);
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("layouts_optimized"));
        
        // Check that memory layout optimizations were applied
        assert!(module.attributes().contains_key("memory_layout_optimizations"));
    }

    #[test]
    fn test_function_inlining_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_function_calls(&context);
        let pass = FunctionInliningPass::new();
        
        assert_eq!(pass.name(), "function-inlining");
        assert!(pass.description().contains("Inlines small functions"));
        assert_eq!(pass.dependencies(), vec!["memory-layout-optimization"]);
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("functions_inlined"));
        
        // Check that function inlining was applied
        assert!(module.attributes().contains_key("function_inlining_applied"));
    }

    #[test]
    fn test_function_specialization_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_specializable_functions(&context);
        let pass = FunctionSpecializationPass::new();
        
        assert_eq!(pass.name(), "function-specialization");
        assert!(pass.description().contains("specialized versions"));
        assert_eq!(pass.dependencies(), vec!["function-inlining"]);
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("functions_specialized"));
        
        // Check that function specialization was applied
        assert!(module.attributes().contains_key("function_specialization_applied"));
    }

    #[test]
    fn test_architecture_specific_optimization_pass() {
        let context = create_test_context();
        let mut module = create_test_module_with_arch_ops(&context);
        let pass = ArchitectureSpecificOptimizationPass::new();
        
        assert_eq!(pass.name(), "architecture-specific-optimization");
        assert!(pass.description().contains("architecture-specific"));
        assert_eq!(pass.dependencies(), vec!["function-specialization"]);
        
        let result = pass.run(&mut module, &context).unwrap();
        assert!(result.changed);
        assert!(result.metrics.contains_key("cpu_optimizations"));
        
        // Check that architecture optimizations were applied
        assert!(module.attributes().contains_key("architecture_optimizations"));
    }

    #[test]
    fn test_standard_mlir_pipeline() {
        let context = create_test_context();
        let mut module = create_comprehensive_test_module(&context);
        let mut pipeline = OptimizationPipeline::create_standard_mlir_pipeline();
        
        let result = pipeline.run(&mut module, &context).unwrap();
        
        // Should run all standard passes successfully
        assert!(result.total_changed);
        assert!(result.passes_run >= 6); // Should run all 6 standard passes
        assert!(result.execution_time_ms > 0.0);
        
        // Check that standard optimizations were applied
        assert!(module.attributes().contains_key("loop_optimizations_applied"));
        assert!(module.attributes().contains_key("vectorization_applied"));
    }

    #[test]
    fn test_comprehensive_pipeline() {
        let context = create_test_context();
        let mut module = create_comprehensive_test_module(&context);
        let mut pipeline = OptimizationPipeline::create_comprehensive_pipeline();
        
        let result = pipeline.run(&mut module, &context).unwrap();
        
        // Should run both Aether and standard passes successfully
        assert!(result.total_changed);
        assert!(result.passes_run > 10); // Should run many passes
        assert!(result.execution_time_ms > 0.0);
    }

    // Helper functions for creating test modules

    fn create_test_module_with_memory_ops(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("memory_layout_test_module").unwrap();
        
        // Add memory operations
        let mut load_op = MLIROperation::new("memref.load".to_string());
        load_op.add_operand(MLIRValue::new("memref1".to_string(), 
            MLIRType::Memref { 
                element_type: Box::new(MLIRType::Float { width: 32 }), 
                shape: vec![1024, 1024] 
            }));
        module.add_operation(load_op).unwrap();
        
        module
    }

    fn create_test_module_with_function_calls(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("function_call_test_module").unwrap();
        
        // Add function call operations
        let mut call_op = MLIROperation::new("func.call".to_string());
        call_op.add_attribute("callee".to_string(), MLIRAttribute::String("small_function".to_string()));
        call_op.add_attribute("function_size".to_string(), MLIRAttribute::Integer(50));
        module.add_operation(call_op).unwrap();
        
        module
    }

    fn create_test_module_with_specializable_functions(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("function_specialization_test_module").unwrap();
        
        // Add function definitions that can be specialized
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("generic_function".to_string()));
        func_op.add_attribute("specializable".to_string(), MLIRAttribute::Boolean(true));
        module.add_operation(func_op).unwrap();
        
        module
    }

    fn create_test_module_with_arch_ops(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("architecture_test_module").unwrap();
        
        // Add operations that can benefit from architecture-specific optimizations
        let mut vectorizable_op = MLIROperation::new("arith.mulf".to_string());
        vectorizable_op.add_operand(MLIRValue::new("a".to_string(), MLIRType::Float { width: 32 }));
        module.add_operation(vectorizable_op).unwrap();
        
        module
    }

    fn create_comprehensive_test_module(context: &MLIRContext) -> MLIRModule {
        let mut module = context.create_module("comprehensive_test_module").unwrap();
        
        // Add operations that will trigger multiple optimization passes
        
        // Loop operations
        let mut loop_op = MLIROperation::new("scf.for".to_string());
        loop_op.add_attribute("upper_bound".to_string(), MLIRAttribute::Integer(3));
        module.add_operation(loop_op).unwrap();
        
        // Vectorizable operations
        let mut arith_op = MLIROperation::new("arith.addf".to_string());
        arith_op.add_operand(MLIRValue::new("a".to_string(), MLIRType::Float { width: 32 }));
        module.add_operation(arith_op).unwrap();
        
        // Memory operations
        let mut load_op = MLIROperation::new("memref.load".to_string());
        load_op.add_operand(MLIRValue::new("memref1".to_string(), 
            MLIRType::Memref { 
                element_type: Box::new(MLIRType::Float { width: 32 }), 
                shape: vec![64, 64] 
            }));
        module.add_operation(load_op).unwrap();
        
        // Function calls
        let mut call_op = MLIROperation::new("func.call".to_string());
        call_op.add_attribute("function_size".to_string(), MLIRAttribute::Integer(30));
        module.add_operation(call_op).unwrap();
        
        module
    }
}