// Test utilities for GPU kernel generation
// Provides compatibility layer for existing tests

use crate::compiler::mlir::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRAttribute};
use crate::compiler::mlir::gpu_dialect::*;
use crate::compiler::mlir::{MLIRError, MLIRPipeline};
use crate::compiler::ast::AST;
use std::collections::HashMap;

/// Mock MLIR context for backward compatibility with tests
pub struct MockMLIRContext {
    inner: MLIRContext,
}

impl MockMLIRContext {
    pub fn new() -> Self {
        MockMLIRContext {
            inner: MLIRContext::new().expect("Failed to create MLIR context"),
        }
    }

    pub fn is_registered_operation(&self, op_name: &str) -> bool {
        let dialect = op_name.split('.').next().unwrap_or("");
        self.inner.is_dialect_registered(dialect)
    }
}

/// Mock MLIR module for backward compatibility with tests
pub struct MockMLIRModule {
    inner: MLIRModule,
    pub operations: Vec<String>, // String representation for backward compatibility
}

impl MockMLIRModule {
    pub fn new() -> Self {
        let context = MLIRContext::new().expect("Failed to create MLIR context");
        let inner = context.create_module("test_module").expect("Failed to create module");
        MockMLIRModule {
            inner,
            operations: Vec::new(),
        }
    }

    pub fn add_operation(&mut self, op: String) {
        self.operations.push(op.clone());
        
        // Try to convert string operation to proper MLIR operation
        if let Ok(mlir_op) = self.parse_string_operation(&op) {
            let _ = self.inner.add_operation(mlir_op);
        }
    }

    pub fn add_attribute(&mut self, key: String, value: String) {
        self.inner.add_attribute(key, value);
    }

    /// Parse a string operation into an MLIR operation (simplified)
    fn parse_string_operation(&self, op_str: &str) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new(self.extract_op_name(op_str));
        
        // Add some basic attributes based on the operation string
        if op_str.contains("kernel") {
            op.add_attribute("kernel".to_string(), MLIRAttribute::Boolean(true));
        }
        if op_str.contains("elementwise") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("elementwise".to_string()));
        }
        if op_str.contains("matmul") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("matmul".to_string()));
        }
        if op_str.contains("reduction") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("reduction".to_string()));
        }
        
        Ok(op)
    }

    fn extract_op_name(&self, op_str: &str) -> String {
        // Extract operation name from string (simplified parsing)
        if let Some(space_pos) = op_str.find(' ') {
            op_str[..space_pos].to_string()
        } else {
            op_str.to_string()
        }
    }
}

impl Clone for MockMLIRModule {
    fn clone(&self) -> Self {
        let context = MLIRContext::new().expect("Failed to create MLIR context");
        let inner = context.create_module("test_module_clone").expect("Failed to create module");
        MockMLIRModule {
            inner,
            operations: self.operations.clone(),
        }
    }
}

/// Mock operation for backward compatibility
#[derive(Debug, Clone)]
pub struct MockOperation {
    pub name: String,
    pub attributes: HashMap<String, String>,
    pub operands: Vec<String>,
    pub results: Vec<String>,
}

impl MockOperation {
    pub fn new(name: &str) -> Self {
        MockOperation {
            name: name.to_string(),
            attributes: HashMap::new(),
            operands: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn add_attribute(&mut self, key: &str, value: &str) {
        self.attributes.insert(key.to_string(), value.to_string());
    }

    pub fn add_operand(&mut self, operand: &str) {
        self.operands.push(operand.to_string());
    }

    pub fn add_result(&mut self, result: &str) {
        self.results.push(result.to_string());
    }
}

/// Test pipeline that creates a mock MLIR pipeline for testing
pub struct TestMLIRPipeline {
    pipeline: MLIRPipeline,
}

impl TestMLIRPipeline {
    pub fn new() -> Self {
        TestMLIRPipeline {
            pipeline: MLIRPipeline::new().expect("Failed to create MLIR pipeline"),
        }
    }

    pub fn compile_ast(&mut self, ast: &AST) -> Result<MockMLIRModule, MLIRError> {
        let mlir_module = self.pipeline.compile_ast(ast)?;
        
        // Convert to mock module for test compatibility
        let mut mock_module = MockMLIRModule::new();
        
        // Add string representations of operations for backward compatibility
        for op in mlir_module.operations() {
            let op_str = format!("{}", op.name);
            mock_module.add_operation(op_str);
        }
        
        Ok(mock_module)
    }

    pub fn lower_to_gpu_kernels(&mut self, module: &mut MockMLIRModule, target: GpuTarget) -> Result<(), MLIRError> {
        self.pipeline.lower_to_gpu_kernels(&mut module.inner, target)
    }

    pub fn optimize_gpu_tensors(&self, module: &mut MockMLIRModule, target: GpuTarget) -> Result<(), MLIRError> {
        self.pipeline.optimize_gpu_tensors(&mut module.inner, target)
    }

    pub fn generate_gpu_code(&self, module: &MockMLIRModule, target: GpuTarget) -> Result<String, MLIRError> {
        self.pipeline.generate_gpu_code(&module.inner, target)
    }
}