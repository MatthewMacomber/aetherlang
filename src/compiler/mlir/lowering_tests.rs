// Tests for dialect lowering infrastructure
// Validates the DialectLowering trait and framework functionality

use super::lowering::{
    DialectLowering, LoweringError, LoweringConfig, TypeConverter, LoweringPass, LoweringStatistics
};
use super::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use std::collections::HashMap;

// ===== MOCK LOWERING IMPLEMENTATIONS FOR TESTING =====

/// Mock lowering implementation for testing the framework
struct MockAetherToStandardLowering {
    source_dialect: String,
    target_dialect: String,
    should_fail: bool,
}

impl MockAetherToStandardLowering {
    fn new() -> Self {
        MockAetherToStandardLowering {
            source_dialect: "aether".to_string(),
            target_dialect: "linalg".to_string(),
            should_fail: false,
        }
    }
    
    fn new_with_failure() -> Self {
        MockAetherToStandardLowering {
            source_dialect: "aether".to_string(),
            target_dialect: "linalg".to_string(),
            should_fail: true,
        }
    }
}

impl DialectLowering for MockAetherToStandardLowering {
    fn lower_operation(&self, op: &MLIROperation) -> Result<Vec<MLIROperation>, LoweringError> {
        if self.should_fail {
            return Err(LoweringError::UnsupportedOperation {
                operation: op.name.clone(),
                source_dialect: self.source_dialect.clone(),
                target_dialect: self.target_dialect.clone(),
            });
        }
        
        match op.name.as_str() {
            "aether.tensor_create" => {
                let mut new_op = MLIROperation::new("linalg.init_tensor".to_string());
                
                // Copy attributes
                for (key, value) in &op.attributes {
                    new_op.add_attribute(key.clone(), value.clone());
                }
                
                // Copy results with converted types
                for result in &op.results {
                    let converted_type = match &result.value_type {
                        MLIRType::AetherTensor { element_type, shape, .. } => {
                            MLIRType::Tensor {
                                element_type: element_type.clone(),
                                shape: shape.clone(),
                            }
                        }
                        other => other.clone(),
                    };
                    new_op.add_result(MLIRValue::new(result.id.clone(), converted_type));
                }
                
                Ok(vec![new_op])
            }
            "aether.tensor_op" => {
                let mut new_op = MLIROperation::new("linalg.generic".to_string());
                
                // Copy operands and results
                for operand in &op.operands {
                    new_op.add_operand(operand.clone());
                }
                for result in &op.results {
                    new_op.add_result(result.clone());
                }
                
                Ok(vec![new_op])
            }
            "aether.matmul" => {
                let mut new_op = MLIROperation::new("linalg.matmul".to_string());
                
                // Copy operands and results
                for operand in &op.operands {
                    new_op.add_operand(operand.clone());
                }
                for result in &op.results {
                    new_op.add_result(result.clone());
                }
                
                Ok(vec![new_op])
            }
            "aether.linear_alloc" => {
                let mut alloc_op = MLIROperation::new("memref.alloc".to_string());
                let mut view_op = MLIROperation::new("memref.cast".to_string());
                
                // Copy attributes and results
                for (key, value) in &op.attributes {
                    alloc_op.add_attribute(key.clone(), value.clone());
                }
                
                for result in &op.results {
                    let converted_type = match &result.value_type {
                        MLIRType::AetherLinear { inner_type } => {
                            match inner_type.as_ref() {
                                MLIRType::AetherTensor { element_type, shape, .. } => {
                                    MLIRType::Memref {
                                        element_type: element_type.clone(),
                                        shape: shape.clone(),
                                    }
                                }
                                other => MLIRType::Memref {
                                    element_type: Box::new(other.clone()),
                                    shape: vec![1],
                                }
                            }
                        }
                        other => other.clone(),
                    };
                    alloc_op.add_result(MLIRValue::new(format!("{}_alloc", result.id), converted_type.clone()));
                    view_op.add_result(MLIRValue::new(result.id.clone(), converted_type));
                }
                
                Ok(vec![alloc_op, view_op])
            }
            _ => Err(LoweringError::UnsupportedOperation {
                operation: op.name.clone(),
                source_dialect: self.source_dialect.clone(),
                target_dialect: self.target_dialect.clone(),
            })
        }
    }
    
    fn get_source_dialect(&self) -> &str {
        &self.source_dialect
    }
    
    fn get_target_dialect(&self) -> &str {
        &self.target_dialect
    }
}

// ===== AETHER TO STANDARD LOWERING TESTS =====

#[cfg(test)]
mod aether_to_standard_tests {
    use super::*;
    use super::super::lowering::AetherToStandardLowering;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_aether_to_standard_lowering_creation() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        assert_eq!(lowering.get_source_dialect(), "aether");
        assert_eq!(lowering.get_target_dialect(), "linalg");
    }

    #[test]
    fn test_tensor_create_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(3),
            MLIRAttribute::Integer(4),
        ]));
        op.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "linalg.init_tensor");
        assert!(result[0].attributes.contains_key("static_sizes"));
        assert_eq!(result[0].results.len(), 1);
        
        match &result[0].results[0].value_type {
            MLIRType::Tensor { element_type, shape } => {
                assert!(matches!(element_type.as_ref(), MLIRType::Float { width: 32 }));
                assert_eq!(*shape, vec![3, 4]);
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_tensor_create_with_initial_value() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(2),
        ]));
        op.add_attribute("initial_value".to_string(), MLIRAttribute::Float(1.0));
        op.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 2); // init_tensor + fill
        assert_eq!(result[0].name, "linalg.init_tensor");
        assert_eq!(result[1].name, "linalg.fill");
        assert!(result[1].attributes.contains_key("value"));
    }

    #[test]
    fn test_tensor_op_elementwise() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.tensor_op".to_string());
        op.add_attribute("op_name".to_string(), MLIRAttribute::String("add".to_string()));
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "linalg.generic");
        assert!(result[0].attributes.contains_key("indexing_maps"));
        assert!(result[0].attributes.contains_key("iterator_types"));
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_tensor_op_reduction() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.tensor_op".to_string());
        op.add_attribute("op_name".to_string(), MLIRAttribute::String("reduce_sum".to_string()));
        op.add_operand(MLIRValue::new("input".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 5],
            device: "cpu".to_string(),
        }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "linalg.generic");
        
        // Check that iterator types include reduction
        if let Some(MLIRAttribute::Array(iterator_types)) = result[0].attributes.get("iterator_types") {
            let has_reduction = iterator_types.iter().any(|attr| {
                if let MLIRAttribute::String(s) = attr {
                    s == "reduction"
                } else {
                    false
                }
            });
            assert!(has_reduction);
        } else {
            panic!("Expected iterator_types attribute");
        }
    }

    #[test]
    fn test_matmul_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.matmul".to_string());
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 5],
            device: "cpu".to_string(),
        }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 5],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "linalg.matmul");
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_matmul_with_transpose() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.matmul".to_string());
        op.add_attribute("transpose_a".to_string(), MLIRAttribute::Boolean(true));
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 3], // Will be transposed to [3, 4]
            device: "cpu".to_string(),
        }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 5],
            device: "cpu".to_string(),
        }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 5],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 2); // transpose + matmul
        assert_eq!(result[0].name, "linalg.transpose");
        assert_eq!(result[1].name, "linalg.matmul");
    }

    #[test]
    fn test_autodiff_forward_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.autodiff_forward".to_string());
        op.add_operand(MLIRValue::new("function".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Float { width: 32 }],
        }));
        op.add_operand(MLIRValue::new("input".to_string(), MLIRType::Float { width: 32 }));
        op.add_operand(MLIRValue::new("tangent".to_string(), MLIRType::Float { width: 32 }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Float { width: 32 }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "func.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("__aether_forward_ad".to_string())));
        assert_eq!(result[0].results.len(), 2); // value + derivative
    }

    #[test]
    fn test_autodiff_reverse_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.autodiff_reverse".to_string());
        op.add_operand(MLIRValue::new("function".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Float { width: 32 }],
        }));
        op.add_operand(MLIRValue::new("input".to_string(), MLIRType::Float { width: 32 }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Float { width: 32 }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 2); // forward + backward
        assert_eq!(result[0].name, "func.call");
        assert_eq!(result[1].name, "func.call");
        assert_eq!(result[0].attributes.get("callee"),
            Some(&MLIRAttribute::String("__aether_reverse_forward".to_string())));
        assert_eq!(result[1].attributes.get("callee"),
            Some(&MLIRAttribute::String("__aether_reverse_backward".to_string())));
    }

    #[test]
    fn test_prob_var_normal_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.prob_var".to_string());
        op.add_attribute("name".to_string(), MLIRAttribute::String("x".to_string()));
        
        let mut dist_dict = HashMap::new();
        dist_dict.insert("type".to_string(), MLIRAttribute::String("normal".to_string()));
        dist_dict.insert("mean".to_string(), MLIRAttribute::Float(0.0));
        dist_dict.insert("std".to_string(), MLIRAttribute::Float(1.0));
        op.add_attribute("distribution".to_string(), MLIRAttribute::Dictionary(dist_dict));
        
        op.add_operand(MLIRValue::new("mean".to_string(), MLIRType::Float { width: 64 }));
        op.add_operand(MLIRValue::new("std".to_string(), MLIRType::Float { width: 64 }));
        op.add_result(MLIRValue::new("dist".to_string(), MLIRType::AetherProbabilistic {
            distribution: "normal".to_string(),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "func.call");
        assert_eq!(result[0].attributes.get("callee"),
            Some(&MLIRAttribute::String("__aether_normal_dist".to_string())));
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_linear_alloc_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.linear_alloc".to_string());
        op.add_attribute("allocation_site".to_string(), MLIRAttribute::String("test_site".to_string()));
        op.add_result(MLIRValue::new("ptr".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![10, 20],
                device: "cpu".to_string(),
            }),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "memref.alloc");
        assert_eq!(result[0].results.len(), 1);
        
        match &result[0].results[0].value_type {
            MLIRType::Memref { element_type, shape } => {
                assert!(matches!(element_type.as_ref(), MLIRType::Float { width: 32 }));
                assert_eq!(*shape, vec![10, 20]);
            }
            _ => panic!("Expected memref type"),
        }
    }

    #[test]
    fn test_linear_drop_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.linear_drop".to_string());
        op.add_operand(MLIRValue::new("ptr".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::Float { width: 32 }),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "memref.dealloc");
        assert_eq!(result[0].operands.len(), 1);
        assert_eq!(result[0].results.len(), 0); // No results for dealloc
    }

    #[test]
    fn test_unsupported_operation() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let op = MLIROperation::new("aether.unsupported_op".to_string());
        
        let result = lowering.lower_operation(&op);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoweringError::UnsupportedOperation { operation, source_dialect, target_dialect } => {
                assert_eq!(operation, "aether.unsupported_op");
                assert_eq!(source_dialect, "aether");
                assert_eq!(target_dialect, "standard");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_comprehensive_lowering_pass() {
        let context = create_test_context();
        let mut module = context.create_module("test_module").unwrap();
        
        // Add various Aether operations
        let mut tensor_create = MLIROperation::new("aether.tensor_create".to_string());
        tensor_create.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(2),
        ]));
        tensor_create.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
            device: "cpu".to_string(),
        }));
        module.add_operation(tensor_create).unwrap();
        
        let mut matmul = MLIROperation::new("aether.matmul".to_string());
        matmul.add_operand(MLIRValue::new("a".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        }));
        matmul.add_operand(MLIRValue::new("b".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 2],
            device: "cpu".to_string(),
        }));
        matmul.add_result(MLIRValue::new("c".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
            device: "cpu".to_string(),
        }));
        module.add_operation(matmul).unwrap();
        
        let mut linear_alloc = MLIROperation::new("aether.linear_alloc".to_string());
        linear_alloc.add_result(MLIRValue::new("ptr".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::Float { width: 32 }),
        }));
        module.add_operation(linear_alloc).unwrap();
        
        // Add a non-Aether operation that should remain unchanged
        let func_op = MLIROperation::new("func.func".to_string());
        module.add_operation(func_op).unwrap();
        
        let lowering = AetherToStandardLowering::new(&context);
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        if let Err(ref e) = result {
            println!("Lowering pass failed: {:?}", e);
        }
        assert!(result.is_ok());
        
        let stats = pass.get_statistics();
        assert_eq!(stats.total_operations, 3); // Only Aether operations
        assert_eq!(stats.successful_lowerings, 3);
        assert_eq!(stats.failed_lowerings, 0);
        
        // Check that operations were converted
        let operations = module.operations();
        let aether_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("aether."))
            .collect();
        let standard_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("linalg.") || op.name.starts_with("memref."))
            .collect();
        let func_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("func."))
            .collect();
        
        assert_eq!(aether_ops.len(), 0); // All Aether ops should be lowered
        assert!(standard_ops.len() >= 3); // Should have at least 3 standard ops
        assert_eq!(func_ops.len(), 1); // func op should remain unchanged
    }
}

// ===== STANDARD TO LLVM LOWERING TESTS =====

#[cfg(test)]
mod standard_to_llvm_tests {
    use super::*;
    use super::super::lowering::StandardToLLVMLowering;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_standard_to_llvm_lowering_creation() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        assert_eq!(lowering.get_source_dialect(), "standard");
        assert_eq!(lowering.get_target_dialect(), "llvm");
    }

    #[test]
    fn test_func_func_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("func.func".to_string());
        op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test_func".to_string()));
        op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("(i32, f32) -> i32".to_string()));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.func");
        assert_eq!(result[0].attributes.get("sym_name"), 
            Some(&MLIRAttribute::String("test_func".to_string())));
        assert!(result[0].attributes.contains_key("function_type"));
    }

    #[test]
    fn test_func_call_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("func.call".to_string());
        op.add_attribute("callee".to_string(), MLIRAttribute::String("test_func".to_string()));
        op.add_operand(MLIRValue::new("arg1".to_string(), MLIRType::Integer { width: 32, signed: true }));
        op.add_operand(MLIRValue::new("arg2".to_string(), MLIRType::Float { width: 32 }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Integer { width: 32, signed: true }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("test_func".to_string())));
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_func_return_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("func.return".to_string());
        op.add_operand(MLIRValue::new("return_val".to_string(), 
            MLIRType::Integer { width: 32, signed: true }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.return");
        assert_eq!(result[0].operands.len(), 1);
        assert_eq!(result[0].results.len(), 0);
    }

    #[test]
    fn test_memref_alloc_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("memref.alloc".to_string());
        op.add_result(MLIRValue::new("memref".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("malloc".to_string())));
        assert_eq!(result[0].results.len(), 1);
        assert!(matches!(result[0].results[0].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_memref_dealloc_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("memref.dealloc".to_string());
        op.add_operand(MLIRValue::new("memref".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("free".to_string())));
        assert_eq!(result[0].operands.len(), 1);
        assert!(matches!(result[0].operands[0].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_memref_load_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("memref.load".to_string());
        op.add_operand(MLIRValue::new("memref".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        op.add_operand(MLIRValue::new("index0".to_string(), MLIRType::Index));
        op.add_operand(MLIRValue::new("index1".to_string(), MLIRType::Index));
        op.add_result(MLIRValue::new("value".to_string(), MLIRType::Float { width: 32 }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.load");
        assert_eq!(result[0].operands.len(), 3); // memref + 2 indices
        assert_eq!(result[0].results.len(), 1);
        assert!(matches!(result[0].operands[0].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_memref_store_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("memref.store".to_string());
        op.add_operand(MLIRValue::new("value".to_string(), MLIRType::Float { width: 32 }));
        op.add_operand(MLIRValue::new("memref".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        op.add_operand(MLIRValue::new("index0".to_string(), MLIRType::Index));
        op.add_operand(MLIRValue::new("index1".to_string(), MLIRType::Index));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.store");
        assert_eq!(result[0].operands.len(), 4); // value + memref + 2 indices
        assert!(matches!(result[0].operands[0].value_type, MLIRType::Float { width: 32 }));
        assert!(matches!(result[0].operands[1].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_arith_addi_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("arith.addi".to_string());
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::Integer { width: 32, signed: true }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::Integer { width: 32, signed: true }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Integer { width: 32, signed: true }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.add");
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_arith_addf_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("arith.addf".to_string());
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::Float { width: 32 }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::Float { width: 32 }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Float { width: 32 }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.fadd");
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_arith_cmpi_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("arith.cmpi".to_string());
        op.add_attribute("predicate".to_string(), MLIRAttribute::String("eq".to_string()));
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::Integer { width: 32, signed: true }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::Integer { width: 32, signed: true }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Integer { width: 1, signed: false }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.icmp");
        assert_eq!(result[0].attributes.get("predicate"), 
            Some(&MLIRAttribute::String("eq".to_string())));
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
        assert!(matches!(result[0].results[0].value_type, MLIRType::Integer { width: 1, signed: false }));
    }

    #[test]
    fn test_arith_constant_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("arith.constant".to_string());
        op.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
        op.add_result(MLIRValue::new("const".to_string(), MLIRType::Integer { width: 32, signed: true }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.mlir.constant");
        assert_eq!(result[0].attributes.get("value"), 
            Some(&MLIRAttribute::Integer(42)));
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_tensor_extract_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("tensor.extract".to_string());
        op.add_operand(MLIRValue::new("tensor".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        op.add_operand(MLIRValue::new("index0".to_string(), MLIRType::Index));
        op.add_operand(MLIRValue::new("index1".to_string(), MLIRType::Index));
        op.add_result(MLIRValue::new("element".to_string(), MLIRType::Float { width: 32 }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.load");
        assert_eq!(result[0].operands.len(), 3); // tensor + 2 indices
        assert!(matches!(result[0].operands[0].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_scf_if_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("scf.if".to_string());
        op.add_operand(MLIRValue::new("condition".to_string(), 
            MLIRType::Integer { width: 1, signed: false }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.cond_br");
        assert_eq!(result[0].operands.len(), 1);
        assert!(result[0].attributes.contains_key("true_dest"));
        assert!(result[0].attributes.contains_key("false_dest"));
    }

    #[test]
    fn test_scf_for_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("scf.for".to_string());
        op.add_operand(MLIRValue::new("lower".to_string(), MLIRType::Index));
        op.add_operand(MLIRValue::new("upper".to_string(), MLIRType::Index));
        op.add_operand(MLIRValue::new("step".to_string(), MLIRType::Index));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert!(result.len() >= 3); // Should generate multiple operations for loop structure
        assert_eq!(result[0].name, "llvm.alloca"); // Loop variable allocation
        assert_eq!(result[1].name, "llvm.store"); // Store initial value
        assert_eq!(result[2].name, "llvm.br"); // Branch to header
    }

    #[test]
    fn test_linalg_generic_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("linalg.generic".to_string());
        op.add_attribute("indexing_maps".to_string(), 
            MLIRAttribute::String("affine_map<(d0, d1) -> (d0, d1)>".to_string()));
        op.add_attribute("iterator_types".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::String("parallel".to_string()),
            MLIRAttribute::String("parallel".to_string()),
        ]));
        op.add_operand(MLIRValue::new("input".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        op.add_result(MLIRValue::new("output".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("__aether_linalg_generic".to_string())));
        assert!(result[0].attributes.contains_key("indexing_maps"));
        assert!(result[0].attributes.contains_key("iterator_types"));
    }

    #[test]
    fn test_function_signature_conversion() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("func.func".to_string());
        op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test_func".to_string()));
        op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("(i32, f32) -> i32".to_string()));
        op.add_attribute("sym_visibility".to_string(), 
            MLIRAttribute::String("public".to_string()));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.func");
        assert_eq!(result[0].attributes.get("sym_name"), 
            Some(&MLIRAttribute::String("test_func".to_string())));
        assert_eq!(result[0].attributes.get("CConv"), 
            Some(&MLIRAttribute::String("ccc".to_string())));
        assert_eq!(result[0].attributes.get("linkage"), 
            Some(&MLIRAttribute::String("public".to_string())));
        assert!(result[0].attributes.contains_key("function_type"));
    }

    #[test]
    fn test_comprehensive_standard_to_llvm_lowering() {
        let context = create_test_context();
        let mut module = context.create_module("test_module").unwrap();
        
        // Add various standard dialect operations
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("main".to_string()));
        func_op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("() -> i32".to_string()));
        module.add_operation(func_op).unwrap();
        
        let mut arith_op = MLIROperation::new("arith.addi".to_string());
        arith_op.add_operand(MLIRValue::new("a".to_string(), 
            MLIRType::Integer { width: 32, signed: true }));
        arith_op.add_operand(MLIRValue::new("b".to_string(), 
            MLIRType::Integer { width: 32, signed: true }));
        arith_op.add_result(MLIRValue::new("c".to_string(), 
            MLIRType::Integer { width: 32, signed: true }));
        module.add_operation(arith_op).unwrap();
        
        let mut memref_alloc = MLIROperation::new("memref.alloc".to_string());
        memref_alloc.add_result(MLIRValue::new("buffer".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![100],
        }));
        module.add_operation(memref_alloc).unwrap();
        
        let mut return_op = MLIROperation::new("func.return".to_string());
        return_op.add_operand(MLIRValue::new("c".to_string(), 
            MLIRType::Integer { width: 32, signed: true }));
        module.add_operation(return_op).unwrap();
        
        // Add a builtin operation that should remain unchanged
        let builtin_op = MLIROperation::new("builtin.module".to_string());
        module.add_operation(builtin_op).unwrap();
        
        let lowering = StandardToLLVMLowering::new(&context);
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        if let Err(ref e) = result {
            println!("Standard to LLVM lowering pass failed: {:?}", e);
        }
        assert!(result.is_ok());
        
        let stats = pass.get_statistics();
        assert_eq!(stats.total_operations, 4); // Only standard dialect operations
        assert_eq!(stats.successful_lowerings, 4);
        assert_eq!(stats.failed_lowerings, 0);
        
        // Check that operations were converted to LLVM dialect
        let operations = module.operations();
        let llvm_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("llvm."))
            .collect();
        let standard_ops: Vec<_> = operations.iter()
            .filter(|op| {
                let dialect = op.name.split('.').next().unwrap_or("");
                matches!(dialect, "func" | "arith" | "memref" | "tensor" | "linalg")
            })
            .collect();
        let builtin_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("builtin."))
            .collect();
        
        assert!(llvm_ops.len() >= 4); // Should have at least 4 LLVM ops
        assert_eq!(standard_ops.len(), 0); // All standard ops should be lowered
        assert_eq!(builtin_ops.len(), 1); // builtin op should remain unchanged
    }

    #[test]
    fn test_type_conversion_to_llvm() {
        let context = create_test_context();
        let mut lowering = StandardToLLVMLowering::new(&context);
        
        // Test integer type conversion
        let int_type = MLIRType::Integer { width: 32, signed: true };
        let converted = lowering.convert_type_to_llvm(&int_type).unwrap();
        assert!(matches!(converted, MLIRType::Integer { width: 32, signed: true }));
        
        // Test float type conversion
        let float_type = MLIRType::Float { width: 64 };
        let converted = lowering.convert_type_to_llvm(&float_type).unwrap();
        assert!(matches!(converted, MLIRType::Float { width: 64 }));
        
        // Test index type conversion
        let index_type = MLIRType::Index;
        let converted = lowering.convert_type_to_llvm(&index_type).unwrap();
        assert!(matches!(converted, MLIRType::Integer { width: 64, signed: true }));
        
        // Test memref type conversion
        let memref_type = MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        };
        let converted = lowering.convert_type_to_llvm(&memref_type).unwrap();
        assert!(matches!(converted, MLIRType::Pointer));
        
        // Test tensor type conversion
        let tensor_type = MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        };
        let converted = lowering.convert_type_to_llvm(&tensor_type).unwrap();
        assert!(matches!(converted, MLIRType::Pointer));
        
        // Test function type conversion
        let func_type = MLIRType::Function {
            inputs: vec![MLIRType::Integer { width: 32, signed: true }],
            outputs: vec![MLIRType::Float { width: 32 }],
        };
        let converted = lowering.convert_type_to_llvm(&func_type).unwrap();
        if let MLIRType::Function { inputs, outputs } = converted {
            assert_eq!(inputs.len(), 1);
            assert_eq!(outputs.len(), 1);
            assert!(matches!(inputs[0], MLIRType::Integer { width: 32, signed: true }));
            assert!(matches!(outputs[0], MLIRType::Float { width: 32 }));
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_aether_type_conversion_error() {
        let context = create_test_context();
        let mut lowering = StandardToLLVMLowering::new(&context);
        
        // Test that Aether types cannot be converted directly
        let aether_tensor = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
            device: "cpu".to_string(),
        };
        
        let result = lowering.convert_type_to_llvm(&aether_tensor);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            LoweringError::TypeConversionError { reason, .. } => {
                assert!(reason.contains("Aether types should be lowered to standard types first"));
            }
            _ => panic!("Expected TypeConversionError"),
        }
    }

    #[test]
    fn test_unsupported_standard_operation() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let op = MLIROperation::new("unknown.operation".to_string());
        
        let result = lowering.lower_operation(&op);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoweringError::UnsupportedOperation { operation, source_dialect, target_dialect } => {
                assert_eq!(operation, "unknown.operation");
                assert_eq!(source_dialect, "standard");
                assert_eq!(target_dialect, "llvm");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_llvm_dialect_verification() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        // Test that all generated operations belong to LLVM dialect
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test".to_string()));
        func_op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("() -> ()".to_string()));
        
        let result = lowering.lower_operation(&func_op).unwrap();
        
        for op in &result {
            assert!(op.name.starts_with("llvm."), 
                "Operation '{}' should be in LLVM dialect", op.name);
        }
    }

    #[test]
    fn test_calling_convention_handling() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test_func".to_string()));
        func_op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("(i32) -> i32".to_string()));
        
        let result = lowering.lower_operation(&func_op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.func");
        
        // Check that calling convention is set
        assert_eq!(result[0].attributes.get("CConv"), 
            Some(&MLIRAttribute::String("ccc".to_string())));
        
        // Check that linkage is set
        assert_eq!(result[0].attributes.get("linkage"), 
            Some(&MLIRAttribute::String("external".to_string())));
    }

    #[test]
    fn test_memory_management_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        // Test memref.alloc lowering
        let mut alloc_op = MLIROperation::new("memref.alloc".to_string());
        alloc_op.add_result(MLIRValue::new("buffer".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        }));
        
        let result = lowering.lower_operation(&alloc_op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("malloc".to_string())));
        
        // Check that size is calculated correctly (10 * 20 * 4 bytes = 800 bytes)
        assert_eq!(result[0].attributes.get("size_bytes"), 
            Some(&MLIRAttribute::Integer(800)));
    }

    #[test]
    fn test_runtime_function_generation() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        // Test that Aether-specific runtime calls are generated
        let mut linalg_op = MLIROperation::new("linalg.matmul".to_string());
        linalg_op.add_operand(MLIRValue::new("a".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
        }));
        linalg_op.add_operand(MLIRValue::new("b".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 2],
        }));
        linalg_op.add_result(MLIRValue::new("c".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
        }));
        
        let result = lowering.lower_operation(&linalg_op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("__aether_matmul".to_string())));
    }

    #[test]
    fn test_linalg_matmul_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("linalg.matmul".to_string());
        op.add_operand(MLIRValue::new("lhs".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
        }));
        op.add_operand(MLIRValue::new("rhs".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 5],
        }));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 5],
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("__aether_matmul".to_string())));
        assert_eq!(result[0].operands.len(), 2);
        assert_eq!(result[0].results.len(), 1);
    }

    #[test]
    fn test_linalg_init_tensor_lowering() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let mut op = MLIROperation::new("linalg.init_tensor".to_string());
        op.add_attribute("static_sizes".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(3),
            MLIRAttribute::Integer(4),
        ]));
        op.add_result(MLIRValue::new("tensor".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llvm.call");
        assert_eq!(result[0].attributes.get("callee"), 
            Some(&MLIRAttribute::String("malloc".to_string())));
        assert!(result[0].attributes.contains_key("size_bytes"));
        assert_eq!(result[0].results.len(), 1);
        assert!(matches!(result[0].results[0].value_type, MLIRType::Pointer));
    }

    #[test]
    fn test_unsupported_operation() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        let op = MLIROperation::new("unsupported.operation".to_string());
        
        let result = lowering.lower_operation(&op);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoweringError::UnsupportedOperation { operation, source_dialect, target_dialect } => {
                assert_eq!(operation, "unsupported.operation");
                assert_eq!(source_dialect, "standard");
                assert_eq!(target_dialect, "llvm");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_type_conversion() {
        let context = create_test_context();
        let mut lowering = StandardToLLVMLowering::new(&context);
        
        // Test integer type conversion
        let int_type = MLIRType::Integer { width: 32, signed: true };
        let converted = lowering.convert_type_to_llvm(&int_type).unwrap();
        assert!(matches!(converted, MLIRType::Integer { width: 32, signed: true }));
        
        // Test float type conversion
        let float_type = MLIRType::Float { width: 64 };
        let converted = lowering.convert_type_to_llvm(&float_type).unwrap();
        assert!(matches!(converted, MLIRType::Float { width: 64 }));
        
        // Test index type conversion
        let index_type = MLIRType::Index;
        let converted = lowering.convert_type_to_llvm(&index_type).unwrap();
        assert!(matches!(converted, MLIRType::Integer { width: 64, signed: true }));
        
        // Test memref type conversion
        let memref_type = MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        };
        let converted = lowering.convert_type_to_llvm(&memref_type).unwrap();
        assert!(matches!(converted, MLIRType::Pointer));
        
        // Test tensor type conversion
        let tensor_type = MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
        };
        let converted = lowering.convert_type_to_llvm(&tensor_type).unwrap();
        assert!(matches!(converted, MLIRType::Pointer));
    }

    #[test]
    fn test_comprehensive_standard_to_llvm_pass() {
        let context = create_test_context();
        let mut module = context.create_module("test_module").unwrap();
        
        // Add various standard operations
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test_func".to_string()));
        func_op.add_attribute("function_type".to_string(), 
            MLIRAttribute::String("() -> ()".to_string()));
        module.add_operation(func_op).unwrap();
        
        let mut alloc_op = MLIROperation::new("memref.alloc".to_string());
        alloc_op.add_result(MLIRValue::new("memref".to_string(), MLIRType::Memref {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10],
        }));
        module.add_operation(alloc_op).unwrap();
        
        let mut matmul_op = MLIROperation::new("linalg.matmul".to_string());
        matmul_op.add_operand(MLIRValue::new("a".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
        }));
        matmul_op.add_operand(MLIRValue::new("b".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 2],
        }));
        matmul_op.add_result(MLIRValue::new("c".to_string(), MLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
        }));
        module.add_operation(matmul_op).unwrap();
        
        // Add a non-standard operation that should remain unchanged
        let builtin_op = MLIROperation::new("builtin.module".to_string());
        module.add_operation(builtin_op).unwrap();
        
        let lowering = StandardToLLVMLowering::new(&context);
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        assert!(result.is_ok());
        
        let stats = pass.get_statistics();
        assert_eq!(stats.total_operations, 3); // Only standard operations
        assert_eq!(stats.successful_lowerings, 3);
        assert_eq!(stats.failed_lowerings, 0);
        
        // Check that operations were converted
        let operations = module.operations();
        let standard_ops: Vec<_> = operations.iter()
            .filter(|op| {
                let dialect = op.name.split('.').next().unwrap_or("");
                matches!(dialect, "func" | "linalg" | "memref" | "tensor" | "arith")
            })
            .collect();
        let llvm_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("llvm."))
            .collect();
        let builtin_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("builtin."))
            .collect();
        
        assert_eq!(standard_ops.len(), 0); // All standard ops should be lowered
        assert!(llvm_ops.len() >= 3); // Should have at least 3 LLVM ops
        assert_eq!(builtin_ops.len(), 1); // builtin op should remain unchanged
    }
}

// ===== BASIC LOWERING TESTS =====

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    fn create_test_module(context: &MLIRContext) -> MLIRModule {
        context.create_module("test_module").unwrap()
    }

    #[test]
    fn test_dialect_lowering_trait_basic() {
        let lowering = MockAetherToStandardLowering::new();
        
        assert_eq!(lowering.get_source_dialect(), "aether");
        assert_eq!(lowering.get_target_dialect(), "linalg");
        
        // Test can_lower_operation
        let aether_op = MLIROperation::new("aether.tensor_create".to_string());
        let other_op = MLIROperation::new("func.func".to_string());
        
        assert!(lowering.can_lower_operation(&aether_op));
        assert!(!lowering.can_lower_operation(&other_op));
    }

    #[test]
    fn test_lowering_config_default() {
        let config = LoweringConfig::default();
        
        assert!(config.preserve_debug_info);
        assert!(!config.aggressive_optimization);
        assert!(config.verify_after_lowering);
        assert!(config.target_hints.is_empty());
    }

    #[test]
    fn test_type_converter_aether_tensor_to_standard() {
        let context = create_test_context();
        let mut converter = TypeConverter::new(&context);
        
        let aether_tensor = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
            device: "cpu".to_string(),
        };
        
        let result = converter.convert_aether_tensor_to_standard(&aether_tensor).unwrap();
        
        match result {
            MLIRType::Tensor { element_type, shape } => {
                assert!(matches!(element_type.as_ref(), MLIRType::Float { width: 32 }));
                assert_eq!(shape, vec![2, 3, 4]);
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_type_converter_aether_linear_to_memref() {
        let context = create_test_context();
        let mut converter = TypeConverter::new(&context);
        
        let aether_linear = MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Integer { width: 32, signed: true }),
                shape: vec![10, 20],
                device: "cpu".to_string(),
            }),
        };
        
        let result = converter.convert_aether_linear_to_memref(&aether_linear).unwrap();
        
        match result {
            MLIRType::Memref { element_type, shape } => {
                assert!(matches!(element_type.as_ref(), MLIRType::Integer { width: 32, signed: true }));
                assert_eq!(shape, vec![10, 20]);
            }
            _ => panic!("Expected memref type"),
        }
    }

    #[test]
    fn test_type_converter_aether_probabilistic_to_standard() {
        let context = create_test_context();
        let mut converter = TypeConverter::new(&context);
        
        let aether_prob = MLIRType::AetherProbabilistic {
            distribution: "normal".to_string(),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        };
        
        let result = converter.convert_aether_probabilistic_to_standard(&aether_prob).unwrap();
        
        match result {
            MLIRType::Float { width: 64 } => {
                // Expected result
            }
            _ => panic!("Expected float type"),
        }
    }

    #[test]
    fn test_type_converter_cache() {
        let context = create_test_context();
        let mut converter = TypeConverter::new(&context);
        
        let aether_tensor = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![5, 5],
            device: "gpu".to_string(),
        };
        
        // First conversion should populate cache
        let result1 = converter.convert_aether_tensor_to_standard(&aether_tensor).unwrap();
        let (cache_size_1, _) = converter.get_cache_stats();
        
        // Second conversion should use cache
        let result2 = converter.convert_aether_tensor_to_standard(&aether_tensor).unwrap();
        let (cache_size_2, _) = converter.get_cache_stats();
        
        assert_eq!(cache_size_1, cache_size_2); // Cache size shouldn't change
        assert_eq!(format!("{:?}", result1), format!("{:?}", result2)); // Results should be identical
        
        // Clear cache
        converter.clear_cache();
        let (cache_size_3, _) = converter.get_cache_stats();
        assert_eq!(cache_size_3, 0);
    }

    #[test]
    fn test_lowering_statistics() {
        let mut stats = LoweringStatistics::new();
        
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_lowerings, 0);
        assert_eq!(stats.failed_lowerings, 0);
        assert_eq!(stats.success_rate(), 0.0);
        
        stats.total_operations = 10;
        stats.successful_lowerings = 8;
        stats.failed_lowerings = 2;
        
        assert_eq!(stats.success_rate(), 0.8);
        
        stats.reset();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_lowerings, 0);
        assert_eq!(stats.failed_lowerings, 0);
    }

    #[test]
    fn test_mock_lowering_tensor_create() {
        let lowering = MockAetherToStandardLowering::new();
        
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(3),
        ]));
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "linalg.init_tensor");
        assert!(result[0].attributes.contains_key("shape"));
        assert_eq!(result[0].results.len(), 1);
        
        match &result[0].results[0].value_type {
            MLIRType::Tensor { element_type, shape } => {
                assert!(matches!(element_type.as_ref(), MLIRType::Float { width: 32 }));
                assert_eq!(*shape, vec![2, 3]);
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_mock_lowering_linear_alloc() {
        let lowering = MockAetherToStandardLowering::new();
        
        let mut op = MLIROperation::new("aether.linear_alloc".to_string());
        op.add_result(MLIRValue::new("result".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Integer { width: 32, signed: true }),
                shape: vec![10],
                device: "cpu".to_string(),
            }),
        }));
        
        let result = lowering.lower_operation(&op).unwrap();
        
        assert_eq!(result.len(), 2); // alloc + cast operations
        assert_eq!(result[0].name, "memref.alloc");
        assert_eq!(result[1].name, "memref.cast");
    }

    #[test]
    fn test_mock_lowering_unsupported_operation() {
        let lowering = MockAetherToStandardLowering::new();
        
        let op = MLIROperation::new("aether.unsupported_op".to_string());
        
        let result = lowering.lower_operation(&op);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            LoweringError::UnsupportedOperation { operation, source_dialect, target_dialect } => {
                assert_eq!(operation, "aether.unsupported_op");
                assert_eq!(source_dialect, "aether");
                assert_eq!(target_dialect, "linalg");
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_lowering_pass_successful_run() {
        let context = create_test_context();
        let mut module = create_test_module(&context);
        
        // Add some Aether operations
        let mut tensor_op = MLIROperation::new("aether.tensor_create".to_string());
        tensor_op.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 4],
            device: "cpu".to_string(),
        }));
        module.add_operation(tensor_op).unwrap();
        
        let matmul_op = MLIROperation::new("aether.matmul".to_string());
        module.add_operation(matmul_op).unwrap();
        
        // Add a non-Aether operation that should be left unchanged
        let func_op = MLIROperation::new("func.func".to_string());
        module.add_operation(func_op).unwrap();
        
        let lowering = MockAetherToStandardLowering::new();
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        assert!(result.is_ok());
        
        let stats = pass.get_statistics();
        assert_eq!(stats.total_operations, 2); // Only Aether operations
        assert_eq!(stats.successful_lowerings, 2);
        assert_eq!(stats.failed_lowerings, 0);
        assert_eq!(stats.success_rate(), 1.0);
        
        // Check that operations were converted
        let operations = module.operations();
        let aether_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("aether."))
            .collect();
        let linalg_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("linalg."))
            .collect();
        let func_ops: Vec<_> = operations.iter()
            .filter(|op| op.name.starts_with("func."))
            .collect();
        
        assert_eq!(aether_ops.len(), 0); // All Aether ops should be lowered
        assert_eq!(linalg_ops.len(), 2); // Should have 2 linalg ops
        assert_eq!(func_ops.len(), 1); // func op should remain unchanged
    }

    #[test]
    fn test_lowering_pass_with_failures() {
        let context = create_test_context();
        let mut module = create_test_module(&context);
        
        // Add an Aether operation
        let tensor_op = MLIROperation::new("aether.tensor_create".to_string());
        module.add_operation(tensor_op).unwrap();
        
        let lowering = MockAetherToStandardLowering::new_with_failure();
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        
        // The pass should continue despite unsupported operations
        assert!(result.is_ok());
        
        let stats = pass.get_statistics();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_lowerings, 0);
        assert_eq!(stats.failed_lowerings, 1);
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.errors.len(), 1);
    }

    #[test]
    fn test_lowering_pass_custom_config() {
        let mut config = LoweringConfig::default();
        config.aggressive_optimization = true;
        config.verify_after_lowering = false;
        config.target_hints.insert("optimization_level".to_string(), "3".to_string());
        
        let lowering = MockAetherToStandardLowering::new();
        let pass = LoweringPass::with_config(lowering, config.clone());
        
        assert_eq!(pass.get_config().aggressive_optimization, true);
        assert_eq!(pass.get_config().verify_after_lowering, false);
        assert_eq!(pass.get_config().target_hints.get("optimization_level"), Some(&"3".to_string()));
    }

    #[test]
    fn test_lowering_error_display() {
        let error = LoweringError::UnsupportedOperation {
            operation: "aether.test_op".to_string(),
            source_dialect: "aether".to_string(),
            target_dialect: "linalg".to_string(),
        };
        
        let error_str = format!("{}", error);
        assert!(error_str.contains("Cannot lower operation 'aether.test_op'"));
        assert!(error_str.contains("from 'aether' to 'linalg'"));
        
        let type_error = LoweringError::TypeConversionError {
            from_type: "AetherTensor".to_string(),
            to_type: "StandardTensor".to_string(),
            reason: "Shape mismatch".to_string(),
        };
        
        let type_error_str = format!("{}", type_error);
        assert!(type_error_str.contains("Type conversion failed"));
        assert!(type_error_str.contains("Shape mismatch"));
    }

    #[test]
    fn test_lowering_pass_verification() {
        let context = create_test_context();
        let mut module = create_test_module(&context);
        
        // Create a mock lowering that produces invalid operations
        struct InvalidLowering;
        
        impl DialectLowering for InvalidLowering {
            fn lower_operation(&self, _op: &MLIROperation) -> Result<Vec<MLIROperation>, LoweringError> {
                // Create an operation with invalid target dialect
                let mut invalid_op = MLIROperation::new("invalid.operation".to_string());
                invalid_op.add_result(MLIRValue::new("".to_string(), MLIRType::Float { width: 32 })); // Empty ID
                Ok(vec![invalid_op])
            }
            
            fn get_source_dialect(&self) -> &str { "aether" }
            fn get_target_dialect(&self) -> &str { "linalg" }
        }
        
        let tensor_op = MLIROperation::new("aether.tensor_create".to_string());
        module.add_operation(tensor_op).unwrap();
        
        let lowering = InvalidLowering;
        let mut pass = LoweringPass::new(lowering);
        
        let result = pass.run(&mut module);
        
        // Should fail verification
        assert!(result.is_err());
        match result.unwrap_err() {
            LoweringError::VerificationError { .. } => {
                // Expected
            }
            other => panic!("Expected VerificationError, got {:?}", other),
        }
    }
}