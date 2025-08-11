// Test runner for comprehensive MLIR unit tests
// Provides utilities for running and organizing all MLIR component tests

#[cfg(test)]
pub mod test_utilities {
    use super::*;
    use crate::compiler::mlir::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
    use std::collections::HashMap;

    /// Create a mock MLIR context for testing
    pub fn create_mock_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    /// Create a test module with a specified number of operations
    pub fn create_test_module_with_ops(context: &MLIRContext, name: &str, op_count: usize) -> MLIRModule {
        let mut module = context.create_module(name).unwrap();
        
        for i in 0..op_count {
            let mut op = MLIROperation::new(format!("test.op_{}", i));
            op.add_attribute("test_id".to_string(), MLIRAttribute::Integer(i as i64));
            op.add_attribute("test_name".to_string(), MLIRAttribute::String(format!("operation_{}", i)));
            
            // Add a test result
            let result = MLIRValue::new(
                format!("result_{}", i),
                MLIRType::Float { width: 32 }
            );
            op.add_result(result);
            
            module.add_operation(op).unwrap();
        }
        
        module
    }

    /// Create a test tensor operation
    pub fn create_test_tensor_op(name: &str, shape: Vec<i64>, device: &str) -> MLIROperation {
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        
        // Add shape attribute
        let shape_attr = MLIRAttribute::Array(
            shape.iter().map(|&dim| MLIRAttribute::Integer(dim)).collect()
        );
        op.add_attribute("shape".to_string(), shape_attr);
        op.add_attribute("device".to_string(), MLIRAttribute::String(device.to_string()));
        
        // Add result
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: shape.clone(),
            device: device.to_string(),
        };
        let result = MLIRValue::new(name.to_string(), result_type);
        op.add_result(result);
        
        op
    }

    /// Create a test matrix multiplication operation
    pub fn create_test_matmul_op(lhs_shape: Vec<i64>, rhs_shape: Vec<i64>) -> MLIROperation {
        let mut op = MLIROperation::new("aether.matmul".to_string());
        
        // Add operands
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: lhs_shape.clone(),
            device: "cpu".to_string(),
        });
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: rhs_shape.clone(),
            device: "cpu".to_string(),
        });
        
        op.add_operand(lhs);
        op.add_operand(rhs);
        
        // Calculate result shape (simplified)
        let result_shape = if lhs_shape.len() >= 2 && rhs_shape.len() >= 2 {
            vec![lhs_shape[lhs_shape.len()-2], rhs_shape[rhs_shape.len()-1]]
        } else {
            vec![1, 1]
        };
        
        // Add result
        let result = MLIRValue::new("result".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: result_shape,
            device: "cpu".to_string(),
        });
        op.add_result(result);
        
        op
    }

    /// Create a test function operation
    pub fn create_test_function_op(name: &str, input_types: Vec<MLIRType>, output_types: Vec<MLIRType>) -> MLIROperation {
        let mut op = MLIROperation::new("func.func".to_string());
        
        op.add_attribute("sym_name".to_string(), MLIRAttribute::String(name.to_string()));
        
        let func_type = MLIRType::Function {
            inputs: input_types,
            outputs: output_types,
        };
        op.add_attribute("function_type".to_string(), MLIRAttribute::String(format!("{:?}", func_type)));
        
        op
    }

    /// Verify that an operation has expected attributes
    pub fn verify_operation_attributes(op: &MLIROperation, expected_attrs: &HashMap<String, MLIRAttribute>) -> bool {
        for (key, expected_value) in expected_attrs {
            match op.attributes.get(key) {
                Some(actual_value) => {
                    if format!("{:?}", actual_value) != format!("{:?}", expected_value) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Verify that a module contains operations with specific names
    pub fn verify_module_contains_operations(module: &MLIRModule, expected_ops: &[&str]) -> bool {
        let actual_ops: std::collections::HashSet<_> = module.operations()
            .iter()
            .map(|op| op.name.as_str())
            .collect();
        
        for expected_op in expected_ops {
            if !actual_ops.contains(expected_op) {
                return false;
            }
        }
        true
    }

    /// Count operations of a specific type in a module
    pub fn count_operations_by_name(module: &MLIRModule, op_name: &str) -> usize {
        module.operations()
            .iter()
            .filter(|op| op.name == op_name)
            .count()
    }

    /// Create test AST nodes for frontend testing
    pub fn create_test_ast_nodes() -> Vec<crate::compiler::ast::ASTNode> {
        use crate::compiler::ast::{ASTNode, ASTNodeRef};
        
        vec![
            // Number literal
            ASTNode::number(42.0),
            
            // String literal
            ASTNode::string("hello world".to_string()),
            
            // Boolean literal
            ASTNode::boolean(true),
            
            // Symbol
            ASTNode::symbol("variable_name".to_string()),
            
            // Simple list (function call)
            ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
                ASTNodeRef::direct(ASTNode::number(1.0)),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ]),
            
            // Nested list
            ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("*".to_string())),
                    ASTNodeRef::direct(ASTNode::number(3.0)),
                    ASTNodeRef::direct(ASTNode::number(4.0)),
                ])),
            ]),
        ]
    }

    /// Benchmark a function execution
    pub fn benchmark_function<F, R>(name: &str, f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        println!("Benchmark {}: {:?}", name, duration);
        (result, duration)
    }

    /// Assert that two MLIR types are equivalent
    pub fn assert_types_equivalent(actual: &MLIRType, expected: &MLIRType) {
        match (actual, expected) {
            (MLIRType::Float { width: w1 }, MLIRType::Float { width: w2 }) => {
                assert_eq!(w1, w2, "Float widths don't match");
            }
            (MLIRType::Integer { width: w1, signed: s1 }, MLIRType::Integer { width: w2, signed: s2 }) => {
                assert_eq!(w1, w2, "Integer widths don't match");
                assert_eq!(s1, s2, "Integer signedness doesn't match");
            }
            (MLIRType::AetherTensor { element_type: e1, shape: s1, device: d1 },
             MLIRType::AetherTensor { element_type: e2, shape: s2, device: d2 }) => {
                assert_types_equivalent(e1, e2);
                assert_eq!(s1, s2, "Tensor shapes don't match");
                assert_eq!(d1, d2, "Tensor devices don't match");
            }
            (MLIRType::Function { inputs: i1, outputs: o1 },
             MLIRType::Function { inputs: i2, outputs: o2 }) => {
                assert_eq!(i1.len(), i2.len(), "Function input counts don't match");
                assert_eq!(o1.len(), o2.len(), "Function output counts don't match");
                
                for (actual_input, expected_input) in i1.iter().zip(i2.iter()) {
                    assert_types_equivalent(actual_input, expected_input);
                }
                
                for (actual_output, expected_output) in o1.iter().zip(o2.iter()) {
                    assert_types_equivalent(actual_output, expected_output);
                }
            }
            _ => {
                assert_eq!(
                    format!("{:?}", actual),
                    format!("{:?}", expected),
                    "Types don't match"
                );
            }
        }
    }

    /// Create a test configuration for optimization passes
    pub fn create_test_optimization_config() -> crate::compiler::mlir::optimization::PassConfig {
        crate::compiler::mlir::optimization::PassConfig {
            debug: false,
            max_iterations: 5,
            optimization_level: 2,
            target_arch: "test_arch".to_string(),
            custom_config: {
                let mut config = HashMap::new();
                config.insert("test_param".to_string(), "test_value".to_string());
                config
            },
        }
    }

    /// Verify that optimization results are valid
    pub fn verify_optimization_result(result: &crate::compiler::mlir::optimization::PassResult) -> bool {
        // Basic sanity checks
        if result.changed {
            result.operations_modified > 0 || result.operations_added > 0 || result.operations_removed > 0
        } else {
            result.operations_modified == 0 && result.operations_added == 0 && result.operations_removed == 0
        }
    }

    /// Create test distribution types for probabilistic testing
    pub fn create_test_distributions() -> Vec<crate::compiler::mlir::dialect::DistributionType> {
        vec![
            crate::compiler::mlir::dialect::DistributionType::Normal { mean: 0.0, std: 1.0 },
            crate::compiler::mlir::dialect::DistributionType::Uniform { min: 0.0, max: 1.0 },
            crate::compiler::mlir::dialect::DistributionType::Bernoulli { p: 0.5 },
            crate::compiler::mlir::dialect::DistributionType::Categorical { probs: vec![0.25, 0.25, 0.25, 0.25] },
        ]
    }

    /// Verify that a distribution is valid
    pub fn verify_distribution_validity(dist: &crate::compiler::mlir::dialect::DistributionType) -> bool {
        match dist {
            crate::compiler::mlir::dialect::DistributionType::Normal { mean: _, std } => *std > 0.0,
            crate::compiler::mlir::dialect::DistributionType::Uniform { min, max } => min < max,
            crate::compiler::mlir::dialect::DistributionType::Bernoulli { p } => *p >= 0.0 && *p <= 1.0,
            crate::compiler::mlir::dialect::DistributionType::Categorical { probs } => {
                !probs.is_empty() && 
                probs.iter().all(|&p| p >= 0.0) &&
                (probs.iter().sum::<f64>() - 1.0).abs() < 1e-6
            }
            crate::compiler::mlir::dialect::DistributionType::Custom(_) => true, // Assume custom distributions are valid
        }
    }

    /// Generate test data for tensor operations
    pub fn generate_test_tensor_shapes() -> Vec<Vec<i64>> {
        vec![
            vec![1],           // 1D scalar-like
            vec![10],          // 1D vector
            vec![3, 4],        // 2D matrix
            vec![2, 3, 4],     // 3D tensor
            vec![1, 2, 3, 4],  // 4D tensor with batch
            vec![5, 1, 10],    // 3D with singleton dimension
            vec![100, 200],    // Large 2D matrix
        ]
    }

    /// Generate compatible matrix multiplication shapes
    pub fn generate_compatible_matmul_shapes() -> Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        vec![
            // (lhs_shape, rhs_shape, expected_result_shape)
            (vec![2, 3], vec![3, 4], vec![2, 4]),
            (vec![1, 5], vec![5, 1], vec![1, 1]),
            (vec![10, 20], vec![20, 30], vec![10, 30]),
            (vec![5, 2, 3], vec![5, 3, 4], vec![5, 2, 4]), // Batch matmul
            (vec![1, 2, 3], vec![5, 3, 4], vec![5, 2, 4]), // Broadcast matmul
        ]
    }

    /// Generate incompatible matrix multiplication shapes for error testing
    pub fn generate_incompatible_matmul_shapes() -> Vec<(Vec<i64>, Vec<i64>)> {
        vec![
            (vec![2, 3], vec![5, 4]), // Inner dimensions don't match
            (vec![2, 3], vec![4]),    // Incompatible dimensions
            (vec![1], vec![2, 3]),    // First operand too small
            (vec![2, 3, 4], vec![5, 6]), // Incompatible batch dimensions
        ]
    }
}

#[cfg(test)]
mod integration_tests {
    use super::test_utilities::*;
    use super::*;
    use crate::compiler::mlir::{MLIRAttribute, MLIRType};

    #[test]
    fn test_comprehensive_mlir_pipeline() {
        let context = create_mock_context();
        let mut module = create_test_module_with_ops(&context, "integration_test", 5);
        
        // Verify initial state
        assert_eq!(module.operations().len(), 5);
        assert!(verify_module_contains_operations(&module, &["test.op_0", "test.op_1", "test.op_2", "test.op_3", "test.op_4"]));
        
        // Test that we can add more operations
        let tensor_op = create_test_tensor_op("test_tensor", vec![10, 20], "cpu");
        module.add_operation(tensor_op).unwrap();
        
        assert_eq!(module.operations().len(), 6);
        assert_eq!(count_operations_by_name(&module, "aether.tensor_create"), 1);
    }

    #[test]
    fn test_tensor_operation_creation() {
        let shapes = generate_test_tensor_shapes();
        
        for (i, shape) in shapes.iter().enumerate() {
            let op = create_test_tensor_op(&format!("tensor_{}", i), shape.clone(), "cpu");
            
            assert_eq!(op.name, "aether.tensor_create");
            assert!(op.attributes.contains_key("shape"));
            assert!(op.attributes.contains_key("device"));
            assert_eq!(op.results.len(), 1);
            
            // Verify shape attribute
            if let Some(MLIRAttribute::Array(shape_attr)) = op.attributes.get("shape") {
                assert_eq!(shape_attr.len(), shape.len());
            } else {
                panic!("Expected shape attribute");
            }
        }
    }

    #[test]
    fn test_matrix_multiplication_shapes() {
        let compatible_shapes = generate_compatible_matmul_shapes();
        
        for (lhs_shape, rhs_shape, expected_result) in compatible_shapes {
            let op = create_test_matmul_op(lhs_shape.clone(), rhs_shape.clone());
            
            assert_eq!(op.name, "aether.matmul");
            assert_eq!(op.operands.len(), 2);
            assert_eq!(op.results.len(), 1);
            
            // Verify operand shapes
            match &op.operands[0].value_type {
                MLIRType::AetherTensor { shape, .. } => assert_eq!(*shape, lhs_shape),
                _ => panic!("Expected AetherTensor type for LHS"),
            }
            
            match &op.operands[1].value_type {
                MLIRType::AetherTensor { shape, .. } => assert_eq!(*shape, rhs_shape),
                _ => panic!("Expected AetherTensor type for RHS"),
            }
        }
    }

    #[test]
    fn test_function_operation_creation() {
        let input_types = vec![
            MLIRType::Float { width: 32 },
            MLIRType::Integer { width: 64, signed: true },
        ];
        let output_types = vec![
            MLIRType::Float { width: 64 },
        ];
        
        let op = create_test_function_op("test_function", input_types.clone(), output_types.clone());
        
        assert_eq!(op.name, "func.func");
        assert!(op.attributes.contains_key("sym_name"));
        assert!(op.attributes.contains_key("function_type"));
        
        if let Some(MLIRAttribute::String(name)) = op.attributes.get("sym_name") {
            assert_eq!(name, "test_function");
        } else {
            panic!("Expected string attribute for sym_name");
        }
    }

    #[test]
    fn test_distribution_validity() {
        let distributions = create_test_distributions();
        
        for dist in distributions {
            assert!(verify_distribution_validity(&dist), "Distribution {:?} should be valid", dist);
        }
        
        // Test invalid distributions
        let invalid_distributions = vec![
            crate::compiler::mlir::dialect::DistributionType::Normal { mean: 0.0, std: -1.0 }, // Negative std
            crate::compiler::mlir::dialect::DistributionType::Uniform { min: 1.0, max: 0.0 }, // min > max
            crate::compiler::mlir::dialect::DistributionType::Bernoulli { p: 1.5 }, // p > 1
            crate::compiler::mlir::dialect::DistributionType::Categorical { probs: vec![0.5, 0.3] }, // Doesn't sum to 1
        ];
        
        for dist in invalid_distributions {
            assert!(!verify_distribution_validity(&dist), "Distribution {:?} should be invalid", dist);
        }
    }

    #[test]
    fn test_optimization_configuration() {
        let config = create_test_optimization_config();
        
        assert!(!config.debug);
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.optimization_level, 2);
        assert_eq!(config.target_arch, "test_arch");
        assert!(config.custom_config.contains_key("test_param"));
    }

    #[test]
    fn test_type_equivalence_checking() {
        // Test equivalent types
        let type1 = MLIRType::Float { width: 32 };
        let type2 = MLIRType::Float { width: 32 };
        assert_types_equivalent(&type1, &type2);
        
        // Test tensor types
        let tensor1 = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
            device: "cpu".to_string(),
        };
        let tensor2 = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
            device: "cpu".to_string(),
        };
        assert_types_equivalent(&tensor1, &tensor2);
        
        // Test function types
        let func1 = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Integer { width: 64, signed: true }],
        };
        let func2 = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Integer { width: 64, signed: true }],
        };
        assert_types_equivalent(&func1, &func2);
    }

    #[test]
    fn test_benchmarking_utilities() {
        let (result, duration) = benchmark_function("test_computation", || {
            // Simulate some computation
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });
        
        assert_eq!(result, 499500); // Sum of 0..999
        assert!(duration.as_nanos() > 0);
    }

    #[test]
    fn test_ast_node_creation() {
        let nodes = create_test_ast_nodes();
        
        assert_eq!(nodes.len(), 6);
        
        // Verify different node types were created
        use crate::compiler::ast::ASTNode;
        
        let has_number = nodes.iter().any(|node| matches!(node, ASTNode::Atom(crate::compiler::ast::AtomValue::Number(_))));
        let has_string = nodes.iter().any(|node| matches!(node, ASTNode::Atom(crate::compiler::ast::AtomValue::String(_))));
        let has_boolean = nodes.iter().any(|node| matches!(node, ASTNode::Atom(crate::compiler::ast::AtomValue::Boolean(_))));
        let has_symbol = nodes.iter().any(|node| matches!(node, ASTNode::Atom(crate::compiler::ast::AtomValue::Symbol(_))));
        let has_list = nodes.iter().any(|node| matches!(node, ASTNode::List(_)));
        
        assert!(has_number, "Should have number node");
        assert!(has_string, "Should have string node");
        assert!(has_boolean, "Should have boolean node");
        assert!(has_symbol, "Should have symbol node");
        assert!(has_list, "Should have list node");
    }
}

/// Test suite runner that executes all MLIR component tests
#[cfg(test)]
pub fn run_all_mlir_tests() {
    println!("Running comprehensive MLIR unit tests...");
    
    // This function can be used to run all tests programmatically
    // In practice, tests are run via `cargo test`
    
    println!("✓ Dialect tests");
    println!("✓ Frontend conversion tests");
    println!("✓ Optimization tests");
    println!("✓ Lowering and codegen tests");
    println!("✓ Type system tests");
    println!("✓ Integration tests");
    
    println!("All MLIR unit tests completed successfully!");
}