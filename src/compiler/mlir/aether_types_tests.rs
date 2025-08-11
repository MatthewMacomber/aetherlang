// Comprehensive tests for Aether MLIR type system
// Tests type conversion, verification, and constraint checking

use super::aether_types::*;
use super::mlir_context::{MLIRType, MLIRError};
use crate::compiler::types::{LinearOwnership, Lifetime};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic AetherMLIRType creation and properties
    #[test]
    fn test_aether_mlir_type_creation() {
        // Test tensor type creation
        let tensor_type = AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
            is_differentiable: true,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        };

        match tensor_type {
            AetherMLIRType::Tensor { shape, is_differentiable, device, .. } => {
                assert_eq!(shape, vec![2, 3, 4]);
                assert!(is_differentiable);
                assert_eq!(device, "cpu");
            }
            _ => panic!("Expected tensor type"),
        }

        // Test probabilistic type creation
        let prob_type = AetherMLIRType::ProbabilisticVariable {
            distribution: DistributionType::Normal { mean: 0.0, std: 1.0 },
            value_type: Box::new(MLIRType::Float { width: 64 }),
            inference_method: InferenceMethod::MCMC,
        };

        match prob_type {
            AetherMLIRType::ProbabilisticVariable { distribution, inference_method, .. } => {
                match distribution {
                    DistributionType::Normal { mean, std } => {
                        assert_eq!(mean, 0.0);
                        assert_eq!(std, 1.0);
                    }
                    _ => panic!("Expected normal distribution"),
                }
                match inference_method {
                    InferenceMethod::MCMC => {}
                    _ => panic!("Expected MCMC inference method"),
                }
            }
            _ => panic!("Expected probabilistic type"),
        }
    }

    /// Test linear type creation and ownership
    #[test]
    fn test_linear_type_creation() {
        let linear_type = AetherMLIRType::LinearType {
            inner_type: Box::new(MLIRType::Integer { width: 32, signed: true }),
            ownership_info: LinearOwnershipInfo {
                ownership: LinearOwnership::Owned,
                allocation_site: Some("test_site".to_string()),
                deallocation_site: None,
                move_semantics: MoveSemanticsInfo {
                    is_movable: true,
                    is_copyable: false,
                    requires_explicit_drop: true,
                    gpu_resource: false,
                },
            },
            lifetime_info: None,
        };

        match linear_type {
            AetherMLIRType::LinearType { ownership_info, .. } => {
                assert_eq!(ownership_info.ownership, LinearOwnership::Owned);
                assert_eq!(ownership_info.allocation_site, Some("test_site".to_string()));
                assert!(ownership_info.move_semantics.is_movable);
                assert!(!ownership_info.move_semantics.is_copyable);
            }
            _ => panic!("Expected linear type"),
        }
    }

    /// Test actor type creation
    #[test]
    fn test_actor_type_creation() {
        let actor_type = AetherMLIRType::ActorType {
            message_type: Box::new(MLIRType::Integer { width: 32, signed: true }),
            state_type: Box::new(MLIRType::Float { width: 64 }),
            mailbox_capacity: Some(100),
        };

        match actor_type {
            AetherMLIRType::ActorType { mailbox_capacity, .. } => {
                assert_eq!(mailbox_capacity, Some(100));
            }
            _ => panic!("Expected actor type"),
        }
    }

    /// Test function type with attributes
    #[test]
    fn test_function_type_creation() {
        let func_type = AetherMLIRType::Function {
            inputs: vec![
                MLIRType::Float { width: 32 },
                MLIRType::Float { width: 32 },
            ],
            outputs: vec![MLIRType::Float { width: 32 }],
            attributes: AetherFunctionAttributes {
                is_differentiable: true,
                is_gpu_kernel: false,
                is_actor_method: false,
                is_pure: true,
                optimization_hints: vec![OptimizationHint::Inline, OptimizationHint::Vectorize],
                calling_convention: CallingConvention::Fast,
            },
        };

        match func_type {
            AetherMLIRType::Function { inputs, outputs, attributes } => {
                assert_eq!(inputs.len(), 2);
                assert_eq!(outputs.len(), 1);
                assert!(attributes.is_differentiable);
                assert!(attributes.is_pure);
                assert_eq!(attributes.optimization_hints.len(), 2);
                match attributes.calling_convention {
                    CallingConvention::Fast => {}
                    _ => panic!("Expected fast calling convention"),
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    /// Test dependent type creation
    #[test]
    fn test_dependent_type_creation() {
        let base_type = AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        };

        let dependent_type = AetherMLIRType::Dependent {
            base_type: Box::new(base_type),
            parameters: vec![
                DependentParameter::Shape("N".to_string()),
                DependentParameter::Int("batch_size".to_string(), 32),
            ],
            constraints: vec![
                TypeConstraint::ShapeConstraint("N".to_string(), ShapeConstraintKind::Positive),
                TypeConstraint::IntConstraint("batch_size".to_string(), IntConstraintKind::GreaterThan(0)),
            ],
        };

        match dependent_type {
            AetherMLIRType::Dependent { parameters, constraints, .. } => {
                assert_eq!(parameters.len(), 2);
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected dependent type"),
        }
    }

    /// Test union and struct types
    #[test]
    fn test_composite_types() {
        // Test union type
        let union_type = AetherMLIRType::Union {
            name: "NumberType".to_string(),
            variants: vec![
                AetherMLIRType::Tensor {
                    element_type: Box::new(MLIRType::Integer { width: 32, signed: true }),
                    shape: vec![1],
                    is_differentiable: false,
                    device: "cpu".to_string(),
                    memory_layout: TensorMemoryLayout::RowMajor,
                },
                AetherMLIRType::Tensor {
                    element_type: Box::new(MLIRType::Float { width: 32 }),
                    shape: vec![1],
                    is_differentiable: false,
                    device: "cpu".to_string(),
                    memory_layout: TensorMemoryLayout::RowMajor,
                },
            ],
            tag_type: MLIRType::Integer { width: 8, signed: false },
        };

        match union_type {
            AetherMLIRType::Union { name, variants, .. } => {
                assert_eq!(name, "NumberType");
                assert_eq!(variants.len(), 2);
            }
            _ => panic!("Expected union type"),
        }

        // Test struct type
        let mut fields = HashMap::new();
        fields.insert("x".to_string(), AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        });
        fields.insert("y".to_string(), AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        });

        let struct_type = AetherMLIRType::Struct {
            name: "Point3D".to_string(),
            fields,
            is_packed: false,
        };

        match struct_type {
            AetherMLIRType::Struct { name, fields, is_packed } => {
                assert_eq!(name, "Point3D");
                assert_eq!(fields.len(), 2);
                assert!(!is_packed);
                assert!(fields.contains_key("x"));
                assert!(fields.contains_key("y"));
            }
            _ => panic!("Expected struct type"),
        }
    }

    /// Test type converter creation and basic functionality
    #[test]
    fn test_type_converter_creation() {
        let mut converter = AetherTypeConverter::new();
        
        // Test that converter starts with empty cache
        assert!(converter.get_cached_type("test_key").is_none());
        
        // Test constraint addition
        let constraint = TypeConstraint::ShapeConstraint(
            "N".to_string(), 
            ShapeConstraintKind::Positive
        );
        converter.add_constraint(constraint);
        
        // Test constraint verification
        assert!(converter.verify_constraints().is_ok());
    }

    /// Test tensor type conversion
    #[test]
    fn test_tensor_type_conversion() {
        let mut converter = AetherTypeConverter::new();
        
        let tensor_type = AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
            is_differentiable: true,
            device: "gpu".to_string(),
            memory_layout: TensorMemoryLayout::GpuOptimized,
        };

        let mlir_type = converter.convert_to_mlir(&tensor_type).unwrap();
        
        match mlir_type {
            MLIRType::AetherTensor { element_type, shape, device } => {
                assert_eq!(shape, vec![2, 3, 4]);
                assert_eq!(device, "gpu");
                match *element_type {
                    MLIRType::Float { width } => assert_eq!(width, 32),
                    _ => panic!("Expected float element type"),
                }
            }
            _ => panic!("Expected Aether tensor type"),
        }
    }

    /// Test probabilistic type conversion
    #[test]
    fn test_probabilistic_type_conversion() {
        let mut converter = AetherTypeConverter::new();
        
        let prob_type = AetherMLIRType::ProbabilisticVariable {
            distribution: DistributionType::Bernoulli { p: 0.7 },
            value_type: Box::new(MLIRType::Integer { width: 1, signed: false }),
            inference_method: InferenceMethod::Variational,
        };

        let mlir_type = converter.convert_to_mlir(&prob_type).unwrap();
        
        match mlir_type {
            MLIRType::AetherProbabilistic { distribution, inner_type } => {
                assert!(distribution.contains("Bernoulli"));
                match *inner_type {
                    MLIRType::Integer { width, signed } => {
                        assert_eq!(width, 1);
                        assert!(!signed);
                    }
                    _ => panic!("Expected integer inner type"),
                }
            }
            _ => panic!("Expected Aether probabilistic type"),
        }
    }

    /// Test linear type conversion
    #[test]
    fn test_linear_type_conversion() {
        let mut converter = AetherTypeConverter::new();
        
        let linear_type = AetherMLIRType::LinearType {
            inner_type: Box::new(MLIRType::Tensor { 
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![100, 100],
            }),
            ownership_info: LinearOwnershipInfo {
                ownership: LinearOwnership::GpuOwned,
                allocation_site: Some("gpu_alloc".to_string()),
                deallocation_site: None,
                move_semantics: MoveSemanticsInfo {
                    is_movable: true,
                    is_copyable: false,
                    requires_explicit_drop: true,
                    gpu_resource: true,
                },
            },
            lifetime_info: None,
        };

        let mlir_type = converter.convert_to_mlir(&linear_type).unwrap();
        
        match mlir_type {
            MLIRType::AetherLinear { inner_type } => {
                match *inner_type {
                    MLIRType::Tensor { shape, .. } => {
                        assert_eq!(shape, vec![100, 100]);
                    }
                    _ => panic!("Expected tensor inner type"),
                }
            }
            _ => panic!("Expected Aether linear type"),
        }
    }

    /// Test function type conversion
    #[test]
    fn test_function_type_conversion() {
        let mut converter = AetherTypeConverter::new();
        
        let func_type = AetherMLIRType::Function {
            inputs: vec![
                MLIRType::Float { width: 32 },
                MLIRType::Float { width: 32 },
            ],
            outputs: vec![MLIRType::Float { width: 32 }],
            attributes: AetherFunctionAttributes {
                is_differentiable: true,
                is_gpu_kernel: true,
                is_actor_method: false,
                is_pure: false,
                optimization_hints: vec![OptimizationHint::Vectorize],
                calling_convention: CallingConvention::GpuKernel,
            },
        };

        let mlir_type = converter.convert_to_mlir(&func_type).unwrap();
        
        match mlir_type {
            MLIRType::Function { inputs, outputs } => {
                assert_eq!(inputs.len(), 2);
                assert_eq!(outputs.len(), 1);
            }
            _ => panic!("Expected function type"),
        }
    }

    /// Test reverse conversion from MLIR to Aether types
    #[test]
    fn test_reverse_type_conversion() {
        let converter = AetherTypeConverter::new();
        
        let mlir_tensor = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![10, 20],
            device: "cpu".to_string(),
        };

        let aether_type = converter.convert_from_mlir(&mlir_tensor).unwrap();
        
        match aether_type {
            AetherMLIRType::Tensor { shape, device, .. } => {
                assert_eq!(shape, vec![10, 20]);
                assert_eq!(device, "cpu");
            }
            _ => panic!("Expected tensor type"),
        }
    }

    /// Test type verifier creation and basic operations
    #[test]
    fn test_type_verifier_creation() {
        let mut verifier = TypeVerifier::new();
        
        // Test tensor creation verification
        let operands = vec![];
        let results = vec![AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        }];

        assert!(verifier.verify_operation_types("aether.tensor_create", &operands, &results).is_ok());
    }

    /// Test matrix multiplication verification
    #[test]
    fn test_matmul_verification() {
        let mut verifier = TypeVerifier::new();
        
        let operands = vec![
            AetherMLIRType::Tensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![2, 3],
                is_differentiable: false,
                device: "cpu".to_string(),
                memory_layout: TensorMemoryLayout::RowMajor,
            },
            AetherMLIRType::Tensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![3, 4],
                is_differentiable: false,
                device: "cpu".to_string(),
                memory_layout: TensorMemoryLayout::RowMajor,
            },
        ];
        
        let results = vec![AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 4],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        }];

        assert!(verifier.verify_operation_types("aether.matmul", &operands, &results).is_ok());
    }

    /// Test invalid matrix multiplication verification
    #[test]
    fn test_invalid_matmul_verification() {
        let mut verifier = TypeVerifier::new();
        
        // Incompatible shapes: [2, 3] x [5, 4] (inner dimensions don't match)
        let operands = vec![
            AetherMLIRType::Tensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![2, 3],
                is_differentiable: false,
                device: "cpu".to_string(),
                memory_layout: TensorMemoryLayout::RowMajor,
            },
            AetherMLIRType::Tensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![5, 4],
                is_differentiable: false,
                device: "cpu".to_string(),
                memory_layout: TensorMemoryLayout::RowMajor,
            },
        ];
        
        let results = vec![AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 4],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        }];

        assert!(verifier.verify_operation_types("aether.matmul", &operands, &results).is_err());
    }

    /// Test automatic differentiation verification
    #[test]
    fn test_autodiff_verification() {
        let mut verifier = TypeVerifier::new();
        
        let operands = vec![
            AetherMLIRType::Function {
                inputs: vec![MLIRType::Float { width: 32 }],
                outputs: vec![MLIRType::Float { width: 32 }],
                attributes: AetherFunctionAttributes {
                    is_differentiable: true,
                    is_gpu_kernel: false,
                    is_actor_method: false,
                    is_pure: true,
                    optimization_hints: vec![],
                    calling_convention: CallingConvention::C,
                },
            },
        ];
        
        let results = vec![AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![1],
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        }];

        assert!(verifier.verify_operation_types("aether.autodiff_forward", &operands, &results).is_ok());
    }

    /// Test probabilistic variable verification
    #[test]
    fn test_prob_var_verification() {
        let mut verifier = TypeVerifier::new();
        
        let operands = vec![];
        let results = vec![AetherMLIRType::ProbabilisticVariable {
            distribution: DistributionType::Normal { mean: 0.0, std: 1.0 },
            value_type: Box::new(MLIRType::Float { width: 64 }),
            inference_method: InferenceMethod::Exact,
        }];

        assert!(verifier.verify_operation_types("aether.prob_var", &operands, &results).is_ok());
    }

    /// Test invalid probabilistic variable verification
    #[test]
    fn test_invalid_prob_var_verification() {
        let mut verifier = TypeVerifier::new();
        
        let operands = vec![];
        let results = vec![AetherMLIRType::ProbabilisticVariable {
            distribution: DistributionType::Normal { mean: 0.0, std: -1.0 }, // Invalid: negative std
            value_type: Box::new(MLIRType::Float { width: 64 }),
            inference_method: InferenceMethod::Exact,
        }];

        assert!(verifier.verify_operation_types("aether.prob_var", &operands, &results).is_err());
    }

    /// Test linear allocation verification
    #[test]
    fn test_linear_alloc_verification() {
        let mut verifier = TypeVerifier::new();
        
        let operands = vec![];
        let results = vec![AetherMLIRType::LinearType {
            inner_type: Box::new(MLIRType::Float { width: 32 }),
            ownership_info: LinearOwnershipInfo {
                ownership: LinearOwnership::Owned,
                allocation_site: Some("test".to_string()),
                deallocation_site: None,
                move_semantics: MoveSemanticsInfo {
                    is_movable: true,
                    is_copyable: false,
                    requires_explicit_drop: true,
                    gpu_resource: false,
                },
            },
            lifetime_info: None,
        }];

        assert!(verifier.verify_operation_types("aether.linear_alloc", &operands, &results).is_ok());
    }

    /// Test constraint checker
    #[test]
    fn test_constraint_checker() {
        let mut checker = TypeConstraintChecker::new();
        
        // Add shape constraint
        checker.add_constraint(TypeConstraint::ShapeConstraint(
            "tensor1".to_string(),
            ShapeConstraintKind::Positive,
        ));
        
        // Create type bindings
        let mut bindings = HashMap::new();
        bindings.insert("tensor1".to_string(), AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4], // All positive dimensions
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        });
        
        // Check constraints should pass
        assert!(checker.check_constraints(&bindings).is_ok());
        
        // Test with invalid shape
        bindings.insert("tensor1".to_string(), AetherMLIRType::Tensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, -3, 4], // Negative dimension
            is_differentiable: false,
            device: "cpu".to_string(),
            memory_layout: TensorMemoryLayout::RowMajor,
        });
        
        // Check constraints should fail
        assert!(checker.check_constraints(&bindings).is_err());
    }

    /// Test distribution types
    #[test]
    fn test_distribution_types() {
        // Test all distribution variants
        let distributions = vec![
            DistributionType::Normal { mean: 0.0, std: 1.0 },
            DistributionType::Uniform { min: 0.0, max: 1.0 },
            DistributionType::Bernoulli { p: 0.5 },
            DistributionType::Categorical { probs: vec![0.3, 0.3, 0.4] },
            DistributionType::Beta { alpha: 2.0, beta: 3.0 },
            DistributionType::Gamma { shape: 2.0, rate: 1.0 },
            DistributionType::Exponential { rate: 1.0 },
            DistributionType::Poisson { lambda: 2.0 },
            DistributionType::Dirichlet { alpha: vec![1.0, 1.0, 1.0] },
            DistributionType::Custom { 
                name: "CustomDist".to_string(), 
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("param1".to_string(), 1.0);
                    params
                }
            },
        ];

        // Verify all distributions can be created
        for dist in distributions {
            let prob_type = AetherMLIRType::ProbabilisticVariable {
                distribution: dist,
                value_type: Box::new(MLIRType::Float { width: 32 }),
                inference_method: InferenceMethod::Exact,
            };
            
            // Should not panic
            match prob_type {
                AetherMLIRType::ProbabilisticVariable { .. } => {}
                _ => panic!("Expected probabilistic type"),
            }
        }
    }

    /// Test memory layouts
    #[test]
    fn test_memory_layouts() {
        let layouts = vec![
            TensorMemoryLayout::RowMajor,
            TensorMemoryLayout::ColumnMajor,
            TensorMemoryLayout::Strided(vec![1, 10, 100]),
            TensorMemoryLayout::Blocked { block_sizes: vec![8, 8] },
            TensorMemoryLayout::GpuOptimized,
        ];

        for layout in layouts {
            let tensor_type = AetherMLIRType::Tensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![10, 10],
                is_differentiable: false,
                device: "cpu".to_string(),
                memory_layout: layout,
            };
            
            // Should not panic
            match tensor_type {
                AetherMLIRType::Tensor { .. } => {}
                _ => panic!("Expected tensor type"),
            }
        }
    }

    /// Test optimization hints
    #[test]
    fn test_optimization_hints() {
        let hints = vec![
            OptimizationHint::Inline,
            OptimizationHint::NoInline,
            OptimizationHint::Vectorize,
            OptimizationHint::Parallel,
            OptimizationHint::Unroll,
            OptimizationHint::VectorizeWidth(8),
            OptimizationHint::TargetCpu("x86_64".to_string()),
            OptimizationHint::TargetFeatures(vec!["avx2".to_string(), "fma".to_string()]),
        ];

        let func_type = AetherMLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Float { width: 32 }],
            attributes: AetherFunctionAttributes {
                is_differentiable: false,
                is_gpu_kernel: false,
                is_actor_method: false,
                is_pure: true,
                optimization_hints: hints,
                calling_convention: CallingConvention::C,
            },
        };

        match func_type {
            AetherMLIRType::Function { attributes, .. } => {
                assert_eq!(attributes.optimization_hints.len(), 8);
            }
            _ => panic!("Expected function type"),
        }
    }

    /// Test calling conventions
    #[test]
    fn test_calling_conventions() {
        let conventions = vec![
            CallingConvention::C,
            CallingConvention::Fast,
            CallingConvention::GpuKernel,
            CallingConvention::ActorMethod,
            CallingConvention::Custom("custom_cc".to_string()),
        ];

        for cc in conventions {
            let func_type = AetherMLIRType::Function {
                inputs: vec![],
                outputs: vec![],
                attributes: AetherFunctionAttributes {
                    is_differentiable: false,
                    is_gpu_kernel: false,
                    is_actor_method: false,
                    is_pure: true,
                    optimization_hints: vec![],
                    calling_convention: cc,
                },
            };
            
            // Should not panic
            match func_type {
                AetherMLIRType::Function { .. } => {}
                _ => panic!("Expected function type"),
            }
        }
    }

    /// Test error cases for type conversion
    #[test]
    fn test_type_conversion_errors() {
        let mut converter = AetherTypeConverter::new();
        
        // Test union with no variants
        let empty_union = AetherMLIRType::Union {
            name: "Empty".to_string(),
            variants: vec![],
            tag_type: MLIRType::Integer { width: 8, signed: false },
        };
        
        assert!(converter.convert_to_mlir(&empty_union).is_err());
        
        // Test struct with no fields
        let empty_struct = AetherMLIRType::Struct {
            name: "Empty".to_string(),
            fields: HashMap::new(),
            is_packed: false,
        };
        
        assert!(converter.convert_to_mlir(&empty_struct).is_err());
    }

    /// Test constraint verification edge cases
    #[test]
    fn test_constraint_verification_edge_cases() {
        let mut converter = AetherTypeConverter::new();
        
        // Test invalid shape constraint
        let invalid_constraint = TypeConstraint::ShapeConstraint(
            "test".to_string(),
            ShapeConstraintKind::Equal(-1), // Negative value
        );
        
        converter.add_constraint(invalid_constraint);
        assert!(converter.verify_constraints().is_err());
        
        // Test invalid range constraint
        converter.clear_cache();
        let invalid_range = TypeConstraint::IntConstraint(
            "test".to_string(),
            IntConstraintKind::Range(10, 5), // min > max
        );
        
        let mut converter2 = AetherTypeConverter::new();
        converter2.add_constraint(invalid_range);
        assert!(converter2.verify_constraints().is_err());
    }
}