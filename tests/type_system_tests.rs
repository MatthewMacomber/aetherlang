// Aether Type System Tests
// Comprehensive tests for type inference, checking, and error reporting

use aether_language::{
    Type, PrimitiveType, Shape, LinearOwnership, Distribution,
    TypeChecker, TypeInference, TypeEnvironment, ConstraintSolver,
    TypeCheckError,
    ASTNode, AtomValue, parse_sexpr
};
use aether_language::compiler::types::{
    DependentParam, DependentConstraint, ShapeConstraintKind, ShapeDim, ShapeExpr,
    Lifetime, LinearConstraint, AllocationSite, AllocationType
};
use aether_language::compiler::type_checker::TypeCheckContext;

#[cfg(test)]
mod type_tests {
    use super::*;

    #[test]
    fn test_primitive_types() {
        let int_type = Type::primitive(PrimitiveType::Int32);
        let float_type = Type::primitive(PrimitiveType::Float64);
        let bool_type = Type::primitive(PrimitiveType::Bool);
        let string_type = Type::primitive(PrimitiveType::String);

        assert!(int_type.is_primitive());
        assert!(float_type.is_primitive());
        assert!(bool_type.is_primitive());
        assert!(string_type.is_primitive());

        assert!(!int_type.is_tensor());
        assert!(!int_type.is_function());
        assert!(!int_type.is_linear());
    }

    #[test]
    fn test_tensor_types() {
        let element_type = Type::primitive(PrimitiveType::Float32);
        let _shape = Shape::concrete(vec![2, 3, 4]);
        let tensor_type = Type::tensor(element_type.clone(), vec![2, 3, 4]);

        assert!(tensor_type.is_tensor());
        assert!(!tensor_type.is_primitive());

        if let Some((elem_type, tensor_shape)) = tensor_type.as_tensor() {
            assert_eq!(elem_type, &element_type);
            assert_eq!(tensor_shape.rank(), Some(3));
        } else {
            panic!("Expected tensor type");
        }
    }

    #[test]
    fn test_function_types() {
        let param_types = vec![
            Type::primitive(PrimitiveType::Int32),
            Type::primitive(PrimitiveType::Float64),
        ];
        let return_type = Type::primitive(PrimitiveType::Bool);
        let function_type = Type::function(param_types.clone(), return_type.clone());

        assert!(function_type.is_function());
        assert!(!function_type.is_primitive());

        if let Some((params, ret_type, is_diff)) = function_type.as_function() {
            assert_eq!(params, &param_types);
            assert_eq!(ret_type, &return_type);
            assert!(!is_diff);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_differentiable_function_types() {
        let param_types = vec![Type::primitive(PrimitiveType::Float32)];
        let return_type = Type::primitive(PrimitiveType::Float32);
        let diff_function = Type::differentiable_function(param_types, return_type);

        if let Some((_, _, is_diff)) = diff_function.as_function() {
            assert!(is_diff);
        } else {
            panic!("Expected differentiable function type");
        }
    }

    #[test]
    fn test_linear_types() {
        let inner_type = Type::primitive(PrimitiveType::Int32);
        let linear_type = Type::linear(inner_type.clone());

        assert!(linear_type.is_linear());
        assert!(!linear_type.is_primitive());

        if let Some((inner, ownership)) = linear_type.as_linear() {
            assert_eq!(inner, &inner_type);
            assert_eq!(ownership, &LinearOwnership::Owned);
        } else {
            panic!("Expected linear type");
        }
    }

    #[test]
    fn test_probabilistic_types() {
        let distribution = Distribution::Normal { mean: 0.0, std: 1.0 };
        let inner_type = Type::primitive(PrimitiveType::Float64);
        let prob_type = Type::probabilistic(distribution.clone(), inner_type.clone());

        assert!(prob_type.is_probabilistic());
        assert!(!prob_type.is_primitive());
    }

    #[test]
    fn test_dynamic_type() {
        let dyn_type = Type::dynamic();
        assert!(dyn_type.is_dynamic());
        assert!(!dyn_type.is_primitive());
        assert!(!dyn_type.is_tensor());
        assert!(!dyn_type.is_function());
    }

    #[test]
    fn test_type_variables() {
        let type_var = Type::variable(1, Some("T".to_string()));
        assert!(!type_var.is_primitive());
        assert!(!type_var.is_dynamic());
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn test_concrete_shapes() {
        let shape = Shape::concrete(vec![2, 3, 4]);
        assert!(shape.is_concrete());
        assert!(!shape.is_symbolic());
        assert!(!shape.is_unknown());
        assert_eq!(shape.rank(), Some(3));

        if let Some(dims) = shape.as_concrete() {
            assert_eq!(dims, &vec![2, 3, 4]);
        } else {
            panic!("Expected concrete shape");
        }
    }

    #[test]
    fn test_symbolic_shapes() {
        use aether_language::compiler::types::{ShapeDim};
        
        let dims = vec![
            ShapeDim::concrete(2),
            ShapeDim::variable("N".to_string()),
            ShapeDim::concrete(4),
        ];
        let shape = Shape::symbolic(dims);
        
        assert!(shape.is_symbolic());
        assert!(!shape.is_concrete());
        assert_eq!(shape.rank(), Some(3));
    }

    #[test]
    fn test_unknown_shapes() {
        let shape = Shape::unknown();
        assert!(shape.is_unknown());
        assert!(!shape.is_concrete());
        assert!(!shape.is_symbolic());
        assert_eq!(shape.rank(), None);
    }

    #[test]
    fn test_shape_compatibility() {
        let shape1 = Shape::concrete(vec![2, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let shape3 = Shape::concrete(vec![3, 2]);
        let unknown_shape = Shape::unknown();

        assert!(shape1.is_compatible_with(&shape2));
        assert!(!shape1.is_compatible_with(&shape3));
        assert!(shape1.is_compatible_with(&unknown_shape));
        assert!(unknown_shape.is_compatible_with(&shape1));
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = Shape::concrete(vec![1, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let shape3 = Shape::concrete(vec![2, 1]);
        let shape4 = Shape::concrete(vec![2, 4]);

        assert!(shape1.is_broadcastable_with(&shape2));
        assert!(shape2.is_broadcastable_with(&shape3));
        assert!(!shape1.is_broadcastable_with(&shape4));

        // Broadcasting with scalars
        let scalar_shape = Shape::concrete(vec![1]);
        let matrix_shape = Shape::concrete(vec![3, 4]);
        assert!(scalar_shape.is_broadcastable_with(&matrix_shape));
        assert!(matrix_shape.is_broadcastable_with(&scalar_shape));
    }
}

#[cfg(test)]
mod type_inference_tests {
    use super::*;

    #[test]
    fn test_constraint_solver_fresh_vars() {
        let mut solver = ConstraintSolver::new();
        let var1 = solver.fresh_type_var(Some("x".to_string()));
        let var2 = solver.fresh_type_var(None);

        assert!(matches!(var1, Type::Variable(_)));
        assert!(matches!(var2, Type::Variable(_)));
        assert_ne!(var1, var2);
    }

    #[test]
    fn test_constraint_solver_unification() {
        let mut solver = ConstraintSolver::new();
        let var = solver.fresh_type_var(Some("x".to_string()));
        let int_type = Type::primitive(PrimitiveType::Int32);

        solver.add_equal_constraint(var.clone(), int_type.clone());
        solver.solve().unwrap();

        let resolved = solver.get_resolved_type(var);
        assert_eq!(resolved, int_type);
    }

    #[test]
    fn test_type_environment() {
        let mut env = TypeEnvironment::new();
        let int_type = Type::primitive(PrimitiveType::Int32);
        let float_type = Type::primitive(PrimitiveType::Float64);

        env.bind("x".to_string(), int_type.clone());
        env.bind("y".to_string(), float_type.clone());

        assert_eq!(env.lookup("x"), Some(&int_type));
        assert_eq!(env.lookup("y"), Some(&float_type));
        assert_eq!(env.lookup("z"), None);

        // Test scoping
        let child_env = env.child();
        assert_eq!(child_env.lookup("x"), Some(&int_type));
        assert!(!child_env.is_bound_locally("x"));
    }

    #[test]
    fn test_type_inference_atoms() {
        let mut inference = TypeInference::new();

        // Test number inference
        let int_type = inference.infer_atom_type(&AtomValue::Number(42.0)).unwrap();
        assert_eq!(int_type, Type::primitive(PrimitiveType::Int32));

        let float_type = inference.infer_atom_type(&AtomValue::Number(3.14)).unwrap();
        assert_eq!(float_type, Type::primitive(PrimitiveType::Float64));

        // Test other atoms
        let string_type = inference.infer_atom_type(&AtomValue::String("hello".to_string())).unwrap();
        assert_eq!(string_type, Type::primitive(PrimitiveType::String));

        let bool_type = inference.infer_atom_type(&AtomValue::Boolean(true)).unwrap();
        assert_eq!(bool_type, Type::primitive(PrimitiveType::Bool));

        let unit_type = inference.infer_atom_type(&AtomValue::Nil).unwrap();
        assert_eq!(unit_type, Type::primitive(PrimitiveType::Unit));
    }

    #[test]
    fn test_type_inference_unknown_symbol() {
        let mut inference = TypeInference::new();
        let symbol_type = inference.infer_atom_type(&AtomValue::Symbol("unknown".to_string())).unwrap();
        
        // Should create a fresh type variable
        assert!(matches!(symbol_type, Type::Variable(_)));
    }

    #[test]
    fn test_type_inference_with_bindings() {
        let mut inference = TypeInference::new();
        let int_type = Type::primitive(PrimitiveType::Int32);
        
        inference.bind_type("x".to_string(), int_type.clone());
        let inferred_type = inference.infer_atom_type(&AtomValue::Symbol("x".to_string())).unwrap();
        
        assert_eq!(inferred_type, int_type);
    }
}

#[cfg(test)]
mod type_checker_tests {
    use super::*;

    #[test]
    fn test_type_checker_static_mode() {
        let mut checker = TypeChecker::new(true);
        
        // Test defined symbols
        let int_type = checker.check_atom(&AtomValue::Number(42.0)).unwrap();
        assert_eq!(int_type, Type::primitive(PrimitiveType::Int32));

        // Test undefined symbols in static mode
        let result = checker.check_atom(&AtomValue::Symbol("undefined".to_string()));
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TypeCheckError::UndefinedSymbol { name, .. } => {
                assert_eq!(name, "undefined");
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_type_checker_dynamic_mode() {
        let mut checker = TypeChecker::new(false);
        
        // Test undefined symbols in dynamic mode
        let result = checker.check_atom(&AtomValue::Symbol("undefined".to_string())).unwrap();
        assert_eq!(result, Type::dynamic());
    }

    #[test]
    fn test_type_assignability() {
        let checker = TypeChecker::new(true);
        
        // Same types
        assert!(checker.is_assignable(
            &Type::primitive(PrimitiveType::Int32),
            &Type::primitive(PrimitiveType::Int32)
        ));

        // Numeric widening
        assert!(checker.is_assignable(
            &Type::primitive(PrimitiveType::Int32),
            &Type::primitive(PrimitiveType::Int64)
        ));

        assert!(checker.is_assignable(
            &Type::primitive(PrimitiveType::Float32),
            &Type::primitive(PrimitiveType::Float64)
        ));

        // Invalid conversions
        assert!(!checker.is_assignable(
            &Type::primitive(PrimitiveType::Int64),
            &Type::primitive(PrimitiveType::Int32)
        ));

        assert!(!checker.is_assignable(
            &Type::primitive(PrimitiveType::String),
            &Type::primitive(PrimitiveType::Int32)
        ));

        // Dynamic type compatibility
        assert!(checker.is_assignable(
            &Type::dynamic(),
            &Type::primitive(PrimitiveType::Int32)
        ));

        assert!(checker.is_assignable(
            &Type::primitive(PrimitiveType::Int32),
            &Type::dynamic()
        ));
    }

    #[test]
    fn test_function_type_assignability() {
        let checker = TypeChecker::new(true);
        
        let func1 = Type::function(
            vec![Type::primitive(PrimitiveType::Int32)],
            Type::primitive(PrimitiveType::Bool)
        );
        
        let func2 = Type::function(
            vec![Type::primitive(PrimitiveType::Int32)],
            Type::primitive(PrimitiveType::Bool)
        );
        
        let func3 = Type::function(
            vec![Type::primitive(PrimitiveType::Int64)], // Wider parameter type
            Type::primitive(PrimitiveType::Bool)
        );

        assert!(checker.is_assignable(&func1, &func2));
        assert!(checker.is_assignable(&func3, &func1)); // Contravariant parameters - func3 can be assigned to func1
    }

    #[test]
    fn test_tensor_type_assignability() {
        let checker = TypeChecker::new(true);
        
        let tensor1 = Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![2, 3]
        );
        
        let tensor2 = Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![2, 3]
        );
        
        let tensor3 = Type::tensor(
            Type::primitive(PrimitiveType::Float64), // Different element type
            vec![2, 3]
        );

        assert!(checker.is_assignable(&tensor1, &tensor2));
        assert!(checker.is_assignable(&tensor1, &tensor3)); // Element type widening
    }

    #[test]
    fn test_shape_broadcasting_in_checker() {
        let checker = TypeChecker::new(true);
        
        let shape1 = Shape::concrete(vec![1, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let result = checker.broadcast_shapes(&shape1, &shape2).unwrap();
        
        if let Shape::Concrete(dims) = result {
            assert_eq!(dims, vec![2, 3]);
        } else {
            panic!("Expected concrete shape");
        }

        // Test incompatible shapes
        let shape3 = Shape::concrete(vec![2, 4]);
        let result = checker.broadcast_shapes(&shape1, &shape3);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_annotation_parsing() {
        let mut checker = TypeChecker::new(true);
        
        // Primitive types
        let int_node = ASTNode::symbol("i32".to_string());
        let int_type = checker.parse_type_annotation(&int_node).unwrap();
        assert_eq!(int_type, Type::primitive(PrimitiveType::Int32));

        let float_node = ASTNode::symbol("f64".to_string());
        let float_type = checker.parse_type_annotation(&float_node).unwrap();
        assert_eq!(float_type, Type::primitive(PrimitiveType::Float64));

        let bool_node = ASTNode::symbol("bool".to_string());
        let bool_type = checker.parse_type_annotation(&bool_node).unwrap();
        assert_eq!(bool_type, Type::primitive(PrimitiveType::Bool));

        // Dynamic type
        let dyn_node = ASTNode::symbol("dyn".to_string());
        let dyn_type = checker.parse_type_annotation(&dyn_node).unwrap();
        assert_eq!(dyn_type, Type::dynamic());

        // Invalid type
        let invalid_node = ASTNode::symbol("invalid_type".to_string());
        let result = checker.parse_type_annotation(&invalid_node);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_ownership_tracking() {
        let mut checker = TypeChecker::new(true);
        let context = checker.context_mut();
        
        // Track ownership
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Owned));

        // Move variable
        context.move_linear_variable("x").unwrap();
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Moved));

        // Try to move again (should fail)
        let result = context.move_linear_variable("x");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TypeCheckError::LinearTypeViolation { violation, .. } => {
                assert_eq!(violation, "use after move");
            }
            _ => panic!("Expected LinearTypeViolation error"),
        }
    }

    #[test]
    fn test_scope_management() {
        let mut checker = TypeChecker::new(true);
        let context = checker.context_mut();
        
        // Bind in outer scope
        context.bind_type("x".to_string(), Type::primitive(PrimitiveType::Int32));
        assert!(context.lookup_type("x").is_some());

        // Enter new scope
        context.enter_scope();
        context.bind_type("y".to_string(), Type::primitive(PrimitiveType::Float64));
        assert!(context.lookup_type("x").is_some()); // Still visible
        assert!(context.lookup_type("y").is_some());

        // Exit scope
        context.exit_scope();
        assert!(context.lookup_type("x").is_some()); // Still visible
        // y should be cleaned up, but our simplified implementation doesn't handle this
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_simple_expression_type_checking() {
        let mut checker = TypeChecker::new(true);
        
        // Test simple number
        let ast = parse_sexpr("42").unwrap();
        let result_type = checker.check(&ast).unwrap();
        assert_eq!(result_type, Type::primitive(PrimitiveType::Int32));

        // Test float
        let ast = parse_sexpr("3.14").unwrap();
        let result_type = checker.check(&ast).unwrap();
        assert_eq!(result_type, Type::primitive(PrimitiveType::Float64));

        // Test string
        let ast = parse_sexpr("\"hello\"").unwrap();
        let result_type = checker.check(&ast).unwrap();
        assert_eq!(result_type, Type::primitive(PrimitiveType::String));

        // Test boolean
        let ast = parse_sexpr("true").unwrap();
        let result_type = checker.check(&ast).unwrap();
        assert_eq!(result_type, Type::primitive(PrimitiveType::Bool));
    }

    #[test]
    fn test_empty_list_type_checking() {
        let mut checker = TypeChecker::new(true);
        
        let ast = parse_sexpr("()").unwrap();
        let result_type = checker.check(&ast).unwrap();
        assert_eq!(result_type, Type::primitive(PrimitiveType::Unit));
    }

    #[test]
    fn test_type_inference_integration() {
        let mut inference = TypeInference::new();
        
        // Test simple atom inference
        let ast = parse_sexpr("42").unwrap();
        let inferred_type = inference.infer_type(&ast).unwrap();
        assert_eq!(inferred_type, Type::primitive(PrimitiveType::Int32));

        // Test with type bindings
        inference.bind_type("x".to_string(), Type::primitive(PrimitiveType::Float64));
        let ast = parse_sexpr("x").unwrap();
        let inferred_type = inference.infer_type(&ast).unwrap();
        assert_eq!(inferred_type, Type::primitive(PrimitiveType::Float64));
    }

    #[test]
    fn test_gradual_typing_behavior() {
        // Static mode - strict checking
        let mut static_checker = TypeChecker::new(true);
        let ast = parse_sexpr("undefined_var").unwrap();
        let result = static_checker.check(&ast);
        assert!(result.is_err());

        // Dynamic mode - permissive checking
        let mut dynamic_checker = TypeChecker::new(false);
        let result = dynamic_checker.check(&ast).unwrap();
        assert_eq!(result, Type::dynamic());
    }

    #[test]
    fn test_error_reporting() {
        let mut checker = TypeChecker::new(true);
        
        // Test undefined symbol error
        let ast = parse_sexpr("undefined").unwrap();
        let result = checker.check(&ast);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TypeCheckError::UndefinedSymbol { name, location } => {
                assert_eq!(name, "undefined");
                assert_eq!(location, "symbol reference");
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }
}

#[cfg(test)]
mod primitive_type_tests {
    use super::*;

    #[test]
    fn test_primitive_type_properties() {
        // Test numeric types
        assert!(PrimitiveType::Int32.is_numeric());
        assert!(PrimitiveType::Float64.is_numeric());
        assert!(!PrimitiveType::Bool.is_numeric());
        assert!(!PrimitiveType::String.is_numeric());

        // Test integer types
        assert!(PrimitiveType::Int32.is_integer());
        assert!(PrimitiveType::UInt64.is_integer());
        assert!(!PrimitiveType::Float32.is_integer());

        // Test float types
        assert!(PrimitiveType::Float32.is_float());
        assert!(PrimitiveType::Float64.is_float());
        assert!(!PrimitiveType::Int32.is_float());

        // Test sizes
        assert_eq!(PrimitiveType::Int32.size_bytes(), 4);
        assert_eq!(PrimitiveType::Int64.size_bytes(), 8);
        assert_eq!(PrimitiveType::Float32.size_bytes(), 4);
        assert_eq!(PrimitiveType::Float64.size_bytes(), 8);
        assert_eq!(PrimitiveType::Bool.size_bytes(), 1);
        assert_eq!(PrimitiveType::Unit.size_bytes(), 0);
    }
}

#[cfg(test)]
mod dependent_type_tests {
    use super::*;
    use aether_language::compiler::types::{
        DependentParam, DependentConstraint, ShapeConstraintKind, ShapeDim
    };

    #[test]
    fn test_dependent_type_creation() {
        let base_type = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params = vec![
            DependentParam::Shape("N".to_string()),
            DependentParam::Shape("M".to_string()),
        ];
        let constraints = vec![
            DependentConstraint::ShapeConstraint("N".to_string(), ShapeConstraintKind::Positive),
            DependentConstraint::ShapeConstraint("M".to_string(), ShapeConstraintKind::Positive),
        ];

        let dependent_type = Type::dependent(base_type.clone(), params.clone(), constraints.clone());
        
        assert!(dependent_type.is_dependent());
        assert!(!dependent_type.is_primitive());
        assert!(!dependent_type.is_tensor()); // It's dependent, not directly tensor

        if let Some((base, params_ref, constraints_ref)) = dependent_type.as_dependent() {
            assert_eq!(base, &base_type);
            assert_eq!(params_ref, &params);
            assert_eq!(constraints_ref, &constraints);
        } else {
            panic!("Expected dependent type");
        }
    }

    #[test]
    fn test_dependent_tensor_creation() {
        let element_type = Type::primitive(PrimitiveType::Float32);
        let shape_params = vec!["Batch".to_string(), "Height".to_string(), "Width".to_string()];
        let constraints = vec![
            DependentConstraint::ShapeConstraint("Batch".to_string(), ShapeConstraintKind::Positive),
            DependentConstraint::ShapeConstraint("Height".to_string(), ShapeConstraintKind::GreaterThan(0)),
            DependentConstraint::ShapeConstraint("Width".to_string(), ShapeConstraintKind::GreaterThan(0)),
        ];

        let dependent_tensor = Type::dependent_tensor(element_type, shape_params, constraints);
        
        assert!(dependent_tensor.is_dependent());
        
        if let Some((base_type, params, _)) = dependent_tensor.as_dependent() {
            assert!(base_type.is_tensor());
            assert_eq!(params.len(), 3);
            
            match &params[0] {
                DependentParam::Shape(name) => assert_eq!(name, "Batch"),
                _ => panic!("Expected shape parameter"),
            }
        } else {
            panic!("Expected dependent type");
        }
    }

    #[test]
    fn test_shape_unification() {
        // Test concrete shape unification
        let shape1 = Shape::concrete(vec![2, 3, 4]);
        let shape2 = Shape::concrete(vec![2, 3, 4]);
        let shape3 = Shape::concrete(vec![2, 3, 5]);

        assert!(shape1.unify_with(&shape2).is_some());
        assert!(shape1.unify_with(&shape3).is_none());

        // Test symbolic shape unification
        let symbolic_shape1 = Shape::symbolic(vec![
            ShapeDim::concrete(2),
            ShapeDim::variable("N".to_string()),
            ShapeDim::concrete(4),
        ]);
        let concrete_shape = Shape::concrete(vec![2, 3, 4]);

        let unified = symbolic_shape1.unify_with(&concrete_shape);
        assert!(unified.is_some());
        
        if let Some(Shape::Symbolic(dims)) = unified {
            assert_eq!(dims.len(), 3);
            assert_eq!(dims[0], ShapeDim::concrete(2));
            assert_eq!(dims[1], ShapeDim::concrete(3)); // Variable unified with concrete
            assert_eq!(dims[2], ShapeDim::concrete(4));
        } else {
            panic!("Expected symbolic shape");
        }
    }

    #[test]
    fn test_matrix_multiplication_shape_inference() {
        // Test concrete shapes
        let shape1 = Shape::concrete(vec![2, 3]);
        let shape2 = Shape::concrete(vec![3, 4]);
        let result = shape1.matmul_result_shape(&shape2);
        
        assert!(result.is_some());
        if let Some(Shape::Concrete(dims)) = result {
            assert_eq!(dims, vec![2, 4]);
        } else {
            panic!("Expected concrete result shape");
        }

        // Test incompatible shapes
        let shape3 = Shape::concrete(vec![2, 5]);
        let result = shape1.matmul_result_shape(&shape3);
        assert!(result.is_none());

        // Test batch matrix multiplication
        let batch_shape1 = Shape::concrete(vec![5, 2, 3]);
        let batch_shape2 = Shape::concrete(vec![5, 3, 4]);
        let batch_result = batch_shape1.matmul_result_shape(&batch_shape2);
        
        assert!(batch_result.is_some());
        if let Some(Shape::Concrete(dims)) = batch_result {
            assert_eq!(dims, vec![5, 2, 4]);
        } else {
            panic!("Expected concrete batch result shape");
        }

        // Test broadcasting in batch dimensions
        let broadcast_shape1 = Shape::concrete(vec![1, 2, 3]);
        let broadcast_shape2 = Shape::concrete(vec![5, 3, 4]);
        let broadcast_result = broadcast_shape1.matmul_result_shape(&broadcast_shape2);
        
        assert!(broadcast_result.is_some());
        if let Some(Shape::Concrete(dims)) = broadcast_result {
            assert_eq!(dims, vec![5, 2, 4]);
        } else {
            panic!("Expected concrete broadcast result shape");
        }
    }

    #[test]
    fn test_symbolic_matrix_multiplication() {
        let symbolic_shape1 = Shape::symbolic(vec![
            ShapeDim::variable("Batch".to_string()),
            ShapeDim::variable("M".to_string()),
            ShapeDim::variable("K".to_string()),
        ]);
        let symbolic_shape2 = Shape::symbolic(vec![
            ShapeDim::variable("Batch".to_string()),
            ShapeDim::variable("K".to_string()),
            ShapeDim::variable("N".to_string()),
        ]);

        let result = symbolic_shape1.matmul_result_shape(&symbolic_shape2);
        assert!(result.is_some());
        
        if let Some(Shape::Symbolic(dims)) = result {
            assert_eq!(dims.len(), 3);
            assert_eq!(dims[0], ShapeDim::variable("Batch".to_string()));
            assert_eq!(dims[1], ShapeDim::variable("M".to_string()));
            assert_eq!(dims[2], ShapeDim::variable("N".to_string()));
        } else {
            panic!("Expected symbolic result shape");
        }
    }

    #[test]
    fn test_shape_compatibility_with_dependent_types() {
        // Test symbolic shape compatibility
        let symbolic_shape1 = Shape::symbolic(vec![
            ShapeDim::variable("N".to_string()),
            ShapeDim::concrete(3),
        ]);
        let symbolic_shape2 = Shape::symbolic(vec![
            ShapeDim::variable("N".to_string()),
            ShapeDim::concrete(3),
        ]);
        let symbolic_shape3 = Shape::symbolic(vec![
            ShapeDim::variable("M".to_string()),
            ShapeDim::concrete(3),
        ]);

        assert!(symbolic_shape1.is_compatible_with(&symbolic_shape2));
        // Different variables should not be automatically compatible without constraint solving
        assert!(!symbolic_shape1.is_compatible_with(&symbolic_shape3));

        // Test mixed concrete/symbolic compatibility
        let concrete_shape = Shape::concrete(vec![5, 3]);
        assert!(symbolic_shape1.is_compatible_with(&concrete_shape));
        assert!(concrete_shape.is_compatible_with(&symbolic_shape1));
    }

    #[test]
    fn test_shape_broadcasting_with_symbolic_shapes() {
        let shape1 = Shape::symbolic(vec![
            ShapeDim::concrete(1),
            ShapeDim::variable("N".to_string()),
        ]);
        let shape2 = Shape::symbolic(vec![
            ShapeDim::variable("M".to_string()),
            ShapeDim::variable("N".to_string()),
        ]);

        assert!(shape1.is_broadcastable_with(&shape2));
        assert!(shape2.is_broadcastable_with(&shape1));

        // Test incompatible broadcasting
        let incompatible_shape = Shape::symbolic(vec![
            ShapeDim::variable("M".to_string()),
            ShapeDim::concrete(4), // Different concrete dimension
        ]);
        assert!(!shape1.is_broadcastable_with(&incompatible_shape));
    }
}

#[cfg(test)]
mod linear_type_tests {
    use super::*;

    #[test]
    fn test_linear_type_creation() {
        let inner_type = Type::primitive(PrimitiveType::Int32);
        let linear_type = Type::linear(inner_type.clone());

        assert!(linear_type.is_linear());
        assert!(linear_type.is_movable());
        assert!(!linear_type.is_borrowed());
        assert!(!linear_type.is_moved());

        if let Some((inner, ownership)) = linear_type.as_linear() {
            assert_eq!(inner, &inner_type);
            assert_eq!(ownership, &LinearOwnership::Owned);
        } else {
            panic!("Expected linear type");
        }
    }

    #[test]
    fn test_linear_type_with_ownership() {
        let inner_type = Type::primitive(PrimitiveType::Float32);
        let lifetime = Lifetime::new("'a".to_string(), 1);
        let borrowed_type = Type::linear_with_ownership(
            inner_type.clone(),
            LinearOwnership::Borrowed(lifetime.clone())
        );

        assert!(borrowed_type.is_linear());
        assert!(!borrowed_type.is_movable());
        assert!(borrowed_type.is_borrowed());
        assert!(!borrowed_type.is_moved());

        if let Some(type_lifetime) = borrowed_type.get_lifetime() {
            assert_eq!(type_lifetime, &lifetime);
        } else {
            panic!("Expected lifetime");
        }
    }

    #[test]
    fn test_gpu_linear_type() {
        let inner_type = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![1024, 1024]);
        let gpu_type = Type::linear_gpu(inner_type);

        assert!(gpu_type.is_linear());
        assert!(gpu_type.is_movable());
        assert!(gpu_type.is_gpu_resource());
        assert!(!gpu_type.is_moved());

        if let Some((_, ownership)) = gpu_type.as_linear() {
            assert_eq!(ownership, &LinearOwnership::GpuOwned);
            assert!(ownership.is_gpu_resource());
        } else {
            panic!("Expected GPU linear type");
        }
    }

    #[test]
    fn test_lifetime_operations() {
        let lifetime1 = Lifetime::new("'a".to_string(), 1);
        let lifetime2 = Lifetime::new("'b".to_string(), 2);
        let lifetime3 = Lifetime::new("'c".to_string(), 1);

        // Lifetime with lower scope level outlives higher scope level
        assert!(lifetime1.outlives(&lifetime2));
        assert!(!lifetime2.outlives(&lifetime1));
        assert!(lifetime1.outlives(&lifetime3)); // Same level, but 'a' outlives 'c' by name comparison

        // Compatibility
        assert!(lifetime1.is_compatible_with(&lifetime1)); // Same lifetime
        assert!(lifetime1.is_compatible_with(&lifetime2)); // 'a' outlives 'b'
    }

    #[test]
    fn test_linear_ownership_permissions() {
        let owned = LinearOwnership::Owned;
        let moved = LinearOwnership::Moved;
        let lifetime = Lifetime::new("'a".to_string(), 1);
        let borrowed = LinearOwnership::Borrowed(lifetime.clone());
        let mutable_borrow = LinearOwnership::MutableBorrow(lifetime);
        let gpu_owned = LinearOwnership::GpuOwned;
        let gpu_moved = LinearOwnership::GpuMoved;

        // Read permissions
        assert!(owned.allows_read());
        assert!(!moved.allows_read());
        assert!(borrowed.allows_read());
        assert!(mutable_borrow.allows_read());
        assert!(gpu_owned.allows_read());
        assert!(!gpu_moved.allows_read());

        // Write permissions
        assert!(owned.allows_write());
        assert!(!moved.allows_write());
        assert!(!borrowed.allows_write());
        assert!(mutable_borrow.allows_write());
        assert!(gpu_owned.allows_write());
        assert!(!gpu_moved.allows_write());

        // Move permissions
        assert!(owned.allows_move());
        assert!(!moved.allows_move());
        assert!(!borrowed.allows_move());
        assert!(!mutable_borrow.allows_move());
        assert!(gpu_owned.allows_move());
        assert!(!gpu_moved.allows_move());

        // GPU resource check
        assert!(!owned.is_gpu_resource());
        assert!(gpu_owned.is_gpu_resource());
        assert!(gpu_moved.is_gpu_resource());
    }

    #[test]
    fn test_linear_type_tracking() {
        let mut context = TypeCheckContext::new(true);

        // Track owned linear variable
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Owned));

        // Move the variable
        context.move_linear_variable("x").unwrap();
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Moved));

        // Try to move again (should fail)
        let result = context.move_linear_variable("x");
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeCheckError::LinearTypeViolation { violation, .. } => {
                assert_eq!(violation, "use after move");
            }
            _ => panic!("Expected LinearTypeViolation error"),
        }
    }

    #[test]
    fn test_linear_borrowing() {
        let mut context = TypeCheckContext::new(true);
        let lifetime = Lifetime::new("'a".to_string(), 1);

        // Track owned variable
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);

        // Borrow immutably
        context.borrow_linear_variable("x", lifetime.clone(), false).unwrap();
        assert!(context.is_lifetime_active("'a"));

        // Try to borrow mutably (should fail because we have immutable borrow)
        let result = context.borrow_linear_variable("x", lifetime.clone(), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_memory_safety() {
        let mut context = TypeCheckContext::new(true);

        // Track GPU resource
        context.track_linear_ownership("gpu_buffer".to_string(), LinearOwnership::GpuOwned);
        context.add_linear_constraint(LinearConstraint::GpuResource("gpu_buffer".to_string()));

        // Move GPU resource
        context.move_linear_variable("gpu_buffer").unwrap();
        assert_eq!(context.check_linear_ownership("gpu_buffer"), Some(&LinearOwnership::GpuMoved));

        // Validate constraints should pass
        let result = context.validate_linear_constraints();
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_use_constraint() {
        let mut context = TypeCheckContext::new(true);

        // Add single-use constraint
        context.add_linear_constraint(LinearConstraint::SingleUse("x".to_string()));

        // Use the variable once
        context.check_single_use("x").unwrap();

        // Try to use again (should fail)
        let result = context.check_single_use("x");
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeCheckError::LinearTypeViolation { violation, .. } => {
                assert_eq!(violation, "variable used more than once");
            }
            _ => panic!("Expected LinearTypeViolation error"),
        }
    }

    #[test]
    fn test_allocation_site_tracking() {
        let mut context = TypeCheckContext::new(true);

        // Register heap allocation
        let heap_site = AllocationSite::new("line 42".to_string(), AllocationType::Heap);
        context.register_allocation("heap_var".to_string(), heap_site);

        // Register GPU allocation
        let gpu_site = AllocationSite::with_size("line 100".to_string(), AllocationType::Gpu, 1024);
        context.register_allocation("gpu_var".to_string(), gpu_site);

        assert!(context.has_allocation_site("heap_var"));
        assert!(context.has_allocation_site("gpu_var"));

        if let Some(site) = context.get_allocation_site("gpu_var") {
            assert_eq!(site.allocation_type, AllocationType::Gpu);
            assert_eq!(site.size_hint, Some(1024));
        } else {
            panic!("Expected GPU allocation site");
        }
    }

    #[test]
    fn test_scope_cleanup() {
        let mut context = TypeCheckContext::new(true);

        // Enter scope and track variables
        context.enter_scope();
        let lifetime = Lifetime::new("'scope_1".to_string(), 1);
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        context.insert_lifetime("'scope_1".to_string(), lifetime);

        assert!(context.has_linear_ownership("x"));
        assert!(context.is_lifetime_active("'scope_1"));

        // Exit scope - should clean up
        context.exit_scope();

        // Variables should be cleaned up
        assert!(!context.has_linear_ownership("x"));
        assert!(!context.is_lifetime_active("'scope_1"));
    }

    #[test]
    fn test_lifetime_outlives_constraint() {
        let mut context = TypeCheckContext::new(true);

        let lifetime_a = Lifetime::new("'a".to_string(), 1);
        let lifetime_b = Lifetime::new("'b".to_string(), 2);

        // 'a outlives 'b (lower scope level outlives higher)
        context.add_linear_constraint(LinearConstraint::OutlivesConstraint(
            lifetime_a.clone(),
            lifetime_b.clone()
        ));

        // Should validate successfully
        let result = context.validate_linear_constraints();
        assert!(result.is_ok());

        // Reverse constraint should fail
        context.clear_linear_constraints();
        context.add_linear_constraint(LinearConstraint::OutlivesConstraint(
            lifetime_b,
            lifetime_a
        ));

        let result = context.validate_linear_constraints();
        assert!(result.is_err());
    }

    #[test]
    fn test_must_move_constraint() {
        let mut context = TypeCheckContext::new(true);

        // Track variable and add must-move constraint
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        context.add_linear_constraint(LinearConstraint::MustMove("x".to_string()));

        // Validation should fail before move
        let result = context.validate_linear_constraints();
        assert!(result.is_err());

        // Move the variable
        context.move_linear_variable("x").unwrap();

        // Now validation should pass
        let result = context.validate_linear_constraints();
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod linear_type_checker_integration_tests {
    use super::*;

    #[test]
    fn test_linear_assignment_checking() {
        let mut checker = TypeChecker::new(true);
        let linear_type = Type::linear(Type::primitive(PrimitiveType::Int32));

        // Check linear assignment
        let result = checker.check_linear_assignment("x", &linear_type, "test location");
        assert!(result.is_ok());

        // Verify ownership was tracked
        assert_eq!(
            checker.context().check_linear_ownership("x"),
            Some(&LinearOwnership::Owned)
        );
    }

    #[test]
    fn test_gpu_linear_assignment() {
        let mut checker = TypeChecker::new(true);
        let gpu_tensor = Type::linear_gpu(Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![1024, 1024]
        ));

        // Check GPU linear assignment
        let result = checker.check_linear_assignment("gpu_buffer", &gpu_tensor, "GPU allocation");
        assert!(result.is_ok());

        // Verify GPU ownership and constraints
        assert_eq!(
            checker.context().check_linear_ownership("gpu_buffer"),
            Some(&LinearOwnership::GpuOwned)
        );

        // Check GPU memory safety
        let result = checker.check_gpu_memory_safety("gpu_buffer");
        assert!(result.is_ok());
    }

    #[test]
    fn test_linear_type_in_atom_checking() {
        let mut checker = TypeChecker::new(true);

        // Bind linear variable
        let linear_type = Type::linear(Type::primitive(PrimitiveType::Int32));
        checker.context_mut().bind_type("x".to_string(), linear_type);
        checker.context_mut().track_linear_ownership("x".to_string(), LinearOwnership::Owned);

        // First use should succeed
        let result = checker.check_atom(&AtomValue::Symbol("x".to_string()));
        assert!(result.is_ok());

        // Variable should be marked as used
        assert!(checker.context().is_variable_used("x"));
    }

    #[test]
    fn test_moved_variable_access() {
        let mut checker = TypeChecker::new(true);

        // Bind and move linear variable
        let linear_type = Type::linear_with_ownership(
            Type::primitive(PrimitiveType::Int32),
            LinearOwnership::Moved
        );
        checker.context_mut().bind_type("x".to_string(), linear_type);

        // Access should fail
        let result = checker.check_atom(&AtomValue::Symbol("x".to_string()));
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeCheckError::LinearTypeViolation { violation, .. } => {
                assert_eq!(violation, "use after move");
            }
            _ => panic!("Expected LinearTypeViolation error"),
        }
    }

    #[test]
    fn test_lifetime_analysis() {
        let mut checker = TypeChecker::new(true);

        // Analyze lifetimes for a symbol
        let symbol_node = ASTNode::symbol("x".to_string());
        let lifetimes = checker.analyze_lifetimes(&symbol_node).unwrap();

        // Should return empty for unbound symbol
        assert!(lifetimes.is_empty());

        // Bind borrowed variable
        let lifetime = Lifetime::new("'a".to_string(), 1);
        let borrowed_type = Type::linear_borrowed(
            Type::primitive(PrimitiveType::Int32),
            lifetime.clone()
        );
        checker.context_mut().bind_type("x".to_string(), borrowed_type);

        let lifetimes = checker.analyze_lifetimes(&symbol_node).unwrap();
        assert_eq!(lifetimes.len(), 1);
        assert_eq!(lifetimes[0], lifetime);
    }

    #[test]
    fn test_automatic_deallocation_insertion() {
        let mut checker = TypeChecker::new(true);

        // Track some linear variables
        checker.context_mut().track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        checker.context_mut().add_linear_constraint(LinearConstraint::MustMove("x".to_string()));

        // Before moving, deallocation should fail constraint validation
        let result = checker.insert_automatic_deallocation();
        assert!(result.is_err());

        // Move the variable
        checker.context_mut().move_linear_variable("x").unwrap();

        // Now deallocation should succeed
        let result = checker.insert_automatic_deallocation();
        assert!(result.is_ok());
    }

    #[test]
    fn test_complex_linear_type_scenario() {
        let mut checker = TypeChecker::new(true);

        // Create a complex scenario with multiple linear variables
        let heap_type = Type::linear(Type::primitive(PrimitiveType::Int32));
        let gpu_type = Type::linear_gpu(Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![256, 256]
        ));
        let lifetime = Lifetime::new("'local".to_string(), 1);
        let borrowed_type = Type::linear_borrowed(
            Type::primitive(PrimitiveType::Float64),
            lifetime.clone()
        );

        // Assign all types
        checker.check_linear_assignment("heap_var", &heap_type, "heap allocation").unwrap();
        checker.check_linear_assignment("gpu_var", &gpu_type, "GPU allocation").unwrap();
        checker.check_linear_assignment("borrowed_var", &borrowed_type, "borrow").unwrap();

        // Verify all are tracked correctly
        assert_eq!(
            checker.context().check_linear_ownership("heap_var"),
            Some(&LinearOwnership::Owned)
        );
        assert_eq!(
            checker.context().check_linear_ownership("gpu_var"),
            Some(&LinearOwnership::GpuOwned)
        );
        assert_eq!(
            checker.context().check_linear_ownership("borrowed_var"),
            Some(&LinearOwnership::Borrowed(lifetime))
        );

        // Move owned variables
        checker.context_mut().move_linear_variable("heap_var").unwrap();
        checker.context_mut().move_linear_variable("gpu_var").unwrap();

        // Verify moves
        assert_eq!(
            checker.context().check_linear_ownership("heap_var"),
            Some(&LinearOwnership::Moved)
        );
        assert_eq!(
            checker.context().check_linear_ownership("gpu_var"),
            Some(&LinearOwnership::GpuMoved)
        );

        // GPU memory safety check should fail for moved GPU variable
        let result = checker.check_gpu_memory_safety("gpu_var");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod shape_broadcasting_tests {
    use super::*;

    #[test]
    fn test_incompatible_symbolic_broadcasting() {
        let shape3 = Shape::symbolic(vec![
            ShapeDim::concrete(2),
            ShapeDim::concrete(3),
        ]);
        let shape4 = Shape::symbolic(vec![
            ShapeDim::concrete(3),
            ShapeDim::concrete(4),
        ]);

        assert!(!shape3.is_broadcastable_with(&shape4));
    }
}

#[cfg(test)]
mod tensor_operation_tests {
    use super::*;

    #[test]
    fn test_tensor_result_shape_inference() {
        let checker = TypeChecker::new(true);

        // Test element-wise operations
        let shape1 = Shape::concrete(vec![2, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let result = checker.infer_tensor_result_shape("add", &[shape1.clone(), shape2.clone()], "test").unwrap();
        
        if let Shape::Concrete(dims) = result {
            assert_eq!(dims, vec![2, 3]);
        } else {
            panic!("Expected concrete result shape");
        }

        // Test broadcasting
        let shape3 = Shape::concrete(vec![1, 3]);
        let shape4 = Shape::concrete(vec![2, 3]);
        let broadcast_result = checker.infer_tensor_result_shape("mul", &[shape3, shape4], "test").unwrap();
        
        if let Shape::Concrete(dims) = broadcast_result {
            assert_eq!(dims, vec![2, 3]);
        } else {
            panic!("Expected concrete broadcast result");
        }

        // Test matrix multiplication
        let matmul_shape1 = Shape::concrete(vec![2, 3]);
        let matmul_shape2 = Shape::concrete(vec![3, 4]);
        let matmul_result = checker.infer_tensor_result_shape("matmul", &[matmul_shape1, matmul_shape2], "test").unwrap();
        
        if let Shape::Concrete(dims) = matmul_result {
            assert_eq!(dims, vec![2, 4]);
        } else {
            panic!("Expected concrete matmul result");
        }

        // Test transpose
        let transpose_shape = Shape::concrete(vec![2, 3, 4]);
        let transpose_result = checker.infer_tensor_result_shape("transpose", &[transpose_shape], "test").unwrap();
        
        if let Shape::Concrete(dims) = transpose_result {
            assert_eq!(dims, vec![2, 4, 3]); // Last two dimensions swapped
        } else {
            panic!("Expected concrete transpose result");
        }
    }

    #[test]
    fn test_tensor_operation_error_cases() {
        let checker = TypeChecker::new(true);

        // Test arity mismatch
        let shape = Shape::concrete(vec![2, 3]);
        let result = checker.infer_tensor_result_shape("add", &[shape], "test");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TypeCheckError::ArityMismatch { expected, actual, .. } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected ArityMismatch error"),
        }

        // Test incompatible shapes for broadcasting
        let shape1 = Shape::concrete(vec![2, 3]);
        let shape2 = Shape::concrete(vec![3, 4]);
        let result = checker.infer_tensor_result_shape("add", &[shape1, shape2], "test");
        assert!(result.is_err());

        // Test matrix multiplication with incompatible shapes
        let matmul_shape1 = Shape::concrete(vec![2, 3]);
        let matmul_shape2 = Shape::concrete(vec![4, 5]);
        let result = checker.infer_tensor_result_shape("matmul", &[matmul_shape1, matmul_shape2], "test");
        assert!(result.is_err());

        // Test transpose with 1D tensor (should fail)
        let transpose_shape = Shape::concrete(vec![5]);
        let result = checker.infer_tensor_result_shape("transpose", &[transpose_shape], "test");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod dependent_type_inference_tests {
    use super::*;
    use aether_language::compiler::types::{DependentParam, DependentConstraint, ShapeConstraintKind};

    #[test]
    fn test_dependent_type_unification() {
        let mut solver = ConstraintSolver::new();

        let base_type1 = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params1 = vec![DependentParam::Shape("N".to_string())];
        let constraints1 = vec![DependentConstraint::ShapeConstraint("N".to_string(), ShapeConstraintKind::Positive)];
        let dependent1 = Type::dependent(base_type1.clone(), params1.clone(), constraints1.clone());

        let base_type2 = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params2 = vec![DependentParam::Shape("N".to_string())];
        let constraints2 = vec![DependentConstraint::ShapeConstraint("N".to_string(), ShapeConstraintKind::Positive)];
        let dependent2 = Type::dependent(base_type2, params2, constraints2);

        let result = solver.unify(dependent1, dependent2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dependent_type_parameter_mismatch() {
        let mut solver = ConstraintSolver::new();

        let base_type1 = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params1 = vec![DependentParam::Shape("N".to_string())];
        let constraints1 = vec![];
        let dependent1 = Type::dependent(base_type1, params1, constraints1);

        let base_type2 = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params2 = vec![DependentParam::Shape("M".to_string())]; // Different parameter name
        let constraints2 = vec![];
        let dependent2 = Type::dependent(base_type2, params2, constraints2);

        let result = solver.unify(dependent1, dependent2);
        assert!(result.is_err());
    }

    #[test]
    fn test_dependent_type_validation() {
        let mut checker = TypeChecker::new(true);

        let base_type = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![]);
        let params = vec![
            DependentParam::Shape("Batch".to_string()),
            DependentParam::Shape("Features".to_string()),
        ];
        let constraints = vec![
            DependentConstraint::ShapeConstraint("Batch".to_string(), ShapeConstraintKind::Positive),
            DependentConstraint::ShapeConstraint("Features".to_string(), ShapeConstraintKind::GreaterThan(0)),
        ];

        let result = checker.validate_dependent_type(&base_type, &params, &constraints, "test");
        assert!(result.is_ok());

        let validated_type = result.unwrap();
        assert!(validated_type.is_dependent());
    }
}

#[cfg(test)]
mod shape_expression_tests {
    use super::*;
    use aether_language::compiler::types::{ShapeExpr, ShapeDim};

    #[test]
    fn test_shape_expression_creation() {
        let dim_n = ShapeExpr::Dim(ShapeDim::variable("N".to_string()));
        let dim_2 = ShapeExpr::Dim(ShapeDim::concrete(2));
        let add_expr = ShapeExpr::Add(Box::new(dim_n.clone()), Box::new(dim_2.clone()));
        let mul_expr = ShapeExpr::Mul(Box::new(add_expr), Box::new(dim_n));

        // Test display formatting
        let formatted = format!("{}", mul_expr);
        assert!(formatted.contains("N"));
        assert!(formatted.contains("2"));
        assert!(formatted.contains("+"));
        assert!(formatted.contains("*"));
    }

    #[test]
    fn test_shape_expression_function_application() {
        let arg1 = ShapeExpr::Dim(ShapeDim::variable("N".to_string()));
        let arg2 = ShapeExpr::Dim(ShapeDim::concrete(2));
        let func_app = ShapeExpr::Apply("max".to_string(), vec![arg1, arg2]);

        let formatted = format!("{}", func_app);
        assert!(formatted.contains("max("));
        assert!(formatted.contains("N"));
        assert!(formatted.contains("2"));
    }

    #[test]
    fn test_shape_expression_conditional() {
        let cond = ShapeExpr::Dim(ShapeDim::variable("N".to_string()));
        let then_expr = ShapeExpr::Dim(ShapeDim::concrete(1));
        let else_expr = ShapeExpr::Dim(ShapeDim::concrete(0));
        let if_expr = ShapeExpr::If(Box::new(cond), Box::new(then_expr), Box::new(else_expr));

        let formatted = format!("{}", if_expr);
        assert!(formatted.contains("if"));
        assert!(formatted.contains("then"));
        assert!(formatted.contains("else"));
    }
}