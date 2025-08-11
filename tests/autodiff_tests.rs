// Aether Automatic Differentiation Tests
// Tests for gradient correctness across various function types

use aether_language::compiler::autodiff::{
    AutoDiffEngine, DifferentiableFunction, DiffMarker, DiffMode, DiffDirection,
    DynamicTape, StaticGraph, TapeOperation, GraphNode, GraphEdge,
};
use aether_language::compiler::ast::{ASTNode, ASTNodeRef, AtomValue};
use aether_language::compiler::types::{Type, PrimitiveType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_engine_creation() {
        let engine = AutoDiffEngine::new();
        
        // Check that primitive derivatives are initialized
        assert!(engine.primitive_derivatives.contains_key("add"));
        assert!(engine.primitive_derivatives.contains_key("mul"));
        assert!(engine.primitive_derivatives.contains_key("sin"));
        assert!(engine.primitive_derivatives.contains_key("exp"));
        
        // Check initial state
        assert!(engine.functions.is_empty());
        assert!(engine.gradient_functions.is_empty());
        assert!(engine.tape.is_none());
        assert!(engine.static_graph.is_none());
    }

    #[test]
    fn test_function_registration() {
        let mut engine = AutoDiffEngine::new();
        
        // Create a simple function: f(x) = x^2
        let func = DifferentiableFunction {
            name: "square".to_string(),
            params: vec!["x".to_string()],
            param_types: vec![Type::primitive(PrimitiveType::Float64)],
            return_type: Type::primitive(PrimitiveType::Float64),
            body: ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ]),
            diff_markers: vec![],
        };
        
        // Register function
        let result = engine.register_function(func);
        assert!(result.is_ok());
        assert!(engine.functions.contains_key("square"));
        
        // Try to register same function again (should fail)
        let func2 = DifferentiableFunction {
            name: "square".to_string(),
            params: vec!["x".to_string()],
            param_types: vec![Type::primitive(PrimitiveType::Float64)],
            return_type: Type::primitive(PrimitiveType::Float64),
            body: ASTNode::symbol("x".to_string()),
            diff_markers: vec![],
        };
        
        let result2 = engine.register_function(func2);
        assert!(result2.is_err());
    }

    #[test]
    fn test_forward_mode_differentiation() {
        let mut engine = AutoDiffEngine::new();
        
        // Test differentiation of x^2 with respect to x
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
        ]);
        
        let result = engine.differentiate(
            &expr,
            "x",
            DiffMode::Dynamic,
            DiffDirection::Forward,
        );
        
        assert!(result.is_ok());
        let grad_expr = result.unwrap();
        
        // Should generate dual number computation
        assert!(grad_expr.is_list());
        if let Some(children) = grad_expr.as_list() {
            assert!(!children.is_empty());
        }
    }

    #[test]
    fn test_reverse_mode_differentiation() {
        let mut engine = AutoDiffEngine::new();
        
        // Test differentiation of sin(x) with respect to x
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("sin".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
        ]);
        
        let result = engine.differentiate(
            &expr,
            "x",
            DiffMode::Dynamic,
            DiffDirection::Reverse,
        );
        
        assert!(result.is_ok());
        let grad_expr = result.unwrap();
        
        // Should generate forward and backward pass
        assert!(grad_expr.is_list());
        if let Some(children) = grad_expr.as_list() {
            assert_eq!(children.len(), 2); // Forward and backward pass
        }
    }

    #[test]
    fn test_higher_order_derivatives() {
        let mut engine = AutoDiffEngine::new();
        
        // Test second derivative of x^3
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ])),
        ]);
        
        let result = engine.higher_order_derivative(
            &expr,
            "x",
            2,
            DiffMode::Dynamic,
        );
        
        assert!(result.is_ok());
        let second_deriv = result.unwrap();
        assert!(second_deriv.is_list());
    }

    #[test]
    fn test_dual_number_computation() {
        let engine = AutoDiffEngine::new();
        
        // Test dual number for variable x
        let var_expr = ASTNode::symbol("x".to_string());
        let result = engine.generate_dual_number_computation(&var_expr, "x");
        
        assert!(result.is_ok());
        let dual_expr = result.unwrap();
        
        // Should be dual(x, 1.0)
        if let Some(children) = dual_expr.as_list() {
            assert_eq!(children.len(), 3);
            if let ASTNodeRef::Direct(op_node) = &children[0] {
                if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                    assert_eq!(op_name, "dual");
                }
            }
        }
        
        // Test dual number for constant
        let const_expr = ASTNode::number(5.0);
        let result2 = engine.generate_dual_number_computation(&const_expr, "x");
        
        assert!(result2.is_ok());
        let dual_const = result2.unwrap();
        
        // Should be dual(5.0, 0.0)
        if let Some(children) = dual_const.as_list() {
            assert_eq!(children.len(), 3);
            if let ASTNodeRef::Direct(deriv_node) = &children[2] {
                if let Some(AtomValue::Number(deriv_val)) = deriv_node.as_atom() {
                    assert_eq!(*deriv_val, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_primitive_derivatives() {
        let engine = AutoDiffEngine::new();
        
        // Test addition derivative
        if let Some(add_deriv) = engine.primitive_derivatives.get("add") {
            let inputs = vec![2.0, 3.0];
            let derivs = vec![1.0, 1.0];
            let result = (add_deriv.forward_fn)(&inputs, &derivs);
            assert_eq!(result, vec![2.0]); // 1.0 + 1.0 = 2.0
            
            let grad_result = (add_deriv.reverse_fn)(&inputs, 1.0);
            assert_eq!(grad_result, vec![1.0, 1.0]);
        }
        
        // Test multiplication derivative
        if let Some(mul_deriv) = engine.primitive_derivatives.get("mul") {
            let inputs = vec![2.0, 3.0];
            let derivs = vec![1.0, 1.0];
            let result = (mul_deriv.forward_fn)(&inputs, &derivs);
            assert_eq!(result, vec![5.0]); // 2.0 * 1.0 + 3.0 * 1.0 = 5.0
            
            let grad_result = (mul_deriv.reverse_fn)(&inputs, 1.0);
            assert_eq!(grad_result, vec![3.0, 2.0]); // [b, a] = [3.0, 2.0]
        }
        
        // Test sine derivative
        if let Some(sin_deriv) = engine.primitive_derivatives.get("sin") {
            let inputs = vec![0.0]; // sin(0) = 0, cos(0) = 1
            let derivs = vec![1.0];
            let result = (sin_deriv.forward_fn)(&inputs, &derivs);
            assert!((result[0] - 1.0).abs() < 1e-10); // cos(0) * 1.0 = 1.0
        }
        
        // Test exponential derivative
        if let Some(exp_deriv) = engine.primitive_derivatives.get("exp") {
            let inputs = vec![0.0]; // exp(0) = 1
            let derivs = vec![1.0];
            let result = (exp_deriv.forward_fn)(&inputs, &derivs);
            assert!((result[0] - 1.0).abs() < 1e-10); // exp(0) * 1.0 = 1.0
        }
    }

    #[test]
    fn test_gradient_function_generation() {
        let mut engine = AutoDiffEngine::new();
        
        // Register a function
        let func = DifferentiableFunction {
            name: "quadratic".to_string(),
            params: vec!["x".to_string()],
            param_types: vec![Type::primitive(PrimitiveType::Float64)],
            return_type: Type::primitive(PrimitiveType::Float64),
            body: ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
                ASTNodeRef::direct(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ])),
                ASTNodeRef::direct(ASTNode::number(1.0)),
            ]),
            diff_markers: vec![],
        };
        
        engine.register_function(func).unwrap();
        
        // Generate gradient function
        let diff_marker = DiffMarker {
            wrt_vars: vec!["x".to_string()],
            direction: DiffDirection::Forward,
            order: 1,
            mode: DiffMode::Dynamic,
        };
        
        let result = engine.generate_gradient_function("quadratic", diff_marker);
        assert!(result.is_ok());
        
        let grad_name = result.unwrap();
        assert_eq!(grad_name, "quadratic_grad");
        assert!(engine.gradient_functions.contains_key(&grad_name));
        
        // Check gradient function properties
        if let Some(grad_func) = engine.get_gradient_function(&grad_name) {
            assert_eq!(grad_func.original_name, "quadratic");
            assert_eq!(grad_func.wrt_vars, vec!["x"]);
            assert_eq!(grad_func.input_types.len(), 1);
            assert_eq!(grad_func.output_types.len(), 1);
        }
    }

    #[test]
    fn test_dynamic_tape() {
        let mut tape = DynamicTape::new();
        
        // Set variables
        tape.set_variable("x".to_string(), 2.0);
        tape.set_variable("y".to_string(), 3.0);
        
        assert_eq!(tape.get_variable("x"), Some(2.0));
        assert_eq!(tape.get_variable("y"), Some(3.0));
        
        // Record operation: z = x * y
        let op = TapeOperation {
            id: 0,
            op_type: "mul".to_string(),
            inputs: vec!["x".to_string(), "y".to_string()],
            output: "z".to_string(),
            params: vec![],
            forward_fn: None,
            reverse_fn: Some(|inputs, grad| vec![grad * inputs[1], grad * inputs[0]]),
        };
        
        tape.record_operation(op);
        tape.set_variable("z".to_string(), 6.0); // 2.0 * 3.0
        
        // Set output gradient
        tape.set_gradient("z".to_string(), 1.0);
        
        // Execute backward pass
        let result = tape.backward_pass();
        assert!(result.is_ok());
        
        // Check gradients
        assert_eq!(tape.get_gradient("x"), Some(3.0)); // dz/dx = y = 3.0
        assert_eq!(tape.get_gradient("y"), Some(2.0)); // dz/dy = x = 2.0
    }

    #[test]
    fn test_static_graph() {
        let mut graph = StaticGraph::new();
        
        // Add nodes: x, y, z = x * y
        let x_node = GraphNode {
            id: 0,
            op_type: "variable".to_string(),
            value_type: Type::primitive(PrimitiveType::Float64),
            params: vec![],
            requires_grad: true,
        };
        
        let y_node = GraphNode {
            id: 1,
            op_type: "variable".to_string(),
            value_type: Type::primitive(PrimitiveType::Float64),
            params: vec![],
            requires_grad: true,
        };
        
        let mul_node = GraphNode {
            id: 2,
            op_type: "mul".to_string(),
            value_type: Type::primitive(PrimitiveType::Float64),
            params: vec![],
            requires_grad: true,
        };
        
        let x_id = graph.add_node(x_node);
        let y_id = graph.add_node(y_node);
        let mul_id = graph.add_node(mul_node);
        
        // Add edges
        graph.add_edge(GraphEdge { from: x_id, to: mul_id, weight: None });
        graph.add_edge(GraphEdge { from: y_id, to: mul_id, weight: None });
        
        // Test topological sort
        let sorted = graph.topological_sort();
        assert!(sorted.is_ok());
        let sorted_nodes = sorted.unwrap();
        
        // The topological sort returns nodes in dependency order
        // Dependencies (x, y) should come after the node that depends on them (mul)
        let x_pos = sorted_nodes.iter().position(|&id| id == x_id).unwrap();
        let y_pos = sorted_nodes.iter().position(|&id| id == y_id).unwrap();
        let mul_pos = sorted_nodes.iter().position(|&id| id == mul_id).unwrap();
        
        // mul should come before x and y in the sorted order
        assert!(mul_pos < x_pos);
        assert!(mul_pos < y_pos);
    }

    #[test]
    fn test_tensor_gradient() {
        let mut engine = AutoDiffEngine::new();
        
        // Create tensor type
        let tensor_type = Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![2, 3],
        );
        
        // Create tensor expression: A * B
        let tensor_expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("matmul".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("A".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("B".to_string())),
        ]);
        
        let result = engine.tensor_gradient(&tensor_expr, "A", &tensor_type);
        assert!(result.is_ok());
        
        let grad_expr = result.unwrap();
        assert!(grad_expr.is_list());
        
        // Should generate tensor gradient computation
        if let Some(children) = grad_expr.as_list() {
            assert!(!children.is_empty());
            if let ASTNodeRef::Direct(op_node) = &children[0] {
                if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                    assert_eq!(op_name, "tensor_gradient");
                }
            }
        }
    }

    #[test]
    fn test_complex_expression_differentiation() {
        let mut engine = AutoDiffEngine::new();
        
        // Test differentiation of sin(x^2 + 1) with respect to x
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("sin".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
                ASTNodeRef::direct(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ])),
                ASTNodeRef::direct(ASTNode::number(1.0)),
            ])),
        ]);
        
        let result = engine.differentiate(
            &expr,
            "x",
            DiffMode::Dynamic,
            DiffDirection::Forward,
        );
        
        assert!(result.is_ok());
        let grad_expr = result.unwrap();
        
        // Should generate nested dual number computation
        assert!(grad_expr.is_list());
    }

    #[test]
    fn test_static_mode_differentiation() {
        let mut engine = AutoDiffEngine::new();
        
        // Test static mode differentiation
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("exp".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
        ]);
        
        let result = engine.differentiate(
            &expr,
            "x",
            DiffMode::Static,
            DiffDirection::Forward,
        );
        

        assert!(result.is_ok());
        let grad_expr = result.unwrap();
        
        // Should generate static differentiation code
        assert!(grad_expr.is_list());
        if let Some(children) = grad_expr.as_list() {
            if let ASTNodeRef::Direct(op_node) = &children[0] {
                if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                    assert_eq!(op_name, "static_differentiate");
                }
            }
        }
    }

    #[test]
    fn test_mixed_mode_differentiation() {
        let mut engine = AutoDiffEngine::new();
        
        // Register a multi-variable function
        let func = DifferentiableFunction {
            name: "multi_var".to_string(),
            params: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            param_types: vec![
                Type::primitive(PrimitiveType::Float64),
                Type::primitive(PrimitiveType::Float64),
                Type::primitive(PrimitiveType::Float64),
            ],
            return_type: Type::primitive(PrimitiveType::Float64),
            body: ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
                ASTNodeRef::direct(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
                ])),
                ASTNodeRef::direct(ASTNode::symbol("z".to_string())),
            ]),
            diff_markers: vec![],
        };
        
        engine.register_function(func).unwrap();
        
        // Generate mixed mode gradient (forward for x, reverse for y, z)
        let diff_marker = DiffMarker {
            wrt_vars: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            direction: DiffDirection::Mixed(vec!["x".to_string()]),
            order: 1,
            mode: DiffMode::Dynamic,
        };
        
        let result = engine.generate_gradient_function("multi_var", diff_marker);
        assert!(result.is_ok());
        
        let grad_name = result.unwrap();
        assert!(engine.gradient_functions.contains_key(&grad_name));
    }

    #[test]
    fn test_error_handling() {
        let mut engine = AutoDiffEngine::new();
        
        // Test differentiation of non-existent function
        let result = engine.generate_gradient_function(
            "non_existent",
            DiffMarker {
                wrt_vars: vec!["x".to_string()],
                direction: DiffDirection::Forward,
                order: 1,
                mode: DiffMode::Dynamic,
            },
        );
        assert!(result.is_err());
        
        // Test invalid parameter count
        let invalid_func = DifferentiableFunction {
            name: "invalid".to_string(),
            params: vec!["x".to_string()],
            param_types: vec![], // Mismatch: 1 param, 0 types
            return_type: Type::primitive(PrimitiveType::Float64),
            body: ASTNode::symbol("x".to_string()),
            diff_markers: vec![],
        };
        
        let result2 = engine.register_function(invalid_func);
        assert!(result2.is_err());
    }

    #[test]
    fn test_differentiation_markers() {
        let diff_marker = DiffMarker {
            wrt_vars: vec!["x".to_string(), "y".to_string()],
            direction: DiffDirection::Reverse,
            order: 2,
            mode: DiffMode::Static,
        };
        
        // Test display formatting
        let marker_str = format!("{}", diff_marker);
        assert!(marker_str.contains("wrt"));
        assert!(marker_str.contains("Reverse"));
        assert!(marker_str.contains("order=2"));
        assert!(marker_str.contains("Static"));
    }

    #[test]
    fn test_computation_graph_building() {
        let mut engine = AutoDiffEngine::new();
        
        // Initialize static graph
        engine.static_graph = Some(StaticGraph::new());
        
        // Build graph for expression: x + y * 2
        let expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("mul".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ])),
        ]);
        
        let result = engine.build_computation_graph(&expr);
        assert!(result.is_ok());
        
        let root_id = result.unwrap();
        if let Some(graph) = &engine.static_graph {
            assert!(graph.get_node(root_id).is_some());
            assert!(!graph.nodes.is_empty());
        }
    }
}