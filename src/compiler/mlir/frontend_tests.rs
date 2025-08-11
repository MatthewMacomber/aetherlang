// Tests for AetherMLIRFrontend AST to MLIR conversion
// Comprehensive test suite for the enhanced frontend implementation

#[cfg(test)]
mod tests {
    use crate::compiler::ast::{AST, ASTNode, ASTNodeRef};
    use crate::compiler::mlir::mlir_context::{MLIRContext, MLIRType};
    use crate::compiler::mlir::frontend::AetherMLIRFrontend;

    /// Helper function to create test MLIR context
    fn create_test_context() -> MLIRContext {
        // Use a simplified mock context for testing
        match MLIRContext::new() {
            Ok(context) => context,
            Err(_) => {
                // Create a minimal mock context for testing
                // This is a simplified implementation for testing purposes
                MLIRContext::new_mock()
            }
        }
    }

    /// Helper function to create simple AST for testing
    fn create_simple_ast(root: ASTNode) -> AST {
        AST::new(root)
    }

    #[test]
    fn test_frontend_creation() {
        let context = create_test_context();
        let _frontend = AetherMLIRFrontend::new(&context);
        
        // Frontend created successfully - basic smoke test
        // Note: Internal fields are private, so we can't test them directly
    }

    #[test]
    fn test_ast_to_module_conversion() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create simple AST: (+ 1 2)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
            ASTNodeRef::direct(ASTNode::number(2.0)),
        ]));

        // Convert AST to module
        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert AST to module: {:?}", result);

        // Verify module contains operations
        assert!(!module.operations().is_empty());
    }

    #[test]
    fn test_variable_declaration() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create AST for (let x 42)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::number(42.0)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert variable declaration: {:?}", result);

        // Verify operation was added to module
        assert!(!module.operations().is_empty());
        let op = &module.operations()[0];
        assert_eq!(op.name, "aether.var");
    }

    #[test]
    fn test_function_definition() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create AST for (defun add (x y) (+ x y))
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("defun".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
            ])),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
            ])),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert function definition: {:?}", result);

        // Verify function operation was added
        assert!(!module.operations().is_empty());
        let op = &module.operations()[0];
        assert_eq!(op.name, "func.func");
    }

    #[test]
    fn test_arithmetic_operations() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test addition: (+ 1 2)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
            ASTNodeRef::direct(ASTNode::number(2.0)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert arithmetic operation: {:?}", result);

        // Verify arithmetic operation was added
        assert!(!module.operations().is_empty());
        let op = &module.operations().last().unwrap();
        assert_eq!(op.name, "arith.addf");
    }

    #[test]
    fn test_complex_expression() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create complex expression: (let x (+ 1 2))
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
                ASTNodeRef::direct(ASTNode::number(1.0)),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ])),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert complex expression: {:?}", result);

        // Verify multiple operations were generated
        assert!(module.operations().len() >= 2); // At least addition and variable declaration
    }

    #[test]
    fn test_function_call_with_arguments() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create function call: (add 1 2)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
            ASTNodeRef::direct(ASTNode::number(2.0)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert function call: {:?}", result);

        // Verify function call operation was created
        assert!(!module.operations().is_empty());
        let ops = module.operations();
        
        // Should have constant operations for arguments and a function call
        assert!(ops.len() >= 3); // 2 constants + 1 call
        
        // Check that we have a function call operation
        let has_call = ops.iter().any(|op| op.name == "func.call");
        assert!(has_call, "Should have func.call operation");
    }

    #[test]
    fn test_control_flow_if_expression() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create if expression: (if (> x 0) x (- x))
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("if".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(">".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::number(0.0)),
            ])),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("-".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ])),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert if expression: {:?}", result);

        // Verify if operation was created
        let ops = module.operations();
        let has_if = ops.iter().any(|op| op.name == "scf.if");
        assert!(has_if, "Should have scf.if operation");
    }

    #[test]
    fn test_while_loop() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create while loop: (while (> x 0) (set x (- x 1)))
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("while".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(">".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::number(0.0)),
            ])),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("set".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                ASTNodeRef::direct(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("-".to_string())),
                    ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
                    ASTNodeRef::direct(ASTNode::number(1.0)),
                ])),
            ])),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert while loop: {:?}", result);

        // Verify while operation was created
        let ops = module.operations();
        let has_while = ops.iter().any(|op| op.name == "scf.while");
        assert!(has_while, "Should have scf.while operation");
    }

    #[test]
    fn test_for_loop() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Create for loop: (for i 0 10 (print i))
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("for".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("i".to_string())),
            ASTNodeRef::direct(ASTNode::number(0.0)),
            ASTNodeRef::direct(ASTNode::number(10.0)),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("print".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("i".to_string())),
            ])),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert for loop: {:?}", result);

        // Verify for operation was created
        let ops = module.operations();
        let has_for = ops.iter().any(|op| op.name == "scf.for");
        assert!(has_for, "Should have scf.for operation");
    }

    #[test]
    fn test_comparison_operations() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test various comparison operations
        let comparisons = vec!["=", "!=", "<", ">", "<=", ">="];
        
        for comp_op in comparisons {
            let ast = create_simple_ast(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(comp_op.to_string())),
                ASTNodeRef::direct(ASTNode::number(1.0)),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ]));

            let result = frontend.convert_ast_to_module(&ast, &mut module);
            assert!(result.is_ok(), "Failed to convert comparison {}: {:?}", comp_op, result);
        }

        // Verify comparison operations were created
        let ops = module.operations();
        let has_cmp = ops.iter().any(|op| op.name == "arith.cmpf");
        assert!(has_cmp, "Should have arith.cmpf operations");
    }

    #[test]
    fn test_logical_operations() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test logical AND: (and true false)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("and".to_string())),
            ASTNodeRef::direct(ASTNode::boolean(true)),
            ASTNodeRef::direct(ASTNode::boolean(false)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert logical AND: {:?}", result);

        // Test logical OR: (or true false)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("or".to_string())),
            ASTNodeRef::direct(ASTNode::boolean(true)),
            ASTNodeRef::direct(ASTNode::boolean(false)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert logical OR: {:?}", result);

        // Test logical NOT: (not true)
        let ast = create_simple_ast(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("not".to_string())),
            ASTNodeRef::direct(ASTNode::boolean(true)),
        ]));

        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok(), "Failed to convert logical NOT: {:?}", result);

        // Verify logical operations were created
        let ops = module.operations();
        let has_logical = ops.iter().any(|op| {
            op.name == "arith.andi" || op.name == "arith.ori" || op.name == "arith.xori"
        });
        assert!(has_logical, "Should have logical operations");
    }

    #[test]
    fn test_type_inference() {
        let context = create_test_context();
        let frontend = AetherMLIRFrontend::new(&context);

        // Test number type inference
        let number_node = ASTNodeRef::direct(ASTNode::number(42.0));
        let number_type = frontend.infer_expression_type(&number_node);
        assert!(number_type.is_ok());
        match number_type.unwrap() {
            MLIRType::Float { width: 64 } => {},
            _ => panic!("Expected Float type for number"),
        }

        // Test boolean type inference
        let bool_node = ASTNodeRef::direct(ASTNode::boolean(true));
        let bool_type = frontend.infer_expression_type(&bool_node);
        assert!(bool_type.is_ok());
        match bool_type.unwrap() {
            MLIRType::Integer { width: 1, signed: false } => {},
            _ => panic!("Expected Integer type for boolean"),
        }

        // Test string type inference
        let string_node = ASTNodeRef::direct(ASTNode::string("hello".to_string()));
        let string_type = frontend.infer_expression_type(&string_node);
        assert!(string_type.is_ok());
        match string_type.unwrap() {
            MLIRType::Memref { .. } => {},
            _ => panic!("Expected Memref type for string"),
        }
    }

    #[test]
    fn test_tensor_creation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test tensor creation: (tensor create [2 3] cpu true f64)
        let tensor_op = ASTNode::symbol("tensor".to_string());
        let create_op = ASTNode::symbol("create".to_string());
        let shape = ASTNode::List(vec![
            ASTNodeRef::direct(ASTNode::number(2.0)),
            ASTNodeRef::direct(ASTNode::number(3.0)),
        ]);
        let device = ASTNode::symbol("cpu".to_string());
        let differentiable = ASTNode::boolean(true);
        let element_type = ASTNode::symbol("f64".to_string());

        let tensor_expr = ASTNode::List(vec![
            ASTNodeRef::direct(tensor_op),
            ASTNodeRef::direct(create_op),
            ASTNodeRef::direct(shape),
            ASTNodeRef::direct(device),
            ASTNodeRef::direct(differentiable),
            ASTNodeRef::direct(element_type),
        ]);

        let result = frontend.traverse_node(&tensor_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_tensor_operations() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test tensor reshape: (tensor reshape tensor_var [4 6])
        let tensor_op = ASTNode::symbol("tensor".to_string());
        let reshape_op = ASTNode::symbol("reshape".to_string());
        let tensor_var = ASTNode::symbol("tensor_var".to_string());
        let new_shape = ASTNode::List(vec![
            ASTNodeRef::direct(ASTNode::number(4.0)),
            ASTNodeRef::direct(ASTNode::number(6.0)),
        ]);

        let reshape_expr = ASTNode::List(vec![
            ASTNodeRef::direct(tensor_op),
            ASTNodeRef::direct(reshape_op),
            ASTNodeRef::direct(tensor_var),
            ASTNodeRef::direct(new_shape),
        ]);

        let result = frontend.traverse_node(&reshape_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_matrix_multiplication() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test matrix multiplication: (matmul a b)
        let matmul_op = ASTNode::symbol("matmul".to_string());
        let a_var = ASTNode::symbol("a".to_string());
        let b_var = ASTNode::symbol("b".to_string());

        let matmul_expr = ASTNode::List(vec![
            ASTNodeRef::direct(matmul_op),
            ASTNodeRef::direct(a_var),
            ASTNodeRef::direct(b_var),
        ]);

        let result = frontend.traverse_node(&matmul_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_automatic_differentiation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test forward mode autodiff: (autodiff forward my_function input)
        let autodiff_op = ASTNode::symbol("autodiff".to_string());
        let mode = ASTNode::symbol("forward".to_string());
        let function = ASTNode::symbol("my_function".to_string());
        let input = ASTNode::symbol("input".to_string());

        let autodiff_expr = ASTNode::List(vec![
            ASTNodeRef::direct(autodiff_op),
            ASTNodeRef::direct(mode),
            ASTNodeRef::direct(function),
            ASTNodeRef::direct(input),
        ]);

        let result = frontend.traverse_node(&autodiff_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_reverse_mode_autodiff() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test reverse mode autodiff: (autodiff reverse my_function input)
        let autodiff_op = ASTNode::symbol("autodiff".to_string());
        let mode = ASTNode::symbol("reverse".to_string());
        let function = ASTNode::symbol("my_function".to_string());
        let input = ASTNode::symbol("input".to_string());

        let autodiff_expr = ASTNode::List(vec![
            ASTNodeRef::direct(autodiff_op),
            ASTNodeRef::direct(mode),
            ASTNodeRef::direct(function),
            ASTNodeRef::direct(input),
        ]);

        let result = frontend.traverse_node(&autodiff_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_gradient_computation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test gradient computation: (gradient my_function input)
        let gradient_op = ASTNode::symbol("gradient".to_string());
        let function = ASTNode::symbol("my_function".to_string());
        let input = ASTNode::symbol("input".to_string());

        let gradient_expr = ASTNode::List(vec![
            ASTNodeRef::direct(gradient_op),
            ASTNodeRef::direct(function),
            ASTNodeRef::direct(input),
        ]);

        let result = frontend.traverse_node(&gradient_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_probabilistic_variable() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test probabilistic variable: (prob-var x (normal 0.0 1.0))
        let prob_var_op = ASTNode::symbol("prob-var".to_string());
        let var_name = ASTNode::symbol("x".to_string());
        let distribution = ASTNode::List(vec![
            ASTNodeRef::direct(ASTNode::symbol("normal".to_string())),
            ASTNodeRef::direct(ASTNode::number(0.0)),
            ASTNodeRef::direct(ASTNode::number(1.0)),
        ]);

        let prob_var_expr = ASTNode::List(vec![
            ASTNodeRef::direct(prob_var_op),
            ASTNodeRef::direct(var_name),
            ASTNodeRef::direct(distribution),
        ]);

        let result = frontend.traverse_node(&prob_var_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_sampling_operation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test sampling: (sample prob_var 10 42)
        let sample_op = ASTNode::symbol("sample".to_string());
        let prob_var = ASTNode::symbol("prob_var".to_string());
        let count = ASTNode::number(10.0);
        let seed = ASTNode::number(42.0);

        let sample_expr = ASTNode::List(vec![
            ASTNodeRef::direct(sample_op),
            ASTNodeRef::direct(prob_var),
            ASTNodeRef::direct(count),
            ASTNodeRef::direct(seed),
        ]);

        let result = frontend.traverse_node(&sample_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_observation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test observation: (observe prob_var observed_value 1.0)
        let observe_op = ASTNode::symbol("observe".to_string());
        let prob_var = ASTNode::symbol("prob_var".to_string());
        let observed_value = ASTNode::number(2.5);
        let weight = ASTNode::number(1.0);

        let observe_expr = ASTNode::List(vec![
            ASTNodeRef::direct(observe_op),
            ASTNodeRef::direct(prob_var),
            ASTNodeRef::direct(observed_value),
            ASTNodeRef::direct(weight),
        ]);

        let result = frontend.traverse_node(&observe_expr, &mut module);
        assert!(result.is_ok());
        // Observe with weight should return log-likelihood
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_linear_type_operations() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test linear type: (linear f64 owned "static" "heap")
        let linear_op = ASTNode::symbol("linear".to_string());
        let inner_type = ASTNode::symbol("f64".to_string());
        let ownership = ASTNode::symbol("owned".to_string());
        let lifetime = ASTNode::symbol("static".to_string());
        let allocation_site = ASTNode::string("heap".to_string());

        let linear_expr = ASTNode::List(vec![
            ASTNodeRef::direct(linear_op),
            ASTNodeRef::direct(inner_type),
            ASTNodeRef::direct(ownership),
            ASTNodeRef::direct(lifetime),
            ASTNodeRef::direct(allocation_site),
        ]);

        let result = frontend.traverse_node(&linear_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_move_operation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test move operation: (move source_var dest_var)
        let move_op = ASTNode::symbol("move".to_string());
        let source = ASTNode::symbol("source_var".to_string());
        let dest = ASTNode::symbol("dest_var".to_string());

        let move_expr = ASTNode::List(vec![
            ASTNodeRef::direct(move_op),
            ASTNodeRef::direct(source),
            ASTNodeRef::direct(dest),
        ]);

        let result = frontend.traverse_node(&move_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_drop_operation() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test drop operation: (drop value immediate)
        let drop_op = ASTNode::symbol("drop".to_string());
        let value = ASTNode::symbol("value".to_string());
        let mode = ASTNode::symbol("immediate".to_string());

        let drop_expr = ASTNode::List(vec![
            ASTNodeRef::direct(drop_op),
            ASTNodeRef::direct(value),
            ASTNodeRef::direct(mode),
        ]);

        let result = frontend.traverse_node(&drop_expr, &mut module);
        assert!(result.is_ok());
        // Drop operations don't return values
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_actor_spawn() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test actor spawn: (spawn actor_type initial_state)
        let spawn_op = ASTNode::symbol("spawn".to_string());
        let actor_type = ASTNode::symbol("MyActor".to_string());
        let initial_state = ASTNode::number(0.0);

        let spawn_expr = ASTNode::List(vec![
            ASTNodeRef::direct(spawn_op),
            ASTNodeRef::direct(actor_type),
            ASTNodeRef::direct(initial_state),
        ]);

        let result = frontend.traverse_node(&spawn_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_message_send() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test message send: (send actor_ref message)
        let send_op = ASTNode::symbol("send".to_string());
        let actor_ref = ASTNode::symbol("actor_ref".to_string());
        let message = ASTNode::string("hello".to_string());

        let send_expr = ASTNode::List(vec![
            ASTNodeRef::direct(send_op),
            ASTNodeRef::direct(actor_ref),
            ASTNodeRef::direct(message),
        ]);

        let result = frontend.traverse_node(&send_expr, &mut module);
        assert!(result.is_ok());
        // Send operations don't return values
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parallel_for() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test parallel for: (parallel-for 0 10 body_func)
        let parallel_for_op = ASTNode::symbol("parallel-for".to_string());
        let lower = ASTNode::number(0.0);
        let upper = ASTNode::number(10.0);
        let body = ASTNode::symbol("body_func".to_string());

        let parallel_for_expr = ASTNode::List(vec![
            ASTNodeRef::direct(parallel_for_op),
            ASTNodeRef::direct(lower),
            ASTNodeRef::direct(upper),
            ASTNodeRef::direct(body),
        ]);

        let result = frontend.traverse_node(&parallel_for_expr, &mut module);
        assert!(result.is_ok());
        // Parallel for operations don't return values
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_complex_ai_workflow() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").expect("Failed to create module");

        // Test complex AI workflow combining multiple constructs
        // (let weights (tensor create [10 5] gpu true f32))
        let let_op = ASTNode::symbol("let".to_string());
        let weights_var = ASTNode::symbol("weights".to_string());
        let tensor_create = ASTNode::List(vec![
            ASTNodeRef::direct(ASTNode::symbol("tensor".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("create".to_string())),
            ASTNodeRef::direct(ASTNode::List(vec![
                ASTNodeRef::direct(ASTNode::number(10.0)),
                ASTNodeRef::direct(ASTNode::number(5.0)),
            ])),
            ASTNodeRef::direct(ASTNode::symbol("gpu".to_string())),
            ASTNodeRef::direct(ASTNode::boolean(true)),
            ASTNodeRef::direct(ASTNode::symbol("f32".to_string())),
        ]);

        let weights_decl = ASTNode::List(vec![
            ASTNodeRef::direct(let_op),
            ASTNodeRef::direct(weights_var),
            ASTNodeRef::direct(tensor_create),
        ]);

        let result = frontend.traverse_node(&weights_decl, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Test gradient computation on the weights
        let gradient_expr = ASTNode::List(vec![
            ASTNodeRef::direct(ASTNode::symbol("gradient".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("loss_function".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("weights".to_string())),
        ]);

        let grad_result = frontend.traverse_node(&gradient_expr, &mut module);
        assert!(grad_result.is_ok());
        assert!(grad_result.unwrap().is_some());
    }
}