// Frontend stage converting AST to Aether MLIR dialect
// Handles the conversion from Aether AST to MLIR representation

use std::collections::HashMap;
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue, NodeId};
use crate::compiler::mlir::{MLIRError, AetherOps};
use crate::compiler::types::LinearOwnership;
use crate::compiler::mlir::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRAttribute, MLIRValue, MLIRType};
use crate::compiler::mlir::test_utils::MockMLIRModule;
use crate::compiler::mlir::aether_types::{AetherMLIRType, AetherTypeConverter, TypeVerifier};
use crate::compiler::mlir::dialect::{AetherOperationBuilder, DistributionType};
use crate::compiler::symbol_table::{SymbolTable, Symbol, SymbolType, SymbolId};

/// Enhanced frontend for converting Aether AST to MLIR with comprehensive AST traversal
pub struct AetherMLIRFrontend<'a> {
    /// MLIR context for operation creation
    context: &'a MLIRContext,
    /// Symbol table for variable and function resolution
    symbol_table: SymbolTable,
    /// Type converter for Aether to MLIR type conversion
    type_converter: AetherTypeConverter,
    /// Type verifier for operation validation
    type_verifier: TypeVerifier,
    /// Operation builder for creating validated operations
    operation_builder: AetherOperationBuilder,
    /// Current function being processed
    current_function: Option<String>,
    /// Current scope depth for nested constructs
    scope_depth: u32,
    /// Value counter for generating unique SSA value names
    value_counter: u32,
    /// AST node cache for graph structure support
    node_cache: HashMap<NodeId, ASTNode>,
    /// Label to value mapping for datum labels
    label_values: HashMap<String, String>,
}

/// Legacy frontend for backward compatibility
pub struct AetherFrontend<'a> {
    context: &'a MLIRContext,
    symbol_table: HashMap<String, String>,
    current_function: Option<String>,
}

impl<'a> AetherMLIRFrontend<'a> {
    /// Create new enhanced Aether MLIR frontend
    pub fn new(context: &'a MLIRContext) -> Self {
        AetherMLIRFrontend {
            context,
            symbol_table: SymbolTable::new(),
            type_converter: AetherTypeConverter::new(),
            type_verifier: TypeVerifier::new(),
            operation_builder: AetherOperationBuilder::new(context),
            current_function: None,
            scope_depth: 0,
            value_counter: 0,
            node_cache: HashMap::new(),
            label_values: HashMap::new(),
        }
    }

    /// Convert AST to MLIR module with comprehensive traversal
    pub fn convert_ast_to_module(&mut self, ast: &AST, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Cache AST nodes for graph structure support
        self.cache_ast_nodes(ast)?;
        
        // Convert root node
        self.traverse_node(&ast.root, module)?;
        
        // Verify all operations in the module
        self.verify_module_operations(module)?;
        
        Ok(())
    }

    /// Cache AST nodes for efficient graph traversal
    fn cache_ast_nodes(&mut self, ast: &AST) -> Result<(), MLIRError> {
        // Cache nodes from the AST's node storage
        for (id, node) in &ast.nodes {
            self.node_cache.insert(*id, node.clone());
        }
        
        // Cache labels
        for (label, id) in &ast.labels {
            if let Some(node) = ast.nodes.get(id) {
                // Generate a value name for the labeled node
                let value_name = self.generate_value_name(&format!("label_{}", label));
                self.label_values.insert(label.clone(), value_name);
            }
        }
        
        Ok(())
    }

    /// Traverse AST node using visitor pattern
    pub fn traverse_node(&mut self, node: &ASTNode, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        match node {
            ASTNode::Atom(atom) => self.visit_atom(atom, module),
            ASTNode::List(children) => self.visit_list(children, module),
            ASTNode::Graph { nodes, edges, labels } => self.visit_graph(nodes, edges, labels, module),
        }
    }

    /// Visit atomic value node
    fn visit_atom(&mut self, atom: &AtomValue, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        match atom {
            AtomValue::Symbol(name) => {
                // Look up symbol in symbol table
                if let Some(symbol) = self.symbol_table.lookup(name) {
                    match symbol.symbol_type {
                        SymbolType::Variable => {
                            Ok(Some(format!("%{}", name)))
                        }
                        SymbolType::Function => {
                            Ok(Some(format!("@{}", name)))
                        }
                        _ => Ok(Some(name.clone()))
                    }
                } else {
                    // Unknown symbol - treat as identifier
                    Ok(Some(name.clone()))
                }
            }
            AtomValue::Number(value) => {
                // Create constant operation for numeric literals
                let const_name = self.generate_value_name("const");
                let mut const_op = MLIROperation::new("arith.constant".to_string());
                const_op.add_attribute("value".to_string(), MLIRAttribute::Float(*value));
                
                let result = MLIRValue::new(const_name.clone(), MLIRType::Float { width: 64 });
                const_op.add_result(result);
                
                module.add_operation(const_op)?;
                Ok(Some(const_name))
            }
            AtomValue::String(value) => {
                // Create string constant
                let str_name = self.generate_value_name("str");
                let mut str_op = MLIROperation::new("arith.constant".to_string());
                str_op.add_attribute("value".to_string(), MLIRAttribute::String(value.clone()));
                
                let result = MLIRValue::new(str_name.clone(), MLIRType::Memref { 
                    element_type: Box::new(MLIRType::Integer { width: 8, signed: false }), 
                    shape: vec![-1] // Dynamic size string
                });
                str_op.add_result(result);
                
                module.add_operation(str_op)?;
                Ok(Some(str_name))
            }
            AtomValue::Boolean(value) => {
                // Create boolean constant
                let bool_name = self.generate_value_name("bool");
                let mut bool_op = MLIROperation::new("arith.constant".to_string());
                bool_op.add_attribute("value".to_string(), MLIRAttribute::Boolean(*value));
                
                let result = MLIRValue::new(bool_name.clone(), MLIRType::Integer { width: 1, signed: false });
                bool_op.add_result(result);
                
                module.add_operation(bool_op)?;
                Ok(Some(bool_name))
            }
            AtomValue::Nil => {
                // Create nil constant
                let nil_name = self.generate_value_name("nil");
                let mut nil_op = MLIROperation::new("arith.constant".to_string());
                nil_op.add_attribute("value".to_string(), MLIRAttribute::String("nil".to_string()));
                
                let result = MLIRValue::new(nil_name.clone(), MLIRType::Integer { width: 1, signed: false });
                nil_op.add_result(result);
                
                module.add_operation(nil_op)?;
                Ok(Some(nil_name))
            }
            AtomValue::Token(_) => {
                // Tokens are not directly converted to values
                Ok(None)
            }
        }
    }

    /// Visit list node (S-expression)
    fn visit_list(&mut self, children: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if children.is_empty() {
            return Ok(None);
        }

        // Get the first element to determine operation type
        let first_child = &children[0];
        if let Some(first_node) = self.resolve_node_ref(first_child) {
            if let ASTNode::Atom(AtomValue::Symbol(op_name)) = first_node {
                let op_name = op_name.clone(); // Clone to avoid borrowing issues
                return self.visit_operation(&op_name, &children[1..], module);
            }
        }

        // If not a recognized operation, treat as function call
        self.visit_function_call(children, module)
    }

    /// Visit graph node with edges and labels
    fn visit_graph(&mut self, nodes: &[ASTNodeRef], edges: &[crate::compiler::ast::GraphEdge], labels: &HashMap<String, NodeId>, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        // Enter new scope for graph processing
        self.enter_scope();
        
        let mut last_value = None;
        
        // Process all nodes in the graph
        for node_ref in nodes {
            if let Some(node) = self.resolve_node_ref(node_ref) {
                let node = node.clone(); // Clone to avoid borrowing issues
                if let Some(value) = self.traverse_node(&node, module)? {
                    last_value = Some(value);
                }
            }
        }
        
        // Process edges (for control flow or data dependencies)
        for edge in edges {
            self.process_graph_edge(edge, module)?;
        }
        
        // Exit scope
        self.exit_scope();
        
        Ok(last_value)
    }

    /// Visit operation based on operation name
    fn visit_operation(&mut self, op_name: &str, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        match op_name {
            // Function definition
            "defun" | "fn" => self.visit_function_definition(args, module),
            
            // Variable declaration
            "let" | "var" => self.visit_variable_declaration(args, module),
            
            // Control flow
            "if" => self.visit_if_expression(args, module),
            "while" => self.visit_while_loop(args, module),
            "for" => self.visit_for_loop(args, module),
            
            // Tensor operations
            "tensor" => self.visit_tensor_operation(args, module),
            "matmul" => self.visit_matrix_multiplication(args, module),
            
            // Automatic differentiation
            "autodiff" | "grad" => self.visit_autodiff_marker(args, module),
            "gradient" => self.visit_gradient_operation(args, module),
            
            // Probabilistic programming
            "prob-var" | "random" => self.visit_probabilistic_variable(args, module),
            "sample" => self.visit_sample_operation(args, module),
            "observe" => self.visit_observe_operation(args, module),
            
            // Linear types
            "linear" => self.visit_linear_type(args, module),
            "move" => self.visit_move_operation(args, module),
            "drop" => self.visit_drop_operation(args, module),
            
            // Concurrency
            "spawn" => self.visit_spawn_actor(args, module),
            "send" => self.visit_send_message(args, module),
            "parallel-for" => self.visit_parallel_for(args, module),
            
            // Arithmetic operations
            "+" | "-" | "*" | "/" | "%" => self.visit_arithmetic_operation(op_name, args, module),
            
            // Comparison operations
            "=" | "==" | "!=" | "<" | ">" | "<=" | ">=" => self.visit_comparison_operation(op_name, args, module),
            
            // Logical operations
            "and" | "or" | "not" => self.visit_logical_operation(op_name, args, module),
            
            // Function call (unknown operation)
            _ => self.visit_function_call(&[ASTNodeRef::direct(ASTNode::symbol(op_name.to_string()))], module),
        }
    }

    /// Visit function definition
    fn visit_function_definition(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 3 {
            return Err(MLIRError::ConversionError("Function definition requires name, parameters, and body".to_string()));
        }

        // Extract function name
        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        // Enter function scope
        self.enter_scope();
        self.current_function = Some(name.clone());

        // Process parameters
        let params = self.process_function_parameters(&args[1])?;
        
        // Infer parameter types (simplified - default to f64)
        let param_types: Vec<MLIRType> = params.iter()
            .map(|_| MLIRType::Float { width: 64 })
            .collect();
        
        // Create function type
        let return_type = MLIRType::Float { width: 64 }; // Simplified - could be inferred from body
        let function_type = MLIRType::Function {
            inputs: param_types.clone(),
            outputs: vec![return_type.clone()],
        };
        
        // Create function operation
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String(name.clone()));
        func_op.add_attribute("type".to_string(), MLIRAttribute::String(format!("{:?}", function_type)));
        
        // Add parameters as attributes
        for (i, param_name) in params.iter().enumerate() {
            func_op.add_attribute(
                format!("param_{}", i), 
                MLIRAttribute::String(param_name.clone())
            );
        }
        
        // Add function to symbol table
        let _func_id = self.symbol_table.add_symbol(name.clone(), SymbolType::Function);
        
        // Create function entry block
        let mut entry_block = MLIROperation::new("func.entry".to_string());
        
        // Add parameter values to entry block
        for (i, (param_name, param_type)) in params.iter().zip(param_types.iter()).enumerate() {
            let param_value_name = self.generate_value_name(&format!("arg_{}", i));
            let param_value = MLIRValue::new(param_value_name.clone(), param_type.clone());
            entry_block.add_result(param_value);
            
            // Add parameter to symbol table in function scope
            let _param_id = self.symbol_table.add_symbol(param_name.clone(), SymbolType::Variable);
        }
        
        // Process function body
        let body_result = self.traverse_node_ref(&args[2], module)?;
        
        // Create return operation
        if let Some(return_value_name) = body_result {
            let mut return_op = MLIROperation::new("func.return".to_string());
            let return_value = MLIRValue::new(return_value_name, return_type);
            return_op.add_operand(return_value);
            module.add_operation(return_op)?;
        }
        
        // Add function to module
        module.add_operation(func_op)?;
        
        // Exit function scope
        self.current_function = None;
        self.exit_scope();
        
        Ok(Some(format!("@{}", name)))
    }

    /// Visit variable declaration
    fn visit_variable_declaration(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Variable declaration requires name and value".to_string()));
        }

        // Extract variable name
        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        // Process initial value
        let initial_value = self.traverse_node_ref(&args[1], module)?;

        // Create variable operation
        let mut var_op = MLIROperation::new("aether.var".to_string());
        var_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        
        // Infer type from initial value (simplified)
        let var_type = MLIRType::Float { width: 64 }; // Default to f64
        var_op.add_attribute("type".to_string(), MLIRAttribute::String("f64".to_string()));
        
        // Add result
        let result_name = self.generate_value_name(&name);
        let result = MLIRValue::new(result_name.clone(), var_type);
        var_op.add_result(result);
        
        // Add to symbol table
        let var_id = self.symbol_table.add_symbol(name.clone(), SymbolType::Variable);
        
        // Add operation to module
        module.add_operation(var_op)?;
        
        Ok(Some(result_name))
    }

    /// Visit if expression
    fn visit_if_expression(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("If expression requires condition and then branch".to_string()));
        }

        // Process condition
        let condition = self.traverse_node_ref(&args[0], module)?;
        
        // Process then branch
        let then_result = self.traverse_node_ref(&args[1], module)?;
        
        // Process else branch if present
        let else_result = if args.len() > 2 {
            Some(self.traverse_node_ref(&args[2], module)?)
        } else {
            None
        };

        // Create if operation using SCF dialect
        let mut if_op = MLIROperation::new("scf.if".to_string());
        
        // Add condition operand
        if let Some(cond_name) = condition {
            let cond_value = MLIRValue::new(cond_name, MLIRType::Integer { width: 1, signed: false });
            if_op.add_operand(cond_value);
        }
        
        // Add result
        let result_name = self.generate_value_name("if_result");
        let result_type = MLIRType::Float { width: 64 }; // Simplified
        let result = MLIRValue::new(result_name.clone(), result_type);
        if_op.add_result(result);
        
        module.add_operation(if_op)?;
        
        Ok(Some(result_name))
    }

    /// Visit while loop
    fn visit_while_loop(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("While loop requires condition and body".to_string()));
        }

        // Enter loop scope
        self.enter_scope();
        
        // Process condition
        let condition = self.traverse_node_ref(&args[0], module)?;
        
        // Process body
        let body_result = self.traverse_node_ref(&args[1], module)?;
        
        // Create while operation using SCF dialect
        let mut while_op = MLIROperation::new("scf.while".to_string());
        
        module.add_operation(while_op)?;
        
        // Exit loop scope
        self.exit_scope();
        
        Ok(None) // While loops don't return values
    }

    /// Visit for loop
    fn visit_for_loop(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 4 {
            return Err(MLIRError::ConversionError("For loop requires variable, start, end, and body".to_string()));
        }

        // Enter loop scope
        self.enter_scope();
        
        // Extract loop variable
        let loop_var = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Loop variable must be a symbol".to_string()));
        };

        // Process bounds
        let start = self.traverse_node_ref(&args[1], module)?;
        let end = self.traverse_node_ref(&args[2], module)?;
        
        // Process body
        let body_result = self.traverse_node_ref(&args[3], module)?;
        
        // Create for operation using SCF dialect
        let mut for_op = MLIROperation::new("scf.for".to_string());
        
        // Add loop bounds
        if let (Some(start_name), Some(end_name)) = (start, end) {
            let start_value = MLIRValue::new(start_name, MLIRType::Index);
            let end_value = MLIRValue::new(end_name, MLIRType::Index);
            for_op.add_operand(start_value);
            for_op.add_operand(end_value);
        }
        
        module.add_operation(for_op)?;
        
        // Exit loop scope
        self.exit_scope();
        
        Ok(None) // For loops don't return values
    }

    /// Generate unique SSA value name
    fn generate_value_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.value_counter);
        self.value_counter += 1;
        name
    }

    /// Enter new scope
    fn enter_scope(&mut self) {
        self.symbol_table.enter_scope();
        self.scope_depth += 1;
    }

    /// Exit current scope
    fn exit_scope(&mut self) {
        self.symbol_table.exit_scope();
        self.scope_depth = self.scope_depth.saturating_sub(1);
    }

    /// Resolve node reference to actual node
    fn resolve_node_ref<'b>(&'b self, node_ref: &'b ASTNodeRef) -> Option<&'b ASTNode> {
        match node_ref {
            ASTNodeRef::Direct(node) => Some(node.as_ref()),
            ASTNodeRef::Id(id) => self.node_cache.get(id),
            ASTNodeRef::Label(_label) => {
                // Would need access to AST for full resolution
                None
            }
        }
    }

    /// Traverse node reference
    fn traverse_node_ref(&mut self, node_ref: &ASTNodeRef, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if let Some(node) = self.resolve_node_ref(node_ref) {
            let node = node.clone(); // Clone to avoid borrowing issues
            self.traverse_node(&node, module)
        } else {
            Ok(None)
        }
    }

    /// Process function parameters
    fn process_function_parameters(&mut self, params_node: &ASTNodeRef) -> Result<Vec<String>, MLIRError> {
        let mut params = Vec::new();
        
        if let Some(ASTNode::List(param_list)) = self.resolve_node_ref(params_node) {
            let param_list = param_list.clone(); // Clone to avoid borrowing issues
            for param_ref in &param_list {
                if let Some(ASTNode::Atom(AtomValue::Symbol(param_name))) = self.resolve_node_ref(param_ref) {
                    let param_name = param_name.clone(); // Clone to avoid borrowing issues
                    // Add parameter to symbol table
                    let _param_id = self.symbol_table.add_symbol(param_name.clone(), SymbolType::Variable);
                    params.push(param_name);
                }
            }
        }
        
        Ok(params)
    }

    /// Process graph edge
    fn process_graph_edge(&mut self, edge: &crate::compiler::ast::GraphEdge, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Graph edges can represent control flow or data dependencies
        // For now, we'll create a simple dependency annotation
        let mut edge_op = MLIROperation::new("aether.graph_edge".to_string());
        edge_op.add_attribute("from".to_string(), MLIRAttribute::Integer(edge.from as i64));
        edge_op.add_attribute("to".to_string(), MLIRAttribute::Integer(edge.to as i64));
        
        if let Some(label) = &edge.label {
            edge_op.add_attribute("label".to_string(), MLIRAttribute::String(label.clone()));
        }
        
        module.add_operation(edge_op)?;
        Ok(())
    }

    /// Visit tensor operation
    fn visit_tensor_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Tensor operation requires arguments".to_string()));
        }

        // Parse tensor operation: (tensor op-name [shape] [device] [differentiable] operands...)
        let op_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            "create".to_string() // Default to tensor creation
        };

        match op_name.as_str() {
            "create" => self.visit_tensor_create(&args[1..], module),
            "reshape" => self.visit_tensor_reshape(&args[1..], module),
            "slice" => self.visit_tensor_slice(&args[1..], module),
            "concat" => self.visit_tensor_concat(&args[1..], module),
            "transpose" => self.visit_tensor_transpose(&args[1..], module),
            "reduce" => self.visit_tensor_reduce(&args[1..], module),
            _ => self.visit_generic_tensor_op(&op_name, &args[1..], module),
        }
    }

    /// Visit tensor creation operation
    fn visit_tensor_create(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Tensor create requires shape specification".to_string()));
        }

        // Parse shape from first argument
        let shape = self.parse_tensor_shape(&args[0])?;
        
        // Parse optional device (default: "cpu")
        let device = if args.len() > 1 {
            if let Some(ASTNode::Atom(AtomValue::Symbol(dev))) = self.resolve_node_ref(&args[1]) {
                dev.clone()
            } else {
                "cpu".to_string()
            }
        } else {
            "cpu".to_string()
        };

        // Parse optional differentiability (default: false)
        let is_differentiable = if args.len() > 2 {
            if let Some(ASTNode::Atom(AtomValue::Boolean(diff))) = self.resolve_node_ref(&args[2]) {
                *diff
            } else {
                false
            }
        } else {
            false
        };

        // Parse optional element type (default: f64)
        let element_type = if args.len() > 3 {
            self.parse_element_type(&args[3])?
        } else {
            MLIRType::Float { width: 64 }
        };

        // Process initial values if provided
        let mut initial_values = Vec::new();
        for arg in &args[4..] {
            if let Some(value_name) = self.traverse_node_ref(arg, module)? {
                let value = MLIRValue::new(value_name, element_type.clone());
                initial_values.push(value);
            }
        }

        // Create tensor creation operation
        let tensor_op = AetherOps::tensor_create(self.context, element_type.clone(), &shape, &device, is_differentiable)?;
        module.add_operation(tensor_op)?;
        
        let result_name = self.generate_value_name("tensor");
        Ok(Some(result_name))
    }

    /// Visit tensor reshape operation
    fn visit_tensor_reshape(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Tensor reshape requires tensor and new shape".to_string()));
        }

        let tensor_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Tensor required for reshape".to_string()))?;
        
        let new_shape = self.parse_tensor_shape(&args[1])?;

        // Create reshape operation
        let mut reshape_op = MLIROperation::new("aether.tensor_reshape".to_string());
        
        let tensor_value = MLIRValue::new(tensor_name, MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape
            device: "cpu".to_string(),
        });
        reshape_op.add_operand(tensor_value);
        
        // Add new shape as attribute
        let shape_attr = MLIRAttribute::Array(
            new_shape.iter().map(|&dim| MLIRAttribute::Integer(dim)).collect()
        );
        reshape_op.add_attribute("new_shape".to_string(), shape_attr);
        
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: new_shape,
            device: "cpu".to_string(),
        };
        let result = MLIRValue::new(self.generate_value_name("reshaped"), result_type);
        reshape_op.add_result(result);
        
        module.add_operation(reshape_op)?;
        
        let result_name = self.generate_value_name("reshaped");
        Ok(Some(result_name))
    }

    /// Visit tensor slice operation
    fn visit_tensor_slice(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Tensor slice requires tensor and slice specification".to_string()));
        }

        let tensor_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Tensor required for slice".to_string()))?;

        // Parse slice specification: [start:end, start:end, ...]
        let slice_spec = self.parse_slice_specification(&args[1])?;

        let mut slice_op = MLIROperation::new("aether.tensor_slice".to_string());
        
        let tensor_value = MLIRValue::new(tensor_name, MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape
            device: "cpu".to_string(),
        });
        slice_op.add_operand(tensor_value);
        
        // Add slice specification as attributes
        for (i, (start, end)) in slice_spec.iter().enumerate() {
            slice_op.add_attribute(format!("slice_{}_start", i), MLIRAttribute::Integer(*start));
            slice_op.add_attribute(format!("slice_{}_end", i), MLIRAttribute::Integer(*end));
        }
        
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape after slicing
            device: "cpu".to_string(),
        };
        let result = MLIRValue::new(self.generate_value_name("sliced"), result_type);
        slice_op.add_result(result);
        
        module.add_operation(slice_op)?;
        
        let result_name = self.generate_value_name("sliced");
        Ok(Some(result_name))
    }

    /// Visit tensor concatenation operation
    fn visit_tensor_concat(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Tensor concat requires at least two tensors".to_string()));
        }

        // Parse axis (default: 0)
        let axis = if let Some(ASTNode::Atom(AtomValue::Number(ax))) = self.resolve_node_ref(&args[0]) {
            *ax as i64
        } else {
            0
        };

        // Process tensor operands
        let mut operands = Vec::new();
        for arg in &args[1..] {
            if let Some(tensor_name) = self.traverse_node_ref(arg, module)? {
                let tensor_value = MLIRValue::new(tensor_name, MLIRType::AetherTensor {
                    element_type: Box::new(MLIRType::Float { width: 64 }),
                    shape: vec![-1], // Dynamic shape
                    device: "cpu".to_string(),
                });
                operands.push(tensor_value);
            }
        }

        let mut concat_op = MLIROperation::new("aether.tensor_concat".to_string());
        concat_op.add_attribute("axis".to_string(), MLIRAttribute::Integer(axis));
        
        for operand in operands {
            concat_op.add_operand(operand);
        }
        
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape after concatenation
            device: "cpu".to_string(),
        };
        let result = MLIRValue::new(self.generate_value_name("concat"), result_type);
        concat_op.add_result(result);
        
        module.add_operation(concat_op)?;
        
        let result_name = self.generate_value_name("concat");
        Ok(Some(result_name))
    }

    /// Visit tensor transpose operation
    fn visit_tensor_transpose(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Tensor transpose requires tensor".to_string()));
        }

        let tensor_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Tensor required for transpose".to_string()))?;

        // Parse optional permutation (default: reverse all dimensions)
        let permutation = if args.len() > 1 {
            self.parse_permutation(&args[1])?
        } else {
            vec![] // Empty means default transpose
        };

        let mut transpose_op = MLIROperation::new("aether.tensor_transpose".to_string());
        
        let tensor_value = MLIRValue::new(tensor_name, MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape
            device: "cpu".to_string(),
        });
        transpose_op.add_operand(tensor_value);
        
        if !permutation.is_empty() {
            let perm_attr = MLIRAttribute::Array(
                permutation.iter().map(|&dim| MLIRAttribute::Integer(dim)).collect()
            );
            transpose_op.add_attribute("permutation".to_string(), perm_attr);
        }
        
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape after transpose
            device: "cpu".to_string(),
        };
        let result = MLIRValue::new(self.generate_value_name("transposed"), result_type);
        transpose_op.add_result(result);
        
        module.add_operation(transpose_op)?;
        
        let result_name = self.generate_value_name("transposed");
        Ok(Some(result_name))
    }

    /// Visit tensor reduction operation
    fn visit_tensor_reduce(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Tensor reduce requires reduction type and tensor".to_string()));
        }

        let reduce_type = if let Some(ASTNode::Atom(AtomValue::Symbol(op))) = self.resolve_node_ref(&args[0]) {
            op.clone()
        } else {
            return Err(MLIRError::ConversionError("Reduction type must be a symbol".to_string()));
        };

        let tensor_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Tensor required for reduce".to_string()))?;

        // Parse optional axes (default: reduce all)
        let axes = if args.len() > 2 {
            self.parse_reduction_axes(&args[2])?
        } else {
            vec![] // Empty means reduce all axes
        };

        let mut reduce_op = MLIROperation::new("aether.tensor_reduce".to_string());
        reduce_op.add_attribute("reduce_type".to_string(), MLIRAttribute::String(reduce_type));
        
        let tensor_value = MLIRValue::new(tensor_name, MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape
            device: "cpu".to_string(),
        });
        reduce_op.add_operand(tensor_value);
        
        if !axes.is_empty() {
            let axes_attr = MLIRAttribute::Array(
                axes.iter().map(|&axis| MLIRAttribute::Integer(axis)).collect()
            );
            reduce_op.add_attribute("axes".to_string(), axes_attr);
        }
        
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape after reduction
            device: "cpu".to_string(),
        };
        let result = MLIRValue::new(self.generate_value_name("reduced"), result_type);
        reduce_op.add_result(result);
        
        module.add_operation(reduce_op)?;
        
        let result_name = self.generate_value_name("reduced");
        Ok(Some(result_name))
    }

    /// Visit generic tensor operation
    fn visit_generic_tensor_op(&mut self, op_name: &str, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        // Process operands
        let mut operands = Vec::new();
        for arg in args {
            if let Some(value_name) = self.traverse_node_ref(arg, module)? {
                let operand = MLIRValue::new(value_name, MLIRType::AetherTensor {
                    element_type: Box::new(MLIRType::Float { width: 64 }),
                    shape: vec![-1], // Dynamic shape
                    device: "cpu".to_string(),
                });
                operands.push(operand);
            }
        }

        // Create generic tensor operation
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Dynamic shape
            device: "cpu".to_string(),
        };

        let tensor_op = AetherOps::tensor_op(self.context, op_name, operands, result_type, HashMap::new())?;
        module.add_operation(tensor_op)?;
        
        let result_name = self.generate_value_name("tensor_result");
        Ok(Some(result_name))
    }

    /// Visit matrix multiplication
    fn visit_matrix_multiplication(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Matrix multiplication requires two operands".to_string()));
        }

        // Process operands
        let lhs_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Left operand required for matmul".to_string()))?;
        let rhs_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Right operand required for matmul".to_string()))?;

        // Create operand values
        let lhs_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![2, 3], // Example shape
            device: "cpu".to_string(),
        };
        let rhs_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![3, 4], // Example shape
            device: "cpu".to_string(),
        };

        let lhs = MLIRValue::new(lhs_name, lhs_type);
        let rhs = MLIRValue::new(rhs_name, rhs_type);

        // Create result type
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![2, 4], // Result shape
            device: "cpu".to_string(),
        };

        // Create matmul operation
        let matmul_op = AetherOps::matmul(self.context, lhs, rhs, result_type, false, false)?;
        module.add_operation(matmul_op)?;
        
        let result_name = self.generate_value_name("matmul_result");
        Ok(Some(result_name))
    }

    /// Visit autodiff marker
    fn visit_autodiff_marker(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Autodiff marker requires function argument".to_string()));
        }

        // Parse autodiff mode and function
        let (mode, function_ref, input_ref) = if args.len() == 1 {
            // (autodiff function) - default to reverse mode
            ("reverse", &args[0], None)
        } else if args.len() == 2 {
            // (autodiff mode function) or (autodiff function input)
            if let Some(ASTNode::Atom(AtomValue::Symbol(mode_str))) = self.resolve_node_ref(&args[0]) {
                if matches!(mode_str.as_str(), "forward" | "reverse" | "both") {
                    (mode_str.as_str(), &args[1], None)
                } else {
                    // First arg is function, second is input
                    ("reverse", &args[0], Some(&args[1]))
                }
            } else {
                ("reverse", &args[0], Some(&args[1]))
            }
        } else {
            // (autodiff mode function input)
            let mode_str = if let Some(ASTNode::Atom(AtomValue::Symbol(mode))) = self.resolve_node_ref(&args[0]) {
                mode.as_str()
            } else {
                "reverse"
            };
            (mode_str, &args[1], Some(&args[2]))
        };

        match mode {
            "forward" => self.visit_autodiff_forward(function_ref, input_ref, module),
            "reverse" => self.visit_autodiff_reverse(function_ref, input_ref, module),
            "both" => self.visit_autodiff_both(function_ref, input_ref, module),
            _ => Err(MLIRError::ConversionError(format!("Unknown autodiff mode: {}", mode))),
        }
    }

    /// Visit forward-mode automatic differentiation
    fn visit_autodiff_forward(&mut self, function_ref: &ASTNodeRef, input_ref: Option<&ASTNodeRef>, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        let function_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(function_ref) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        // Create function value
        let function_value = MLIRValue::new(format!("@{}", function_name), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![MLIRType::Float { width: 64 }],
        });

        // Process input and tangent
        let (input_value, tangent_value) = if let Some(input_ref) = input_ref {
            let input_name = self.traverse_node_ref(input_ref, module)?
                .ok_or_else(|| MLIRError::ConversionError("Input required for forward autodiff".to_string()))?;
            
            let input_val = MLIRValue::new(input_name, MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![-1],
                device: "cpu".to_string(),
            });

            // Create unit tangent vector
            let tangent_name = self.generate_value_name("tangent");
            let mut tangent_op = MLIROperation::new("aether.tensor_create".to_string());
            tangent_op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![MLIRAttribute::Integer(-1)]));
            tangent_op.add_attribute("device".to_string(), MLIRAttribute::String("cpu".to_string()));
            tangent_op.add_attribute("is_differentiable".to_string(), MLIRAttribute::Boolean(true));
            tangent_op.add_attribute("init_value".to_string(), MLIRAttribute::Float(1.0));
            
            let tangent_result = MLIRValue::new(tangent_name.clone(), MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![-1],
                device: "cpu".to_string(),
            });
            tangent_op.add_result(tangent_result.clone());
            module.add_operation(tangent_op)?;

            (input_val, tangent_result)
        } else {
            // Create default input and tangent
            let input_val = MLIRValue::new("input".to_string(), MLIRType::Float { width: 64 });
            let tangent_val = MLIRValue::new("tangent".to_string(), MLIRType::Float { width: 64 });
            (input_val, tangent_val)
        };

        // Create forward autodiff operation
        let result_type = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![MLIRType::Float { width: 64 }, MLIRType::Float { width: 64 }], // value and derivative
        };

        let forward_op = AetherOps::autodiff_forward(self.context, function_value, input_value, tangent_value, result_type)?;
        module.add_operation(forward_op)?;
        
        let result_name = self.generate_value_name("forward_autodiff");
        Ok(Some(result_name))
    }

    /// Visit reverse-mode automatic differentiation
    fn visit_autodiff_reverse(&mut self, function_ref: &ASTNodeRef, input_ref: Option<&ASTNodeRef>, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        let function_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(function_ref) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        // Create function value
        let function_value = MLIRValue::new(format!("@{}", function_name), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![MLIRType::Float { width: 64 }],
        });

        // Process input
        let input_value = if let Some(input_ref) = input_ref {
            let input_name = self.traverse_node_ref(input_ref, module)?
                .ok_or_else(|| MLIRError::ConversionError("Input required for reverse autodiff".to_string()))?;
            
            MLIRValue::new(input_name, MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![-1],
                device: "cpu".to_string(),
            })
        } else {
            MLIRValue::new("input".to_string(), MLIRType::Float { width: 64 })
        };

        // Create reverse autodiff operation
        let result_type = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![MLIRType::Float { width: 64 }, MLIRType::Function {
                inputs: vec![MLIRType::Float { width: 64 }],
                outputs: vec![MLIRType::Float { width: 64 }],
            }], // value and gradient function
        };

        let reverse_op = AetherOps::autodiff_reverse(self.context, function_value, input_value, result_type)?;
        module.add_operation(reverse_op)?;
        
        let result_name = self.generate_value_name("reverse_autodiff");
        Ok(Some(result_name))
    }

    /// Visit both forward and reverse mode automatic differentiation
    fn visit_autodiff_both(&mut self, function_ref: &ASTNodeRef, input_ref: Option<&ASTNodeRef>, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        // Generate both forward and reverse mode operations
        let forward_result = self.visit_autodiff_forward(function_ref, input_ref, module)?;
        let reverse_result = self.visit_autodiff_reverse(function_ref, input_ref, module)?;

        // Create a combined result operation
        let mut both_op = MLIROperation::new("aether.autodiff_both".to_string());
        
        if let Some(forward_name) = forward_result {
            let forward_val = MLIRValue::new(forward_name, MLIRType::Function {
                inputs: vec![MLIRType::Float { width: 64 }],
                outputs: vec![MLIRType::Float { width: 64 }, MLIRType::Float { width: 64 }],
            });
            both_op.add_operand(forward_val);
        }

        if let Some(reverse_name) = reverse_result {
            let reverse_val = MLIRValue::new(reverse_name, MLIRType::Function {
                inputs: vec![MLIRType::Float { width: 64 }],
                outputs: vec![MLIRType::Float { width: 64 }, MLIRType::Function {
                    inputs: vec![MLIRType::Float { width: 64 }],
                    outputs: vec![MLIRType::Float { width: 64 }],
                }],
            });
            both_op.add_operand(reverse_val);
        }

        let result_type = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![
                MLIRType::Float { width: 64 }, // value
                MLIRType::Float { width: 64 }, // forward derivative
                MLIRType::Function {           // reverse gradient function
                    inputs: vec![MLIRType::Float { width: 64 }],
                    outputs: vec![MLIRType::Float { width: 64 }],
                }
            ],
        };
        let result = MLIRValue::new(self.generate_value_name("both_autodiff"), result_type);
        both_op.add_result(result);

        module.add_operation(both_op)?;
        
        let result_name = self.generate_value_name("both_autodiff");
        Ok(Some(result_name))
    }

    /// Visit gradient computation
    fn visit_gradient_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Gradient requires function and input".to_string()));
        }

        let function_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let input_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Input required for gradient".to_string()))?;

        let function_value = MLIRValue::new(format!("@{}", function_name), MLIRType::Function {
            inputs: vec![MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![-1],
                device: "cpu".to_string(),
            }],
            outputs: vec![MLIRType::Float { width: 64 }],
        });

        let input_value = MLIRValue::new(input_name, MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1],
            device: "cpu".to_string(),
        });

        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![-1], // Same shape as input
            device: "cpu".to_string(),
        };

        let gradient_op = AetherOps::gradient(self.context, function_value, input_value, result_type)?;
        module.add_operation(gradient_op)?;
        
        let result_name = self.generate_value_name("gradient");
        Ok(Some(result_name))
    }

    /// Visit probabilistic variable
    fn visit_probabilistic_variable(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Probabilistic variable requires name and distribution".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        let distribution_spec = &args[1];
        let (distribution, parameters) = self.parse_distribution_specification(distribution_spec, module)?;

        // Create probabilistic variable operation using the operation builder
        let prob_var_op = self.operation_builder.build_prob_var(&name, distribution, parameters)?;
        module.add_operation(prob_var_op)?;

        // Add to symbol table
        let _var_id = self.symbol_table.add_symbol(name.clone(), SymbolType::Variable);
        
        let result_name = self.generate_value_name(&format!("prob_{}", name));
        Ok(Some(result_name))
    }

    /// Parse distribution specification from AST
    fn parse_distribution_specification(&mut self, dist_spec: &ASTNodeRef, module: &mut MLIRModule) -> Result<(DistributionType, Vec<MLIRValue>), MLIRError> {
        // Clone the node to avoid borrowing issues
        let resolved_node = if let Some(node) = self.resolve_node_ref(dist_spec) {
            node.clone()
        } else {
            return Err(MLIRError::ConversionError("Invalid distribution specification".to_string()));
        };

        match resolved_node {
            ASTNode::Atom(AtomValue::Symbol(dist_name)) => {
                // Simple distribution name with default parameters
                let distribution = match dist_name.as_str() {
                    "normal" => DistributionType::Normal { mean: 0.0, std: 1.0 },
                    "uniform" => DistributionType::Uniform { min: 0.0, max: 1.0 },
                    "bernoulli" => DistributionType::Bernoulli { p: 0.5 },
                    _ => DistributionType::Custom(dist_name),
                };
                Ok((distribution, vec![]))
            }
            ASTNode::List(dist_args) => {
                // Distribution with parameters: (normal mean std) or (uniform min max)
                if dist_args.is_empty() {
                    return Err(MLIRError::ConversionError("Distribution specification cannot be empty".to_string()));
                }

                let dist_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&dist_args[0]) {
                    name.clone()
                } else {
                    return Err(MLIRError::ConversionError("Distribution name must be a symbol".to_string()));
                };

                // Process parameters
                let mut parameters = Vec::new();
                for param_ref in &dist_args[1..] {
                    if let Some(param_name) = self.traverse_node_ref(param_ref, module)? {
                        let param_value = MLIRValue::new(param_name, MLIRType::Float { width: 64 });
                        parameters.push(param_value);
                    }
                }

                // Create distribution with parsed parameters
                let distribution = match dist_name.as_str() {
                    "normal" => {
                        let mean = if parameters.len() > 0 { 
                            self.extract_float_value(&parameters[0])? 
                        } else { 0.0 };
                        let std = if parameters.len() > 1 { 
                            self.extract_float_value(&parameters[1])? 
                        } else { 1.0 };
                        DistributionType::Normal { mean, std }
                    }
                    "uniform" => {
                        let min = if parameters.len() > 0 { 
                            self.extract_float_value(&parameters[0])? 
                        } else { 0.0 };
                        let max = if parameters.len() > 1 { 
                            self.extract_float_value(&parameters[1])? 
                        } else { 1.0 };
                        DistributionType::Uniform { min, max }
                    }
                    "bernoulli" => {
                        let p = if parameters.len() > 0 { 
                            self.extract_float_value(&parameters[0])? 
                        } else { 0.5 };
                        DistributionType::Bernoulli { p }
                    }
                    "categorical" => {
                        let probs = parameters.iter()
                            .map(|p| self.extract_float_value(p))
                            .collect::<Result<Vec<_>, _>>()?;
                        DistributionType::Categorical { probs }
                    }
                    _ => DistributionType::Custom(dist_name),
                };

                Ok((distribution, parameters))
            }
            _ => Err(MLIRError::ConversionError("Invalid distribution specification".to_string())),
        }
    }

    /// Extract float value from MLIR value (simplified)
    fn extract_float_value(&self, _value: &MLIRValue) -> Result<f64, MLIRError> {
        // In a real implementation, this would extract the actual value
        // For now, return a default value
        Ok(1.0)
    }

    /// Visit sample operation
    fn visit_sample_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Sample operation requires probabilistic variable".to_string()));
        }

        let prob_var_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Probabilistic variable required for sample".to_string()))?;

        // Parse optional sample count (default: 1)
        let sample_count = if args.len() > 1 {
            if let Some(ASTNode::Atom(AtomValue::Number(count))) = self.resolve_node_ref(&args[1]) {
                *count as i64
            } else {
                1
            }
        } else {
            1
        };

        // Parse optional random seed
        let seed = if args.len() > 2 {
            if let Some(ASTNode::Atom(AtomValue::Number(s))) = self.resolve_node_ref(&args[2]) {
                Some(*s as u64)
            } else {
                None
            }
        } else {
            None
        };

        let prob_var = MLIRValue::new(prob_var_name, MLIRType::AetherProbabilistic {
            distribution: "normal".to_string(),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        });

        // Create enhanced sample operation
        let mut sample_op = MLIROperation::new("aether.sample".to_string());
        sample_op.add_operand(prob_var);
        sample_op.add_attribute("count".to_string(), MLIRAttribute::Integer(sample_count));
        
        if let Some(seed_val) = seed {
            sample_op.add_attribute("seed".to_string(), MLIRAttribute::Integer(seed_val as i64));
        }

        // Determine result type based on sample count
        let result_type = if sample_count == 1 {
            MLIRType::Float { width: 64 }
        } else {
            MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 64 }),
                shape: vec![sample_count],
                device: "cpu".to_string(),
            }
        };

        let result = MLIRValue::new(self.generate_value_name("sample"), result_type);
        sample_op.add_result(result);
        module.add_operation(sample_op)?;
        
        let result_name = self.generate_value_name("sample");
        Ok(Some(result_name))
    }

    /// Visit observe operation
    fn visit_observe_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Observe operation requires variable and observed value".to_string()));
        }

        let prob_var_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Probabilistic variable required for observe".to_string()))?;
        let observed_value_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Observed value required for observe".to_string()))?;

        // Parse optional observation weight (default: 1.0)
        let weight = if args.len() > 2 {
            if let Some(ASTNode::Atom(AtomValue::Number(w))) = self.resolve_node_ref(&args[2]) {
                *w
            } else {
                1.0
            }
        } else {
            1.0
        };

        let prob_var = MLIRValue::new(prob_var_name, MLIRType::AetherProbabilistic {
            distribution: "normal".to_string(),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        });
        let observed_value = MLIRValue::new(observed_value_name, MLIRType::Float { width: 64 });

        // Create enhanced observe operation
        let mut observe_op = MLIROperation::new("aether.observe".to_string());
        observe_op.add_operand(prob_var);
        observe_op.add_operand(observed_value);
        observe_op.add_attribute("weight".to_string(), MLIRAttribute::Float(weight));
        
        // Add optional log-likelihood result
        if args.len() > 3 {
            let log_likelihood_type = MLIRType::Float { width: 64 };
            let log_likelihood = MLIRValue::new(self.generate_value_name("log_likelihood"), log_likelihood_type);
            observe_op.add_result(log_likelihood);
        }

        module.add_operation(observe_op)?;
        
        // Return log-likelihood if requested
        if args.len() > 3 {
            let result_name = self.generate_value_name("log_likelihood");
            Ok(Some(result_name))
        } else {
            Ok(None) // Standard observe operations don't produce values
        }
    }

    /// Visit linear type annotation
    fn visit_linear_type(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Linear type requires inner type".to_string()));
        }

        // Parse linear type specification: (linear inner-type [ownership] [lifetime])
        let inner_type = self.parse_type_specification(&args[0])?;
        
        // Parse optional ownership (default: "owned")
        let ownership = if args.len() > 1 {
            if let Some(ASTNode::Atom(AtomValue::Symbol(own))) = self.resolve_node_ref(&args[1]) {
                match own.as_str() {
                    "owned" => LinearOwnership::Owned,
                    "borrowed" => LinearOwnership::Borrowed(crate::compiler::types::Lifetime {
                        name: "static".to_string(),
                        scope_level: 0,
                        start_location: None,
                        end_location: None,
                    }),
                    "moved" => LinearOwnership::Moved,
                    _ => LinearOwnership::Owned,
                }
            } else {
                LinearOwnership::Owned
            }
        } else {
            LinearOwnership::Owned
        };

        // Parse optional lifetime
        let lifetime = if args.len() > 2 {
            if let Some(ASTNode::Atom(AtomValue::Symbol(lt))) = self.resolve_node_ref(&args[2]) {
                Some(lt.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Parse optional allocation site
        let allocation_site = if args.len() > 3 {
            if let Some(ASTNode::Atom(AtomValue::String(site))) = self.resolve_node_ref(&args[3]) {
                Some(site.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Create linear allocation operation
        let linear_alloc_op = AetherOps::linear_alloc(
            self.context, 
            inner_type.clone(), 
            None, // No size specified
            allocation_site.as_deref().unwrap_or("unknown")
        )?;
        module.add_operation(linear_alloc_op)?;

        // Create linear type annotation operation
        let mut linear_type_op = MLIROperation::new("aether.linear_type".to_string());
        linear_type_op.add_attribute("inner_type".to_string(), MLIRAttribute::String(format!("{:?}", inner_type)));
        linear_type_op.add_attribute("ownership".to_string(), MLIRAttribute::String(format!("{:?}", ownership)));
        
        if let Some(lt) = &lifetime {
            linear_type_op.add_attribute("lifetime".to_string(), MLIRAttribute::String(lt.clone()));
        }
        
        if let Some(site) = &allocation_site {
            linear_type_op.add_attribute("allocation_site".to_string(), MLIRAttribute::String(site.clone()));
        }

        let result_type = MLIRType::AetherLinear {
            inner_type: Box::new(inner_type),
        };
        let result = MLIRValue::new(self.generate_value_name("linear"), result_type);
        linear_type_op.add_result(result);
        
        module.add_operation(linear_type_op)?;
        
        let result_name = self.generate_value_name("linear");
        Ok(Some(result_name))
    }

    /// Visit move operation
    fn visit_move_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Move operation requires source value".to_string()));
        }

        let source_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Source value required for move".to_string()))?;

        // Parse optional destination variable name
        let dest_name = if args.len() > 1 {
            if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[1]) {
                Some(name.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Infer source type (simplified - assume linear type)
        let source_type = MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        };

        let source = MLIRValue::new(source_name.clone(), source_type.clone());

        // Create move operation with ownership transfer tracking
        let mut move_op = MLIROperation::new("aether.linear_move".to_string());
        move_op.add_operand(source);
        move_op.add_attribute("source_name".to_string(), MLIRAttribute::String(source_name.clone()));
        
        if let Some(dest) = &dest_name {
            move_op.add_attribute("dest_name".to_string(), MLIRAttribute::String(dest.clone()));
        }

        // Add move semantics attributes
        move_op.add_attribute("move_type".to_string(), MLIRAttribute::String("ownership_transfer".to_string()));
        move_op.add_attribute("invalidates_source".to_string(), MLIRAttribute::Boolean(true));

        let result_name = dest_name.as_ref().map(|s| s.clone()).unwrap_or_else(|| self.generate_value_name("moved"));
        let result = MLIRValue::new(result_name.clone(), source_type);
        move_op.add_result(result);

        module.add_operation(move_op)?;

        // Update symbol table to mark source as moved
        if let Some(_symbol) = self.symbol_table.lookup(&source_name) {
            // In a real implementation, we'd mark the symbol as moved/invalid
        }

        // Add destination to symbol table if named
        if let Some(dest) = dest_name {
            let _dest_id = self.symbol_table.add_symbol(dest, SymbolType::Variable);
        }
        
        Ok(Some(result_name))
    }

    /// Visit drop operation
    fn visit_drop_operation(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Drop operation requires value to drop".to_string()));
        }

        let value_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Value required for drop".to_string()))?;

        // Parse optional drop mode (default: "immediate")
        let drop_mode = if args.len() > 1 {
            if let Some(ASTNode::Atom(AtomValue::Symbol(mode))) = self.resolve_node_ref(&args[1]) {
                mode.clone()
            } else {
                "immediate".to_string()
            }
        } else {
            "immediate".to_string()
        };

        let value_type = MLIRType::AetherLinear {
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        };

        let value = MLIRValue::new(value_name.clone(), value_type);

        // Create enhanced drop operation
        let mut drop_op = MLIROperation::new("aether.linear_drop".to_string());
        drop_op.add_operand(value);
        drop_op.add_attribute("value_name".to_string(), MLIRAttribute::String(value_name.clone()));
        drop_op.add_attribute("drop_mode".to_string(), MLIRAttribute::String(drop_mode));
        drop_op.add_attribute("deterministic".to_string(), MLIRAttribute::Boolean(true));

        module.add_operation(drop_op)?;

        // Update symbol table to mark value as dropped
        if let Some(_symbol) = self.symbol_table.lookup(&value_name) {
            // In a real implementation, we'd mark the symbol as dropped/invalid
        }
        
        Ok(None) // Drop operations don't produce values
    }

    /// Visit spawn actor
    fn visit_spawn_actor(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Spawn operation requires actor type".to_string()));
        }

        let actor_type = MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }], // Message type
            outputs: vec![MLIRType::Float { width: 64 }], // State type
        };

        let initial_state = if args.len() > 1 {
            let state_name = self.traverse_node_ref(&args[1], module)?
                .ok_or_else(|| MLIRError::ConversionError("Initial state value required".to_string()))?;
            Some(MLIRValue::new(state_name, MLIRType::Float { width: 64 }))
        } else {
            None
        };

        let result_type = actor_type.clone();
        let spawn_op = AetherOps::spawn_actor(self.context, actor_type, initial_state, result_type)?;
        module.add_operation(spawn_op)?;
        
        let result_name = self.generate_value_name("actor_ref");
        Ok(Some(result_name))
    }

    /// Visit send message
    fn visit_send_message(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Send operation requires actor and message".to_string()));
        }

        let actor_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Actor reference required for send".to_string()))?;
        let message_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Message required for send".to_string()))?;

        let actor_ref = MLIRValue::new(actor_name, MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 64 }],
            outputs: vec![MLIRType::Float { width: 64 }],
        });
        let message = MLIRValue::new(message_name, MLIRType::Float { width: 64 });

        let send_op = AetherOps::send_message(self.context, actor_ref, message)?;
        module.add_operation(send_op)?;
        
        Ok(None) // Send operations don't produce values
    }

    /// Visit parallel for
    fn visit_parallel_for(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() < 3 {
            return Err(MLIRError::ConversionError("Parallel for requires bounds and body".to_string()));
        }

        let lower_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Lower bound required for parallel for".to_string()))?;
        let upper_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Upper bound required for parallel for".to_string()))?;

        let lower_bound = MLIRValue::new(lower_name, MLIRType::Index);
        let upper_bound = MLIRValue::new(upper_name, MLIRType::Index);

        // Create a dummy body function for now
        let body_function = MLIRValue::new("body_func".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Index],
            outputs: vec![],
        });

        let step = None; // Optional step parameter

        let parallel_for_op = AetherOps::parallel_for(self.context, lower_bound, upper_bound, step, body_function)?;
        module.add_operation(parallel_for_op)?;
        
        Ok(None) // Parallel for operations don't produce values
    }

    /// Visit arithmetic operation
    fn visit_arithmetic_operation(&mut self, op_name: &str, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() != 2 {
            return Err(MLIRError::ConversionError(format!("Arithmetic operation {} requires exactly 2 arguments", op_name)));
        }

        let lhs_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Left operand required for arithmetic".to_string()))?;
        let rhs_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Right operand required for arithmetic".to_string()))?;

        // Infer operand types
        let lhs_type = self.infer_expression_type(&args[0])?;
        let rhs_type = self.infer_expression_type(&args[1])?;
        
        // Determine result type and operation variant
        let (mlir_op_name, result_type) = match (&lhs_type, &rhs_type) {
            (MLIRType::Integer { width: w1, signed: s1 }, MLIRType::Integer { width: w2, signed: s2 }) => {
                // Integer arithmetic
                let result_width = (*w1).max(*w2);
                let result_signed = *s1 || *s2;
                let result_type = MLIRType::Integer { width: result_width, signed: result_signed };
                
                let op_name = match op_name {
                    "+" => "arith.addi",
                    "-" => "arith.subi", 
                    "*" => "arith.muli",
                    "/" => if result_signed { "arith.divsi" } else { "arith.divui" },
                    "%" => if result_signed { "arith.remsi" } else { "arith.remui" },
                    _ => return Err(MLIRError::ConversionError(format!("Unknown arithmetic operation: {}", op_name))),
                };
                (op_name, result_type)
            }
            _ => {
                // Default to floating point arithmetic
                let op_name = match op_name {
                    "+" => "arith.addf",
                    "-" => "arith.subf",
                    "*" => "arith.mulf",
                    "/" => "arith.divf",
                    "%" => "arith.remf",
                    _ => return Err(MLIRError::ConversionError(format!("Unknown arithmetic operation: {}", op_name))),
                };
                (op_name, MLIRType::Float { width: 64 })
            }
        };

        // Create arithmetic operation
        let mut arith_op = MLIROperation::new(mlir_op_name.to_string());
        
        // Add operands with inferred types
        let lhs = MLIRValue::new(lhs_name, lhs_type);
        let rhs = MLIRValue::new(rhs_name, rhs_type);
        arith_op.add_operand(lhs);
        arith_op.add_operand(rhs);
        
        // Add result
        let result_name = self.generate_value_name(&format!("{}_result", op_name));
        let result = MLIRValue::new(result_name.clone(), result_type);
        arith_op.add_result(result);
        
        module.add_operation(arith_op)?;
        
        Ok(Some(result_name))
    }

    /// Visit comparison operation
    fn visit_comparison_operation(&mut self, op_name: &str, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.len() != 2 {
            return Err(MLIRError::ConversionError(format!("Comparison operation {} requires exactly 2 arguments", op_name)));
        }

        let lhs_name = self.traverse_node_ref(&args[0], module)?
            .ok_or_else(|| MLIRError::ConversionError("Left operand required for comparison".to_string()))?;
        let rhs_name = self.traverse_node_ref(&args[1], module)?
            .ok_or_else(|| MLIRError::ConversionError("Right operand required for comparison".to_string()))?;

        let mlir_op_name = match op_name {
            "=" | "==" => "arith.cmpf",
            "!=" => "arith.cmpf",
            "<" => "arith.cmpf",
            ">" => "arith.cmpf",
            "<=" => "arith.cmpf",
            ">=" => "arith.cmpf",
            _ => return Err(MLIRError::ConversionError(format!("Unknown comparison operation: {}", op_name))),
        };

        // Create comparison operation
        let mut cmp_op = MLIROperation::new(mlir_op_name.to_string());
        
        // Add predicate attribute
        let predicate = match op_name {
            "=" | "==" => "eq",
            "!=" => "ne",
            "<" => "lt",
            ">" => "gt",
            "<=" => "le",
            ">=" => "ge",
            _ => "eq",
        };
        cmp_op.add_attribute("predicate".to_string(), MLIRAttribute::String(predicate.to_string()));
        
        // Add operands
        let lhs = MLIRValue::new(lhs_name, MLIRType::Float { width: 64 });
        let rhs = MLIRValue::new(rhs_name, MLIRType::Float { width: 64 });
        cmp_op.add_operand(lhs);
        cmp_op.add_operand(rhs);
        
        // Add result
        let result_name = self.generate_value_name("cmp_result");
        let result = MLIRValue::new(result_name.clone(), MLIRType::Integer { width: 1, signed: false });
        cmp_op.add_result(result);
        
        module.add_operation(cmp_op)?;
        
        Ok(Some(result_name))
    }

    /// Visit logical operation
    fn visit_logical_operation(&mut self, op_name: &str, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        match op_name {
            "not" => {
                if args.len() != 1 {
                    return Err(MLIRError::ConversionError("Not operation requires exactly 1 argument".to_string()));
                }
                
                let operand_name = self.traverse_node_ref(&args[0], module)?
                    .ok_or_else(|| MLIRError::ConversionError("Operand required for not".to_string()))?;
                
                // Create not operation
                let mut not_op = MLIROperation::new("arith.xori".to_string());
                
                let operand = MLIRValue::new(operand_name, MLIRType::Integer { width: 1, signed: false });
                let true_const = MLIRValue::new("true".to_string(), MLIRType::Integer { width: 1, signed: false });
                not_op.add_operand(operand);
                not_op.add_operand(true_const);
                
                let result_name = self.generate_value_name("not_result");
                let result = MLIRValue::new(result_name.clone(), MLIRType::Integer { width: 1, signed: false });
                not_op.add_result(result);
                
                module.add_operation(not_op)?;
                Ok(Some(result_name))
            }
            "and" | "or" => {
                if args.len() != 2 {
                    return Err(MLIRError::ConversionError(format!("{} operation requires exactly 2 arguments", op_name)));
                }
                
                let lhs_name = self.traverse_node_ref(&args[0], module)?
                    .ok_or_else(|| MLIRError::ConversionError("Left operand required for logical operation".to_string()))?;
                let rhs_name = self.traverse_node_ref(&args[1], module)?
                    .ok_or_else(|| MLIRError::ConversionError("Right operand required for logical operation".to_string()))?;
                
                let mlir_op_name = match op_name {
                    "and" => "arith.andi",
                    "or" => "arith.ori",
                    _ => unreachable!(),
                };
                
                // Create logical operation
                let mut logical_op = MLIROperation::new(mlir_op_name.to_string());
                
                let lhs = MLIRValue::new(lhs_name, MLIRType::Integer { width: 1, signed: false });
                let rhs = MLIRValue::new(rhs_name, MLIRType::Integer { width: 1, signed: false });
                logical_op.add_operand(lhs);
                logical_op.add_operand(rhs);
                
                let result_name = self.generate_value_name(&format!("{}_result", op_name));
                let result = MLIRValue::new(result_name.clone(), MLIRType::Integer { width: 1, signed: false });
                logical_op.add_result(result);
                
                module.add_operation(logical_op)?;
                Ok(Some(result_name))
            }
            _ => Err(MLIRError::ConversionError(format!("Unknown logical operation: {}", op_name)))
        }
    }

    /// Visit function call
    fn visit_function_call(&mut self, args: &[ASTNodeRef], module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Function call requires function name".to_string()));
        }

        let func_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        // Process arguments with type inference
        let mut call_args = Vec::new();
        let mut arg_types = Vec::new();
        for arg in &args[1..] {
            if let Some(arg_name) = self.traverse_node_ref(arg, module)? {
                let arg_type = self.infer_expression_type(arg)?;
                let arg_value = MLIRValue::new(arg_name, arg_type.clone());
                call_args.push(arg_value);
                arg_types.push(arg_type);
            }
        }

        // Infer return type based on function name and arguments
        let return_type = self.infer_function_return_type(&func_name, &arg_types)?;

        // Create function call operation
        let mut call_op = MLIROperation::new("func.call".to_string());
        call_op.add_attribute("callee".to_string(), MLIRAttribute::String(func_name.clone()));
        
        // Add arguments
        for arg in call_args {
            call_op.add_operand(arg);
        }
        
        // Add result if function returns something
        let result_name = self.generate_value_name(&format!("call_{}", func_name));
        let result = MLIRValue::new(result_name.clone(), return_type);
        call_op.add_result(result);
        
        module.add_operation(call_op)?;
        
        Ok(Some(result_name))
    }

    /// Verify all operations in the module
    fn verify_module_operations(&mut self, module: &MLIRModule) -> Result<(), MLIRError> {
        for operation in module.operations() {
            // Basic operation verification
            if operation.name.is_empty() {
                return Err(MLIRError::VerificationError("Operation name cannot be empty".to_string()));
            }
            
            // Verify operation has valid dialect prefix
            if !operation.name.contains('.') {
                return Err(MLIRError::VerificationError(format!("Operation {} must have dialect prefix", operation.name)));
            }
        }
        
        Ok(())
    }

    /// Infer type of an expression from AST node
    pub fn infer_expression_type(&self, node_ref: &ASTNodeRef) -> Result<MLIRType, MLIRError> {
        if let Some(node) = self.resolve_node_ref(node_ref) {
            match node {
                ASTNode::Atom(AtomValue::Number(_)) => Ok(MLIRType::Float { width: 64 }),
                ASTNode::Atom(AtomValue::Boolean(_)) => Ok(MLIRType::Integer { width: 1, signed: false }),
                ASTNode::Atom(AtomValue::String(_)) => Ok(MLIRType::Memref { 
                    element_type: Box::new(MLIRType::Integer { width: 8, signed: false }), 
                    shape: vec![-1] 
                }),
                ASTNode::Atom(AtomValue::Symbol(name)) => {
                    // Look up symbol in symbol table
                    if let Some(symbol) = self.symbol_table.lookup(name) {
                        match symbol.symbol_type {
                            SymbolType::Variable => Ok(MLIRType::Float { width: 64 }), // Default variable type
                            SymbolType::Function => Ok(MLIRType::Function {
                                inputs: vec![],
                                outputs: vec![MLIRType::Float { width: 64 }],
                            }),
                            _ => Ok(MLIRType::Float { width: 64 }),
                        }
                    } else {
                        Ok(MLIRType::Float { width: 64 }) // Default for unknown symbols
                    }
                }
                ASTNode::List(children) if !children.is_empty() => {
                    // For lists, infer type based on operation
                    if let Some(ASTNode::Atom(AtomValue::Symbol(op_name))) = self.resolve_node_ref(&children[0]) {
                        match op_name.as_str() {
                            "+" | "-" | "*" | "/" | "%" => Ok(MLIRType::Float { width: 64 }),
                            "=" | "==" | "!=" | "<" | ">" | "<=" | ">=" => Ok(MLIRType::Integer { width: 1, signed: false }),
                            "and" | "or" | "not" => Ok(MLIRType::Integer { width: 1, signed: false }),
                            "tensor" => Ok(MLIRType::AetherTensor {
                                element_type: Box::new(MLIRType::Float { width: 64 }),
                                shape: vec![1], // Default shape
                                device: "cpu".to_string(),
                            }),
                            _ => Ok(MLIRType::Float { width: 64 }), // Default for unknown operations
                        }
                    } else {
                        Ok(MLIRType::Float { width: 64 })
                    }
                }
                _ => Ok(MLIRType::Float { width: 64 }), // Default fallback
            }
        } else {
            Ok(MLIRType::Float { width: 64 }) // Default for unresolved references
        }
    }

    /// Infer function return type based on name and argument types
    fn infer_function_return_type(&self, func_name: &str, _arg_types: &[MLIRType]) -> Result<MLIRType, MLIRError> {
        // For now, use simple heuristics based on function name
        match func_name {
            // Arithmetic functions return numeric types
            "add" | "sub" | "mul" | "div" | "mod" | "abs" | "sqrt" | "sin" | "cos" | "tan" => {
                Ok(MLIRType::Float { width: 64 })
            }
            // Comparison functions return boolean
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                Ok(MLIRType::Integer { width: 1, signed: false })
            }
            // Tensor functions return tensors
            "matmul" | "transpose" | "reshape" => {
                Ok(MLIRType::AetherTensor {
                    element_type: Box::new(MLIRType::Float { width: 64 }),
                    shape: vec![1], // Simplified
                    device: "cpu".to_string(),
                })
            }
            // String functions return strings
            "concat" | "substr" => Ok(MLIRType::Memref { 
                element_type: Box::new(MLIRType::Integer { width: 8, signed: false }), 
                shape: vec![-1] 
            }),
            // Default to float for unknown functions
            _ => Ok(MLIRType::Float { width: 64 }),
        }
    }

    /// Parse tensor shape from AST node
    fn parse_tensor_shape(&self, shape_node: &ASTNodeRef) -> Result<Vec<i64>, MLIRError> {
        match self.resolve_node_ref(shape_node) {
            Some(ASTNode::List(dims)) => {
                let mut shape = Vec::new();
                for dim_ref in dims {
                    if let Some(ASTNode::Atom(AtomValue::Number(dim))) = self.resolve_node_ref(dim_ref) {
                        shape.push(*dim as i64);
                    } else {
                        return Err(MLIRError::ConversionError("Tensor dimension must be a number".to_string()));
                    }
                }
                Ok(shape)
            }
            Some(ASTNode::Atom(AtomValue::Number(dim))) => {
                // Single dimension
                Ok(vec![*dim as i64])
            }
            _ => Err(MLIRError::ConversionError("Invalid tensor shape specification".to_string())),
        }
    }

    /// Parse element type from AST node
    fn parse_element_type(&self, type_node: &ASTNodeRef) -> Result<MLIRType, MLIRError> {
        match self.resolve_node_ref(type_node) {
            Some(ASTNode::Atom(AtomValue::Symbol(type_name))) => {
                match type_name.as_str() {
                    "f32" => Ok(MLIRType::Float { width: 32 }),
                    "f64" => Ok(MLIRType::Float { width: 64 }),
                    "i8" => Ok(MLIRType::Integer { width: 8, signed: true }),
                    "i16" => Ok(MLIRType::Integer { width: 16, signed: true }),
                    "i32" => Ok(MLIRType::Integer { width: 32, signed: true }),
                    "i64" => Ok(MLIRType::Integer { width: 64, signed: true }),
                    "u8" => Ok(MLIRType::Integer { width: 8, signed: false }),
                    "u16" => Ok(MLIRType::Integer { width: 16, signed: false }),
                    "u32" => Ok(MLIRType::Integer { width: 32, signed: false }),
                    "u64" => Ok(MLIRType::Integer { width: 64, signed: false }),
                    "bool" => Ok(MLIRType::Integer { width: 1, signed: false }),
                    _ => Err(MLIRError::ConversionError(format!("Unknown element type: {}", type_name))),
                }
            }
            _ => Err(MLIRError::ConversionError("Element type must be a symbol".to_string())),
        }
    }

    /// Parse slice specification from AST node
    fn parse_slice_specification(&self, slice_node: &ASTNodeRef) -> Result<Vec<(i64, i64)>, MLIRError> {
        match self.resolve_node_ref(slice_node) {
            Some(ASTNode::List(slices)) => {
                let mut slice_spec = Vec::new();
                for slice_ref in slices {
                    if let Some(ASTNode::List(range)) = self.resolve_node_ref(slice_ref) {
                        if range.len() == 2 {
                            let start = if let Some(ASTNode::Atom(AtomValue::Number(s))) = self.resolve_node_ref(&range[0]) {
                                *s as i64
                            } else {
                                return Err(MLIRError::ConversionError("Slice start must be a number".to_string()));
                            };
                            let end = if let Some(ASTNode::Atom(AtomValue::Number(e))) = self.resolve_node_ref(&range[1]) {
                                *e as i64
                            } else {
                                return Err(MLIRError::ConversionError("Slice end must be a number".to_string()));
                            };
                            slice_spec.push((start, end));
                        } else {
                            return Err(MLIRError::ConversionError("Slice range must have start and end".to_string()));
                        }
                    } else {
                        return Err(MLIRError::ConversionError("Slice specification must be a list of ranges".to_string()));
                    }
                }
                Ok(slice_spec)
            }
            _ => Err(MLIRError::ConversionError("Invalid slice specification".to_string())),
        }
    }

    /// Parse permutation from AST node
    fn parse_permutation(&self, perm_node: &ASTNodeRef) -> Result<Vec<i64>, MLIRError> {
        match self.resolve_node_ref(perm_node) {
            Some(ASTNode::List(dims)) => {
                let mut permutation = Vec::new();
                for dim_ref in dims {
                    if let Some(ASTNode::Atom(AtomValue::Number(dim))) = self.resolve_node_ref(dim_ref) {
                        permutation.push(*dim as i64);
                    } else {
                        return Err(MLIRError::ConversionError("Permutation dimension must be a number".to_string()));
                    }
                }
                Ok(permutation)
            }
            _ => Err(MLIRError::ConversionError("Invalid permutation specification".to_string())),
        }
    }

    /// Parse reduction axes from AST node
    fn parse_reduction_axes(&self, axes_node: &ASTNodeRef) -> Result<Vec<i64>, MLIRError> {
        match self.resolve_node_ref(axes_node) {
            Some(ASTNode::List(axes)) => {
                let mut reduction_axes = Vec::new();
                for axis_ref in axes {
                    if let Some(ASTNode::Atom(AtomValue::Number(axis))) = self.resolve_node_ref(axis_ref) {
                        reduction_axes.push(*axis as i64);
                    } else {
                        return Err(MLIRError::ConversionError("Reduction axis must be a number".to_string()));
                    }
                }
                Ok(reduction_axes)
            }
            Some(ASTNode::Atom(AtomValue::Number(axis))) => {
                // Single axis
                Ok(vec![*axis as i64])
            }
            _ => Err(MLIRError::ConversionError("Invalid reduction axes specification".to_string())),
        }
    }

    /// Parse type specification from AST node
    fn parse_type_specification(&self, type_node: &ASTNodeRef) -> Result<MLIRType, MLIRError> {
        match self.resolve_node_ref(type_node) {
            Some(ASTNode::Atom(AtomValue::Symbol(type_name))) => {
                self.parse_element_type(type_node)
            }
            Some(ASTNode::List(type_spec)) => {
                // Complex type specification like (tensor f64 [2 3])
                if type_spec.is_empty() {
                    return Err(MLIRError::ConversionError("Type specification cannot be empty".to_string()));
                }

                let type_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&type_spec[0]) {
                    name.clone()
                } else {
                    return Err(MLIRError::ConversionError("Type name must be a symbol".to_string()));
                };

                match type_name.as_str() {
                    "tensor" => {
                        let element_type = if type_spec.len() > 1 {
                            self.parse_element_type(&type_spec[1])?
                        } else {
                            MLIRType::Float { width: 64 }
                        };

                        let shape = if type_spec.len() > 2 {
                            self.parse_tensor_shape(&type_spec[2])?
                        } else {
                            vec![-1] // Dynamic shape
                        };

                        Ok(MLIRType::AetherTensor {
                            element_type: Box::new(element_type),
                            shape,
                            device: "cpu".to_string(),
                        })
                    }
                    "function" => {
                        // Function type specification: (function [input-types] [output-types])
                        let inputs = if type_spec.len() > 1 {
                            self.parse_type_list(&type_spec[1])?
                        } else {
                            vec![]
                        };

                        let outputs = if type_spec.len() > 2 {
                            self.parse_type_list(&type_spec[2])?
                        } else {
                            vec![]
                        };

                        Ok(MLIRType::Function { inputs, outputs })
                    }
                    _ => self.parse_element_type(&type_spec[0]),
                }
            }
            _ => Err(MLIRError::ConversionError("Invalid type specification".to_string())),
        }
    }

    /// Parse list of types from AST node
    fn parse_type_list(&self, type_list_node: &ASTNodeRef) -> Result<Vec<MLIRType>, MLIRError> {
        match self.resolve_node_ref(type_list_node) {
            Some(ASTNode::List(types)) => {
                let mut type_list = Vec::new();
                for type_ref in types {
                    let parsed_type = self.parse_type_specification(type_ref)?;
                    type_list.push(parsed_type);
                }
                Ok(type_list)
            }
            _ => Err(MLIRError::ConversionError("Type list must be a list".to_string())),
        }
    }
}

impl<'a> AetherFrontend<'a> {
    /// Create new Aether frontend (legacy)
    pub fn new(context: &'a MLIRContext) -> Self {
        AetherFrontend {
            context,
            symbol_table: HashMap::new(),
            current_function: None,
        }
    }

    /// Convert AST to MLIR module (for backward compatibility)
    pub fn convert_ast(&mut self, ast: &AST) -> Result<MockMLIRModule, MLIRError> {
        let mut module = MockMLIRModule::new();

        // Convert root node
        self.convert_node_to_mock(&ast.root, &mut module)?;

        Ok(module)
    }

    /// Convert AST directly to MLIR module
    pub fn convert_ast_to_module(&mut self, ast: &AST, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Convert root node
        self.convert_node(&ast.root, module)?;
        Ok(())
    }

    /// Convert AST node to MLIR operations
    fn convert_node(&mut self, node: &ASTNode, module: &mut MLIRModule) -> Result<Option<String>, MLIRError> {
        match node {
            ASTNode::Atom(atom) => self.convert_atom(atom),
            ASTNode::List(children) => self.convert_list_to_mlir(children, module),
            ASTNode::Graph { nodes, .. } => self.convert_graph_to_mlir(nodes, module),
        }
    }

    /// Convert AST node to Mock MLIR operations (for backward compatibility)
    fn convert_node_to_mock(&mut self, node: &ASTNode, module: &mut crate::compiler::mlir::test_utils::MockMLIRModule) -> Result<Option<String>, MLIRError> {
        match node {
            ASTNode::Atom(atom) => self.convert_atom(atom),
            ASTNode::List(children) => self.convert_list(children, module),
            ASTNode::Graph { nodes, .. } => self.convert_graph(nodes, module),
        }
    }

    /// Convert atomic value to MLIR
    fn convert_atom(&mut self, atom: &AtomValue) -> Result<Option<String>, MLIRError> {
        match atom {
            AtomValue::Symbol(name) => {
                if let Some(value) = self.symbol_table.get(name) {
                    Ok(Some(value.clone()))
                } else {
                    Ok(Some(name.clone()))
                }
            }
            AtomValue::Number(value) => Ok(Some(format!("const_{}", value))),
            AtomValue::String(value) => Ok(Some(format!("str_{}", value))),
            AtomValue::Boolean(value) => Ok(Some(format!("bool_{}", value))),
            AtomValue::Nil => Ok(Some("nil".to_string())),
            AtomValue::Token(_) => Ok(None),
        }
    }

    /// Convert list to real MLIR operations
    fn convert_list_to_mlir(
        &mut self,
        children: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if children.is_empty() {
            return Ok(None);
        }

        // Get the first element to determine operation type
        let first_child = children.first().unwrap();
        if let Some(first_node) = self.resolve_node_ref(first_child) {
            if let ASTNode::Atom(AtomValue::Symbol(op_name)) = first_node {
                let op_name = op_name.clone();
                return self.convert_operation_to_mlir(&op_name, &children[1..], module);
            }
        }

        // If not a recognized operation, treat as function call
        self.convert_function_call_to_mlir(children, module)
    }

    /// Convert list to MLIR operations
    fn convert_list(
        &mut self,
        children: &[ASTNodeRef],
        module: &mut crate::compiler::mlir::test_utils::MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if children.is_empty() {
            return Ok(None);
        }

        // Get the first element to determine operation type
        let first_child = children.first().unwrap();
        if let Some(first_node) = self.resolve_node_ref(first_child) {
            if let ASTNode::Atom(AtomValue::Symbol(op_name)) = first_node {
                let op_name = op_name.clone(); // Clone to avoid borrowing issues
                return self.convert_operation(&op_name, &children[1..], module);
            }
        }

        // If not a recognized operation, treat as function call
        self.convert_function_call(children, module)
    }

    /// Convert graph structure to MLIR
    fn convert_graph(
        &mut self,
        nodes: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        let mut last_value = None;
        for node_ref in nodes {
            if let Some(node) = self.resolve_node_ref(node_ref) {
                let node = node.clone(); // Clone to avoid borrowing issues
                if let Some(value) = self.convert_node_to_mock(&node, module)? {
                    last_value = Some(value);
                }
            }
        }
        Ok(last_value)
    }

    /// Convert recognized operations to MLIR
    fn convert_operation(
        &mut self,
        op_name: &str,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        match op_name {
            "defun" => self.convert_function_definition(args, module),
            "let" => self.convert_variable_declaration(args, module),
            "tensor" => self.convert_tensor_operation(args, module),
            "autodiff" => self.convert_autodiff_marker(args, module),
            "prob-var" => self.convert_probabilistic_variable(args, module),
            "linear" => self.convert_linear_type(args, module),
            "+" | "-" | "*" | "/" => self.convert_arithmetic_operation(op_name, args, module),
            _ => self.convert_function_call(&[ASTNodeRef::direct(ASTNode::symbol(op_name.to_string()))], module),
        }
    }

    /// Convert function definition
    fn convert_function_definition(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 3 {
            return Err(MLIRError::ConversionError("Function definition requires name, parameters, and body".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let context = &self.context;
        // Create function operation directly since func_def doesn't exist
        let mut func_op = MLIROperation::new("func.func".to_string());
        func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String(name.clone()));
        func_op.add_attribute("function_type".to_string(), MLIRAttribute::String("f64 -> f64".to_string()));
        module.add_operation(format!("aether.func @{}", name));
        
        Ok(Some(name))
    }

    /// Convert variable declaration
    fn convert_variable_declaration(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Variable declaration requires name and value".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        let initial_value = if let Some(value_node) = self.resolve_node_ref(&args[1]) {
            let value_node = value_node.clone(); // Clone to avoid borrowing issues
            self.convert_node_to_mock(&value_node, module)?
        } else {
            None
        };

        let context = &self.context;
        // Create variable operation directly since var_decl doesn't exist
        let mut var_op = MLIROperation::new("aether.var".to_string());
        var_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        var_op.add_attribute("type".to_string(), MLIRAttribute::String("f64".to_string()));
        module.add_operation(format!("aether.var %{} : f64", name));
        
        self.symbol_table.insert(name.clone(), format!("%{}", name));
        Ok(Some(format!("%{}", name)))
    }

    /// Convert tensor operation
    fn convert_tensor_operation(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Tensor operation requires arguments".to_string()));
        }

        let op_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            "generic".to_string()
        };

        let mut operands = Vec::new();
        for arg in &args[1..] {
            if let Some(arg_node) = self.resolve_node_ref(arg) {
                let arg_node = arg_node.clone(); // Clone to avoid borrowing issues
                if let Some(value) = self.convert_node_to_mock(&arg_node, module)? {
                    operands.push(value);
                }
            }
        }
        let operand_refs: Vec<&str> = operands.iter().map(|s| s.as_str()).collect();

        let context = &self.context;
        // Create tensor operation directly with proper types
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 64 }),
            shape: vec![1],
            device: "cpu".to_string(),
        };
        let mut tensor_op = MLIROperation::new("aether.tensor_op".to_string());
        tensor_op.add_attribute("op_name".to_string(), MLIRAttribute::String(op_name.clone()));
        module.add_operation(format!("aether.tensor_op \"{}\"", op_name));
        
        Ok(Some("tensor_result".to_string()))
    }

    /// Convert autodiff marker
    fn convert_autodiff_marker(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Autodiff marker requires function argument".to_string()));
        }

        let function_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let _context = &self.context;
        // Create autodiff operation directly since autodiff_marker doesn't exist
        module.add_operation(format!("aether.autodiff @{}", function_name));
        
        Ok(None)
    }

    /// Convert probabilistic variable
    fn convert_probabilistic_variable(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Probabilistic variable requires name and distribution".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        let distribution = if let Some(ASTNode::Atom(AtomValue::Symbol(dist))) = self.resolve_node_ref(&args[1]) {
            dist.clone()
        } else {
            return Err(MLIRError::ConversionError("Distribution must be a symbol".to_string()));
        };

        let mut parameters = Vec::new();
        for param_ref in &args[2..] {
            if let Some(param_node) = self.resolve_node_ref(param_ref) {
                let param_node = param_node.clone(); // Clone to avoid borrowing issues
                if let Some(value) = self.convert_node_to_mock(&param_node, module)? {
                    parameters.push(value);
                }
            }
        }
        let param_refs: Vec<&str> = parameters.iter().map(|s| s.as_str()).collect();

        let context = &self.context;
        // Create probabilistic variable operation directly with proper types
        let result_type = MLIRType::AetherProbabilistic {
            distribution: format!("{:?}", distribution),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        };
        let mut prob_var_op = MLIROperation::new("aether.prob_var".to_string());
        prob_var_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        prob_var_op.add_attribute("distribution".to_string(), MLIRAttribute::String(format!("{:?}", distribution)));
        module.add_operation(format!("aether.prob_var %{} ~ {}", name, distribution));
        
        Ok(Some(format!("%{}", name)))
    }

    /// Convert linear type annotation
    fn convert_linear_type(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Linear type requires inner type".to_string()));
        }

        let inner_type = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            "f64".to_string()
        };

        let context = &self.context;
        // Create linear type operation directly since linear_type doesn't exist
        let mut linear_op = MLIROperation::new("aether.linear_type".to_string());
        linear_op.add_attribute("inner_type".to_string(), MLIRAttribute::String(inner_type.clone()));
        linear_op.add_attribute("ownership".to_string(), MLIRAttribute::String("owned".to_string()));
        module.add_operation(format!("aether.linear_type linear<{}>", inner_type));
        
        Ok(Some(format!("linear<{}>", inner_type)))
    }

    /// Convert arithmetic operations
    fn convert_arithmetic_operation(
        &mut self,
        op_name: &str,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() != 2 {
            return Err(MLIRError::ConversionError(format!("Arithmetic operation {} requires exactly 2 arguments", op_name)));
        }

        let lhs = if let Some(lhs_node) = self.resolve_node_ref(&args[0]) {
            let lhs_node = lhs_node.clone(); // Clone to avoid borrowing issues
            self.convert_node_to_mock(&lhs_node, module)?
        } else {
            None
        };

        let rhs = if let Some(rhs_node) = self.resolve_node_ref(&args[1]) {
            let rhs_node = rhs_node.clone(); // Clone to avoid borrowing issues
            self.convert_node_to_mock(&rhs_node, module)?
        } else {
            None
        };

        if let (Some(lhs_val), Some(rhs_val)) = (lhs, rhs) {
            let mlir_op = match op_name {
                "+" => "arith.addf",
                "-" => "arith.subf",
                "*" => "arith.mulf",
                "/" => "arith.divf",
                _ => return Err(MLIRError::ConversionError(format!("Unknown arithmetic operation: {}", op_name))),
            };

            module.add_operation(format!("{} {}, {} : f64", mlir_op, lhs_val, rhs_val));
            Ok(Some("arith_result".to_string()))
        } else {
            Err(MLIRError::ConversionError("Invalid operands for arithmetic operation".to_string()))
        }
    }

    /// Convert function call
    fn convert_function_call(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MockMLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Function call requires function name".to_string()));
        }

        let func_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        module.add_operation(format!("func.call @{}", func_name));
        Ok(Some("call_result".to_string()))
    }

    /// Resolve node reference to actual node
    fn resolve_node_ref<'b>(&self, node_ref: &'b ASTNodeRef) -> Option<&'b ASTNode> {
        match node_ref {
            ASTNodeRef::Direct(node) => Some(node.as_ref()),
            ASTNodeRef::Id(_) | ASTNodeRef::Label(_) => {
                // Would need access to AST for full resolution
                None
            }
        }
    }

    // Methods for converting to real MLIR modules

    /// Convert graph structure to real MLIR
    fn convert_graph_to_mlir(
        &mut self,
        nodes: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        let mut last_value = None;
        for node_ref in nodes {
            if let Some(node) = self.resolve_node_ref(node_ref) {
                let node = node.clone();
                if let Some(value) = self.convert_node(&node, module)? {
                    last_value = Some(value);
                }
            }
        }
        Ok(last_value)
    }

    /// Convert recognized operations to real MLIR
    fn convert_operation_to_mlir(
        &mut self,
        op_name: &str,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        match op_name {
            "defun" => self.convert_function_definition_to_mlir(args, module),
            "let" => self.convert_variable_declaration_to_mlir(args, module),
            "tensor" => self.convert_tensor_operation_to_mlir(args, module),
            "autodiff" => self.convert_autodiff_marker_to_mlir(args, module),
            "prob-var" => self.convert_probabilistic_variable_to_mlir(args, module),
            "linear" => self.convert_linear_type_to_mlir(args, module),
            "+" | "-" | "*" | "/" => self.convert_arithmetic_operation_to_mlir(op_name, args, module),
            _ => self.convert_function_call_to_mlir(&[ASTNodeRef::direct(ASTNode::symbol(op_name.to_string()))], module),
        }
    }

    /// Convert function definition to real MLIR
    fn convert_function_definition_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 3 {
            return Err(MLIRError::ConversionError("Function definition requires name, parameters, and body".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let mut func_op = MLIROperation::new(format!("aether.func"));
        func_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        func_op.add_attribute("type".to_string(), MLIRAttribute::String("f64 -> f64".to_string()));
        module.add_operation(func_op)?;
        
        Ok(Some(name))
    }

    /// Convert variable declaration to real MLIR
    fn convert_variable_declaration_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Variable declaration requires name and value".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        let mut var_op = MLIROperation::new("aether.var".to_string());
        var_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        var_op.add_attribute("type".to_string(), MLIRAttribute::String("f64".to_string()));
        module.add_operation(var_op)?;
        
        self.symbol_table.insert(name.clone(), format!("%{}", name));
        Ok(Some(format!("%{}", name)))
    }

    /// Convert tensor operation to real MLIR
    fn convert_tensor_operation_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Tensor operation requires arguments".to_string()));
        }

        let op_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            "generic".to_string()
        };

        let mut tensor_op = MLIROperation::new("aether.tensor_op".to_string());
        tensor_op.add_attribute("operation".to_string(), MLIRAttribute::String(op_name));
        tensor_op.add_attribute("type".to_string(), MLIRAttribute::String("tensor<f64>".to_string()));
        module.add_operation(tensor_op)?;
        
        Ok(Some("tensor_result".to_string()))
    }

    /// Convert autodiff marker to real MLIR
    fn convert_autodiff_marker_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Autodiff marker requires function argument".to_string()));
        }

        let function_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let mut autodiff_op = MLIROperation::new("aether.autodiff".to_string());
        autodiff_op.add_attribute("function".to_string(), MLIRAttribute::String(function_name));
        autodiff_op.add_attribute("mode".to_string(), MLIRAttribute::String("reverse".to_string()));
        module.add_operation(autodiff_op)?;
        
        Ok(None)
    }

    /// Convert probabilistic variable to real MLIR
    fn convert_probabilistic_variable_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() < 2 {
            return Err(MLIRError::ConversionError("Probabilistic variable requires name and distribution".to_string()));
        }

        let name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Variable name must be a symbol".to_string()));
        };

        let distribution = if let Some(ASTNode::Atom(AtomValue::Symbol(dist))) = self.resolve_node_ref(&args[1]) {
            dist.clone()
        } else {
            return Err(MLIRError::ConversionError("Distribution must be a symbol".to_string()));
        };

        let mut prob_var_op = MLIROperation::new("aether.prob_var".to_string());
        prob_var_op.add_attribute("name".to_string(), MLIRAttribute::String(name.clone()));
        prob_var_op.add_attribute("distribution".to_string(), MLIRAttribute::String(distribution));
        module.add_operation(prob_var_op)?;
        
        Ok(Some(format!("%{}", name)))
    }

    /// Convert linear type annotation to real MLIR
    fn convert_linear_type_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Linear type requires inner type".to_string()));
        }

        let inner_type = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            "f64".to_string()
        };

        let mut linear_op = MLIROperation::new("aether.linear_type".to_string());
        linear_op.add_attribute("inner_type".to_string(), MLIRAttribute::String(inner_type.clone()));
        linear_op.add_attribute("ownership".to_string(), MLIRAttribute::String("owned".to_string()));
        module.add_operation(linear_op)?;
        
        Ok(Some(format!("linear<{}>", inner_type)))
    }

    /// Convert arithmetic operations to real MLIR
    fn convert_arithmetic_operation_to_mlir(
        &mut self,
        op_name: &str,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.len() != 2 {
            return Err(MLIRError::ConversionError(format!("Arithmetic operation {} requires exactly 2 arguments", op_name)));
        }

        let mlir_op_name = match op_name {
            "+" => "arith.addf",
            "-" => "arith.subf",
            "*" => "arith.mulf",
            "/" => "arith.divf",
            _ => return Err(MLIRError::ConversionError(format!("Unknown arithmetic operation: {}", op_name))),
        };

        let mut arith_op = MLIROperation::new(mlir_op_name.to_string());
        arith_op.add_attribute("type".to_string(), MLIRAttribute::String("f64".to_string()));
        module.add_operation(arith_op)?;
        
        Ok(Some("arith_result".to_string()))
    }

    /// Convert function call to real MLIR
    fn convert_function_call_to_mlir(
        &mut self,
        args: &[ASTNodeRef],
        module: &mut MLIRModule,
    ) -> Result<Option<String>, MLIRError> {
        if args.is_empty() {
            return Err(MLIRError::ConversionError("Function call requires function name".to_string()));
        }

        let func_name = if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = self.resolve_node_ref(&args[0]) {
            name.clone()
        } else {
            return Err(MLIRError::ConversionError("Function name must be a symbol".to_string()));
        };

        let mut call_op = MLIROperation::new("func.call".to_string());
        call_op.add_attribute("callee".to_string(), MLIRAttribute::String(func_name));
        module.add_operation(call_op)?;
        
        Ok(Some("call_result".to_string()))
    }
}