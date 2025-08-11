// Aether Automatic Differentiation Engine
// Dual-mode AD supporting both dynamic tape and static graph compilation

use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::compiler::ast::{ASTNode, ASTNodeRef, AtomValue};
use crate::compiler::types::{Type, PrimitiveType};

/// Differentiation mode for AD engine
#[derive(Debug, Clone, PartialEq)]
pub enum DiffMode {
    /// Dynamic tape-based differentiation for development
    Dynamic,
    /// Static graph compilation for production
    Static,
}

/// Differentiation direction
#[derive(Debug, Clone, PartialEq)]
pub enum DiffDirection {
    /// Forward mode differentiation
    Forward,
    /// Reverse mode differentiation (backpropagation)
    Reverse,
    /// Mixed mode (forward for some variables, reverse for others)
    Mixed(Vec<String>), // Variables to use forward mode for
}

/// Differentiation marker for functions and expressions
#[derive(Debug, Clone, PartialEq)]
pub struct DiffMarker {
    /// Variables to differentiate with respect to
    pub wrt_vars: Vec<String>,
    /// Differentiation direction
    pub direction: DiffDirection,
    /// Order of differentiation (1 for first derivative, 2 for second, etc.)
    pub order: u32,
    /// Whether to use dynamic or static mode
    pub mode: DiffMode,
}

/// Gradient computation function metadata
#[derive(Debug, Clone)]
pub struct GradientFunction {
    /// Original function name
    pub original_name: String,
    /// Generated gradient function name
    pub gradient_name: String,
    /// Variables being differentiated
    pub wrt_vars: Vec<String>,
    /// Input parameter types
    pub input_types: Vec<Type>,
    /// Output gradient types
    pub output_types: Vec<Type>,
    /// Generated AST for gradient computation
    pub gradient_ast: ASTNode,
    /// Differentiation metadata
    pub diff_marker: DiffMarker,
}

/// Automatic differentiation engine
#[derive(Debug)]
pub struct AutoDiffEngine {
    /// Registered differentiable functions
    pub functions: HashMap<String, DifferentiableFunction>,
    /// Generated gradient functions
    pub gradient_functions: HashMap<String, GradientFunction>,
    /// Primitive operation derivatives
    pub primitive_derivatives: HashMap<String, PrimitiveDerivative>,
    /// Tape for dynamic mode differentiation
    pub tape: Option<DynamicTape>,
    /// Static computation graph
    pub static_graph: Option<StaticGraph>,
}

/// Differentiable function definition
#[derive(Debug, Clone)]
pub struct DifferentiableFunction {
    /// Function name
    pub name: String,
    /// Function parameters
    pub params: Vec<String>,
    /// Parameter types
    pub param_types: Vec<Type>,
    /// Return type
    pub return_type: Type,
    /// Function body AST
    pub body: ASTNode,
    /// Differentiation markers
    pub diff_markers: Vec<DiffMarker>,
}

/// Primitive operation derivative definition
#[derive(Debug, Clone)]
pub struct PrimitiveDerivative {
    /// Operation name (e.g., "add", "mul", "sin")
    pub op_name: String,
    /// Forward mode derivative function
    pub forward_fn: fn(&[f64], &[f64]) -> Vec<f64>,
    /// Reverse mode derivative function
    pub reverse_fn: fn(&[f64], f64) -> Vec<f64>,
    /// Symbolic derivative expression
    pub symbolic_expr: Option<ASTNode>,
}

/// Dynamic tape for tape-based differentiation
#[derive(Debug, Clone)]
pub struct DynamicTape {
    /// Tape operations
    pub operations: Vec<TapeOperation>,
    /// Variable values
    pub variables: HashMap<String, f64>,
    /// Variable gradients
    pub gradients: HashMap<String, f64>,
    /// Next operation ID
    next_op_id: u32,
}

/// Operation recorded on the dynamic tape
#[derive(Debug, Clone)]
pub struct TapeOperation {
    /// Operation ID
    pub id: u32,
    /// Operation type
    pub op_type: String,
    /// Input variable names
    pub inputs: Vec<String>,
    /// Output variable name
    pub output: String,
    /// Operation parameters
    pub params: Vec<f64>,
    /// Forward derivative function
    pub forward_fn: Option<fn(&[f64], &[f64]) -> Vec<f64>>,
    /// Reverse derivative function
    pub reverse_fn: Option<fn(&[f64], f64) -> Vec<f64>>,
}

/// Static computation graph for compile-time optimization
#[derive(Debug, Clone)]
pub struct StaticGraph {
    /// Graph nodes
    pub nodes: HashMap<u32, GraphNode>,
    /// Graph edges
    pub edges: Vec<GraphEdge>,
    /// Input nodes
    pub inputs: Vec<u32>,
    /// Output nodes
    pub outputs: Vec<u32>,
    /// Next node ID
    next_node_id: u32,
}

/// Node in the static computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node ID
    pub id: u32,
    /// Operation type
    pub op_type: String,
    /// Node value type
    pub value_type: Type,
    /// Operation parameters
    pub params: Vec<f64>,
    /// Whether node requires gradient
    pub requires_grad: bool,
}

/// Edge in the static computation graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID
    pub from: u32,
    /// Target node ID
    pub to: u32,
    /// Edge weight (for weighted operations)
    pub weight: Option<f64>,
}

impl AutoDiffEngine {
    /// Create new automatic differentiation engine
    pub fn new() -> Self {
        let mut engine = AutoDiffEngine {
            functions: HashMap::new(),
            gradient_functions: HashMap::new(),
            primitive_derivatives: HashMap::new(),
            tape: None,
            static_graph: None,
        };
        
        // Initialize primitive derivatives
        engine.init_primitive_derivatives();
        engine
    }

    /// Initialize primitive operation derivatives
    fn init_primitive_derivatives(&mut self) {
        // Addition: d/dx(a + b) = da/dx + db/dx
        self.primitive_derivatives.insert("add".to_string(), PrimitiveDerivative {
            op_name: "add".to_string(),
            forward_fn: |_inputs, derivs| vec![derivs[0] + derivs[1]],
            reverse_fn: |_inputs, grad| vec![grad, grad],
            symbolic_expr: Some(ASTNode::symbol("add_derivative".to_string())),
        });

        // Subtraction: d/dx(a - b) = da/dx - db/dx
        self.primitive_derivatives.insert("sub".to_string(), PrimitiveDerivative {
            op_name: "sub".to_string(),
            forward_fn: |_inputs, derivs| vec![derivs[0] - derivs[1]],
            reverse_fn: |_inputs, grad| vec![grad, -grad],
            symbolic_expr: Some(ASTNode::symbol("sub_derivative".to_string())),
        });

        // Multiplication: d/dx(a * b) = a * db/dx + b * da/dx
        self.primitive_derivatives.insert("mul".to_string(), PrimitiveDerivative {
            op_name: "mul".to_string(),
            forward_fn: |inputs, derivs| vec![inputs[0] * derivs[1] + inputs[1] * derivs[0]],
            reverse_fn: |inputs, grad| vec![grad * inputs[1], grad * inputs[0]],
            symbolic_expr: Some(ASTNode::symbol("mul_derivative".to_string())),
        });

        // Division: d/dx(a / b) = (b * da/dx - a * db/dx) / b^2
        self.primitive_derivatives.insert("div".to_string(), PrimitiveDerivative {
            op_name: "div".to_string(),
            forward_fn: |inputs, derivs| {
                let b_squared = inputs[1] * inputs[1];
                vec![(inputs[1] * derivs[0] - inputs[0] * derivs[1]) / b_squared]
            },
            reverse_fn: |inputs, grad| {
                let b_squared = inputs[1] * inputs[1];
                vec![grad / inputs[1], -grad * inputs[0] / b_squared]
            },
            symbolic_expr: Some(ASTNode::symbol("div_derivative".to_string())),
        });

        // Sine: d/dx(sin(x)) = cos(x) * dx/dx
        self.primitive_derivatives.insert("sin".to_string(), PrimitiveDerivative {
            op_name: "sin".to_string(),
            forward_fn: |inputs, derivs| vec![inputs[0].cos() * derivs[0]],
            reverse_fn: |inputs, grad| vec![grad * inputs[0].cos()],
            symbolic_expr: Some(ASTNode::symbol("sin_derivative".to_string())),
        });

        // Cosine: d/dx(cos(x)) = -sin(x) * dx/dx
        self.primitive_derivatives.insert("cos".to_string(), PrimitiveDerivative {
            op_name: "cos".to_string(),
            forward_fn: |inputs, derivs| vec![-inputs[0].sin() * derivs[0]],
            reverse_fn: |inputs, grad| vec![-grad * inputs[0].sin()],
            symbolic_expr: Some(ASTNode::symbol("cos_derivative".to_string())),
        });

        // Exponential: d/dx(exp(x)) = exp(x) * dx/dx
        self.primitive_derivatives.insert("exp".to_string(), PrimitiveDerivative {
            op_name: "exp".to_string(),
            forward_fn: |inputs, derivs| vec![inputs[0].exp() * derivs[0]],
            reverse_fn: |inputs, grad| vec![grad * inputs[0].exp()],
            symbolic_expr: Some(ASTNode::symbol("exp_derivative".to_string())),
        });

        // Natural logarithm: d/dx(ln(x)) = 1/x * dx/dx
        self.primitive_derivatives.insert("ln".to_string(), PrimitiveDerivative {
            op_name: "ln".to_string(),
            forward_fn: |inputs, derivs| vec![derivs[0] / inputs[0]],
            reverse_fn: |inputs, grad| vec![grad / inputs[0]],
            symbolic_expr: Some(ASTNode::symbol("ln_derivative".to_string())),
        });

        // Power: d/dx(x^n) = n * x^(n-1) * dx/dx
        self.primitive_derivatives.insert("pow".to_string(), PrimitiveDerivative {
            op_name: "pow".to_string(),
            forward_fn: |inputs, derivs| {
                let base = inputs[0];
                let exp = inputs[1];
                vec![exp * base.powf(exp - 1.0) * derivs[0] + base.powf(exp) * base.ln() * derivs[1]]
            },
            reverse_fn: |inputs, grad| {
                let base = inputs[0];
                let exp = inputs[1];
                vec![
                    grad * exp * base.powf(exp - 1.0),
                    grad * base.powf(exp) * base.ln()
                ]
            },
            symbolic_expr: Some(ASTNode::symbol("pow_derivative".to_string())),
        });

        // Matrix multiplication derivatives (simplified for scalars here)
        self.primitive_derivatives.insert("matmul".to_string(), PrimitiveDerivative {
            op_name: "matmul".to_string(),
            forward_fn: |inputs, derivs| {
                // Simplified: for matrices A and B, d/dx(A*B) involves tensor operations
                // This is a placeholder for the full tensor implementation
                vec![derivs[0] * inputs[1] + inputs[0] * derivs[1]]
            },
            reverse_fn: |inputs, grad| {
                // Simplified reverse mode for matrix multiplication
                vec![grad * inputs[1], grad * inputs[0]]
            },
            symbolic_expr: Some(ASTNode::symbol("matmul_derivative".to_string())),
        });
    }

    /// Register a differentiable function
    pub fn register_function(&mut self, func: DifferentiableFunction) -> Result<(), String> {
        // Validate function definition
        if func.params.len() != func.param_types.len() {
            return Err("Parameter count mismatch".to_string());
        }

        // Check if function is already registered
        if self.functions.contains_key(&func.name) {
            return Err(format!("Function '{}' already registered", func.name));
        }

        self.functions.insert(func.name.clone(), func);
        Ok(())
    }

    /// Generate gradient function for a differentiable function
    pub fn generate_gradient_function(
        &mut self,
        func_name: &str,
        diff_marker: DiffMarker,
    ) -> Result<String, String> {
        let func = self.functions.get(func_name)
            .ok_or_else(|| format!("Function '{}' not found", func_name))?
            .clone();

        let gradient_name = format!("{}_grad", func_name);

        // Generate gradient computation AST based on differentiation mode
        let gradient_ast = match diff_marker.mode {
            DiffMode::Dynamic => self.generate_dynamic_gradient(&func, &diff_marker)?,
            DiffMode::Static => self.generate_static_gradient(&func, &diff_marker)?,
        };

        // Determine output types (gradients have same shape as inputs)
        let mut output_types = Vec::new();
        for var in &diff_marker.wrt_vars {
            if let Some(param_idx) = func.params.iter().position(|p| p == var) {
                output_types.push(func.param_types[param_idx].clone());
            }
        }

        let gradient_func = GradientFunction {
            original_name: func_name.to_string(),
            gradient_name: gradient_name.clone(),
            wrt_vars: diff_marker.wrt_vars.clone(),
            input_types: func.param_types.clone(),
            output_types,
            gradient_ast,
            diff_marker,
        };

        self.gradient_functions.insert(gradient_name.clone(), gradient_func);
        Ok(gradient_name)
    }

    /// Generate dynamic gradient computation
    fn generate_dynamic_gradient(
        &mut self,
        func: &DifferentiableFunction,
        diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Initialize dynamic tape if not present
        if self.tape.is_none() {
            self.tape = Some(DynamicTape::new());
        }

        match diff_marker.direction {
            DiffDirection::Forward => self.generate_forward_mode_gradient(func, diff_marker),
            DiffDirection::Reverse => self.generate_reverse_mode_gradient(func, diff_marker),
            DiffDirection::Mixed(_) => self.generate_mixed_mode_gradient(func, diff_marker),
        }
    }

    /// Generate static gradient computation
    fn generate_static_gradient(
        &mut self,
        func: &DifferentiableFunction,
        diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Initialize static graph if not present
        if self.static_graph.is_none() {
            self.static_graph = Some(StaticGraph::new());
        }

        // Build computation graph from function body
        self.build_computation_graph(&func.body)?;

        // Generate optimized gradient computation
        match diff_marker.direction {
            DiffDirection::Forward => self.generate_static_forward_gradient(func, diff_marker),
            DiffDirection::Reverse => self.generate_static_reverse_gradient(func, diff_marker),
            DiffDirection::Mixed(_) => self.generate_static_mixed_gradient(func, diff_marker),
        }
    }

    /// Generate forward mode gradient
    fn generate_forward_mode_gradient(
        &self,
        func: &DifferentiableFunction,
        diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Forward mode: compute derivatives alongside values
        let mut gradient_exprs = Vec::new();

        // Create dual numbers for each variable we're differentiating with respect to
        for wrt_var in &diff_marker.wrt_vars {
            let dual_computation = self.generate_dual_number_computation(&func.body, wrt_var)?;
            gradient_exprs.push(ASTNodeRef::direct(dual_computation));
        }

        Ok(ASTNode::list(gradient_exprs))
    }

    /// Generate reverse mode gradient (backpropagation)
    fn generate_reverse_mode_gradient(
        &self,
        func: &DifferentiableFunction,
        diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Reverse mode: build computation graph and backpropagate
        let forward_pass = self.generate_forward_pass(&func.body)?;
        let backward_pass = self.generate_backward_pass(&func.body, &diff_marker.wrt_vars)?;

        // Combine forward and backward passes
        let gradient_computation = ASTNode::list(vec![
            ASTNodeRef::direct(forward_pass),
            ASTNodeRef::direct(backward_pass),
        ]);

        Ok(gradient_computation)
    }

    /// Generate mixed mode gradient
    fn generate_mixed_mode_gradient(
        &self,
        func: &DifferentiableFunction,
        diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        if let DiffDirection::Mixed(forward_vars) = &diff_marker.direction {
            let mut gradient_exprs = Vec::new();

            // Use forward mode for specified variables
            for var in forward_vars {
                if diff_marker.wrt_vars.contains(var) {
                    let dual_computation = self.generate_dual_number_computation(&func.body, var)?;
                    gradient_exprs.push(ASTNodeRef::direct(dual_computation));
                }
            }

            // Use reverse mode for remaining variables
            let reverse_vars: Vec<String> = diff_marker.wrt_vars.iter()
                .filter(|var| !forward_vars.contains(var))
                .cloned()
                .collect();

            if !reverse_vars.is_empty() {
                let reverse_computation = self.generate_backward_pass(&func.body, &reverse_vars)?;
                gradient_exprs.push(ASTNodeRef::direct(reverse_computation));
            }

            Ok(ASTNode::list(gradient_exprs))
        } else {
            Err("Invalid mixed mode configuration".to_string())
        }
    }

    /// Generate dual number computation for forward mode
    pub fn generate_dual_number_computation(&self, expr: &ASTNode, wrt_var: &str) -> Result<ASTNode, String> {
        match expr {
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                if name == wrt_var {
                    // Variable we're differentiating with respect to: dual(value, 1.0)
                    Ok(ASTNode::list(vec![
                        ASTNodeRef::direct(ASTNode::symbol("dual".to_string())),
                        ASTNodeRef::direct(ASTNode::symbol(name.clone())),
                        ASTNodeRef::direct(ASTNode::number(1.0)),
                    ]))
                } else {
                    // Other variable: dual(value, 0.0)
                    Ok(ASTNode::list(vec![
                        ASTNodeRef::direct(ASTNode::symbol("dual".to_string())),
                        ASTNodeRef::direct(ASTNode::symbol(name.clone())),
                        ASTNodeRef::direct(ASTNode::number(0.0)),
                    ]))
                }
            }
            ASTNode::Atom(AtomValue::Number(value)) => {
                // Constant: dual(value, 0.0)
                Ok(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("dual".to_string())),
                    ASTNodeRef::direct(ASTNode::number(*value)),
                    ASTNodeRef::direct(ASTNode::number(0.0)),
                ]))
            }
            ASTNode::List(children) => {
                if let Some(ASTNodeRef::Direct(op_node)) = children.first() {
                    if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                        // Apply dual number arithmetic rules
                        self.generate_dual_operation(op_name, &children[1..], wrt_var)
                    } else {
                        Err("Invalid operation in expression".to_string())
                    }
                } else {
                    Err("Empty expression list".to_string())
                }
            }
            _ => Err("Unsupported expression type for differentiation".to_string()),
        }
    }

    /// Generate dual number operation
    fn generate_dual_operation(&self, op_name: &str, args: &[ASTNodeRef], wrt_var: &str) -> Result<ASTNode, String> {
        let dual_args: Result<Vec<ASTNode>, String> = args.iter()
            .map(|arg_ref| {
                if let ASTNodeRef::Direct(arg_node) = arg_ref {
                    self.generate_dual_number_computation(arg_node, wrt_var)
                } else {
                    Err("Unsupported argument reference type".to_string())
                }
            })
            .collect();

        let dual_args = dual_args?;
        let dual_op_name = format!("dual_{}", op_name);

        let mut result_args = vec![ASTNodeRef::direct(ASTNode::symbol(dual_op_name))];
        for dual_arg in dual_args {
            result_args.push(ASTNodeRef::direct(dual_arg));
        }

        Ok(ASTNode::list(result_args))
    }

    /// Generate forward pass for reverse mode
    fn generate_forward_pass(&self, expr: &ASTNode) -> Result<ASTNode, String> {
        // Build computation graph and record operations
        match expr {
            ASTNode::List(children) => {
                if let Some(ASTNodeRef::Direct(op_node)) = children.first() {
                    if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                        let forward_args: Result<Vec<ASTNode>, String> = children[1..].iter()
                            .map(|arg_ref| {
                                if let ASTNodeRef::Direct(arg_node) = arg_ref {
                                    self.generate_forward_pass(arg_node)
                                } else {
                                    Ok(ASTNode::symbol("arg_ref".to_string()))
                                }
                            })
                            .collect();

                        let forward_args = forward_args?;
                        let mut result_args = vec![ASTNodeRef::direct(ASTNode::symbol(format!("forward_{}", op_name)))];
                        for arg in forward_args {
                            result_args.push(ASTNodeRef::direct(arg));
                        }

                        Ok(ASTNode::list(result_args))
                    } else {
                        Err("Invalid operation in forward pass".to_string())
                    }
                } else {
                    Err("Empty expression in forward pass".to_string())
                }
            }
            _ => Ok(expr.clone()),
        }
    }

    /// Generate backward pass for reverse mode
    fn generate_backward_pass(&self, expr: &ASTNode, wrt_vars: &[String]) -> Result<ASTNode, String> {
        // Generate backpropagation code
        let mut backward_exprs = Vec::new();

        // Initialize gradient of output to 1.0
        backward_exprs.push(ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("set_grad".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("output".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
        ])));

        // Generate backward operations
        let backward_ops = self.generate_backward_operations(expr, wrt_vars)?;
        for op in backward_ops {
            backward_exprs.push(ASTNodeRef::direct(op));
        }

        // Collect gradients for requested variables
        for var in wrt_vars {
            backward_exprs.push(ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("get_grad".to_string())),
                ASTNodeRef::direct(ASTNode::symbol(var.clone())),
            ])));
        }

        Ok(ASTNode::list(backward_exprs))
    }

    /// Generate backward operations for reverse mode
    fn generate_backward_operations(&self, expr: &ASTNode, wrt_vars: &[String]) -> Result<Vec<ASTNode>, String> {
        let mut operations = Vec::new();

        match expr {
            ASTNode::List(children) => {
                if let Some(ASTNodeRef::Direct(op_node)) = children.first() {
                    if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                        // Generate backward operation for this node
                        if let Some(_primitive_deriv) = self.primitive_derivatives.get(op_name) {
                            let backward_op = ASTNode::list(vec![
                                ASTNodeRef::direct(ASTNode::symbol(format!("backward_{}", op_name))),
                                ASTNodeRef::direct(ASTNode::symbol("grad_output".to_string())),
                            ]);
                            operations.push(backward_op);
                        }

                        // Recursively generate backward operations for children
                        for child_ref in &children[1..] {
                            if let ASTNodeRef::Direct(child_node) = child_ref {
                                let child_ops = self.generate_backward_operations(child_node, wrt_vars)?;
                                operations.extend(child_ops);
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(operations)
    }

    /// Build computation graph for static mode
    pub fn build_computation_graph(&mut self, expr: &ASTNode) -> Result<u32, String> {
        match expr {
            ASTNode::Atom(AtomValue::Symbol(_name)) => {
                // Variable node
                let graph = self.static_graph.as_mut()
                    .ok_or("Static graph not initialized")?;
                let node_id = graph.add_node(GraphNode {
                    id: graph.next_node_id,
                    op_type: "variable".to_string(),
                    value_type: Type::primitive(PrimitiveType::Float64), // Default type
                    params: Vec::new(),
                    requires_grad: true,
                });
                Ok(node_id)
            }
            ASTNode::Atom(AtomValue::Number(value)) => {
                // Constant node
                let graph = self.static_graph.as_mut()
                    .ok_or("Static graph not initialized")?;
                let node_id = graph.add_node(GraphNode {
                    id: graph.next_node_id,
                    op_type: "constant".to_string(),
                    value_type: Type::primitive(PrimitiveType::Float64),
                    params: vec![*value],
                    requires_grad: false,
                });
                Ok(node_id)
            }
            ASTNode::List(children) => {
                if let Some(ASTNodeRef::Direct(op_node)) = children.first() {
                    if let Some(AtomValue::Symbol(op_name)) = op_node.as_atom() {
                        // Operation node
                        let mut input_nodes = Vec::new();
                        for child_ref in &children[1..] {
                            if let ASTNodeRef::Direct(child_node) = child_ref {
                                let child_id = self.build_computation_graph(child_node)?;
                                input_nodes.push(child_id);
                            }
                        }

                        let graph = self.static_graph.as_mut()
                            .ok_or("Static graph not initialized")?;
                        let node_id = graph.add_node(GraphNode {
                            id: graph.next_node_id,
                            op_type: op_name.clone(),
                            value_type: Type::primitive(PrimitiveType::Float64),
                            params: Vec::new(),
                            requires_grad: true,
                        });

                        // Add edges from inputs to this node
                        for input_id in input_nodes {
                            graph.add_edge(GraphEdge {
                                from: input_id,
                                to: node_id,
                                weight: None,
                            });
                        }

                        Ok(node_id)
                    } else {
                        Err("Invalid operation node".to_string())
                    }
                } else {
                    Err("Empty operation list".to_string())
                }
            }
            _ => Err("Unsupported expression type in computation graph".to_string()),
        }
    }

    /// Generate static forward gradient
    fn generate_static_forward_gradient(
        &self,
        func: &DifferentiableFunction,
        _diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Use static graph to generate optimized forward mode code
        Ok(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("static_forward_gradient".to_string())),
            ASTNodeRef::direct(ASTNode::symbol(func.name.clone())),
        ]))
    }

    /// Generate static reverse gradient
    fn generate_static_reverse_gradient(
        &self,
        func: &DifferentiableFunction,
        _diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Use static graph to generate optimized reverse mode code
        Ok(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("static_reverse_gradient".to_string())),
            ASTNodeRef::direct(ASTNode::symbol(func.name.clone())),
        ]))
    }

    /// Generate static mixed gradient
    fn generate_static_mixed_gradient(
        &self,
        func: &DifferentiableFunction,
        _diff_marker: &DiffMarker,
    ) -> Result<ASTNode, String> {
        // Use static graph to generate optimized mixed mode code
        Ok(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("static_mixed_gradient".to_string())),
            ASTNodeRef::direct(ASTNode::symbol(func.name.clone())),
        ]))
    }

    /// Differentiate expression with respect to variable
    pub fn differentiate(
        &mut self,
        expr: &ASTNode,
        wrt_var: &str,
        mode: DiffMode,
        direction: DiffDirection,
    ) -> Result<ASTNode, String> {
        let diff_marker = DiffMarker {
            wrt_vars: vec![wrt_var.to_string()],
            direction,
            order: 1,
            mode: mode.clone(),
        };

        match mode {
            DiffMode::Dynamic => {
                match diff_marker.direction {
                    DiffDirection::Forward => self.generate_dual_number_computation(expr, wrt_var),
                    DiffDirection::Reverse => {
                        let forward = self.generate_forward_pass(expr)?;
                        let backward = self.generate_backward_pass(expr, &[wrt_var.to_string()])?;
                        Ok(ASTNode::list(vec![
                            ASTNodeRef::direct(forward),
                            ASTNodeRef::direct(backward),
                        ]))
                    }
                    DiffDirection::Mixed(_) => {
                        Err("Mixed mode not supported for single variable differentiation".to_string())
                    }
                }
            }
            DiffMode::Static => {
                // Initialize static graph if not present
                if self.static_graph.is_none() {
                    self.static_graph = Some(StaticGraph::new());
                }
                
                // Build computation graph and generate optimized code
                self.build_computation_graph(expr)?;
                Ok(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol("static_differentiate".to_string())),
                    ASTNodeRef::direct(expr.clone()),
                    ASTNodeRef::direct(ASTNode::symbol(wrt_var.to_string())),
                ]))
            }
        }
    }

    /// Compute higher-order derivatives
    pub fn higher_order_derivative(
        &mut self,
        expr: &ASTNode,
        wrt_var: &str,
        order: u32,
        mode: DiffMode,
    ) -> Result<ASTNode, String> {
        if order == 0 {
            return Ok(expr.clone());
        }

        let mut current_expr = expr.clone();
        for _i in 0..order {
            current_expr = self.differentiate(
                &current_expr,
                wrt_var,
                mode.clone(),
                DiffDirection::Forward, // Use forward mode for higher-order derivatives
            )?;
        }

        Ok(current_expr)
    }

    /// Integrate AD with tensor operations
    pub fn tensor_gradient(
        &mut self,
        tensor_expr: &ASTNode,
        wrt_tensor: &str,
        tensor_type: &Type,
    ) -> Result<ASTNode, String> {
        // Validate tensor type
        if !tensor_type.is_tensor() {
            return Err("Expected tensor type for tensor gradient".to_string());
        }

        // Generate tensor-aware gradient computation
        let tensor_grad = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("tensor_gradient".to_string())),
            ASTNodeRef::direct(tensor_expr.clone()),
            ASTNodeRef::direct(ASTNode::symbol(wrt_tensor.to_string())),
            ASTNodeRef::direct(ASTNode::symbol("tensor_type".to_string())),
        ]);

        Ok(tensor_grad)
    }

    /// Get registered function
    pub fn get_function(&self, name: &str) -> Option<&DifferentiableFunction> {
        self.functions.get(name)
    }

    /// Get generated gradient function
    pub fn get_gradient_function(&self, name: &str) -> Option<&GradientFunction> {
        self.gradient_functions.get(name)
    }

    /// Check if function is differentiable
    pub fn is_differentiable(&self, func_name: &str) -> bool {
        self.functions.contains_key(func_name)
    }
}

impl DynamicTape {
    /// Create new dynamic tape
    pub fn new() -> Self {
        DynamicTape {
            operations: Vec::new(),
            variables: HashMap::new(),
            gradients: HashMap::new(),
            next_op_id: 0,
        }
    }

    /// Record operation on tape
    pub fn record_operation(&mut self, op: TapeOperation) {
        self.operations.push(op);
        self.next_op_id += 1;
    }

    /// Set variable value
    pub fn set_variable(&mut self, name: String, value: f64) {
        self.variables.insert(name, value);
    }

    /// Get variable value
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }

    /// Set gradient value
    pub fn set_gradient(&mut self, name: String, grad: f64) {
        self.gradients.insert(name, grad);
    }

    /// Get gradient value
    pub fn get_gradient(&self, name: &str) -> Option<f64> {
        self.gradients.get(name).copied()
    }

    /// Clear tape
    pub fn clear(&mut self) {
        self.operations.clear();
        self.variables.clear();
        self.gradients.clear();
        self.next_op_id = 0;
    }

    /// Execute forward pass
    pub fn forward_pass(&mut self) -> Result<(), String> {
        for _op in &self.operations {
            // Execute forward operation
            // This would involve actual computation based on op_type
        }
        Ok(())
    }

    /// Execute backward pass
    pub fn backward_pass(&mut self) -> Result<(), String> {
        // Clone operations to avoid borrowing issues
        let operations = self.operations.clone();
        
        // Reverse iterate through operations
        for op in operations.iter().rev() {
            // Execute backward operation using reverse_fn
            if let Some(reverse_fn) = op.reverse_fn {
                let inputs: Vec<f64> = op.inputs.iter()
                    .filter_map(|name| self.get_variable(name))
                    .collect();
                
                let output_grad = self.get_gradient(&op.output).unwrap_or(0.0);
                let input_grads = reverse_fn(&inputs, output_grad);
                
                // Accumulate gradients
                for (input_name, grad) in op.inputs.iter().zip(input_grads.iter()) {
                    let current_grad = self.get_gradient(input_name).unwrap_or(0.0);
                    self.set_gradient(input_name.clone(), current_grad + grad);
                }
            }
        }
        Ok(())
    }
}

impl StaticGraph {
    /// Create new static computation graph
    pub fn new() -> Self {
        StaticGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_node_id: 0,
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, node: GraphNode) -> u32 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        let mut node = node;
        node.id = id;
        self.nodes.insert(id, node);
        id
    }

    /// Add edge to graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Get node by ID
    pub fn get_node(&self, id: u32) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Topological sort of nodes
    pub fn topological_sort(&self) -> Result<Vec<u32>, String> {
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut result = Vec::new();

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id) {
                self.topological_sort_util(*node_id, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn topological_sort_util(
        &self,
        node_id: u32,
        visited: &mut HashSet<u32>,
        temp_visited: &mut HashSet<u32>,
        result: &mut Vec<u32>,
    ) -> Result<(), String> {
        if temp_visited.contains(&node_id) {
            return Err("Cycle detected in computation graph".to_string());
        }

        if visited.contains(&node_id) {
            return Ok(());
        }

        temp_visited.insert(node_id);

        // Visit all nodes that this node depends on (incoming edges)
        for edge in &self.edges {
            if edge.to == node_id {
                self.topological_sort_util(edge.from, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);

        Ok(())
    }

    /// Generate forward pass code
    pub fn generate_forward_code(&self) -> Result<ASTNode, String> {
        let sorted_nodes = self.topological_sort()?;
        let mut forward_ops = Vec::new();

        for node_id in sorted_nodes {
            if let Some(node) = self.get_node(node_id) {
                let forward_op = ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol(format!("compute_{}", node.op_type))),
                    ASTNodeRef::direct(ASTNode::number(node_id as f64)),
                ]);
                forward_ops.push(ASTNodeRef::direct(forward_op));
            }
        }

        Ok(ASTNode::list(forward_ops))
    }

    /// Generate backward pass code
    pub fn generate_backward_code(&self) -> Result<ASTNode, String> {
        let sorted_nodes = self.topological_sort()?;
        let mut backward_ops = Vec::new();

        // Reverse order for backward pass
        for node_id in sorted_nodes.iter().rev() {
            if let Some(node) = self.get_node(*node_id) {
                if node.requires_grad {
                    let backward_op = ASTNode::list(vec![
                        ASTNodeRef::direct(ASTNode::symbol(format!("backward_{}", node.op_type))),
                        ASTNodeRef::direct(ASTNode::number(*node_id as f64)),
                    ]);
                    backward_ops.push(ASTNodeRef::direct(backward_op));
                }
            }
        }

        Ok(ASTNode::list(backward_ops))
    }
}

impl fmt::Display for DiffMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@diff(wrt={:?}, dir={:?}, order={}, mode={:?})", 
               self.wrt_vars, self.direction, self.order, self.mode)
    }
}

impl fmt::Display for DiffMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffMode::Dynamic => write!(f, "dynamic"),
            DiffMode::Static => write!(f, "static"),
        }
    }
}

impl fmt::Display for DiffDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffDirection::Forward => write!(f, "forward"),
            DiffDirection::Reverse => write!(f, "reverse"),
            DiffDirection::Mixed(vars) => write!(f, "mixed({:?})", vars),
        }
    }
}