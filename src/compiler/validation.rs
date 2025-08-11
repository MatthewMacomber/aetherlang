// Comprehensive Validation Framework for Aether Compiler
// Implements multi-stage validation with syntax, semantic, and type checking

use std::collections::{HashMap, HashSet};
use crate::compiler::ast::{ASTNode, ASTNodeRef, AtomValue};
use crate::compiler::comprehensive_errors::{CompilerError, SourceLocation};
use crate::compiler::types::{Type, PrimitiveType, Shape};
use crate::compiler::symbol_table::SymbolTable;

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, Vec<CompilerError>>;

/// Core validation trait that all validators must implement
pub trait AetherValidator {
    /// Validate syntax of Aether code
    fn validate_syntax(&self, code: &str) -> ValidationResult<()>;
    
    /// Validate semantic correctness of AST
    fn validate_semantics(&self, ast: &ASTNode) -> ValidationResult<()>;
    
    /// Validate type correctness and inference
    fn validate_types(&self, ast: &ASTNode) -> ValidationResult<()>;
}

/// Validation context containing symbol information and state
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Symbol table for tracking defined variables and functions
    pub symbol_table: SymbolTable,
    /// Type bindings for variables
    pub type_bindings: HashMap<String, Type>,
    /// Function signatures
    pub function_signatures: HashMap<String, FunctionSignature>,
    /// Current scope level
    pub scope_level: u32,
    /// Defined variables in current scope
    pub defined_variables: HashSet<String>,
    /// Used variables (for unused variable detection)
    pub used_variables: HashSet<String>,
}

/// Function signature information
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub location: SourceLocation,
}

/// Function parameter information
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
    pub optional: bool,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new() -> Self {
        let mut context = ValidationContext {
            symbol_table: SymbolTable::new(),
            type_bindings: HashMap::new(),
            function_signatures: HashMap::new(),
            scope_level: 0,
            defined_variables: HashSet::new(),
            used_variables: HashSet::new(),
        };
        context.init_builtins();
        context
    }

    /// Initialize built-in functions and types
    fn init_builtins(&mut self) {
        // Built-in arithmetic functions
        self.add_builtin_function("+", vec![
            Parameter { name: "a".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
            Parameter { name: "b".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
        ], Type::Primitive(PrimitiveType::Int32));
        
        self.add_builtin_function("-", vec![
            Parameter { name: "a".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
            Parameter { name: "b".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
        ], Type::Primitive(PrimitiveType::Int32));
        
        self.add_builtin_function("*", vec![
            Parameter { name: "a".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
            Parameter { name: "b".to_string(), param_type: Type::Primitive(PrimitiveType::Int32), optional: false },
        ], Type::Primitive(PrimitiveType::Int32));
        
        self.add_builtin_function("/", vec![
            Parameter { name: "a".to_string(), param_type: Type::Primitive(PrimitiveType::Float32), optional: false },
            Parameter { name: "b".to_string(), param_type: Type::Primitive(PrimitiveType::Float32), optional: false },
        ], Type::Primitive(PrimitiveType::Float32));
    }

    /// Add a built-in function to the context
    fn add_builtin_function(&mut self, name: &str, parameters: Vec<Parameter>, return_type: Type) {
        let signature = FunctionSignature {
            name: name.to_string(),
            parameters,
            return_type,
            location: SourceLocation::unknown(),
        };
        self.function_signatures.insert(name.to_string(), signature);
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scope_level += 1;
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if self.scope_level > 0 {
            self.scope_level -= 1;
        }
    }

    /// Define a variable in current scope
    pub fn define_variable(&mut self, name: String, var_type: Type) {
        self.defined_variables.insert(name.clone());
        self.type_bindings.insert(name, var_type);
    }

    /// Mark a variable as used
    pub fn use_variable(&mut self, name: &str) {
        self.used_variables.insert(name.to_string());
    }

    /// Check if a variable is defined
    pub fn is_variable_defined(&self, name: &str) -> bool {
        self.defined_variables.contains(name)
    }

    /// Get variable type
    pub fn get_variable_type(&self, name: &str) -> Option<&Type> {
        self.type_bindings.get(name)
    }

    /// Check if a function is defined
    pub fn is_function_defined(&self, name: &str) -> bool {
        self.function_signatures.contains_key(name)
    }

    /// Get function signature
    pub fn get_function_signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.function_signatures.get(name)
    }
}
/// Syntax validator for Aether code
pub struct SyntaxValidator {
    context: ValidationContext,
}

impl SyntaxValidator {
    pub fn new() -> Self {
        SyntaxValidator {
            context: ValidationContext::new(),
        }
    }

    /// Check for balanced parentheses in S-expressions
    fn check_balanced_parens(&self, code: &str) -> ValidationResult<()> {
        // Use the validation context to track state
        self.context.set_current_operation("balanced_parens_check");
        
        // Continue with the existing implementation
        let mut stack = Vec::new();
        let mut line = 1;
        let mut column = 1;
        let mut errors = Vec::new();

        for ch in code.chars() {
            match ch {
                '(' => {
                    stack.push((ch, line, column));
                }
                ')' => {
                    if stack.is_empty() {
                        errors.push(CompilerError::SyntaxError {
                            location: SourceLocation::new("<input>".to_string(), line, column),
                            message: "Unmatched closing parenthesis".to_string(),
                            suggestions: vec!["Check for missing opening parenthesis".to_string()],
                        });
                    } else {
                        stack.pop();
                    }
                }
                '\n' => {
                    line += 1;
                    column = 1;
                    continue;
                }
                _ => {}
            }
            column += 1;
        }

        // Check for unmatched opening parentheses
        for (_, line, column) in stack {
            errors.push(CompilerError::SyntaxError {
                location: SourceLocation::new("<input>".to_string(), line, column),
                message: "Unmatched opening parenthesis".to_string(),
                suggestions: vec!["Check for missing closing parenthesis".to_string()],
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate basic syntax rules
    fn check_syntax_rules(&self, code: &str) -> ValidationResult<()> {
        let mut errors = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            let line_number = line_num + 1;
            
            // Check for invalid characters
            for (col_num, ch) in line.char_indices() {
                let column = col_num + 1;
                if ch.is_control() && ch != '\t' && ch != '\n' && ch != '\r' {
                    errors.push(CompilerError::SyntaxError {
                        location: SourceLocation::new("<input>".to_string(), line_number, column),
                        message: format!("Invalid control character: {:?}", ch),
                        suggestions: vec!["Remove or replace invalid control characters".to_string()],
                    });
                }
            }

            // Check for unterminated strings
            let mut in_string = false;
            let mut escaped = false;
            for (col_num, ch) in line.char_indices() {
                let column = col_num + 1;
                match ch {
                    '"' if !escaped => in_string = !in_string,
                    '\\' if in_string => escaped = !escaped,
                    _ => escaped = false,
                }
            }
            
            if in_string {
                errors.push(CompilerError::SyntaxError {
                    location: SourceLocation::new("<input>".to_string(), line_number, line.len()),
                    message: "Unterminated string literal".to_string(),
                    suggestions: vec!["Add closing quote to string literal".to_string()],
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl AetherValidator for SyntaxValidator {
    fn validate_syntax(&self, code: &str) -> ValidationResult<()> {
        let mut all_errors = Vec::new();

        // Check balanced parentheses
        if let Err(mut errors) = self.check_balanced_parens(code) {
            all_errors.append(&mut errors);
        }

        // Check basic syntax rules
        if let Err(mut errors) = self.check_syntax_rules(code) {
            all_errors.append(&mut errors);
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }

    fn validate_semantics(&self, _ast: &ASTNode) -> ValidationResult<()> {
        // Syntax validator doesn't perform semantic validation
        Ok(())
    }

    fn validate_types(&self, _ast: &ASTNode) -> ValidationResult<()> {
        // Syntax validator doesn't perform type validation
        Ok(())
    }
}

/// Semantic validator for Aether AST
pub struct SemanticValidator {
    context: ValidationContext,
}

impl SemanticValidator {
    pub fn new() -> Self {
        SemanticValidator {
            context: ValidationContext::new(),
        }
    }

    /// Validate variable and function references
    fn validate_references(&mut self, ast: &ASTNode) -> ValidationResult<()> {
        let mut errors = Vec::new();
        self.validate_references_recursive(ast, &mut errors);
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Recursively validate references in AST
    fn validate_references_recursive(&mut self, ast: &ASTNode, errors: &mut Vec<CompilerError>) {
        match ast {
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                // Check if it's a variable reference
                if !self.context.is_variable_defined(name) && !self.context.is_function_defined(name) {
                    // Suggest similar names
                    let suggestions = self.suggest_similar_names(name);
                    let suggestion_text = if suggestions.is_empty() {
                        String::new()
                    } else {
                        format!(" Did you mean: {}?", suggestions.join(", "))
                    };
                    
                    errors.push(CompilerError::UndefinedVariable {
                        name: name.clone(),
                        location: SourceLocation::unknown(),
                        suggestions,
                    });
                } else {
                    self.context.use_variable(name);
                }
            }
            ASTNode::List(nodes) => {
                // Handle function calls and special forms
                if let Some(ASTNodeRef::Direct(first_node)) = nodes.first() {
                    if let ASTNode::Atom(AtomValue::Symbol(func_name)) = first_node.as_ref() {
                        match func_name.as_str() {
                            "let" => self.validate_let_binding(nodes, errors),
                            "defn" => self.validate_function_definition(nodes, errors),
                            _ => {
                                // Regular function call
                                if !self.context.is_function_defined(func_name) {
                                    let suggestions = self.suggest_similar_names(func_name);
                                    errors.push(CompilerError::UndefinedFunction {
                                        name: func_name.clone(),
                                        location: SourceLocation::unknown(),
                                        suggestions,
                                    });
                                }
                                
                                // Validate arguments
                                for node_ref in nodes.iter().skip(1) {
                                    if let ASTNodeRef::Direct(node) = node_ref {
                                        self.validate_references_recursive(node, errors);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Validate let binding syntax and semantics
    fn validate_let_binding(&mut self, nodes: &[ASTNodeRef], errors: &mut Vec<CompilerError>) {
        if nodes.len() < 3 {
            errors.push(CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "let binding requires at least variable name and value".to_string(),
                suggestions: vec!["Use format: (let variable-name value)".to_string()],
            });
            return;
        }

        // Extract variable name
        if let Some(ASTNodeRef::Direct(var_node)) = nodes.get(1) {
            if let ASTNode::Atom(AtomValue::Symbol(var_name)) = var_node.as_ref() {
                // Validate the value expression
                if let Some(ASTNodeRef::Direct(value_node)) = nodes.get(2) {
                    self.validate_references_recursive(value_node, errors);
                    
                    // Define the variable in context
                    self.context.define_variable(var_name.clone(), Type::Dynamic);
                }
            } else {
                errors.push(CompilerError::SyntaxError {
                    location: SourceLocation::unknown(),
                    message: "let binding variable name must be a symbol".to_string(),
                    suggestions: vec!["Use a valid identifier for the variable name".to_string()],
                });
            }
        }
    }

    /// Validate function definition
    fn validate_function_definition(&mut self, nodes: &[ASTNodeRef], errors: &mut Vec<CompilerError>) {
        if nodes.len() < 4 {
            errors.push(CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "function definition requires name, parameters, and body".to_string(),
                suggestions: vec!["Use format: (defn function-name [params] body)".to_string()],
            });
            return;
        }

        // Extract function name
        if let Some(ASTNodeRef::Direct(name_node)) = nodes.get(1) {
            if let ASTNode::Atom(AtomValue::Symbol(func_name)) = name_node.as_ref() {
                // Enter new scope for function
                self.context.enter_scope();

                // Validate parameters
                if let Some(ASTNodeRef::Direct(params_node)) = nodes.get(2) {
                    if let ASTNode::List(param_refs) = params_node.as_ref() {
                        let mut parameters = Vec::new();
                        for param_ref in param_refs {
                            if let ASTNodeRef::Direct(param_node) = param_ref {
                                if let ASTNode::Atom(AtomValue::Symbol(param_name)) = param_node.as_ref() {
                                    parameters.push(Parameter {
                                        name: param_name.clone(),
                                        param_type: Type::Dynamic,
                                        optional: false,
                                    });
                                    self.context.define_variable(param_name.clone(), Type::Dynamic);
                                }
                            }
                        }

                        // Create function signature
                        let signature = FunctionSignature {
                            name: func_name.clone(),
                            parameters,
                            return_type: Type::Dynamic,
                            location: SourceLocation::unknown(),
                        };
                        self.context.function_signatures.insert(func_name.clone(), signature);
                    }
                }

                // Validate function body
                for node_ref in nodes.iter().skip(3) {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.validate_references_recursive(node, errors);
                    }
                }

                // Exit function scope
                self.context.exit_scope();
            }
        }
    }

    /// Suggest similar variable/function names for typos
    fn suggest_similar_names(&self, name: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Check defined variables
        for var_name in &self.context.defined_variables {
            if self.is_similar(name, var_name) {
                suggestions.push(var_name.clone());
            }
        }
        
        // Check function names
        for func_name in self.context.function_signatures.keys() {
            if self.is_similar(name, func_name) {
                suggestions.push(func_name.clone());
            }
        }
        
        suggestions.sort();
        suggestions.truncate(3); // Limit to 3 suggestions
        suggestions
    }

    /// Simple similarity check (Levenshtein distance <= 2)
    fn is_similar(&self, a: &str, b: &str) -> bool {
        if a == b {
            return false; // Don't suggest exact matches
        }
        
        let len_diff = (a.len() as i32 - b.len() as i32).abs();
        if len_diff > 2 {
            return false;
        }
        
        // Simple edit distance check
        self.edit_distance(a, b) <= 2
    }

    /// Calculate edit distance between two strings
    fn edit_distance(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let a_len = a_chars.len();
        let b_len = b_chars.len();
        
        let mut dp = vec![vec![0; b_len + 1]; a_len + 1];
        
        for i in 0..=a_len {
            dp[i][0] = i;
        }
        for j in 0..=b_len {
            dp[0][j] = j;
        }
        
        for i in 1..=a_len {
            for j in 1..=b_len {
                let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
                dp[i][j] = std::cmp::min(
                    std::cmp::min(dp[i-1][j] + 1, dp[i][j-1] + 1),
                    dp[i-1][j-1] + cost
                );
            }
        }
        
        dp[a_len][b_len]
    }
}

impl AetherValidator for SemanticValidator {
    fn validate_syntax(&self, _code: &str) -> ValidationResult<()> {
        // Semantic validator doesn't perform syntax validation
        Ok(())
    }

    fn validate_semantics(&self, ast: &ASTNode) -> ValidationResult<()> {
        let mut validator = self.clone();
        validator.validate_references(ast)
    }

    fn validate_types(&self, _ast: &ASTNode) -> ValidationResult<()> {
        // Semantic validator doesn't perform type validation
        Ok(())
    }
}

impl Clone for SemanticValidator {
    fn clone(&self) -> Self {
        SemanticValidator {
            context: self.context.clone(),
        }
    }
}

/// Type validator for Aether AST with tensor shape validation
pub struct TypeValidator {
    context: ValidationContext,
}

impl TypeValidator {
    pub fn new() -> Self {
        TypeValidator {
            context: ValidationContext::new(),
        }
    }

    /// Validate types throughout the AST
    fn validate_types_recursive(&mut self, ast: &ASTNode) -> ValidationResult<Type> {
        match ast {
            ASTNode::Atom(atom) => self.validate_atom_type(atom),
            ASTNode::List(nodes) => self.validate_list_type(nodes),
            ASTNode::Graph { nodes, .. } => {
                // For graph nodes, validate all contained nodes
                let mut errors = Vec::new();
                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        if let Err(mut node_errors) = self.validate_types_recursive(node) {
                            errors.append(&mut node_errors);
                        }
                    }
                }
                
                if errors.is_empty() {
                    Ok(Type::Dynamic)
                } else {
                    Err(errors)
                }
            }
        }
    }

    /// Validate atomic value types
    fn validate_atom_type(&mut self, atom: &AtomValue) -> ValidationResult<Type> {
        match atom {
            AtomValue::Number(_) => Ok(Type::Primitive(PrimitiveType::Float64)),
            AtomValue::String(_) => Ok(Type::Primitive(PrimitiveType::String)),
            AtomValue::Boolean(_) => Ok(Type::Primitive(PrimitiveType::Bool)),
            AtomValue::Nil => Ok(Type::Dynamic),
            AtomValue::Symbol(name) => {
                if let Some(var_type) = self.context.get_variable_type(name) {
                    Ok(var_type.clone())
                } else if self.context.is_function_defined(name) {
                    // Return function type
                    if let Some(signature) = self.context.get_function_signature(name) {
                        Ok(Type::Function {
                            params: signature.parameters.iter().map(|p| p.param_type.clone()).collect(),
                            return_type: Box::new(signature.return_type.clone()),
                            is_differentiable: false,
                        })
                    } else {
                        Ok(Type::Dynamic)
                    }
                } else {
                    Err(vec![CompilerError::UndefinedVariable {
                        name: name.clone(),
                        location: SourceLocation::unknown(),
                        suggestions: Vec::new(),
                    }])
                }
            }
            AtomValue::Token(_) => Ok(Type::Dynamic),
        }
    }

    /// Validate list (function call or special form) types
    fn validate_list_type(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.is_empty() {
            return Ok(Type::Dynamic);
        }

        // Get the first element (function or operator)
        if let Some(ASTNodeRef::Direct(first_node)) = nodes.first() {
            if let ASTNode::Atom(AtomValue::Symbol(func_name)) = first_node.as_ref() {
                match func_name.as_str() {
                    "let" => self.validate_let_type(nodes),
                    "defn" => self.validate_function_def_type(nodes),
                    "tensor" => self.validate_tensor_creation(nodes),
                    "reshape" => self.validate_tensor_reshape(nodes),
                    "matmul" => self.validate_matrix_multiplication(nodes),
                    "+" | "-" | "*" | "/" => self.validate_arithmetic_operation(func_name, nodes),
                    _ => self.validate_function_call_type(func_name, nodes),
                }
            } else {
                // First element is not a symbol, validate as expression
                self.validate_types_recursive(first_node)
            }
        } else {
            Ok(Type::Dynamic)
        }
    }

    /// Validate let binding type
    fn validate_let_type(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 3 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "let binding requires variable name and value".to_string(),
                suggestions: vec!["Use format: (let variable-name value)".to_string()],
            }]);
        }

        // Validate the value expression and infer type
        if let Some(ASTNodeRef::Direct(value_node)) = nodes.get(2) {
            let value_type = self.validate_types_recursive(value_node)?;
            
            // Bind the variable to its inferred type
            if let Some(ASTNodeRef::Direct(var_node)) = nodes.get(1) {
                if let ASTNode::Atom(AtomValue::Symbol(var_name)) = var_node.as_ref() {
                    self.context.define_variable(var_name.clone(), value_type.clone());
                }
            }
            
            Ok(value_type)
        } else {
            Ok(Type::Dynamic)
        }
    }

    /// Validate function definition type
    fn validate_function_def_type(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 4 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "function definition requires name, parameters, and body".to_string(),
                suggestions: vec!["Use format: (defn function-name [params] body)".to_string()],
            }]);
        }

        // Enter new scope for function
        self.context.enter_scope();

        // Validate function body and infer return type
        let mut return_type = Type::Dynamic;
        for node_ref in nodes.iter().skip(3) {
            if let ASTNodeRef::Direct(node) = node_ref {
                return_type = self.validate_types_recursive(node)?;
            }
        }

        // Exit function scope
        self.context.exit_scope();

        Ok(Type::Function {
            params: vec![Type::Dynamic], // Simplified for now
            return_type: Box::new(return_type),
            is_differentiable: false,
        })
    }

    /// Validate tensor creation with shape checking
    fn validate_tensor_creation(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 2 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "tensor creation requires data".to_string(),
                suggestions: vec!["Use format: (tensor data)".to_string()],
            }]);
        }

        // Validate tensor data and infer shape
        if let Some(ASTNodeRef::Direct(data_node)) = nodes.get(1) {
            let (element_type, shape) = self.infer_tensor_shape(data_node)?;
            
            Ok(Type::Tensor {
                element_type: Box::new(element_type),
                shape,
            })
        } else {
            Ok(Type::Dynamic)
        }
    }

    /// Infer tensor shape from nested list structure
    fn infer_tensor_shape(&mut self, node: &ASTNode) -> ValidationResult<(Type, Shape)> {
        match node {
            ASTNode::List(elements) => {
                if elements.is_empty() {
                    return Ok((Type::Dynamic, Shape::Unknown));
                }

                let mut dimensions = vec![elements.len()];
                let mut element_type = Type::Dynamic;

                // Check first element to determine inner structure
                if let Some(ASTNodeRef::Direct(first_elem)) = elements.first() {
                    match first_elem.as_ref() {
                        ASTNode::List(_) => {
                            // Nested list - recurse to get inner dimensions
                            let (inner_type, inner_shape) = self.infer_tensor_shape(first_elem)?;
                            element_type = inner_type;
                            
                            if let Shape::Concrete(inner_dims) = inner_shape {
                                dimensions.extend(inner_dims);
                            }
                        }
                        ASTNode::Atom(atom) => {
                            element_type = self.validate_atom_type(atom)?;
                        }
                        _ => {}
                    }
                }

                // Validate all elements have consistent shape
                for (i, elem_ref) in elements.iter().enumerate() {
                    if let ASTNodeRef::Direct(elem_node) = elem_ref {
                        match elem_node.as_ref() {
                            ASTNode::List(inner_elements) => {
                                if inner_elements.len() != *dimensions.get(1).unwrap_or(&0) {
                                    return Err(vec![CompilerError::TensorShapeMismatch {
                                        expected: dimensions.clone(),
                                        found: vec![elements.len(), inner_elements.len()],
                                        location: SourceLocation::unknown(),
                                    }]);
                                }
                            }
                            ASTNode::Atom(atom) => {
                                let atom_type = self.validate_atom_type(atom)?;
                                if !self.types_compatible(&element_type, &atom_type) {
                                    return Err(vec![CompilerError::TypeMismatch {
                                        expected: format!("{:?}", element_type),
                                        found: format!("{:?}", atom_type),
                                        location: SourceLocation::unknown(),
                                        context: Some("tensor element type validation".to_string()),
                                    }]);
                                }
                            }
                            _ => {}
                        }
                    }
                }

                Ok((element_type, Shape::Concrete(dimensions)))
            }
            ASTNode::Atom(atom) => {
                let atom_type = self.validate_atom_type(atom)?;
                Ok((atom_type, Shape::Concrete(vec![]))) // Scalar
            }
            _ => Ok((Type::Dynamic, Shape::Unknown)),
        }
    }

    /// Validate tensor reshape operation
    fn validate_tensor_reshape(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 3 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "reshape requires tensor and new shape".to_string(),
                suggestions: vec!["Use format: (reshape tensor [new-dimensions])".to_string()],
            }]);
        }

        // Validate input tensor
        if let Some(ASTNodeRef::Direct(tensor_node)) = nodes.get(1) {
            let tensor_type = self.validate_types_recursive(tensor_node)?;
            
            if let Type::Tensor { element_type, shape } = tensor_type {
                // Validate new shape
                if let Some(ASTNodeRef::Direct(shape_node)) = nodes.get(2) {
                    let new_shape = self.parse_shape_specification(shape_node)?;
                    
                    // Check that total elements remain the same
                    if let (Shape::Concrete(old_dims), Shape::Concrete(new_dims)) = (&shape, &new_shape) {
                        let old_total: usize = old_dims.iter().product();
                        let new_total: usize = new_dims.iter().product();
                        
                        if old_total != new_total {
                            return Err(vec![CompilerError::TensorShapeMismatch {
                                expected: old_dims.clone(),
                                found: new_dims.clone(),
                                location: SourceLocation::unknown(),
                            }]);
                        }
                    }
                    
                    Ok(Type::Tensor {
                        element_type,
                        shape: new_shape,
                    })
                } else {
                    Ok(Type::Tensor {
                        element_type,
                        shape: Shape::Unknown,
                    })
                }
            } else {
                Err(vec![CompilerError::TypeMismatch {
                    expected: "tensor".to_string(),
                    found: format!("{:?}", tensor_type),
                    location: SourceLocation::unknown(),
                    context: Some("tensor reshape operation".to_string()),
                }])
            }
        } else {
            Ok(Type::Dynamic)
        }
    }

    /// Validate matrix multiplication with shape compatibility
    fn validate_matrix_multiplication(&mut self, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 3 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: "matrix multiplication requires two tensors".to_string(),
                suggestions: vec!["Use format: (matmul tensor1 tensor2)".to_string()],
            }]);
        }

        // Validate both operands
        let left_type = if let Some(ASTNodeRef::Direct(left_node)) = nodes.get(1) {
            self.validate_types_recursive(left_node)?
        } else {
            return Ok(Type::Dynamic);
        };

        let right_type = if let Some(ASTNodeRef::Direct(right_node)) = nodes.get(2) {
            self.validate_types_recursive(right_node)?
        } else {
            return Ok(Type::Dynamic);
        };

        // Check tensor types and shapes
        match (&left_type, &right_type) {
            (Type::Tensor { element_type: left_elem, shape: left_shape },
             Type::Tensor { element_type: right_elem, shape: right_shape }) => {
                
                // Check element type compatibility
                if !self.types_compatible(&left_elem, &right_elem) {
                    return Err(vec![CompilerError::TypeMismatch {
                        expected: format!("{:?}", left_elem),
                        found: format!("{:?}", right_elem),
                        location: SourceLocation::unknown(),
                        context: Some("matrix multiplication element types".to_string()),
                    }]);
                }

                // Check shape compatibility for matrix multiplication
                match (left_shape, right_shape) {
                    (Shape::Concrete(left_dims), Shape::Concrete(right_dims)) => {
                        if left_dims.len() >= 2 && right_dims.len() >= 2 {
                            let left_cols = left_dims[left_dims.len() - 1];
                            let right_rows = right_dims[right_dims.len() - 2];
                            
                            if left_cols != right_rows {
                                return Err(vec![CompilerError::TensorShapeMismatch {
                                    expected: left_dims.clone(),
                                    found: right_dims.clone(),
                                    location: SourceLocation::unknown(),
                                }]);
                            }

                            // Result shape: [..., left_rows, right_cols]
                            let mut result_dims = left_dims[..left_dims.len()-1].to_vec();
                            result_dims.push(right_dims[right_dims.len() - 1]);
                            
                            Ok(Type::Tensor {
                                element_type: Box::new((**left_elem).clone()),
                                shape: Shape::Concrete(result_dims),
                            })
                        } else {
                            Err(vec![CompilerError::SyntaxError {
                                location: SourceLocation::unknown(),
                                message: "matrix multiplication requires at least 2D tensors".to_string(),
                                suggestions: vec!["Ensure tensors have at least 2 dimensions".to_string()],
                            }])
                        }
                    }
                    _ => Ok(Type::Tensor {
                        element_type: Box::new((**left_elem).clone()),
                        shape: Shape::Unknown,
                    }),
                }
            }
            _ => Err(vec![CompilerError::TypeMismatch {
                expected: "tensor".to_string(),
                found: format!("left: {:?}, right: {:?}", left_type, right_type),
                location: SourceLocation::unknown(),
                context: Some("matrix multiplication operands".to_string()),
            }])
        }
    }

    /// Validate arithmetic operations
    fn validate_arithmetic_operation(&mut self, op: &str, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if nodes.len() < 3 {
            return Err(vec![CompilerError::SyntaxError {
                location: SourceLocation::unknown(),
                message: format!("arithmetic operation '{}' requires two operands", op),
                suggestions: vec![format!("Use format: ({} operand1 operand2)", op)],
            }]);
        }

        let left_type = if let Some(ASTNodeRef::Direct(left_node)) = nodes.get(1) {
            self.validate_types_recursive(left_node)?
        } else {
            return Ok(Type::Dynamic);
        };

        let right_type = if let Some(ASTNodeRef::Direct(right_node)) = nodes.get(2) {
            self.validate_types_recursive(right_node)?
        } else {
            return Ok(Type::Dynamic);
        };

        // Check type compatibility
        if self.types_compatible(&left_type, &right_type) {
            Ok(self.promote_numeric_type(&left_type, &right_type))
        } else {
            Err(vec![CompilerError::TypeMismatch {
                expected: format!("{:?}", left_type),
                found: format!("{:?}", right_type),
                location: SourceLocation::unknown(),
                context: Some("arithmetic operation operands".to_string()),
            }])
        }
    }

    /// Validate function call type
    fn validate_function_call_type(&mut self, func_name: &str, nodes: &[ASTNodeRef]) -> ValidationResult<Type> {
        if let Some(signature) = self.context.get_function_signature(func_name).cloned() {
            // Validate argument count
            let expected_args = signature.parameters.len();
            let actual_args = nodes.len() - 1; // Subtract function name

            if actual_args != expected_args {
                return Err(vec![CompilerError::SyntaxError {
                    location: SourceLocation::unknown(),
                    message: format!("function '{}' expects {} arguments, got {}", 
                                    func_name, expected_args, actual_args),
                    suggestions: vec![format!("Provide exactly {} arguments to function '{}'", expected_args, func_name)],
                }]);
            }

            // Validate argument types
            for (i, param) in signature.parameters.iter().enumerate() {
                if let Some(ASTNodeRef::Direct(arg_node)) = nodes.get(i + 1) {
                    let arg_type = self.validate_types_recursive(arg_node)?;
                    if !self.types_compatible(&param.param_type, &arg_type) {
                        return Err(vec![CompilerError::TypeMismatch {
                            expected: format!("{:?}", param.param_type),
                            found: format!("{:?}", arg_type),
                            location: SourceLocation::unknown(),
                            context: Some(format!("function '{}' argument validation", func_name)),
                        }]);
                    }
                }
            }

            Ok(signature.return_type.clone())
        } else {
            Err(vec![CompilerError::UndefinedFunction {
                name: func_name.to_string(),
                location: SourceLocation::unknown(),
                suggestions: Vec::new(),
            }])
        }
    }

    /// Parse shape specification from AST node
    fn parse_shape_specification(&self, node: &ASTNode) -> ValidationResult<Shape> {
        match node {
            ASTNode::List(elements) => {
                let mut dimensions = Vec::new();
                for elem_ref in elements {
                    if let ASTNodeRef::Direct(elem_node) = elem_ref {
                        if let ASTNode::Atom(AtomValue::Number(n)) = elem_node.as_ref() {
                            if *n >= 0.0 && n.fract() == 0.0 {
                                dimensions.push(*n as usize);
                            } else {
                                return Err(vec![CompilerError::SyntaxError {
                                    location: SourceLocation::unknown(),
                                    message: "shape dimensions must be non-negative integers".to_string(),
                                    suggestions: vec!["Use positive integer values for tensor dimensions".to_string()],
                                }]);
                            }
                        }
                    }
                }
                Ok(Shape::Concrete(dimensions))
            }
            _ => Ok(Shape::Unknown),
        }
    }

    /// Check if two types are compatible
    fn types_compatible(&self, type1: &Type, type2: &Type) -> bool {
        match (type1, type2) {
            (Type::Dynamic, _) | (_, Type::Dynamic) => true,
            (Type::Primitive(p1), Type::Primitive(p2)) => self.primitives_compatible(p1, p2),
            (Type::Tensor { element_type: e1, .. }, Type::Tensor { element_type: e2, .. }) => {
                self.types_compatible(e1, e2)
            }
            _ => type1 == type2,
        }
    }

    /// Check if primitive types are compatible
    fn primitives_compatible(&self, p1: &PrimitiveType, p2: &PrimitiveType) -> bool {
        match (p1, p2) {
            (PrimitiveType::Int32, PrimitiveType::Float32) |
            (PrimitiveType::Float32, PrimitiveType::Int32) |
            (PrimitiveType::Int32, PrimitiveType::Float64) |
            (PrimitiveType::Float64, PrimitiveType::Int32) |
            (PrimitiveType::Float32, PrimitiveType::Float64) |
            (PrimitiveType::Float64, PrimitiveType::Float32) => true,
            _ => p1 == p2,
        }
    }

    /// Promote numeric types for arithmetic operations
    fn promote_numeric_type(&self, type1: &Type, type2: &Type) -> Type {
        match (type1, type2) {
            (Type::Primitive(PrimitiveType::Float64), _) |
            (_, Type::Primitive(PrimitiveType::Float64)) => Type::Primitive(PrimitiveType::Float64),
            (Type::Primitive(PrimitiveType::Float32), _) |
            (_, Type::Primitive(PrimitiveType::Float32)) => Type::Primitive(PrimitiveType::Float32),
            _ => type1.clone(),
        }
    }
}

impl AetherValidator for TypeValidator {
    fn validate_syntax(&self, _code: &str) -> ValidationResult<()> {
        // Type validator doesn't perform syntax validation
        Ok(())
    }

    fn validate_semantics(&self, _ast: &ASTNode) -> ValidationResult<()> {
        // Type validator doesn't perform semantic validation
        Ok(())
    }

    fn validate_types(&self, ast: &ASTNode) -> ValidationResult<()> {
        let mut validator = self.clone();
        validator.validate_types_recursive(ast).map(|_| ())
    }
}

impl Clone for TypeValidator {
    fn clone(&self) -> Self {
        TypeValidator {
            context: self.context.clone(),
        }
    }
}

/// Comprehensive validator that orchestrates multi-stage validation
pub struct ComprehensiveValidator {
    syntax_validator: SyntaxValidator,
    semantic_validator: SemanticValidator,
    type_validator: TypeValidator,
}

impl ComprehensiveValidator {
    /// Create a new comprehensive validator
    pub fn new() -> Self {
        ComprehensiveValidator {
            syntax_validator: SyntaxValidator::new(),
            semantic_validator: SemanticValidator::new(),
            type_validator: TypeValidator::new(),
        }
    }

    /// Validate a complete Aether file with multi-stage validation
    pub fn validate_file(&self, file_path: &str) -> ValidationResult<ValidationReport> {
        // Read the file
        let code = std::fs::read_to_string(file_path)
            .map_err(|e| vec![CompilerError::IoError {
                message: format!("Failed to read file '{}': {}", file_path, e),
                location: SourceLocation::unknown(),
            }])?;

        self.validate_code(&code, file_path)
    }

    /// Validate Aether code with multi-stage validation
    pub fn validate_code(&self, code: &str, file_path: &str) -> ValidationResult<ValidationReport> {
        let mut all_errors = Vec::new();
        let mut warnings = Vec::new();

        // Stage 1: Syntax validation
        if let Err(mut syntax_errors) = self.syntax_validator.validate_syntax(code) {
            all_errors.append(&mut syntax_errors);
            // If syntax validation fails, we can't proceed to semantic/type validation
            return Ok(ValidationReport {
                file_path: file_path.to_string(),
                success: false,
                errors: all_errors,
                warnings,
                metrics: ValidationMetrics::default(),
            });
        }

        // Parse the code to AST for semantic and type validation
        let ast = match self.parse_code(code) {
            Ok(ast) => ast,
            Err(parse_errors) => {
                all_errors.extend(parse_errors);
                return Ok(ValidationReport {
                    file_path: file_path.to_string(),
                    success: false,
                    errors: all_errors,
                    warnings,
                    metrics: ValidationMetrics::default(),
                });
            }
        };

        // Stage 2: Semantic validation
        if let Err(mut semantic_errors) = self.semantic_validator.validate_semantics(&ast) {
            all_errors.append(&mut semantic_errors);
        }

        // Stage 3: Type validation
        if let Err(mut type_errors) = self.type_validator.validate_types(&ast) {
            all_errors.append(&mut type_errors);
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(code, &ast);

        // Generate warnings for potential issues
        self.generate_warnings(&ast, &mut warnings);

        Ok(ValidationReport {
            file_path: file_path.to_string(),
            success: all_errors.is_empty(),
            errors: all_errors,
            warnings,
            metrics,
        })
    }

    /// Parse code into AST (simplified parser for validation)
    fn parse_code(&self, code: &str) -> ValidationResult<ASTNode> {
        // This is a simplified parser for demonstration
        // In a real implementation, this would use the full Aether parser
        
        let trimmed = code.trim();
        if trimmed.is_empty() {
            return Ok(ASTNode::List(vec![]));
        }

        // Simple S-expression parsing
        if trimmed.starts_with('(') && trimmed.ends_with(')') {
            let inner = &trimmed[1..trimmed.len()-1];
            let tokens = self.tokenize(inner);
            let nodes = tokens.into_iter()
                .map(|token| ASTNodeRef::Direct(Box::new(self.parse_token(token))))
                .collect();
            Ok(ASTNode::List(nodes))
        } else {
            // Single atom
            Ok(self.parse_token(trimmed.to_string()))
        }
    }

    /// Simple tokenizer for S-expressions
    fn tokenize(&self, input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_string = false;
        let mut paren_depth = 0;

        for ch in input.chars() {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current_token.push(ch);
                }
                '(' if !in_string => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    paren_depth += 1;
                    current_token.push(ch);
                }
                ')' if !in_string => {
                    current_token.push(ch);
                    paren_depth -= 1;
                    if paren_depth == 0 {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                ' ' | '\t' | '\n' | '\r' if !in_string && paren_depth == 0 => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    /// Parse a single token into an AST node
    fn parse_token(&self, token: String) -> ASTNode {
        if token.starts_with('"') && token.ends_with('"') {
            // String literal
            let content = token[1..token.len()-1].to_string();
            ASTNode::Atom(AtomValue::String(content))
        } else if let Ok(num) = token.parse::<f64>() {
            // Number literal
            ASTNode::Atom(AtomValue::Number(num))
        } else if token == "true" || token == "false" {
            // Boolean literal
            ASTNode::Atom(AtomValue::Boolean(token == "true"))
        } else if token == "nil" {
            // Nil literal
            ASTNode::Atom(AtomValue::Nil)
        } else {
            // Symbol
            ASTNode::Atom(AtomValue::Symbol(token))
        }
    }

    /// Calculate validation metrics
    fn calculate_metrics(&self, code: &str, ast: &ASTNode) -> ValidationMetrics {
        let lines_of_code = code.lines().filter(|line| !line.trim().is_empty()).count();
        let (functions_validated, types_checked, tensor_operations_validated) = self.count_ast_elements(ast);

        ValidationMetrics {
            lines_of_code,
            functions_validated,
            types_checked,
            tensor_operations_validated,
        }
    }

    /// Count various elements in the AST for metrics
    fn count_ast_elements(&self, ast: &ASTNode) -> (usize, usize, usize) {
        let mut functions = 0;
        let mut types = 0;
        let mut tensor_ops = 0;

        self.count_elements_recursive(ast, &mut functions, &mut types, &mut tensor_ops);
        (functions, types, tensor_ops)
    }

    /// Recursively count elements in AST
    fn count_elements_recursive(&self, ast: &ASTNode, functions: &mut usize, types: &mut usize, tensor_ops: &mut usize) {
        match ast {
            ASTNode::List(nodes) => {
                if let Some(ASTNodeRef::Direct(first)) = nodes.first() {
                    if let ASTNode::Atom(AtomValue::Symbol(name)) = first.as_ref() {
                        match name.as_str() {
                            "defn" => *functions += 1,
                            "tensor" | "reshape" | "matmul" => *tensor_ops += 1,
                            _ => {}
                        }
                    }
                }

                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.count_elements_recursive(node, functions, types, tensor_ops);
                    }
                }
            }
            ASTNode::Graph { nodes, .. } => {
                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.count_elements_recursive(node, functions, types, tensor_ops);
                    }
                }
            }
            _ => {
                *types += 1; // Count atoms as type instances
            }
        }
    }

    /// Generate warnings for potential issues
    fn generate_warnings(&self, ast: &ASTNode, warnings: &mut Vec<String>) {
        // Check for potential performance issues
        self.check_performance_warnings(ast, warnings);
        
        // Check for style issues
        self.check_style_warnings(ast, warnings);
    }

    /// Check for performance-related warnings
    fn check_performance_warnings(&self, ast: &ASTNode, warnings: &mut Vec<String>) {
        match ast {
            ASTNode::List(nodes) => {
                if let Some(ASTNodeRef::Direct(first)) = nodes.first() {
                    if let ASTNode::Atom(AtomValue::Symbol(name)) = first.as_ref() {
                        match name.as_str() {
                            "reshape" => {
                                warnings.push("Consider avoiding frequent tensor reshaping for better performance".to_string());
                            }
                            "+" | "-" | "*" | "/" if nodes.len() > 10 => {
                                warnings.push("Large arithmetic expressions may benefit from vectorization".to_string());
                            }
                            _ => {}
                        }
                    }
                }

                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.check_performance_warnings(node, warnings);
                    }
                }
            }
            ASTNode::Graph { nodes, .. } => {
                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.check_performance_warnings(node, warnings);
                    }
                }
            }
            _ => {}
        }
    }

    /// Check for style-related warnings
    fn check_style_warnings(&self, ast: &ASTNode, warnings: &mut Vec<String>) {
        match ast {
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                if name.len() > 50 {
                    warnings.push(format!("Very long symbol name '{}' may hurt readability", name));
                }
                if name.contains("__") {
                    warnings.push(format!("Double underscore in symbol '{}' is discouraged", name));
                }
            }
            ASTNode::List(nodes) => {
                if nodes.len() > 20 {
                    warnings.push("Very long expression may benefit from being split into smaller parts".to_string());
                }

                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.check_style_warnings(node, warnings);
                    }
                }
            }
            ASTNode::Graph { nodes, .. } => {
                for node_ref in nodes {
                    if let ASTNodeRef::Direct(node) = node_ref {
                        self.check_style_warnings(node, warnings);
                    }
                }
            }
            _ => {}
        }
    }
}

impl AetherValidator for ComprehensiveValidator {
    fn validate_syntax(&self, code: &str) -> ValidationResult<()> {
        self.syntax_validator.validate_syntax(code)
    }

    fn validate_semantics(&self, ast: &ASTNode) -> ValidationResult<()> {
        self.semantic_validator.validate_semantics(ast)
    }

    fn validate_types(&self, ast: &ASTNode) -> ValidationResult<()> {
        self.type_validator.validate_types(ast)
    }
}

/// Validation report containing results and metrics
#[derive(Debug)]
pub struct ValidationReport {
    pub file_path: String,
    pub success: bool,
    pub errors: Vec<CompilerError>,
    pub warnings: Vec<String>,
    pub metrics: ValidationMetrics,
}

/// Validation metrics for reporting
#[derive(Debug, Default)]
pub struct ValidationMetrics {
    pub lines_of_code: usize,
    pub functions_validated: usize,
    pub types_checked: usize,
    pub tensor_operations_validated: usize,
}

impl ValidationReport {
    /// Create a success report
    pub fn success(file_path: String) -> Self {
        ValidationReport {
            file_path,
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
        }
    }

    /// Check if validation was successful
    pub fn is_success(&self) -> bool {
        self.success && self.errors.is_empty()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_validator_balanced_parens() {
        let validator = SyntaxValidator::new();
        
        // Valid balanced parentheses
        assert!(validator.validate_syntax("(+ 1 2)").is_ok());
        assert!(validator.validate_syntax("(defn foo (x) (+ x 1))").is_ok());
        
        // Invalid unbalanced parentheses
        assert!(validator.validate_syntax("(+ 1 2").is_err());
        assert!(validator.validate_syntax("+ 1 2)").is_err());
    }

    #[test]
    fn test_semantic_validator_undefined_variables() {
        let validator = SemanticValidator::new();
        let ast = ASTNode::Atom(AtomValue::Symbol("undefined_var".to_string()));
        
        let result = validator.validate_semantics(&ast);
        assert!(result.is_err());
        
        if let Err(errors) = result {
            assert!(matches!(errors[0], CompilerError::UndefinedVariable { .. }));
        }
    }

    #[test]
    fn test_type_validator_tensor_shapes() {
        let validator = TypeValidator::new();
        
        // Create a simple tensor creation AST
        let tensor_data = ASTNode::List(vec![
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(1.0)))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(2.0)))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(3.0)))),
        ]);
        
        let tensor_ast = ASTNode::List(vec![
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Symbol("tensor".to_string())))),
            ASTNodeRef::Direct(Box::new(tensor_data)),
        ]);
        
        let result = validator.validate_types(&tensor_ast);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comprehensive_validator() {
        let validator = ComprehensiveValidator::new();
        
        // Test valid code
        let valid_code = "(+ 1 2)";
        let result = validator.validate_code(valid_code, "test.ae");
        assert!(result.is_ok());
        
        if let Ok(report) = result {
            assert!(report.is_success());
        }
        
        // Test invalid code
        let invalid_code = "(+ 1 2";
        let result = validator.validate_code(invalid_code, "test.ae");
        assert!(result.is_ok()); // Returns Ok(report) but report.success is false
        
        if let Ok(report) = result {
            assert!(!report.is_success());
            assert!(report.error_count() > 0);
        }
    }
}