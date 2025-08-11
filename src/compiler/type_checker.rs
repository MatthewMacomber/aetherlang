// Aether Type Checker
// Static and dynamic type validation with gradual typing support

use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::compiler::types::{
    Type, PrimitiveType, Shape, LinearOwnership, 
    Lifetime, LinearConstraint, AllocationSite, AllocationType
};
use crate::compiler::type_inference::{TypeInference, InferenceError};
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
use crate::compiler::symbol_table::SymbolTable;

/// Type checking error
#[derive(Debug, Clone, PartialEq)]
pub enum TypeCheckError {
    /// Type mismatch
    TypeMismatch {
        expected: Type,
        actual: Type,
        location: String,
    },
    /// Undefined variable or function
    UndefinedSymbol {
        name: String,
        location: String,
    },
    /// Invalid operation for type
    InvalidOperation {
        operation: String,
        operand_types: Vec<Type>,
        location: String,
    },
    /// Shape mismatch in tensor operations
    ShapeMismatch {
        expected: Shape,
        actual: Shape,
        operation: String,
        location: String,
    },
    /// Linear type violation
    LinearTypeViolation {
        variable: String,
        violation: String,
        location: String,
    },
    /// Arity mismatch in function call
    ArityMismatch {
        expected: usize,
        actual: usize,
        function: String,
        location: String,
    },
    /// Type annotation parsing error
    InvalidTypeAnnotation {
        annotation: String,
        error: String,
        location: String,
    },
    /// Inference error
    InferenceError(InferenceError),
}

/// Type checking result
pub type TypeCheckResult<T> = Result<T, TypeCheckError>;

/// Type checking context
#[derive(Debug, Clone)]
pub struct TypeCheckContext {
    /// Symbol table for name resolution
    symbol_table: SymbolTable,
    /// Type bindings for variables
    type_bindings: HashMap<String, Type>,
    /// Linear type ownership tracking
    linear_ownership: HashMap<String, LinearOwnership>,
    /// Linear type constraints
    linear_constraints: Vec<LinearConstraint>,
    /// Allocation sites for memory management
    allocation_sites: HashMap<String, AllocationSite>,
    /// Active lifetimes in current scope
    active_lifetimes: HashMap<String, Lifetime>,
    /// Variables that have been used (for single-use checking)
    used_variables: HashSet<String>,
    /// Current scope level
    scope_level: u32,
    /// Whether to perform static checking (vs dynamic)
    static_mode: bool,
}

impl TypeCheckContext {
    /// Create new type checking context
    pub fn new(static_mode: bool) -> Self {
        let mut context = TypeCheckContext {
            symbol_table: SymbolTable::new(),
            type_bindings: HashMap::new(),
            linear_ownership: HashMap::new(),
            linear_constraints: Vec::new(),
            allocation_sites: HashMap::new(),
            active_lifetimes: HashMap::new(),
            used_variables: HashSet::new(),
            scope_level: 0,
            static_mode,
        };
        
        // Initialize built-in types and functions
        context.init_builtins();
        context
    }

    /// Initialize built-in types and functions
    fn init_builtins(&mut self) {
        // Built-in types
        self.bind_type("i8".to_string(), Type::primitive(PrimitiveType::Int8));
        self.bind_type("i16".to_string(), Type::primitive(PrimitiveType::Int16));
        self.bind_type("i32".to_string(), Type::primitive(PrimitiveType::Int32));
        self.bind_type("i64".to_string(), Type::primitive(PrimitiveType::Int64));
        self.bind_type("u8".to_string(), Type::primitive(PrimitiveType::UInt8));
        self.bind_type("u16".to_string(), Type::primitive(PrimitiveType::UInt16));
        self.bind_type("u32".to_string(), Type::primitive(PrimitiveType::UInt32));
        self.bind_type("u64".to_string(), Type::primitive(PrimitiveType::UInt64));
        self.bind_type("f32".to_string(), Type::primitive(PrimitiveType::Float32));
        self.bind_type("f64".to_string(), Type::primitive(PrimitiveType::Float64));
        self.bind_type("bool".to_string(), Type::primitive(PrimitiveType::Bool));
        self.bind_type("string".to_string(), Type::primitive(PrimitiveType::String));
        self.bind_type("char".to_string(), Type::primitive(PrimitiveType::Char));

        // Built-in functions
        self.bind_function("+".to_string(), vec![
            Type::primitive(PrimitiveType::Int32),
            Type::primitive(PrimitiveType::Int32)
        ], Type::primitive(PrimitiveType::Int32));

        self.bind_function("-".to_string(), vec![
            Type::primitive(PrimitiveType::Int32),
            Type::primitive(PrimitiveType::Int32)
        ], Type::primitive(PrimitiveType::Int32));

        self.bind_function("*".to_string(), vec![
            Type::primitive(PrimitiveType::Int32),
            Type::primitive(PrimitiveType::Int32)
        ], Type::primitive(PrimitiveType::Int32));

        self.bind_function("/".to_string(), vec![
            Type::primitive(PrimitiveType::Int32),
            Type::primitive(PrimitiveType::Int32)
        ], Type::primitive(PrimitiveType::Int32));

        // Tensor operations
        self.bind_tensor_operations();
    }

    /// Bind tensor operations
    fn bind_tensor_operations(&mut self) {
        // Matrix multiplication
        let tensor_f32 = Type::tensor(
            Type::primitive(PrimitiveType::Float32),
            vec![] // Shape will be inferred
        );
        self.bind_function("matmul".to_string(), vec![tensor_f32.clone(), tensor_f32.clone()], tensor_f32);

        // Element-wise operations
        let tensor_any = Type::tensor(Type::dynamic(), vec![]);
        self.bind_function("tensor_add".to_string(), vec![tensor_any.clone(), tensor_any.clone()], tensor_any.clone());
        self.bind_function("tensor_mul".to_string(), vec![tensor_any.clone(), tensor_any.clone()], tensor_any);
    }

    /// Bind type for variable
    pub fn bind_type(&mut self, name: String, type_: Type) {
        self.type_bindings.insert(name, type_);
    }

    /// Bind function type
    pub fn bind_function(&mut self, name: String, params: Vec<Type>, return_type: Type) {
        let function_type = Type::function(params, return_type);
        self.bind_type(name, function_type);
    }

    /// Look up type for variable
    pub fn lookup_type(&self, name: &str) -> Option<&Type> {
        self.type_bindings.get(name)
    }

    /// Enter new scope
    pub fn enter_scope(&mut self) {
        self.scope_level += 1;
        self.symbol_table.enter_scope();
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        // Check for unused linear variables before scope exit
        self.check_unused_linear_variables();
        
        // Clean up linear ownership for variables going out of scope
        let vars_to_remove: Vec<String> = self.linear_ownership
            .keys()
            .filter(|var| {
                // Remove variables that belong to current scope
                if let Some(lifetime) = self.active_lifetimes.get(*var) {
                    lifetime.scope_level >= self.scope_level
                } else {
                    true // Remove if no lifetime info
                }
            })
            .cloned()
            .collect();

        for var in vars_to_remove {
            if let Some(ownership) = self.linear_ownership.get(&var) {
                match ownership {
                    LinearOwnership::Owned | LinearOwnership::GpuOwned => {
                        // Variable goes out of scope - automatically deallocated
                        self.insert_deallocation(&var);
                    }
                    LinearOwnership::Borrowed(_) | LinearOwnership::Shared(_) | LinearOwnership::MutableBorrow(_) => {
                        // Borrowed references just expire
                    }
                    LinearOwnership::Moved | LinearOwnership::GpuMoved => {
                        // Already moved, nothing to do
                    }
                }
                self.linear_ownership.remove(&var);
            }
            self.allocation_sites.remove(&var);
        }

        // Clean up lifetimes for current scope
        self.active_lifetimes.retain(|_, lifetime| lifetime.scope_level < self.scope_level);
        
        // Clean up used variables for current scope
        self.used_variables.clear();

        self.scope_level -= 1;
        self.symbol_table.exit_scope();
    }

    /// Track linear ownership
    pub fn track_linear_ownership(&mut self, var: String, ownership: LinearOwnership) {
        self.linear_ownership.insert(var, ownership);
    }

    /// Check linear ownership
    pub fn check_linear_ownership(&self, var: &str) -> Option<&LinearOwnership> {
        self.linear_ownership.get(var)
    }

    /// Check if variable has been used
    pub fn is_variable_used(&self, var: &str) -> bool {
        self.used_variables.contains(var)
    }

    /// Check if lifetime is active
    pub fn is_lifetime_active(&self, lifetime_name: &str) -> bool {
        self.active_lifetimes.contains_key(lifetime_name)
    }

    /// Check if allocation site exists
    pub fn has_allocation_site(&self, var: &str) -> bool {
        self.allocation_sites.contains_key(var)
    }

    /// Get allocation site
    pub fn get_allocation_site(&self, var: &str) -> Option<&AllocationSite> {
        self.allocation_sites.get(var)
    }

    /// Check if variable has linear ownership
    pub fn has_linear_ownership(&self, var: &str) -> bool {
        self.linear_ownership.contains_key(var)
    }

    /// Clear linear constraints (for testing)
    pub fn clear_linear_constraints(&mut self) {
        self.linear_constraints.clear();
    }

    /// Insert lifetime (for testing)
    pub fn insert_lifetime(&mut self, name: String, lifetime: Lifetime) {
        self.active_lifetimes.insert(name, lifetime);
    }

    /// Move linear variable (mark as moved)
    pub fn move_linear_variable(&mut self, var: &str) -> TypeCheckResult<()> {
        if let Some(ownership) = self.linear_ownership.get_mut(var) {
            match ownership {
                LinearOwnership::Owned => {
                    *ownership = LinearOwnership::Moved;
                    self.used_variables.insert(var.to_string());
                    Ok(())
                }
                LinearOwnership::GpuOwned => {
                    *ownership = LinearOwnership::GpuMoved;
                    self.used_variables.insert(var.to_string());
                    Ok(())
                }
                LinearOwnership::Moved | LinearOwnership::GpuMoved => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "use after move".to_string(),
                        location: "unknown".to_string(),
                    })
                }
                LinearOwnership::Borrowed(_) | LinearOwnership::Shared(_) | LinearOwnership::MutableBorrow(_) => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "cannot move borrowed value".to_string(),
                        location: "unknown".to_string(),
                    })
                }
            }
        } else {
            Ok(()) // Non-linear variable
        }
    }

    /// Borrow linear variable
    pub fn borrow_linear_variable(&mut self, var: &str, lifetime: Lifetime, mutable: bool) -> TypeCheckResult<()> {
        if let Some(ownership) = self.linear_ownership.get(var).cloned() {
            match ownership {
                LinearOwnership::Owned | LinearOwnership::GpuOwned => {
                    // Can borrow owned values - change the original variable's ownership
                    let new_ownership = if mutable {
                        LinearOwnership::MutableBorrow(lifetime.clone())
                    } else {
                        LinearOwnership::Borrowed(lifetime.clone())
                    };
                    self.linear_ownership.insert(var.to_string(), new_ownership);
                    self.active_lifetimes.insert(lifetime.name.clone(), lifetime);
                    Ok(())
                }
                LinearOwnership::Borrowed(_) if !mutable => {
                    // Can create additional immutable borrows - change to shared
                    self.linear_ownership.insert(var.to_string(), LinearOwnership::Shared(lifetime.clone()));
                    self.active_lifetimes.insert(lifetime.name.clone(), lifetime);
                    Ok(())
                }
                LinearOwnership::Shared(_) if !mutable => {
                    // Can create additional shared borrows
                    self.active_lifetimes.insert(lifetime.name.clone(), lifetime);
                    Ok(())
                }
                LinearOwnership::Moved | LinearOwnership::GpuMoved => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "cannot borrow moved value".to_string(),
                        location: "unknown".to_string(),
                    })
                }
                LinearOwnership::MutableBorrow(_) => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "cannot borrow mutably borrowed value".to_string(),
                        location: "unknown".to_string(),
                    })
                }
                LinearOwnership::Borrowed(_) if mutable => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "cannot mutably borrow immutably borrowed value".to_string(),
                        location: "unknown".to_string(),
                    })
                }
                LinearOwnership::Shared(_) if mutable => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "cannot mutably borrow shared value".to_string(),
                        location: "unknown".to_string(),
                    })
                }
                // These cases are already handled above with conditions
                LinearOwnership::Borrowed(_) | LinearOwnership::Shared(_) => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "invalid borrow state".to_string(),
                        location: "unknown".to_string(),
                    })
                }
            }
        } else {
            Ok(()) // Non-linear variable
        }
    }

    /// Add linear constraint
    pub fn add_linear_constraint(&mut self, constraint: LinearConstraint) {
        self.linear_constraints.push(constraint);
    }

    /// Check single-use constraint
    pub fn check_single_use(&mut self, var: &str) -> TypeCheckResult<()> {
        if self.used_variables.contains(var) {
            Err(TypeCheckError::LinearTypeViolation {
                variable: var.to_string(),
                violation: "variable used more than once".to_string(),
                location: "unknown".to_string(),
            })
        } else {
            self.used_variables.insert(var.to_string());
            Ok(())
        }
    }

    /// Register allocation site
    pub fn register_allocation(&mut self, var: String, site: AllocationSite) {
        self.allocation_sites.insert(var, site);
    }

    /// Insert automatic deallocation
    fn insert_deallocation(&self, var: &str) {
        // In a real implementation, this would insert deallocation code
        // For now, we just track that deallocation should happen
        if let Some(site) = self.allocation_sites.get(var) {
            match site.allocation_type {
                AllocationType::Heap => {
                    // Insert free() call
                    println!("Inserting deallocation for heap variable: {}", var);
                }
                AllocationType::Gpu => {
                    // Insert GPU memory free
                    println!("Inserting GPU deallocation for variable: {}", var);
                }
                AllocationType::Stack => {
                    // Stack variables are automatically deallocated
                }
                AllocationType::Shared => {
                    // Shared memory requires special handling
                    println!("Inserting shared memory deallocation for variable: {}", var);
                }
            }
        }
    }

    /// Check for unused linear variables
    fn check_unused_linear_variables(&self) {
        for (var, ownership) in &self.linear_ownership {
            match ownership {
                LinearOwnership::Owned | LinearOwnership::GpuOwned => {
                    if !self.used_variables.contains(var) {
                        // In a real implementation, this might be a warning or error
                        println!("Warning: unused linear variable '{}' will be automatically deallocated", var);
                    }
                }
                _ => {}
            }
        }
    }

    /// Validate all linear constraints
    pub fn validate_linear_constraints(&self) -> TypeCheckResult<()> {
        for constraint in &self.linear_constraints {
            match constraint {
                LinearConstraint::SingleUse(var) => {
                    if !self.used_variables.contains(var) {
                        return Err(TypeCheckError::LinearTypeViolation {
                            variable: var.clone(),
                            violation: "variable must be used exactly once".to_string(),
                            location: "constraint validation".to_string(),
                        });
                    }
                }
                LinearConstraint::MustMove(var) => {
                    if let Some(ownership) = self.linear_ownership.get(var) {
                        if !matches!(ownership, LinearOwnership::Moved | LinearOwnership::GpuMoved) {
                            return Err(TypeCheckError::LinearTypeViolation {
                                variable: var.clone(),
                                violation: "variable must be moved before scope ends".to_string(),
                                location: "constraint validation".to_string(),
                            });
                        }
                    }
                }
                LinearConstraint::NoCopy(var) => {
                    // Check that variable hasn't been copied (simplified)
                    if self.used_variables.contains(var) {
                        // In a real implementation, we'd track copy operations
                    }
                }
                LinearConstraint::GpuResource(var) => {
                    if let Some(ownership) = self.linear_ownership.get(var) {
                        if !ownership.is_gpu_resource() {
                            return Err(TypeCheckError::LinearTypeViolation {
                                variable: var.clone(),
                                violation: "variable must be GPU resource".to_string(),
                                location: "constraint validation".to_string(),
                            });
                        }
                    }
                }
                LinearConstraint::OutlivesConstraint(lifetime1, lifetime2) => {
                    if !lifetime1.outlives(lifetime2) {
                        return Err(TypeCheckError::LinearTypeViolation {
                            variable: format!("lifetime '{}'", lifetime1.name),
                            violation: format!("must outlive lifetime '{}'", lifetime2.name),
                            location: "constraint validation".to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Type checker
pub struct TypeChecker {
    context: TypeCheckContext,
    inference: TypeInference,
}

impl TypeChecker {
    /// Create new type checker
    pub fn new(static_mode: bool) -> Self {
        TypeChecker {
            context: TypeCheckContext::new(static_mode),
            inference: TypeInference::new(),
        }
    }



    /// Type check AST
    pub fn check(&mut self, ast: &AST) -> TypeCheckResult<Type> {
        self.check_node(&ast.root)
    }

    /// Type check AST node
    fn check_node(&mut self, node: &ASTNode) -> TypeCheckResult<Type> {
        match node {
            ASTNode::Atom(atom) => self.check_atom(atom),
            ASTNode::List(elements) => self.check_list(elements),
            ASTNode::Graph { nodes, .. } => {
                // For graph nodes, check first node (simplified)
                if let Some(_first_ref) = nodes.first() {
                    // Would need proper AST node resolution
                    Ok(Type::dynamic())
                } else {
                    Ok(Type::primitive(PrimitiveType::Unit))
                }
            }
        }
    }

    /// Type check atomic value
    pub fn check_atom(&mut self, atom: &AtomValue) -> TypeCheckResult<Type> {
        match atom {
            AtomValue::Symbol(name) => {
                // First, check if the symbol exists and get its type
                let type_opt = self.context.lookup_type(name).cloned();
                
                if let Some(type_) = type_opt {
                    // Check linear ownership if applicable
                    if let Type::Linear { ownership, inner_type: _ } = &type_ {
                        match ownership {
                            LinearOwnership::Moved | LinearOwnership::GpuMoved => {
                                return Err(TypeCheckError::LinearTypeViolation {
                                    variable: name.clone(),
                                    violation: "use after move".to_string(),
                                    location: "symbol reference".to_string(),
                                });
                            }
                            LinearOwnership::Owned | LinearOwnership::GpuOwned => {
                                // Check single-use constraint for linear types
                                self.context.check_single_use(name)?;
                                // Mark as used but don't move yet (that happens on assignment/call)
                                self.context.used_variables.insert(name.clone());
                            }
                            LinearOwnership::Borrowed(_) | LinearOwnership::Shared(_) | LinearOwnership::MutableBorrow(_) => {
                                // Borrowed values can be used multiple times within their lifetime
                            }
                        }
                    }
                    Ok(type_)
                } else if self.context.static_mode {
                    Err(TypeCheckError::UndefinedSymbol {
                        name: name.clone(),
                        location: "symbol reference".to_string(),
                    })
                } else {
                    // In dynamic mode, unknown symbols have dynamic type
                    Ok(Type::dynamic())
                }
            }
            AtomValue::Number(value) => {
                // Infer numeric type
                if value.fract() == 0.0 {
                    if *value >= i32::MIN as f64 && *value <= i32::MAX as f64 {
                        Ok(Type::primitive(PrimitiveType::Int32))
                    } else {
                        Ok(Type::primitive(PrimitiveType::Int64))
                    }
                } else {
                    Ok(Type::primitive(PrimitiveType::Float64))
                }
            }
            AtomValue::String(_) => Ok(Type::primitive(PrimitiveType::String)),
            AtomValue::Boolean(_) => Ok(Type::primitive(PrimitiveType::Bool)),
            AtomValue::Nil => Ok(Type::primitive(PrimitiveType::Unit)),
            AtomValue::Token(_) => Ok(Type::dynamic()), // Tokens are dynamically typed
        }
    }

    /// Type check list (function application or special form)
    fn check_list(&mut self, elements: &[ASTNodeRef]) -> TypeCheckResult<Type> {
        if elements.is_empty() {
            return Ok(Type::primitive(PrimitiveType::Unit));
        }

        // For now, we'll handle this as a simplified function application
        // In a real implementation, we'd need to handle special forms like let, if, etc.
        
        // Check if first element is a known function
        // This is simplified - would need proper AST node resolution
        
        // For demonstration, let's handle some basic cases
        Ok(Type::dynamic())
    }

    /// Check function application
    fn check_function_application(
        &mut self,
        function_type: &Type,
        args: &[Type],
        location: &str,
    ) -> TypeCheckResult<Type> {
        match function_type {
            Type::Function { params, return_type, .. } => {
                if params.len() != args.len() {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: params.len(),
                        actual: args.len(),
                        function: "unknown".to_string(),
                        location: location.to_string(),
                    });
                }

                // Check parameter types
                for (param_type, arg_type) in params.iter().zip(args.iter()) {
                    if !self.is_assignable(arg_type, param_type) {
                        return Err(TypeCheckError::TypeMismatch {
                            expected: param_type.clone(),
                            actual: arg_type.clone(),
                            location: location.to_string(),
                        });
                    }
                }

                Ok((**return_type).clone())
            }
            Type::Dynamic => {
                // Dynamic function - return dynamic type
                Ok(Type::dynamic())
            }
            _ => Err(TypeCheckError::InvalidOperation {
                operation: "function call".to_string(),
                operand_types: vec![function_type.clone()],
                location: location.to_string(),
            }),
        }
    }

    /// Check if source type is assignable to target type
    pub fn is_assignable(&self, source: &Type, target: &Type) -> bool {
        match (source, target) {
            // Dynamic type is assignable to/from anything
            (Type::Dynamic, _) | (_, Type::Dynamic) => true,
            
            // Same types
            (s, t) if s == t => true,
            
            // Numeric conversions
            (Type::Primitive(s), Type::Primitive(t)) => self.is_numeric_convertible(s, t),
            
            // Function types (simplified)
            (Type::Function { params: p1, return_type: r1, .. },
             Type::Function { params: p2, return_type: r2, .. }) => {
                p1.len() == p2.len() &&
                p1.iter().zip(p2.iter()).all(|(a, b)| self.is_assignable(b, a)) && // Contravariant
                self.is_assignable(r1, r2) // Covariant
            }
            
            // Tensor types
            (Type::Tensor { element_type: e1, shape: s1 },
             Type::Tensor { element_type: e2, shape: s2 }) => {
                self.is_assignable(e1, e2) && s1.is_compatible_with(s2)
            }
            
            // Linear types
            (Type::Linear { inner_type: i1, ownership: o1 },
             Type::Linear { inner_type: i2, ownership: o2 }) => {
                self.is_assignable(i1, i2) && o1 == o2
            }
            
            _ => false,
        }
    }

    /// Check numeric convertibility
    fn is_numeric_convertible(&self, source: &PrimitiveType, target: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        match (source, target) {
            // Same type
            (s, t) if s == t => true,
            
            // Integer widening
            (Int8, Int16) | (Int8, Int32) | (Int8, Int64) => true,
            (Int16, Int32) | (Int16, Int64) => true,
            (Int32, Int64) => true,
            
            // Unsigned integer widening
            (UInt8, UInt16) | (UInt8, UInt32) | (UInt8, UInt64) => true,
            (UInt16, UInt32) | (UInt16, UInt64) => true,
            (UInt32, UInt64) => true,
            
            // Float widening
            (Float32, Float64) => true,
            
            // Integer to float (with potential precision loss)
            (Int8, Float32) | (Int8, Float64) => true,
            (Int16, Float32) | (Int16, Float64) => true,
            (Int32, Float64) => true,
            (UInt8, Float32) | (UInt8, Float64) => true,
            (UInt16, Float32) | (UInt16, Float64) => true,
            (UInt32, Float64) => true,
            
            _ => false,
        }
    }

    /// Check tensor operation
    fn check_tensor_operation(
        &mut self,
        operation: &str,
        operands: &[Type],
        location: &str,
    ) -> TypeCheckResult<Type> {
        match operation {
            "add" | "sub" | "mul" | "div" => {
                if operands.len() != 2 {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: 2,
                        actual: operands.len(),
                        function: operation.to_string(),
                        location: location.to_string(),
                    });
                }

                match (&operands[0], &operands[1]) {
                    (Type::Tensor { element_type: e1, shape: s1 },
                     Type::Tensor { element_type: e2, shape: s2 }) => {
                        // Check element type compatibility
                        if !self.is_assignable(e1, e2) && !self.is_assignable(e2, e1) {
                            return Err(TypeCheckError::TypeMismatch {
                                expected: e1.as_ref().clone(),
                                actual: e2.as_ref().clone(),
                                location: location.to_string(),
                            });
                        }

                        // Check shape compatibility (broadcasting)
                        if !s1.is_broadcastable_with(s2) {
                            return Err(TypeCheckError::ShapeMismatch {
                                expected: s1.clone(),
                                actual: s2.clone(),
                                operation: operation.to_string(),
                                location: location.to_string(),
                            });
                        }

                        // Result has promoted element type and broadcast shape
                        let result_element_type = if self.is_assignable(e1, e2) { e2.as_ref().clone() } else { e1.as_ref().clone() };
                        let _result_shape = self.broadcast_shapes(s1, s2)?;
                        
                        Ok(Type::tensor(result_element_type, vec![])) // Simplified shape
                    }
                    _ => Err(TypeCheckError::InvalidOperation {
                        operation: operation.to_string(),
                        operand_types: operands.to_vec(),
                        location: location.to_string(),
                    }),
                }
            }
            "matmul" => {
                if operands.len() != 2 {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: 2,
                        actual: operands.len(),
                        function: operation.to_string(),
                        location: location.to_string(),
                    });
                }

                match (&operands[0], &operands[1]) {
                    (Type::Tensor { element_type: e1, shape: s1 },
                     Type::Tensor { element_type: e2, shape: s2 }) => {
                        // Check element type compatibility
                        if !self.is_assignable(e1, e2) && !self.is_assignable(e2, e1) {
                            return Err(TypeCheckError::TypeMismatch {
                                expected: e1.as_ref().clone(),
                                actual: e2.as_ref().clone(),
                                location: location.to_string(),
                            });
                        }

                        // Check matrix multiplication shape compatibility
                        self.check_matmul_shapes(s1, s2, location)?;

                        // Result type
                        let result_element_type = if self.is_assignable(e1, e2) { e2.as_ref().clone() } else { e1.as_ref().clone() };
                        Ok(Type::tensor(result_element_type, vec![])) // Simplified
                    }
                    _ => Err(TypeCheckError::InvalidOperation {
                        operation: operation.to_string(),
                        operand_types: operands.to_vec(),
                        location: location.to_string(),
                    }),
                }
            }
            _ => Err(TypeCheckError::InvalidOperation {
                operation: operation.to_string(),
                operand_types: operands.to_vec(),
                location: location.to_string(),
            }),
        }
    }

    /// Broadcast two shapes
    pub fn broadcast_shapes(&self, shape1: &Shape, shape2: &Shape) -> TypeCheckResult<Shape> {
        match (shape1, shape2) {
            (Shape::Unknown, _) => Ok(shape2.clone()),
            (_, Shape::Unknown) => Ok(shape1.clone()),
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) => {
                let max_len = dims1.len().max(dims2.len());
                let mut result_dims = Vec::with_capacity(max_len);
                
                for i in 0..max_len {
                    let dim1 = dims1.get(dims1.len().saturating_sub(max_len - i)).copied().unwrap_or(1);
                    let dim2 = dims2.get(dims2.len().saturating_sub(max_len - i)).copied().unwrap_or(1);
                    
                    if dim1 == 1 {
                        result_dims.push(dim2);
                    } else if dim2 == 1 {
                        result_dims.push(dim1);
                    } else if dim1 == dim2 {
                        result_dims.push(dim1);
                    } else {
                        return Err(TypeCheckError::ShapeMismatch {
                            expected: shape1.clone(),
                            actual: shape2.clone(),
                            operation: "broadcast".to_string(),
                            location: "shape broadcasting".to_string(),
                        });
                    }
                }
                
                Ok(Shape::concrete(result_dims))
            }
            _ => Ok(Shape::unknown()), // Simplified for symbolic shapes
        }
    }

    /// Check matrix multiplication shape compatibility
    fn check_matmul_shapes(&self, shape1: &Shape, shape2: &Shape, location: &str) -> TypeCheckResult<()> {
        match (shape1, shape2) {
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) => {
                if dims1.len() < 2 || dims2.len() < 2 {
                    return Err(TypeCheckError::ShapeMismatch {
                        expected: shape1.clone(),
                        actual: shape2.clone(),
                        operation: "matrix multiplication requires at least 2D tensors".to_string(),
                        location: location.to_string(),
                    });
                }
                
                let inner1 = dims1[dims1.len() - 1];
                let inner2 = dims2[dims2.len() - 2];
                
                if inner1 != inner2 {
                    return Err(TypeCheckError::ShapeMismatch {
                        expected: shape1.clone(),
                        actual: shape2.clone(),
                        operation: "matrix multiplication inner dimensions must match".to_string(),
                        location: location.to_string(),
                    });
                }
                
                Ok(())
            }
            (Shape::Symbolic(dims1), Shape::Symbolic(dims2)) => {
                if dims1.len() < 2 || dims2.len() < 2 {
                    return Err(TypeCheckError::ShapeMismatch {
                        expected: shape1.clone(),
                        actual: shape2.clone(),
                        operation: "matrix multiplication requires at least 2D tensors".to_string(),
                        location: location.to_string(),
                    });
                }
                
                // Check if inner dimensions can be unified
                let inner1 = &dims1[dims1.len() - 1];
                let inner2 = &dims2[dims2.len() - 2];
                
                if !shape1.dims_compatible(inner1, inner2) {
                    return Err(TypeCheckError::ShapeMismatch {
                        expected: shape1.clone(),
                        actual: shape2.clone(),
                        operation: "matrix multiplication inner dimensions must be compatible".to_string(),
                        location: location.to_string(),
                    });
                }
                
                Ok(())
            }
            _ => Ok(()), // Mixed or computed shapes - would need constraint solving
        }
    }

    /// Infer result shape for tensor operation
    pub fn infer_tensor_result_shape(
        &self,
        operation: &str,
        operand_shapes: &[Shape],
        location: &str,
    ) -> TypeCheckResult<Shape> {
        match operation {
            "add" | "sub" | "mul" | "div" => {
                if operand_shapes.len() != 2 {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: 2,
                        actual: operand_shapes.len(),
                        function: operation.to_string(),
                        location: location.to_string(),
                    });
                }
                
                self.broadcast_shapes(&operand_shapes[0], &operand_shapes[1])
            }
            "matmul" => {
                if operand_shapes.len() != 2 {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: 2,
                        actual: operand_shapes.len(),
                        function: operation.to_string(),
                        location: location.to_string(),
                    });
                }
                
                self.check_matmul_shapes(&operand_shapes[0], &operand_shapes[1], location)?;
                
                if let Some(result_shape) = operand_shapes[0].matmul_result_shape(&operand_shapes[1]) {
                    Ok(result_shape)
                } else {
                    Err(TypeCheckError::ShapeMismatch {
                        expected: operand_shapes[0].clone(),
                        actual: operand_shapes[1].clone(),
                        operation: "matrix multiplication shape inference failed".to_string(),
                        location: location.to_string(),
                    })
                }
            }
            "transpose" => {
                if operand_shapes.len() != 1 {
                    return Err(TypeCheckError::ArityMismatch {
                        expected: 1,
                        actual: operand_shapes.len(),
                        function: operation.to_string(),
                        location: location.to_string(),
                    });
                }
                
                match &operand_shapes[0] {
                    Shape::Concrete(dims) => {
                        if dims.len() < 2 {
                            return Err(TypeCheckError::ShapeMismatch {
                                expected: operand_shapes[0].clone(),
                                actual: operand_shapes[0].clone(),
                                operation: "transpose requires at least 2D tensor".to_string(),
                                location: location.to_string(),
                            });
                        }
                        
                        let mut transposed_dims = dims.clone();
                        let len = transposed_dims.len();
                        transposed_dims.swap(len - 2, len - 1);
                        Ok(Shape::Concrete(transposed_dims))
                    }
                    Shape::Symbolic(dims) => {
                        if dims.len() < 2 {
                            return Err(TypeCheckError::ShapeMismatch {
                                expected: operand_shapes[0].clone(),
                                actual: operand_shapes[0].clone(),
                                operation: "transpose requires at least 2D tensor".to_string(),
                                location: location.to_string(),
                            });
                        }
                        
                        let mut transposed_dims = dims.clone();
                        let len = transposed_dims.len();
                        transposed_dims.swap(len - 2, len - 1);
                        Ok(Shape::Symbolic(transposed_dims))
                    }
                    _ => Ok(operand_shapes[0].clone()), // Unknown or computed shapes
                }
            }
            "reshape" => {
                // For reshape, we'd need the target shape as a parameter
                // This is simplified - in practice, reshape would take shape as argument
                Ok(Shape::Unknown)
            }
            _ => Err(TypeCheckError::InvalidOperation {
                operation: operation.to_string(),
                operand_types: vec![], // Would need to convert shapes to types
                location: location.to_string(),
            }),
        }
    }

    /// Check dependent type constraints
    pub fn check_dependent_constraints(
        &self,
        constraints: &[crate::compiler::types::DependentConstraint],
        _location: &str,
    ) -> TypeCheckResult<()> {
        use crate::compiler::types::{DependentConstraint, ShapeConstraintKind};
        
        for constraint in constraints {
            match constraint {
                DependentConstraint::ShapeConstraint(_var, kind) => {
                    match kind {
                        ShapeConstraintKind::Positive => {
                            // Would need to check that the shape variable is positive
                            // This requires constraint solving which is simplified here
                        }
                        ShapeConstraintKind::Equal(_) |
                        ShapeConstraintKind::GreaterThan(_) |
                        ShapeConstraintKind::LessThan(_) => {
                            // Would need constraint solving
                        }
                    }
                }
                DependentConstraint::IntConstraint(_, _) => {
                    // Would need constraint solving for integer constraints
                }
                DependentConstraint::Relationship(_, _, _) => {
                    // Would need constraint solving for parameter relationships
                }
            }
        }
        
        Ok(())
    }

    /// Validate dependent type
    pub fn validate_dependent_type(
        &mut self,
        base_type: &Type,
        params: &[crate::compiler::types::DependentParam],
        constraints: &[crate::compiler::types::DependentConstraint],
        location: &str,
    ) -> TypeCheckResult<Type> {
        // Check that the base type is valid
        let _validated_base = self.check_type_validity(base_type, location)?;
        
        // Check dependent constraints
        self.check_dependent_constraints(constraints, location)?;
        
        // For now, return the base type with dependent information
        // In a full implementation, we'd need constraint solving
        Ok(Type::Dependent {
            base_type: Box::new(base_type.clone()),
            params: params.to_vec(),
            constraints: constraints.to_vec(),
        })
    }

    /// Check if a type is valid (helper method)
    fn check_type_validity(&mut self, type_: &Type, location: &str) -> TypeCheckResult<Type> {
        match type_ {
            Type::Dynamic | Type::Primitive(_) => Ok(type_.clone()),
            Type::Tensor { element_type, shape } => {
                let validated_element = self.check_type_validity(element_type, location)?;
                // Validate shape constraints if any
                Ok(Type::Tensor {
                    element_type: Box::new(validated_element),
                    shape: shape.clone(),
                })
            }
            Type::Function { params, return_type, is_differentiable } => {
                let mut validated_params = Vec::new();
                for param in params {
                    validated_params.push(self.check_type_validity(param, location)?);
                }
                let validated_return = self.check_type_validity(return_type, location)?;
                
                Ok(Type::Function {
                    params: validated_params,
                    return_type: Box::new(validated_return),
                    is_differentiable: *is_differentiable,
                })
            }
            Type::Dependent { base_type, params, constraints } => {
                self.validate_dependent_type(base_type, params, constraints, location)
            }
            _ => Ok(type_.clone()), // Other types are assumed valid for now
        }
    }

    /// Parse type annotation from AST node
    pub fn parse_type_annotation(&mut self, node: &ASTNode) -> TypeCheckResult<Type> {
        match node {
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                match name.as_str() {
                    "i8" => Ok(Type::primitive(PrimitiveType::Int8)),
                    "i16" => Ok(Type::primitive(PrimitiveType::Int16)),
                    "i32" => Ok(Type::primitive(PrimitiveType::Int32)),
                    "i64" => Ok(Type::primitive(PrimitiveType::Int64)),
                    "u8" => Ok(Type::primitive(PrimitiveType::UInt8)),
                    "u16" => Ok(Type::primitive(PrimitiveType::UInt16)),
                    "u32" => Ok(Type::primitive(PrimitiveType::UInt32)),
                    "u64" => Ok(Type::primitive(PrimitiveType::UInt64)),
                    "f32" => Ok(Type::primitive(PrimitiveType::Float32)),
                    "f64" => Ok(Type::primitive(PrimitiveType::Float64)),
                    "bool" => Ok(Type::primitive(PrimitiveType::Bool)),
                    "string" => Ok(Type::primitive(PrimitiveType::String)),
                    "char" => Ok(Type::primitive(PrimitiveType::Char)),
                    "dyn" => Ok(Type::dynamic()),
                    _ => Err(TypeCheckError::InvalidTypeAnnotation {
                        annotation: name.clone(),
                        error: "unknown type".to_string(),
                        location: "type annotation".to_string(),
                    }),
                }
            }
            ASTNode::List(elements) => {
                if elements.is_empty() {
                    return Ok(Type::primitive(PrimitiveType::Unit));
                }

                // Parse complex type annotations like Tensor<f32, [2, 3]>
                // This is simplified - would need proper parsing
                Ok(Type::dynamic())
            }
            _ => Err(TypeCheckError::InvalidTypeAnnotation {
                annotation: format!("{:?}", node),
                error: "invalid type annotation syntax".to_string(),
                location: "type annotation".to_string(),
            }),
        }
    }

    /// Get type checking context
    pub fn context(&self) -> &TypeCheckContext {
        &self.context
    }

    /// Get mutable type checking context
    pub fn context_mut(&mut self) -> &mut TypeCheckContext {
        &mut self.context
    }

    /// Check linear type assignment
    pub fn check_linear_assignment(&mut self, var: &str, value_type: &Type, location: &str) -> TypeCheckResult<()> {
        if let Type::Linear { ownership, .. } = value_type {
            match ownership {
                LinearOwnership::Owned => {
                    // Register as owned linear variable
                    self.context.track_linear_ownership(var.to_string(), LinearOwnership::Owned);
                    self.context.register_allocation(
                        var.to_string(),
                        AllocationSite::new(location.to_string(), AllocationType::Heap)
                    );
                }
                LinearOwnership::GpuOwned => {
                    // Register as GPU-owned linear variable
                    self.context.track_linear_ownership(var.to_string(), LinearOwnership::GpuOwned);
                    self.context.register_allocation(
                        var.to_string(),
                        AllocationSite::new(location.to_string(), AllocationType::Gpu)
                    );
                    self.context.add_linear_constraint(LinearConstraint::GpuResource(var.to_string()));
                }
                LinearOwnership::Borrowed(lifetime) => {
                    // Register borrowed reference
                    self.context.track_linear_ownership(var.to_string(), ownership.clone());
                    self.context.active_lifetimes.insert(lifetime.name.clone(), lifetime.clone());
                }
                _ => {
                    return Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "invalid ownership for assignment".to_string(),
                        location: location.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Analyze lifetime requirements
    pub fn analyze_lifetimes(&mut self, node: &ASTNode) -> TypeCheckResult<Vec<Lifetime>> {
        // Simplified lifetime analysis
        // In a real implementation, this would perform comprehensive lifetime inference
        let mut lifetimes = Vec::new();
        
        match node {
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                if let Some(type_) = self.context.lookup_type(name) {
                    if let Some(lifetime) = type_.get_lifetime() {
                        lifetimes.push(lifetime.clone());
                    }
                }
            }
            ASTNode::List(elements) => {
                for _element in elements {
                    // Would need to resolve ASTNodeRef and recurse
                    // For now, create a placeholder lifetime
                    let lifetime = Lifetime::new(
                        format!("'scope_{}", self.context.scope_level),
                        self.context.scope_level
                    );
                    lifetimes.push(lifetime);
                    break; // Simplified
                }
            }
            _ => {}
        }
        
        Ok(lifetimes)
    }

    /// Insert automatic memory deallocation
    pub fn insert_automatic_deallocation(&mut self) -> TypeCheckResult<()> {
        // This would be called at scope exit to insert deallocation code
        // For now, we validate that all linear constraints are satisfied
        self.context.validate_linear_constraints()
    }

    /// Check GPU memory safety
    pub fn check_gpu_memory_safety(&self, var: &str) -> TypeCheckResult<()> {
        if let Some(ownership) = self.context.check_linear_ownership(var) {
            match ownership {
                LinearOwnership::GpuOwned => {
                    // GPU resource is properly owned
                    Ok(())
                }
                LinearOwnership::GpuMoved => {
                    Err(TypeCheckError::LinearTypeViolation {
                        variable: var.to_string(),
                        violation: "GPU resource has been moved".to_string(),
                        location: "GPU memory safety check".to_string(),
                    })
                }
                _ => {
                    // Not a GPU resource
                    Ok(())
                }
            }
        } else {
            Ok(())
        }
    }
}

impl From<InferenceError> for TypeCheckError {
    fn from(error: InferenceError) -> Self {
        TypeCheckError::InferenceError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ast::ASTNode;

    #[test]
    fn test_type_checker_creation() {
        let checker = TypeChecker::new(true);
        assert!(checker.context.static_mode);
        
        let dynamic_checker = TypeChecker::new(false);
        assert!(!dynamic_checker.context.static_mode);
    }

    #[test]
    fn test_check_atom_types() {
        let mut checker = TypeChecker::new(true);
        
        let int_type = checker.check_atom(&AtomValue::Number(42.0)).unwrap();
        assert_eq!(int_type, Type::primitive(PrimitiveType::Int32));
        
        let float_type = checker.check_atom(&AtomValue::Number(3.14)).unwrap();
        assert_eq!(float_type, Type::primitive(PrimitiveType::Float64));
        
        let string_type = checker.check_atom(&AtomValue::String("hello".to_string())).unwrap();
        assert_eq!(string_type, Type::primitive(PrimitiveType::String));
        
        let bool_type = checker.check_atom(&AtomValue::Boolean(true)).unwrap();
        assert_eq!(bool_type, Type::primitive(PrimitiveType::Bool));
    }

    #[test]
    fn test_undefined_symbol_static_mode() {
        let mut checker = TypeChecker::new(true);
        
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
    fn test_undefined_symbol_dynamic_mode() {
        let mut checker = TypeChecker::new(false);
        
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
        
        // Dynamic type
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
    fn test_shape_broadcasting() {
        let checker = TypeChecker::new(true);
        
        let shape1 = Shape::concrete(vec![1, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let result = checker.broadcast_shapes(&shape1, &shape2).unwrap();
        
        if let Shape::Concrete(dims) = result {
            assert_eq!(dims, vec![2, 3]);
        } else {
            panic!("Expected concrete shape");
        }
    }

    #[test]
    fn test_parse_primitive_type_annotations() {
        let mut checker = TypeChecker::new(true);
        
        let int_node = ASTNode::symbol("i32".to_string());
        let int_type = checker.parse_type_annotation(&int_node).unwrap();
        assert_eq!(int_type, Type::primitive(PrimitiveType::Int32));
        
        let float_node = ASTNode::symbol("f64".to_string());
        let float_type = checker.parse_type_annotation(&float_node).unwrap();
        assert_eq!(float_type, Type::primitive(PrimitiveType::Float64));
        
        let dyn_node = ASTNode::symbol("dyn".to_string());
        let dyn_type = checker.parse_type_annotation(&dyn_node).unwrap();
        assert_eq!(dyn_type, Type::dynamic());
    }

    #[test]
    fn test_linear_ownership_tracking() {
        let mut context = TypeCheckContext::new(true);
        
        context.track_linear_ownership("x".to_string(), LinearOwnership::Owned);
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Owned));
        
        context.move_linear_variable("x").unwrap();
        assert_eq!(context.check_linear_ownership("x"), Some(&LinearOwnership::Moved));
        
        let result = context.move_linear_variable("x");
        assert!(result.is_err());
    }
}

impl fmt::Display for TypeCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeCheckError::TypeMismatch { expected, actual, location } => {
                write!(f, "Type mismatch at {}: expected {}, found {}", location, expected, actual)
            }
            TypeCheckError::UndefinedSymbol { name, location } => {
                write!(f, "Undefined symbol '{}' at {}", name, location)
            }
            TypeCheckError::InvalidOperation { operation, operand_types, location } => {
                write!(f, "Invalid operation '{}' at {} with operands: {:?}", operation, location, operand_types)
            }
            TypeCheckError::ShapeMismatch { expected, actual, operation, location } => {
                write!(f, "Shape mismatch in {} at {}: expected {:?}, found {:?}", operation, location, expected, actual)
            }
            TypeCheckError::LinearTypeViolation { variable, violation, location } => {
                write!(f, "Linear type violation for '{}' at {}: {}", variable, location, violation)
            }
            TypeCheckError::ArityMismatch { expected, actual, function, location } => {
                write!(f, "Arity mismatch for '{}' at {}: expected {} arguments, found {}", function, location, expected, actual)
            }
            TypeCheckError::InvalidTypeAnnotation { annotation, error, location } => {
                write!(f, "Invalid type annotation '{}' at {}: {}", annotation, location, error)
            }
            TypeCheckError::InferenceError(err) => {
                write!(f, "Type inference error: {:?}", err)
            }
        }
    }
}

impl std::error::Error for TypeCheckError {}