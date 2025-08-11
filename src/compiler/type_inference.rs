// Aether Type Inference Engine
// Constraint-based type inference for gradual typing

use std::collections::HashMap;
use crate::compiler::types::{
    Type, TypeVar, TypeConstraint, PrimitiveType, Shape,
    LinearOwnership
};
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};

/// Type inference error
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    /// Type mismatch between expected and actual types
    TypeMismatch {
        expected: Type,
        actual: Type,
        location: String,
    },
    /// Unresolved type variable
    UnresolvedTypeVar(u32),
    /// Conflicting constraints on type variable
    ConflictingConstraints {
        var_id: u32,
        constraint1: TypeConstraint,
        constraint2: TypeConstraint,
    },
    /// Shape mismatch in tensor operations
    ShapeMismatch {
        expected: Shape,
        actual: Shape,
        operation: String,
    },
    /// Undefined variable or function
    UndefinedSymbol(String),
    /// Invalid operation for type
    InvalidOperation {
        operation: String,
        type_: Type,
    },
    /// Linear type violation (use after move, etc.)
    LinearTypeViolation {
        variable: String,
        violation: String,
    },
}

/// Type inference result
pub type InferenceResult<T> = Result<T, InferenceError>;

/// Type environment for variable bindings
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    bindings: HashMap<String, Type>,
    parent: Option<Box<TypeEnvironment>>,
}

impl TypeEnvironment {
    /// Create new empty environment
    pub fn new() -> Self {
        TypeEnvironment {
            bindings: HashMap::new(),
            parent: None,
        }
    }

    /// Create child environment
    pub fn child(&self) -> Self {
        TypeEnvironment {
            bindings: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Bind variable to type
    pub fn bind(&mut self, name: String, type_: Type) {
        self.bindings.insert(name, type_);
    }

    /// Look up variable type
    pub fn lookup(&self, name: &str) -> Option<&Type> {
        self.bindings.get(name).or_else(|| {
            self.parent.as_ref().and_then(|parent| parent.lookup(name))
        })
    }

    /// Check if variable is bound in current scope
    pub fn is_bound_locally(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
}

/// Constraint solver for type inference
#[derive(Debug)]
pub struct ConstraintSolver {
    constraints: Vec<Constraint>,
    type_vars: HashMap<u32, TypeVar>,
    substitutions: HashMap<u32, Type>,
    next_var_id: u32,
}

/// Internal constraint representation
#[derive(Debug, Clone)]
enum Constraint {
    /// Type equality constraint
    Equal(Type, Type),
    /// Subtype constraint
    Subtype(Type, Type),
    /// Shape compatibility constraint
    ShapeCompatible(Shape, Shape, String),
    /// Linear ownership constraint
    LinearOwnership(String, LinearOwnership),
}

impl ConstraintSolver {
    /// Create new constraint solver
    pub fn new() -> Self {
        ConstraintSolver {
            constraints: Vec::new(),
            type_vars: HashMap::new(),
            substitutions: HashMap::new(),
            next_var_id: 0,
        }
    }

    /// Generate fresh type variable
    pub fn fresh_type_var(&mut self, name: Option<String>) -> Type {
        let id = self.next_var_id;
        self.next_var_id += 1;
        
        let type_var = TypeVar {
            id,
            name: name.clone(),
            constraints: Vec::new(),
        };
        
        self.type_vars.insert(id, type_var);
        Type::Variable(TypeVar { id, name, constraints: Vec::new() })
    }

    /// Add equality constraint
    pub fn add_equal_constraint(&mut self, type1: Type, type2: Type) {
        self.constraints.push(Constraint::Equal(type1, type2));
    }

    /// Add subtype constraint
    pub fn add_subtype_constraint(&mut self, subtype: Type, supertype: Type) {
        self.constraints.push(Constraint::Subtype(subtype, supertype));
    }

    /// Add shape compatibility constraint
    pub fn add_shape_constraint(&mut self, shape1: Shape, shape2: Shape, operation: String) {
        self.constraints.push(Constraint::ShapeCompatible(shape1, shape2, operation));
    }

    /// Solve all constraints
    pub fn solve(&mut self) -> InferenceResult<()> {
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for constraint in self.constraints.clone() {
                match constraint {
                    Constraint::Equal(type1, type2) => {
                        if self.unify(type1, type2)? {
                            changed = true;
                        }
                    }
                    Constraint::Subtype(sub, sup) => {
                        if self.check_subtype(&sub, &sup)? {
                            changed = true;
                        }
                    }
                    Constraint::ShapeCompatible(shape1, shape2, op) => {
                        self.check_shape_compatibility(&shape1, &shape2, &op)?;
                    }
                    Constraint::LinearOwnership(_, _) => {
                        // Linear ownership constraints are checked separately
                    }
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            // Could indicate infinite loop in constraint solving
            // For now, we'll continue - in practice, we'd want better cycle detection
        }

        Ok(())
    }

    /// Unify two types
    pub fn unify(&mut self, type1: Type, type2: Type) -> InferenceResult<bool> {
        let type1 = self.apply_substitution(type1);
        let type2 = self.apply_substitution(type2);

        match (type1, type2) {
            // Same types unify trivially
            (t1, t2) if t1 == t2 => Ok(false),

            // Dynamic type unifies with anything
            (Type::Dynamic, _) | (_, Type::Dynamic) => Ok(false),

            // Type variable unification
            (Type::Variable(var), t) | (t, Type::Variable(var)) => {
                if let Type::Variable(other_var) = &t {
                    if var.id == other_var.id {
                        return Ok(false);
                    }
                }
                
                if self.occurs_check(var.id, &t) {
                    return Err(InferenceError::ConflictingConstraints {
                        var_id: var.id,
                        constraint1: TypeConstraint::Equal(Type::Variable(var.clone())),
                        constraint2: TypeConstraint::Equal(t),
                    });
                }
                
                self.substitutions.insert(var.id, t);
                Ok(true)
            }

            // Function type unification
            (Type::Function { params: p1, return_type: r1, is_differentiable: d1 },
             Type::Function { params: p2, return_type: r2, is_differentiable: d2 }) => {
                if p1.len() != p2.len() || d1 != d2 {
                    return Err(InferenceError::TypeMismatch {
                        expected: Type::Function { params: p1, return_type: r1, is_differentiable: d1 },
                        actual: Type::Function { params: p2, return_type: r2, is_differentiable: d2 },
                        location: "function unification".to_string(),
                    });
                }

                let mut changed = false;
                for (param1, param2) in p1.into_iter().zip(p2.into_iter()) {
                    if self.unify(param1, param2)? {
                        changed = true;
                    }
                }

                if self.unify(*r1, *r2)? {
                    changed = true;
                }

                Ok(changed)
            }

            // Tensor type unification
            (Type::Tensor { element_type: e1, shape: s1 },
             Type::Tensor { element_type: e2, shape: s2 }) => {
                let mut changed = false;
                
                if self.unify(*e1, *e2)? {
                    changed = true;
                }

                if !s1.is_compatible_with(&s2) {
                    return Err(InferenceError::ShapeMismatch {
                        expected: s1,
                        actual: s2,
                        operation: "tensor unification".to_string(),
                    });
                }

                Ok(changed)
            }

            // Linear type unification
            (Type::Linear { inner_type: i1, ownership: o1 },
             Type::Linear { inner_type: i2, ownership: o2 }) => {
                if o1 != o2 {
                    return Err(InferenceError::LinearTypeViolation {
                        variable: "unknown".to_string(),
                        violation: format!("ownership mismatch: {:?} vs {:?}", o1, o2),
                    });
                }

                self.unify(*i1, *i2)
            }

            // Dependent type unification
            (Type::Dependent { base_type: b1, params: p1, constraints: c1 },
             Type::Dependent { base_type: b2, params: p2, constraints: c2 }) => {
                // Check if parameters are compatible
                if p1.len() != p2.len() {
                    return Err(InferenceError::TypeMismatch {
                        expected: Type::Dependent { base_type: b1.clone(), params: p1.clone(), constraints: c1.clone() },
                        actual: Type::Dependent { base_type: b2.clone(), params: p2.clone(), constraints: c2.clone() },
                        location: "dependent type parameter count mismatch".to_string(),
                    });
                }

                // Check parameter compatibility (simplified)
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    if !self.params_compatible(param1, param2) {
                        return Err(InferenceError::TypeMismatch {
                            expected: Type::Dependent { base_type: b1.clone(), params: p1.clone(), constraints: c1.clone() },
                            actual: Type::Dependent { base_type: b2.clone(), params: p2.clone(), constraints: c2.clone() },
                            location: "dependent type parameter mismatch".to_string(),
                        });
                    }
                }

                // Unify base types
                let changed = self.unify(*b1, *b2)?;

                // Merge constraints (simplified)
                // In a full implementation, we'd need constraint solving
                
                Ok(changed)
            }

            // Other type mismatches
            (t1, t2) => Err(InferenceError::TypeMismatch {
                expected: t1,
                actual: t2,
                location: "type unification".to_string(),
            }),
        }
    }

    /// Check if type variable occurs in type (prevents infinite types)
    fn occurs_check(&self, var_id: u32, type_: &Type) -> bool {
        match type_ {
            Type::Variable(var) => var.id == var_id,
            Type::Function { params, return_type, .. } => {
                params.iter().any(|p| self.occurs_check(var_id, p)) ||
                self.occurs_check(var_id, return_type)
            }
            Type::Tensor { element_type, .. } => self.occurs_check(var_id, element_type),
            Type::Linear { inner_type, .. } => self.occurs_check(var_id, inner_type),
            Type::Probabilistic { inner_type, .. } => self.occurs_check(var_id, inner_type),
            Type::Struct { fields, .. } => fields.values().any(|t| self.occurs_check(var_id, t)),
            Type::Union { variants, .. } => variants.iter().any(|t| self.occurs_check(var_id, t)),
            Type::Dependent { base_type, .. } => self.occurs_check(var_id, base_type),
            _ => false,
        }
    }

    /// Apply current substitutions to type
    fn apply_substitution(&self, type_: Type) -> Type {
        match type_ {
            Type::Variable(var) => {
                if let Some(substitution) = self.substitutions.get(&var.id) {
                    self.apply_substitution(substitution.clone())
                } else {
                    Type::Variable(var)
                }
            }
            Type::Function { params, return_type, is_differentiable } => {
                Type::Function {
                    params: params.into_iter().map(|p| self.apply_substitution(p)).collect(),
                    return_type: Box::new(self.apply_substitution(*return_type)),
                    is_differentiable,
                }
            }
            Type::Tensor { element_type, shape } => {
                Type::Tensor {
                    element_type: Box::new(self.apply_substitution(*element_type)),
                    shape,
                }
            }
            Type::Linear { inner_type, ownership } => {
                Type::Linear {
                    inner_type: Box::new(self.apply_substitution(*inner_type)),
                    ownership,
                }
            }
            Type::Probabilistic { distribution, inner_type } => {
                Type::Probabilistic {
                    distribution,
                    inner_type: Box::new(self.apply_substitution(*inner_type)),
                }
            }
            Type::Dependent { base_type, params, constraints } => {
                Type::Dependent {
                    base_type: Box::new(self.apply_substitution(*base_type)),
                    params,
                    constraints,
                }
            }
            other => other,
        }
    }

    /// Check subtype relationship
    fn check_subtype(&self, subtype: &Type, supertype: &Type) -> InferenceResult<bool> {
        match (subtype, supertype) {
            // Dynamic is subtype of everything and everything is subtype of dynamic
            (Type::Dynamic, _) | (_, Type::Dynamic) => Ok(false),
            
            // Same types
            (t1, t2) if t1 == t2 => Ok(false),
            
            // Numeric subtyping (simplified)
            (Type::Primitive(p1), Type::Primitive(p2)) => {
                Ok(self.is_numeric_subtype(p1, p2))
            }
            
            // Function subtyping (contravariant in parameters, covariant in return)
            (Type::Function { params: p1, return_type: r1, .. },
             Type::Function { params: p2, return_type: r2, .. }) => {
                if p1.len() != p2.len() {
                    return Ok(false);
                }
                
                // Parameters are contravariant
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    if !self.check_subtype(param2, param1)? {
                        return Ok(false);
                    }
                }
                
                // Return type is covariant
                self.check_subtype(r1, r2)
            }
            
            _ => Ok(false),
        }
    }

    /// Check numeric subtype relationship
    fn is_numeric_subtype(&self, sub: &PrimitiveType, sup: &PrimitiveType) -> bool {
        use PrimitiveType::*;
        match (sub, sup) {
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
            
            // Integer to float conversion
            (Int8, Float32) | (Int8, Float64) => true,
            (Int16, Float32) | (Int16, Float64) => true,
            (Int32, Float64) => true,
            
            _ => false,
        }
    }

    /// Check shape compatibility
    fn check_shape_compatibility(&self, shape1: &Shape, shape2: &Shape, operation: &str) -> InferenceResult<()> {
        match operation {
            "add" | "sub" | "mul" | "div" => {
                if !shape1.is_compatible_with(shape2) && !shape1.is_broadcastable_with(shape2) {
                    return Err(InferenceError::ShapeMismatch {
                        expected: shape1.clone(),
                        actual: shape2.clone(),
                        operation: operation.to_string(),
                    });
                }
            }
            "matmul" => {
                // Matrix multiplication requires inner dimensions to match
                if let (Some(dims1), Some(dims2)) = (shape1.as_concrete(), shape2.as_concrete()) {
                    if dims1.len() < 2 || dims2.len() < 2 {
                        return Err(InferenceError::ShapeMismatch {
                            expected: shape1.clone(),
                            actual: shape2.clone(),
                            operation: "matrix multiplication (requires at least 2D tensors)".to_string(),
                        });
                    }
                    
                    let inner1 = dims1[dims1.len() - 1];
                    let inner2 = dims2[dims2.len() - 2];
                    
                    if inner1 != inner2 {
                        return Err(InferenceError::ShapeMismatch {
                            expected: shape1.clone(),
                            actual: shape2.clone(),
                            operation: "matrix multiplication (inner dimensions must match)".to_string(),
                        });
                    }
                }
            }
            _ => {
                // Unknown operation, assume compatible
            }
        }
        
        Ok(())
    }

    /// Get final type for variable after solving
    pub fn get_resolved_type(&self, type_: Type) -> Type {
        self.apply_substitution(type_)
    }

    /// Check if dependent parameters are compatible
    fn params_compatible(&self, param1: &crate::compiler::types::DependentParam, param2: &crate::compiler::types::DependentParam) -> bool {
        use crate::compiler::types::DependentParam;
        
        match (param1, param2) {
            (DependentParam::Shape(s1), DependentParam::Shape(s2)) => s1 == s2,
            (DependentParam::Int(i1), DependentParam::Int(i2)) => i1 == i2,
            (DependentParam::Bool(b1), DependentParam::Bool(b2)) => b1 == b2,
            _ => false, // Different parameter types are incompatible
        }
    }
}

/// Type inference engine
pub struct TypeInference {
    solver: ConstraintSolver,
    environment: TypeEnvironment,
}

impl TypeInference {
    /// Create new type inference engine
    pub fn new() -> Self {
        TypeInference {
            solver: ConstraintSolver::new(),
            environment: TypeEnvironment::new(),
        }
    }

    /// Infer type of AST node
    pub fn infer_type(&mut self, ast: &AST) -> InferenceResult<Type> {
        self.infer_node_type(&ast.root)
    }

    /// Infer type of AST node
    fn infer_node_type(&mut self, node: &ASTNode) -> InferenceResult<Type> {
        match node {
            ASTNode::Atom(atom) => self.infer_atom_type(atom),
            ASTNode::List(elements) => self.infer_list_type(elements),
            ASTNode::Graph { nodes, .. } => {
                // For graph nodes, infer type of first node (simplified)
                if let Some(_first_ref) = nodes.first() {
                    // Would need AST context to resolve references properly
                    Ok(self.solver.fresh_type_var(None))
                } else {
                    Ok(Type::primitive(PrimitiveType::Unit))
                }
            }
        }
    }

    /// Infer type of atomic value
    pub fn infer_atom_type(&mut self, atom: &AtomValue) -> InferenceResult<Type> {
        match atom {
            AtomValue::Symbol(name) => {
                if let Some(type_) = self.environment.lookup(name) {
                    Ok(type_.clone())
                } else {
                    // Unknown symbol - create fresh type variable
                    Ok(self.solver.fresh_type_var(Some(name.clone())))
                }
            }
            AtomValue::Number(value) => {
                // Infer numeric type based on value
                if value.fract() == 0.0 {
                    // Integer
                    if *value >= i32::MIN as f64 && *value <= i32::MAX as f64 {
                        Ok(Type::primitive(PrimitiveType::Int32))
                    } else {
                        Ok(Type::primitive(PrimitiveType::Int64))
                    }
                } else {
                    // Float
                    Ok(Type::primitive(PrimitiveType::Float64))
                }
            }
            AtomValue::String(_) => Ok(Type::primitive(PrimitiveType::String)),
            AtomValue::Boolean(_) => Ok(Type::primitive(PrimitiveType::Bool)),
            AtomValue::Nil => Ok(Type::primitive(PrimitiveType::Unit)),
            AtomValue::Token(_) => {
                // Token type - create fresh type variable
                Ok(self.solver.fresh_type_var(Some("token".to_string())))
            }
        }
    }

    /// Infer type of list (function application or special form)
    fn infer_list_type(&mut self, elements: &[ASTNodeRef]) -> InferenceResult<Type> {
        if elements.is_empty() {
            return Ok(Type::primitive(PrimitiveType::Unit));
        }

        // For now, assume first element is function and rest are arguments
        // In a real implementation, we'd need to handle special forms like let, if, etc.
        
        let result_type = self.solver.fresh_type_var(Some("result".to_string()));
        
        // Create function type constraint
        let mut param_types = Vec::new();
        for _arg_ref in &elements[1..] {
            // Would need to resolve ASTNodeRef to actual node
            param_types.push(self.solver.fresh_type_var(None));
        }
        
        let _function_type = Type::function(param_types, result_type.clone());
        
        // Add constraint that first element has function type
        // (This is simplified - would need proper AST node resolution)
        
        Ok(result_type)
    }

    /// Add type binding to environment
    pub fn bind_type(&mut self, name: String, type_: Type) {
        self.environment.bind(name, type_);
    }

    /// Enter new scope
    pub fn enter_scope(&mut self) {
        self.environment = self.environment.child();
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if let Some(parent) = self.environment.parent.take() {
            self.environment = *parent;
        }
    }

    /// Solve all constraints
    pub fn solve_constraints(&mut self) -> InferenceResult<()> {
        self.solver.solve()
    }

    /// Get resolved type
    pub fn resolve_type(&self, type_: Type) -> Type {
        self.solver.get_resolved_type(type_)
    }
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_type_var() {
        let mut solver = ConstraintSolver::new();
        let var1 = solver.fresh_type_var(Some("x".to_string()));
        let var2 = solver.fresh_type_var(None);
        
        assert!(matches!(var1, Type::Variable(_)));
        assert!(matches!(var2, Type::Variable(_)));
        assert_ne!(var1, var2);
    }

    #[test]
    fn test_unify_same_types() {
        let mut solver = ConstraintSolver::new();
        let int_type = Type::primitive(PrimitiveType::Int32);
        
        let result = solver.unify(int_type.clone(), int_type.clone());
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No change expected
    }

    #[test]
    fn test_unify_type_variable() {
        let mut solver = ConstraintSolver::new();
        let var = solver.fresh_type_var(Some("x".to_string()));
        let int_type = Type::primitive(PrimitiveType::Int32);
        
        let result = solver.unify(var.clone(), int_type.clone());
        assert!(result.is_ok());
        assert!(result.unwrap()); // Change expected
        
        let resolved = solver.get_resolved_type(var);
        assert_eq!(resolved, int_type);
    }

    #[test]
    fn test_shape_compatibility() {
        let shape1 = Shape::concrete(vec![2, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let shape3 = Shape::concrete(vec![3, 2]);
        
        assert!(shape1.is_compatible_with(&shape2));
        assert!(!shape1.is_compatible_with(&shape3));
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = Shape::concrete(vec![1, 3]);
        let shape2 = Shape::concrete(vec![2, 3]);
        let shape3 = Shape::concrete(vec![2, 4]);
        
        assert!(shape1.is_broadcastable_with(&shape2));
        assert!(!shape1.is_broadcastable_with(&shape3));
    }

    #[test]
    fn test_type_environment() {
        let mut env = TypeEnvironment::new();
        let int_type = Type::primitive(PrimitiveType::Int32);
        
        env.bind("x".to_string(), int_type.clone());
        assert_eq!(env.lookup("x"), Some(&int_type));
        assert_eq!(env.lookup("y"), None);
        
        let child_env = env.child();
        assert_eq!(child_env.lookup("x"), Some(&int_type));
    }

    #[test]
    fn test_infer_atom_types() {
        let mut inference = TypeInference::new();
        
        let number_type = inference.infer_atom_type(&AtomValue::Number(42.0)).unwrap();
        assert_eq!(number_type, Type::primitive(PrimitiveType::Int32));
        
        let float_type = inference.infer_atom_type(&AtomValue::Number(3.14)).unwrap();
        assert_eq!(float_type, Type::primitive(PrimitiveType::Float64));
        
        let string_type = inference.infer_atom_type(&AtomValue::String("hello".to_string())).unwrap();
        assert_eq!(string_type, Type::primitive(PrimitiveType::String));
        
        let bool_type = inference.infer_atom_type(&AtomValue::Boolean(true)).unwrap();
        assert_eq!(bool_type, Type::primitive(PrimitiveType::Bool));
    }
}