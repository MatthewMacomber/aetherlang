// Aether Type System
// Gradual typing with dependent types for tensor shapes and linear types for resource safety

use std::collections::HashMap;
use std::fmt;

/// Type identifier for type references
pub type TypeId = u32;

/// Shape variable for dependent types
pub type ShapeVar = String;

/// Core type representation supporting gradual typing
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Dynamic type (unknown at compile time)
    Dynamic,
    /// Primitive types
    Primitive(PrimitiveType),
    /// Tensor type with shape information
    Tensor {
        element_type: Box<Type>,
        shape: Shape,
    },
    /// Function type
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
        is_differentiable: bool,
    },
    /// Linear type for resource management
    Linear {
        inner_type: Box<Type>,
        ownership: LinearOwnership,
    },
    /// Probabilistic type for uncertainty
    Probabilistic {
        distribution: Distribution,
        inner_type: Box<Type>,
    },
    /// Composite types
    Struct {
        name: String,
        fields: HashMap<String, Type>,
    },
    /// Union/sum types
    Union {
        name: String,
        variants: Vec<Type>,
    },
    /// Type variable for inference
    Variable(TypeVar),
    /// Reference to another type
    Reference(TypeId),
    /// Dependent type with value-level parameters
    Dependent {
        base_type: Box<Type>,
        params: Vec<DependentParam>,
        constraints: Vec<DependentConstraint>,
    },
}

/// Primitive type kinds
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveType {
    // Numeric types
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    // Other primitives
    Bool,
    String,
    Char,
    Unit,
}

/// Tensor shape representation for dependent types
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// Concrete dimensions known at compile time
    Concrete(Vec<usize>),
    /// Symbolic dimensions with variables
    Symbolic(Vec<ShapeDim>),
    /// Computed shape from expression
    Computed(ShapeExpr),
    /// Unknown shape (for gradual typing)
    Unknown,
}

/// Shape dimension (concrete or symbolic)
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeDim {
    Concrete(usize),
    Variable(ShapeVar),
}

/// Shape expression for computed shapes
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeExpr {
    Dim(ShapeDim),
    Add(Box<ShapeExpr>, Box<ShapeExpr>),
    Sub(Box<ShapeExpr>, Box<ShapeExpr>),
    Mul(Box<ShapeExpr>, Box<ShapeExpr>),
    Div(Box<ShapeExpr>, Box<ShapeExpr>),
    /// Function application in shape expressions (for dependent types)
    Apply(String, Vec<ShapeExpr>),
    /// Conditional shape expression
    If(Box<ShapeExpr>, Box<ShapeExpr>, Box<ShapeExpr>),
}

/// Dependent type parameter (value-level parameter in types)
#[derive(Debug, Clone, PartialEq)]
pub enum DependentParam {
    /// Shape parameter
    Shape(ShapeVar),
    /// Integer parameter
    Int(String),
    /// Boolean parameter
    Bool(String),
}

/// Dependent type constraint
#[derive(Debug, Clone, PartialEq)]
pub enum DependentConstraint {
    /// Shape constraint (e.g., N > 0)
    ShapeConstraint(ShapeVar, ShapeConstraintKind),
    /// Integer constraint
    IntConstraint(String, IntConstraintKind),
    /// Relationship between parameters
    Relationship(DependentParam, RelationshipKind, DependentParam),
}

/// Kind of shape constraint
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeConstraintKind {
    /// Dimension must be positive
    Positive,
    /// Dimension must equal specific value
    Equal(usize),
    /// Dimension must be greater than value
    GreaterThan(usize),
    /// Dimension must be less than value
    LessThan(usize),
}

/// Kind of integer constraint
#[derive(Debug, Clone, PartialEq)]
pub enum IntConstraintKind {
    /// Must be positive
    Positive,
    /// Must equal specific value
    Equal(i64),
    /// Must be greater than value
    GreaterThan(i64),
    /// Must be less than value
    LessThan(i64),
}

/// Relationship between dependent parameters
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipKind {
    Equal,
    GreaterThan,
    LessThan,
    /// For matrix multiplication: inner dimensions must match
    MatMulCompatible,
    /// For broadcasting: shapes must be broadcastable
    BroadcastCompatible,
}

/// Linear type ownership for memory management
#[derive(Debug, Clone, PartialEq)]
pub enum LinearOwnership {
    /// Owned value (can be moved)
    Owned,
    /// Borrowed reference with lifetime
    Borrowed(Lifetime),
    /// Moved value (no longer accessible)
    Moved,
    /// Shared reference (multiple immutable borrows allowed)
    Shared(Lifetime),
    /// Mutable reference (exclusive borrow)
    MutableBorrow(Lifetime),
    /// GPU memory ownership (special handling for GPU resources)
    GpuOwned,
    /// GPU memory moved (deallocated on GPU)
    GpuMoved,
}

/// Lifetime for borrowed references
#[derive(Debug, Clone, PartialEq)]
pub struct Lifetime {
    pub name: String,
    pub scope_level: u32,
    pub start_location: Option<String>,
    pub end_location: Option<String>,
}

/// Linear type constraint for compile-time validation
#[derive(Debug, Clone, PartialEq)]
pub enum LinearConstraint {
    /// Variable must be used exactly once
    SingleUse(String),
    /// Variable must be moved before scope ends
    MustMove(String),
    /// Variable cannot be copied
    NoCopy(String),
    /// GPU resource must be explicitly deallocated
    GpuResource(String),
    /// Lifetime must not outlive another lifetime
    OutlivesConstraint(Lifetime, Lifetime),
}

/// Memory allocation site information
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationSite {
    pub location: String,
    pub allocation_type: AllocationType,
    pub size_hint: Option<usize>,
    pub alignment: Option<usize>,
}

/// Type of memory allocation
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationType {
    /// Stack allocation
    Stack,
    /// Heap allocation
    Heap,
    /// GPU memory allocation
    Gpu,
    /// Shared memory allocation
    Shared,
}

/// Probability distribution for probabilistic types
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    Normal { mean: f64, std: f64 },
    Uniform { min: f64, max: f64 },
    Bernoulli { p: f64 },
    Categorical { probs: Vec<f64> },
    Custom(String),
}

/// Type variable for type inference
#[derive(Debug, Clone, PartialEq)]
pub struct TypeVar {
    pub id: u32,
    pub name: Option<String>,
    pub constraints: Vec<TypeConstraint>,
}

/// Type constraint for inference
#[derive(Debug, Clone, PartialEq)]
pub enum TypeConstraint {
    /// Type must be equal to another type
    Equal(Type),
    /// Type must be a subtype of another type
    Subtype(Type),
    /// Type must support specific operations
    HasOperation(String),
    /// Type must have specific shape properties
    ShapeConstraint(ShapeConstraint),
}

/// Shape constraint for tensor types
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeConstraint {
    /// Shape must have specific rank
    Rank(usize),
    /// Shape dimensions must be compatible for operation
    Compatible(Shape),
    /// Shape must be broadcastable with another shape
    Broadcastable(Shape),
}

impl Type {
    /// Create dynamic type
    pub fn dynamic() -> Self {
        Type::Dynamic
    }

    /// Create primitive type
    pub fn primitive(prim: PrimitiveType) -> Self {
        Type::Primitive(prim)
    }

    /// Create tensor type with concrete shape
    pub fn tensor(element_type: Type, dimensions: Vec<usize>) -> Self {
        Type::Tensor {
            element_type: Box::new(element_type),
            shape: Shape::Concrete(dimensions),
        }
    }

    /// Create function type
    pub fn function(params: Vec<Type>, return_type: Type) -> Self {
        Type::Function {
            params,
            return_type: Box::new(return_type),
            is_differentiable: false,
        }
    }

    /// Create differentiable function type
    pub fn differentiable_function(params: Vec<Type>, return_type: Type) -> Self {
        Type::Function {
            params,
            return_type: Box::new(return_type),
            is_differentiable: true,
        }
    }

    /// Create linear type
    pub fn linear(inner_type: Type) -> Self {
        Type::Linear {
            inner_type: Box::new(inner_type),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Create linear type with specific ownership
    pub fn linear_with_ownership(inner_type: Type, ownership: LinearOwnership) -> Self {
        Type::Linear {
            inner_type: Box::new(inner_type),
            ownership,
        }
    }

    /// Create borrowed linear type
    pub fn linear_borrowed(inner_type: Type, lifetime: Lifetime) -> Self {
        Type::Linear {
            inner_type: Box::new(inner_type),
            ownership: LinearOwnership::Borrowed(lifetime),
        }
    }

    /// Create GPU-owned linear type
    pub fn linear_gpu(inner_type: Type) -> Self {
        Type::Linear {
            inner_type: Box::new(inner_type),
            ownership: LinearOwnership::GpuOwned,
        }
    }

    /// Create probabilistic type
    pub fn probabilistic(distribution: Distribution, inner_type: Type) -> Self {
        Type::Probabilistic {
            distribution,
            inner_type: Box::new(inner_type),
        }
    }

    /// Create struct type
    pub fn struct_type(name: String, fields: HashMap<String, Type>) -> Self {
        Type::Struct { name, fields }
    }

    /// Create union type
    pub fn union(name: String, variants: Vec<Type>) -> Self {
        Type::Union { name, variants }
    }

    /// Create type variable
    pub fn variable(id: u32, name: Option<String>) -> Self {
        Type::Variable(TypeVar {
            id,
            name,
            constraints: Vec::new(),
        })
    }

    /// Create dependent type
    pub fn dependent(base_type: Type, params: Vec<DependentParam>, constraints: Vec<DependentConstraint>) -> Self {
        Type::Dependent {
            base_type: Box::new(base_type),
            params,
            constraints,
        }
    }

    /// Create dependent tensor type with shape parameters
    pub fn dependent_tensor(element_type: Type, shape_params: Vec<ShapeVar>, constraints: Vec<DependentConstraint>) -> Self {
        let params = shape_params.into_iter()
            .map(DependentParam::Shape)
            .collect();
        
        let tensor_type = Type::Tensor {
            element_type: Box::new(element_type),
            shape: Shape::Unknown, // Will be resolved from parameters
        };
        
        Type::Dependent {
            base_type: Box::new(tensor_type),
            params,
            constraints,
        }
    }

    /// Check if type is dynamic
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Type::Dynamic)
    }

    /// Check if type is primitive
    pub fn is_primitive(&self) -> bool {
        matches!(self, Type::Primitive(_))
    }

    /// Check if type is tensor
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. })
    }

    /// Check if type is function
    pub fn is_function(&self) -> bool {
        matches!(self, Type::Function { .. })
    }

    /// Check if type is linear
    pub fn is_linear(&self) -> bool {
        matches!(self, Type::Linear { .. })
    }

    /// Check if type is probabilistic
    pub fn is_probabilistic(&self) -> bool {
        matches!(self, Type::Probabilistic { .. })
    }

    /// Check if type is dependent
    pub fn is_dependent(&self) -> bool {
        matches!(self, Type::Dependent { .. })
    }

    /// Get tensor element type and shape
    pub fn as_tensor(&self) -> Option<(&Type, &Shape)> {
        match self {
            Type::Tensor { element_type, shape } => Some((element_type, shape)),
            _ => None,
        }
    }

    /// Get function parameter and return types
    pub fn as_function(&self) -> Option<(&Vec<Type>, &Type, bool)> {
        match self {
            Type::Function { params, return_type, is_differentiable } => {
                Some((params, return_type, *is_differentiable))
            }
            _ => None,
        }
    }

    /// Get linear inner type and ownership
    pub fn as_linear(&self) -> Option<(&Type, &LinearOwnership)> {
        match self {
            Type::Linear { inner_type, ownership } => Some((inner_type, ownership)),
            _ => None,
        }
    }

    /// Check if type is movable (owned or GPU-owned)
    pub fn is_movable(&self) -> bool {
        match self {
            Type::Linear { ownership, .. } => matches!(ownership, 
                LinearOwnership::Owned | LinearOwnership::GpuOwned),
            _ => true, // Non-linear types are always movable
        }
    }

    /// Check if type is borrowed
    pub fn is_borrowed(&self) -> bool {
        match self {
            Type::Linear { ownership, .. } => matches!(ownership, 
                LinearOwnership::Borrowed(_) | LinearOwnership::Shared(_) | LinearOwnership::MutableBorrow(_)),
            _ => false,
        }
    }

    /// Check if type has been moved
    pub fn is_moved(&self) -> bool {
        match self {
            Type::Linear { ownership, .. } => matches!(ownership, 
                LinearOwnership::Moved | LinearOwnership::GpuMoved),
            _ => false,
        }
    }

    /// Check if type is GPU resource
    pub fn is_gpu_resource(&self) -> bool {
        match self {
            Type::Linear { ownership, .. } => matches!(ownership, 
                LinearOwnership::GpuOwned | LinearOwnership::GpuMoved),
            _ => false,
        }
    }

    /// Get lifetime if type is borrowed
    pub fn get_lifetime(&self) -> Option<&Lifetime> {
        match self {
            Type::Linear { ownership, .. } => match ownership {
                LinearOwnership::Borrowed(lifetime) |
                LinearOwnership::Shared(lifetime) |
                LinearOwnership::MutableBorrow(lifetime) => Some(lifetime),
                _ => None,
            },
            _ => None,
        }
    }

    /// Get dependent type information
    pub fn as_dependent(&self) -> Option<(&Type, &Vec<DependentParam>, &Vec<DependentConstraint>)> {
        match self {
            Type::Dependent { base_type, params, constraints } => Some((base_type, params, constraints)),
            _ => None,
        }
    }
}

impl Shape {
    /// Create concrete shape
    pub fn concrete(dimensions: Vec<usize>) -> Self {
        Shape::Concrete(dimensions)
    }

    /// Create symbolic shape
    pub fn symbolic(dimensions: Vec<ShapeDim>) -> Self {
        Shape::Symbolic(dimensions)
    }

    /// Create unknown shape
    pub fn unknown() -> Self {
        Shape::Unknown
    }

    /// Get rank (number of dimensions)
    pub fn rank(&self) -> Option<usize> {
        match self {
            Shape::Concrete(dims) => Some(dims.len()),
            Shape::Symbolic(dims) => Some(dims.len()),
            Shape::Computed(_) => None, // Would need evaluation
            Shape::Unknown => None,
        }
    }

    /// Check if shape is concrete
    pub fn is_concrete(&self) -> bool {
        matches!(self, Shape::Concrete(_))
    }

    /// Check if shape is symbolic
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Shape::Symbolic(_))
    }

    /// Check if shape is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, Shape::Unknown)
    }

    /// Get concrete dimensions if available
    pub fn as_concrete(&self) -> Option<&Vec<usize>> {
        match self {
            Shape::Concrete(dims) => Some(dims),
            _ => None,
        }
    }

    /// Check if two shapes are compatible for operations
    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        match (self, other) {
            (Shape::Unknown, _) | (_, Shape::Unknown) => true, // Unknown is compatible with anything
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) => {
                dims1.len() == dims2.len() && dims1.iter().zip(dims2.iter()).all(|(d1, d2)| d1 == d2)
            }
            (Shape::Symbolic(dims1), Shape::Symbolic(dims2)) => {
                // Check if symbolic shapes have same structure
                dims1.len() == dims2.len() && 
                dims1.iter().zip(dims2.iter()).all(|(d1, d2)| self.dims_compatible(d1, d2))
            }
            (Shape::Concrete(concrete_dims), Shape::Symbolic(symbolic_dims)) |
            (Shape::Symbolic(symbolic_dims), Shape::Concrete(concrete_dims)) => {
                // Check if concrete shape matches symbolic constraints
                concrete_dims.len() == symbolic_dims.len() &&
                concrete_dims.iter().zip(symbolic_dims.iter()).all(|(concrete, symbolic)| {
                    match symbolic {
                        ShapeDim::Concrete(expected) => concrete == expected,
                        ShapeDim::Variable(_) => true, // Variable can match any concrete value
                    }
                })
            }
            (Shape::Computed(_), _) | (_, Shape::Computed(_)) => {
                // Would need expression evaluation for computed shapes
                true // Simplified for now
            }
        }
    }

    /// Check if two shape dimensions are compatible
    pub fn dims_compatible(&self, dim1: &ShapeDim, dim2: &ShapeDim) -> bool {
        match (dim1, dim2) {
            (ShapeDim::Concrete(d1), ShapeDim::Concrete(d2)) => d1 == d2,
            (ShapeDim::Variable(v1), ShapeDim::Variable(v2)) => v1 == v2,
            (ShapeDim::Variable(_), ShapeDim::Concrete(_)) |
            (ShapeDim::Concrete(_), ShapeDim::Variable(_)) => true, // Variable can unify with concrete
        }
    }

    /// Check if shape can be broadcast with another shape
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        match (self, other) {
            (Shape::Unknown, _) | (_, Shape::Unknown) => true,
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) => {
                // NumPy-style broadcasting rules
                let max_len = dims1.len().max(dims2.len());
                let padded1: Vec<usize> = std::iter::repeat(1)
                    .take(max_len - dims1.len())
                    .chain(dims1.iter().copied())
                    .collect();
                let padded2: Vec<usize> = std::iter::repeat(1)
                    .take(max_len - dims2.len())
                    .chain(dims2.iter().copied())
                    .collect();
                
                padded1.iter().zip(padded2.iter()).all(|(d1, d2)| {
                    *d1 == 1 || *d2 == 1 || d1 == d2
                })
            }
            (Shape::Symbolic(dims1), Shape::Symbolic(dims2)) => {
                // Symbolic broadcasting - check if dimensions can be broadcast
                let max_len = dims1.len().max(dims2.len());
                for i in 0..max_len {
                    let dim1 = dims1.get(dims1.len().saturating_sub(max_len - i));
                    let dim2 = dims2.get(dims2.len().saturating_sub(max_len - i));
                    
                    match (dim1, dim2) {
                        (Some(ShapeDim::Concrete(1)), _) | (_, Some(ShapeDim::Concrete(1))) => continue,
                        (Some(d1), Some(d2)) if self.dims_compatible(d1, d2) => continue,
                        (None, _) | (_, None) => continue, // Missing dimensions are treated as 1
                        _ => return false,
                    }
                }
                true
            }
            _ => true, // Mixed cases simplified for now
        }
    }

    /// Unify two shapes, returning the most specific unified shape
    pub fn unify_with(&self, other: &Shape) -> Option<Shape> {
        match (self, other) {
            (Shape::Unknown, s) | (s, Shape::Unknown) => Some(s.clone()),
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) if dims1 == dims2 => Some(self.clone()),
            (Shape::Concrete(dims), Shape::Symbolic(symbolic_dims)) |
            (Shape::Symbolic(symbolic_dims), Shape::Concrete(dims)) => {
                if dims.len() != symbolic_dims.len() {
                    return None;
                }
                
                let mut unified_dims = Vec::new();
                for (concrete, symbolic) in dims.iter().zip(symbolic_dims.iter()) {
                    match symbolic {
                        ShapeDim::Concrete(expected) if concrete == expected => {
                            unified_dims.push(ShapeDim::Concrete(*concrete));
                        }
                        ShapeDim::Variable(_) => {
                            unified_dims.push(ShapeDim::Concrete(*concrete));
                        }
                        _ => return None,
                    }
                }
                Some(Shape::Symbolic(unified_dims))
            }
            (Shape::Symbolic(dims1), Shape::Symbolic(dims2)) => {
                if dims1.len() != dims2.len() {
                    return None;
                }
                
                let mut unified_dims = Vec::new();
                for (d1, d2) in dims1.iter().zip(dims2.iter()) {
                    match (d1, d2) {
                        (ShapeDim::Concrete(c1), ShapeDim::Concrete(c2)) if c1 == c2 => {
                            unified_dims.push(ShapeDim::Concrete(*c1));
                        }
                        (ShapeDim::Variable(v1), ShapeDim::Variable(v2)) if v1 == v2 => {
                            unified_dims.push(ShapeDim::Variable(v1.clone()));
                        }
                        (ShapeDim::Concrete(c), ShapeDim::Variable(_)) |
                        (ShapeDim::Variable(_), ShapeDim::Concrete(c)) => {
                            unified_dims.push(ShapeDim::Concrete(*c));
                        }
                        _ => return None,
                    }
                }
                Some(Shape::Symbolic(unified_dims))
            }
            _ => None, // Cannot unify computed shapes without evaluation
        }
    }

    /// Infer result shape for matrix multiplication
    pub fn matmul_result_shape(&self, other: &Shape) -> Option<Shape> {
        match (self, other) {
            (Shape::Concrete(dims1), Shape::Concrete(dims2)) => {
                if dims1.len() < 2 || dims2.len() < 2 {
                    return None;
                }
                
                let inner1 = dims1[dims1.len() - 1];
                let inner2 = dims2[dims2.len() - 2];
                
                if inner1 != inner2 {
                    return None;
                }
                
                let mut result_dims = Vec::new();
                
                // Batch dimensions (all but last 2)
                let batch_dims1 = &dims1[..dims1.len() - 2];
                let batch_dims2 = &dims2[..dims2.len() - 2];
                
                // Broadcast batch dimensions
                let max_batch_len = batch_dims1.len().max(batch_dims2.len());
                for i in 0..max_batch_len {
                    let dim1 = batch_dims1.get(batch_dims1.len().saturating_sub(max_batch_len - i)).copied().unwrap_or(1);
                    let dim2 = batch_dims2.get(batch_dims2.len().saturating_sub(max_batch_len - i)).copied().unwrap_or(1);
                    
                    if dim1 == 1 {
                        result_dims.push(dim2);
                    } else if dim2 == 1 {
                        result_dims.push(dim1);
                    } else if dim1 == dim2 {
                        result_dims.push(dim1);
                    } else {
                        return None; // Incompatible batch dimensions
                    }
                }
                
                // Matrix dimensions: [M, K] @ [K, N] -> [M, N]
                result_dims.push(dims1[dims1.len() - 2]); // M
                result_dims.push(dims2[dims2.len() - 1]); // N
                
                Some(Shape::Concrete(result_dims))
            }
            (Shape::Symbolic(dims1), Shape::Symbolic(dims2)) => {
                if dims1.len() < 2 || dims2.len() < 2 {
                    return None;
                }
                
                // For symbolic shapes, we can still infer the structure
                let mut result_dims = Vec::new();
                
                // Batch dimensions
                let batch_dims1 = &dims1[..dims1.len() - 2];
                let batch_dims2 = &dims2[..dims2.len() - 2];
                
                let max_batch_len = batch_dims1.len().max(batch_dims2.len());
                for i in 0..max_batch_len {
                    let dim1 = batch_dims1.get(batch_dims1.len().saturating_sub(max_batch_len - i));
                    let dim2 = batch_dims2.get(batch_dims2.len().saturating_sub(max_batch_len - i));
                    
                    match (dim1, dim2) {
                        (Some(d1), Some(d2)) => {
                            if self.dims_compatible(d1, d2) {
                                result_dims.push(d1.clone());
                            } else {
                                return None;
                            }
                        }
                        (Some(d), None) | (None, Some(d)) => result_dims.push(d.clone()),
                        (None, None) => {} // No batch dimension
                    }
                }
                
                // Matrix dimensions
                result_dims.push(dims1[dims1.len() - 2].clone()); // M
                result_dims.push(dims2[dims2.len() - 1].clone()); // N
                
                Some(Shape::Symbolic(result_dims))
            }
            _ => None, // Mixed or computed shapes not supported yet
        }
    }
}

impl ShapeDim {
    /// Create concrete dimension
    pub fn concrete(size: usize) -> Self {
        ShapeDim::Concrete(size)
    }

    /// Create variable dimension
    pub fn variable(name: String) -> Self {
        ShapeDim::Variable(name)
    }

    /// Check if dimension is concrete
    pub fn is_concrete(&self) -> bool {
        matches!(self, ShapeDim::Concrete(_))
    }

    /// Get concrete size if available
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            ShapeDim::Concrete(size) => Some(*size),
            _ => None,
        }
    }
}

impl PrimitiveType {
    /// Check if type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, 
            PrimitiveType::Int8 | PrimitiveType::Int16 | PrimitiveType::Int32 | PrimitiveType::Int64 |
            PrimitiveType::UInt8 | PrimitiveType::UInt16 | PrimitiveType::UInt32 | PrimitiveType::UInt64 |
            PrimitiveType::Float32 | PrimitiveType::Float64
        )
    }

    /// Check if type is integer
    pub fn is_integer(&self) -> bool {
        matches!(self,
            PrimitiveType::Int8 | PrimitiveType::Int16 | PrimitiveType::Int32 | PrimitiveType::Int64 |
            PrimitiveType::UInt8 | PrimitiveType::UInt16 | PrimitiveType::UInt32 | PrimitiveType::UInt64
        )
    }

    /// Check if type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, PrimitiveType::Float32 | PrimitiveType::Float64)
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            PrimitiveType::Int8 | PrimitiveType::UInt8 => 1,
            PrimitiveType::Int16 | PrimitiveType::UInt16 => 2,
            PrimitiveType::Int32 | PrimitiveType::UInt32 | PrimitiveType::Float32 => 4,
            PrimitiveType::Int64 | PrimitiveType::UInt64 | PrimitiveType::Float64 => 8,
            PrimitiveType::Bool => 1,
            PrimitiveType::Char => 4, // UTF-32
            PrimitiveType::String => 0, // Variable size
            PrimitiveType::Unit => 0,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Dynamic => write!(f, "dyn"),
            Type::Primitive(prim) => write!(f, "{}", prim),
            Type::Tensor { element_type, shape } => {
                write!(f, "Tensor<{}, {}>", element_type, shape)
            }
            Type::Function { params, return_type, is_differentiable } => {
                let diff_marker = if *is_differentiable { "@diff " } else { "" };
                write!(f, "{}({}) -> {}", diff_marker, 
                    params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", "),
                    return_type)
            }
            Type::Linear { inner_type, ownership } => {
                write!(f, "linear<{}, {:?}>", inner_type, ownership)
            }
            Type::Probabilistic { distribution, inner_type } => {
                write!(f, "prob<{:?}, {}>", distribution, inner_type)
            }
            Type::Struct { name, .. } => write!(f, "struct {}", name),
            Type::Union { name, .. } => write!(f, "union {}", name),
            Type::Variable(var) => {
                if let Some(name) = &var.name {
                    write!(f, "'{}", name)
                } else {
                    write!(f, "'t{}", var.id)
                }
            }
            Type::Reference(id) => write!(f, "ref({})", id),
            Type::Dependent { base_type, params, constraints } => {
                write!(f, "{}[", base_type)?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    match param {
                        DependentParam::Shape(var) => write!(f, "{}", var)?,
                        DependentParam::Int(var) => write!(f, "{}: int", var)?,
                        DependentParam::Bool(var) => write!(f, "{}: bool", var)?,
                    }
                }
                if !constraints.is_empty() {
                    write!(f, " where ")?;
                    for (i, constraint) in constraints.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{:?}", constraint)?;
                    }
                }
                write!(f, "]")
            }
        }
    }
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveType::Int8 => write!(f, "i8"),
            PrimitiveType::Int16 => write!(f, "i16"),
            PrimitiveType::Int32 => write!(f, "i32"),
            PrimitiveType::Int64 => write!(f, "i64"),
            PrimitiveType::UInt8 => write!(f, "u8"),
            PrimitiveType::UInt16 => write!(f, "u16"),
            PrimitiveType::UInt32 => write!(f, "u32"),
            PrimitiveType::UInt64 => write!(f, "u64"),
            PrimitiveType::Float32 => write!(f, "f32"),
            PrimitiveType::Float64 => write!(f, "f64"),
            PrimitiveType::Bool => write!(f, "bool"),
            PrimitiveType::String => write!(f, "string"),
            PrimitiveType::Char => write!(f, "char"),
            PrimitiveType::Unit => write!(f, "()"),
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shape::Concrete(dims) => {
                write!(f, "[{}]", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
            }
            Shape::Symbolic(dims) => {
                write!(f, "[{}]", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
            }
            Shape::Computed(expr) => write!(f, "[{}]", expr),
            Shape::Unknown => write!(f, "[?]"),
        }
    }
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Concrete(size) => write!(f, "{}", size),
            ShapeDim::Variable(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for ShapeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeExpr::Dim(dim) => write!(f, "{}", dim),
            ShapeExpr::Add(left, right) => write!(f, "({} + {})", left, right),
            ShapeExpr::Sub(left, right) => write!(f, "({} - {})", left, right),
            ShapeExpr::Mul(left, right) => write!(f, "({} * {})", left, right),
            ShapeExpr::Div(left, right) => write!(f, "({} / {})", left, right),
            ShapeExpr::Apply(func, args) => {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            ShapeExpr::If(cond, then_expr, else_expr) => {
                write!(f, "if {} then {} else {}", cond, then_expr, else_expr)
            }
        }
    }
}

impl Lifetime {
    /// Create new lifetime
    pub fn new(name: String, scope_level: u32) -> Self {
        Lifetime {
            name,
            scope_level,
            start_location: None,
            end_location: None,
        }
    }

    /// Create lifetime with location information
    pub fn with_location(name: String, scope_level: u32, start_location: String) -> Self {
        Lifetime {
            name,
            scope_level,
            start_location: Some(start_location),
            end_location: None,
        }
    }

    /// Check if this lifetime outlives another
    pub fn outlives(&self, other: &Lifetime) -> bool {
        self.scope_level <= other.scope_level
    }

    /// Check if lifetimes are compatible for borrowing
    pub fn is_compatible_with(&self, other: &Lifetime) -> bool {
        self.name == other.name || self.outlives(other)
    }
}

impl LinearOwnership {
    /// Check if ownership allows reading
    pub fn allows_read(&self) -> bool {
        !matches!(self, LinearOwnership::Moved | LinearOwnership::GpuMoved)
    }

    /// Check if ownership allows writing
    pub fn allows_write(&self) -> bool {
        matches!(self, 
            LinearOwnership::Owned | 
            LinearOwnership::GpuOwned | 
            LinearOwnership::MutableBorrow(_)
        )
    }

    /// Check if ownership allows moving
    pub fn allows_move(&self) -> bool {
        matches!(self, LinearOwnership::Owned | LinearOwnership::GpuOwned)
    }

    /// Check if this is a GPU resource
    pub fn is_gpu_resource(&self) -> bool {
        matches!(self, LinearOwnership::GpuOwned | LinearOwnership::GpuMoved)
    }
}

impl AllocationSite {
    /// Create new allocation site
    pub fn new(location: String, allocation_type: AllocationType) -> Self {
        AllocationSite {
            location,
            allocation_type,
            size_hint: None,
            alignment: None,
        }
    }

    /// Create allocation site with size hint
    pub fn with_size(location: String, allocation_type: AllocationType, size: usize) -> Self {
        AllocationSite {
            location,
            allocation_type,
            size_hint: Some(size),
            alignment: None,
        }
    }
}