// Minimal MLIR implementation for Aether
// Self-contained MLIR-like IR system without external dependencies
// Designed for bootstrapping the Aether compiler

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use crate::compiler::mlir::error_handling::{MLIRCompilationError, SourceLocation, VerificationError, VerificationErrorType};

/// Minimal MLIR context that doesn't depend on external MLIR libraries
pub struct MinimalMLIRContext {
    /// Registered dialects
    dialects: HashMap<String, Arc<dyn Dialect>>,
    /// Type registry
    types: TypeRegistry,
    /// Operation registry
    operations: OperationRegistry,
    /// Unique ID counter for operations and values
    next_id: u64,
}

impl MinimalMLIRContext {
    /// Get a type from the registry
    pub fn get_type(&self, name: &str) -> Option<MinimalMLIRType> {
        // Use the types registry to look up types
        self.types.get_type(name)
    }
    
    /// Verify an operation using the registry
    pub fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MinimalMLIRError> {
        // Use the operations registry to verify operations
        self.operations.verify_operation(op)
    }
}

/// Minimal MLIR module
pub struct MinimalMLIRModule {
    /// Module name
    name: String,
    /// Operations in the module
    operations: Vec<MinimalMLIROperation>,
    /// Module attributes
    attributes: HashMap<String, MinimalMLIRAttribute>,
    /// Symbol table for named operations
    symbol_table: HashMap<String, usize>,
}

/// Minimal MLIR operation
#[derive(Debug, Clone)]
pub struct MinimalMLIROperation {
    /// Unique operation ID
    pub id: u64,
    /// Operation name (e.g., "arith.addf", "func.func")
    pub name: String,
    /// Input operands
    pub operands: Vec<MinimalMLIRValue>,
    /// Output results
    pub results: Vec<MinimalMLIRValue>,
    /// Operation attributes
    pub attributes: HashMap<String, MinimalMLIRAttribute>,
    /// Nested regions
    pub regions: Vec<MinimalMLIRRegion>,
    /// Source location for error reporting
    pub location: Option<SourceLocation>,
}

/// Minimal MLIR value
#[derive(Debug, Clone)]
pub struct MinimalMLIRValue {
    /// Unique value ID
    pub id: u64,
    /// Value type
    pub value_type: MinimalMLIRType,
    /// Optional name for debugging
    pub name: Option<String>,
}

/// Minimal MLIR type system
#[derive(Debug, Clone, PartialEq)]
pub enum MinimalMLIRType {
    /// Void type
    None,
    /// Integer type with bit width
    Integer { width: u32, signed: bool },
    /// Floating point type
    Float { width: u32 },
    /// Index type (platform-dependent integer)
    Index,
    /// Function type
    Function { inputs: Vec<MinimalMLIRType>, outputs: Vec<MinimalMLIRType> },
    /// Tensor type
    Tensor { element_type: Box<MinimalMLIRType>, shape: Vec<i64> },
    /// Memref type
    Memref { element_type: Box<MinimalMLIRType>, shape: Vec<i64> },
    /// Pointer type
    Pointer { pointee_type: Box<MinimalMLIRType> },
    /// Custom Aether types
    AetherTensor { element_type: Box<MinimalMLIRType>, shape: Vec<i64>, device: String },
    AetherLinear { inner_type: Box<MinimalMLIRType> },
    AetherProbabilistic { distribution: String, inner_type: Box<MinimalMLIRType> },
}

/// Minimal MLIR attribute
#[derive(Debug, Clone, PartialEq)]
pub enum MinimalMLIRAttribute {
    /// String attribute
    String(String),
    /// Integer attribute
    Integer(i64),
    /// Float attribute
    Float(f64),
    /// Boolean attribute
    Boolean(bool),
    /// Array attribute
    Array(Vec<MinimalMLIRAttribute>),
    /// Dictionary attribute
    Dictionary(HashMap<String, MinimalMLIRAttribute>),
    /// Type attribute
    Type(MinimalMLIRType),
    /// Symbol reference
    SymbolRef(String),
}

/// Minimal MLIR region (contains blocks)
#[derive(Debug, Clone)]
pub struct MinimalMLIRRegion {
    /// Blocks in this region
    pub blocks: Vec<MinimalMLIRBlock>,
}

/// Minimal MLIR block (contains operations)
#[derive(Debug, Clone)]
pub struct MinimalMLIRBlock {
    /// Block arguments
    pub arguments: Vec<MinimalMLIRValue>,
    /// Operations in this block
    pub operations: Vec<MinimalMLIROperation>,
    /// Block label for branching
    pub label: Option<String>,
}

/// Dialect trait for registering operations and types
pub trait Dialect: Send + Sync {
    /// Get dialect name
    fn name(&self) -> &str;
    
    /// Get supported operations
    fn operations(&self) -> Vec<&str>;
    
    /// Get supported types
    fn types(&self) -> Vec<&str>;
    
    /// Verify operation is valid for this dialect
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError>;
}

/// Type registry for managing MLIR types
pub struct TypeRegistry {
    /// Registered type constructors
    constructors: HashMap<String, Box<dyn Fn(&[MinimalMLIRAttribute]) -> Result<MinimalMLIRType, MinimalMLIRError>>>,
}

impl TypeRegistry {
    /// Get a type by name
    pub fn get_type(&self, name: &str) -> Option<MinimalMLIRType> {
        // For now, return a basic type - in a full implementation this would use constructors
        match name {
            "i32" => Some(MinimalMLIRType::Integer(32)),
            "f32" => Some(MinimalMLIRType::Float(32)),
            _ => None,
        }
    }
}

/// Operation registry for managing MLIR operations
pub struct OperationRegistry {
    /// Registered operation verifiers
    verifiers: HashMap<String, Box<dyn Fn(&MinimalMLIROperation) -> Result<(), MinimalMLIRError>>>,
}

impl OperationRegistry {
    /// Verify an operation
    pub fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MinimalMLIRError> {
        // For now, basic verification - in a full implementation this would use verifiers
        if op.name.is_empty() {
            return Err(MinimalMLIRError::InvalidOperation("Empty operation name".to_string()));
        }
        Ok(())
    }
}

/// Minimal MLIR error types
#[derive(Debug, Clone)]
pub enum MinimalMLIRError {
    /// Context creation failed
    ContextCreation(String),
    /// Module creation failed
    ModuleCreation(String),
    /// Operation creation failed
    OperationCreation(String),
    /// Type error
    TypeError(String),
    /// Verification error
    VerificationError(String),
    /// Dialect error
    DialectError(String),
    /// Value error
    ValueError(String),
    /// Region error
    RegionError(String),
}

impl fmt::Display for MinimalMLIRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MinimalMLIRError::ContextCreation(msg) => write!(f, "Context creation error: {}", msg),
            MinimalMLIRError::ModuleCreation(msg) => write!(f, "Module creation error: {}", msg),
            MinimalMLIRError::OperationCreation(msg) => write!(f, "Operation creation error: {}", msg),
            MinimalMLIRError::TypeError(msg) => write!(f, "Type error: {}", msg),
            MinimalMLIRError::VerificationError(msg) => write!(f, "Verification error: {}", msg),
            MinimalMLIRError::DialectError(msg) => write!(f, "Dialect error: {}", msg),
            MinimalMLIRError::ValueError(msg) => write!(f, "Value error: {}", msg),
            MinimalMLIRError::RegionError(msg) => write!(f, "Region error: {}", msg),
        }
    }
}

impl std::error::Error for MinimalMLIRError {}

impl MinimalMLIRContext {
    /// Create a new minimal MLIR context
    pub fn new() -> Result<Self, MinimalMLIRError> {
        let mut context = MinimalMLIRContext {
            dialects: HashMap::new(),
            types: TypeRegistry::new(),
            operations: OperationRegistry::new(),
            next_id: 1,
        };
        
        // Register built-in dialects
        context.register_builtin_dialects()?;
        
        Ok(context)
    }
    
    /// Register built-in dialects
    fn register_builtin_dialects(&mut self) -> Result<(), MinimalMLIRError> {
        // Register builtin dialect
        self.register_dialect(Arc::new(BuiltinDialect::new()))?;
        
        // Register arithmetic dialect
        self.register_dialect(Arc::new(ArithDialect::new()))?;
        
        // Register function dialect
        self.register_dialect(Arc::new(FuncDialect::new()))?;
        
        // Register tensor dialect
        self.register_dialect(Arc::new(TensorDialect::new()))?;
        
        // Register Aether dialect
        self.register_dialect(Arc::new(AetherDialect::new()))?;
        
        Ok(())
    }
    
    /// Register a dialect
    pub fn register_dialect(&mut self, dialect: Arc<dyn Dialect>) -> Result<(), MinimalMLIRError> {
        let name = dialect.name().to_string();
        
        if self.dialects.contains_key(&name) {
            return Err(MinimalMLIRError::DialectError(
                format!("Dialect '{}' already registered", name)
            ));
        }
        
        self.dialects.insert(name, dialect);
        Ok(())
    }
    
    /// Check if dialect is registered
    pub fn is_dialect_registered(&self, name: &str) -> bool {
        self.dialects.contains_key(name)
    }
    
    /// Get registered dialects
    pub fn get_registered_dialects(&self) -> Vec<String> {
        self.dialects.keys().cloned().collect()
    }
    
    /// Create a new module
    pub fn create_module(&mut self, name: &str) -> Result<MinimalMLIRModule, MinimalMLIRError> {
        Ok(MinimalMLIRModule {
            name: name.to_string(),
            operations: Vec::new(),
            attributes: HashMap::new(),
            symbol_table: HashMap::new(),
        })
    }
    
    /// Create a new operation
    pub fn create_operation(&mut self, name: &str) -> Result<MinimalMLIROperation, MinimalMLIRError> {
        let id = self.next_id;
        self.next_id += 1;
        
        Ok(MinimalMLIROperation {
            id,
            name: name.to_string(),
            operands: Vec::new(),
            results: Vec::new(),
            attributes: HashMap::new(),
            regions: Vec::new(),
            location: None,
        })
    }
    
    /// Create a new value
    pub fn create_value(&mut self, value_type: MinimalMLIRType) -> MinimalMLIRValue {
        let id = self.next_id;
        self.next_id += 1;
        
        MinimalMLIRValue {
            id,
            value_type,
            name: None,
        }
    }
    
    /// Verify an operation
    pub fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MinimalMLIRError> {
        // Extract dialect name from operation name
        let dialect_name = if let Some(dot_pos) = op.name.find('.') {
            &op.name[..dot_pos]
        } else {
            "builtin"
        };
        
        // Find the dialect and verify
        if let Some(dialect) = self.dialects.get(dialect_name) {
            dialect.verify_operation(op).map_err(|e| {
                MinimalMLIRError::VerificationError(format!("Operation verification failed: {}", e))
            })
        } else {
            Err(MinimalMLIRError::DialectError(
                format!("Unknown dialect: {}", dialect_name)
            ))
        }
    }
    
    /// Verify a module
    pub fn verify_module(&self, module: &MinimalMLIRModule) -> Result<(), MinimalMLIRError> {
        for op in &module.operations {
            self.verify_operation(op)?;
        }
        Ok(())
    }
}

impl MinimalMLIRModule {
    /// Add an operation to the module
    pub fn add_operation(&mut self, op: MinimalMLIROperation) -> Result<(), MinimalMLIRError> {
        // Add to symbol table if it has a symbol name
        if let Some(MinimalMLIRAttribute::String(symbol)) = op.attributes.get("sym_name") {
            let index = self.operations.len();
            self.symbol_table.insert(symbol.clone(), index);
        }
        
        self.operations.push(op);
        Ok(())
    }
    
    /// Replace all operations
    pub fn replace_operations(&mut self, operations: Vec<MinimalMLIROperation>) -> Result<(), MinimalMLIRError> {
        self.operations = operations;
        self.rebuild_symbol_table();
        Ok(())
    }
    
    /// Rebuild symbol table after operations change
    fn rebuild_symbol_table(&mut self) {
        self.symbol_table.clear();
        for (index, op) in self.operations.iter().enumerate() {
            if let Some(MinimalMLIRAttribute::String(symbol)) = op.attributes.get("sym_name") {
                self.symbol_table.insert(symbol.clone(), index);
            }
        }
    }
    
    /// Get operations
    pub fn get_operations(&self) -> &[MinimalMLIROperation] {
        &self.operations
    }
    
    /// Get mutable operations
    pub fn get_operations_mut(&mut self) -> &mut Vec<MinimalMLIROperation> {
        &mut self.operations
    }
    
    /// Find operation by symbol name
    pub fn find_operation_by_symbol(&self, symbol: &str) -> Option<&MinimalMLIROperation> {
        self.symbol_table.get(symbol)
            .and_then(|&index| self.operations.get(index))
    }
    
    /// Add module attribute
    pub fn add_attribute(&mut self, name: String, attr: MinimalMLIRAttribute) {
        self.attributes.insert(name, attr);
    }
    
    /// Get module attributes
    pub fn get_attributes(&self) -> &HashMap<String, MinimalMLIRAttribute> {
        &self.attributes
    }
    
    /// Get module name
    pub fn get_name(&self) -> &str {
        &self.name
    }
    
    /// Print module to string
    pub fn to_string(&self) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("module @{} {{\n", self.name));
        
        // Add module attributes
        for (name, attr) in &self.attributes {
            output.push_str(&format!("  // {}={}\n", name, attr.to_string()));
        }
        
        // Add operations
        for op in &self.operations {
            output.push_str(&format!("  {}\n", op.to_string()));
        }
        
        output.push_str("}\n");
        output
    }
}

impl MinimalMLIROperation {
    /// Add operand
    pub fn add_operand(&mut self, operand: MinimalMLIRValue) {
        self.operands.push(operand);
    }
    
    /// Add result
    pub fn add_result(&mut self, result: MinimalMLIRValue) {
        self.results.push(result);
    }
    
    /// Add attribute
    pub fn add_attribute(&mut self, name: String, attr: MinimalMLIRAttribute) {
        self.attributes.insert(name, attr);
    }
    
    /// Add region
    pub fn add_region(&mut self, region: MinimalMLIRRegion) {
        self.regions.push(region);
    }
    
    /// Set source location
    pub fn set_location(&mut self, location: SourceLocation) {
        self.location = Some(location);
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        let mut output = String::new();
        
        // Add results
        if !self.results.is_empty() {
            let results: Vec<String> = self.results.iter()
                .map(|r| format!("%{}", r.id))
                .collect();
            output.push_str(&format!("{} = ", results.join(", ")));
        }
        
        // Add operation name
        output.push_str(&self.name);
        
        // Add operands
        if !self.operands.is_empty() {
            let operands: Vec<String> = self.operands.iter()
                .map(|o| format!("%{}", o.id))
                .collect();
            output.push_str(&format!("({})", operands.join(", ")));
        }
        
        // Add attributes
        if !self.attributes.is_empty() {
            let attrs: Vec<String> = self.attributes.iter()
                .map(|(k, v)| format!("{}={}", k, v.to_string()))
                .collect();
            output.push_str(&format!(" {{{}}}", attrs.join(", ")));
        }
        
        // Add type information
        if !self.results.is_empty() {
            let types: Vec<String> = self.results.iter()
                .map(|r| r.value_type.to_string())
                .collect();
            output.push_str(&format!(" : {}", types.join(", ")));
        }
        
        output
    }
}

impl MinimalMLIRType {
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            MinimalMLIRType::None => "none".to_string(),
            MinimalMLIRType::Integer { width, signed } => {
                if *signed {
                    format!("i{}", width)
                } else {
                    format!("ui{}", width)
                }
            }
            MinimalMLIRType::Float { width } => format!("f{}", width),
            MinimalMLIRType::Index => "index".to_string(),
            MinimalMLIRType::Function { inputs, outputs } => {
                let input_types: Vec<String> = inputs.iter().map(|t| t.to_string()).collect();
                let output_types: Vec<String> = outputs.iter().map(|t| t.to_string()).collect();
                format!("({}) -> ({})", input_types.join(", "), output_types.join(", "))
            }
            MinimalMLIRType::Tensor { element_type, shape } => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("tensor<{}x{}>", shape_str.join("x"), element_type.to_string())
            }
            MinimalMLIRType::Memref { element_type, shape } => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("memref<{}x{}>", shape_str.join("x"), element_type.to_string())
            }
            MinimalMLIRType::Pointer { pointee_type } => {
                format!("!llvm.ptr<{}>", pointee_type.to_string())
            }
            MinimalMLIRType::AetherTensor { element_type, shape, device } => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("!aether.tensor<{}x{}, {}>", shape_str.join("x"), element_type.to_string(), device)
            }
            MinimalMLIRType::AetherLinear { inner_type } => {
                format!("!aether.linear<{}>", inner_type.to_string())
            }
            MinimalMLIRType::AetherProbabilistic { distribution, inner_type } => {
                format!("!aether.prob<{}, {}>", distribution, inner_type.to_string())
            }
        }
    }
    
    /// Get size in bits if applicable
    pub fn size_bits(&self) -> Option<u32> {
        match self {
            MinimalMLIRType::Integer { width, .. } => Some(*width),
            MinimalMLIRType::Float { width } => Some(*width),
            MinimalMLIRType::Index => Some(64), // Assume 64-bit
            _ => None,
        }
    }
    
    /// Check if type is a tensor
    pub fn is_tensor(&self) -> bool {
        matches!(self, MinimalMLIRType::Tensor { .. } | MinimalMLIRType::AetherTensor { .. })
    }
    
    /// Check if type is a memref
    pub fn is_memref(&self) -> bool {
        matches!(self, MinimalMLIRType::Memref { .. })
    }
}

impl MinimalMLIRAttribute {
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            MinimalMLIRAttribute::String(s) => format!("\"{}\"", s),
            MinimalMLIRAttribute::Integer(i) => i.to_string(),
            MinimalMLIRAttribute::Float(f) => f.to_string(),
            MinimalMLIRAttribute::Boolean(b) => b.to_string(),
            MinimalMLIRAttribute::Array(arr) => {
                let elements: Vec<String> = arr.iter().map(|a| a.to_string()).collect();
                format!("[{}]", elements.join(", "))
            }
            MinimalMLIRAttribute::Dictionary(dict) => {
                let pairs: Vec<String> = dict.iter()
                    .map(|(k, v)| format!("{}={}", k, v.to_string()))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            }
            MinimalMLIRAttribute::Type(t) => t.to_string(),
            MinimalMLIRAttribute::SymbolRef(s) => format!("@{}", s),
        }
    }
}

impl TypeRegistry {
    /// Create new type registry
    pub fn new() -> Self {
        TypeRegistry {
            constructors: HashMap::new(),
        }
    }
}

impl OperationRegistry {
    /// Create new operation registry
    pub fn new() -> Self {
        OperationRegistry {
            verifiers: HashMap::new(),
        }
    }
}

// Built-in dialect implementations
pub struct BuiltinDialect;
pub struct ArithDialect;
pub struct FuncDialect;
pub struct TensorDialect;
pub struct AetherDialect;

impl BuiltinDialect {
    pub fn new() -> Self {
        BuiltinDialect
    }
}

impl Dialect for BuiltinDialect {
    fn name(&self) -> &str {
        "builtin"
    }
    
    fn operations(&self) -> Vec<&str> {
        vec!["module", "func", "return"]
    }
    
    fn types(&self) -> Vec<&str> {
        vec!["i1", "i8", "i16", "i32", "i64", "f16", "f32", "f64", "index"]
    }
    
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError> {
        // Basic verification for builtin operations
        if op.name.is_empty() {
            return Err(MLIRCompilationError::OperationCreation {
                operation: op.name.clone(),
                error: "Operation name cannot be empty".to_string(),
                location: op.location.clone().unwrap_or_default(),
                operand_types: vec![],
                expected_signature: None,
            });
        }
        Ok(())
    }
}

impl ArithDialect {
    pub fn new() -> Self {
        ArithDialect
    }
}

impl Dialect for ArithDialect {
    fn name(&self) -> &str {
        "arith"
    }
    
    fn operations(&self) -> Vec<&str> {
        vec!["addi", "addf", "subi", "subf", "muli", "mulf", "divi", "divf"]
    }
    
    fn types(&self) -> Vec<&str> {
        vec![]
    }
    
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError> {
        // Verify arithmetic operations have correct number of operands
        if op.operands.len() != 2 {
            return Err(MLIRCompilationError::OperationCreation {
                operation: op.name.clone(),
                error: format!("Arithmetic operation {} requires exactly 2 operands, got {}", op.name, op.operands.len()),
                location: op.location.clone().unwrap_or_default(),
                operand_types: op.operands.iter().map(|o| o.value_type.to_string()).collect(),
                expected_signature: Some("(T, T) -> T".to_string()),
            });
        }
        Ok(())
    }
}

impl FuncDialect {
    pub fn new() -> Self {
        FuncDialect
    }
}

impl Dialect for FuncDialect {
    fn name(&self) -> &str {
        "func"
    }
    
    fn operations(&self) -> Vec<&str> {
        vec!["func", "call", "return"]
    }
    
    fn types(&self) -> Vec<&str> {
        vec![]
    }
    
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError> {
        match op.name.as_str() {
            "func.func" => {
                // Function must have sym_name attribute
                if !op.attributes.contains_key("sym_name") {
                    return Err(MLIRCompilationError::OperationCreation {
                        operation: op.name.clone(),
                        error: "Function operation must have sym_name attribute".to_string(),
                        location: op.location.clone().unwrap_or_default(),
                        operand_types: vec![],
                        expected_signature: None,
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl TensorDialect {
    pub fn new() -> Self {
        TensorDialect
    }
}

impl Dialect for TensorDialect {
    fn name(&self) -> &str {
        "tensor"
    }
    
    fn operations(&self) -> Vec<&str> {
        vec!["empty", "extract", "insert", "cast", "reshape"]
    }
    
    fn types(&self) -> Vec<&str> {
        vec![]
    }
    
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError> {
        // Verify tensor operations work with tensor types
        for operand in &op.operands {
            if !operand.value_type.is_tensor() && op.name.contains("tensor") {
                return Err(MLIRCompilationError::OperationCreation {
                    operation: op.name.clone(),
                    error: format!("Tensor operation {} requires tensor operands", op.name),
                    location: op.location.clone().unwrap_or_default(),
                    operand_types: op.operands.iter().map(|o| o.value_type.to_string()).collect(),
                    expected_signature: Some("tensor<...> -> tensor<...>".to_string()),
                });
            }
        }
        Ok(())
    }
}

impl AetherDialect {
    pub fn new() -> Self {
        AetherDialect
    }
}

impl Dialect for AetherDialect {
    fn name(&self) -> &str {
        "aether"
    }
    
    fn operations(&self) -> Vec<&str> {
        vec!["tensor_create", "linear_resource", "prob_sample", "autodiff"]
    }
    
    fn types(&self) -> Vec<&str> {
        vec!["tensor", "linear", "prob"]
    }
    
    fn verify_operation(&self, op: &MinimalMLIROperation) -> Result<(), MLIRCompilationError> {
        // Verify Aether-specific operations
        match op.name.as_str() {
            "aether.tensor_create" => {
                // Must have shape and element type attributes
                if !op.attributes.contains_key("shape") || !op.attributes.contains_key("element_type") {
                    return Err(MLIRCompilationError::OperationCreation {
                        operation: op.name.clone(),
                        error: "tensor_create requires shape and element_type attributes".to_string(),
                        location: op.location.clone().unwrap_or_default(),
                        operand_types: vec![],
                        expected_signature: Some("() -> !aether.tensor<...>".to_string()),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for SourceLocation {
    fn default() -> Self {
        SourceLocation {
            file: None,
            line: 0,
            column: 0,
            offset: 0,
            length: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let context = MinimalMLIRContext::new();
        assert!(context.is_ok());
    }

    #[test]
    fn test_module_creation() {
        let mut context = MinimalMLIRContext::new().unwrap();
        let module = context.create_module("test_module");
        assert!(module.is_ok());
        assert_eq!(module.unwrap().name, "test_module");
    }

    #[test]
    fn test_operation_creation() {
        let mut context = MinimalMLIRContext::new().unwrap();
        let op = context.create_operation("arith.addf");
        assert!(op.is_ok());
        assert_eq!(op.unwrap().name, "arith.addf");
    }

    #[test]
    fn test_dialect_registration() {
        let context = MinimalMLIRContext::new().unwrap();
        assert!(context.is_dialect_registered("builtin"));
        assert!(context.is_dialect_registered("arith"));
        assert!(context.is_dialect_registered("func"));
        assert!(context.is_dialect_registered("tensor"));
        assert!(context.is_dialect_registered("aether"));
    }

    #[test]
    fn test_operation_verification() {
        let mut context = MinimalMLIRContext::new().unwrap();
        let mut op = context.create_operation("arith.addf").unwrap();
        
        // Should fail with no operands
        let result = context.verify_operation(&op);
        assert!(result.is_err());
        
        // Add operands
        let val1 = context.create_value(MinimalMLIRType::Float { width: 32 });
        let val2 = context.create_value(MinimalMLIRType::Float { width: 32 });
        op.add_operand(val1);
        op.add_operand(val2);
        
        // Should pass with correct operands
        let result = context.verify_operation(&op);
        assert!(result.is_ok());
    }

    #[test]
    fn test_module_operations() {
        let mut context = MinimalMLIRContext::new().unwrap();
        let mut module = context.create_module("test").unwrap();
        let op = context.create_operation("builtin.module").unwrap();
        
        let result = module.add_operation(op);
        assert!(result.is_ok());
        assert_eq!(module.get_operations().len(), 1);
    }

    #[test]
    fn test_type_string_representation() {
        let int_type = MinimalMLIRType::Integer { width: 32, signed: true };
        assert_eq!(int_type.to_string(), "i32");
        
        let float_type = MinimalMLIRType::Float { width: 64 };
        assert_eq!(float_type.to_string(), "f64");
        
        let tensor_type = MinimalMLIRType::Tensor {
            element_type: Box::new(MinimalMLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
        };
        assert_eq!(tensor_type.to_string(), "tensor<2x3x4xf32>");
    }
}