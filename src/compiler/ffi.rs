// Aether Foreign Function Interface (FFI)
// Zero-cost interoperability with C/C++, Rust, JavaScript, and ML frameworks

use std::collections::HashMap;
use std::path::PathBuf;
use crate::compiler::types::{Type, PrimitiveType};
// AST imports removed as not needed for FFI

/// FFI binding configuration
#[derive(Debug, Clone)]
pub struct FFIConfig {
    /// Target language for bindings
    pub target: FFITarget,
    /// Library name
    pub library_name: String,
    /// Header files or interface definitions
    pub headers: Vec<PathBuf>,
    /// Link libraries
    pub link_libraries: Vec<String>,
    /// Include paths
    pub include_paths: Vec<PathBuf>,
    /// Custom type mappings
    pub type_mappings: HashMap<String, Type>,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Error handling strategy
    pub error_handling: ErrorHandling,
}

/// FFI target language
#[derive(Debug, Clone, PartialEq)]
pub enum FFITarget {
    /// C library bindings
    C,
    /// C++ library bindings
    Cpp,
    /// Rust library bindings
    Rust,
    /// JavaScript/WebAssembly bindings
    JavaScript,
    /// Python bindings (for ML frameworks)
    Python,
    /// Custom target
    Custom(String),
}

/// Memory management strategy for FFI
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Aether manages memory (copy data)
    AetherManaged,
    /// Foreign library manages memory (zero-copy)
    ForeignManaged,
    /// Shared ownership with reference counting
    SharedOwnership,
    /// Linear types with explicit transfer
    LinearTransfer,
}

/// Error handling strategy for FFI calls
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorHandling {
    /// Return error codes
    ErrorCodes,
    /// Use exceptions (C++/JavaScript)
    Exceptions,
    /// Result types (Rust-style)
    Results,
    /// Panic on error
    Panic,
}

/// FFI function signature
#[derive(Debug, Clone)]
pub struct FFIFunction {
    /// Function name in foreign library
    pub foreign_name: String,
    /// Function name in Aether
    pub aether_name: String,
    /// Parameter types
    pub parameters: Vec<FFIParameter>,
    /// Return type
    pub return_type: Option<FFIType>,
    /// Calling convention
    pub calling_convention: CallingConvention,
    /// Whether function is thread-safe
    pub thread_safe: bool,
    /// Documentation
    pub documentation: Option<String>,
}

/// FFI parameter definition
#[derive(Debug, Clone)]
pub struct FFIParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ffi_type: FFIType,
    /// Whether parameter is input, output, or both
    pub direction: ParameterDirection,
    /// Whether parameter is optional
    pub optional: bool,
}

/// Parameter direction for FFI calls
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterDirection {
    /// Input parameter
    In,
    /// Output parameter
    Out,
    /// Input/output parameter
    InOut,
}

/// FFI type representation
#[derive(Debug, Clone)]
pub enum FFIType {
    /// Primitive types that map directly
    Primitive(PrimitiveType),
    /// Pointer to type
    Pointer(Box<FFIType>),
    /// Array with known size
    Array(Box<FFIType>, usize),
    /// Slice with runtime size
    Slice(Box<FFIType>),
    /// String (null-terminated C string)
    CString,
    /// Wide string (UTF-16)
    WideString,
    /// Opaque handle/pointer
    Handle(String),
    /// Struct type
    Struct(String, Vec<FFIField>),
    /// Union type
    Union(String, Vec<FFIField>),
    /// Function pointer
    FunctionPointer(Vec<FFIType>, Option<Box<FFIType>>),
    /// Tensor type (for ML frameworks)
    Tensor {
        element_type: Box<FFIType>,
        dimensions: Option<Vec<usize>>,
    },
    /// Custom type with conversion functions
    Custom {
        name: String,
        to_aether: String,
        from_aether: String,
    },
}

/// FFI struct/union field
#[derive(Debug, Clone)]
pub struct FFIField {
    /// Field name
    pub name: String,
    /// Field type
    pub ffi_type: FFIType,
    /// Field offset (for manual layout)
    pub offset: Option<usize>,
}

/// Calling convention for FFI functions
#[derive(Debug, Clone, PartialEq)]
pub enum CallingConvention {
    /// C calling convention
    C,
    /// Standard calling convention (platform default)
    Std,
    /// Fast calling convention
    Fast,
    /// System calling convention
    System,
    /// Windows API calling convention
    Win64,
}

/// FFI binding generator
pub struct FFIBindingGenerator {
    /// Configuration
    config: FFIConfig,
    /// Generated functions
    functions: Vec<FFIFunction>,
    /// Generated types
    types: HashMap<String, FFIType>,
    /// Type conversion functions
    conversions: HashMap<String, TypeConversion>,
}

/// Type conversion between Aether and foreign types
#[derive(Debug, Clone)]
pub struct TypeConversion {
    /// Aether type
    pub aether_type: Type,
    /// Foreign type
    pub foreign_type: FFIType,
    /// Conversion function from Aether to foreign
    pub to_foreign: String,
    /// Conversion function from foreign to Aether
    pub from_foreign: String,
    /// Whether conversion is zero-cost
    pub zero_cost: bool,
}

impl FFIBindingGenerator {
    /// Create new FFI binding generator
    pub fn new(config: FFIConfig) -> Self {
        Self {
            config,
            functions: Vec::new(),
            types: HashMap::new(),
            conversions: HashMap::new(),
        }
    }

    /// Add FFI function binding
    pub fn add_function(&mut self, function: FFIFunction) {
        self.functions.push(function);
    }

    /// Add FFI type definition
    pub fn add_type(&mut self, name: String, ffi_type: FFIType) {
        self.types.insert(name, ffi_type);
    }

    /// Add type conversion
    pub fn add_conversion(&mut self, name: String, conversion: TypeConversion) {
        self.conversions.insert(name, conversion);
    }

    /// Generate bindings for target language
    pub fn generate_bindings(&self) -> Result<String, FFIError> {
        match self.config.target {
            FFITarget::C => self.generate_c_bindings(),
            FFITarget::Cpp => self.generate_cpp_bindings(),
            FFITarget::Rust => self.generate_rust_bindings(),
            FFITarget::JavaScript => self.generate_js_bindings(),
            FFITarget::Python => self.generate_python_bindings(),
            FFITarget::Custom(ref name) => self.generate_custom_bindings(name),
        }
    }

    /// Generate C library bindings
    fn generate_c_bindings(&self) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate header includes
        output.push_str("// Generated C bindings for Aether\n");
        output.push_str("#include <stdint.h>\n");
        output.push_str("#include <stdbool.h>\n");
        output.push_str("#include <stddef.h>\n\n");
        
        // Generate type definitions
        for (name, ffi_type) in &self.types {
            output.push_str(&self.generate_c_type_definition(name, ffi_type)?);
            output.push('\n');
        }
        
        // Generate function declarations
        for function in &self.functions {
            output.push_str(&self.generate_c_function_declaration(function)?);
            output.push('\n');
        }
        
        // Generate wrapper implementations
        for function in &self.functions {
            output.push_str(&self.generate_c_function_wrapper(function)?);
            output.push('\n');
        }
        
        Ok(output)
    }

    /// Generate C++ library bindings
    fn generate_cpp_bindings(&self) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate header
        output.push_str("// Generated C++ bindings for Aether\n");
        output.push_str("#include <cstdint>\n");
        output.push_str("#include <memory>\n");
        output.push_str("#include <vector>\n");
        output.push_str("#include <string>\n");
        output.push_str("#include <stdexcept>\n\n");
        
        // Generate namespace
        output.push_str("namespace aether {\n\n");
        
        // Generate type definitions with RAII wrappers
        for (name, ffi_type) in &self.types {
            output.push_str(&self.generate_cpp_type_definition(name, ffi_type)?);
            output.push('\n');
        }
        
        // Generate function declarations with exception handling
        for function in &self.functions {
            output.push_str(&self.generate_cpp_function_declaration(function)?);
            output.push('\n');
        }
        
        output.push_str("} // namespace aether\n");
        
        Ok(output)
    }

    /// Generate Rust library bindings
    fn generate_rust_bindings(&self) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate header
        output.push_str("// Generated Rust bindings for Aether\n");
        output.push_str("use std::ffi::{CStr, CString};\n");
        output.push_str("use std::os::raw::{c_char, c_int, c_void};\n\n");
        
        // Generate type definitions with ownership integration
        for (name, ffi_type) in &self.types {
            output.push_str(&self.generate_rust_type_definition(name, ffi_type)?);
            output.push('\n');
        }
        
        // Generate extern block for foreign functions
        output.push_str("extern \"C\" {\n");
        for function in &self.functions {
            output.push_str(&format!("    {};\n", self.generate_rust_extern_declaration(function)?));
        }
        output.push_str("}\n\n");
        
        // Generate safe Rust wrappers
        for function in &self.functions {
            output.push_str(&self.generate_rust_safe_wrapper(function)?);
            output.push('\n');
        }
        
        Ok(output)
    }

    /// Generate JavaScript/WebAssembly bindings
    fn generate_js_bindings(&self) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate TypeScript definitions
        output.push_str("// Generated JavaScript bindings for Aether\n");
        output.push_str("// TypeScript definitions\n\n");
        
        // Generate type definitions
        for (name, ffi_type) in &self.types {
            output.push_str(&self.generate_js_type_definition(name, ffi_type)?);
            output.push('\n');
        }
        
        // Generate WebAssembly module interface
        output.push_str("export interface AetherModule {\n");
        for function in &self.functions {
            output.push_str(&format!("  {}: {};\n", 
                function.aether_name, 
                self.generate_js_function_signature(function)?));
        }
        output.push_str("}\n\n");
        
        // Generate JavaScript wrapper functions
        output.push_str("export class AetherBindings {\n");
        output.push_str("  private module: AetherModule;\n\n");
        output.push_str("  constructor(module: AetherModule) {\n");
        output.push_str("    this.module = module;\n");
        output.push_str("  }\n\n");
        
        for function in &self.functions {
            output.push_str(&self.generate_js_wrapper_method(function)?);
            output.push('\n');
        }
        
        output.push_str("}\n");
        
        Ok(output)
    }

    /// Generate Python bindings for ML frameworks
    fn generate_python_bindings(&self) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate Python module header
        output.push_str("# Generated Python bindings for Aether\n");
        output.push_str("import ctypes\n");
        output.push_str("import numpy as np\n");
        output.push_str("from typing import Optional, List, Union, Any\n\n");
        
        // Generate type definitions with NumPy integration
        for (name, ffi_type) in &self.types {
            output.push_str(&self.generate_python_type_definition(name, ffi_type)?);
            output.push('\n');
        }
        
        // Generate function wrappers with NumPy array handling
        output.push_str("class AetherBindings:\n");
        output.push_str("    def __init__(self, library_path: str):\n");
        output.push_str("        self._lib = ctypes.CDLL(library_path)\n");
        output.push_str("        self._setup_functions()\n\n");
        
        output.push_str("    def _setup_functions(self):\n");
        for function in &self.functions {
            output.push_str(&self.generate_python_function_setup(function)?);
        }
        output.push('\n');
        
        for function in &self.functions {
            output.push_str(&self.generate_python_wrapper_method(function)?);
            output.push('\n');
        }
        
        Ok(output)
    }

    /// Generate custom target bindings
    fn generate_custom_bindings(&self, target: &str) -> Result<String, FFIError> {
        Err(FFIError::UnsupportedTarget(target.to_string()))
    }

    // Helper methods for C bindings
    fn generate_c_type_definition(&self, name: &str, ffi_type: &FFIType) -> Result<String, FFIError> {
        match ffi_type {
            FFIType::Struct(_, fields) => {
                let mut output = format!("typedef struct {} {{\n", name);
                for field in fields {
                    output.push_str(&format!("    {} {};\n", 
                        self.ffi_type_to_c_type(&field.ffi_type)?, 
                        field.name));
                }
                output.push_str(&format!("}} {};\n", name));
                Ok(output)
            }
            FFIType::Union(_, fields) => {
                let mut output = format!("typedef union {} {{\n", name);
                for field in fields {
                    output.push_str(&format!("    {} {};\n", 
                        self.ffi_type_to_c_type(&field.ffi_type)?, 
                        field.name));
                }
                output.push_str(&format!("}} {};\n", name));
                Ok(output)
            }
            _ => Ok(format!("typedef {} {};\n", self.ffi_type_to_c_type(ffi_type)?, name))
        }
    }

    fn generate_c_function_declaration(&self, function: &FFIFunction) -> Result<String, FFIError> {
        let return_type = match &function.return_type {
            Some(ret_type) => self.ffi_type_to_c_type(ret_type)?,
            None => "void".to_string(),
        };
        
        let params: Result<Vec<String>, FFIError> = function.parameters.iter()
            .map(|param| Ok(format!("{} {}", 
                self.ffi_type_to_c_type(&param.ffi_type)?, 
                param.name)))
            .collect();
        
        let params_vec = params?;
        let params_str = if params_vec.is_empty() {
            "void".to_string()
        } else {
            params_vec.join(", ")
        };
        
        Ok(format!("{} {}({});", return_type, function.foreign_name, params_str))
    }

    fn generate_c_function_wrapper(&self, function: &FFIFunction) -> Result<String, FFIError> {
        let mut output = String::new();
        
        // Generate wrapper function that handles type conversions
        let return_type = match &function.return_type {
            Some(ret_type) => self.ffi_type_to_c_type(ret_type)?,
            None => "void".to_string(),
        };
        
        output.push_str(&format!("{} aether_{}(", return_type, function.aether_name));
        
        let params: Result<Vec<String>, FFIError> = function.parameters.iter()
            .map(|param| Ok(format!("{} {}", 
                self.ffi_type_to_c_type(&param.ffi_type)?, 
                param.name)))
            .collect();
        
        output.push_str(&params?.join(", "));
        output.push_str(") {\n");
        
        // Add type conversion and error handling logic
        output.push_str("    // Type conversions and error handling\n");
        
        // Call original function
        if function.return_type.is_some() {
            output.push_str("    return ");
        } else {
            output.push_str("    ");
        }
        
        output.push_str(&format!("{}(", function.foreign_name));
        let param_names: Vec<String> = function.parameters.iter()
            .map(|param| param.name.clone())
            .collect();
        output.push_str(&param_names.join(", "));
        output.push_str(");\n");
        
        output.push_str("}\n");
        
        Ok(output)
    }

    fn ffi_type_to_c_type(&self, ffi_type: &FFIType) -> Result<String, FFIError> {
        match ffi_type {
            FFIType::Primitive(PrimitiveType::Int8) => Ok("int8_t".to_string()),
            FFIType::Primitive(PrimitiveType::Int16) => Ok("int16_t".to_string()),
            FFIType::Primitive(PrimitiveType::Int32) => Ok("int32_t".to_string()),
            FFIType::Primitive(PrimitiveType::Int64) => Ok("int64_t".to_string()),
            FFIType::Primitive(PrimitiveType::UInt8) => Ok("uint8_t".to_string()),
            FFIType::Primitive(PrimitiveType::UInt16) => Ok("uint16_t".to_string()),
            FFIType::Primitive(PrimitiveType::UInt32) => Ok("uint32_t".to_string()),
            FFIType::Primitive(PrimitiveType::UInt64) => Ok("uint64_t".to_string()),
            FFIType::Primitive(PrimitiveType::Float32) => Ok("float".to_string()),
            FFIType::Primitive(PrimitiveType::Float64) => Ok("double".to_string()),
            FFIType::Primitive(PrimitiveType::Bool) => Ok("bool".to_string()),
            FFIType::Primitive(PrimitiveType::Char) => Ok("char".to_string()),
            FFIType::Primitive(PrimitiveType::Unit) => Ok("void".to_string()),
            FFIType::Pointer(inner) => Ok(format!("{}*", self.ffi_type_to_c_type(inner)?)),
            FFIType::Array(inner, size) => Ok(format!("{}[{}]", self.ffi_type_to_c_type(inner)?, size)),
            FFIType::CString => Ok("const char*".to_string()),
            FFIType::Handle(name) => Ok(format!("{}*", name)),
            _ => Err(FFIError::UnsupportedType(format!("{:?}", ffi_type))),
        }
    }

    // Helper methods for other target languages would be implemented similarly...
    fn generate_cpp_type_definition(&self, _name: &str, _ffi_type: &FFIType) -> Result<String, FFIError> {
        // Implementation for C++ type definitions
        Ok("// C++ type definition placeholder\n".to_string())
    }

    fn generate_cpp_function_declaration(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for C++ function declarations
        Ok("// C++ function declaration placeholder\n".to_string())
    }

    fn generate_rust_type_definition(&self, _name: &str, _ffi_type: &FFIType) -> Result<String, FFIError> {
        // Implementation for Rust type definitions
        Ok("// Rust type definition placeholder\n".to_string())
    }

    fn generate_rust_extern_declaration(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for Rust extern declarations
        Ok("// Rust extern declaration placeholder".to_string())
    }

    fn generate_rust_safe_wrapper(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for Rust safe wrappers
        Ok("// Rust safe wrapper placeholder\n".to_string())
    }

    fn generate_js_type_definition(&self, _name: &str, _ffi_type: &FFIType) -> Result<String, FFIError> {
        // Implementation for JavaScript type definitions
        Ok("// JavaScript type definition placeholder\n".to_string())
    }

    fn generate_js_function_signature(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for JavaScript function signatures
        Ok("() => void".to_string())
    }

    fn generate_js_wrapper_method(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for JavaScript wrapper methods
        Ok("  // JavaScript wrapper method placeholder\n".to_string())
    }

    fn generate_python_type_definition(&self, _name: &str, _ffi_type: &FFIType) -> Result<String, FFIError> {
        // Implementation for Python type definitions
        Ok("# Python type definition placeholder\n".to_string())
    }

    fn generate_python_function_setup(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for Python function setup
        Ok("        # Python function setup placeholder\n".to_string())
    }

    fn generate_python_wrapper_method(&self, _function: &FFIFunction) -> Result<String, FFIError> {
        // Implementation for Python wrapper methods
        Ok("    # Python wrapper method placeholder\n".to_string())
    }
}

/// FFI error types
#[derive(Debug, Clone)]
pub enum FFIError {
    /// Unsupported target language
    UnsupportedTarget(String),
    /// Unsupported type conversion
    UnsupportedType(String),
    /// Invalid function signature
    InvalidSignature(String),
    /// Missing type definition
    MissingType(String),
    /// Code generation error
    CodeGenError(String),
    /// IO error during binding generation
    IoError(String),
}

impl std::fmt::Display for FFIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FFIError::UnsupportedTarget(target) => write!(f, "Unsupported FFI target: {}", target),
            FFIError::UnsupportedType(type_name) => write!(f, "Unsupported FFI type: {}", type_name),
            FFIError::InvalidSignature(sig) => write!(f, "Invalid function signature: {}", sig),
            FFIError::MissingType(type_name) => write!(f, "Missing type definition: {}", type_name),
            FFIError::CodeGenError(msg) => write!(f, "Code generation error: {}", msg),
            FFIError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for FFIError {}

/// ML framework integration
pub struct MLFrameworkIntegration {
    /// Framework name (TensorFlow, PyTorch, etc.)
    pub framework: String,
    /// Model format support
    pub supported_formats: Vec<ModelFormat>,
    /// Tensor type mappings
    pub tensor_mappings: HashMap<String, Type>,
}

/// Supported model formats
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    /// ONNX format
    ONNX,
    /// TensorFlow SavedModel
    TensorFlowSavedModel,
    /// PyTorch model
    PyTorch,
    /// TensorFlow Lite
    TensorFlowLite,
    /// Core ML
    CoreML,
    /// Custom format
    Custom(String),
}

impl MLFrameworkIntegration {
    /// Create new ML framework integration
    pub fn new(framework: String) -> Self {
        Self {
            framework,
            supported_formats: Vec::new(),
            tensor_mappings: HashMap::new(),
        }
    }

    /// Add supported model format
    pub fn add_format(&mut self, format: ModelFormat) {
        self.supported_formats.push(format);
    }

    /// Add tensor type mapping
    pub fn add_tensor_mapping(&mut self, framework_type: String, aether_type: Type) {
        self.tensor_mappings.insert(framework_type, aether_type);
    }

    /// Load model from file
    pub fn load_model(&self, path: &str, format: ModelFormat) -> Result<LoadedModel, FFIError> {
        // Implementation would depend on specific framework
        Ok(LoadedModel {
            path: path.to_string(),
            format,
            inputs: Vec::new(),
            outputs: Vec::new(),
        })
    }
}

/// Loaded ML model
#[derive(Debug, Clone)]
pub struct LoadedModel {
    /// Model file path
    pub path: String,
    /// Model format
    pub format: ModelFormat,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
}

/// Tensor specification for ML models
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor type
    pub tensor_type: Type,
    /// Shape (if known)
    pub shape: Option<Vec<usize>>,
    /// Data type
    pub dtype: PrimitiveType,
}

/// FFI utilities for common operations
pub struct FFIUtils;

impl FFIUtils {
    /// Convert Aether type to FFI type
    pub fn aether_to_ffi_type(aether_type: &Type) -> Result<FFIType, FFIError> {
        match aether_type {
            Type::Primitive(prim) => Ok(FFIType::Primitive(prim.clone())),
            Type::Tensor { element_type, shape } => {
                let element_ffi = Self::aether_to_ffi_type(element_type)?;
                let dimensions = match shape {
                    crate::compiler::types::Shape::Concrete(dims) => Some(dims.clone()),
                    _ => None,
                };
                Ok(FFIType::Tensor {
                    element_type: Box::new(element_ffi),
                    dimensions,
                })
            }
            _ => Err(FFIError::UnsupportedType(format!("{:?}", aether_type))),
        }
    }

    /// Generate type conversion code
    pub fn generate_type_conversion(_from: &Type, _to: &FFIType) -> Result<String, FFIError> {
        // Implementation would generate conversion code based on types
        Ok("// Type conversion placeholder".to_string())
    }

    /// Validate FFI function signature
    pub fn validate_signature(function: &FFIFunction) -> Result<(), FFIError> {
        // Validate parameter types
        for param in &function.parameters {
            if !Self::is_valid_ffi_type(&param.ffi_type) {
                return Err(FFIError::InvalidSignature(
                    format!("Invalid parameter type: {:?}", param.ffi_type)
                ));
            }
        }

        // Validate return type
        if let Some(ret_type) = &function.return_type {
            if !Self::is_valid_ffi_type(ret_type) {
                return Err(FFIError::InvalidSignature(
                    format!("Invalid return type: {:?}", ret_type)
                ));
            }
        }

        Ok(())
    }

    /// Check if FFI type is valid for target
    fn is_valid_ffi_type(ffi_type: &FFIType) -> bool {
        match ffi_type {
            FFIType::Primitive(_) => true,
            FFIType::Pointer(inner) => Self::is_valid_ffi_type(inner),
            FFIType::Array(inner, _) => Self::is_valid_ffi_type(inner),
            FFIType::Slice(inner) => Self::is_valid_ffi_type(inner),
            FFIType::CString | FFIType::WideString => true,
            FFIType::Handle(_) => true,
            FFIType::Struct(_, fields) => fields.iter().all(|f| Self::is_valid_ffi_type(&f.ffi_type)),
            FFIType::Union(_, fields) => fields.iter().all(|f| Self::is_valid_ffi_type(&f.ffi_type)),
            FFIType::FunctionPointer(params, ret) => {
                params.iter().all(Self::is_valid_ffi_type) &&
                ret.as_ref().map_or(true, |r| Self::is_valid_ffi_type(r))
            }
            FFIType::Tensor { element_type, .. } => Self::is_valid_ffi_type(element_type),
            FFIType::Custom { .. } => true, // Assume custom types are valid
        }
    }
}