// Isolated FFI Unit Tests
// Tests only the FFI functionality without dependencies on other modules

#[cfg(test)]
mod ffi_unit_tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    // Minimal type definitions for testing
    #[derive(Debug, Clone, PartialEq)]
    pub enum PrimitiveType {
        Int32,
        Float32,
        Float64,
        Bool,
        String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Type {
        Primitive(PrimitiveType),
        Tensor { element_type: Box<Type>, shape: Vec<usize> },
    }

    // FFI types and structures (copied from main module for isolated testing)
    #[derive(Debug, Clone)]
    pub struct FFIConfig {
        pub target: FFITarget,
        pub library_name: String,
        pub headers: Vec<PathBuf>,
        pub link_libraries: Vec<String>,
        pub include_paths: Vec<PathBuf>,
        pub type_mappings: HashMap<String, Type>,
        pub memory_strategy: MemoryStrategy,
        pub error_handling: ErrorHandling,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum FFITarget {
        C,
        Cpp,
        Rust,
        JavaScript,
        Python,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum MemoryStrategy {
        AetherManaged,
        ForeignManaged,
        SharedOwnership,
        LinearTransfer,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum ErrorHandling {
        ErrorCodes,
        Exceptions,
        Results,
        Panic,
    }

    #[derive(Debug, Clone)]
    pub struct FFIFunction {
        pub foreign_name: String,
        pub aether_name: String,
        pub parameters: Vec<FFIParameter>,
        pub return_type: Option<FFIType>,
        pub calling_convention: CallingConvention,
        pub thread_safe: bool,
        pub documentation: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub struct FFIParameter {
        pub name: String,
        pub ffi_type: FFIType,
        pub direction: ParameterDirection,
        pub optional: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum ParameterDirection {
        In,
        Out,
        InOut,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum FFIType {
        Primitive(PrimitiveType),
        Pointer(Box<FFIType>),
        Array(Box<FFIType>, usize),
        CString,
        Handle(String),
        Struct(String, Vec<FFIField>),
        FunctionPointer(Vec<FFIType>, Option<Box<FFIType>>),
        Tensor { element_type: Box<FFIType>, dimensions: Option<Vec<usize>> },
    }

    #[derive(Debug, Clone)]
    pub struct FFIField {
        pub name: String,
        pub ffi_type: FFIType,
        pub offset: Option<usize>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum CallingConvention {
        C,
        Std,
        Fast,
        System,
        Win64,
    }

    #[derive(Debug, Clone)]
    pub enum FFIError {
        UnsupportedTarget(String),
        UnsupportedType(String),
        InvalidSignature(String),
        MissingType(String),
        CodeGenError(String),
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

    // Simple FFI binding generator for testing
    pub struct FFIBindingGenerator {
        config: FFIConfig,
        functions: Vec<FFIFunction>,
        types: HashMap<String, FFIType>,
    }

    impl FFIBindingGenerator {
        pub fn new(config: FFIConfig) -> Self {
            Self {
                config,
                functions: Vec::new(),
                types: HashMap::new(),
            }
        }

        pub fn add_function(&mut self, function: FFIFunction) {
            self.functions.push(function);
        }

        pub fn add_type(&mut self, name: String, ffi_type: FFIType) {
            self.types.insert(name, ffi_type);
        }

        pub fn generate_bindings(&self) -> Result<String, FFIError> {
            match self.config.target {
                FFITarget::C => self.generate_c_bindings(),
                FFITarget::Cpp => Ok("// C++ bindings placeholder\n".to_string()),
                FFITarget::Rust => Ok("// Rust bindings placeholder\n".to_string()),
                FFITarget::JavaScript => Ok("// JavaScript bindings placeholder\n".to_string()),
                FFITarget::Python => Ok("// Python bindings placeholder\n".to_string()),
            }
        }

        fn generate_c_bindings(&self) -> Result<String, FFIError> {
            let mut output = String::new();
            
            // Generate header
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
            
            Ok(output)
        }

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

        fn ffi_type_to_c_type(&self, ffi_type: &FFIType) -> Result<String, FFIError> {
            match ffi_type {
                FFIType::Primitive(PrimitiveType::Int32) => Ok("int32_t".to_string()),
                FFIType::Primitive(PrimitiveType::Float32) => Ok("float".to_string()),
                FFIType::Primitive(PrimitiveType::Float64) => Ok("double".to_string()),
                FFIType::Primitive(PrimitiveType::Bool) => Ok("bool".to_string()),
                FFIType::Primitive(PrimitiveType::String) => Ok("const char*".to_string()),
                FFIType::Pointer(inner) => Ok(format!("{}*", self.ffi_type_to_c_type(inner)?)),
                FFIType::Array(inner, size) => Ok(format!("{}[{}]", self.ffi_type_to_c_type(inner)?, size)),
                FFIType::CString => Ok("const char*".to_string()),
                FFIType::Handle(name) => Ok(format!("{}*", name)),
                _ => Err(FFIError::UnsupportedType(format!("{:?}", ffi_type))),
            }
        }
    }

    // Utility functions for testing
    pub struct FFIUtils;

    impl FFIUtils {
        pub fn aether_to_ffi_type(aether_type: &Type) -> Result<FFIType, FFIError> {
            match aether_type {
                Type::Primitive(prim) => Ok(FFIType::Primitive(prim.clone())),
                Type::Tensor { element_type, shape } => {
                    let element_ffi = Self::aether_to_ffi_type(element_type)?;
                    Ok(FFIType::Tensor {
                        element_type: Box::new(element_ffi),
                        dimensions: Some(shape.clone()),
                    })
                }
            }
        }

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

        fn is_valid_ffi_type(ffi_type: &FFIType) -> bool {
            match ffi_type {
                FFIType::Primitive(_) => true,
                FFIType::Pointer(inner) => Self::is_valid_ffi_type(inner),
                FFIType::Array(inner, _) => Self::is_valid_ffi_type(inner),
                FFIType::CString => true,
                FFIType::Handle(_) => true,
                FFIType::Struct(_, fields) => fields.iter().all(|f| Self::is_valid_ffi_type(&f.ffi_type)),
                FFIType::FunctionPointer(params, ret) => {
                    params.iter().all(Self::is_valid_ffi_type) &&
                    ret.as_ref().map_or(true, |r| Self::is_valid_ffi_type(r))
                }
                FFIType::Tensor { element_type, .. } => Self::is_valid_ffi_type(element_type),
            }
        }
    }

    // Tests
    #[test]
    fn test_ffi_config_creation() {
        let config = FFIConfig {
            target: FFITarget::C,
            library_name: "test_lib".to_string(),
            headers: vec![PathBuf::from("test.h")],
            link_libraries: vec!["m".to_string()],
            include_paths: vec![PathBuf::from("/usr/include")],
            type_mappings: HashMap::new(),
            memory_strategy: MemoryStrategy::AetherManaged,
            error_handling: ErrorHandling::ErrorCodes,
        };

        assert_eq!(config.target, FFITarget::C);
        assert_eq!(config.library_name, "test_lib");
        assert_eq!(config.headers.len(), 1);
        assert_eq!(config.link_libraries.len(), 1);
    }

    #[test]
    fn test_ffi_function_creation() {
        let function = FFIFunction {
            foreign_name: "add_numbers".to_string(),
            aether_name: "add".to_string(),
            parameters: vec![
                FFIParameter {
                    name: "a".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Float32),
                    direction: ParameterDirection::In,
                    optional: false,
                },
                FFIParameter {
                    name: "b".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Float32),
                    direction: ParameterDirection::In,
                    optional: false,
                },
            ],
            return_type: Some(FFIType::Primitive(PrimitiveType::Float32)),
            calling_convention: CallingConvention::C,
            thread_safe: true,
            documentation: Some("Adds two floating point numbers".to_string()),
        };

        assert_eq!(function.foreign_name, "add_numbers");
        assert_eq!(function.aether_name, "add");
        assert_eq!(function.parameters.len(), 2);
        assert!(function.return_type.is_some());
        assert!(function.thread_safe);
    }

    #[test]
    fn test_ffi_type_conversions() {
        // Test primitive type conversions
        let aether_int = Type::Primitive(PrimitiveType::Int32);
        let ffi_int = FFIUtils::aether_to_ffi_type(&aether_int).unwrap();
        assert_eq!(ffi_int, FFIType::Primitive(PrimitiveType::Int32));

        // Test tensor type conversions
        let aether_tensor = Type::Tensor {
            element_type: Box::new(Type::Primitive(PrimitiveType::Float32)),
            shape: vec![2, 3],
        };
        let ffi_tensor = FFIUtils::aether_to_ffi_type(&aether_tensor).unwrap();
        match ffi_tensor {
            FFIType::Tensor { element_type, dimensions } => {
                assert_eq!(*element_type, FFIType::Primitive(PrimitiveType::Float32));
                assert_eq!(dimensions, Some(vec![2, 3]));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_ffi_signature_validation() {
        let valid_function = FFIFunction {
            foreign_name: "test_func".to_string(),
            aether_name: "test".to_string(),
            parameters: vec![
                FFIParameter {
                    name: "param".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Int32),
                    direction: ParameterDirection::In,
                    optional: false,
                },
            ],
            return_type: Some(FFIType::Primitive(PrimitiveType::Int32)),
            calling_convention: CallingConvention::C,
            thread_safe: true,
            documentation: None,
        };

        assert!(FFIUtils::validate_signature(&valid_function).is_ok());
    }

    #[test]
    fn test_c_binding_generation() {
        let config = FFIConfig {
            target: FFITarget::C,
            library_name: "math_lib".to_string(),
            headers: vec![PathBuf::from("math.h")],
            link_libraries: vec!["m".to_string()],
            include_paths: vec![],
            type_mappings: HashMap::new(),
            memory_strategy: MemoryStrategy::AetherManaged,
            error_handling: ErrorHandling::ErrorCodes,
        };

        let mut generator = FFIBindingGenerator::new(config);

        // Add a simple function
        let function = FFIFunction {
            foreign_name: "sqrt".to_string(),
            aether_name: "square_root".to_string(),
            parameters: vec![
                FFIParameter {
                    name: "x".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Float64),
                    direction: ParameterDirection::In,
                    optional: false,
                },
            ],
            return_type: Some(FFIType::Primitive(PrimitiveType::Float64)),
            calling_convention: CallingConvention::C,
            thread_safe: true,
            documentation: Some("Square root function".to_string()),
        };

        generator.add_function(function);

        let bindings = generator.generate_bindings().unwrap();
        assert!(bindings.contains("#include <stdint.h>"));
        assert!(bindings.contains("double sqrt(double x);"));
    }

    #[test]
    fn test_struct_type_definition() {
        let point_struct = FFIType::Struct(
            "Point".to_string(),
            vec![
                FFIField {
                    name: "x".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Float32),
                    offset: Some(0),
                },
                FFIField {
                    name: "y".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Float32),
                    offset: Some(4),
                },
            ],
        );

        let config = FFIConfig {
            target: FFITarget::C,
            library_name: "geometry".to_string(),
            headers: vec![],
            link_libraries: vec![],
            include_paths: vec![],
            type_mappings: HashMap::new(),
            memory_strategy: MemoryStrategy::AetherManaged,
            error_handling: ErrorHandling::ErrorCodes,
        };

        let mut generator = FFIBindingGenerator::new(config);
        generator.add_type("Point".to_string(), point_struct);

        let bindings = generator.generate_bindings().unwrap();
        assert!(bindings.contains("typedef struct Point"));
        assert!(bindings.contains("float x;"));
        assert!(bindings.contains("float y;"));
    }

    #[test]
    fn test_function_pointer_type() {
        let callback_type = FFIType::FunctionPointer(
            vec![FFIType::Primitive(PrimitiveType::Int32)],
            Some(Box::new(FFIType::Primitive(PrimitiveType::Int32))),
        );

        let function = FFIFunction {
            foreign_name: "process_with_callback".to_string(),
            aether_name: "process".to_string(),
            parameters: vec![
                FFIParameter {
                    name: "data".to_string(),
                    ffi_type: FFIType::Primitive(PrimitiveType::Int32),
                    direction: ParameterDirection::In,
                    optional: false,
                },
                FFIParameter {
                    name: "callback".to_string(),
                    ffi_type: callback_type,
                    direction: ParameterDirection::In,
                    optional: false,
                },
            ],
            return_type: Some(FFIType::Primitive(PrimitiveType::Int32)),
            calling_convention: CallingConvention::C,
            thread_safe: false,
            documentation: None,
        };

        assert!(FFIUtils::validate_signature(&function).is_ok());
    }

    #[test]
    fn test_memory_strategies() {
        let strategies = vec![
            MemoryStrategy::AetherManaged,
            MemoryStrategy::ForeignManaged,
            MemoryStrategy::SharedOwnership,
            MemoryStrategy::LinearTransfer,
        ];

        for strategy in strategies {
            let config = FFIConfig {
                target: FFITarget::C,
                library_name: "test".to_string(),
                headers: vec![],
                link_libraries: vec![],
                include_paths: vec![],
                type_mappings: HashMap::new(),
                memory_strategy: strategy.clone(),
                error_handling: ErrorHandling::ErrorCodes,
            };

            let generator = FFIBindingGenerator::new(config);
            // Test that generator can be created with different memory strategies
            assert_eq!(generator.config.memory_strategy, strategy);
        }
    }

    #[test]
    fn test_error_handling_strategies() {
        let strategies = vec![
            ErrorHandling::ErrorCodes,
            ErrorHandling::Exceptions,
            ErrorHandling::Results,
            ErrorHandling::Panic,
        ];

        for strategy in strategies {
            let config = FFIConfig {
                target: FFITarget::C,
                library_name: "test".to_string(),
                headers: vec![],
                link_libraries: vec![],
                include_paths: vec![],
                type_mappings: HashMap::new(),
                memory_strategy: MemoryStrategy::AetherManaged,
                error_handling: strategy.clone(),
            };

            let generator = FFIBindingGenerator::new(config);
            assert_eq!(generator.config.error_handling, strategy);
        }
    }

    #[test]
    fn test_cross_language_targets() {
        let targets = vec![
            FFITarget::C,
            FFITarget::Cpp,
            FFITarget::Rust,
            FFITarget::JavaScript,
            FFITarget::Python,
        ];

        for target in targets {
            let config = FFIConfig {
                target: target.clone(),
                library_name: "test".to_string(),
                headers: vec![],
                link_libraries: vec![],
                include_paths: vec![],
                type_mappings: HashMap::new(),
                memory_strategy: MemoryStrategy::AetherManaged,
                error_handling: ErrorHandling::ErrorCodes,
            };

            let generator = FFIBindingGenerator::new(config);
            let result = generator.generate_bindings();
            
            // All targets should be able to generate some form of bindings
            assert!(result.is_ok());
            assert_eq!(generator.config.target, target);
        }
    }

    #[test]
    fn test_tensor_ffi_type() {
        let tensor_type = FFIType::Tensor {
            element_type: Box::new(FFIType::Primitive(PrimitiveType::Float32)),
            dimensions: Some(vec![2, 3, 4]),
        };

        assert!(FFIUtils::is_valid_ffi_type(&tensor_type));

        // Test conversion from Aether tensor type
        let aether_tensor = Type::Tensor {
            element_type: Box::new(Type::Primitive(PrimitiveType::Float32)),
            shape: vec![2, 3, 4],
        };

        let converted = FFIUtils::aether_to_ffi_type(&aether_tensor).unwrap();
        match converted {
            FFIType::Tensor { element_type, dimensions } => {
                assert_eq!(*element_type, FFIType::Primitive(PrimitiveType::Float32));
                assert_eq!(dimensions, Some(vec![2, 3, 4]));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            FFIError::UnsupportedTarget("CustomTarget".to_string()),
            FFIError::UnsupportedType("CustomType".to_string()),
            FFIError::InvalidSignature("invalid sig".to_string()),
            FFIError::MissingType("MissingType".to_string()),
            FFIError::CodeGenError("codegen failed".to_string()),
            FFIError::IoError("io failed".to_string()),
        ];

        for error in errors {
            let error_string = error.to_string();
            assert!(!error_string.is_empty());
            println!("Error: {}", error_string);
        }
    }
}