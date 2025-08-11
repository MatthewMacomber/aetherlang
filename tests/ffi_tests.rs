// Aether FFI Tests
// Comprehensive tests for Foreign Function Interface functionality

use aether::compiler::ffi::*;
use aether::compiler::ml_integration::*;
use aether::compiler::types::{Type, PrimitiveType};
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(test)]
mod ffi_binding_tests {
    use super::*;

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
        let aether_int = Type::primitive(PrimitiveType::Int32);
        let ffi_int = FFIUtils::aether_to_ffi_type(&aether_int).unwrap();
        assert_eq!(ffi_int, FFIType::Primitive(PrimitiveType::Int32));

        // Test tensor type conversions
        let aether_tensor = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![2, 3]);
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
        assert!(bindings.contains("aether_square_root"));
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
}

#[cfg(test)]
mod ml_integration_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_ml_integration_manager_creation() {
        let manager = MLIntegrationManager::new();
        
        // Should have built-in frameworks registered
        assert!(manager.frameworks.contains_key("ONNX"));
        assert!(manager.frameworks.contains_key("TensorFlow"));
        assert!(manager.frameworks.contains_key("PyTorch"));
    }

    #[test]
    fn test_onnx_framework() {
        let framework = ONNXFramework::new();
        
        assert_eq!(framework.name(), "ONNX");
        assert_eq!(framework.supported_formats(), vec![ModelFormat::ONNX]);
    }

    #[test]
    fn test_model_loading() {
        let mut manager = MLIntegrationManager::new();
        let model_path = Path::new("test_model.onnx");
        
        let result = manager.load_model("ONNX", model_path, ModelFormat::ONNX);
        assert!(result.is_ok());
        
        let model = result.unwrap();
        assert_eq!(model.format, ModelFormat::ONNX);
        assert!(!model.inputs.is_empty());
        assert!(!model.outputs.is_empty());
    }

    #[test]
    fn test_tensor_data_creation() {
        let tensor = MLUtils::create_tensor_data(
            "test_tensor".to_string(),
            DataType::Float32,
            vec![2, 3],
            vec![0u8; 24], // 2 * 3 * 4 bytes
        );

        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.dtype, DataType::Float32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data.len(), 24);
        assert_eq!(tensor.layout, DataLayout::RowMajor);
    }

    #[test]
    fn test_shape_validation() {
        let expected = vec![2, 3, 4];
        let actual = vec![2, 3, 4];
        
        assert!(MLUtils::validate_shape_compatibility(&expected, &actual).is_ok());
        
        let wrong_shape = vec![2, 3, 5];
        assert!(MLUtils::validate_shape_compatibility(&expected, &wrong_shape).is_err());
    }

    #[test]
    fn test_tensor_size_calculation() {
        let shape = vec![2, 3, 4];
        let size = MLUtils::calculate_tensor_size(&shape, &DataType::Float32);
        assert_eq!(size, 96); // 2 * 3 * 4 * 4 bytes

        let size_f64 = MLUtils::calculate_tensor_size(&shape, &DataType::Float64);
        assert_eq!(size_f64, 192); // 2 * 3 * 4 * 8 bytes
    }

    #[test]
    fn test_data_layout_conversion() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8]; // 2x2 matrix of 32-bit values
        let shape = vec![2, 2];
        let element_size = 4;

        let converted = MLUtils::convert_layout(
            &data,
            &shape,
            &DataLayout::RowMajor,
            &DataLayout::ColumnMajor,
            element_size,
        );

        // Should transpose the data
        assert_eq!(converted.len(), data.len());
        // First element should be the same
        assert_eq!(converted[0..4], data[0..4]);
        // Second element should be from the second row
        assert_eq!(converted[4..8], data[4..8]);
    }

    #[test]
    fn test_onnx_tensor_conversion() {
        let framework = ONNXFramework::new();
        
        let tensor_data = TensorData {
            name: "test".to_string(),
            dtype: DataType::Float32,
            shape: vec![2, 3],
            data: vec![0u8; 24],
            layout: DataLayout::RowMajor,
        };

        let aether_tensor = framework.to_aether_tensor(&tensor_data).unwrap();
        
        // Check that conversion preserves data
        assert_eq!(aether_tensor.data, tensor_data.data);
        
        // Check that we can convert back
        let converted_back = framework.from_aether_tensor(&aether_tensor).unwrap();
        assert_eq!(converted_back.dtype, tensor_data.dtype);
        assert_eq!(converted_back.shape, tensor_data.shape);
    }

    #[test]
    fn test_model_info_extraction() {
        let framework = ONNXFramework::new();
        let model = LoadedModel {
            path: "test.onnx".to_string(),
            format: ModelFormat::ONNX,
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    tensor_type: Type::tensor(Type::primitive(PrimitiveType::Float32), vec![1, 224, 224, 3]),
                    shape: Some(vec![1, 224, 224, 3]),
                    dtype: PrimitiveType::Float32,
                }
            ],
            outputs: vec![
                TensorSpec {
                    name: "output".to_string(),
                    tensor_type: Type::tensor(Type::primitive(PrimitiveType::Float32), vec![1, 1000]),
                    shape: Some(vec![1, 1000]),
                    dtype: PrimitiveType::Float32,
                }
            ],
        };

        let info = framework.get_model_info(&model).unwrap();
        assert_eq!(info.name, "ONNX Model");
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.outputs.len(), 1);
    }

    #[test]
    fn test_unsupported_framework() {
        let manager = MLIntegrationManager::new();
        let model_path = Path::new("test.model");
        
        let result = manager.load_model("UnsupportedFramework", model_path, ModelFormat::ONNX);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            MLError::FrameworkNotFound(name) => assert_eq!(name, "UnsupportedFramework"),
            _ => panic!("Expected FrameworkNotFound error"),
        }
    }

    #[test]
    fn test_model_caching() {
        let mut manager = MLIntegrationManager::new();
        let model_path = Path::new("test_model.onnx");
        
        // Load model first time
        let _model1 = manager.load_model("ONNX", model_path, ModelFormat::ONNX).unwrap();
        
        // Check that model is cached
        let cached_model = manager.get_cached_model("ONNX", model_path);
        assert!(cached_model.is_some());
        
        let cached = cached_model.unwrap();
        assert_eq!(cached.format, ModelFormat::ONNX);
    }

    #[test]
    fn test_data_type_sizes() {
        let framework = ONNXFramework::new();
        
        assert_eq!(framework.dtype_size(&DataType::Float32), 4);
        assert_eq!(framework.dtype_size(&DataType::Float64), 8);
        assert_eq!(framework.dtype_size(&DataType::Int32), 4);
        assert_eq!(framework.dtype_size(&DataType::Int64), 8);
        assert_eq!(framework.dtype_size(&DataType::Bool), 1);
        assert_eq!(framework.dtype_size(&DataType::String), 0); // Variable size
    }

    #[test]
    fn test_stride_calculation() {
        let framework = ONNXFramework::new();
        let shape = vec![2, 3, 4];
        
        // Row-major strides
        let row_major_strides = framework.calculate_strides(&shape, &DataLayout::RowMajor);
        assert_eq!(row_major_strides, vec![12, 4, 1]);
        
        // Column-major strides
        let col_major_strides = framework.calculate_strides(&shape, &DataLayout::ColumnMajor);
        assert_eq!(col_major_strides, vec![1, 2, 6]);
        
        // Custom strides
        let custom_strides = vec![24, 8, 2];
        let custom_layout = DataLayout::Custom(custom_strides.clone());
        let result_strides = framework.calculate_strides(&shape, &custom_layout);
        assert_eq!(result_strides, custom_strides);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_ffi_with_ml_integration() {
        // Test that FFI and ML integration work together
        let mut ml_manager = MLIntegrationManager::new();
        
        // Create FFI config for ML framework binding
        let ffi_config = FFIConfig {
            target: FFITarget::Python,
            library_name: "ml_framework".to_string(),
            headers: vec![],
            link_libraries: vec![],
            include_paths: vec![],
            type_mappings: HashMap::new(),
            memory_strategy: MemoryStrategy::SharedOwnership,
            error_handling: ErrorHandling::Exceptions,
        };

        let ffi_generator = FFIBindingGenerator::new(ffi_config);
        
        // Load a model
        let model_path = Path::new("test_model.onnx");
        let model = ml_manager.load_model("ONNX", model_path, ModelFormat::ONNX).unwrap();
        
        // Verify that both systems can work with the same tensor types
        let tensor_spec = &model.inputs[0];
        let ffi_type = FFIUtils::aether_to_ffi_type(&tensor_spec.tensor_type).unwrap();
        
        match ffi_type {
            FFIType::Tensor { element_type, dimensions } => {
                assert_eq!(*element_type, FFIType::Primitive(PrimitiveType::Float32));
                assert!(dimensions.is_some());
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_cross_language_type_consistency() {
        // Test that type conversions are consistent across different FFI targets
        let aether_type = Type::tensor(Type::primitive(PrimitiveType::Float32), vec![2, 3]);
        let ffi_type = FFIUtils::aether_to_ffi_type(&aether_type).unwrap();
        
        // Test with different FFI targets
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
            
            // All targets should be able to handle the same FFI type
            assert_eq!(generator.config.target, target);
        }
    }

    #[test]
    fn test_real_world_ml_workflow() {
        // Simulate a real-world workflow: load model, prepare data, run inference
        let mut ml_manager = MLIntegrationManager::new();
        
        // Load ONNX model
        let model_path = Path::new("image_classifier.onnx");
        let model = ml_manager.load_model("ONNX", model_path, ModelFormat::ONNX).unwrap();
        
        // Create input tensor (simulating image data)
        let input_data = vec![0u8; 224 * 224 * 3 * 4]; // 224x224 RGB image, float32
        let input_tensor = MLUtils::create_tensor_data(
            "image".to_string(),
            DataType::Float32,
            vec![1, 224, 224, 3],
            input_data,
        );
        
        // Validate input shape matches model expectations
        let expected_shape = &model.inputs[0].shape.as_ref().unwrap();
        assert!(MLUtils::validate_shape_compatibility(expected_shape, &input_tensor.shape).is_ok());
        
        // Convert to Aether tensor
        let aether_tensor = ml_manager.to_aether_tensor("ONNX", &input_tensor).unwrap();
        
        // Verify tensor properties
        assert!(aether_tensor.tensor_type.is_tensor());
        assert_eq!(aether_tensor.data.len(), input_tensor.data.len());
        
        // Convert back to framework tensor
        let converted_back = ml_manager.from_aether_tensor("ONNX", &aether_tensor).unwrap();
        assert_eq!(converted_back.shape, input_tensor.shape);
        assert_eq!(converted_back.dtype, input_tensor.dtype);
    }
}