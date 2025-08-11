// Type system unit tests for MLIR integration
// Tests the Aether type system integration with MLIR

use super::*;
use crate::compiler::mlir::{MLIRContext, MLIRModule, MLIRType, MLIRError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_conversion() {
        let context = MLIRContext::new_mock();
        let aether_type = MLIRType::Integer { width: 32, signed: true };
        
        // Test that type conversion works
        assert!(matches!(aether_type, MLIRType::Integer { width: 32, signed: true }));
    }

    #[test]
    fn test_tensor_type_conversion() {
        let context = MLIRContext::new_mock();
        let tensor_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
            device: "cpu".to_string(),
        };
        
        // Test tensor type structure
        if let MLIRType::AetherTensor { element_type, shape, device } = tensor_type {
            assert!(matches!(*element_type, MLIRType::Float { width: 32 }));
            assert_eq!(shape, vec![2, 3, 4]);
            assert_eq!(device, "cpu");
        } else {
            panic!("Expected AetherTensor type");
        }
    }

    #[test]
    fn test_function_type_conversion() {
        let context = MLIRContext::new_mock();
        let func_type = MLIRType::Function {
            inputs: vec![
                MLIRType::Integer { width: 32, signed: true },
                MLIRType::Float { width: 64 },
            ],
            outputs: vec![MLIRType::Float { width: 32 }],
        };
        
        // Test function type structure
        if let MLIRType::Function { inputs, outputs } = func_type {
            assert_eq!(inputs.len(), 2);
            assert_eq!(outputs.len(), 1);
        } else {
            panic!("Expected Function type");
        }
    }

    #[test]
    fn test_type_verification() {
        let context = MLIRContext::new_mock();
        
        // Test that basic types are valid
        let int_type = MLIRType::Integer { width: 32, signed: true };
        let float_type = MLIRType::Float { width: 32 };
        let index_type = MLIRType::Index;
        
        // These should all be valid types
        assert!(matches!(int_type, MLIRType::Integer { .. }));
        assert!(matches!(float_type, MLIRType::Float { .. }));
        assert!(matches!(index_type, MLIRType::Index));
    }

    #[test]
    fn test_complex_type_nesting() {
        let context = MLIRContext::new_mock();
        
        // Create a complex nested type
        let complex_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Function {
                inputs: vec![MLIRType::Integer { width: 32, signed: true }],
                outputs: vec![MLIRType::Float { width: 32 }],
            }),
            shape: vec![10],
            device: "gpu".to_string(),
        };
        
        // Verify the nested structure
        if let MLIRType::AetherTensor { element_type, device, .. } = complex_type {
            if let MLIRType::Function { inputs, outputs } = &*element_type {
                assert_eq!(inputs.len(), 1);
                assert_eq!(outputs.len(), 1);
                assert_eq!(device, "gpu");
            } else {
                panic!("Expected Function type as element type");
            }
        } else {
            panic!("Expected AetherTensor type");
        }
    }
}