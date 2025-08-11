// Aether Runtime Module
// Handles memory management, execution, and system interfaces

pub mod tensor;
pub mod tensor_ops;
pub mod memory_layout;
pub mod native_runtime;
pub mod error_handling;
pub mod runtime_library;
pub mod llvm_runtime_declarations;
pub mod runtime_linking;
pub mod executable_linker;

#[cfg(test)]
pub mod runtime_tests;

pub use tensor::*;
pub use tensor_ops::*;
pub use memory_layout::*;
pub use native_runtime::*;
pub use error_handling::*;
pub use runtime_library::*;
pub use llvm_runtime_declarations::*;
pub use runtime_linking::*;
pub use executable_linker::*;