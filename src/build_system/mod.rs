// Build system management for Aether compiler
// Provides orchestration, error handling, and environment validation

pub mod manager;
pub mod config;
pub mod environment;
pub mod error_handler;
pub mod fix_engine;
pub mod rust_compiler;
pub mod aether_compiler;
pub mod workflow;
pub mod windows;

pub use manager::{BuildSystemManager, CompilerBinary, Executable, TestResults, BuildSystemError, CompilationResult};
pub use config::*;
pub use environment::*;
pub use error_handler::*;
pub use fix_engine::*;
pub use rust_compiler::{RustCompiler, ToolchainInfo};
pub use aether_compiler::*;
pub use workflow::*;
pub use windows::*;

// Re-export TestRunner from testing module
pub use crate::testing::test_runner::TestRunner;