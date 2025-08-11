// Build configuration management for Aether compiler

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Build configuration for the Aether compiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub rust_toolchain: ToolchainConfig,
    pub aether_config: AetherConfig,
    pub target_config: TargetConfig,
    pub optimization_config: OptimizationConfig,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            rust_toolchain: ToolchainConfig::default(),
            aether_config: AetherConfig::default(),
            target_config: TargetConfig::default(),
            optimization_config: OptimizationConfig::default(),
        }
    }
}

/// Rust toolchain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainConfig {
    pub version: String,
    pub features: Vec<String>,
    pub dependencies: Vec<Dependency>,
}

impl Default for ToolchainConfig {
    fn default() -> Self {
        Self {
            version: "stable".to_string(),
            features: vec!["mlir".to_string()],
            dependencies: vec![
                Dependency {
                    name: "melior".to_string(),
                    version: "0.19".to_string(),
                    optional: true,
                },
                Dependency {
                    name: "mlir-sys".to_string(),
                    version: "0.3".to_string(),
                    optional: true,
                },
                Dependency {
                    name: "tempfile".to_string(),
                    version: "3.8".to_string(),
                    optional: false,
                },
            ],
        }
    }
}

/// Aether compiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherConfig {
    pub compiler_path: Option<PathBuf>,
    pub source_files: Vec<PathBuf>,
    pub output_directory: PathBuf,
}

impl Default for AetherConfig {
    fn default() -> Self {
        Self {
            compiler_path: None,
            source_files: Vec::new(),
            output_directory: PathBuf::from("target"),
        }
    }
}

/// Target platform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    pub platform: Platform,
    pub architecture: Architecture,
    pub executable_format: ExecutableFormat,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            platform: Platform::Windows,
            architecture: Architecture::X86_64,
            executable_format: ExecutableFormat::PE,
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub level: OptimizationLevel,
    pub enable_mlir_optimizations: bool,
    pub enable_gpu_optimizations: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Debug,
            enable_mlir_optimizations: true,
            enable_gpu_optimizations: false,
        }
    }
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub optional: bool,
}

/// Target platform
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    WebAssembly,
}

/// Target architecture
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Wasm32,
}

/// Executable format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutableFormat {
    PE,    // Windows
    ELF,   // Linux
    MachO, // macOS
    Wasm,  // WebAssembly
}

/// Optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,
    Release,
    ReleaseLTO,
}