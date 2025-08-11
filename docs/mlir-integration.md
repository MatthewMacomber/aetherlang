# MLIR Integration for Aether

This document describes the MLIR (Multi-Level Intermediate Representation) integration in the Aether programming language compiler.

## Overview

The Aether compiler includes MLIR integration to provide a robust, production-ready compilation pipeline that converts Aether's tokenized AST through MLIR dialects to optimized native executables via LLVM.

## Features

- **Real MLIR-C API Bindings**: Rust bindings to the MLIR C API using safe wrappers
- **Build System Integration**: Proper linking against MLIR and LLVM libraries
- **Stub Implementation**: Fallback implementation when MLIR is not available
- **Comprehensive Testing**: Integration tests to verify MLIR functionality

## Build Configuration

### With MLIR Support

To build with real MLIR support, you need to have MLIR/LLVM 18.x installed:

```bash
# Enable MLIR feature
cargo build --features mlir

# Run tests with MLIR
cargo test --features mlir
```

### Environment Variables

Set these environment variables to help the build system find MLIR:

```bash
# For MLIR/LLVM 18.x
export MLIR_SYS_180_PREFIX=/path/to/mlir/installation
export LLVM_SYS_180_PREFIX=/path/to/llvm/installation
```

### Without MLIR (Stub Mode)

The default build uses stub implementations:

```bash
# Build without MLIR (default)
cargo build

# Run tests in stub mode
cargo test
```

## API Usage

### Basic MLIR Context

```rust
use aether_language::compiler::mlir::{MLIRContext, MLIRPipeline};

// Create MLIR context
let context = MLIRContext::new()?;

// Check if dialects are registered
assert!(context.is_dialect_registered("builtin"));
assert!(context.is_dialect_registered("func"));

// Create a module
let module = context.create_module("my_module")?;

// Verify the module
module.verify()?;
```

### MLIR Pipeline

```rust
use aether_language::compiler::mlir::MLIRPipeline;
use aether_language::compiler::ast::AST;

// Create pipeline
let pipeline = MLIRPipeline::new()?;

// Compile AST to MLIR
let ast = /* your AST */;
let mlir_module = pipeline.compile_ast(&ast)?;

// Apply lowering passes
pipeline.lower_to_standard(&mut mlir_module)?;
```

### Safe MLIR Bindings

```rust
use aether_language::compiler::mlir::{SafeMlirContext, SafeMlirModule};

// Create safe context
let context = SafeMlirContext::new()?;

// Create types
let i32_type = context.create_i32_type();
let f64_type = context.create_f64_type();
let tensor_type = context.create_tensor_type(&[2, 3, 4], f32_type);

// Create attributes
let string_attr = context.create_string_attr("example");
let bool_attr = context.create_bool_attr(true);

// Create locations
let unknown_loc = context.create_unknown_location();
let file_loc = context.create_file_location("example.ae", 10, 5);
```

## Testing

### Running MLIR Tests

```bash
# Run all MLIR integration tests
cargo test mlir_integration_tests -- --nocapture

# Run simple MLIR tests
cargo test --test simple_mlir_test -- --nocapture

# Run specific test
cargo test test_mlir_context_creation -- --nocapture
```

### Test Coverage

The integration includes tests for:

- MLIR context creation and destruction
- Dialect registration and loading
- Module creation and verification
- Type and attribute creation
- Error handling and diagnostics
- Performance and memory usage

## Architecture

### Components

1. **MLIR Bindings** (`src/compiler/mlir/bindings.rs`)
   - Safe Rust wrappers around MLIR-C API
   - Automatic resource management
   - Stub implementations for fallback

2. **MLIR Context** (`src/compiler/mlir/mlir_context.rs`)
   - High-level MLIR context management
   - Dialect registration
   - Module creation and verification

3. **Build System** (`build.rs`)
   - MLIR/LLVM library detection and linking
   - Cross-platform build support
   - Environment variable configuration

### Error Handling

The MLIR integration provides comprehensive error handling:

```rust
use aether_language::compiler::mlir::MLIRError;

match result {
    Ok(value) => { /* success */ },
    Err(MLIRError::ContextCreationError(msg)) => { /* handle context error */ },
    Err(MLIRError::DialectError(msg)) => { /* handle dialect error */ },
    Err(MLIRError::ModuleError(msg)) => { /* handle module error */ },
    Err(MLIRError::VerificationError(msg)) => { /* handle verification error */ },
    // ... other error types
}
```

## Installation

### Prerequisites

For real MLIR support, you need:

- MLIR/LLVM 18.x development libraries
- C++ compiler (for linking)
- CMake (for MLIR build detection)

### Platform-Specific Instructions

#### Linux/macOS

```bash
# Install LLVM/MLIR via package manager
# Ubuntu/Debian:
sudo apt-get install llvm-18-dev mlir-18-dev

# macOS with Homebrew:
brew install llvm@18

# Set environment variables
export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
export LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
```

#### Windows

```bash
# Download LLVM/MLIR pre-built binaries
# Set environment variables
set MLIR_SYS_180_PREFIX=C:\path\to\llvm
set LLVM_SYS_180_PREFIX=C:\path\to\llvm
```

## Troubleshooting

### Common Issues

1. **MLIR not found**: Set `MLIR_SYS_180_PREFIX` environment variable
2. **Linking errors**: Ensure all required system libraries are installed
3. **Version mismatch**: Use MLIR/LLVM version 18.x

### Debug Information

Enable debug output during build:

```bash
RUST_LOG=debug cargo build --features mlir
```

### Fallback to Stub Mode

If MLIR is not available, the build automatically falls back to stub mode:

```rust
use aether_language::compiler::mlir::{is_mlir_available, get_mlir_version};

if !is_mlir_available() {
    println!("Running in stub mode: {}", get_mlir_version());
}
```

## Future Enhancements

- Support for additional MLIR dialects
- Custom Aether dialect implementation
- GPU compilation through MLIR
- Advanced optimization passes
- WebAssembly target support through MLIR

## Contributing

When contributing to MLIR integration:

1. Ensure tests pass in both MLIR and stub modes
2. Add appropriate error handling
3. Update documentation for new features
4. Test on multiple platforms
5. Follow Rust safety guidelines for FFI code