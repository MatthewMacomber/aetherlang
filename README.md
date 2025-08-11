# Aether Programming Language - Compiler & Tools

Aether is a next-generation programming language designed from first principles for an era where Artificial Intelligence is a primary user and collaborator in software development. This README covers the complete toolchain for building, testing, and deploying Aether programs.

## Overview

Aether solves the "two-language problem" in AI development by providing a unified, high-performance environment that bridges research prototyping and production deployment. The language features:

- **AI-Native Design**: Tokenized AST as canonical representation, optimized for LLM generation
- **Performance Without Compromise**: Eliminates Python's bottlenecks while maintaining research-friendly flexibility  
- **Safety by Design**: Advanced type system prevents common AI bugs at compile time
- **Universal Deployment**: Single codebase compiles to native, WebAssembly, GPU kernels, and mobile targets

## Quick Start

### Prerequisites

- Rust 1.70+ (for building the compiler)
- LLVM/MLIR 18.0+ (optional, for advanced compilation features)
- Git (for version control)

### Installation

```bash
# Clone the repository
git clone https://github.com/aether-lang/aether
cd aether

# Build the compiler and tools
make build

# Or build with Cargo directly
cargo build --release

# Install system-wide (optional)
make install
```

### Hello World

Create a file `hello.ae`:

```lisp
# Hello World Aether Program
(func main ()
  (call print "Hello, World from Aether!")
  (return 0))
```

Compile and run:

```bash
# Compile to native executable
aetherc build hello.ae --target native

# Run the program
./hello
```

## Core Tools

### 1. `aetherc` - The Aether Compiler

The main compiler that transforms Aether source code into various target formats.

#### Basic Usage

```bash
# Compile to native executable
aetherc build input.ae --target native

# Compile to WebAssembly
aetherc build input.ae --target wasm32

# Compile with GPU acceleration
aetherc build input.ae --target gpu --device cuda

# Compile with optimizations
aetherc build input.ae --target native --optimize --release
```

#### Compilation Targets

- **Native**: `--target native` - Produces platform-specific executables (ELF/Mach-O/PE)
- **WebAssembly**: `--target wasm32` - Browser-compatible WebAssembly modules
- **GPU**: `--target gpu --device cuda|opencl|vulkan` - GPU compute kernels
- **Mobile**: `--target mobile --platform ios|android` - Mobile-optimized bytecode
- **Cloud**: `--target cloud` - Microservice-ready binaries with RPC stubs

#### Advanced Options

```bash
# Enable debug information
aetherc build input.ae --debug --symbols

# Cross-compilation
aetherc build input.ae --target native --arch x86_64 --os linux

# Link external libraries
aetherc build input.ae --link-lib math --link-lib pthread

# Generate intermediate representations
aetherc build input.ae --emit-llvm --emit-mlir --emit-ast

# Profile-guided optimization
aetherc build input.ae --pgo-profile profile.data
```

### 2. `aetherfmt` - Code Formatter & Transpiler

Bidirectional transpiler that converts between human-readable and machine-optimized syntax.

#### Usage

```bash
# Format to human-readable syntax
aetherfmt --to-sweet input.ae

# Format to canonical S-expressions
aetherfmt --to-canonical input.ae

# Format in-place
aetherfmt --in-place *.ae

# Check formatting without changes
aetherfmt --check src/
```

#### Format Examples

**Human-readable "sweet" syntax:**
```aether
func fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}
```

**Canonical S-expression syntax:**
```lisp
(func fibonacci ((n i32)) i32
  (if (<= n 1)
    (return n)
    (return (+ (call fibonacci (- n 1))
               (call fibonacci (- n 2))))))
```

### 3. `aether-analyze` - Static Analyzer

Comprehensive static analysis tool for performance, correctness, and optimization insights.

#### Usage

```bash
# Full analysis suite
aether-analyze input.ae

# Memory analysis
aether-analyze --memory input.ae

# Tensor shape analysis
aether-analyze --shapes input.ae

# GPU occupancy analysis
aether-analyze --gpu-occupancy input.ae

# Performance bottleneck detection
aether-analyze --performance input.ae

# Generate analysis report
aether-analyze input.ae --output report.json --format json
```

#### Analysis Categories

- **Memory Safety**: Detects use-after-free, memory leaks, buffer overflows
- **Tensor Shapes**: Validates tensor operations and shape compatibility
- **GPU Utilization**: Analyzes GPU memory usage and kernel efficiency
- **Performance**: Identifies bottlenecks and optimization opportunities
- **Concurrency**: Detects race conditions and deadlocks
- **AI Model Validation**: Verifies neural network architectures

### 4. `aether-playground` - Interactive REPL

Browser-based interactive development environment for experimentation and learning.

#### Usage

```bash
# Start local playground server
aether-playground --port 8080

# Open in browser automatically
aether-playground --open

# Enable GPU acceleration in playground
aether-playground --gpu --device cuda

# Load project context
aether-playground --project ./my-project
```

#### Features

- Real-time compilation and execution
- Interactive tensor visualization
- GPU kernel profiling
- Model architecture visualization
- Collaborative editing support
- Export to various formats

### 5. `aether-test` - Test Runner

Comprehensive testing framework for Aether programs.

#### Usage

```bash
# Run all tests
aether-test

# Run specific test file
aether-test tests/math_test.ae

# Run tests with coverage
aether-test --coverage

# Run performance benchmarks
aether-test --benchmark

# Generate test report
aether-test --report html --output test-results/
```

### 6. `aether-file-test` - File Compilation Testing

Specialized tool for testing file compilation across the entire project.

#### Usage

```bash
# Run all file compilation tests
aether-file-test run

# Run with custom configuration
aether-file-test run --config custom-config.toml

# Generate additional test cases
aether-file-test generate --categories core,types,ai

# Validate test setup
aether-file-test validate

# Show project statistics
aether-file-test stats
```

#### Configuration (`aether-file-test.toml`)

```toml
project_root = "."
compiler_path = "./target/release/aetherc.exe"
output_directory = "target/file_compilation_tests"
test_directories = ["examples", "tests"]
generate_additional_tests = true
test_categories = ["CoreLanguage", "TypeSystem", "AIFeatures", "ErrorHandling"]
report_format = "Console"
max_parallel_compilations = 4
verbose = true

[compilation_timeout]
secs = 30
nanos = 0
```

## Build System Integration

### Make Targets

The project includes a comprehensive Makefile for common development tasks:

```bash
# Build everything
make build

# Run all tests
make test

# Run only file compilation tests
make test-files

# Clean build artifacts
make clean

# Install tools system-wide
make install

# Development mode with file watching
make watch

# Generate documentation
make docs

# Run benchmarks
make benchmark
```

### Cargo Integration

Standard Rust build commands work seamlessly:

```bash
# Build all binaries
cargo build --release

# Build specific tool
cargo build --bin aetherc --release

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Run benchmarks
cargo bench
```

## Language Features & Examples

### Basic Syntax

Aether uses S-expressions as its canonical form, with optional human-readable syntax:

```lisp
# Variables and functions
(let x 42)
(let y (+ x 8))

(func greet (name)
  (call print "Hello," name))

# Control flow
(if (> x 0)
  (call print "Positive")
  (call print "Non-positive"))

# Loops
(for i (range 10)
  (call print "Iteration" i))
```

### AI-Native Features

```lisp
# Tensor operations
(let matrix (tensor [[1 2] [3 4]]))
(let result (matmul matrix (transpose matrix)))

# Neural network definition
(model SimpleNet
  (layer dense 784 512 relu)
  (layer dense 512 256 relu)
  (layer dense 256 10 softmax))

# Automatic differentiation
(let loss (mse-loss predictions targets))
(let gradients (backward loss))
```

### Concurrency & Distribution

```lisp
# Actor-based concurrency
(actor DataProcessor
  (state buffer [])
  (handler process-data (data)
    (set! buffer (append buffer data))
    (when (> (length buffer) 1000)
      (call flush-buffer))))

# Distributed computation
(distributed-map process-chunk data-chunks
  :nodes 4
  :gpu-enabled true)
```

## Advanced Usage

### Custom Compilation Pipelines

```bash
# Multi-stage compilation with custom passes
aetherc build input.ae \
  --pass tensor-fusion \
  --pass gpu-optimization \
  --pass dead-code-elimination \
  --target gpu

# Profile-guided optimization workflow
aetherc build input.ae --profile-generate
./input  # Run with representative data
aetherc build input.ae --profile-use=profile.data --optimize
```

### Integration with External Tools

```bash
# Export to ONNX format
aetherc export model.ae --format onnx --output model.onnx

# Generate C++ bindings
aetherc bind model.ae --language cpp --output bindings/

# Create Docker deployment
aetherc deploy model.ae --target docker --optimize-size
```

### Development Workflow

```bash
# Watch mode for development
aether-playground --watch src/

# Continuous testing
make watch  # Runs tests on file changes

# Performance profiling
aether-analyze --profile input.ae
aether-test --benchmark --compare-baseline
```

## Troubleshooting

### Common Issues

1. **MLIR/LLVM not found**
   ```bash
   # Install LLVM/MLIR
   # Ubuntu/Debian:
   sudo apt install llvm-18-dev mlir-18-tools
   
   # macOS:
   brew install llvm@18
   
   # Set environment variable
   export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
   ```

2. **Compilation timeout**
   ```bash
   # Increase timeout in configuration
   aether-file-test run --compilation-timeout 60
   ```

3. **GPU compilation issues**
   ```bash
   # Check GPU drivers and CUDA installation
   nvidia-smi  # For NVIDIA GPUs
   
   # Use CPU fallback
   aetherc build input.ae --target native --no-gpu
   ```

### Debug Mode

```bash
# Enable verbose logging
AETHER_LOG=debug aetherc build input.ae

# Generate debug symbols
aetherc build input.ae --debug --symbols

# Use debug build of compiler
cargo build --bin aetherc  # Debug mode
./target/debug/aetherc build input.ae
```

## Contributing

### Development Setup

```bash
# Set up development environment
make env-setup

# Run development tests
make dev-test

# Check code formatting
cargo fmt --check

# Run linter
cargo clippy
```

### Testing

```bash
# Run full test suite
make test

# Run specific test categories
aether-file-test run --test-categories CoreLanguage,TypeSystem

# Add new test cases
aether-file-test generate --categories custom
```

## Performance & Optimization

### Compilation Performance

- Use `--parallel` flag for multi-threaded compilation
- Enable incremental compilation with `--incremental`
- Use `--cache-dir` to persist compilation cache

### Runtime Performance

- Profile with `aether-analyze --performance`
- Use profile-guided optimization (PGO)
- Enable GPU acceleration for tensor operations
- Consider WebAssembly for cross-platform deployment

## License

This project is licensed under MIT OR Apache-2.0. See LICENSE files for details.

## Resources

- [Language Specification](./Concept%20Documents/)
- [API Documentation](https://docs.rs/aether-language)
- [Examples](./examples/)
- [Community Forum](https://github.com/aether-lang/aether/discussions)
- [Issue Tracker](https://github.com/aether-lang/aether/issues)

---

For more detailed information about specific tools, run `<tool-name> --help` or consult the individual tool documentation.