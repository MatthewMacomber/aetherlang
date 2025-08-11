# Aether File Compilation Testing - Build System Integration

This document describes the build system integration for the Aether file compilation testing framework.

## Overview

The build system integration provides seamless integration with:
- Cargo build process
- CI/CD pipelines (GitHub Actions, GitLab CI, etc.)
- Make-based build systems
- Caching mechanisms for improved performance
- Automated regression testing

## Features

### 🔧 Cargo Integration

- **Automatic Testing**: Run file compilation tests as part of `cargo test`
- **Build Script Integration**: Optional testing during `cargo build`
- **Incremental Testing**: Only test changed files for faster feedback
- **Caching**: Intelligent caching of compilation and execution results

### 🚀 CI/CD Pipeline Support

- **GitHub Actions**: Pre-configured workflow for comprehensive testing
- **Artifact Generation**: JUnit XML, JSON, and summary reports
- **Multi-platform Testing**: Windows, macOS, and Linux support
- **Performance Benchmarks**: Automated performance regression detection

### ⚡ Performance Optimizations

- **Smart Caching**: Cache compilation results based on file content hashes
- **Parallel Execution**: Configurable parallel compilation and execution
- **Incremental Updates**: Git-based change detection
- **Resource Management**: Configurable timeouts and resource limits

## Quick Start

### 1. Basic Usage

```bash
# Build the testing tools
make build

# Run file compilation tests
make test-files

# Run all tests (including Cargo tests)
make test
```

### 2. Configuration

Create or modify `aether-file-test.toml`:

```toml
[testing]
project_root = "."
compiler_path = "aetherc"
output_directory = "target/file_compilation_tests"
test_directories = ["examples", "tests"]
compilation_timeout = 30
execution_timeout = 10
generate_additional_tests = true
verbose = false

[cargo_integration]
enable_caching = true
cache_expiration_hours = 24
incremental_testing = true
fail_build_on_test_failure = false
generate_ci_artifacts = true
```

### 3. Cargo Integration

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
# Your existing dev dependencies...

[[test]]
name = "file_compilation_integration_test"
path = "tests/file_compilation_integration_test.rs"
```

Run with cargo:

```bash
cargo test file_compilation_integration_test
```

## Build System Integration

### Cargo Build Script

The build script (`build.rs`) can automatically run file compilation tests:

```bash
# Enable automatic testing during build
export AETHER_RUN_FILE_TESTS=1
cargo build
```

### Make Integration

Use the provided Makefile for comprehensive build management:

```bash
# Available targets
make help

# Common workflows
make build          # Build compiler and tools
make test           # Run all tests
make test-files     # Run file compilation tests only
make clean          # Clean all artifacts
make ci-test        # Run CI/CD tests
```

### CI/CD Integration

#### GitHub Actions

The provided workflow (`.github/workflows/aether-file-compilation-tests.yml`) includes:

- Multi-platform testing (Windows, macOS, Linux)
- Multiple Rust versions (stable, beta)
- Artifact generation and upload
- Performance benchmarking
- Regression testing

#### Custom CI Systems

For other CI systems, use the CLI directly:

```bash
# Build the tools
cargo build --release --bin aetherc
cargo build --release --bin aether-file-test

# Run tests with CI-friendly output
./target/release/aether-file-test run \
  --format json \
  --output target/ci_artifacts \
  --verbose

# Check exit code for CI success/failure
echo "Exit code: $?"
```

## Caching System

### How It Works

The caching system provides intelligent caching of:
- **Compilation Results**: Based on source file and compiler hashes
- **Execution Results**: Based on executable file hashes
- **Test Generation**: Cached generated test files

### Configuration

```toml
[cache]
max_age_hours = 24
max_size_mb = 100
enable_compression = true
cleanup_interval_hours = 6
enable_validation = true
```

### Cache Management

```bash
# View cache statistics
./target/release/aether-file-test stats

# Clean cache
make clean-cache

# Clear all cache
./target/release/aether-file-test clean --include-generated
```

## Advanced Features

### Incremental Testing

Only test files that have changed since the last run:

```bash
# Enable incremental testing
./target/release/aether-file-test run --incremental

# Or configure in aether-file-test.toml
[cargo_integration]
incremental_testing = true
```

### Custom Test Categories

Generate and test specific categories of tests:

```bash
# Generate specific test categories
./target/release/aether-file-test generate --categories core,types,ai

# Run specific test categories
./target/release/aether-file-test run --test-categories core,types
```

### Performance Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Or directly
./target/release/aether-file-test run --test-categories performance
```

## Environment Variables

Control behavior through environment variables:

```bash
# Enable file tests during build
export AETHER_RUN_FILE_TESTS=1

# Fail build on test failures
export AETHER_FAIL_BUILD_ON_TEST_FAILURE=1

# Set log level
export AETHER_LOG_LEVEL=debug

# Enable profiling
export AETHER_ENABLE_PROFILING=1
```

## Troubleshooting

### Common Issues

1. **Compiler Not Found**
   ```bash
   # Ensure aetherc is built and in PATH
   cargo build --release --bin aetherc
   export PATH="$PWD/target/release:$PATH"
   ```

2. **Permission Issues**
   ```bash
   # Ensure output directories are writable
   mkdir -p target/file_compilation_tests
   chmod 755 target/file_compilation_tests
   ```

3. **Cache Issues**
   ```bash
   # Clear cache if results seem stale
   make clean-cache
   ```

### Debug Mode

Enable verbose output for debugging:

```bash
./target/release/aether-file-test run --verbose
```

### Validation

Validate your setup:

```bash
./target/release/aether-file-test validate
```

## Integration Examples

### GitHub Actions

```yaml
- name: Run Aether File Tests
  run: |
    cargo build --release --bin aether-file-test
    ./target/release/aether-file-test run --format json --output artifacts/

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: artifacts/
```

### GitLab CI

```yaml
test:aether-files:
  script:
    - cargo build --release --bin aether-file-test
    - ./target/release/aether-file-test run --format json --output artifacts/
  artifacts:
    reports:
      junit: artifacts/test_results.xml
    paths:
      - artifacts/
```

### Jenkins

```groovy
stage('Aether File Tests') {
    steps {
        sh 'cargo build --release --bin aether-file-test'
        sh './target/release/aether-file-test run --format json --output artifacts/'
    }
    post {
        always {
            junit 'artifacts/test_results.xml'
            archiveArtifacts 'artifacts/**'
        }
    }
}
```

## Performance Considerations

### Parallel Execution

Configure parallel execution for better performance:

```toml
[testing]
max_parallel_compilations = 4
max_parallel_executions = 2
```

### Resource Limits

Set appropriate timeouts:

```toml
[testing]
compilation_timeout = 30  # seconds
execution_timeout = 10    # seconds
```

### Cache Optimization

Optimize cache settings:

```toml
[cache]
max_size_mb = 500         # Increase for large projects
cleanup_interval_hours = 12  # More frequent cleanup
```

## Contributing

When contributing to the build integration:

1. Test on multiple platforms (Windows, macOS, Linux)
2. Ensure backward compatibility
3. Update documentation
4. Add appropriate tests
5. Consider performance impact

## Support

For issues with build integration:

1. Check the troubleshooting section
2. Run with `--verbose` for detailed output
3. Validate your setup with `validate` command
4. Check GitHub Issues for known problems