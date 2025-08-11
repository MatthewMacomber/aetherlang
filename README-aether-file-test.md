# Aether File Compilation Testing CLI

The `aether-file-test` CLI tool provides comprehensive testing capabilities for compiling all Aether source files to executables. It supports batch processing, parallel execution, and multiple output formats.

## Installation

Build the CLI tool using Cargo:

```bash
cargo build --bin aether-file-test --release
```

The executable will be available at `target/release/aether-file-test` (or `target/release/aether-file-test.exe` on Windows).

## Basic Usage

### Run Complete Test Suite

```bash
# Run all tests with default settings
aether-file-test run

# Run with verbose output
aether-file-test --verbose run

# Run with custom compiler and output directory
aether-file-test --compiler ./aetherc --output ./test-results run
```

### Configuration File

Create a configuration file to avoid repeating command-line arguments:

```bash
# Generate a default configuration file
aether-file-test --save-config aether-test.toml

# Use configuration file
aether-file-test --config aether-test.toml run
```

### Validate Setup

```bash
# Validate configuration and environment
aether-file-test validate

# Validate with custom settings
aether-file-test --compiler ./aetherc --project-root ./my-project validate
```

### Discover Files

```bash
# List all Aether files in the project
aether-file-test discover

# Show detailed information
aether-file-test discover --detailed

# Filter by specific directories
aether-file-test discover --filter-dirs examples,tests
```

### Generate Test Files

```bash
# Generate additional test files
aether-file-test generate

# Generate specific categories
aether-file-test generate --categories core,types,ai
```

### Project Statistics

```bash
# Show project statistics
aether-file-test stats
```

### Clean Up

```bash
# Clean test artifacts
aether-file-test clean

# Also remove generated test files
aether-file-test clean --include-generated
```

## Command-Line Options

### Global Options

- `--project-root <PATH>`: Project root directory to scan (default: current directory)
- `--compiler <PATH>`: Path to aetherc compiler executable (default: "aetherc")
- `--output <PATH>`: Output directory for test artifacts (default: "target/file_compilation_tests")
- `--directories <DIRS>`: Comma-separated list of directories to include in testing
- `--format <FORMAT>`: Report output format (console, json, html, markdown)
- `--verbose`: Enable verbose output
- `--config <PATH>`: Configuration file path
- `--compile-timeout <SECONDS>`: Compilation timeout (default: 30)
- `--exec-timeout <SECONDS>`: Execution timeout (default: 10)
- `--max-parallel-compile <N>`: Maximum parallel compilation processes
- `--max-parallel-exec <N>`: Maximum parallel execution processes
- `--no-generate`: Disable generation of additional test files
- `--test-categories <CATEGORIES>`: Test categories to generate (core, types, ai, errors, performance)
- `--keep-artifacts`: Keep temporary artifacts after testing
- `--save-config <PATH>`: Save configuration to file and exit

### Run Command Options

- `--compile-only`: Only compile files, don't execute them
- `--execute-only`: Only execute already compiled files
- `--report-only`: Generate report from existing results

### Generate Command Options

- `--categories <CATEGORIES>`: Specific categories to generate
- `--output <PATH>`: Output directory for generated files

### Discover Command Options

- `--detailed`: Show detailed file information
- `--filter-dirs <DIRS>`: Filter by specific directories

### Clean Command Options

- `--include-generated`: Also remove generated test files

## Configuration File Format

The configuration file uses TOML format:

```toml
# Project root directory to scan for Aether files
project_root = "."

# Path to the aetherc compiler executable
compiler_path = "aetherc"

# Directory for output files (executables, reports, generated tests)
output_directory = "target/file_compilation_tests"

# Directories to include in testing
test_directories = ["examples", "tests"]

# Timeout for compilation operations (in seconds)
compilation_timeout = 30

# Timeout for executable execution (in seconds)
execution_timeout = 10

# Whether to generate additional test files
generate_additional_tests = true

# Categories of tests to generate
test_categories = ["CoreLanguage", "TypeSystem", "AIFeatures", "ErrorHandling"]

# Format for the final report
report_format = "Console"

# Whether to clean up temporary artifacts after testing
cleanup_artifacts = true

# Maximum number of parallel compilation processes
max_parallel_compilations = 8

# Maximum number of parallel executions
max_parallel_executions = 4

# Enable verbose logging
verbose = false
```

## Report Formats

### Console (Default)

Human-readable output with colored text and progress indicators.

### JSON

Machine-readable JSON format for programmatic consumption:

```bash
aether-file-test --format json run > results.json
```

### HTML

Styled HTML report (planned feature):

```bash
aether-file-test --format html run
```

### Markdown

Markdown format for documentation:

```bash
aether-file-test --format markdown run > TESTING_REPORT.md
```

## Examples

### Basic Testing Workflow

```bash
# 1. Validate setup
aether-file-test validate

# 2. Discover files
aether-file-test discover --detailed

# 3. Run tests
aether-file-test --verbose run

# 4. Clean up
aether-file-test clean
```

### CI/CD Integration

```bash
# Generate JSON report for CI systems
aether-file-test --format json --no-generate run > test-results.json

# Validate exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

### Custom Test Categories

```bash
# Generate only core language tests
aether-file-test generate --categories core

# Run tests on specific directories
aether-file-test --directories examples run
```

## Error Handling

The CLI provides detailed error messages and appropriate exit codes:

- Exit code 0: Success
- Exit code 1: General error (configuration, compilation, execution failures)

Common error scenarios:

1. **Compiler not found**: Ensure `aetherc` is in PATH or specify `--compiler` option
2. **Project root not found**: Verify the `--project-root` path exists
3. **Permission errors**: Ensure write access to output directory
4. **Timeout errors**: Increase `--compile-timeout` or `--exec-timeout` values

## Performance Tuning

### Parallel Execution

Adjust parallel processing based on your system:

```bash
# Use all CPU cores for compilation
aether-file-test --max-parallel-compile $(nproc) run

# Limit parallel executions to avoid resource exhaustion
aether-file-test --max-parallel-exec 4 run
```

### Timeout Configuration

Set appropriate timeouts based on your project complexity:

```bash
# Increase timeouts for complex projects
aether-file-test --compile-timeout 60 --exec-timeout 30 run
```

### Memory Management

For large projects, consider:

- Using `--no-generate` to skip test file generation
- Running tests in smaller batches using `--directories`
- Increasing system swap space if needed

## Troubleshooting

### Common Issues

1. **"Compiler not found" error**
   - Solution: Install aetherc or specify path with `--compiler`

2. **"Permission denied" errors**
   - Solution: Ensure write permissions to output directory

3. **Timeout errors**
   - Solution: Increase timeout values or check for infinite loops in code

4. **Out of memory errors**
   - Solution: Reduce parallel execution count or increase system memory

### Debug Mode

Enable verbose output for detailed debugging:

```bash
aether-file-test --verbose run
```

### Log Files

Check the output directory for detailed logs:

```
target/file_compilation_tests/
├── compilation_logs/
├── execution_logs/
└── reports/
```