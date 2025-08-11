# Aether Language Makefile
# Provides build system integration for file compilation testing

.PHONY: all build test test-files clean install help
.DEFAULT_GOAL := help

# Configuration
CARGO := cargo
AETHER_FILE_TEST := ./target/release/aether-file-test
CONFIG_FILE := aether-file-test.toml

# Build targets
all: build test ## Build everything and run tests

build: ## Build the Aether compiler and tools
	@echo "Building Aether compiler and tools..."
	$(CARGO) build --release --bin aetherc
	$(CARGO) build --release --bin aether-file-test
	@echo "✅ Build completed"

build-debug: ## Build in debug mode
	@echo "Building Aether compiler and tools (debug)..."
	$(CARGO) build --bin aetherc
	$(CARGO) build --bin aether-file-test
	@echo "✅ Debug build completed"

# Test targets
test: build ## Run all tests including file compilation tests
	@echo "Running Cargo tests..."
	$(CARGO) test
	@echo "Running file compilation tests..."
	$(MAKE) test-files
	@echo "✅ All tests completed"

test-files: ## Run file compilation tests only
	@echo "Running Aether file compilation tests..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) run --config $(CONFIG_FILE) --verbose; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

test-files-quick: ## Run file compilation tests without generating additional tests
	@echo "Running quick file compilation tests..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) run --no-generate --format console; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

test-files-ci: build ## Run file compilation tests for CI/CD
	@echo "Running file compilation tests for CI/CD..."
	$(AETHER_FILE_TEST) run \
		--config $(CONFIG_FILE) \
		--format json \
		--output target/ci_artifacts \
		--verbose

test-incremental: ## Run incremental file compilation tests
	@echo "Running incremental file compilation tests..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) run --config $(CONFIG_FILE) --incremental; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

# Development targets
dev-test: build-debug ## Run tests in development mode
	@echo "Running development tests..."
	$(CARGO) test
	./target/debug/aether-file-test run --verbose

watch: ## Watch for changes and run tests
	@echo "Watching for changes..."
	$(CARGO) watch -x "build --bin aetherc" -x "build --bin aether-file-test" -s "make test-files-quick"

# Validation targets
validate: ## Validate configuration and setup
	@echo "Validating Aether file compilation testing setup..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) validate --config $(CONFIG_FILE); \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

discover: ## Discover Aether files in the project
	@echo "Discovering Aether files..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) discover --detailed; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

stats: ## Show project statistics
	@echo "Gathering project statistics..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) stats; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

# Generation targets
generate-tests: ## Generate additional test files
	@echo "Generating additional test files..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) generate --categories core,types,ai,errors; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

generate-config: ## Generate default configuration file
	@echo "Generating default configuration file..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) --save-config $(CONFIG_FILE); \
		echo "✅ Configuration saved to $(CONFIG_FILE)"; \
	else \
		echo "❌ aether-file-test binary not found. Run 'make build' first."; \
		exit 1; \
	fi

# Cleanup targets
clean: ## Clean build artifacts and test results
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	@echo "Cleaning test artifacts..."
	$(MAKE) clean-tests
	@echo "✅ Cleanup completed"

clean-tests: ## Clean test artifacts only
	@echo "Cleaning test artifacts..."
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) clean --include-generated; \
	else \
		rm -rf target/file_compilation_tests target/aether_test_cache target/ci_artifacts; \
	fi

clean-cache: ## Clean test cache only
	@echo "Cleaning test cache..."
	rm -rf target/aether_test_cache
	@echo "✅ Cache cleaned"

# Installation targets
install: build ## Install Aether tools to system
	@echo "Installing Aether tools..."
	$(CARGO) install --path . --bin aetherc
	$(CARGO) install --path . --bin aether-file-test
	@echo "✅ Installation completed"

install-dev: build-debug ## Install development version
	@echo "Installing development version..."
	$(CARGO) install --path . --bin aetherc --debug
	$(CARGO) install --path . --bin aether-file-test --debug
	@echo "✅ Development installation completed"

# CI/CD targets
ci-setup: ## Set up CI/CD environment
	@echo "Setting up CI/CD environment..."
	@mkdir -p target/ci_artifacts
	@mkdir -p target/aether_test_cache
	@echo "✅ CI/CD environment ready"

ci-test: ci-setup build ## Run tests for CI/CD
	@echo "Running CI/CD tests..."
	$(MAKE) test-files-ci
	@echo "✅ CI/CD tests completed"

ci-artifacts: ## Generate CI/CD artifacts
	@echo "Generating CI/CD artifacts..."
	@if [ -d "target/ci_artifacts" ]; then \
		echo "📊 Test artifacts generated in target/ci_artifacts/"; \
		ls -la target/ci_artifacts/; \
	else \
		echo "❌ No CI artifacts found. Run 'make ci-test' first."; \
	fi

# Performance targets
benchmark: build ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(CARGO) bench
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		$(AETHER_FILE_TEST) run --test-categories performance --format json; \
	fi

profile: build-debug ## Run with profiling enabled
	@echo "Running with profiling..."
	AETHER_ENABLE_PROFILING=1 $(MAKE) test-files

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	$(CARGO) doc --no-deps --open

docs-test: ## Test documentation examples
	@echo "Testing documentation examples..."
	$(CARGO) test --doc

# Environment targets
env-check: ## Check environment setup
	@echo "Checking environment setup..."
	@echo "Rust version: $$(rustc --version)"
	@echo "Cargo version: $$(cargo --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Available Aether files: $$(find . -name '*.ae' | wc -l)"
	@if command -v aetherc >/dev/null 2>&1; then \
		echo "✅ aetherc is available in PATH"; \
	else \
		echo "⚠️ aetherc not found in PATH"; \
	fi

env-setup: ## Set up development environment
	@echo "Setting up development environment..."
	@echo "Installing Rust components..."
	rustup component add rustfmt clippy
	@echo "Creating necessary directories..."
	@mkdir -p target/file_compilation_tests
	@mkdir -p target/aether_test_cache
	@mkdir -p target/ci_artifacts
	@echo "✅ Development environment ready"

# Help target
help: ## Show this help message
	@echo "Aether Language Build System"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make build          # Build the compiler and tools"
	@echo "  make test           # Run all tests"
	@echo "  make test-files     # Run file compilation tests only"
	@echo "  make clean          # Clean all artifacts"
	@echo "  make ci-test        # Run CI/CD tests"
	@echo ""
	@echo "Configuration:"
	@echo "  Config file: $(CONFIG_FILE)"
	@echo "  Test binary: $(AETHER_FILE_TEST)"

# Version information
version: ## Show version information
	@echo "Aether Language Build System"
	@echo "Version: 0.1.0"
	@echo "Rust version: $$(rustc --version)"
	@echo "Cargo version: $$(cargo --version)"
	@if [ -f "$(AETHER_FILE_TEST)" ]; then \
		echo "File test version: $$($(AETHER_FILE_TEST) --version)"; \
	fi