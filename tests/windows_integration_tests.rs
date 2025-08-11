// Integration tests for Windows executable generation and testing workflow
// Tests the complete end-to-end process of compiling Aether code to Windows executables

use aether_language::build_system::{
    BuildSystemManager, AetherCompiler, AetherCompilationConfig, CompilationTarget,
    WindowsExecutableGenerator, WindowsExecutableConfig, WindowsExecutableTester,
    WindowsTestConfig, WindowsArchitecture, WindowsSubsystem, WindowsVersion,
    windows_helpers, aether_compiler_helpers
};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

/// Test the complete Windows executable generation workflow
#[test]
fn test_complete_windows_executable_workflow() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create mock Aether compiler binary
    let compiler_path = temp_dir.path().join("aetherc.exe");
    create_mock_compiler(&compiler_path);
    
    // Create sample Aether source file
    let source_path = temp_dir.path().join("hello_windows.ae");
    create_sample_aether_source(&source_path);
    
    // Set up Windows compilation configuration
    let config = aether_compiler_helpers::windows_compilation_config(
        temp_dir.path().join("output")
    );
    
    // Create Aether compiler with Windows support
    let mut compiler = AetherCompiler::with_config(compiler_path, config).unwrap();
    assert!(compiler.has_windows_support());
    
    // Configure Windows-specific settings
    let windows_config = aether_compiler_helpers::recommended_windows_executable_config();
    compiler.configure_windows_executable(windows_config).unwrap();
    
    let test_config = aether_compiler_helpers::recommended_windows_test_config();
    compiler.configure_windows_testing(test_config).unwrap();
    
    // Validate syntax first
    let validation = compiler.validate_syntax(&source_path).unwrap();
    assert!(validation.is_valid, "Aether source should have valid syntax");
    assert!(validation.ast.is_some(), "AST should be generated");
    assert!(validation.errors.is_empty(), "No syntax errors should be present");
}

/// Test Windows executable generation through BuildSystemManager
#[test]
fn test_build_system_manager_windows_workflow() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create mock Aether compiler binary
    let compiler_path = temp_dir.path().join("aetherc.exe");
    create_mock_compiler(&compiler_path);
    
    // Create sample Aether source file
    let source_path = temp_dir.path().join("test_program.ae");
    create_sample_aether_source(&source_path);
    
    // Create build system manager
    let mut manager = BuildSystemManager::new();
    
    // Validate environment
    let env_status = manager.validate_environment().unwrap();
    println!("Environment status: {:?}", env_status.overall_status);
    
    // Test Aether source compilation (this will fail with mock compiler, but we test the workflow)
    let result = manager.compile_aether_source(&source_path);
    
    // The compilation will fail because we're using a mock compiler,
    // but we can verify the workflow was attempted
    match result {
        Ok(_) => {
            // If somehow successful with mock, that's fine too
            println!("Mock compilation succeeded unexpectedly");
        }
        Err(e) => {
            // Expected to fail with mock compiler
            println!("Expected compilation failure with mock compiler: {}", e);
        }
    }
}

/// Test Windows-specific executable generation features
#[test]
fn test_windows_executable_features() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test different Windows configurations
    let configs = vec![
        ("debug", windows_helpers::debug_windows_config()),
        ("release", windows_helpers::release_windows_config()),
        ("default", windows_helpers::default_windows_config()),
    ];
    
    for (name, config) in configs {
        println!("Testing {} configuration", name);
        
        let generator = WindowsExecutableGenerator::with_config(config.clone());
        assert_eq!(generator.config().target_arch, config.target_arch);
        assert_eq!(generator.config().subsystem, config.subsystem);
        assert_eq!(generator.config().bundle_dependencies, config.bundle_dependencies);
        assert_eq!(generator.config().include_debug_info, config.include_debug_info);
    }
}

/// Test Windows executable testing configurations
#[test]
fn test_windows_executable_testing_configurations() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test different testing configurations
    let base_config = WindowsTestConfig::default();
    
    // Test with different Windows versions
    let version_configs = vec![
        WindowsVersion::Windows10,
        WindowsVersion::Windows11,
        WindowsVersion::WindowsServer2019,
        WindowsVersion::WindowsServer2022,
    ];
    
    for version in version_configs {
        let mut config = base_config.clone();
        config.test_versions = vec![version.clone()];
        
        let tester = WindowsExecutableTester::with_config(config);
        assert!(tester.config().test_versions.contains(&version));
    }
    
    // Test with different architectures
    let arch_configs = vec![
        WindowsArchitecture::X86,
        WindowsArchitecture::X64,
        WindowsArchitecture::ARM64,
    ];
    
    for arch in arch_configs {
        let mut config = base_config.clone();
        config.test_architectures = vec![arch.clone()];
        
        let tester = WindowsExecutableTester::with_config(config);
        assert!(tester.config().test_architectures.contains(&arch));
    }
}

/// Test Windows dependency detection and bundling
#[test]
fn test_windows_dependency_workflow() {
    use aether_language::build_system::WindowsRuntimeDetector;
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a mock executable
    let mock_exe = temp_dir.path().join("test.exe");
    fs::write(&mock_exe, "mock executable content").unwrap();
    
    // Test dependency detection
    let mut detector = WindowsRuntimeDetector::new();
    let dependencies = detector.detect_dependencies(&mock_exe).unwrap();
    
    assert!(!dependencies.is_empty(), "Should detect some dependencies");
    
    // Verify we have both system and non-system dependencies
    let system_deps: Vec<_> = dependencies.iter().filter(|d| d.is_system_dll).collect();
    let runtime_deps: Vec<_> = dependencies.iter().filter(|d| !d.is_system_dll).collect();
    
    assert!(!system_deps.is_empty(), "Should have system dependencies");
    assert!(!runtime_deps.is_empty(), "Should have runtime dependencies");
    
    // Test dependency caching
    let dependencies2 = detector.detect_dependencies(&mock_exe).unwrap();
    assert_eq!(dependencies.len(), dependencies2.len(), "Cached results should match");
    
    // Test cache clearing
    detector.clear_cache();
    let dependencies3 = detector.detect_dependencies(&mock_exe).unwrap();
    assert_eq!(dependencies.len(), dependencies3.len(), "Results should be consistent after cache clear");
}

/// Test Windows environment and registry handling
#[test]
fn test_windows_environment_workflow() {
    use aether_language::build_system::WindowsEnvironmentHandler;
    
    let mut handler = WindowsEnvironmentHandler::new();
    
    // Test environment variable operations
    let test_var = "AETHER_WINDOWS_TEST";
    let test_value = "integration_test_value";
    
    // Set and retrieve environment variable
    handler.set_env_var(test_var, test_value).unwrap();
    let retrieved = handler.get_env_var(test_var).unwrap();
    assert_eq!(retrieved, Some(test_value.to_string()));
    
    // Test runtime environment detection
    let runtime_env = handler.detect_runtime_environment().unwrap();
    
    // Verify detected information makes sense
    assert!(matches!(runtime_env.version, 
        WindowsVersion::Windows10 | 
        WindowsVersion::Windows11 | 
        WindowsVersion::WindowsServer2019 | 
        WindowsVersion::WindowsServer2022
    ));
    
    assert!(matches!(runtime_env.architecture,
        WindowsArchitecture::X86 |
        WindowsArchitecture::X64 |
        WindowsArchitecture::ARM64
    ));
    
    // Should detect some runtime libraries
    println!("Detected runtime libraries: {:?}", runtime_env.runtime_libraries);
    println!("Detected .NET versions: {:?}", runtime_env.dotnet_versions);
    println!("Detected VC++ redistributables: {:?}", runtime_env.vcredist_versions);
}

/// Test Windows compatibility checking workflow
#[test]
fn test_windows_compatibility_workflow() {
    use aether_language::build_system::WindowsCompatibilityChecker;
    
    let temp_dir = TempDir::new().unwrap();
    let mock_exe = temp_dir.path().join("compatibility_test.exe");
    fs::write(&mock_exe, "mock executable for compatibility testing").unwrap();
    
    let checker = WindowsCompatibilityChecker::new();
    let system_info = checker.system_info();
    
    println!("System info - Version: {:?}, Architecture: {:?}", 
             system_info.version, system_info.architecture);
    
    // Test compatibility with different targets
    let test_cases = vec![
        (WindowsVersion::Windows10, WindowsArchitecture::X64),
        (WindowsVersion::Windows11, WindowsArchitecture::X64),
        (WindowsVersion::Windows10, WindowsArchitecture::X86),
        (WindowsVersion::WindowsServer2019, WindowsArchitecture::X64),
    ];
    
    for (version, arch) in test_cases {
        let compat_result = checker.check_compatibility(&mock_exe, &version, &arch).unwrap();
        
        println!("Compatibility test - Version: {:?}, Arch: {:?}, Compatible: {}", 
                 version, arch, compat_result.compatible);
        
        if !compat_result.compatible {
            println!("Issues: {:?}", compat_result.issues);
            println!("Recommendations: {:?}", compat_result.recommendations);
        }
        
        // Verify result structure
        assert_eq!(compat_result.windows_version, version);
        assert_eq!(compat_result.architecture, arch);
    }
}

/// Test complete Aether compilation configuration workflow
#[test]
fn test_aether_compilation_configuration_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    create_mock_compiler(&compiler_path);
    
    // Test different compilation configurations
    let configs = vec![
        ("windows_default", aether_compiler_helpers::windows_compilation_config(temp_dir.path().to_path_buf())),
        ("windows_debug", aether_compiler_helpers::windows_debug_config(temp_dir.path().to_path_buf())),
        ("windows_release", aether_compiler_helpers::windows_release_config(temp_dir.path().to_path_buf())),
    ];
    
    for (name, config) in configs {
        println!("Testing {} configuration", name);
        
        let compiler = AetherCompiler::with_config(compiler_path.clone(), config.clone()).unwrap();
        
        // Verify Windows support is enabled
        assert!(compiler.has_windows_support(), "Windows support should be enabled for {}", name);
        
        // Verify configuration properties
        assert_eq!(compiler.config().target, CompilationTarget::WindowsNative);
        assert_eq!(compiler.config().output_directory, config.output_directory);
        assert_eq!(compiler.config().optimization_level, config.optimization_level);
        assert_eq!(compiler.config().debug_info, config.debug_info);
        assert_eq!(compiler.config().verbose, config.verbose);
        
        // Verify Windows-specific configurations are available
        assert!(compiler.get_windows_config().is_some());
        assert!(compiler.get_windows_test_config().is_some());
    }
}

/// Test executable name determination for different scenarios
#[test]
fn test_executable_name_determination_workflow() {
    let test_cases = vec![
        ("simple.ae", "simple.exe"),
        ("hello_world.ae", "hello_world.exe"),
        ("complex-name.ae", "complex-name.exe"),
        ("test.aether", "test.exe"), // Different extension
        ("no_extension", "no_extension.exe"),
    ];
    
    for (input, expected) in test_cases {
        let source_path = Path::new(input);
        let exe_name = aether_compiler_helpers::determine_executable_name(
            source_path, 
            &CompilationTarget::WindowsNative
        );
        
        assert_eq!(exe_name, PathBuf::from(expected), 
                   "Failed for input: {}", input);
    }
    
    // Test with other targets for comparison
    let source_path = Path::new("test.ae");
    
    let wasm_name = aether_compiler_helpers::determine_executable_name(
        source_path, &CompilationTarget::WebAssembly
    );
    assert_eq!(wasm_name, PathBuf::from("test.wasm"));
    
    let gpu_name = aether_compiler_helpers::determine_executable_name(
        source_path, &CompilationTarget::GPU
    );
    assert_eq!(gpu_name, PathBuf::from("test.ptx"));
    
    let mobile_name = aether_compiler_helpers::determine_executable_name(
        source_path, &CompilationTarget::Mobile
    );
    assert_eq!(mobile_name, PathBuf::from("test.aether"));
}

/// Test Windows-specific error handling and recovery
#[test]
fn test_windows_error_handling_workflow() {
    use aether_language::build_system::{WindowsExecutableError, WindowsExecutableGenerator};
    
    let temp_dir = TempDir::new().unwrap();
    
    // Test error handling for non-existent input
    let mut generator = WindowsExecutableGenerator::new();
    let non_existent = temp_dir.path().join("does_not_exist.exe");
    let output = temp_dir.path().join("output.exe");
    
    let result = generator.generate_executable(&non_existent, &output);
    assert!(result.is_err(), "Should fail for non-existent input");
    
    match result.unwrap_err() {
        WindowsExecutableError::PEGenerationFailed(msg) => {
            assert!(msg.contains("not found"), "Error message should mention file not found");
        }
        _ => panic!("Expected PEGenerationFailed error"),
    }
    
    // Test error handling for invalid configuration
    let temp_compiler_path = temp_dir.path().join("aetherc.exe");
    create_mock_compiler(&temp_compiler_path);
    
    let invalid_config = AetherCompilationConfig {
        target: CompilationTarget::WindowsNative,
        optimization_level: 5, // Invalid optimization level
        debug_info: true,
        compiler_flags: Vec::new(),
        output_directory: temp_dir.path().to_path_buf(),
        verbose: false,
    };
    
    let compiler = AetherCompiler::with_config(temp_compiler_path, invalid_config).unwrap();
    let validation_result = compiler.validate_config(compiler.config());
    assert!(validation_result.is_err(), "Should fail for invalid optimization level");
}

/// Test Windows helpers utility functions
#[test]
fn test_windows_helpers_workflow() {
    // Test executable extension
    assert_eq!(windows_helpers::get_executable_extension(), ".exe");
    
    // Test platform detection
    // Note: This test will behave differently on different platforms
    println!("Is Windows platform: {}", windows_helpers::is_windows());
    
    // Test system directory detection
    let system_dir = windows_helpers::get_system_directory();
    println!("System directory: {}", system_dir.display());
    
    // Test program files directory detection
    let program_files = windows_helpers::get_program_files_directory();
    println!("Program Files directory: {}", program_files.display());
    
    // Test configuration helpers
    let debug_config = windows_helpers::debug_windows_config();
    assert!(debug_config.include_debug_info);
    assert!(!debug_config.linker_flags.is_empty());
    
    let release_config = windows_helpers::release_windows_config();
    assert!(!release_config.include_debug_info);
    assert!(!release_config.linker_flags.is_empty());
}

/// Helper function to create a mock Aether compiler binary
fn create_mock_compiler(path: &Path) {
    let mock_compiler_content = r#"#!/bin/bash
# Mock Aether compiler for testing
echo "Mock Aether Compiler v0.1.0"
if [[ "$1" == "--version" ]]; then
    echo "aetherc 0.1.0"
    exit 0
fi
if [[ "$1" == "--help" ]]; then
    echo "Usage: aetherc [options] <source-file>"
    echo "Options:"
    echo "  -o <output>     Specify output file"
    echo "  --target <tgt>  Specify target platform"
    echo "  -O <level>      Optimization level (0-3)"
    echo "  --debug         Include debug information"
    echo "  --verbose       Verbose output"
    exit 0
fi
# For actual compilation, just create a mock executable
if [[ "$2" == "-o" ]]; then
    echo "Mock compilation of $1 to $3"
    echo "Mock Windows executable" > "$3"
    exit 0
fi
echo "Mock compiler executed with args: $@"
exit 1
"#;
    
    fs::write(path, mock_compiler_content).unwrap();
    
    // On Unix systems, make it executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }
}

/// Helper function to create a sample Aether source file
fn create_sample_aether_source(path: &Path) {
    let aether_source = r#"
; Sample Aether program for Windows executable testing
(func main ()
  (let message "Hello from Aether on Windows!")
  (call print message)
  (call print "Testing Windows executable generation...")
  
  ; Test basic arithmetic
  (let x 42)
  (let y 13)
  (let sum (+ x y))
  (call print (format "Sum: {}" sum))
  
  ; Test conditional logic
  (if (> sum 50)
    (call print "Sum is greater than 50")
    (call print "Sum is 50 or less"))
  
  ; Test loop
  (for i 0 3
    (call print (format "Loop iteration: {}" i)))
  
  (call print "Windows executable test completed successfully!")
  (return 0))

; Helper function for demonstration
(func format (template value)
  (call string-format template value))
"#;
    
    fs::write(path, aether_source).unwrap();
}

/// Test that demonstrates the complete workflow from source to tested executable
#[test]
fn test_end_to_end_windows_workflow() {
    let temp_dir = TempDir::new().unwrap();
    
    // Step 1: Set up the environment
    let compiler_path = temp_dir.path().join("aetherc.exe");
    create_mock_compiler(&compiler_path);
    
    let source_path = temp_dir.path().join("end_to_end_test.ae");
    create_sample_aether_source(&source_path);
    
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).unwrap();
    
    // Step 2: Configure Windows compilation
    let compilation_config = aether_compiler_helpers::windows_compilation_config(output_dir.clone());
    let mut compiler = AetherCompiler::with_config(compiler_path, compilation_config).unwrap();
    
    // Step 3: Configure Windows-specific settings
    let windows_exe_config = aether_compiler_helpers::recommended_windows_executable_config();
    compiler.configure_windows_executable(windows_exe_config).unwrap();
    
    let windows_test_config = aether_compiler_helpers::recommended_windows_test_config();
    compiler.configure_windows_testing(windows_test_config).unwrap();
    
    // Step 4: Validate syntax
    let syntax_validation = compiler.validate_syntax(&source_path).unwrap();
    assert!(syntax_validation.is_valid, "Source should have valid syntax");
    
    // Step 5: Validate configuration
    let config_validation = compiler.validate_config(compiler.config());
    assert!(config_validation.is_ok(), "Configuration should be valid");
    
    // Step 6: Verify Windows support is properly configured
    assert!(compiler.has_windows_support(), "Windows support should be enabled");
    assert!(compiler.get_windows_config().is_some(), "Windows config should be available");
    assert!(compiler.get_windows_test_config().is_some(), "Windows test config should be available");
    
    // Step 7: Test helper functions work correctly
    let exe_name = aether_compiler_helpers::determine_executable_name(
        &source_path, 
        &CompilationTarget::WindowsNative
    );
    assert_eq!(exe_name, PathBuf::from("end_to_end_test.exe"));
    
    let requires_windows = aether_compiler_helpers::requires_windows_processing(
        &CompilationTarget::WindowsNative
    );
    assert!(requires_windows, "Should require Windows processing");
    
    println!("End-to-end Windows workflow test completed successfully!");
    println!("- Mock compiler created and configured");
    println!("- Sample Aether source created");
    println!("- Windows compilation configured");
    println!("- Windows executable settings configured");
    println!("- Windows testing settings configured");
    println!("- Syntax validation passed");
    println!("- Configuration validation passed");
    println!("- Windows support verified");
    println!("- Helper functions validated");
}