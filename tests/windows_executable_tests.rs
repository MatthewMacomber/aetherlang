// Tests for Windows-specific executable generation and testing
// Validates PE format generation, dependency bundling, and compatibility testing

use aether_language::build_system::{
    AetherCompiler, AetherCompilationConfig, CompilationTarget,
    WindowsExecutableGenerator, WindowsExecutableConfig, WindowsExecutableTester,
    WindowsTestConfig, WindowsArchitecture, WindowsSubsystem, WindowsVersion,
    windows_helpers, aether_compiler_helpers
};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_windows_executable_config_creation() {
    let config = WindowsExecutableConfig::default();
    assert_eq!(config.target_arch, WindowsArchitecture::X64);
    assert_eq!(config.subsystem, WindowsSubsystem::Console);
    assert!(config.bundle_dependencies);
    assert!(config.include_debug_info);
}

#[test]
fn test_windows_executable_generator_creation() {
    let generator = WindowsExecutableGenerator::new();
    let config = generator.config();
    assert_eq!(config.target_arch, WindowsArchitecture::X64);
    assert_eq!(config.subsystem, WindowsSubsystem::Console);
}

#[test]
fn test_windows_executable_tester_creation() {
    let tester = WindowsExecutableTester::new();
    let config = tester.config();
    assert_eq!(config.timeout, Duration::from_secs(30));
    assert!(config.test_compatibility);
    assert!(config.validate_dependencies);
}

#[test]
fn test_aether_compiler_windows_support() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();

    // Test with Windows target
    let windows_config = AetherCompilationConfig {
        target: CompilationTarget::WindowsNative,
        optimization_level: 2,
        debug_info: true,
        compiler_flags: Vec::new(),
        output_directory: temp_dir.path().to_path_buf(),
        verbose: false,
    };

    let compiler = AetherCompiler::with_config(compiler_path, windows_config).unwrap();
    assert!(compiler.has_windows_support());
    assert!(compiler.get_windows_config().is_some());
    assert!(compiler.get_windows_test_config().is_some());
}

#[test]
fn test_aether_compiler_non_windows_target() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();

    // Test with non-Windows target
    let wasm_config = AetherCompilationConfig {
        target: CompilationTarget::WebAssembly,
        optimization_level: 2,
        debug_info: true,
        compiler_flags: Vec::new(),
        output_directory: temp_dir.path().to_path_buf(),
        verbose: false,
    };

    let compiler = AetherCompiler::with_config(compiler_path, wasm_config).unwrap();
    assert!(!compiler.has_windows_support());
    assert!(compiler.get_windows_config().is_none());
    assert!(compiler.get_windows_test_config().is_none());
}

#[test]
fn test_windows_compilation_config_helpers() {
    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().to_path_buf();

    // Test Windows compilation config
    let config = aether_compiler_helpers::windows_compilation_config(output_dir.clone());
    assert_eq!(config.target, CompilationTarget::WindowsNative);
    assert_eq!(config.optimization_level, 2);
    assert!(config.debug_info);

    // Test Windows debug config
    let debug_config = aether_compiler_helpers::windows_debug_config(output_dir.clone());
    assert_eq!(debug_config.target, CompilationTarget::WindowsNative);
    assert_eq!(debug_config.optimization_level, 0);
    assert!(debug_config.debug_info);
    assert!(debug_config.verbose);

    // Test Windows release config
    let release_config = aether_compiler_helpers::windows_release_config(output_dir);
    assert_eq!(release_config.target, CompilationTarget::WindowsNative);
    assert_eq!(release_config.optimization_level, 3);
    assert!(!release_config.debug_info);
    assert!(!release_config.verbose);
}

#[test]
fn test_windows_executable_name_determination() {
    let source_path = Path::new("test.ae");
    let exe_name = aether_compiler_helpers::determine_executable_name(
        source_path, 
        &CompilationTarget::WindowsNative
    );
    assert_eq!(exe_name, PathBuf::from("test.exe"));
}

#[test]
fn test_windows_processing_requirement() {
    assert!(aether_compiler_helpers::requires_windows_processing(&CompilationTarget::WindowsNative));
    assert!(!aether_compiler_helpers::requires_windows_processing(&CompilationTarget::WebAssembly));
    assert!(!aether_compiler_helpers::requires_windows_processing(&CompilationTarget::GPU));
    assert!(!aether_compiler_helpers::requires_windows_processing(&CompilationTarget::Mobile));
}

#[test]
fn test_recommended_windows_configs() {
    let exe_config = aether_compiler_helpers::recommended_windows_executable_config();
    assert_eq!(exe_config.target_arch, WindowsArchitecture::X64);
    assert_eq!(exe_config.subsystem, WindowsSubsystem::Console);
    assert!(exe_config.bundle_dependencies);

    let test_config = aether_compiler_helpers::recommended_windows_test_config();
    assert_eq!(test_config.timeout, Duration::from_secs(60));
    assert!(test_config.test_compatibility);
    assert!(test_config.validate_dependencies);
}

#[test]
fn test_windows_helpers() {
    assert_eq!(windows_helpers::get_executable_extension(), ".exe");
    
    let debug_config = windows_helpers::debug_windows_config();
    assert!(debug_config.include_debug_info);
    assert!(!debug_config.linker_flags.is_empty());
    
    let release_config = windows_helpers::release_windows_config();
    assert!(!release_config.include_debug_info);
    assert!(!release_config.linker_flags.is_empty());
}

#[test]
fn test_windows_architecture_variants() {
    let architectures = vec![
        WindowsArchitecture::X86,
        WindowsArchitecture::X64,
        WindowsArchitecture::ARM64,
    ];
    
    for arch in architectures {
        let mut config = WindowsExecutableConfig::default();
        config.target_arch = arch.clone();
        
        let generator = WindowsExecutableGenerator::with_config(config);
        assert_eq!(generator.config().target_arch, arch);
    }
}

#[test]
fn test_windows_subsystem_variants() {
    let subsystems = vec![
        WindowsSubsystem::Console,
        WindowsSubsystem::Windows,
        WindowsSubsystem::Native,
    ];
    
    for subsystem in subsystems {
        let mut config = WindowsExecutableConfig::default();
        config.subsystem = subsystem.clone();
        
        let generator = WindowsExecutableGenerator::with_config(config);
        assert_eq!(generator.config().subsystem, subsystem);
    }
}

#[test]
fn test_windows_version_compatibility() {
    let versions = vec![
        WindowsVersion::Windows10,
        WindowsVersion::Windows11,
        WindowsVersion::WindowsServer2019,
        WindowsVersion::WindowsServer2022,
    ];
    
    for version in versions {
        let mut config = WindowsTestConfig::default();
        config.test_versions = vec![version.clone()];
        
        let tester = WindowsExecutableTester::with_config(config);
        assert!(tester.config().test_versions.contains(&version));
    }
}

#[test]
fn test_aether_compiler_windows_configuration() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();

    let config = aether_compiler_helpers::windows_compilation_config(temp_dir.path().to_path_buf());
    let mut compiler = AetherCompiler::with_config(compiler_path, config).unwrap();

    // Test Windows executable configuration
    let windows_config = aether_compiler_helpers::recommended_windows_executable_config();
    let result = compiler.configure_windows_executable(windows_config.clone());
    assert!(result.is_ok());

    // Verify configuration was applied
    let applied_config = compiler.get_windows_config().unwrap();
    assert_eq!(applied_config.target_arch, windows_config.target_arch);
    assert_eq!(applied_config.subsystem, windows_config.subsystem);

    // Test Windows testing configuration
    let test_config = aether_compiler_helpers::recommended_windows_test_config();
    let result = compiler.configure_windows_testing(test_config.clone());
    assert!(result.is_ok());

    // Verify test configuration was applied
    let applied_test_config = compiler.get_windows_test_config().unwrap();
    assert_eq!(applied_test_config.timeout, test_config.timeout);
    assert_eq!(applied_test_config.test_compatibility, test_config.test_compatibility);
}

#[test]
fn test_aether_compiler_target_switching() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();

    // Start with Windows target
    let windows_config = aether_compiler_helpers::windows_compilation_config(temp_dir.path().to_path_buf());
    let mut compiler = AetherCompiler::with_config(compiler_path, windows_config).unwrap();
    assert!(compiler.has_windows_support());

    // Switch to WebAssembly target
    let wasm_config = AetherCompilationConfig {
        target: CompilationTarget::WebAssembly,
        optimization_level: 2,
        debug_info: true,
        compiler_flags: Vec::new(),
        output_directory: temp_dir.path().to_path_buf(),
        verbose: false,
    };
    compiler.update_config(wasm_config);
    assert!(!compiler.has_windows_support());

    // Switch back to Windows target
    let windows_config2 = aether_compiler_helpers::windows_compilation_config(temp_dir.path().to_path_buf());
    compiler.update_config(windows_config2);
    assert!(compiler.has_windows_support());
}

#[test]
fn test_windows_executable_generation_mock() {
    let temp_dir = TempDir::new().unwrap();
    let compiler_path = temp_dir.path().join("aetherc.exe");
    fs::write(&compiler_path, "mock compiler").unwrap();

    // Create a mock Aether source file
    let source_path = temp_dir.path().join("hello.ae");
    fs::write(&source_path, "(func main () (call print \"Hello, Windows!\") (return 0))").unwrap();

    let config = aether_compiler_helpers::windows_compilation_config(temp_dir.path().to_path_buf());
    let mut compiler = AetherCompiler::with_config(compiler_path, config).unwrap();

    // Test syntax validation
    let validation = compiler.validate_syntax(&source_path).unwrap();
    assert!(validation.is_valid);
    assert!(validation.ast.is_some());
}

#[test]
fn test_windows_dependency_detection_simulation() {
    use aether_language::build_system::WindowsRuntimeDetector;
    
    let temp_dir = TempDir::new().unwrap();
    let mock_exe = temp_dir.path().join("test.exe");
    fs::write(&mock_exe, "mock executable").unwrap();

    let mut detector = WindowsRuntimeDetector::new();
    let dependencies = detector.detect_dependencies(&mock_exe).unwrap();
    
    // Should detect some common dependencies
    assert!(!dependencies.is_empty());
    
    // Should include system DLLs
    let has_system_dll = dependencies.iter().any(|dep| dep.is_system_dll);
    assert!(has_system_dll);
    
    // Should include runtime DLLs
    let has_runtime_dll = dependencies.iter().any(|dep| !dep.is_system_dll);
    assert!(has_runtime_dll);
}

#[test]
fn test_windows_environment_detection() {
    use aether_language::build_system::WindowsEnvironmentHandler;
    
    let mut handler = WindowsEnvironmentHandler::new();
    
    // Test environment variable access
    handler.set_env_var("AETHER_TEST_VAR", "test_value").unwrap();
    let value = handler.get_env_var("AETHER_TEST_VAR").unwrap();
    assert_eq!(value, Some("test_value".to_string()));
    
    // Test runtime environment detection
    let runtime_env = handler.detect_runtime_environment().unwrap();
    
    // Should detect some Windows version
    assert!(matches!(runtime_env.version, 
        WindowsVersion::Windows10 | 
        WindowsVersion::Windows11 | 
        WindowsVersion::WindowsServer2019 | 
        WindowsVersion::WindowsServer2022
    ));
    
    // Should detect architecture
    assert!(matches!(runtime_env.architecture,
        WindowsArchitecture::X86 |
        WindowsArchitecture::X64 |
        WindowsArchitecture::ARM64
    ));
}

#[test]
fn test_windows_compatibility_checker() {
    use aether_language::build_system::WindowsCompatibilityChecker;
    
    let temp_dir = TempDir::new().unwrap();
    let mock_exe = temp_dir.path().join("test.exe");
    fs::write(&mock_exe, "mock executable").unwrap();

    let checker = WindowsCompatibilityChecker::new();
    let system_info = checker.system_info();
    
    // Should have detected system information
    assert!(matches!(system_info.version, 
        WindowsVersion::Windows10 | 
        WindowsVersion::Windows11 | 
        WindowsVersion::WindowsServer2019 | 
        WindowsVersion::WindowsServer2022
    ));
    
    // Test compatibility check
    let compat_result = checker.check_compatibility(
        &mock_exe,
        &WindowsVersion::Windows10,
        &WindowsArchitecture::X64
    ).unwrap();
    
    // Should provide compatibility information
    assert!(compat_result.windows_version == WindowsVersion::Windows10);
    assert!(compat_result.architecture == WindowsArchitecture::X64);
}

#[test]
fn test_pe_format_analysis() {
    use aether_language::build_system::{PEFormatInfo, WindowsSubsystem};
    
    let pe_info = PEFormatInfo {
        signature: "PE00".to_string(),
        machine_type: "IMAGE_FILE_MACHINE_AMD64".to_string(),
        section_count: 4,
        entry_point: 0x1000,
        image_base: 0x140000000,
        subsystem: WindowsSubsystem::Console,
        dll_characteristics: 0x8160,
    };
    
    assert_eq!(pe_info.signature, "PE00");
    assert_eq!(pe_info.machine_type, "IMAGE_FILE_MACHINE_AMD64");
    assert_eq!(pe_info.section_count, 4);
    assert_eq!(pe_info.entry_point, 0x1000);
    assert_eq!(pe_info.image_base, 0x140000000);
    assert_eq!(pe_info.subsystem, WindowsSubsystem::Console);
    assert_eq!(pe_info.dll_characteristics, 0x8160);
}

#[test]
fn test_windows_version_info() {
    use aether_language::build_system::WindowsVersionInfo;
    
    let version_info = WindowsVersionInfo::default();
    assert_eq!(version_info.file_version, "0.1.0.0");
    assert_eq!(version_info.product_version, "0.1.0");
    assert_eq!(version_info.company_name, "Aether Language Team");
    assert_eq!(version_info.file_description, "Aether Language Executable");
    assert_eq!(version_info.product_name, "Aether Language");
    assert!(version_info.copyright.contains("2024"));
}

#[test]
fn test_windows_executable_result_structure() {
    use aether_language::build_system::{WindowsExecutableResult, PEFormatInfo, WindowsSubsystem};
    use std::time::Duration;
    
    let temp_dir = TempDir::new().unwrap();
    let exe_path = temp_dir.path().join("test.exe");
    
    let pe_info = PEFormatInfo {
        signature: "PE00".to_string(),
        machine_type: "IMAGE_FILE_MACHINE_AMD64".to_string(),
        section_count: 4,
        entry_point: 0x1000,
        image_base: 0x140000000,
        subsystem: WindowsSubsystem::Console,
        dll_characteristics: 0x8160,
    };
    
    let result = WindowsExecutableResult {
        executable_path: exe_path.clone(),
        pe_info,
        bundled_dependencies: Vec::new(),
        generation_time: Duration::from_millis(100),
        success: true,
        errors: Vec::new(),
    };
    
    assert_eq!(result.executable_path, exe_path);
    assert!(result.success);
    assert!(result.errors.is_empty());
    assert_eq!(result.generation_time, Duration::from_millis(100));
}

#[test]
fn test_windows_test_result_structure() {
    use aether_language::build_system::{
        WindowsTestResult, WindowsExecutionResult, WindowsCompatibilityResult,
        WindowsDependencyResult, WindowsVersion, WindowsArchitecture
    };
    use std::time::Duration;
    
    let execution_result = WindowsExecutionResult {
        exit_code: 0,
        stdout: "Hello, Windows!".to_string(),
        stderr: String::new(),
        execution_time: Duration::from_millis(50),
        peak_memory_usage: 1024 * 1024, // 1MB
        success: true,
    };
    
    let compatibility_result = WindowsCompatibilityResult {
        windows_version: WindowsVersion::Windows10,
        architecture: WindowsArchitecture::X64,
        compatible: true,
        issues: Vec::new(),
        recommendations: Vec::new(),
    };
    
    let dependency_result = WindowsDependencyResult {
        dependency_name: "kernel32.dll".to_string(),
        valid: true,
        found_path: Some(PathBuf::from("C:\\Windows\\System32\\kernel32.dll")),
        version_mismatch: None,
        suggestions: Vec::new(),
    };
    
    let test_result = WindowsTestResult {
        success: true,
        execution_results: vec![execution_result],
        compatibility_results: vec![compatibility_result],
        dependency_results: vec![dependency_result],
        test_duration: Duration::from_millis(200),
    };
    
    assert!(test_result.success);
    assert_eq!(test_result.execution_results.len(), 1);
    assert_eq!(test_result.compatibility_results.len(), 1);
    assert_eq!(test_result.dependency_results.len(), 1);
    assert_eq!(test_result.test_duration, Duration::from_millis(200));
}