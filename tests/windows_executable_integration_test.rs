// Integration test for Windows executable generation and testing
// Tests the complete Windows-specific functionality for task 10

use aether_language::build_system::{
    WindowsExecutableGenerator, WindowsExecutableConfig, WindowsExecutableTester,
    WindowsTestConfig, WindowsArchitecture, WindowsSubsystem, WindowsVersion,
    WindowsRuntimeDetector, WindowsEnvironmentHandler, WindowsCompatibilityChecker,
};
use std::path::Path;
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_windows_pe_executable_generation() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create a mock input binary
    let input_binary = temp_dir.path().join("input.exe");
    fs::write(&input_binary, "mock binary content").unwrap();
    
    // Create output path
    let output_path = temp_dir.path().join("output.exe");
    
    // Create Windows executable generator
    let mut generator = WindowsExecutableGenerator::new();
    
    // Generate Windows executable
    let result = generator.generate_executable(&input_binary, &output_path);
    
    assert!(result.is_ok());
    let exe_result = result.unwrap();
    assert!(exe_result.success);
    assert!(exe_result.executable_path.exists());
    assert_eq!(exe_result.pe_info.signature, "PE00");
}

#[test]
fn test_windows_runtime_dependency_detection() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create a mock executable
    let mock_exe = temp_dir.path().join("test.exe");
    fs::write(&mock_exe, "mock executable").unwrap();
    
    // Create dependency detector
    let mut detector = WindowsRuntimeDetector::new();
    
    // Detect dependencies
    let dependencies = detector.detect_dependencies(&mock_exe);
    
    assert!(dependencies.is_ok());
    let deps = dependencies.unwrap();
    assert!(!deps.is_empty());
    
    // Should detect some system DLLs
    let has_system_dll = deps.iter().any(|dep| dep.is_system_dll);
    assert!(has_system_dll);
    
    // Should detect some runtime DLLs
    let has_runtime_dll = deps.iter().any(|dep| !dep.is_system_dll);
    assert!(has_runtime_dll);
}

#[test]
fn test_windows_environment_detection() {
    let mut handler = WindowsEnvironmentHandler::new();
    
    // Test environment variable handling
    handler.set_env_var("AETHER_TEST_VAR", "test_value").unwrap();
    let value = handler.get_env_var("AETHER_TEST_VAR").unwrap();
    assert_eq!(value, Some("test_value".to_string()));
    
    // Test runtime environment detection
    let runtime_env = handler.detect_runtime_environment();
    assert!(runtime_env.is_ok());
    
    let env = runtime_env.unwrap();
    // Should detect some Windows version
    assert!(matches!(env.version, 
        WindowsVersion::Windows10 | 
        WindowsVersion::Windows11 | 
        WindowsVersion::WindowsServer2019 | 
        WindowsVersion::WindowsServer2022
    ));
    
    // Should detect architecture
    assert!(matches!(env.architecture,
        WindowsArchitecture::X86 |
        WindowsArchitecture::X64 |
        WindowsArchitecture::ARM64
    ));
}

#[test]
fn test_windows_executable_testing() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create a mock executable that can be "executed"
    let mock_exe = temp_dir.path().join("test.exe");
    fs::write(&mock_exe, "mock executable").unwrap();
    
    // Create test configuration
    let test_config = WindowsTestConfig {
        timeout: Duration::from_secs(10),
        test_compatibility: true,
        test_versions: vec![WindowsVersion::Windows10],
        test_architectures: vec![WindowsArchitecture::X64],
        validate_dependencies: true,
    };
    
    // Create tester
    let mut tester = WindowsExecutableTester::with_config(test_config);
    
    // Test the executable (this will fail since it's not a real executable, but we test the structure)
    let test_result = tester.test_executable(&mock_exe);
    
    // The test should complete (even if it fails execution)
    assert!(test_result.is_ok());
    let result = test_result.unwrap();
    
    // Should have execution results
    assert!(!result.execution_results.is_empty());
    
    // Should have compatibility results
    assert!(!result.compatibility_results.is_empty());
    
    // Should have dependency results
    assert!(!result.dependency_results.is_empty());
}

#[test]
fn test_windows_compatibility_checker() {
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
    );
    
    assert!(compat_result.is_ok());
    let result = compat_result.unwrap();
    
    // Should provide compatibility information
    assert_eq!(result.windows_version, WindowsVersion::Windows10);
    assert_eq!(result.architecture, WindowsArchitecture::X64);
}

#[test]
fn test_windows_executable_config_variants() {
    // Test different architecture configurations
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
    
    // Test different subsystem configurations
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
fn test_windows_registry_and_environment_handling() {
    let mut handler = WindowsEnvironmentHandler::new();
    
    // Test registry value access (simulated)
    let registry_value = handler.get_registry_value(
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
        "ProductName"
    );
    
    // Should return some result (even if simulated)
    assert!(registry_value.is_ok());
    
    // Test environment variable operations
    let test_vars = vec![
        ("PATH", "should exist"),
        ("SYSTEMROOT", "should exist"),
        ("PROGRAMFILES", "should exist"),
    ];
    
    for (var_name, _description) in test_vars {
        let result = handler.get_env_var(var_name);
        assert!(result.is_ok());
        // Most of these should exist on Windows systems
    }
}

#[test]
fn test_windows_version_info_structure() {
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
fn test_pe_format_info_structure() {
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