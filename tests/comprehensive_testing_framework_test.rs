// Comprehensive Testing Framework Validation
// Tests the complete testing and validation framework implementation

use aether_language::testing::{
    TestConfig, TestContext, TestResult, TestStatus,
    unit_testing::{UnitTest, UnitTestSuite, UnitAssert},
    property_testing::{PropertyTest, PropertyConfig, Generators, Shrinkers},
    integration_testing::{IntegrationTest, IntegrationConfig, CompilationTarget, IntegrationScenarios},
    benchmark_testing::{BenchmarkTest, BenchmarkConfig, BenchmarkScenarios},
    correctness_validation::{CorrectnessTestSuite, CorrectnessConfig},
    meta_testing::{MetaTestSuite, MetaTestConfig},
    test_runner::{TestRunner, ExecutionStrategy, TestFilter},
    test_reporting::{TestReporter, ReportConfig, ReportFormat},
};
use std::time::Duration;

#[test]
fn test_unit_testing_framework() {
    // Test basic unit test functionality
    let test = UnitTest::new("sample_unit_test", || {
        UnitAssert::equals(2 + 2, 4)?;
        UnitAssert::is_true(true)?;
        UnitAssert::approx_equals(3.14159, 3.14160, 0.001)?;
        Ok(())
    });

    let mut context = TestContext::new(TestConfig::default());
    let result = test.run(&mut context);
    
    assert_eq!(result.status, TestStatus::Passed);
    assert_eq!(result.name, "sample_unit_test");
}

#[test]
fn test_unit_test_suite() {
    let suite = UnitTestSuite::new("arithmetic_tests")
        .add_test(UnitTest::new("addition", || {
            UnitAssert::equals(5 + 3, 8)
        }))
        .add_test(UnitTest::new("subtraction", || {
            UnitAssert::equals(10 - 4, 6)
        }))
        .add_test(UnitTest::new("multiplication", || {
            UnitAssert::equals(6 * 7, 42)
        }));

    assert_eq!(suite.name, "arithmetic_tests");
    assert_eq!(suite.tests.len(), 3);
}

#[test]
fn test_property_testing_framework() {
    // Test property that addition is commutative
    let commutative_property = PropertyTest::new(
        "addition_commutative",
        |rng| {
            let a = Generators::i32_range(-100, 100)(rng);
            let b = Generators::i32_range(-100, 100)(rng);
            (a, b)
        },
        |(a, b)| {
            if a + b == b + a {
                Ok(())
            } else {
                Err("Addition is not commutative".to_string())
            }
        }
    ).with_config(PropertyConfig {
        test_cases: 50,
        ..PropertyConfig::default()
    });

    let mut context = TestContext::new(TestConfig::default());
    let result = commutative_property.run(&mut context);
    
    assert_eq!(result.status, TestStatus::Passed);
}

#[test]
fn test_property_generators_and_shrinkers() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // Test generators
    let int_gen = Generators::i32_range(1, 10);
    let value = int_gen(&mut rng);
    assert!(value >= 1 && value <= 10);
    
    let string_gen = Generators::string(5);
    let s = string_gen(&mut rng);
    assert!(s.len() <= 5);
    
    // Test shrinkers
    let shrinks = Shrinkers::i32(&42);
    assert!(shrinks.contains(&0));
    assert!(shrinks.contains(&21));
    
    let string_shrinks = Shrinkers::string("hello");
    assert!(string_shrinks.contains(&String::new()));
    assert!(string_shrinks.contains(&"hell".to_string()));
}

#[test]
fn test_integration_testing_framework() {
    // Test integration test creation
    let test = IntegrationTest::new(
        "hello_world_test",
        r#"
        fn main() {
            println!("Hello, World!");
        }
        "#
    ).expect_output("Hello, World!")
     .expect_exit_code(0)
     .with_config(IntegrationConfig {
         timeout: Duration::from_secs(10),
         target_platforms: vec![CompilationTarget::Native],
         ..IntegrationConfig::default()
     });

    assert_eq!(test.name, "hello_world_test");
    assert_eq!(test.expected_output, Some("Hello, World!".to_string()));
    assert_eq!(test.expected_exit_code, 0);
}

#[test]
fn test_integration_scenarios() {
    let arithmetic_test = IntegrationScenarios::arithmetic_test();
    assert_eq!(arithmetic_test.name, "arithmetic_basic");
    assert!(arithmetic_test.source_code.contains("let a = 5"));
    
    let tensor_test = IntegrationScenarios::tensor_test();
    assert_eq!(tensor_test.name, "tensor_basic");
    assert!(tensor_test.source_code.contains("tensor"));
    
    let comprehensive_suite = IntegrationScenarios::comprehensive_suite();
    assert_eq!(comprehensive_suite.name, "comprehensive_integration");
    assert!(comprehensive_suite.tests.len() >= 5);
}

#[test]
fn test_benchmark_testing_framework() {
    let benchmark = BenchmarkTest::new(
        "simple_computation",
        || {
            // Simple computation for benchmarking
            let mut sum = 0;
            for i in 0..1000 {
                sum += i * i;
            }
            if sum > 0 { Ok(()) } else { Err("Invalid sum".to_string()) }
        }
    ).with_config(BenchmarkConfig {
        warmup_iterations: 3,
        measurement_iterations: 10,
        ..BenchmarkConfig::default()
    });

    let result = benchmark.run_benchmark();
    assert!(result.is_ok());
    
    let measurement = result.unwrap();
    assert_eq!(measurement.name, "simple_computation");
    assert_eq!(measurement.iterations, 10);
    assert!(measurement.mean_time > Duration::from_nanos(0));
}

#[test]
fn test_benchmark_scenarios() {
    let tensor_suite = BenchmarkScenarios::tensor_operations();
    assert_eq!(tensor_suite.name, "tensor_operations");
    assert_eq!(tensor_suite.benchmarks.len(), 3);
    
    let compilation_suite = BenchmarkScenarios::compilation_performance();
    assert_eq!(compilation_suite.name, "compilation_performance");
    assert_eq!(compilation_suite.benchmarks.len(), 3);
    
    let comprehensive_suites = BenchmarkScenarios::comprehensive_suite();
    assert_eq!(comprehensive_suites.len(), 4);
}

#[test]
fn test_correctness_validation_framework() {
    let suite = CorrectnessTestSuite::new("ai_correctness")
        .with_config(CorrectnessConfig {
            numerical_tolerance: 1e-6,
            gradient_tolerance: 1e-4,
            ..CorrectnessConfig::default()
        });

    let tests = suite.create_all_tests();
    assert!(!tests.is_empty());
    assert!(tests.len() >= 8); // Should have multiple correctness tests
    
    // Verify test names are unique
    let mut names = std::collections::HashSet::new();
    for test in &tests {
        assert!(names.insert(test.name()));
    }
}

#[test]
fn test_meta_testing_framework() {
    let config = MetaTestConfig {
        coverage_threshold: 0.8,
        reliability_threshold: 0.95,
        framework_completeness_checks: true,
        ..MetaTestConfig::default()
    };
    
    let suite = MetaTestSuite::new("framework_validation", config);
    let tests = suite.create_all_tests();
    
    assert_eq!(suite.name, "framework_validation");
    assert_eq!(tests.len(), 3); // Should have 3 meta-tests
    
    // Test names should be descriptive
    let test_names: Vec<&str> = tests.iter().map(|t| t.name()).collect();
    assert!(test_names.contains(&"framework_correctness"));
    assert!(test_names.contains(&"framework_completeness"));
    assert!(test_names.contains(&"framework_reliability"));
}

#[test]
fn test_test_runner() {
    let config = TestConfig {
        timeout: Duration::from_secs(5),
        parallel_execution: false, // Use sequential for predictable testing
        verbose: false,
        ..TestConfig::default()
    };
    
    let runner = TestRunner::new(config)
        .with_execution_strategy(ExecutionStrategy::Sequential);

    let tests: Vec<Box<dyn aether_language::testing::TestCase>> = vec![
        Box::new(UnitTest::new("test1", || Ok(()))),
        Box::new(UnitTest::new("test2", || Ok(()))),
        Box::new(UnitTest::new("test3", || Err("Expected failure".to_string()))),
    ];

    let result = runner.run_tests(tests);
    
    assert_eq!(result.total_tests, 3);
    assert_eq!(result.passed, 2);
    assert_eq!(result.failed, 1);
    assert_eq!(result.errors, 0);
    assert!(!result.is_successful());
    assert_eq!(result.success_rate(), 2.0 / 3.0);
}

#[test]
fn test_test_filter() {
    let filter = TestFilter {
        name_pattern: Some("unit".to_string()),
        tags: vec!["fast".to_string()],
        exclude_tags: vec!["slow".to_string()],
        test_types: vec!["unit".to_string()],
        only_failed: false,
    };

    assert!(filter.matches("unit_test_1", &["fast".to_string()], "unit"));
    assert!(!filter.matches("integration_test_1", &["fast".to_string()], "unit")); // Wrong name
    assert!(!filter.matches("unit_test_1", &["slow".to_string()], "unit")); // Excluded tag
    assert!(!filter.matches("unit_test_1", &["fast".to_string()], "integration")); // Wrong type
}

#[test]
fn test_test_reporting() {
    use aether_language::testing::test_runner::TestRunResult;
    
    // Create sample test results
    let run_result = TestRunResult {
        total_tests: 3,
        passed: 2,
        failed: 1,
        errors: 0,
        skipped: 0,
        duration: Duration::from_millis(150),
        results: vec![
            TestResult::passed("test1".to_string(), Duration::from_millis(50)),
            TestResult::passed("test2".to_string(), Duration::from_millis(75)),
            TestResult::failed("test3".to_string(), Duration::from_millis(25), "assertion failed".to_string()),
        ],
    };

    // Test console report
    let console_config = ReportConfig {
        format: ReportFormat::Console,
        color_output: false,
        ..ReportConfig::default()
    };
    let console_reporter = TestReporter::new(console_config);
    let console_report = console_reporter.generate_report(&run_result).unwrap();
    
    assert!(console_report.contains("Test Results"));
    assert!(console_report.contains("2 passed"));
    assert!(console_report.contains("1 failed"));

    // Test JSON report
    let json_config = ReportConfig {
        format: ReportFormat::Json,
        ..ReportConfig::default()
    };
    let json_reporter = TestReporter::new(json_config);
    let json_report = json_reporter.generate_report(&run_result).unwrap();
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json_report).unwrap();
    assert_eq!(parsed["summary"]["total_tests"].as_u64().unwrap(), 3);
    assert_eq!(parsed["summary"]["passed"].as_u64().unwrap(), 2);

    // Test Markdown report
    let md_config = ReportConfig {
        format: ReportFormat::Markdown,
        ..ReportConfig::default()
    };
    let md_reporter = TestReporter::new(md_config);
    let md_report = md_reporter.generate_report(&run_result).unwrap();
    
    assert!(md_report.contains("# Aether Test Report"));
    assert!(md_report.contains("**Total Tests**: 3"));
    assert!(md_report.contains("**Passed**: 2 ✅"));
}

#[test]
fn test_comprehensive_framework_integration() {
    // This test demonstrates the complete testing framework working together
    
    // 1. Create various types of tests
    let unit_tests: Vec<Box<dyn aether_language::testing::TestCase>> = vec![
        Box::new(UnitTest::new("unit_arithmetic", || {
            UnitAssert::equals(2 + 2, 4)?;
            UnitAssert::equals(5 * 6, 30)?;
            Ok(())
        })),
        Box::new(UnitTest::new("unit_strings", || {
            UnitAssert::equals("hello".len(), 5)?;
            UnitAssert::contains(&["a", "b", "c"], &"b")?;
            Ok(())
        })),
    ];

    let property_tests: Vec<Box<dyn aether_language::testing::TestCase>> = vec![
        Box::new(PropertyTest::new(
            "property_reverse_reverse",
            |rng| Generators::string(10)(rng),
            |s| {
                let reversed_twice: String = s.chars().rev().collect::<String>().chars().rev().collect();
                if reversed_twice == *s {
                    Ok(())
                } else {
                    Err("Reverse of reverse should equal original".to_string())
                }
            }
        ).with_config(PropertyConfig {
            test_cases: 20,
            ..PropertyConfig::default()
        })),
    ];

    let benchmark_tests: Vec<Box<dyn aether_language::testing::TestCase>> = vec![
        Box::new(BenchmarkTest::new(
            "benchmark_vector_sum",
            || {
                let vec: Vec<i32> = (0..1000).collect();
                let sum: i32 = vec.iter().sum();
                if sum > 0 { Ok(()) } else { Err("Invalid sum".to_string()) }
            }
        ).with_config(BenchmarkConfig {
            measurement_iterations: 5,
            ..BenchmarkConfig::default()
        })),
    ];

    // 2. Combine all tests
    let mut all_tests = Vec::new();
    all_tests.extend(unit_tests);
    all_tests.extend(property_tests);
    all_tests.extend(benchmark_tests);

    // 3. Run tests with runner
    let config = TestConfig {
        parallel_execution: false,
        verbose: false,
        ..TestConfig::default()
    };
    let runner = TestRunner::new(config);
    let run_result = runner.run_tests(all_tests);

    // 4. Verify results
    assert_eq!(run_result.total_tests, 4);
    assert!(run_result.passed >= 3); // Most tests should pass
    assert!(run_result.success_rate() > 0.5);

    // 5. Generate report
    let report_config = ReportConfig {
        format: ReportFormat::Console,
        color_output: false,
        include_statistics: true,
        ..ReportConfig::default()
    };
    let reporter = TestReporter::new(report_config);
    let report = reporter.generate_report(&run_result).unwrap();

    // 6. Verify report contains expected information
    assert!(report.contains("Test Results"));
    assert!(report.contains("Statistics"));
    assert!(report.len() > 100); // Should be a substantial report
}

#[test]
fn test_framework_completeness() {
    // Verify that all major testing framework components are available and functional
    
    // Unit testing
    let _unit_test = UnitTest::new("test", || Ok(()));
    let _unit_suite = UnitTestSuite::new("suite");
    
    // Property testing
    let _property_test = PropertyTest::new("prop", |rng| 42, |_| Ok(()));
    let _property_config = PropertyConfig::default();
    
    // Integration testing
    let _integration_test = IntegrationTest::new("int", "code");
    let _integration_config = IntegrationConfig::default();
    
    // Benchmark testing
    let _benchmark_test = BenchmarkTest::new("bench", || Ok(()));
    let _benchmark_config = BenchmarkConfig::default();
    
    // Correctness validation
    let _correctness_suite = CorrectnessTestSuite::new("correctness");
    let _correctness_config = CorrectnessConfig::default();
    
    // Meta testing
    let _meta_suite = MetaTestSuite::new("meta", MetaTestConfig::default());
    
    // Test runner
    let _runner = TestRunner::new(TestConfig::default());
    let _filter = TestFilter::default();
    
    // Test reporting
    let _reporter = TestReporter::new(ReportConfig::default());
    
    // If we reach here, all components are available
    assert!(true);
}

#[test]
fn test_error_handling_and_edge_cases() {
    // Test that the framework handles various error conditions gracefully
    
    // Test failing unit test
    let failing_test = UnitTest::new("failing_test", || {
        Err("This test is designed to fail".to_string())
    });
    
    let mut context = TestContext::new(TestConfig::default());
    let result = failing_test.run(&mut context);
    assert_eq!(result.status, TestStatus::Failed);
    assert!(result.message.is_some());
    
    // Test property test with always-failing property
    let failing_property = PropertyTest::new(
        "always_fails",
        |rng| Generators::i32_range(1, 10)(rng),
        |_| Err("Always fails".to_string())
    ).with_config(PropertyConfig {
        test_cases: 3,
        ..PropertyConfig::default()
    });
    
    let result = failing_property.run(&mut context);
    assert_eq!(result.status, TestStatus::Failed);
    
    // Test empty test suite
    let empty_tests: Vec<Box<dyn aether_language::testing::TestCase>> = vec![];
    let runner = TestRunner::new(TestConfig::default());
    let result = runner.run_tests(empty_tests);
    
    assert_eq!(result.total_tests, 0);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
    assert!(result.is_successful()); // Empty test suite is considered successful
}