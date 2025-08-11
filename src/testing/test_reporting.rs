// Test Reporting
// Comprehensive test result reporting and analysis

use super::{TestResult, TestStatus, test_runner::{TestRunResult, TestStatistics}};
use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize};

/// Report format options
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFormat {
    Console,
    Json,
    Xml,
    Html,
    Markdown,
    Csv,
}

/// Report configuration
#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub format: ReportFormat,
    pub output_path: Option<String>,
    pub include_passed: bool,
    pub include_failed: bool,
    pub include_errors: bool,
    pub include_skipped: bool,
    pub include_statistics: bool,
    pub include_metadata: bool,
    pub verbose: bool,
    pub color_output: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        ReportConfig {
            format: ReportFormat::Console,
            output_path: None,
            include_passed: true,
            include_failed: true,
            include_errors: true,
            include_skipped: true,
            include_statistics: true,
            include_metadata: false,
            verbose: false,
            color_output: true,
        }
    }
}

/// Test reporter for generating various report formats
pub struct TestReporter {
    config: ReportConfig,
}

impl TestReporter {
    pub fn new(config: ReportConfig) -> Self {
        TestReporter { config }
    }

    /// Generate report from test run results
    pub fn generate_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        match self.config.format {
            ReportFormat::Console => self.generate_console_report(run_result),
            ReportFormat::Json => self.generate_json_report(run_result),
            ReportFormat::Xml => self.generate_xml_report(run_result),
            ReportFormat::Html => self.generate_html_report(run_result),
            ReportFormat::Markdown => self.generate_markdown_report(run_result),
            ReportFormat::Csv => self.generate_csv_report(run_result),
        }
    }

    /// Save report to file
    pub fn save_report(&self, report: &str) -> Result<(), String> {
        if let Some(path) = &self.config.output_path {
            fs::write(path, report)
                .map_err(|e| format!("Failed to write report to {}: {}", path, e))?;
        }
        Ok(())
    }

    /// Generate and save report in one step
    pub fn generate_and_save(&self, run_result: &TestRunResult) -> Result<String, String> {
        let report = self.generate_report(run_result)?;
        self.save_report(&report)?;
        Ok(report)
    }

    fn generate_console_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let mut report = String::new();
        
        // Header
        report.push_str(&self.format_header("Test Results"));
        report.push('\n');

        // Summary
        report.push_str(&self.format_summary(run_result));
        report.push('\n');

        // Statistics
        if self.config.include_statistics {
            let stats = TestStatistics::from_results(&run_result.results);
            report.push_str(&self.format_statistics(&stats));
            report.push('\n');
        }

        // Test details
        report.push_str(&self.format_test_details(run_result));

        Ok(report)
    }

    fn generate_json_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let json_report = JsonReport::from_run_result(run_result);
        serde_json::to_string_pretty(&json_report)
            .map_err(|e| format!("Failed to serialize JSON report: {}", e))
    }

    fn generate_xml_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<testsuites>\n");
        xml.push_str(&format!("  <testsuite name=\"AetherTests\" tests=\"{}\" failures=\"{}\" errors=\"{}\" time=\"{:.3}\">\n",
                             run_result.total_tests, run_result.failed, run_result.errors, run_result.duration.as_secs_f64()));

        for result in &run_result.results {
            xml.push_str(&format!("    <testcase name=\"{}\" time=\"{:.3}\"",
                                 result.name, result.duration.as_secs_f64()));
            
            match result.status {
                TestStatus::Passed => {
                    xml.push_str(" />\n");
                }
                TestStatus::Failed => {
                    xml.push_str(">\n");
                    xml.push_str(&format!("      <failure message=\"{}\">{}</failure>\n",
                                         result.message.as_deref().unwrap_or("Test failed"),
                                         result.error_details.as_deref().unwrap_or("")));
                    xml.push_str("    </testcase>\n");
                }
                TestStatus::Error => {
                    xml.push_str(">\n");
                    xml.push_str(&format!("      <error message=\"{}\">{}</error>\n",
                                         result.message.as_deref().unwrap_or("Test error"),
                                         result.error_details.as_deref().unwrap_or("")));
                    xml.push_str("    </testcase>\n");
                }
                TestStatus::Skipped => {
                    xml.push_str(">\n");
                    xml.push_str("      <skipped />\n");
                    xml.push_str("    </testcase>\n");
                }
                TestStatus::Timeout => {
                    xml.push_str(">\n");
                    xml.push_str(&format!("      <error message=\"Timeout\">{}</error>\n",
                                         result.message.as_deref().unwrap_or("Test timed out")));
                    xml.push_str("    </testcase>\n");
                }
            }
        }

        xml.push_str("  </testsuite>\n");
        xml.push_str("</testsuites>\n");
        Ok(xml)
    }

    fn generate_html_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <title>Aether Test Report</title>\n");
        html.push_str("  <style>\n");
        html.push_str(include_str!("report_styles.css"));
        html.push_str("  </style>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("  <h1>Aether Test Report</h1>\n");

        // Summary
        html.push_str("  <div class=\"summary\">\n");
        html.push_str(&format!("    <h2>Summary</h2>\n"));
        html.push_str(&format!("    <p>Total Tests: {}</p>\n", run_result.total_tests));
        html.push_str(&format!("    <p class=\"passed\">Passed: {}</p>\n", run_result.passed));
        html.push_str(&format!("    <p class=\"failed\">Failed: {}</p>\n", run_result.failed));
        html.push_str(&format!("    <p class=\"error\">Errors: {}</p>\n", run_result.errors));
        html.push_str(&format!("    <p>Duration: {:?}</p>\n", run_result.duration));
        html.push_str("  </div>\n");

        // Statistics
        if self.config.include_statistics {
            let stats = TestStatistics::from_results(&run_result.results);
            html.push_str("  <div class=\"statistics\">\n");
            html.push_str("    <h2>Statistics</h2>\n");
            html.push_str(&format!("    <p>Success Rate: {:.1}%</p>\n", stats.success_rate * 100.0));
            html.push_str(&format!("    <p>Tests per Second: {:.2}</p>\n", stats.tests_per_second));
            html.push_str(&format!("    <p>Average Duration: {:?}</p>\n", stats.average_duration));
            html.push_str("  </div>\n");
        }

        // Test results
        html.push_str("  <div class=\"results\">\n");
        html.push_str("    <h2>Test Results</h2>\n");
        html.push_str("    <table>\n");
        html.push_str("      <tr><th>Test Name</th><th>Status</th><th>Duration</th><th>Message</th></tr>\n");

        for result in &run_result.results {
            let status_class = match result.status {
                TestStatus::Passed => "passed",
                TestStatus::Failed => "failed",
                TestStatus::Error => "error",
                TestStatus::Skipped => "skipped",
                TestStatus::Timeout => "timeout",
            };

            html.push_str(&format!("      <tr class=\"{}\">\n", status_class));
            html.push_str(&format!("        <td>{}</td>\n", result.name));
            html.push_str(&format!("        <td>{:?}</td>\n", result.status));
            html.push_str(&format!("        <td>{:?}</td>\n", result.duration));
            html.push_str(&format!("        <td>{}</td>\n", 
                                 result.message.as_deref().unwrap_or("")));
            html.push_str("      </tr>\n");
        }

        html.push_str("    </table>\n");
        html.push_str("  </div>\n");
        html.push_str("</body>\n</html>\n");

        Ok(html)
    }

    fn generate_markdown_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let mut md = String::new();
        
        md.push_str("# Aether Test Report\n\n");

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Total Tests**: {}\n", run_result.total_tests));
        md.push_str(&format!("- **Passed**: {} ✅\n", run_result.passed));
        md.push_str(&format!("- **Failed**: {} ❌\n", run_result.failed));
        md.push_str(&format!("- **Errors**: {} ⚠️\n", run_result.errors));
        md.push_str(&format!("- **Skipped**: {} ⏭️\n", run_result.skipped));
        md.push_str(&format!("- **Duration**: {:?}\n", run_result.duration));
        md.push_str(&format!("- **Success Rate**: {:.1}%\n\n", run_result.success_rate() * 100.0));

        // Statistics
        if self.config.include_statistics {
            let stats = TestStatistics::from_results(&run_result.results);
            md.push_str("## Statistics\n\n");
            md.push_str(&format!("- **Average Duration**: {:?}\n", stats.average_duration));
            md.push_str(&format!("- **Tests per Second**: {:.2}\n", stats.tests_per_second));
            
            if let Some((name, duration)) = &stats.fastest_test {
                md.push_str(&format!("- **Fastest Test**: {} ({:?})\n", name, duration));
            }
            
            if let Some((name, duration)) = &stats.slowest_test {
                md.push_str(&format!("- **Slowest Test**: {} ({:?})\n", name, duration));
            }
            
            md.push_str("\n");
        }

        // Failed tests
        let failed_tests = run_result.failed_tests();
        if !failed_tests.is_empty() {
            md.push_str("## Failed Tests\n\n");
            for result in failed_tests {
                md.push_str(&format!("### {} ❌\n\n", result.name));
                md.push_str(&format!("- **Status**: {:?}\n", result.status));
                md.push_str(&format!("- **Duration**: {:?}\n", result.duration));
                
                if let Some(message) = &result.message {
                    md.push_str(&format!("- **Message**: {}\n", message));
                }
                
                if let Some(error) = &result.error_details {
                    md.push_str(&format!("- **Error Details**:\n```\n{}\n```\n", error));
                }
                
                md.push_str("\n");
            }
        }

        // All test results (if verbose)
        if self.config.verbose {
            md.push_str("## All Test Results\n\n");
            md.push_str("| Test Name | Status | Duration | Message |\n");
            md.push_str("|-----------|--------|----------|----------|\n");

            for result in &run_result.results {
                let status_emoji = match result.status {
                    TestStatus::Passed => "✅",
                    TestStatus::Failed => "❌",
                    TestStatus::Error => "⚠️",
                    TestStatus::Skipped => "⏭️",
                    TestStatus::Timeout => "⏰",
                };

                md.push_str(&format!("| {} | {} {:?} | {:?} | {} |\n",
                                   result.name,
                                   status_emoji,
                                   result.status,
                                   result.duration,
                                   result.message.as_deref().unwrap_or("")));
            }
        }

        Ok(md)
    }

    fn generate_csv_report(&self, run_result: &TestRunResult) -> Result<String, String> {
        let mut csv = String::new();
        
        // Header
        csv.push_str("Test Name,Status,Duration (ms),Message,Error Details\n");

        // Data rows
        for result in &run_result.results {
            csv.push_str(&format!("\"{}\",{},{},\"{}\",\"{}\"\n",
                                 result.name,
                                 format!("{:?}", result.status),
                                 result.duration.as_millis(),
                                 result.message.as_deref().unwrap_or("").replace("\"", "\"\""),
                                 result.error_details.as_deref().unwrap_or("").replace("\"", "\"\"")));
        }

        Ok(csv)
    }

    fn format_header(&self, title: &str) -> String {
        if self.config.color_output {
            format!("\x1b[1;36m{}\x1b[0m", title)
        } else {
            title.to_string()
        }
    }

    fn format_summary(&self, run_result: &TestRunResult) -> String {
        let mut summary = String::new();
        
        if self.config.color_output {
            summary.push_str(&format!("Tests: \x1b[32m{} passed\x1b[0m, \x1b[31m{} failed\x1b[0m, \x1b[33m{} errors\x1b[0m, {} skipped ({} total) in {:?}\n",
                                    run_result.passed, run_result.failed, run_result.errors, run_result.skipped, run_result.total_tests, run_result.duration));
        } else {
            summary.push_str(&run_result.summary());
            summary.push('\n');
        }

        summary.push_str(&format!("Success rate: {:.1}%\n", run_result.success_rate() * 100.0));
        summary
    }

    fn format_statistics(&self, stats: &TestStatistics) -> String {
        let mut output = String::new();
        output.push_str("Statistics:\n");
        output.push_str(&format!("  Average duration: {:?}\n", stats.average_duration));
        output.push_str(&format!("  Tests per second: {:.2}\n", stats.tests_per_second));
        
        if let Some((name, duration)) = &stats.fastest_test {
            output.push_str(&format!("  Fastest test: {} ({:?})\n", name, duration));
        }
        
        if let Some((name, duration)) = &stats.slowest_test {
            output.push_str(&format!("  Slowest test: {} ({:?})\n", name, duration));
        }
        
        output
    }

    fn format_test_details(&self, run_result: &TestRunResult) -> String {
        let mut details = String::new();

        // Failed tests
        if self.config.include_failed {
            let failed_tests = run_result.failed_tests();
            if !failed_tests.is_empty() {
                details.push_str("Failed Tests:\n");
                for result in failed_tests {
                    details.push_str(&self.format_test_result(result));
                }
                details.push('\n');
            }
        }

        // All tests (if verbose)
        if self.config.verbose {
            details.push_str("All Test Results:\n");
            for result in &run_result.results {
                if self.should_include_result(result) {
                    details.push_str(&self.format_test_result(result));
                }
            }
        }

        details
    }

    fn format_test_result(&self, result: &TestResult) -> String {
        let mut output = String::new();
        
        let status_symbol = match result.status {
            TestStatus::Passed => if self.config.color_output { "\x1b[32m✓\x1b[0m" } else { "✓" },
            TestStatus::Failed => if self.config.color_output { "\x1b[31m✗\x1b[0m" } else { "✗" },
            TestStatus::Error => if self.config.color_output { "\x1b[33m⚠\x1b[0m" } else { "⚠" },
            TestStatus::Skipped => if self.config.color_output { "\x1b[36m-\x1b[0m" } else { "-" },
            TestStatus::Timeout => if self.config.color_output { "\x1b[35m⏰\x1b[0m" } else { "⏰" },
        };

        output.push_str(&format!("  {} {} ({:?})\n", status_symbol, result.name, result.duration));

        if let Some(message) = &result.message {
            output.push_str(&format!("    Message: {}\n", message));
        }

        if let Some(error) = &result.error_details {
            output.push_str(&format!("    Error: {}\n", error));
        }

        if self.config.include_metadata && !result.metadata.is_empty() {
            output.push_str("    Metadata:\n");
            for (key, value) in &result.metadata {
                output.push_str(&format!("      {}: {}\n", key, value));
            }
        }

        output
    }

    fn should_include_result(&self, result: &TestResult) -> bool {
        match result.status {
            TestStatus::Passed => self.config.include_passed,
            TestStatus::Failed => self.config.include_failed,
            TestStatus::Error => self.config.include_errors,
            TestStatus::Skipped => self.config.include_skipped,
            TestStatus::Timeout => self.config.include_errors,
        }
    }
}

/// JSON report structure
#[derive(Debug, Serialize, Deserialize)]
struct JsonReport {
    summary: JsonSummary,
    statistics: JsonStatistics,
    results: Vec<JsonTestResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonSummary {
    total_tests: usize,
    passed: usize,
    failed: usize,
    errors: usize,
    skipped: usize,
    duration_ms: u64,
    success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonStatistics {
    average_duration_ms: u64,
    tests_per_second: f64,
    fastest_test: Option<(String, u64)>,
    slowest_test: Option<(String, u64)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonTestResult {
    name: String,
    status: String,
    duration_ms: u64,
    message: Option<String>,
    error_details: Option<String>,
    metadata: HashMap<String, String>,
}

impl JsonReport {
    fn from_run_result(run_result: &TestRunResult) -> Self {
        let stats = TestStatistics::from_results(&run_result.results);
        
        JsonReport {
            summary: JsonSummary {
                total_tests: run_result.total_tests,
                passed: run_result.passed,
                failed: run_result.failed,
                errors: run_result.errors,
                skipped: run_result.skipped,
                duration_ms: run_result.duration.as_millis() as u64,
                success_rate: run_result.success_rate(),
            },
            statistics: JsonStatistics {
                average_duration_ms: stats.average_duration.as_millis() as u64,
                tests_per_second: stats.tests_per_second,
                fastest_test: stats.fastest_test.map(|(name, dur)| (name, dur.as_millis() as u64)),
                slowest_test: stats.slowest_test.map(|(name, dur)| (name, dur.as_millis() as u64)),
            },
            results: run_result.results.iter().map(|r| JsonTestResult {
                name: r.name.clone(),
                status: format!("{:?}", r.status),
                duration_ms: r.duration.as_millis() as u64,
                message: r.message.clone(),
                error_details: r.error_details.clone(),
                metadata: r.metadata.clone(),
            }).collect(),
        }
    }
}

/// Report comparison utilities
pub struct ReportComparison;

impl ReportComparison {
    /// Compare two test run results
    pub fn compare_runs(baseline: &TestRunResult, current: &TestRunResult) -> ComparisonReport {
        let baseline_stats = TestStatistics::from_results(&baseline.results);
        let current_stats = TestStatistics::from_results(&current.results);

        ComparisonReport {
            baseline_summary: format!("{} tests, {} passed, {} failed", 
                                    baseline.total_tests, baseline.passed, baseline.failed),
            current_summary: format!("{} tests, {} passed, {} failed", 
                                   current.total_tests, current.passed, current.failed),
            success_rate_change: current.success_rate() - baseline.success_rate(),
            performance_change: current_stats.average_duration.as_secs_f64() / baseline_stats.average_duration.as_secs_f64() - 1.0,
            new_failures: Self::find_new_failures(baseline, current),
            fixed_tests: Self::find_fixed_tests(baseline, current),
        }
    }

    fn find_new_failures(baseline: &TestRunResult, current: &TestRunResult) -> Vec<String> {
        let baseline_failed: std::collections::HashSet<_> = baseline.results.iter()
            .filter(|r| r.status == TestStatus::Failed || r.status == TestStatus::Error)
            .map(|r| &r.name)
            .collect();

        current.results.iter()
            .filter(|r| (r.status == TestStatus::Failed || r.status == TestStatus::Error) 
                       && !baseline_failed.contains(&r.name))
            .map(|r| r.name.clone())
            .collect()
    }

    fn find_fixed_tests(baseline: &TestRunResult, current: &TestRunResult) -> Vec<String> {
        let current_failed: std::collections::HashSet<_> = current.results.iter()
            .filter(|r| r.status == TestStatus::Failed || r.status == TestStatus::Error)
            .map(|r| &r.name)
            .collect();

        baseline.results.iter()
            .filter(|r| (r.status == TestStatus::Failed || r.status == TestStatus::Error) 
                       && !current_failed.contains(&r.name))
            .map(|r| r.name.clone())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub baseline_summary: String,
    pub current_summary: String,
    pub success_rate_change: f64,
    pub performance_change: f64,
    pub new_failures: Vec<String>,
    pub fixed_tests: Vec<String>,
}

// CSS styles for HTML reports (would normally be in a separate file)
const CSS_STYLES: &str = r#"
body { font-family: Arial, sans-serif; margin: 20px; }
h1, h2 { color: #333; }
.summary, .statistics, .results { margin: 20px 0; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #f2f2f2; }
.passed { color: green; }
.failed { color: red; }
.error { color: orange; }
.skipped { color: blue; }
.timeout { color: purple; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_sample_results() -> TestRunResult {
        TestRunResult {
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
        }
    }

    #[test]
    fn test_report_config_default() {
        let config = ReportConfig::default();
        assert_eq!(config.format, ReportFormat::Console);
        assert!(config.include_passed);
        assert!(config.include_failed);
        assert!(config.color_output);
    }

    #[test]
    fn test_console_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Console,
            color_output: false,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        assert!(report.contains("Test Results"));
        assert!(report.contains("2 passed"));
        assert!(report.contains("1 failed"));
        assert!(report.contains("Success rate: 66.7%"));
    }

    #[test]
    fn test_json_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Json,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&report).unwrap();
        assert!(parsed["summary"]["total_tests"].as_u64().unwrap() == 3);
        assert!(parsed["summary"]["passed"].as_u64().unwrap() == 2);
        assert!(parsed["summary"]["failed"].as_u64().unwrap() == 1);
    }

    #[test]
    fn test_xml_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Xml,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        assert!(report.contains("<?xml version=\"1.0\""));
        assert!(report.contains("<testsuites>"));
        assert!(report.contains("<testsuite"));
        assert!(report.contains("tests=\"3\""));
        assert!(report.contains("failures=\"1\""));
    }

    #[test]
    fn test_markdown_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Markdown,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        assert!(report.contains("# Aether Test Report"));
        assert!(report.contains("## Summary"));
        assert!(report.contains("**Total Tests**: 3"));
        assert!(report.contains("**Passed**: 2 ✅"));
        assert!(report.contains("**Failed**: 1 ❌"));
    }

    #[test]
    fn test_csv_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Csv,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        assert!(report.contains("Test Name,Status,Duration (ms),Message,Error Details"));
        assert!(report.contains("test1,Passed,50"));
        assert!(report.contains("test3,Failed,25"));
    }

    #[test]
    fn test_html_report_generation() {
        let config = ReportConfig {
            format: ReportFormat::Html,
            ..ReportConfig::default()
        };
        let reporter = TestReporter::new(config);
        let run_result = create_sample_results();

        let report = reporter.generate_report(&run_result).unwrap();
        
        assert!(report.contains("<!DOCTYPE html>"));
        assert!(report.contains("<title>Aether Test Report</title>"));
        assert!(report.contains("<h1>Aether Test Report</h1>"));
        assert!(report.contains("Total Tests: 3"));
    }

    #[test]
    fn test_report_comparison() {
        let baseline = TestRunResult {
            total_tests: 2,
            passed: 1,
            failed: 1,
            errors: 0,
            skipped: 0,
            duration: Duration::from_millis(100),
            results: vec![
                TestResult::passed("test1".to_string(), Duration::from_millis(50)),
                TestResult::failed("test2".to_string(), Duration::from_millis(50), "error".to_string()),
            ],
        };

        let current = TestRunResult {
            total_tests: 2,
            passed: 2,
            failed: 0,
            errors: 0,
            skipped: 0,
            duration: Duration::from_millis(80),
            results: vec![
                TestResult::passed("test1".to_string(), Duration::from_millis(40)),
                TestResult::passed("test2".to_string(), Duration::from_millis(40)),
            ],
        };

        let comparison = ReportComparison::compare_runs(&baseline, &current);
        
        assert!(comparison.success_rate_change > 0.0); // Improvement
        assert!(comparison.performance_change < 0.0); // Faster
        assert!(comparison.fixed_tests.contains(&"test2".to_string()));
        assert!(comparison.new_failures.is_empty());
    }
}