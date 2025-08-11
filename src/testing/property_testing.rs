// Property-Based Testing Framework
// Provides property-based testing capabilities for Aether language components

use super::{TestCase, TestResult, TestContext};
use std::time::{Duration, Instant};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Property test configuration
#[derive(Debug, Clone)]
pub struct PropertyConfig {
    pub test_cases: usize,
    pub max_shrink_iterations: usize,
    pub seed: Option<u64>,
    pub timeout_per_case: Option<Duration>,
    pub verbose_shrinking: bool,
}

impl Default for PropertyConfig {
    fn default() -> Self {
        PropertyConfig {
            test_cases: 100,
            max_shrink_iterations: 100,
            seed: None,
            timeout_per_case: Some(Duration::from_millis(100)),
            verbose_shrinking: false,
        }
    }
}

/// Trait for generating test data
pub trait Arbitrary: Clone + std::fmt::Debug {
    fn arbitrary(rng: &mut StdRng) -> Self;
    fn shrink(&self) -> Vec<Self> { Vec::new() }
}

/// Property test case
pub struct PropertyTest<T> {
    pub name: String,
    pub property: Box<dyn Fn(&T) -> Result<(), String> + Send + Sync>,
    pub config: PropertyConfig,
    pub generator: Box<dyn Fn(&mut StdRng) -> T + Send + Sync>,
    pub shrinker: Option<Box<dyn Fn(&T) -> Vec<T> + Send + Sync>>,
}

impl<T> PropertyTest<T>
where
    T: Clone + std::fmt::Debug + 'static,
{
    pub fn new<F, G>(name: &str, generator: G, property: F) -> Self
    where
        F: Fn(&T) -> Result<(), String> + Send + Sync + 'static,
        G: Fn(&mut StdRng) -> T + Send + Sync + 'static,
    {
        PropertyTest {
            name: name.to_string(),
            property: Box::new(property),
            config: PropertyConfig::default(),
            generator: Box::new(generator),
            shrinker: None,
        }
    }

    pub fn with_config(mut self, config: PropertyConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_shrinker<S>(mut self, shrinker: S) -> Self
    where
        S: Fn(&T) -> Vec<T> + Send + Sync + 'static,
    {
        self.shrinker = Some(Box::new(shrinker));
        self
    }

    pub fn run_property(&self) -> PropertyTestResult<T> {
        let seed = self.config.seed.unwrap_or_else(|| rand::random());
        let mut rng = StdRng::seed_from_u64(seed);
        
        let mut passed_cases = 0;
        let mut total_time = Duration::new(0, 0);
        
        for case_num in 0..self.config.test_cases {
            let test_data = (self.generator)(&mut rng);
            let start = Instant::now();
            
            match (self.property)(&test_data) {
                Ok(()) => {
                    passed_cases += 1;
                    total_time += start.elapsed();
                }
                Err(error) => {
                    // Property failed, try to shrink
                    let shrunk_data = self.shrink_failing_case(&test_data, &error);
                    return PropertyTestResult::Failed {
                        case_number: case_num,
                        original_input: test_data,
                        shrunk_input: shrunk_data.clone(),
                        error: error.clone(),
                        shrink_steps: 0, // TODO: track shrink steps
                        seed,
                    };
                }
            }
            
            // Check timeout per case
            if let Some(timeout) = self.config.timeout_per_case {
                if start.elapsed() > timeout {
                    return PropertyTestResult::Timeout {
                        case_number: case_num,
                        input: test_data,
                        timeout,
                        seed,
                    };
                }
            }
        }
        
        PropertyTestResult::Passed {
            cases_tested: passed_cases,
            total_time,
            seed,
        }
    }

    fn shrink_failing_case(&self, original: &T, original_error: &str) -> Option<T> {
        if let Some(shrinker) = &self.shrinker {
            let mut current = original.clone();
            let mut current_error = original_error.to_string();
            
            for _ in 0..self.config.max_shrink_iterations {
                let candidates = shrinker(&current);
                let mut found_smaller = false;
                
                for candidate in candidates {
                    if let Err(error) = (self.property)(&candidate) {
                        // This candidate also fails, use it as the new current
                        current = candidate;
                        current_error = error;
                        found_smaller = true;
                        
                        if self.config.verbose_shrinking {
                            println!("Shrunk to: {:?}", current);
                        }
                        break;
                    }
                }
                
                if !found_smaller {
                    break;
                }
            }
            
            Some(current)
        } else {
            None
        }
    }
}

impl<T> TestCase for PropertyTest<T>
where
    T: Clone + std::fmt::Debug + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let start = Instant::now();
        let result = self.run_property();
        let duration = start.elapsed();

        match result {
            PropertyTestResult::Passed { cases_tested, .. } => {
                TestResult::passed(self.name.clone(), duration)
                    .with_metadata("cases_tested".to_string(), cases_tested.to_string())
            }
            PropertyTestResult::Failed { case_number, original_input, shrunk_input, error, seed, .. } => {
                let message = if let Some(shrunk) = shrunk_input {
                    format!("Property failed at case {}: {} (shrunk from {:?} to {:?})", 
                           case_number, error, original_input, shrunk)
                } else {
                    format!("Property failed at case {}: {} (input: {:?})", 
                           case_number, error, original_input)
                };
                TestResult::failed(self.name.clone(), duration, message)
                    .with_metadata("seed".to_string(), seed.to_string())
                    .with_metadata("case_number".to_string(), case_number.to_string())
            }
            PropertyTestResult::Timeout { case_number, input, timeout, seed } => {
                let message = format!("Property test timed out at case {} after {:?} (input: {:?})", 
                                    case_number, timeout, input);
                TestResult::error(self.name.clone(), duration, message)
                    .with_metadata("seed".to_string(), seed.to_string())
                    .with_metadata("timeout".to_string(), format!("{:?}", timeout))
            }
        }
    }
}

/// Result of running a property test
#[derive(Debug, Clone)]
pub enum PropertyTestResult<T> {
    Passed {
        cases_tested: usize,
        total_time: Duration,
        seed: u64,
    },
    Failed {
        case_number: usize,
        original_input: T,
        shrunk_input: Option<T>,
        error: String,
        shrink_steps: usize,
        seed: u64,
    },
    Timeout {
        case_number: usize,
        input: T,
        timeout: Duration,
        seed: u64,
    },
}

/// Built-in generators for common types
pub struct Generators;

impl Generators {
    pub fn i32_range(min: i32, max: i32) -> impl Fn(&mut StdRng) -> i32 {
        move |rng| rng.gen_range(min..=max)
    }

    pub fn f64_range(min: f64, max: f64) -> impl Fn(&mut StdRng) -> f64 {
        move |rng| rng.gen_range(min..=max)
    }

    pub fn bool() -> impl Fn(&mut StdRng) -> bool {
        |rng| rng.gen()
    }

    pub fn string(max_len: usize) -> impl Fn(&mut StdRng) -> String {
        move |rng| {
            let len = rng.gen_range(0..=max_len);
            (0..len)
                .map(|_| rng.gen_range(b'a'..=b'z') as char)
                .collect()
        }
    }

    pub fn vec<T, F>(generator: F, max_len: usize) -> impl Fn(&mut StdRng) -> Vec<T>
    where
        F: Fn(&mut StdRng) -> T + Clone,
    {
        move |rng| {
            let len = rng.gen_range(0..=max_len);
            (0..len).map(|_| generator(rng)).collect()
        }
    }

    pub fn option<T, F>(generator: F, none_probability: f64) -> impl Fn(&mut StdRng) -> Option<T>
    where
        F: Fn(&mut StdRng) -> T,
    {
        move |rng| {
            if rng.gen::<f64>() < none_probability {
                None
            } else {
                Some(generator(rng))
            }
        }
    }

    pub fn tuple2<T1, T2, F1, F2>(gen1: F1, gen2: F2) -> impl Fn(&mut StdRng) -> (T1, T2)
    where
        F1: Fn(&mut StdRng) -> T1,
        F2: Fn(&mut StdRng) -> T2,
    {
        move |rng| (gen1(rng), gen2(rng))
    }
}

/// Built-in shrinkers for common types
pub struct Shrinkers;

impl Shrinkers {
    pub fn i32(value: &i32) -> Vec<i32> {
        let mut shrinks = Vec::new();
        
        // Shrink towards zero
        if *value != 0 {
            shrinks.push(0);
        }
        
        // Shrink by halving
        if value.abs() > 1 {
            shrinks.push(value / 2);
            if *value > 0 {
                shrinks.push(value - 1);
            } else {
                shrinks.push(value + 1);
            }
        }
        
        shrinks
    }

    pub fn f64(value: &f64) -> Vec<f64> {
        let mut shrinks = Vec::new();
        
        if *value != 0.0 {
            shrinks.push(0.0);
        }
        
        if value.abs() > 1.0 {
            shrinks.push(value / 2.0);
        }
        
        shrinks
    }

    pub fn string(value: &str) -> Vec<String> {
        let mut shrinks = Vec::new();
        
        // Empty string
        if !value.is_empty() {
            shrinks.push(String::new());
        }
        
        // Remove characters from the end
        if value.len() > 1 {
            shrinks.push(value[..value.len() - 1].to_string());
        }
        
        // Remove characters from the middle
        if value.len() > 2 {
            let mid = value.len() / 2;
            let mut chars: Vec<char> = value.chars().collect();
            chars.remove(mid);
            shrinks.push(chars.into_iter().collect());
        }
        
        shrinks
    }

    pub fn vec<T: Clone>(value: &[T]) -> Vec<Vec<T>> {
        let mut shrinks = Vec::new();
        
        // Empty vector
        if !value.is_empty() {
            shrinks.push(Vec::new());
        }
        
        // Remove elements
        if value.len() > 1 {
            // Remove last element
            shrinks.push(value[..value.len() - 1].to_vec());
            
            // Remove first element
            shrinks.push(value[1..].to_vec());
            
            // Remove middle element
            if value.len() > 2 {
                let mid = value.len() / 2;
                let mut vec = value.to_vec();
                vec.remove(mid);
                shrinks.push(vec);
            }
        }
        
        shrinks
    }
}

/// Macro for creating property tests
#[macro_export]
macro_rules! property_test {
    ($name:expr, $generator:expr, $property:expr) => {
        PropertyTest::new($name, $generator, $property)
    };
    ($name:expr, $generator:expr, cases = $cases:expr, $property:expr) => {
        PropertyTest::new($name, $generator, $property)
            .with_config(PropertyConfig {
                test_cases: $cases,
                ..PropertyConfig::default()
            })
    };
}

// Implement Arbitrary for common types
impl Arbitrary for i32 {
    fn arbitrary(rng: &mut StdRng) -> Self {
        rng.gen()
    }

    fn shrink(&self) -> Vec<Self> {
        Shrinkers::i32(self)
    }
}

impl Arbitrary for f64 {
    fn arbitrary(rng: &mut StdRng) -> Self {
        rng.gen()
    }

    fn shrink(&self) -> Vec<Self> {
        Shrinkers::f64(self)
    }
}

impl Arbitrary for bool {
    fn arbitrary(rng: &mut StdRng) -> Self {
        rng.gen()
    }
}

impl Arbitrary for String {
    fn arbitrary(rng: &mut StdRng) -> Self {
        let len = rng.gen_range(0..=20);
        (0..len)
            .map(|_| rng.gen_range(b'a'..=b'z') as char)
            .collect()
    }

    fn shrink(&self) -> Vec<Self> {
        Shrinkers::string(self)
    }
}

impl<T: Arbitrary> Arbitrary for Vec<T> {
    fn arbitrary(rng: &mut StdRng) -> Self {
        let len = rng.gen_range(0..=10);
        (0..len).map(|_| T::arbitrary(rng)).collect()
    }

    fn shrink(&self) -> Vec<Self> {
        Shrinkers::vec(self)
    }
}

impl<T: Arbitrary> Arbitrary for Option<T> {
    fn arbitrary(rng: &mut StdRng) -> Self {
        if rng.gen_bool(0.3) {
            None
        } else {
            Some(T::arbitrary(rng))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_config_default() {
        let config = PropertyConfig::default();
        assert_eq!(config.test_cases, 100);
        assert_eq!(config.max_shrink_iterations, 100);
        assert!(config.seed.is_none());
        assert!(!config.verbose_shrinking);
    }

    #[test]
    fn test_generators() {
        let mut rng = StdRng::seed_from_u64(42);
        
        let int_gen = Generators::i32_range(1, 10);
        let value = int_gen(&mut rng);
        assert!(value >= 1 && value <= 10);
        
        let string_gen = Generators::string(5);
        let s = string_gen(&mut rng);
        assert!(s.len() <= 5);
        
        // TODO: Fix vec generator with closures
        // let vec_gen = Generators::vec(Generators::i32_range(0, 100), 3);
        // let vec = vec_gen(&mut rng);
        // assert!(vec.len() <= 3);
        // assert!(vec.iter().all(|&x| x >= 0 && x <= 100));
    }

    #[test]
    fn test_shrinkers() {
        let shrinks = Shrinkers::i32(&42);
        assert!(shrinks.contains(&0));
        assert!(shrinks.contains(&21));
        
        let shrinks = Shrinkers::string("hello");
        assert!(shrinks.contains(&String::new()));
        assert!(shrinks.contains(&"hell".to_string()));
        
        let shrinks = Shrinkers::vec(&vec![1, 2, 3, 4]);
        assert!(shrinks.contains(&vec![]));
        assert!(shrinks.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn test_arbitrary_implementations() {
        let mut rng = StdRng::seed_from_u64(123);
        
        let _: i32 = Arbitrary::arbitrary(&mut rng);
        let _: f64 = Arbitrary::arbitrary(&mut rng);
        let _: bool = Arbitrary::arbitrary(&mut rng);
        let _: String = Arbitrary::arbitrary(&mut rng);
        let _: Vec<i32> = Arbitrary::arbitrary(&mut rng);
        let _: Option<i32> = Arbitrary::arbitrary(&mut rng);
    }

    #[test]
    fn test_simple_property() {
        let property = PropertyTest::new(
            "addition_commutative",
            |rng: &mut StdRng| (rng.gen::<i32>() % 1000, rng.gen::<i32>() % 1000),
            |(a, b)| {
                if a + b == b + a {
                    Ok(())
                } else {
                    Err("Addition is not commutative".to_string())
                }
            }
        ).with_config(PropertyConfig {
            test_cases: 10,
            ..PropertyConfig::default()
        });

        let result = property.run_property();
        match result {
            PropertyTestResult::Passed { cases_tested, .. } => {
                assert_eq!(cases_tested, 10);
            }
            _ => panic!("Property should have passed"),
        }
    }
}