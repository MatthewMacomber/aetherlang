// Aether Standard Library Tests
// Comprehensive tests for core data structures, math, I/O, and string processing

use aether::stdlib::*;
use aether::compiler::autodiff::{DiffMarker};
use std::fs;
use std::path::Path;

#[cfg(test)]
mod collections_tests {
    use super::*;

    #[test]
    fn test_linear_vec_ownership() {
        let vec = LinearVec::new()
            .push(1)
            .push(2)
            .push(3);
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(1), Some(&2));
        
        // Test consuming operations
        let (vec, popped) = vec.pop();
        assert_eq!(popped, Some(3));
        assert_eq!(vec.len(), 2);
        
        // Test functional operations
        let doubled = vec.map(|x| x * 2);
        assert_eq!(doubled.get(0), Some(&2));
        assert_eq!(doubled.get(1), Some(&4));
    }

    #[test]
    fn test_linear_map_operations() {
        let (map, _) = LinearMap::new().insert("key1", 10);
        let (map, _) = map.insert("key2", 20);
        let (map, _) = map.insert("key3", 30);
        
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"key2"), Some(&20));
        assert!(map.contains_key(&"key1"));
        
        let keys = map.keys();
        assert_eq!(keys.len(), 3);
        
        let (map, removed) = map.remove(&"key2");
        assert_eq!(removed, Some(20));
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&"key2"));
    }

    #[test]
    fn test_linear_set_operations() {
        let (set1, _) = LinearSet::new().insert(1);
        let (set1, _) = set1.insert(2);
        let (set1, _) = set1.insert(3);
        
        let (set2, _) = LinearSet::new().insert(2);
        let (set2, _) = set2.insert(3);
        let (set2, _) = set2.insert(4);
        
        assert_eq!(set1.len(), 3);
        assert_eq!(set2.len(), 3);
        
        let union = set1.clone().union(set2.clone());
        assert_eq!(union.len(), 4);
        
        let intersection = set1.intersection(set2);
        assert_eq!(intersection.len(), 2);
        assert!(intersection.contains(&2));
        assert!(intersection.contains(&3));
    }

    #[test]
    fn test_collection_memory_optimization() {
        let vec = LinearVec::new().push(1).push(2).push(3);
        let optimizer = CollectionUtils::optimize_vector_layout(&vec);
        
        // Test that optimizer is created successfully
        // In a real implementation, this would test actual memory layout optimization
        
        let memory_usage = CollectionUtils::estimate_memory_usage::<i32>(100);
        assert!(memory_usage > 0);
    }
}

#[cfg(test)]
mod math_tests {
    use super::*;
    use math::*;

    #[test]
    fn test_diff_scalar_arithmetic() {
        let x = DiffScalar::with_gradient(3.0, 1.0);
        let y = DiffScalar::with_gradient(4.0, 1.0);
        
        let sum = x + y;
        assert_eq!(sum.value(), 7.0);
        assert_eq!(sum.gradient(), Some(2.0));
        
        let x = DiffScalar::with_gradient(3.0, 1.0);
        let y = DiffScalar::with_gradient(4.0, 1.0);
        let product = x * y;
        assert_eq!(product.value(), 12.0);
        assert_eq!(product.gradient(), Some(7.0)); // 1*4 + 3*1
    }

    #[test]
    fn test_trigonometric_functions() {
        use std::f64::consts::PI;
        
        let x = DiffScalar::with_gradient(PI / 2.0, 1.0);
        let sin_result = trig::sin(x);
        assert!((sin_result.value() - 1.0).abs() < 1e-10);
        assert!((sin_result.gradient().unwrap() - 0.0).abs() < 1e-10); // cos(π/2) = 0
        
        let x = DiffScalar::with_gradient(0.0, 1.0);
        let cos_result = trig::cos(x);
        assert_eq!(cos_result.value(), 1.0);
        assert_eq!(cos_result.gradient(), Some(0.0)); // -sin(0) = 0
        
        let x = DiffScalar::with_gradient(PI / 4.0, 1.0);
        let tan_result = trig::tan(x);
        assert!((tan_result.value() - 1.0).abs() < 1e-10);
        assert!((tan_result.gradient().unwrap() - 2.0).abs() < 1e-10); // sec²(π/4) = 2
    }

    #[test]
    fn test_exponential_functions() {
        let x = DiffScalar::with_gradient(0.0, 1.0);
        let exp_result = exp_log::exp(x);
        assert_eq!(exp_result.value(), 1.0);
        assert_eq!(exp_result.gradient(), Some(1.0));
        
        let x = DiffScalar::with_gradient(1.0, 1.0);
        let ln_result = exp_log::ln(x);
        assert_eq!(ln_result.value(), 0.0);
        assert_eq!(ln_result.gradient(), Some(1.0));
        
        let x = DiffScalar::with_gradient(4.0, 1.0);
        let sqrt_result = exp_log::sqrt(x);
        assert_eq!(sqrt_result.value(), 2.0);
        assert_eq!(sqrt_result.gradient(), Some(0.25)); // 1/(2*sqrt(4)) = 1/4
    }

    #[test]
    fn test_power_function() {
        let base = DiffScalar::with_gradient(2.0, 1.0);
        let exponent = DiffScalar::new(3.0);
        
        let result = exp_log::pow(base, exponent);
        assert_eq!(result.value(), 8.0); // 2³ = 8
        assert_eq!(result.gradient(), Some(12.0)); // 3 * 2² = 12
        
        let base = DiffScalar::new(2.0);
        let exponent = DiffScalar::with_gradient(3.0, 1.0);
        
        let result = exp_log::pow(base, exponent);
        assert_eq!(result.value(), 8.0); // 2³ = 8
        let expected_grad = 8.0 * (2.0_f64).ln(); // 8 * ln(2)
        assert!((result.gradient().unwrap() - expected_grad).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let x = DiffScalar::with_gradient(0.0, 1.0);
        
        let sinh_result = hyperbolic::sinh(x);
        assert_eq!(sinh_result.value(), 0.0);
        assert_eq!(sinh_result.gradient(), Some(1.0)); // cosh(0) = 1
        
        let cosh_result = hyperbolic::cosh(DiffScalar::with_gradient(0.0, 1.0));
        assert_eq!(cosh_result.value(), 1.0);
        assert_eq!(cosh_result.gradient(), Some(0.0)); // sinh(0) = 0
        
        let tanh_result = hyperbolic::tanh(DiffScalar::with_gradient(0.0, 1.0));
        assert_eq!(tanh_result.value(), 0.0);
        assert_eq!(tanh_result.gradient(), Some(1.0)); // 1 - tanh²(0) = 1
    }

    #[test]
    fn test_statistical_functions() {
        let values = LinearVec::new()
            .push(1.0)
            .push(2.0)
            .push(3.0)
            .push(4.0)
            .push(5.0);
        
        assert_eq!(stats::mean(&values), 3.0);
        assert_eq!(stats::variance(&values), 2.5);
        assert!((stats::std_dev(&values) - 1.5811388300841898).abs() < 1e-10);
        
        let x = DiffScalar::with_gradient(0.0, 1.0);
        let pdf = stats::normal_pdf(x, 0.0, 1.0);
        let expected = 1.0 / (2.0 * constants::PI).sqrt();
        assert!((pdf.value() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_math_operations() {
        use aether::runtime::tensor::{Tensor, TensorData, TensorDType};
        
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = Tensor::new(
            TensorData::Float32(data),
            vec![2, 2],
        ).unwrap();
        
        let sin_result = tensor_math::TensorMath::sin(&tensor).unwrap();
        if let TensorData::Float32(result_data) = sin_result.data() {
            assert!((result_data[0] - 0.0f32.sin()).abs() < 1e-6);
            assert!((result_data[1] - 1.0f32.sin()).abs() < 1e-6);
            assert!((result_data[2] - 2.0f32.sin()).abs() < 1e-6);
            assert!((result_data[3] - 3.0f32.sin()).abs() < 1e-6);
        } else {
            panic!("Expected Float32 tensor data");
        }
        
        let exp_result = tensor_math::TensorMath::exp(&tensor).unwrap();
        if let TensorData::Float32(result_data) = exp_result.data() {
            assert!((result_data[0] - 0.0f32.exp()).abs() < 1e-6);
            assert!((result_data[1] - 1.0f32.exp()).abs() < 1e-6);
        } else {
            panic!("Expected Float32 tensor data");
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        use aether::runtime::tensor::{Tensor, TensorData, TensorDType};
        
        // 2x3 matrix
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(
            TensorData::Float32(a_data),
            vec![2, 3],
        ).unwrap();
        
        // 3x2 matrix
        let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = Tensor::new(
            TensorData::Float32(b_data),
            vec![3, 2],
        ).unwrap();
        
        let result = tensor_math::TensorMath::matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &vec![2, 2]);
        
        if let TensorData::Float32(result_data) = result.data() {
            // Expected result: [[58, 64], [139, 154]]
            assert_eq!(result_data[0], 58.0); // 1*7 + 2*9 + 3*11
            assert_eq!(result_data[1], 64.0); // 1*8 + 2*10 + 3*12
            assert_eq!(result_data[2], 139.0); // 4*7 + 5*9 + 6*11
            assert_eq!(result_data[3], 154.0); // 4*8 + 5*10 + 6*12
        } else {
            panic!("Expected Float32 tensor data");
        }
    }
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_linear_file_operations() {
        let test_path = "test_stdlib_file.txt";
        let test_content = "Hello, Aether Standard Library!";
        
        // Create and write to file
        let file = LinearFile::create(test_path).unwrap();
        let file = file.write_string(test_content).unwrap();
        
        // Verify file exists
        assert!(file.exists());
        
        // Read from file
        let (file, content) = file.read_to_string().unwrap();
        assert_eq!(content, test_content);
        
        // Test file size
        let size = file.size().unwrap();
        assert_eq!(size, test_content.len() as u64);
        
        // Clean up
        file.delete().unwrap();
        assert!(!Path::new(test_path).exists());
    }

    #[test]
    fn test_buffered_io_operations() {
        let test_path = "test_stdlib_buffered.txt";
        let lines = vec!["First line", "Second line", "Third line"];
        
        // Write lines using buffered writer
        let mut writer = LinearWriter::new(test_path).unwrap();
        for line in &lines {
            writer = writer.write_line(line).unwrap();
        }
        writer.close().unwrap();
        
        // Read lines using buffered reader
        let reader = LinearReader::new(test_path).unwrap();
        let read_lines = reader.read_lines().unwrap();
        
        assert_eq!(read_lines.len(), 3);
        for i in 0..3 {
            assert_eq!(read_lines.get(i).unwrap(), &lines[i].to_string());
        }
        
        // Test reading exact bytes
        let reader = LinearReader::new(test_path).unwrap();
        let (_, bytes) = reader.read_exact(5).unwrap();
        assert_eq!(bytes.len(), 5);
        
        // Clean up
        fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_directory_operations() {
        let test_dir = "test_stdlib_directory";
        
        // Create directory
        let dir = LinearDirectory::create(test_dir).unwrap();
        assert!(dir.path().exists());
        
        // Create test files in directory
        let test_file1 = Path::new(test_dir).join("file1.txt");
        let test_file2 = Path::new(test_dir).join("file2.txt");
        fs::write(&test_file1, "content1").unwrap();
        fs::write(&test_file2, "content2").unwrap();
        
        // Create subdirectory
        let sub_dir = Path::new(test_dir).join("subdir");
        fs::create_dir(&sub_dir).unwrap();
        
        // List contents
        let (dir, contents) = dir.list_contents().unwrap();
        assert_eq!(contents.len(), 3); // 2 files + 1 directory
        
        // List files only
        let (dir, files) = dir.list_files().unwrap();
        assert_eq!(files.len(), 2);
        
        // List directories only
        let (dir, dirs) = dir.list_dirs().unwrap();
        assert_eq!(dirs.len(), 1);
        
        // Clean up
        dir.delete_recursive().unwrap();
        assert!(!Path::new(test_dir).exists());
    }

    #[test]
    fn test_io_error_handling() {
        // Test file not found error
        let result = LinearFile::open("nonexistent_file.txt");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IOError::FileNotFound(_) => {},
            _ => panic!("Expected FileNotFound error"),
        }
        
        // Test invalid directory
        let result = LinearDirectory::open("nonexistent_directory");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IOError::FileNotFound(_) => {},
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_serialization() {
        use io::serialization::JsonSerializer;
        
        // Test vector serialization
        let vec = LinearVec::new()
            .push("apple".to_string())
            .push("banana".to_string())
            .push("cherry".to_string());
        
        let json = JsonSerializer::serialize_vec(&vec).unwrap();
        assert_eq!(json, "[\"apple\", \"banana\", \"cherry\"]");
        
        // Test map serialization
        let (map, _) = LinearMap::new().insert("fruit".to_string(), "apple".to_string());
        let (map, _) = map.insert("color".to_string(), "red".to_string());
        
        let json = JsonSerializer::serialize_map(&map).unwrap();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("fruit") && json.contains("apple"));
        assert!(json.contains("color") && json.contains("red"));
    }
}

#[cfg(test)]
mod string_tests {
    use super::*;

    #[test]
    fn test_linear_string_basic_operations() {
        let s1 = LinearString::from_str("Hello");
        let s2 = LinearString::from_str(" World");
        
        let combined = s1.append(s2);
        assert_eq!(combined.as_str(), "Hello World");
        assert_eq!(combined.len(), 11);
        assert_eq!(combined.char_count(), 11);
        
        let upper = combined.to_uppercase();
        assert_eq!(upper.as_str(), "HELLO WORLD");
        
        let lower = upper.to_lowercase();
        assert_eq!(lower.as_str(), "hello world");
        
        let trimmed = LinearString::from_str("  test  ").trim();
        assert_eq!(trimmed.as_str(), "test");
    }

    #[test]
    fn test_string_manipulation() {
        let text = LinearString::from_str("Hello World Hello");
        
        // Test replacement
        let replaced_all = text.clone().replace("Hello", "Hi");
        assert_eq!(replaced_all.as_str(), "Hi World Hi");
        
        let replaced_first = text.replace_first("Hello", "Hi");
        assert_eq!(replaced_first.as_str(), "Hi World Hello");
        
        // Test substring operations
        let sub = text.substring(0, 5).unwrap();
        assert_eq!(sub.as_str(), "Hello");
        
        // Test character operations
        assert_eq!(text.char_at(0), Some('H'));
        assert_eq!(text.char_at(6), Some('W'));
        
        // Test search operations
        assert_eq!(text.find("World"), Some(6));
        assert_eq!(text.rfind("Hello"), Some(12));
        assert!(text.contains("World"));
        assert!(text.starts_with("Hello"));
        assert!(text.ends_with("Hello"));
    }

    #[test]
    fn test_string_splitting() {
        let text = LinearString::from_str("apple,banana,cherry");
        let parts = text.split(",");
        
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.get(0).unwrap().as_str(), "apple");
        assert_eq!(parts.get(1).unwrap().as_str(), "banana");
        assert_eq!(parts.get(2).unwrap().as_str(), "cherry");
        
        let text = LinearString::from_str("one two three four");
        let words = text.split_whitespace();
        assert_eq!(words.len(), 4);
        assert_eq!(words.get(0).unwrap().as_str(), "one");
        assert_eq!(words.get(3).unwrap().as_str(), "four");
        
        let multiline = LinearString::from_str("line1\nline2\nline3");
        let lines = multiline.lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines.get(1).unwrap().as_str(), "line2");
    }

    #[test]
    fn test_string_utilities() {
        // Test joining
        let strings = LinearVec::new()
            .push(LinearString::from_str("apple"))
            .push(LinearString::from_str("banana"))
            .push(LinearString::from_str("cherry"));
        
        let joined = StringUtils::join(strings, ", ");
        assert_eq!(joined.as_str(), "apple, banana, cherry");
        
        // Test type checking
        let numeric = LinearString::from_str("12345");
        assert!(StringUtils::is_numeric(&numeric));
        
        let alpha = LinearString::from_str("hello");
        assert!(StringUtils::is_alphabetic(&alpha));
        
        let alphanum = LinearString::from_str("hello123");
        assert!(StringUtils::is_alphanumeric(&alphanum));
        
        // Test conversions
        let int_str = LinearString::from_str("42");
        assert_eq!(StringUtils::to_int(&int_str).unwrap(), 42);
        
        let float_str = LinearString::from_str("3.14");
        assert!((StringUtils::to_float(&float_str).unwrap() - 3.14).abs() < 1e-10);
        
        let from_int = StringUtils::from_int(123);
        assert_eq!(from_int.as_str(), "123");
        
        let from_float = StringUtils::from_float(2.718);
        assert_eq!(from_float.as_str(), "2.718");
    }

    #[test]
    fn test_levenshtein_distance() {
        let s1 = LinearString::from_str("kitten");
        let s2 = LinearString::from_str("sitting");
        
        let distance = StringUtils::levenshtein_distance(&s1, &s2);
        assert_eq!(distance, 3);
        
        let s3 = LinearString::from_str("hello");
        let s4 = LinearString::from_str("hello");
        let distance2 = StringUtils::levenshtein_distance(&s3, &s4);
        assert_eq!(distance2, 0);
        
        let s5 = LinearString::from_str("abc");
        let s6 = LinearString::from_str("def");
        let distance3 = StringUtils::levenshtein_distance(&s5, &s6);
        assert_eq!(distance3, 3);
    }

    #[test]
    fn test_string_padding_and_formatting() {
        let s = LinearString::from_str("test");
        
        let padded_left = s.clone().pad_left(8, '0');
        assert_eq!(padded_left.as_str(), "0000test");
        
        let padded_right = s.pad_right(8, '-');
        assert_eq!(padded_right.as_str(), "test----");
        
        let reversed = LinearString::from_str("hello").reverse();
        assert_eq!(reversed.as_str(), "olleh");
        
        let repeated = LinearString::from_str("abc").repeat(3);
        assert_eq!(repeated.as_str(), "abcabcabc");
    }

    #[test]
    fn test_basic_regex() {
        use string::regex::*;
        
        // Test literal matching
        let regex = Regex::new("hello").unwrap();
        let text = LinearString::from_str("hello world");
        
        let m = regex.find(&text).unwrap();
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 5);
        assert_eq!(m.text.as_str(), "hello");
        
        assert!(regex.is_match(&LinearString::from_str("hello")));
        assert!(!regex.is_match(&LinearString::from_str("world")));
        
        // Test replacement
        let replaced = regex.replace_first(text, "hi");
        assert_eq!(replaced.as_str(), "hi world");
        
        // Test find all
        let text = LinearString::from_str("hello world hello");
        let matches = regex.find_all(&text);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches.get(0).unwrap().start, 0);
        assert_eq!(matches.get(1).unwrap().start, 12);
    }

    #[test]
    fn test_regex_splitting() {
        use string::regex::*;
        
        let regex = Regex::new(",").unwrap();
        let text = LinearString::from_str("a,b,c,d");
        let parts = regex.split(text);
        
        assert_eq!(parts.len(), 4);
        assert_eq!(parts.get(0).unwrap().as_str(), "a");
        assert_eq!(parts.get(1).unwrap().as_str(), "b");
        assert_eq!(parts.get(2).unwrap().as_str(), "c");
        assert_eq!(parts.get(3).unwrap().as_str(), "d");
    }

    #[test]
    fn test_string_equality() {
        let s1 = LinearString::from_str("hello");
        let s2 = LinearString::from_str("hello");
        let s3 = LinearString::from_str("world");
        
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1, "hello");
        assert_eq!(s1, "hello".to_string());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_math_with_collections() {
        // Test using mathematical functions with collections
        let values = LinearVec::new()
            .push(DiffScalar::with_gradient(1.0, 1.0))
            .push(DiffScalar::with_gradient(2.0, 1.0))
            .push(DiffScalar::with_gradient(3.0, 1.0));
        
        // Apply sine function to all values
        let sin_values = values.map(|x| math::trig::sin(x));
        
        assert_eq!(sin_values.len(), 3);
        assert!((sin_values.get(0).unwrap().value() - 1.0_f64.sin()).abs() < 1e-10);
        assert!((sin_values.get(1).unwrap().value() - 2.0_f64.sin()).abs() < 1e-10);
        assert!((sin_values.get(2).unwrap().value() - 3.0_f64.sin()).abs() < 1e-10);
    }

    #[test]
    fn test_io_with_string_processing() {
        let test_path = "test_integration.txt";
        let lines = vec![
            "apple,red,sweet",
            "banana,yellow,sweet", 
            "lemon,yellow,sour"
        ];
        
        // Write CSV-like data
        let mut writer = LinearWriter::new(test_path).unwrap();
        for line in &lines {
            writer = writer.write_line(line).unwrap();
        }
        writer.close().unwrap();
        
        // Read and process the data
        let reader = LinearReader::new(test_path).unwrap();
        let read_lines = reader.read_lines().unwrap();
        
        // Parse CSV data using string processing
        let mut parsed_data = LinearVec::new();
        for i in 0..read_lines.len() {
            if let Some(line) = read_lines.get(i) {
                let linear_line = LinearString::from_str(line);
                let fields = linear_line.split(",");
                parsed_data = parsed_data.push(fields);
            }
        }
        
        assert_eq!(parsed_data.len(), 3);
        
        // Check first row
        if let Some(first_row) = parsed_data.get(0) {
            assert_eq!(first_row.len(), 3);
            assert_eq!(first_row.get(0).unwrap().as_str(), "apple");
            assert_eq!(first_row.get(1).unwrap().as_str(), "red");
            assert_eq!(first_row.get(2).unwrap().as_str(), "sweet");
        }
        
        // Clean up
        fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_collections_with_string_processing() {
        // Create a map of string data
        let (mut data_map, _) = LinearMap::new().insert(
            LinearString::from_str("fruits"),
            LinearString::from_str("apple,banana,cherry")
        );
        let (data_map, _) = data_map.insert(
            LinearString::from_str("colors"),
            LinearString::from_str("red,yellow,red")
        );
        
        // Process the data
        let fruits_key = LinearString::from_str("fruits");
        if let Some(fruits_str) = data_map.get(&fruits_key) {
            let fruits = fruits_str.clone().split(",");
            assert_eq!(fruits.len(), 3);
            assert_eq!(fruits.get(0).unwrap().as_str(), "apple");
        }
        
        // Create a set of processed strings
        let mut word_set = LinearSet::new();
        let keys = data_map.keys();
        
        for i in 0..keys.len() {
            if let Some(key) = keys.get(i) {
                let upper_key = (*key).clone().to_uppercase();
                let (new_set, _) = word_set.insert(upper_key);
                word_set = new_set;
            }
        }
        
        assert_eq!(word_set.len(), 2);
        assert!(word_set.contains(&LinearString::from_str("FRUITS")));
        assert!(word_set.contains(&LinearString::from_str("COLORS")));
    }

    #[test]
    fn test_performance_with_large_collections() {
        // Test performance with larger datasets
        let mut large_vec = LinearVec::new();
        
        // Create a large vector
        for i in 0..1000 {
            large_vec = large_vec.push(i);
        }
        
        assert_eq!(large_vec.len(), 1000);
        
        // Test functional operations on large dataset
        let doubled = large_vec.map(|x| x * 2);
        assert_eq!(doubled.len(), 1000);
        assert_eq!(doubled.get(0), Some(&0));
        assert_eq!(doubled.get(999), Some(&1998));
        
        // Test filtering
        let evens = LinearVec::new()
            .push(1).push(2).push(3).push(4).push(5).push(6)
            .filter(|&x| x % 2 == 0);
        assert_eq!(evens.len(), 3);
        
        // Test folding
        let sum = LinearVec::new()
            .push(1).push(2).push(3).push(4).push(5)
            .fold(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }
}