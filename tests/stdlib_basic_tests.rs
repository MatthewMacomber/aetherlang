// Basic Standard Library Tests
// Tests for core functionality without complex dependencies

#[cfg(test)]
mod basic_collections_tests {
    use aether::stdlib::collections::{LinearVec, LinearMap, LinearSet};

    #[test]
    fn test_linear_vec_basic() {
        let vec = LinearVec::new()
            .push(1)
            .push(2)
            .push(3);
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(1), Some(&2));
        
        let (vec, popped) = vec.pop();
        assert_eq!(popped, Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_linear_map_basic() {
        let (map, _) = LinearMap::new().insert("key1", 10);
        let (map, _) = map.insert("key2", 20);
        
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"key1"), Some(&10));
        assert!(map.contains_key(&"key2"));
    }

    #[test]
    fn test_linear_set_basic() {
        let (set, _) = LinearSet::new().insert(1);
        let (set, _) = set.insert(2);
        
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
    }
}

#[cfg(test)]
mod basic_string_tests {
    use aether::stdlib::string::{LinearString, StringUtils};
    use aether::stdlib::collections::LinearVec;

    #[test]
    fn test_linear_string_basic() {
        let s1 = LinearString::from_str("Hello");
        let s2 = LinearString::from_str(" World");
        
        let combined = s1.append(s2);
        assert_eq!(combined.as_str(), "Hello World");
        assert_eq!(combined.len(), 11);
    }

    #[test]
    fn test_string_utilities() {
        let strings = LinearVec::new()
            .push(LinearString::from_str("apple"))
            .push(LinearString::from_str("banana"));
        
        let joined = StringUtils::join(strings, ", ");
        assert_eq!(joined.as_str(), "apple, banana");
        
        let numeric = LinearString::from_str("12345");
        assert!(StringUtils::is_numeric(&numeric));
    }
}

#[cfg(test)]
mod basic_math_tests {
    use aether::stdlib::math::{DiffScalar, constants};

    #[test]
    fn test_diff_scalar_basic() {
        let x = DiffScalar::with_gradient(3.0, 1.0);
        let y = DiffScalar::with_gradient(4.0, 1.0);
        
        let sum = x + y;
        assert_eq!(sum.value(), 7.0);
        assert_eq!(sum.gradient(), Some(2.0));
    }

    #[test]
    fn test_constants() {
        assert!((constants::PI - 3.14159265358979323846).abs() < 1e-10);
        assert!((constants::E - 2.71828182845904523536).abs() < 1e-10);
    }

    #[test]
    fn test_trig_functions() {
        use aether::stdlib::math::trig;
        
        let x = DiffScalar::with_gradient(0.0, 1.0);
        let sin_result = trig::sin(x);
        assert_eq!(sin_result.value(), 0.0);
        assert_eq!(sin_result.gradient(), Some(1.0)); // cos(0) = 1
    }
}

#[cfg(test)]
mod basic_io_tests {
    use aether::stdlib::io::{LinearFile, IOError};
    use std::fs;

    #[test]
    fn test_linear_file_basic() {
        let test_path = "test_basic_file.txt";
        let test_content = "Hello, Basic Test!";
        
        // Create and write to file
        let file = LinearFile::create(test_path).unwrap();
        let file = file.write_string(test_content).unwrap();
        
        // Read from file
        let (_, content) = file.read_to_string().unwrap();
        assert_eq!(content, test_content);
        
        // Clean up
        let file = LinearFile::open(test_path).unwrap();
        file.delete().unwrap();
    }

    #[test]
    fn test_io_error_handling() {
        let result = LinearFile::open("nonexistent_file.txt");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IOError::FileNotFound(_) => {},
            _ => panic!("Expected FileNotFound error"),
        }
    }
}