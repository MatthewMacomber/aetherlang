// Tests for Sweet Syntax Transpiler
// Comprehensive tests for bidirectional transpilation and round-trip preservation

use aether_language::{sweet_to_sexpr, sexpr_to_sweet_string, SweetSyntaxTranspiler, parse_sexpr};

#[cfg(test)]
mod sweet_syntax_tests {
    use super::*;

    #[test]
    fn test_simple_identifiers() {
        let sweet = "hello";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "hello");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "hello");
    }

    #[test]
    fn test_numbers() {
        let sweet = "42";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "42");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "42");
    }

    #[test]
    fn test_floating_point_numbers() {
        let sweet = "3.14";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "3.14");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "3.14");
    }

    #[test]
    fn test_strings() {
        let sweet = "\"hello world\"";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "\"hello world\"");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "\"hello world\"");
    }

    #[test]
    fn test_booleans() {
        let sweet = "true";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "true");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "true");
        
        let sweet = "false";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "false");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "false");
    }

    #[test]
    fn test_nil() {
        let sweet = "nil";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "nil");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "nil");
    }

    #[test]
    fn test_simple_infix_expressions() {
        let sweet = "{x + y}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(+ x y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{x + y}");
    }

    #[test]
    fn test_complex_infix_expressions() {
        let sweet = "{x + y * z}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(+ x (* y z))");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{x + {y * z}}");
    }

    #[test]
    fn test_parenthesized_expressions() {
        let sweet = "{(x + y) * z}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(* (+ x y) z)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{{x + y} * z}");
    }

    #[test]
    fn test_function_calls() {
        let sweet = "add(x, y)";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(add x y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "add(x, y)");
    }

    #[test]
    fn test_function_calls_with_expressions() {
        let sweet = "add({x + 1}, {y * 2})";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(add (+ x 1) (* y 2))");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "add({x + 1}, {y * 2})");
    }

    #[test]
    fn test_nested_function_calls() {
        let sweet = "add(multiply(x, 2), y)";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(add (multiply x 2) y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "add(multiply(x, 2), y)");
    }

    #[test]
    fn test_comparison_operators() {
        let sweet = "{x == y}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(== x y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{x == y}");
    }

    #[test]
    fn test_logical_operators() {
        let sweet = "{x && y}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(&& x y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{x && y}");
    }

    #[test]
    fn test_unary_operators() {
        let sweet = "{!x}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(! x)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "(!x)");
    }

    #[test]
    fn test_negative_numbers() {
        let sweet = "{-42}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(- 42)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "(-42)");
    }

    #[test]
    fn test_operator_precedence() {
        // Test that multiplication has higher precedence than addition
        let sweet = "{a + b * c}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(+ a (* b c))");
        
        // Test that comparison has lower precedence than arithmetic
        let sweet = "{a + b == c * d}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(== (+ a b) (* c d))");
    }

    #[test]
    fn test_right_associative_assignment() {
        let sweet = "{a = b = c}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(= a (= b c))");
    }

    #[test]
    fn test_left_associative_arithmetic() {
        let sweet = "{a - b - c}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(- (- a b) c)");
    }

    #[test]
    fn test_mixed_function_calls_and_infix() {
        let sweet = "max({a + b}, {c * d})";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(max (+ a b) (* c d))");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "max({a + b}, {c * d})");
    }

    #[test]
    fn test_special_forms_not_converted_to_function_calls() {
        // Test that special forms like 'if' are not converted to function call syntax
        let sexpr = "(if (> x 0) x (- x))";
        let sweet = sexpr_to_sweet_string(sexpr).unwrap();
        assert_eq!(sweet, "(if {x > 0} x (-x))");
    }

    #[test]
    fn test_round_trip_preservation() {
        let test_cases = vec![
            "hello",
            "42",
            "3.14",
            "\"hello world\"",
            "true",
            "false",
            "nil",
            "{x + y}",
            "{x * y + z}",
            "add(x, y)",
            "max({a + b}, c)",
            "{x == y}",
            "{x && y}",
            "{!x}",
            "{a + b * c}",
        ];

        for test_case in test_cases {
            let sexpr = sweet_to_sexpr(test_case).unwrap();
            let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
            
            // Parse both versions to AST and compare for semantic equivalence
            let original_ast = {
                let mut transpiler = SweetSyntaxTranspiler::new(test_case).unwrap();
                transpiler.parse_to_sexpr().unwrap()
            };
            
            let round_trip_ast = {
                let mut transpiler = SweetSyntaxTranspiler::new(&back_to_sweet).unwrap();
                transpiler.parse_to_sexpr().unwrap()
            };
            
            // Compare the AST structures for semantic equivalence
            assert_eq!(format!("{}", original_ast.root), format!("{}", round_trip_ast.root),
                "Round-trip failed for: {} -> {} -> {}", test_case, sexpr, back_to_sweet);
        }
    }

    #[test]
    fn test_error_handling_unmatched_delimiters() {
        let result = sweet_to_sexpr("{x + y");
        assert!(result.is_err());
        
        let result = sweet_to_sexpr("add(x, y");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_handling_invalid_expressions() {
        let result = sweet_to_sexpr("{+}");
        assert!(result.is_err());
        
        let result = sweet_to_sexpr("add(,)");
        assert!(result.is_err());
    }

    #[test]
    fn test_string_escapes() {
        let sweet = "\"hello\\nworld\"";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "\"hello\nworld\"");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "\"hello\nworld\"");
    }

    #[test]
    fn test_complex_nested_expressions() {
        let sweet = "calculate({(a + b) * (c - d)}, max(x, y))";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(calculate (* (+ a b) (- c d)) (max x y))");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "calculate({{a + b} * {c - d}}, max(x, y))");
    }

    #[test]
    fn test_multiple_operators_same_precedence() {
        let sweet = "{a + b - c + d}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        // Left associative: ((a + b) - c) + d
        assert_eq!(sexpr, "(+ (- (+ a b) c) d)");
    }

    #[test]
    fn test_bitwise_operators() {
        let sweet = "{a & b | c}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        // & has higher precedence than |
        assert_eq!(sexpr, "(| (& a b) c)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{{a & b} | c}");
    }

    #[test]
    fn test_assignment_operators() {
        let sweet = "{x += y}";
        let sexpr = sweet_to_sexpr(sweet).unwrap();
        assert_eq!(sexpr, "(+= x y)");
        
        let back_to_sweet = sexpr_to_sweet_string(&sexpr).unwrap();
        assert_eq!(back_to_sweet, "{x += y}");
    }
}

#[cfg(test)]
mod indentation_tests {
    use super::*;

    // Note: Full indentation support would require more complex parsing
    // These tests demonstrate the basic structure for future implementation
    
    #[test]
    fn test_simple_indentation_parsing() {
        // For now, we'll test that the lexer can handle basic indentation
        // Full indentation-sensitive parsing would be implemented in a future iteration
        let sweet = "hello";
        let result = sweet_to_sexpr(sweet);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_malformed_infix_expressions() {
        let malformed_cases = vec![
            "{+ +}",           // Two operators in a row
            "{x +}",           // Missing right operand
            "{+ x}",           // Missing left operand for binary operator
            "{x y}",           // Missing operator
        ];

        for case in malformed_cases {
            let result = sweet_to_sexpr(case);
            assert!(result.is_err(), "Expected error for malformed case: {}", case);
        }
    }

    #[test]
    fn test_malformed_function_calls() {
        let malformed_cases = vec![
            "func(",          // Unclosed function call
            "func(,)",        // Empty argument with comma
            "func(x,,y)",     // Double comma
        ];

        for case in malformed_cases {
            let result = sweet_to_sexpr(case);
            assert!(result.is_err(), "Expected error for malformed case: {}", case);
        }
    }
}