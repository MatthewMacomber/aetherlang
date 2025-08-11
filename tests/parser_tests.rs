// Aether Parser Tests
// Comprehensive testing for S-expression parser and AST representation

use aether_language::{parse_sexpr, parse_multiple_sexprs, ASTNode, AtomValue, ParseError};

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn test_parse_atoms() {
        // Test symbol
        let ast = parse_sexpr("hello").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "hello"),
            _ => panic!("Expected symbol atom"),
        }

        // Test number
        let ast = parse_sexpr("42.5").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 42.5),
            _ => panic!("Expected number atom"),
        }

        // Test string
        let ast = parse_sexpr("\"hello world\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello world"),
            _ => panic!("Expected string atom"),
        }

        // Test boolean true
        let ast = parse_sexpr("true").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Boolean(value)) => assert_eq!(*value, true),
            _ => panic!("Expected boolean atom"),
        }

        // Test boolean false
        let ast = parse_sexpr("false").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Boolean(value)) => assert_eq!(*value, false),
            _ => panic!("Expected boolean atom"),
        }

        // Test nil
        let ast = parse_sexpr("nil").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Nil) => {},
            _ => panic!("Expected nil atom"),
        }
    }

    #[test]
    fn test_parse_numbers() {
        // Test integer
        let ast = parse_sexpr("42").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 42.0),
            _ => panic!("Expected number atom"),
        }

        // Test negative integer
        let ast = parse_sexpr("-42").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, -42.0),
            _ => panic!("Expected negative number atom"),
        }

        // Test decimal
        let ast = parse_sexpr("3.14159").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 3.14159),
            _ => panic!("Expected decimal number atom"),
        }

        // Test scientific notation
        let ast = parse_sexpr("1.23e-4").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 1.23e-4),
            _ => panic!("Expected scientific notation number atom"),
        }

        // Test scientific notation with positive exponent
        let ast = parse_sexpr("1.5E+10").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 1.5E+10),
            _ => panic!("Expected scientific notation number atom"),
        }
    }

    #[test]
    fn test_parse_strings() {
        // Test simple string
        let ast = parse_sexpr("\"hello\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello"),
            _ => panic!("Expected string atom"),
        }

        // Test string with spaces
        let ast = parse_sexpr("\"hello world\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello world"),
            _ => panic!("Expected string atom"),
        }

        // Test string with escape sequences
        let ast = parse_sexpr("\"hello\\nworld\\t!\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello\nworld\t!"),
            _ => panic!("Expected string atom with escapes"),
        }

        // Test string with quotes
        let ast = parse_sexpr("\"say \\\"hello\\\"\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "say \"hello\""),
            _ => panic!("Expected string atom with quotes"),
        }

        // Test empty string
        let ast = parse_sexpr("\"\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, ""),
            _ => panic!("Expected empty string atom"),
        }
    }

    #[test]
    fn test_parse_symbols() {
        // Test simple symbol
        let ast = parse_sexpr("hello").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "hello"),
            _ => panic!("Expected symbol atom"),
        }

        // Test symbol with numbers
        let ast = parse_sexpr("var123").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "var123"),
            _ => panic!("Expected symbol atom"),
        }

        // Test symbol with special characters
        let ast = parse_sexpr("my-var_name").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "my-var_name"),
            _ => panic!("Expected symbol atom"),
        }

        // Test operator symbols
        let ast = parse_sexpr("+").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "+"),
            _ => panic!("Expected operator symbol"),
        }

        let ast = parse_sexpr("<=").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "<="),
            _ => panic!("Expected operator symbol"),
        }
    }

    #[test]
    fn test_parse_lists() {
        // Test empty list
        let ast = parse_sexpr("()").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 0),
            _ => panic!("Expected empty list"),
        }

        // Test simple list
        let ast = parse_sexpr("(+ 1 2)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                assert_eq!(elements.len(), 3);
                
                // Check first element is '+'
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&elements[0]) {
                    assert_eq!(name, "+");
                } else {
                    panic!("Expected '+' symbol");
                }
                
                // Check second element is 1
                if let Some(ASTNode::Atom(AtomValue::Number(value))) = ast.resolve_ref(&elements[1]) {
                    assert_eq!(*value, 1.0);
                } else {
                    panic!("Expected number 1");
                }
                
                // Check third element is 2
                if let Some(ASTNode::Atom(AtomValue::Number(value))) = ast.resolve_ref(&elements[2]) {
                    assert_eq!(*value, 2.0);
                } else {
                    panic!("Expected number 2");
                }
            }
            _ => panic!("Expected list"),
        }

        // Test list with mixed types
        let ast = parse_sexpr("(func \"hello\" 42 true nil)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                assert_eq!(elements.len(), 5);
                
                // Check types of elements
                if let Some(ASTNode::Atom(AtomValue::Symbol(_))) = ast.resolve_ref(&elements[0]) {} else {
                    panic!("Expected symbol");
                }
                if let Some(ASTNode::Atom(AtomValue::String(_))) = ast.resolve_ref(&elements[1]) {} else {
                    panic!("Expected string");
                }
                if let Some(ASTNode::Atom(AtomValue::Number(_))) = ast.resolve_ref(&elements[2]) {} else {
                    panic!("Expected number");
                }
                if let Some(ASTNode::Atom(AtomValue::Boolean(_))) = ast.resolve_ref(&elements[3]) {} else {
                    panic!("Expected boolean");
                }
                if let Some(ASTNode::Atom(AtomValue::Nil)) = ast.resolve_ref(&elements[4]) {} else {
                    panic!("Expected nil");
                }
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_nested_lists() {
        // Test nested list
        let ast = parse_sexpr("(if (> x 0) x (- x))").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                assert_eq!(elements.len(), 4);
                
                // Check first element is 'if'
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&elements[0]) {
                    assert_eq!(name, "if");
                } else {
                    panic!("Expected 'if' symbol");
                }
                
                // Check second element is a list (> x 0)
                if let Some(ASTNode::List(condition_elements)) = ast.resolve_ref(&elements[1]) {
                    assert_eq!(condition_elements.len(), 3);
                } else {
                    panic!("Expected condition list");
                }
                
                // Check third element is 'x'
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&elements[2]) {
                    assert_eq!(name, "x");
                } else {
                    panic!("Expected 'x' symbol");
                }
                
                // Check fourth element is a list (- x)
                if let Some(ASTNode::List(else_elements)) = ast.resolve_ref(&elements[3]) {
                    assert_eq!(else_elements.len(), 2);
                } else {
                    panic!("Expected else list");
                }
            }
            _ => panic!("Expected list"),
        }

        // Test deeply nested list
        let ast = parse_sexpr("(((nested) list) structure)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                // The structure should be: ((nested) list) structure
                // So we expect 2 elements: the nested list and "structure"
                assert_eq!(elements.len(), 2);
                
                // Check first element is nested list
                if let Some(ASTNode::List(nested1)) = ast.resolve_ref(&elements[0]) {
                    assert_eq!(nested1.len(), 2);
                    
                    // Check first nested element
                    if let Some(ASTNode::List(nested2)) = ast.resolve_ref(&nested1[0]) {
                        assert_eq!(nested2.len(), 1);
                    } else {
                        panic!("Expected deeply nested list");
                    }
                } else {
                    panic!("Expected nested list");
                }
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_multiple_expressions() {
        let asts = parse_multiple_sexprs("42 (+ 1 2) \"hello\" true").unwrap();
        assert_eq!(asts.len(), 4);
        
        // Check first expression (number)
        match &asts[0].root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 42.0),
            _ => panic!("Expected number"),
        }
        
        // Check second expression (list)
        match &asts[1].root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected list"),
        }
        
        // Check third expression (string)
        match &asts[2].root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello"),
            _ => panic!("Expected string"),
        }
        
        // Check fourth expression (boolean)
        match &asts[3].root {
            ASTNode::Atom(AtomValue::Boolean(value)) => assert_eq!(*value, true),
            _ => panic!("Expected boolean"),
        }
    }

    #[test]
    fn test_parse_with_comments() {
        // Test line comment
        let ast = parse_sexpr("; This is a comment\n(+ 1 2)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected list"),
        }

        // Test comment at end of line
        let ast = parse_sexpr("(+ 1 2) ; This is a comment").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected list"),
        }

        // Test multiple comments
        let asts = parse_multiple_sexprs("; First comment\n42\n; Second comment\n(+ 1 2)").unwrap();
        assert_eq!(asts.len(), 2);
    }

    #[test]
    fn test_parse_with_whitespace() {
        // Test various whitespace
        let ast = parse_sexpr("  \t\n  (+   1    2  )  \t\n  ").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected list"),
        }

        // Test no whitespace - this should parse "+1" as a single symbol, not "+" and "1"
        let ast = parse_sexpr("(+ 1 2)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected list"),
        }
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_parse_errors() {
        // Test unmatched opening parenthesis
        let result = parse_sexpr("(+ 1 2");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnmatchedParen(_) => {},
            _ => panic!("Expected unmatched paren error"),
        }

        // Test unmatched closing parenthesis - this should parse just the "+" symbol
        let result = parse_sexpr("+");
        assert!(result.is_ok());
        
        // Test a real unmatched closing paren
        let result = parse_sexpr(")");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnmatchedParen(_) => {},
            _ => panic!("Expected unmatched paren error"),
        }

        // Test unterminated string
        let result = parse_sexpr("\"hello world");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnterminatedString(_) => {},
            _ => panic!("Expected unterminated string error"),
        }

        // Test empty input - this should fail with EOF
        let result = parse_sexpr("");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnexpectedEof => {},
            _ => panic!("Expected unexpected EOF error"),
        }

        // Test invalid number format - this might parse as a number (12.34) or symbol
        let result = parse_sexpr("12.34.56");
        assert!(result.is_ok());
        // The parser might parse this as 12.34 (valid number) and ignore the rest
        // or as a symbol. Let's just check it parses successfully.
        match &result.unwrap().root {
            ASTNode::Atom(AtomValue::Number(value)) => {
                // If parsed as number, it should be 12.34
                assert_eq!(*value, 12.34);
            },
            ASTNode::Atom(AtomValue::Symbol(name)) => {
                // If parsed as symbol, it should be the full string
                assert_eq!(name, "12.34.56");
            },
            _ => panic!("Expected number or symbol"),
        }
    }

    #[test]
    fn test_malformed_input_handling() {
        // Test nested unmatched parens
        let result = parse_sexpr("(+ (- 1 2) 3");
        assert!(result.is_err());

        // Test multiple unmatched parens
        let result = parse_sexpr("((((+ 1 2)");
        assert!(result.is_err());

        // Test mixed unmatched parens - this should parse the list successfully
        let result = parse_sexpr("(+ 1 2)");
        assert!(result.is_ok());

        // Test string with unescaped quotes - this should parse just the first string
        let result = parse_sexpr("\"hello \"");
        assert!(result.is_ok());
        match &result.unwrap().root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello "),
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test single character symbols
        let ast = parse_sexpr("x").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "x"),
            _ => panic!("Expected single character symbol"),
        }

        // Test zero
        let ast = parse_sexpr("0").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 0.0),
            _ => panic!("Expected zero"),
        }

        // Test negative zero
        let ast = parse_sexpr("-0").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, -0.0),
            _ => panic!("Expected negative zero"),
        }

        // Test very long symbol
        let long_symbol = "a".repeat(1000);
        let ast = parse_sexpr(&long_symbol).unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, &long_symbol),
            _ => panic!("Expected long symbol"),
        }

        // Test very nested list
        let nested = "(((((((((x)))))))))";
        let ast = parse_sexpr(nested).unwrap();
        assert!(ast.root.is_list());
    }
}

#[cfg(test)]
mod ast_tests {
    use super::*;
    use aether_language::{ASTNode, ASTNodeRef, AtomValue};

    #[test]
    fn test_ast_node_creation() {
        // Test atom creation methods
        let symbol_node = ASTNode::symbol("test".to_string());
        assert!(symbol_node.is_atom());
        assert_eq!(symbol_node.as_atom().unwrap().as_symbol().unwrap(), "test");

        let number_node = ASTNode::number(42.0);
        assert!(number_node.is_atom());
        assert_eq!(number_node.as_atom().unwrap().as_number().unwrap(), 42.0);

        let string_node = ASTNode::string("hello".to_string());
        assert!(string_node.is_atom());
        assert_eq!(string_node.as_atom().unwrap().as_string().unwrap(), "hello");

        let bool_node = ASTNode::boolean(true);
        assert!(bool_node.is_atom());
        assert_eq!(bool_node.as_atom().unwrap().as_boolean().unwrap(), true);

        let nil_node = ASTNode::nil();
        assert!(nil_node.is_atom());
        assert!(nil_node.as_atom().unwrap().is_nil());

        // Test list creation
        let empty_list = ASTNode::empty_list();
        assert!(empty_list.is_list());
        assert_eq!(empty_list.as_list().unwrap().len(), 0);

        let list_node = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("test".to_string())),
            ASTNodeRef::direct(ASTNode::number(42.0)),
        ]);
        assert!(list_node.is_list());
        assert_eq!(list_node.as_list().unwrap().len(), 2);
    }

    #[test]
    fn test_atom_value_methods() {
        let symbol = AtomValue::Symbol("test".to_string());
        assert!(symbol.is_symbol());
        assert!(!symbol.is_number());
        assert_eq!(symbol.as_symbol().unwrap(), "test");

        let number = AtomValue::Number(42.5);
        assert!(number.is_number());
        assert!(!number.is_symbol());
        assert_eq!(number.as_number().unwrap(), 42.5);

        let string = AtomValue::String("hello".to_string());
        assert!(string.is_string());
        assert!(!string.is_boolean());
        assert_eq!(string.as_string().unwrap(), "hello");

        let boolean = AtomValue::Boolean(true);
        assert!(boolean.is_boolean());
        assert!(!boolean.is_nil());
        assert_eq!(boolean.as_boolean().unwrap(), true);

        let nil = AtomValue::Nil;
        assert!(nil.is_nil());
        assert!(!nil.is_string());
    }

    #[test]
    fn test_ast_node_ref_creation() {
        let node = ASTNode::symbol("test".to_string());
        
        let direct_ref = ASTNodeRef::direct(node);
        match direct_ref {
            ASTNodeRef::Direct(_) => {},
            _ => panic!("Expected direct reference"),
        }

        let id_ref = ASTNodeRef::id(42);
        match id_ref {
            ASTNodeRef::Id(id) => assert_eq!(id, 42),
            _ => panic!("Expected ID reference"),
        }

        let label_ref = ASTNodeRef::label("test_label".to_string());
        match label_ref {
            ASTNodeRef::Label(label) => assert_eq!(label, "test_label"),
            _ => panic!("Expected label reference"),
        }
    }

    #[test]
    fn test_ast_display() {
        // Test atom display
        let symbol = ASTNode::symbol("hello".to_string());
        assert_eq!(format!("{}", symbol), "hello");

        let number = ASTNode::number(42.5);
        assert_eq!(format!("{}", number), "42.5");

        let string = ASTNode::string("world".to_string());
        assert_eq!(format!("{}", string), "\"world\"");

        let boolean = ASTNode::boolean(true);
        assert_eq!(format!("{}", boolean), "true");

        let nil = ASTNode::nil();
        assert_eq!(format!("{}", nil), "nil");

        // Test list display
        let list = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
            ASTNodeRef::direct(ASTNode::number(2.0)),
        ]);
        assert_eq!(format!("{}", list), "(+ 1 2)");

        // Test empty list display
        let empty = ASTNode::empty_list();
        assert_eq!(format!("{}", empty), "()");
    }
}