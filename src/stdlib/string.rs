// Aether Standard Library - String Processing
// String manipulation and regular expression support with linear ownership

use crate::compiler::types::LinearOwnership;
use crate::stdlib::collections::LinearVec;

/// Linear string with ownership tracking
#[derive(Debug, Clone)]
pub struct LinearString {
    data: String,
    ownership: LinearOwnership,
}

impl LinearString {
    /// Create a new linear string
    pub fn new() -> Self {
        Self {
            data: String::new(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Create from string slice
    pub fn from_str(s: &str) -> Self {
        Self {
            data: s.to_string(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Create from owned string
    pub fn from_string(s: String) -> Self {
        Self {
            data: s,
            ownership: LinearOwnership::Owned,
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get as string slice
    pub fn as_str(&self) -> &str {
        &self.data
    }

    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.data
    }

    /// Append another string, consuming both
    pub fn append(mut self, other: LinearString) -> Self {
        self.data.push_str(&other.data);
        self
    }

    /// Append string slice
    pub fn append_str(mut self, s: &str) -> Self {
        self.data.push_str(s);
        self
    }

    /// Prepend string slice
    pub fn prepend_str(self, s: &str) -> Self {
        Self {
            data: format!("{}{}", s, self.data),
            ownership: self.ownership,
        }
    }

    /// Convert to uppercase
    pub fn to_uppercase(self) -> Self {
        Self {
            data: self.data.to_uppercase(),
            ownership: self.ownership,
        }
    }

    /// Convert to lowercase
    pub fn to_lowercase(self) -> Self {
        Self {
            data: self.data.to_lowercase(),
            ownership: self.ownership,
        }
    }

    /// Trim whitespace
    pub fn trim(self) -> Self {
        Self {
            data: self.data.trim().to_string(),
            ownership: self.ownership,
        }
    }

    /// Trim start whitespace
    pub fn trim_start(self) -> Self {
        Self {
            data: self.data.trim_start().to_string(),
            ownership: self.ownership,
        }
    }

    /// Trim end whitespace
    pub fn trim_end(self) -> Self {
        Self {
            data: self.data.trim_end().to_string(),
            ownership: self.ownership,
        }
    }

    /// Split by delimiter
    pub fn split(self, delimiter: &str) -> LinearVec<LinearString> {
        let parts: Vec<LinearString> = self.data
            .split(delimiter)
            .map(|s| LinearString::from_str(s))
            .collect();
        
        parts.into_iter().fold(LinearVec::new(), |vec, part| vec.push(part))
    }

    /// Split by whitespace
    pub fn split_whitespace(self) -> LinearVec<LinearString> {
        let parts: Vec<LinearString> = self.data
            .split_whitespace()
            .map(|s| LinearString::from_str(s))
            .collect();
        
        parts.into_iter().fold(LinearVec::new(), |vec, part| vec.push(part))
    }

    /// Split into lines
    pub fn lines(self) -> LinearVec<LinearString> {
        let lines: Vec<LinearString> = self.data
            .lines()
            .map(|s| LinearString::from_str(s))
            .collect();
        
        lines.into_iter().fold(LinearVec::new(), |vec, line| vec.push(line))
    }

    /// Replace all occurrences
    pub fn replace(self, from: &str, to: &str) -> Self {
        Self {
            data: self.data.replace(from, to),
            ownership: self.ownership,
        }
    }

    /// Replace first occurrence
    pub fn replace_first(self, from: &str, to: &str) -> Self {
        if let Some(pos) = self.data.find(from) {
            let mut result = String::with_capacity(self.data.len());
            result.push_str(&self.data[..pos]);
            result.push_str(to);
            result.push_str(&self.data[pos + from.len()..]);
            
            Self {
                data: result,
                ownership: self.ownership,
            }
        } else {
            self
        }
    }

    /// Check if contains substring
    pub fn contains(&self, needle: &str) -> bool {
        self.data.contains(needle)
    }

    /// Check if starts with prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.data.starts_with(prefix)
    }

    /// Check if ends with suffix
    pub fn ends_with(&self, suffix: &str) -> bool {
        self.data.ends_with(suffix)
    }

    /// Find first occurrence of substring
    pub fn find(&self, needle: &str) -> Option<usize> {
        self.data.find(needle)
    }

    /// Find last occurrence of substring
    pub fn rfind(&self, needle: &str) -> Option<usize> {
        self.data.rfind(needle)
    }

    /// Get substring
    pub fn substring(&self, start: usize, end: usize) -> Option<LinearString> {
        if start <= end && end <= self.data.len() {
            Some(LinearString::from_str(&self.data[start..end]))
        } else {
            None
        }
    }

    /// Get character at index
    pub fn char_at(&self, index: usize) -> Option<char> {
        self.data.chars().nth(index)
    }

    /// Get character count (not byte count)
    pub fn char_count(&self) -> usize {
        self.data.chars().count()
    }

    /// Reverse the string
    pub fn reverse(self) -> Self {
        Self {
            data: self.data.chars().rev().collect(),
            ownership: self.ownership,
        }
    }

    /// Repeat the string n times
    pub fn repeat(self, n: usize) -> Self {
        Self {
            data: self.data.repeat(n),
            ownership: self.ownership,
        }
    }

    /// Pad left with character to reach target length
    pub fn pad_left(self, target_len: usize, pad_char: char) -> Self {
        if self.data.len() >= target_len {
            return self;
        }
        
        let pad_count = target_len - self.data.len();
        let padding: String = std::iter::repeat(pad_char).take(pad_count).collect();
        
        Self {
            data: format!("{}{}", padding, self.data),
            ownership: self.ownership,
        }
    }

    /// Pad right with character to reach target length
    pub fn pad_right(self, target_len: usize, pad_char: char) -> Self {
        if self.data.len() >= target_len {
            return self;
        }
        
        let pad_count = target_len - self.data.len();
        let padding: String = std::iter::repeat(pad_char).take(pad_count).collect();
        
        Self {
            data: format!("{}{}", self.data, padding),
            ownership: self.ownership,
        }
    }
}

impl Default for LinearString {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for LinearString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl PartialEq for LinearString {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl PartialEq<str> for LinearString {
    fn eq(&self, other: &str) -> bool {
        self.data == other
    }
}

impl PartialEq<String> for LinearString {
    fn eq(&self, other: &String) -> bool {
        &self.data == other
    }
}

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Join strings with delimiter
    pub fn join(strings: LinearVec<LinearString>, delimiter: &str) -> LinearString {
        if strings.is_empty() {
            return LinearString::new();
        }
        
        let mut result = String::new();
        for i in 0..strings.len() {
            if let Some(s) = strings.get(i) {
                if i > 0 {
                    result.push_str(delimiter);
                }
                result.push_str(s.as_str());
            }
        }
        
        LinearString::from_string(result)
    }

    /// Check if string is numeric
    pub fn is_numeric(s: &LinearString) -> bool {
        s.as_str().chars().all(|c| c.is_ascii_digit())
    }

    /// Check if string is alphabetic
    pub fn is_alphabetic(s: &LinearString) -> bool {
        s.as_str().chars().all(|c| c.is_alphabetic())
    }

    /// Check if string is alphanumeric
    pub fn is_alphanumeric(s: &LinearString) -> bool {
        s.as_str().chars().all(|c| c.is_alphanumeric())
    }

    /// Convert string to integer
    pub fn to_int(s: &LinearString) -> Result<i64, String> {
        s.as_str().parse::<i64>().map_err(|e| e.to_string())
    }

    /// Convert string to float
    pub fn to_float(s: &LinearString) -> Result<f64, String> {
        s.as_str().parse::<f64>().map_err(|e| e.to_string())
    }

    /// Convert integer to string
    pub fn from_int(value: i64) -> LinearString {
        LinearString::from_string(value.to_string())
    }

    /// Convert float to string
    pub fn from_float(value: f64) -> LinearString {
        LinearString::from_string(value.to_string())
    }

    /// Calculate Levenshtein distance between two strings
    pub fn levenshtein_distance(s1: &LinearString, s2: &LinearString) -> usize {
        let s1_chars: Vec<char> = s1.as_str().chars().collect();
        let s2_chars: Vec<char> = s2.as_str().chars().collect();
        
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();
        
        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i-1] == s2_chars[j-1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i-1][j] + 1,     // deletion
                        matrix[i][j-1] + 1      // insertion
                    ),
                    matrix[i-1][j-1] + cost     // substitution
                );
            }
        }
        
        matrix[len1][len2]
    }
}

/// Regular expression support
pub mod regex {
    use super::*;

    /// Regular expression error types
    #[derive(Debug, Clone)]
    pub enum RegexError {
        InvalidPattern(String),
        CompilationError(String),
        MatchError(String),
    }

    impl std::fmt::Display for RegexError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RegexError::InvalidPattern(msg) => write!(f, "Invalid regex pattern: {}", msg),
                RegexError::CompilationError(msg) => write!(f, "Regex compilation error: {}", msg),
                RegexError::MatchError(msg) => write!(f, "Regex match error: {}", msg),
            }
        }
    }

    impl std::error::Error for RegexError {}

    /// Match result
    #[derive(Debug, Clone)]
    pub struct Match {
        pub text: LinearString,
        pub start: usize,
        pub end: usize,
        pub groups: LinearVec<LinearString>,
    }

    /// Simple regex implementation (basic patterns only)
    pub struct Regex {
        pattern: String,
        compiled: CompiledPattern,
    }

    #[derive(Debug, Clone)]
    enum CompiledPattern {
        Literal(String),
        Wildcard,
        CharClass(Vec<char>),
        Sequence(Vec<CompiledPattern>),
        Optional(Box<CompiledPattern>),
        Repeat(Box<CompiledPattern>),
        Group(Box<CompiledPattern>),
    }

    impl Regex {
        /// Compile a regex pattern
        pub fn new(pattern: &str) -> Result<Self, RegexError> {
            let compiled = Self::compile_pattern(pattern)?;
            Ok(Self {
                pattern: pattern.to_string(),
                compiled,
            })
        }

        /// Find first match in text
        pub fn find(&self, text: &LinearString) -> Option<Match> {
            self.find_at(text, 0)
        }

        /// Find first match starting at position
        pub fn find_at(&self, text: &LinearString, start: usize) -> Option<Match> {
            let text_str = text.as_str();
            let chars: Vec<char> = text_str.chars().collect();
            
            for i in start..chars.len() {
                if let Some((end, groups)) = self.match_at(&chars, i) {
                    let matched_text = chars[i..end].iter().collect::<String>();
                    return Some(Match {
                        text: LinearString::from_string(matched_text),
                        start: i,
                        end,
                        groups,
                    });
                }
            }
            
            None
        }

        /// Find all matches in text
        pub fn find_all(&self, text: &LinearString) -> LinearVec<Match> {
            let mut matches = LinearVec::new();
            let mut start = 0;
            
            while let Some(m) = self.find_at(text, start) {
                start = m.end;
                matches = matches.push(m);
            }
            
            matches
        }

        /// Check if pattern matches entire text
        pub fn is_match(&self, text: &LinearString) -> bool {
            let chars: Vec<char> = text.as_str().chars().collect();
            if let Some((end, _)) = self.match_at(&chars, 0) {
                end == chars.len()
            } else {
                false
            }
        }

        /// Replace first match
        pub fn replace_first(&self, text: LinearString, replacement: &str) -> LinearString {
            if let Some(m) = self.find(&text) {
                let mut result = String::new();
                result.push_str(&text.as_str()[..m.start]);
                result.push_str(replacement);
                result.push_str(&text.as_str()[m.end..]);
                LinearString::from_string(result)
            } else {
                text
            }
        }

        /// Replace all matches
        pub fn replace_all(&self, text: LinearString, replacement: &str) -> LinearString {
            let matches = self.find_all(&text);
            if matches.is_empty() {
                return text;
            }
            
            let mut result = String::new();
            let mut last_end = 0;
            
            for i in 0..matches.len() {
                if let Some(m) = matches.get(i) {
                    result.push_str(&text.as_str()[last_end..m.start]);
                    result.push_str(replacement);
                    last_end = m.end;
                }
            }
            
            result.push_str(&text.as_str()[last_end..]);
            LinearString::from_string(result)
        }

        /// Split text by regex matches
        pub fn split(&self, text: LinearString) -> LinearVec<LinearString> {
            let matches = self.find_all(&text);
            if matches.is_empty() {
                return LinearVec::new().push(text);
            }
            
            let mut parts = LinearVec::new();
            let mut last_end = 0;
            
            for i in 0..matches.len() {
                if let Some(m) = matches.get(i) {
                    if m.start > last_end {
                        let part = &text.as_str()[last_end..m.start];
                        parts = parts.push(LinearString::from_str(part));
                    }
                    last_end = m.end;
                }
            }
            
            if last_end < text.len() {
                let part = &text.as_str()[last_end..];
                parts = parts.push(LinearString::from_str(part));
            }
            
            parts
        }

        // Simple pattern compilation (supports basic patterns only)
        fn compile_pattern(pattern: &str) -> Result<CompiledPattern, RegexError> {
            if pattern.is_empty() {
                return Ok(CompiledPattern::Literal(String::new()));
            }
            
            // Very basic pattern matching - in practice would need full regex parser
            if pattern == "." {
                Ok(CompiledPattern::Wildcard)
            } else if pattern.starts_with('[') && pattern.ends_with(']') {
                let chars: Vec<char> = pattern[1..pattern.len()-1].chars().collect();
                Ok(CompiledPattern::CharClass(chars))
            } else if pattern.ends_with('*') {
                let inner = Self::compile_pattern(&pattern[..pattern.len()-1])?;
                Ok(CompiledPattern::Repeat(Box::new(inner)))
            } else if pattern.ends_with('?') {
                let inner = Self::compile_pattern(&pattern[..pattern.len()-1])?;
                Ok(CompiledPattern::Optional(Box::new(inner)))
            } else {
                Ok(CompiledPattern::Literal(pattern.to_string()))
            }
        }

        // Match pattern at specific position
        fn match_at(&self, chars: &[char], pos: usize) -> Option<(usize, LinearVec<LinearString>)> {
            self.match_pattern(&self.compiled, chars, pos)
        }

        fn match_pattern(&self, pattern: &CompiledPattern, chars: &[char], pos: usize) -> Option<(usize, LinearVec<LinearString>)> {
            match pattern {
                CompiledPattern::Literal(s) => {
                    let pattern_chars: Vec<char> = s.chars().collect();
                    if pos + pattern_chars.len() <= chars.len() {
                        if chars[pos..pos + pattern_chars.len()] == pattern_chars {
                            Some((pos + pattern_chars.len(), LinearVec::new()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                CompiledPattern::Wildcard => {
                    if pos < chars.len() {
                        Some((pos + 1, LinearVec::new()))
                    } else {
                        None
                    }
                },
                CompiledPattern::CharClass(allowed) => {
                    if pos < chars.len() && allowed.contains(&chars[pos]) {
                        Some((pos + 1, LinearVec::new()))
                    } else {
                        None
                    }
                },
                CompiledPattern::Optional(inner) => {
                    // Try to match, but succeed even if it doesn't match
                    if let Some(result) = self.match_pattern(inner, chars, pos) {
                        Some(result)
                    } else {
                        Some((pos, LinearVec::new()))
                    }
                },
                CompiledPattern::Repeat(inner) => {
                    let mut current_pos = pos;
                    let mut groups = LinearVec::new();
                    
                    // Match zero or more times
                    while let Some((new_pos, mut inner_groups)) = self.match_pattern(inner, chars, current_pos) {
                        current_pos = new_pos;
                        // Merge groups
                        for i in 0..inner_groups.len() {
                            if let Some(group) = inner_groups.get(i) {
                                groups = groups.push(group.clone());
                            }
                        }
                    }
                    
                    Some((current_pos, groups))
                },
                _ => None, // Other patterns not implemented in this basic version
            }
        }
    }

    /// Common regex patterns
    pub struct CommonPatterns;

    impl CommonPatterns {
        /// Email pattern (simplified)
        pub fn email() -> Result<Regex, RegexError> {
            Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        }

        /// URL pattern (simplified)
        pub fn url() -> Result<Regex, RegexError> {
            Regex::new(r"https?://[^\s]+")
        }

        /// Phone number pattern (US format)
        pub fn phone_us() -> Result<Regex, RegexError> {
            Regex::new(r"\d{3}-\d{3}-\d{4}")
        }

        /// Integer pattern
        pub fn integer() -> Result<Regex, RegexError> {
            Regex::new(r"-?\d+")
        }

        /// Float pattern
        pub fn float() -> Result<Regex, RegexError> {
            Regex::new(r"-?\d+\.\d+")
        }

        /// Whitespace pattern
        pub fn whitespace() -> Result<Regex, RegexError> {
            Regex::new(r"\s+")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_string_basic_operations() {
        let s1 = LinearString::from_str("Hello");
        let s2 = LinearString::from_str(" World");
        
        let combined = s1.append(s2);
        assert_eq!(combined.as_str(), "Hello World");
        assert_eq!(combined.len(), 11);
        
        let upper = combined.to_uppercase();
        assert_eq!(upper.as_str(), "HELLO WORLD");
        
        let trimmed = LinearString::from_str("  test  ").trim();
        assert_eq!(trimmed.as_str(), "test");
    }

    #[test]
    fn test_string_splitting() {
        let text = LinearString::from_str("apple,banana,cherry");
        let parts = text.split(",");
        
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.get(0).unwrap().as_str(), "apple");
        assert_eq!(parts.get(1).unwrap().as_str(), "banana");
        assert_eq!(parts.get(2).unwrap().as_str(), "cherry");
    }

    #[test]
    fn test_string_replacement() {
        let text = LinearString::from_str("Hello World Hello");
        let replaced = text.replace("Hello", "Hi");
        assert_eq!(replaced.as_str(), "Hi World Hi");
        
        let text = LinearString::from_str("Hello World Hello");
        let replaced_first = text.replace_first("Hello", "Hi");
        assert_eq!(replaced_first.as_str(), "Hi World Hello");
    }

    #[test]
    fn test_string_utilities() {
        let strings = LinearVec::new()
            .push(LinearString::from_str("apple"))
            .push(LinearString::from_str("banana"))
            .push(LinearString::from_str("cherry"));
        
        let joined = StringUtils::join(strings, ", ");
        assert_eq!(joined.as_str(), "apple, banana, cherry");
        
        let numeric = LinearString::from_str("12345");
        assert!(StringUtils::is_numeric(&numeric));
        
        let alpha = LinearString::from_str("hello");
        assert!(StringUtils::is_alphabetic(&alpha));
        
        let int_result = StringUtils::to_int(&LinearString::from_str("42"));
        assert_eq!(int_result.unwrap(), 42);
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
    }

    #[test]
    fn test_basic_regex() {
        use super::regex::*;
        
        // Test literal matching
        let regex = Regex::new("hello").unwrap();
        let text = LinearString::from_str("hello world");
        assert!(regex.is_match(&LinearString::from_str("hello")));
        
        let m = regex.find(&text).unwrap();
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 5);
        assert_eq!(m.text.as_str(), "hello");
    }

    #[test]
    fn test_regex_replacement() {
        use super::regex::*;
        
        let regex = Regex::new("world").unwrap();
        let text = LinearString::from_str("hello world");
        let replaced = regex.replace_first(text, "universe");
        assert_eq!(replaced.as_str(), "hello universe");
    }

    #[test]
    fn test_string_padding() {
        let s = LinearString::from_str("test");
        let padded_left = s.pad_left(8, '0');
        assert_eq!(padded_left.as_str(), "0000test");
        
        let s = LinearString::from_str("test");
        let padded_right = s.pad_right(8, '-');
        assert_eq!(padded_right.as_str(), "test----");
    }

    #[test]
    fn test_string_reverse() {
        let s = LinearString::from_str("hello");
        let reversed = s.reverse();
        assert_eq!(reversed.as_str(), "olleh");
    }

    #[test]
    fn test_string_repeat() {
        let s = LinearString::from_str("abc");
        let repeated = s.repeat(3);
        assert_eq!(repeated.as_str(), "abcabcabc");
    }
}