// Aether Standard Library - I/O Operations
// Async I/O with error handling and resource management

use crate::compiler::types::LinearOwnership;
use crate::stdlib::collections::LinearVec;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{self, Read, Write, BufRead, BufReader, BufWriter};

/// I/O error types with detailed context
#[derive(Debug, Clone)]
pub enum IOError {
    FileNotFound(String),
    PermissionDenied(String),
    InvalidPath(String),
    ReadError(String),
    WriteError(String),
    NetworkError(String),
    SerializationError(String),
    AsyncError(String),
}

impl std::fmt::Display for IOError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IOError::FileNotFound(path) => write!(f, "File not found: {}", path),
            IOError::PermissionDenied(path) => write!(f, "Permission denied: {}", path),
            IOError::InvalidPath(path) => write!(f, "Invalid path: {}", path),
            IOError::ReadError(msg) => write!(f, "Read error: {}", msg),
            IOError::WriteError(msg) => write!(f, "Write error: {}", msg),
            IOError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            IOError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            IOError::AsyncError(msg) => write!(f, "Async error: {}", msg),
        }
    }
}

impl std::error::Error for IOError {}

impl From<std::io::Error> for IOError {
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            io::ErrorKind::NotFound => IOError::FileNotFound(error.to_string()),
            io::ErrorKind::PermissionDenied => IOError::PermissionDenied(error.to_string()),
            io::ErrorKind::InvalidInput => IOError::InvalidPath(error.to_string()),
            _ => IOError::ReadError(error.to_string()),
        }
    }
}

/// Result type for I/O operations
pub type IOResult<T> = Result<T, IOError>;

/// Linear file handle with automatic resource cleanup
#[derive(Debug)]
pub struct LinearFile {
    path: PathBuf,
    ownership: LinearOwnership,
    metadata: Option<fs::Metadata>,
}

impl LinearFile {
    /// Open a file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = fs::metadata(&path).map_err(IOError::from)?;
        
        Ok(Self {
            path,
            ownership: LinearOwnership::Owned,
            metadata: Some(metadata),
        })
    }

    /// Create a new file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(IOError::from)?;
        }
        
        Ok(Self {
            path,
            ownership: LinearOwnership::Owned,
            metadata: None,
        })
    }

    /// Read entire file contents as string
    pub fn read_to_string(self) -> IOResult<(Self, String)> {
        let contents = fs::read_to_string(&self.path).map_err(IOError::from)?;
        Ok((self, contents))
    }

    /// Read entire file contents as bytes
    pub fn read_to_bytes(self) -> IOResult<(Self, LinearVec<u8>)> {
        let bytes = fs::read(&self.path).map_err(IOError::from)?;
        let linear_bytes = bytes.into_iter().fold(LinearVec::new(), |vec, byte| vec.push(byte));
        Ok((self, linear_bytes))
    }

    /// Write string contents to file
    pub fn write_string(self, contents: &str) -> IOResult<Self> {
        fs::write(&self.path, contents).map_err(IOError::from)?;
        Ok(self)
    }

    /// Write bytes to file
    pub fn write_bytes(self, bytes: &LinearVec<u8>) -> IOResult<Self> {
        let byte_vec = bytes.clone().into_vec();
        fs::write(&self.path, byte_vec).map_err(IOError::from)?;
        Ok(self)
    }

    /// Append string to file
    pub fn append_string(self, contents: &str) -> IOResult<Self> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(IOError::from)?;
        
        file.write_all(contents.as_bytes()).map_err(IOError::from)?;
        Ok(self)
    }

    /// Get file metadata
    pub fn metadata(&self) -> Option<&fs::Metadata> {
        self.metadata.as_ref()
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file size
    pub fn size(&self) -> IOResult<u64> {
        if let Some(metadata) = &self.metadata {
            Ok(metadata.len())
        } else {
            let metadata = fs::metadata(&self.path).map_err(IOError::from)?;
            Ok(metadata.len())
        }
    }

    /// Check if file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Delete the file
    pub fn delete(self) -> IOResult<()> {
        fs::remove_file(&self.path).map_err(IOError::from)?;
        Ok(())
    }
}

/// Buffered file reader with linear ownership
pub struct LinearReader {
    reader: BufReader<fs::File>,
    ownership: LinearOwnership,
}

impl LinearReader {
    /// Create a new buffered reader
    pub fn new<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let file = fs::File::open(path).map_err(IOError::from)?;
        let reader = BufReader::new(file);
        
        Ok(Self {
            reader,
            ownership: LinearOwnership::Owned,
        })
    }

    /// Read a line from the file
    pub fn read_line(mut self) -> IOResult<(Self, Option<String>)> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => Ok((self, None)), // EOF
            Ok(_) => {
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                Ok((self, Some(line)))
            },
            Err(e) => Err(IOError::from(e)),
        }
    }

    /// Read all lines from the file
    pub fn read_lines(self) -> IOResult<LinearVec<String>> {
        let mut lines = LinearVec::new();
        let mut current_reader = self;
        
        loop {
            let (reader, line) = current_reader.read_line()?;
            current_reader = reader;
            
            match line {
                Some(line) => lines = lines.push(line),
                None => break,
            }
        }
        
        Ok(lines)
    }

    /// Read exact number of bytes
    pub fn read_exact(mut self, count: usize) -> IOResult<(Self, LinearVec<u8>)> {
        let mut buffer = vec![0u8; count];
        self.reader.read_exact(&mut buffer).map_err(IOError::from)?;
        
        let linear_buffer = buffer.into_iter().fold(LinearVec::new(), |vec, byte| vec.push(byte));
        Ok((self, linear_buffer))
    }
}

/// Buffered file writer with linear ownership
pub struct LinearWriter {
    writer: BufWriter<fs::File>,
    ownership: LinearOwnership,
}

impl LinearWriter {
    /// Create a new buffered writer
    pub fn new<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let file = fs::File::create(path).map_err(IOError::from)?;
        let writer = BufWriter::new(file);
        
        Ok(Self {
            writer,
            ownership: LinearOwnership::Owned,
        })
    }

    /// Write a string line
    pub fn write_line(mut self, line: &str) -> IOResult<Self> {
        writeln!(self.writer, "{}", line).map_err(|e| IOError::WriteError(e.to_string()))?;
        Ok(self)
    }

    /// Write bytes
    pub fn write_bytes(mut self, bytes: &LinearVec<u8>) -> IOResult<Self> {
        let byte_vec = bytes.clone().into_vec();
        self.writer.write_all(&byte_vec).map_err(|e| IOError::WriteError(e.to_string()))?;
        Ok(self)
    }

    /// Flush the writer
    pub fn flush(mut self) -> IOResult<Self> {
        self.writer.flush().map_err(|e| IOError::WriteError(e.to_string()))?;
        Ok(self)
    }

    /// Close the writer (consumes it)
    pub fn close(mut self) -> IOResult<()> {
        self.writer.flush().map_err(|e| IOError::WriteError(e.to_string()))?;
        Ok(())
    }
}

/// Directory operations with linear ownership
pub struct LinearDirectory {
    path: PathBuf,
    ownership: LinearOwnership,
}

impl LinearDirectory {
    /// Open a directory
    pub fn open<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let path = path.as_ref().to_path_buf();
        
        if !path.exists() {
            return Err(IOError::FileNotFound(path.to_string_lossy().to_string()));
        }
        
        if !path.is_dir() {
            return Err(IOError::InvalidPath("Path is not a directory".to_string()));
        }
        
        Ok(Self {
            path,
            ownership: LinearOwnership::Owned,
        })
    }

    /// Create a directory
    pub fn create<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path).map_err(IOError::from)?;
        
        Ok(Self {
            path,
            ownership: LinearOwnership::Owned,
        })
    }

    /// List directory contents
    pub fn list_contents(self) -> IOResult<(Self, LinearVec<PathBuf>)> {
        let entries = fs::read_dir(&self.path).map_err(IOError::from)?;
        let mut contents = LinearVec::new();
        
        for entry in entries {
            let entry = entry.map_err(IOError::from)?;
            contents = contents.push(entry.path());
        }
        
        Ok((self, contents))
    }

    /// List files only
    pub fn list_files(self) -> IOResult<(Self, LinearVec<PathBuf>)> {
        let (dir, contents) = self.list_contents()?;
        let files = contents.filter(|path| path.is_file());
        Ok((dir, files))
    }

    /// List subdirectories only
    pub fn list_dirs(self) -> IOResult<(Self, LinearVec<PathBuf>)> {
        let (dir, contents) = self.list_contents()?;
        let dirs = contents.filter(|path| path.is_dir());
        Ok((dir, dirs))
    }

    /// Get directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Delete directory (must be empty)
    pub fn delete(self) -> IOResult<()> {
        fs::remove_dir(&self.path).map_err(IOError::from)?;
        Ok(())
    }

    /// Delete directory recursively
    pub fn delete_recursive(self) -> IOResult<()> {
        fs::remove_dir_all(&self.path).map_err(IOError::from)?;
        Ok(())
    }
}

/// Async I/O operations
pub mod async_io {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Async file operations
    pub struct AsyncFile {
        path: PathBuf,
        ownership: LinearOwnership,
    }

    impl AsyncFile {
        /// Create async file handle
        pub fn new<P: AsRef<Path>>(path: P) -> Self {
            Self {
                path: path.as_ref().to_path_buf(),
                ownership: LinearOwnership::Owned,
            }
        }

        /// Async read file to string
        pub fn read_to_string(self) -> AsyncReadString {
            AsyncReadString {
                file: self,
                state: AsyncReadState::Pending,
            }
        }

        /// Async write string to file
        pub fn write_string(self, content: String) -> AsyncWriteString {
            AsyncWriteString {
                file: self,
                content,
                state: AsyncWriteState::Pending,
            }
        }
    }

    /// Async read operation state
    enum AsyncReadState {
        Pending,
        Reading,
        Complete(String),
        Error(IOError),
    }

    /// Future for async string reading
    pub struct AsyncReadString {
        file: AsyncFile,
        state: AsyncReadState,
    }

    impl Future for AsyncReadString {
        type Output = IOResult<(AsyncFile, String)>;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            match &self.state {
                AsyncReadState::Pending => {
                    self.state = AsyncReadState::Reading;
                    Poll::Pending
                },
                AsyncReadState::Reading => {
                    // In a real implementation, this would use async I/O
                    match fs::read_to_string(&self.file.path) {
                        Ok(content) => {
                            self.state = AsyncReadState::Complete(content.clone());
                            Poll::Ready(Ok((
                                AsyncFile {
                                    path: self.file.path.clone(),
                                    ownership: LinearOwnership::Owned,
                                },
                                content
                            )))
                        },
                        Err(e) => {
                            let error = IOError::from(e);
                            self.state = AsyncReadState::Error(error.clone());
                            Poll::Ready(Err(error))
                        }
                    }
                },
                AsyncReadState::Complete(content) => {
                    Poll::Ready(Ok((
                        AsyncFile {
                            path: self.file.path.clone(),
                            ownership: LinearOwnership::Owned,
                        },
                        content.clone()
                    )))
                },
                AsyncReadState::Error(error) => Poll::Ready(Err(error.clone())),
            }
        }
    }

    /// Async write operation state
    enum AsyncWriteState {
        Pending,
        Writing,
        Complete,
        Error(IOError),
    }

    /// Future for async string writing
    pub struct AsyncWriteString {
        file: AsyncFile,
        content: String,
        state: AsyncWriteState,
    }

    impl Future for AsyncWriteString {
        type Output = IOResult<AsyncFile>;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            match &self.state {
                AsyncWriteState::Pending => {
                    self.state = AsyncWriteState::Writing;
                    Poll::Pending
                },
                AsyncWriteState::Writing => {
                    // In a real implementation, this would use async I/O
                    match fs::write(&self.file.path, &self.content) {
                        Ok(_) => {
                            self.state = AsyncWriteState::Complete;
                            Poll::Ready(Ok(AsyncFile {
                                path: self.file.path.clone(),
                                ownership: LinearOwnership::Owned,
                            }))
                        },
                        Err(e) => {
                            let error = IOError::from(e);
                            self.state = AsyncWriteState::Error(error.clone());
                            Poll::Ready(Err(error))
                        }
                    }
                },
                AsyncWriteState::Complete => {
                    Poll::Ready(Ok(AsyncFile {
                        path: self.file.path.clone(),
                        ownership: LinearOwnership::Owned,
                    }))
                },
                AsyncWriteState::Error(error) => Poll::Ready(Err(error.clone())),
            }
        }
    }
}

/// Serialization utilities
pub mod serialization {
    use super::*;
    use crate::stdlib::collections::{LinearVec, LinearMap};

    /// JSON serialization support
    pub struct JsonSerializer;

    impl JsonSerializer {
        /// Serialize a map to JSON string
        pub fn serialize_map<K, V>(map: &LinearMap<K, V>) -> IOResult<String>
        where
            K: std::fmt::Display + Eq + std::hash::Hash,
            V: std::fmt::Display,
        {
            let mut json = String::from("{");
            let keys = map.keys();
            let values = map.values();
            
            for i in 0..keys.len() {
                if i > 0 {
                    json.push_str(", ");
                }
                if let (Some(key), Some(value)) = (keys.get(i), values.get(i)) {
                    json.push_str(&format!("\"{}\": \"{}\"", key, value));
                }
            }
            
            json.push('}');
            Ok(json)
        }

        /// Serialize a vector to JSON array string
        pub fn serialize_vec<T>(vec: &LinearVec<T>) -> IOResult<String>
        where
            T: std::fmt::Display,
        {
            let mut json = String::from("[");
            
            for i in 0..vec.len() {
                if i > 0 {
                    json.push_str(", ");
                }
                if let Some(item) = vec.get(i) {
                    json.push_str(&format!("\"{}\"", item));
                }
            }
            
            json.push(']');
            Ok(json)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_linear_file_operations() {
        let test_path = "test_file.txt";
        let test_content = "Hello, Aether!";
        
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
    fn test_buffered_io() {
        let test_path = "test_buffered.txt";
        let lines = vec!["Line 1", "Line 2", "Line 3"];
        
        // Write lines
        let mut writer = LinearWriter::new(test_path).unwrap();
        for line in &lines {
            writer = writer.write_line(line).unwrap();
        }
        writer.close().unwrap();
        
        // Read lines
        let reader = LinearReader::new(test_path).unwrap();
        let read_lines = reader.read_lines().unwrap();
        
        assert_eq!(read_lines.len(), 3);
        for i in 0..3 {
            assert_eq!(read_lines.get(i), Some(&lines[i].to_string()));
        }
        
        // Clean up
        fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_directory_operations() {
        let test_dir = "test_directory";
        
        // Create directory
        let dir = LinearDirectory::create(test_dir).unwrap();
        assert!(dir.path().exists());
        
        // Create test file in directory
        let test_file = Path::new(test_dir).join("test.txt");
        fs::write(&test_file, "test content").unwrap();
        
        // List contents
        let (dir, contents) = dir.list_contents().unwrap();
        assert_eq!(contents.len(), 1);
        
        // List files
        let (dir, files) = dir.list_files().unwrap();
        assert_eq!(files.len(), 1);
        
        // Clean up
        dir.delete_recursive().unwrap();
    }

    #[test]
    fn test_error_handling() {
        // Test file not found
        let result = LinearFile::open("nonexistent_file.txt");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IOError::FileNotFound(_) => {},
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_serialization() {
        use super::serialization::JsonSerializer;
        use crate::stdlib::collections::{LinearVec, LinearMap};
        
        // Test vector serialization
        let vec = LinearVec::new()
            .push("item1".to_string())
            .push("item2".to_string())
            .push("item3".to_string());
        
        let json = JsonSerializer::serialize_vec(&vec).unwrap();
        assert_eq!(json, "[\"item1\", \"item2\", \"item3\"]");
        
        // Test map serialization
        let (map, _) = LinearMap::new().insert("key1".to_string(), "value1".to_string());
        let (map, _) = map.insert("key2".to_string(), "value2".to_string());
        
        let json = JsonSerializer::serialize_map(&map).unwrap();
        // Note: HashMap iteration order is not guaranteed, so we just check it's valid JSON structure
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("key1") && json.contains("value1"));
        assert!(json.contains("key2") && json.contains("value2"));
    }
}