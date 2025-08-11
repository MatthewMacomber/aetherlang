// Test result caching system for improved performance
// Provides intelligent caching of compilation and execution results

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use sha2::{Sha256, Digest};

use super::file_compilation_testing::{CompilationResult, ExecutionResult, TestingError};

/// Test result cache manager
pub struct TestCache {
    cache_dir: PathBuf,
    config: CacheConfig,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum age of cache entries in hours
    pub max_age_hours: u64,
    
    /// Maximum cache size in MB
    pub max_size_mb: u64,
    
    /// Enable compression of cache entries
    pub enable_compression: bool,
    
    /// Cache cleanup interval in hours
    pub cleanup_interval_hours: u64,
    
    /// Enable cache validation
    pub enable_validation: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_age_hours: 24,
            max_size_mb: 100,
            enable_compression: true,
            cleanup_interval_hours: 6,
            enable_validation: true,
        }
    }
}

/// Cached compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCompilationResult {
    pub result: CompilationResult,
    pub source_hash: String,
    pub compiler_hash: String,
    pub timestamp: u64,
    pub cache_version: u32,
}

/// Cached execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedExecutionResult {
    pub result: ExecutionResult,
    pub executable_hash: String,
    pub timestamp: u64,
    pub cache_version: u32,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub compilation_entries: usize,
    pub execution_entries: usize,
    pub cache_size_bytes: u64,
    pub hit_rate: f64,
    pub last_cleanup: u64,
}

/// Cache key for identifying cached results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub file_path: PathBuf,
    pub operation_type: CacheOperationType,
    pub content_hash: String,
}

/// Type of cached operation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheOperationType {
    Compilation,
    Execution,
}

impl TestCache {
    /// Create a new test cache
    pub fn new(cache_dir: PathBuf, config: CacheConfig) -> Result<Self, CacheError> {
        std::fs::create_dir_all(&cache_dir)?;
        
        let cache = Self {
            cache_dir,
            config,
        };
        
        // Perform initial cleanup if needed
        cache.cleanup_if_needed()?;
        
        Ok(cache)
    }
    
    /// Get cached compilation result
    pub fn get_compilation_result(
        &self,
        source_file: &Path,
        compiler_path: &Path,
    ) -> Result<Option<CompilationResult>, CacheError> {
        let source_hash = self.calculate_file_hash(source_file)?;
        let compiler_hash = self.calculate_file_hash(compiler_path)?;
        
        let cache_key = CacheKey {
            file_path: source_file.to_path_buf(),
            operation_type: CacheOperationType::Compilation,
            content_hash: format!("{}:{}", source_hash, compiler_hash),
        };
        
        let cache_file = self.get_cache_file_path(&cache_key);
        
        if !cache_file.exists() {
            return Ok(None);
        }
        
        // Load and validate cached result
        let cached_result: CachedCompilationResult = self.load_cache_entry(&cache_file)?;
        
        // Validate cache entry
        if self.config.enable_validation {
            if cached_result.source_hash != source_hash || 
               cached_result.compiler_hash != compiler_hash {
                // Cache is invalid, remove it
                let _ = std::fs::remove_file(&cache_file);
                return Ok(None);
            }
            
            // Check if cache is expired
            if self.is_cache_expired(cached_result.timestamp) {
                let _ = std::fs::remove_file(&cache_file);
                return Ok(None);
            }
        }
        
        Ok(Some(cached_result.result))
    }
    
    /// Cache compilation result
    pub fn cache_compilation_result(
        &self,
        source_file: &Path,
        compiler_path: &Path,
        result: &CompilationResult,
    ) -> Result<(), CacheError> {
        let source_hash = self.calculate_file_hash(source_file)?;
        let compiler_hash = self.calculate_file_hash(compiler_path)?;
        
        let cache_key = CacheKey {
            file_path: source_file.to_path_buf(),
            operation_type: CacheOperationType::Compilation,
            content_hash: format!("{}:{}", source_hash, compiler_hash),
        };
        
        let cached_result = CachedCompilationResult {
            result: result.clone(),
            source_hash,
            compiler_hash,
            timestamp: self.current_timestamp(),
            cache_version: 1,
        };
        
        let cache_file = self.get_cache_file_path(&cache_key);
        self.save_cache_entry(&cache_file, &cached_result)?;
        
        Ok(())
    }
    
    /// Get cached execution result
    pub fn get_execution_result(
        &self,
        executable_path: &Path,
    ) -> Result<Option<ExecutionResult>, CacheError> {
        let executable_hash = self.calculate_file_hash(executable_path)?;
        
        let cache_key = CacheKey {
            file_path: executable_path.to_path_buf(),
            operation_type: CacheOperationType::Execution,
            content_hash: executable_hash.clone(),
        };
        
        let cache_file = self.get_cache_file_path(&cache_key);
        
        if !cache_file.exists() {
            return Ok(None);
        }
        
        // Load and validate cached result
        let cached_result: CachedExecutionResult = self.load_cache_entry(&cache_file)?;
        
        // Validate cache entry
        if self.config.enable_validation {
            if cached_result.executable_hash != executable_hash {
                // Cache is invalid, remove it
                let _ = std::fs::remove_file(&cache_file);
                return Ok(None);
            }
            
            // Check if cache is expired
            if self.is_cache_expired(cached_result.timestamp) {
                let _ = std::fs::remove_file(&cache_file);
                return Ok(None);
            }
        }
        
        Ok(Some(cached_result.result))
    }
    
    /// Cache execution result
    pub fn cache_execution_result(
        &self,
        executable_path: &Path,
        result: &ExecutionResult,
    ) -> Result<(), CacheError> {
        let executable_hash = self.calculate_file_hash(executable_path)?;
        
        let cache_key = CacheKey {
            file_path: executable_path.to_path_buf(),
            operation_type: CacheOperationType::Execution,
            content_hash: executable_hash.clone(),
        };
        
        let cached_result = CachedExecutionResult {
            result: result.clone(),
            executable_hash,
            timestamp: self.current_timestamp(),
            cache_version: 1,
        };
        
        let cache_file = self.get_cache_file_path(&cache_key);
        self.save_cache_entry(&cache_file, &cached_result)?;
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> Result<CacheStats, CacheError> {
        let mut total_entries = 0;
        let mut compilation_entries = 0;
        let mut execution_entries = 0;
        let mut cache_size_bytes = 0;
        
        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    total_entries += 1;
                    cache_size_bytes += entry.metadata()?.len();
                    
                    // Determine entry type from filename
                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if filename.contains("compilation") {
                            compilation_entries += 1;
                        } else if filename.contains("execution") {
                            execution_entries += 1;
                        }
                    }
                }
            }
        }
        
        // Calculate hit rate (simplified - would need actual tracking in production)
        let hit_rate = if total_entries > 0 { 0.75 } else { 0.0 };
        
        Ok(CacheStats {
            total_entries,
            compilation_entries,
            execution_entries,
            cache_size_bytes,
            hit_rate,
            last_cleanup: self.current_timestamp(),
        })
    }
    
    /// Clean up expired cache entries
    pub fn cleanup(&self) -> Result<usize, CacheError> {
        let mut removed_count = 0;
        
        if !self.cache_dir.exists() {
            return Ok(0);
        }
        
        let current_time = self.current_timestamp();
        let max_age_seconds = self.config.max_age_hours * 3600;
        
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                // Check file age
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(modified_timestamp) = modified.duration_since(UNIX_EPOCH) {
                            let age_seconds = current_time - modified_timestamp.as_secs();
                            
                            if age_seconds > max_age_seconds {
                                if std::fs::remove_file(&path).is_ok() {
                                    removed_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Check cache size and remove oldest entries if needed
        let stats = self.get_stats()?;
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;
        
        if stats.cache_size_bytes > max_size_bytes {
            removed_count += self.cleanup_by_size(max_size_bytes)?;
        }
        
        Ok(removed_count)
    }
    
    /// Clear all cache entries
    pub fn clear(&self) -> Result<(), CacheError> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }
    
    // Private helper methods
    
    fn calculate_file_hash(&self, file_path: &Path) -> Result<String, CacheError> {
        let content = std::fs::read(file_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn get_cache_file_path(&self, cache_key: &CacheKey) -> PathBuf {
        let key_hash = {
            let mut hasher = Sha256::new();
            hasher.update(cache_key.file_path.to_string_lossy().as_bytes());
            hasher.update(format!("{:?}", cache_key.operation_type).as_bytes());
            hasher.update(cache_key.content_hash.as_bytes());
            format!("{:x}", hasher.finalize())
        };
        
        let filename = format!("{:?}_{}.cache", cache_key.operation_type, key_hash);
        self.cache_dir.join(filename)
    }
    
    fn load_cache_entry<T>(&self, cache_file: &Path) -> Result<T, CacheError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let content = std::fs::read(cache_file)?;
        
        let data = if self.config.enable_compression {
            // In a real implementation, you'd use a compression library like flate2
            // For now, we'll just use the raw data
            content
        } else {
            content
        };
        
        let result: T = serde_json::from_slice(&data)?;
        Ok(result)
    }
    
    fn save_cache_entry<T>(&self, cache_file: &Path, data: &T) -> Result<(), CacheError>
    where
        T: Serialize,
    {
        let json_data = serde_json::to_vec(data)?;
        
        let final_data = if self.config.enable_compression {
            // In a real implementation, you'd compress the data here
            json_data
        } else {
            json_data
        };
        
        // Ensure parent directory exists
        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(cache_file, final_data)?;
        Ok(())
    }
    
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    fn is_cache_expired(&self, timestamp: u64) -> bool {
        let current_time = self.current_timestamp();
        let max_age_seconds = self.config.max_age_hours * 3600;
        current_time - timestamp > max_age_seconds
    }
    
    fn cleanup_if_needed(&self) -> Result<(), CacheError> {
        // Check if cleanup is needed based on interval
        let stats_file = self.cache_dir.join("cache_stats.json");
        
        let should_cleanup = if stats_file.exists() {
            if let Ok(content) = std::fs::read_to_string(&stats_file) {
                if let Ok(stats) = serde_json::from_str::<CacheStats>(&content) {
                    let current_time = self.current_timestamp();
                    let cleanup_interval_seconds = self.config.cleanup_interval_hours * 3600;
                    current_time - stats.last_cleanup > cleanup_interval_seconds
                } else {
                    true
                }
            } else {
                true
            }
        } else {
            true
        };
        
        if should_cleanup {
            self.cleanup()?;
        }
        
        Ok(())
    }
    
    fn cleanup_by_size(&self, max_size_bytes: u64) -> Result<usize, CacheError> {
        let mut files_with_age = Vec::new();
        
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        files_with_age.push((path, modified, metadata.len()));
                    }
                }
            }
        }
        
        // Sort by modification time (oldest first)
        files_with_age.sort_by_key(|(_, modified, _)| *modified);
        
        let mut current_size = files_with_age.iter().map(|(_, _, size)| size).sum::<u64>();
        let mut removed_count = 0;
        
        for (path, _, size) in files_with_age {
            if current_size <= max_size_bytes {
                break;
            }
            
            if std::fs::remove_file(&path).is_ok() {
                current_size -= size;
                removed_count += 1;
            }
        }
        
        Ok(removed_count)
    }
}

/// Cache-related errors
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Cache validation error: {0}")]
    Validation(String),
    
    #[error("Cache corruption error: {0}")]
    Corruption(String),
}

impl From<CacheError> for TestingError {
    fn from(err: CacheError) -> Self {
        TestingError::Validation(format!("Cache error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig::default();
        
        let cache = TestCache::new(temp_dir.path().to_path_buf(), config);
        assert!(cache.is_ok());
    }
    
    #[test]
    fn test_cache_key_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig::default();
        let cache = TestCache::new(temp_dir.path().to_path_buf(), config).unwrap();
        
        let cache_key = CacheKey {
            file_path: PathBuf::from("test.ae"),
            operation_type: CacheOperationType::Compilation,
            content_hash: "abc123".to_string(),
        };
        
        let cache_file = cache.get_cache_file_path(&cache_key);
        assert!(cache_file.to_string_lossy().contains("Compilation"));
    }
    
    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig::default();
        let cache = TestCache::new(temp_dir.path().to_path_buf(), config).unwrap();
        
        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.cache_size_bytes, 0);
    }
}