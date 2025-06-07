// error.rs - Error handling for CrowdFace

use thiserror::Error;
use std::io;

/// Result type for CrowdFace operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for CrowdFace
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Segmentation error: {0}")]
    Segmentation(String),

    #[error("Matting error: {0}")]
    Matting(String),

    #[error("Injection error: {0}")]
    Injection(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("AWS error: {0}")]
    Aws(String),

    #[error("Video processing error: {0}")]
    VideoProcessing(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("FFI error: {0}")]
    Ffi(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Unknown(s.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Unknown(s)
    }
}
