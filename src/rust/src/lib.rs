// lib.rs - Main library file for CrowdFace

mod config;
mod error;
pub mod models;
mod pipeline;
mod utils;
pub mod api;

// Re-exports
pub use config::Config;
pub use error::{Error, Result};
pub use models::{segmentation, matting, injection};
pub use pipeline::{Pipeline, PipelineBuilder};
pub use api::{ApiServer, ApiClient};

/// CrowdFace version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the CrowdFace library with the given configuration
pub fn init(config: Config) -> Result<()> {
    utils::logging::init()?;
    utils::gpu::check_gpu_availability(&config.gpu)?;
    Ok(())
}

/// Create a new pipeline with default configuration
pub fn create_pipeline() -> PipelineBuilder {
    PipelineBuilder::new()
}

/// Check if CUDA is available
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        utils::gpu::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Check if AWS integration is available
pub fn aws_available() -> bool {
    #[cfg(feature = "aws")]
    {
        true
    }
    #[cfg(not(feature = "aws"))]
    {
        false
    }
}

/// Get system information
pub fn system_info() -> serde_json::Value {
    utils::system::get_system_info()
}
