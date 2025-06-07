// utils/gpu.rs - GPU utilities

use crate::error::{Error, Result};
use crate::config::GpuConfig;

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // In a real implementation, this would check if CUDA is available
        // using the tch crate (PyTorch bindings for Rust)
        true
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get available GPUs
pub struct GpuInfo {
    /// GPU ID
    pub id: i32,
    
    /// GPU name
    pub name: String,
    
    /// Total memory in MB
    pub total_memory_mb: usize,
    
    /// Free memory in MB
    pub free_memory_mb: usize,
}

/// Check GPU availability
pub fn check_gpu_availability(config: &GpuConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    
    if !is_cuda_available() {
        return Err(Error::Gpu("CUDA is not available".to_string()));
    }
    
    // In a real implementation, this would check if the specified GPUs are available
    
    Ok(())
}

/// Get available GPUs
pub fn get_available_gpus() -> Result<Vec<GpuInfo>> {
    if !is_cuda_available() {
        return Ok(Vec::new());
    }
    
    // In a real implementation, this would get the available GPUs
    // using the tch crate (PyTorch bindings for Rust)
    
    // Return a dummy GPU for demonstration
    Ok(vec![
        GpuInfo {
            id: 0,
            name: "NVIDIA GeForce RTX 3080".to_string(),
            total_memory_mb: 10240,
            free_memory_mb: 8192,
        }
    ])
}

/// Check if a specific GPU is available
pub fn is_gpu_available(device_id: i32) -> bool {
    if !is_cuda_available() {
        return false;
    }
    
    // In a real implementation, this would check if the specified GPU is available
    
    device_id == 0 // Assume only GPU 0 is available for demonstration
}
