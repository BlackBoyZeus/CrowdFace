// utils/system.rs - System information utilities

use crate::error::Result;
use serde_json::{json, Value};

/// Get system information
pub fn get_system_info() -> Value {
    let os_info = get_os_info();
    let cpu_info = get_cpu_info();
    let memory_info = get_memory_info();
    let gpu_info = get_gpu_info();
    
    json!({
        "os": os_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "crowdface": {
            "version": crate::VERSION,
            "cuda_available": crate::cuda_available(),
            "aws_available": crate::aws_available(),
        }
    })
}

/// Get OS information
fn get_os_info() -> Value {
    json!({
        "name": std::env::consts::OS,
        "family": std::env::consts::FAMILY,
        "arch": std::env::consts::ARCH,
    })
}

/// Get CPU information
fn get_cpu_info() -> Value {
    json!({
        "num_cores": num_cpus::get(),
        "num_physical_cores": num_cpus::get_physical(),
    })
}

/// Get memory information
fn get_memory_info() -> Value {
    // In a real implementation, this would get actual memory information
    json!({
        "total_mb": 16384,
        "free_mb": 8192,
    })
}

/// Get GPU information
fn get_gpu_info() -> Value {
    // Get available GPUs
    let gpus = match super::gpu::get_available_gpus() {
        Ok(gpus) => gpus,
        Err(_) => Vec::new(),
    };
    
    // Convert to JSON
    let gpu_json: Vec<Value> = gpus.iter().map(|gpu| {
        json!({
            "id": gpu.id,
            "name": gpu.name,
            "total_memory_mb": gpu.total_memory_mb,
            "free_memory_mb": gpu.free_memory_mb,
        })
    }).collect();
    
    json!({
        "cuda_available": super::gpu::is_cuda_available(),
        "devices": gpu_json,
    })
}
