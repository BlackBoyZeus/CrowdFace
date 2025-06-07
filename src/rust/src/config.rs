// config.rs - Configuration for CrowdFace

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for CrowdFace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// General configuration
    pub general: GeneralConfig,
    
    /// GPU configuration
    pub gpu: GpuConfig,
    
    /// Model configuration
    pub models: ModelConfig,
    
    /// Pipeline configuration
    pub pipeline: PipelineConfig,
    
    /// API configuration
    pub api: ApiConfig,
    
    /// AWS configuration
    #[serde(default)]
    pub aws: AwsConfig,
}

/// General configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
    
    /// Temporary directory
    #[serde(default = "default_temp_dir")]
    pub temp_dir: PathBuf,
    
    /// Number of worker threads
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    #[serde(default = "default_true")]
    pub enabled: bool,
    
    /// GPU device IDs to use
    #[serde(default)]
    pub device_ids: Vec<i32>,
    
    /// Memory limit per GPU in MB
    #[serde(default = "default_gpu_memory_limit")]
    pub memory_limit_mb: usize,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Segmentation model configuration
    pub segmentation: SegmentationConfig,
    
    /// Matting model configuration
    pub matting: MattingConfig,
    
    /// Injection model configuration
    pub injection: InjectionConfig,
}

/// Segmentation model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationConfig {
    /// Model path
    pub model_path: PathBuf,
    
    /// Model variant
    #[serde(default = "default_sam2_variant")]
    pub variant: String,
    
    /// Confidence threshold
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    
    /// Downsample ratio
    #[serde(default = "default_downsample_ratio")]
    pub downsample_ratio: f32,
}

/// Matting model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MattingConfig {
    /// Model path
    pub model_path: PathBuf,
    
    /// Model variant
    #[serde(default = "default_rvm_variant")]
    pub variant: String,
    
    /// Sequence chunk size
    #[serde(default = "default_sequence_chunk")]
    pub sequence_chunk: usize,
}

/// Injection model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Model path
    pub model_path: PathBuf,
    
    /// Model variant
    #[serde(default = "default_bagel_variant")]
    pub variant: String,
    
    /// CFG scale
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Input resolution
    #[serde(default = "default_input_resolution")]
    pub input_resolution: (usize, usize),
    
    /// Output resolution
    #[serde(default = "default_output_resolution")]
    pub output_resolution: (usize, usize),
    
    /// Frame rate
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f32,
    
    /// Buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    
    /// Real-time mode
    #[serde(default = "default_true")]
    pub real_time: bool,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Host
    #[serde(default = "default_host")]
    pub host: String,
    
    /// Port
    #[serde(default = "default_port")]
    pub port: u16,
    
    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,
    
    /// API keys
    #[serde(default)]
    pub api_keys: Vec<String>,
}

/// AWS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AwsConfig {
    /// Enable AWS integration
    #[serde(default)]
    pub enabled: bool,
    
    /// Region
    #[serde(default)]
    pub region: Option<String>,
    
    /// S3 bucket
    #[serde(default)]
    pub s3_bucket: Option<String>,
    
    /// SageMaker endpoint
    #[serde(default)]
    pub sagemaker_endpoint: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            gpu: GpuConfig::default(),
            models: ModelConfig::default(),
            pipeline: PipelineConfig::default(),
            api: ApiConfig::default(),
            aws: AwsConfig::default(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            temp_dir: default_temp_dir(),
            worker_threads: default_worker_threads(),
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            device_ids: vec![0],
            memory_limit_mb: default_gpu_memory_limit(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            segmentation: SegmentationConfig::default(),
            matting: MattingConfig::default(),
            injection: InjectionConfig::default(),
        }
    }
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/sam2.1_hiera_base_plus.pt"),
            variant: default_sam2_variant(),
            confidence_threshold: default_confidence_threshold(),
            downsample_ratio: default_downsample_ratio(),
        }
    }
}

impl Default for MattingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/rvm_mobilenetv3.pth"),
            variant: default_rvm_variant(),
            sequence_chunk: default_sequence_chunk(),
        }
    }
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/bagel-7b-mot"),
            variant: default_bagel_variant(),
            cfg_scale: default_cfg_scale(),
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_resolution: default_input_resolution(),
            output_resolution: default_output_resolution(),
            frame_rate: default_frame_rate(),
            buffer_size: default_buffer_size(),
            real_time: default_true(),
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            cors_enabled: default_true(),
            api_keys: Vec::new(),
        }
    }
}

// Default functions
fn default_log_level() -> String {
    "info".to_string()
}

fn default_temp_dir() -> PathBuf {
    std::env::temp_dir().join("crowdface")
}

fn default_worker_threads() -> usize {
    num_cpus::get()
}

fn default_true() -> bool {
    true
}

fn default_gpu_memory_limit() -> usize {
    4096 // 4GB
}

fn default_sam2_variant() -> String {
    "base_plus".to_string()
}

fn default_rvm_variant() -> String {
    "mobilenetv3".to_string()
}

fn default_bagel_variant() -> String {
    "7b".to_string()
}

fn default_confidence_threshold() -> f32 {
    0.5
}

fn default_downsample_ratio() -> f32 {
    0.25
}

fn default_sequence_chunk() -> usize {
    12
}

fn default_cfg_scale() -> f32 {
    7.0
}

fn default_input_resolution() -> (usize, usize) {
    (1920, 1080) // Full HD
}

fn default_output_resolution() -> (usize, usize) {
    (1920, 1080) // Full HD
}

fn default_frame_rate() -> f32 {
    30.0
}

fn default_buffer_size() -> usize {
    30 // 1 second at 30 FPS
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}
