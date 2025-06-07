// utils/logging.rs - Logging utilities

use crate::error::Result;
use crate::config::GeneralConfig;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

/// Initialize the logger
pub fn init_logger(config: &GeneralConfig) -> Result<()> {
    let level = match config.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| crate::error::Error::Config(format!("Failed to set global default subscriber: {}", e)))?;
    
    info!("Logger initialized with level: {}", level);
    
    Ok(())
}

/// Log a message at the info level
pub fn log_info(message: &str) {
    info!("{}", message);
}

/// Initialize the logger with default configuration
pub fn init() -> Result<()> {
    init_logger(&GeneralConfig::default())
}
