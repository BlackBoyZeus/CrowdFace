// models/mod.rs - Models module

pub mod segmentation;
pub mod matting;
pub mod injection;

// Re-exports
pub use segmentation::{SegmentationModel, SAM2Model, SegmentationResult, Mask};
pub use matting::{MattingModel, RVMModel, MattingResult};
pub use injection::{InjectionModel, BAGELModel, InjectionResult, PlacementParams, PlacementInfo};

/// Create all models with default configuration
pub fn create_default_models() -> crate::error::Result<(SAM2Model, RVMModel, BAGELModel)> {
    let segmentation = segmentation::create_sam2_model()?;
    let matting = matting::create_rvm_model()?;
    let injection = injection::create_bagel_model()?;
    
    Ok((segmentation, matting, injection))
}
