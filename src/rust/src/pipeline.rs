// pipeline.rs - Pipeline implementation

use crate::error::{Error, Result};
use crate::config::Config;
use crate::models::{
    SegmentationModel, SAM2Model, SegmentationResult,
    MattingModel, RVMModel, MattingResult,
    InjectionModel, BAGELModel, InjectionResult, PlacementParams
};
use std::sync::{Arc, Mutex};

/// Video frame representation
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame data (RGB)
    pub data: Vec<u8>,
    
    /// Width of the frame
    pub width: usize,
    
    /// Height of the frame
    pub height: usize,
    
    /// Frame index
    pub index: usize,
    
    /// Timestamp in milliseconds
    pub timestamp: u64,
}

/// Pipeline processing result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Original frame
    pub original_frame: Frame,
    
    /// Segmentation result
    pub segmentation: Option<SegmentationResult>,
    
    /// Matting result
    pub matting: Option<MattingResult>,
    
    /// Injection result
    pub injection: Option<InjectionResult>,
    
    /// Final composited frame
    pub composited_frame: Frame,
    
    /// Processing time in milliseconds
    pub processing_time: u64,
}

/// Pipeline for video processing
pub struct Pipeline {
    config: Config,
    segmentation_model: Arc<Mutex<Box<dyn SegmentationModel>>>,
    matting_model: Arc<Mutex<Box<dyn MattingModel>>>,
    injection_model: Arc<Mutex<Box<dyn InjectionModel>>>,
    frame_buffer: Vec<Frame>,
    result_buffer: Vec<PipelineResult>,
}

/// Pipeline builder
pub struct PipelineBuilder {
    config: Config,
    segmentation_model: Option<Box<dyn SegmentationModel>>,
    matting_model: Option<Box<dyn MattingModel>>,
    injection_model: Option<Box<dyn InjectionModel>>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            segmentation_model: None,
            matting_model: None,
            injection_model: None,
        }
    }
    
    /// Set the configuration
    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }
    
    /// Set the segmentation model
    pub fn with_segmentation_model<M: SegmentationModel + 'static>(mut self, model: M) -> Self {
        self.segmentation_model = Some(Box::new(model));
        self
    }
    
    /// Set the matting model
    pub fn with_matting_model<M: MattingModel + 'static>(mut self, model: M) -> Self {
        self.matting_model = Some(Box::new(model));
        self
    }
    
    /// Set the injection model
    pub fn with_injection_model<M: InjectionModel + 'static>(mut self, model: M) -> Self {
        self.injection_model = Some(Box::new(model));
        self
    }
    
    /// Build the pipeline
    pub fn build(self) -> Result<Pipeline> {
        // Create default models if not provided
        let segmentation_model = match self.segmentation_model {
            Some(model) => model,
            None => {
                let mut model = SAM2Model::new(self.config.models.segmentation.clone());
                model.init()?;
                Box::new(model)
            }
        };
        
        let matting_model = match self.matting_model {
            Some(model) => model,
            None => {
                let mut model = RVMModel::new(self.config.models.matting.clone());
                model.init()?;
                Box::new(model)
            }
        };
        
        let injection_model = match self.injection_model {
            Some(model) => model,
            None => {
                let mut model = BAGELModel::new(self.config.models.injection.clone());
                model.init()?;
                Box::new(model)
            }
        };
        
        Ok(Pipeline {
            config: self.config,
            segmentation_model: Arc::new(Mutex::new(segmentation_model)),
            matting_model: Arc::new(Mutex::new(matting_model)),
            injection_model: Arc::new(Mutex::new(injection_model)),
            frame_buffer: Vec::new(),
            result_buffer: Vec::new(),
        })
    }
}

impl Pipeline {
    /// Process a single frame
    pub fn process_frame(&mut self, frame: Frame, ad_content: &[u8]) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Segmentation
        let segmentation_result = {
            let mut model = self.segmentation_model.lock().map_err(|_| Error::Pipeline("Failed to lock segmentation model".to_string()))?;
            model.process_frame(&frame.data, frame.width, frame.height)?
        };
        
        // Step 2: Matting
        let matting_result = {
            let mut model = self.matting_model.lock().map_err(|_| Error::Pipeline("Failed to lock matting model".to_string()))?;
            model.process_frame(&frame.data, frame.width, frame.height)?
        };
        
        // Step 3: Injection
        let injection_result = {
            let model = self.injection_model.lock().map_err(|_| Error::Pipeline("Failed to lock injection model".to_string()))?;
            model.process_frame(&frame.data, &matting_result.alpha, ad_content, frame.width, frame.height)?
        };
        
        // Create result
        let composited_frame = Frame {
            data: injection_result.composited.clone(),
            width: frame.width,
            height: frame.height,
            index: frame.index,
            timestamp: frame.timestamp,
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let result = PipelineResult {
            original_frame: frame,
            segmentation: Some(segmentation_result),
            matting: Some(matting_result),
            injection: Some(injection_result),
            composited_frame,
            processing_time,
        };
        
        // Add to result buffer
        self.result_buffer.push(result.clone());
        
        Ok(result)
    }
    
    /// Process a batch of frames
    pub fn process_batch(&mut self, frames: Vec<Frame>, ad_contents: Vec<&[u8]>) -> Result<Vec<PipelineResult>> {
        if frames.len() != ad_contents.len() {
            return Err(Error::Pipeline("Number of frames and ad contents must match".to_string()));
        }
        
        let mut results = Vec::with_capacity(frames.len());
        
        for (i, frame) in frames.into_iter().enumerate() {
            let result = self.process_frame(frame, ad_contents[i])?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Reset all models
    pub fn reset(&mut self) -> Result<()> {
        {
            let mut model = self.matting_model.lock().map_err(|_| Error::Pipeline("Failed to lock matting model".to_string()))?;
            model.reset_state()?;
        }
        
        self.frame_buffer.clear();
        self.result_buffer.clear();
        
        Ok(())
    }
    
    /// Set ad placement parameters
    pub fn set_placement_params(&mut self, params: PlacementParams) -> Result<()> {
        let mut model = self.injection_model.lock().map_err(|_| Error::Pipeline("Failed to lock injection model".to_string()))?;
        model.set_placement_params(params)?;
        Ok(())
    }
    
    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Get the result buffer
    pub fn result_buffer(&self) -> &[PipelineResult] {
        &self.result_buffer
    }
    
    /// Clear the result buffer
    pub fn clear_result_buffer(&mut self) {
        self.result_buffer.clear();
    }
}
