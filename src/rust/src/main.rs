// main.rs - Main entry point for CrowdFace server

use crowdface::{
    Config, init,
    create_pipeline,
    api::ApiServer,
    models::{create_default_models, PlacementParams},
};

fn main() -> crowdface::Result<()> {
    // Load configuration
    let config = Config::default();
    
    // Initialize CrowdFace
    init(config.clone())?;
    
    // Create pipeline
    let pipeline = create_pipeline()
        .with_config(config.clone())
        .build()?;
    
    // Create API server
    let server = ApiServer::new(config.api.clone(), pipeline);
    
    // Start server
    server.start()?;
    
    println!("CrowdFace server started on {}:{}", config.api.host, config.api.port);
    println!("Press Ctrl+C to stop");
    
    // Wait for Ctrl+C
    let (tx, rx) = std::sync::mpsc::channel();
    ctrlc::set_handler(move || {
        tx.send(()).expect("Could not send signal on channel");
    }).expect("Error setting Ctrl-C handler");
    
    rx.recv().expect("Could not receive from channel");
    
    // Stop server
    server.stop()?;
    
    println!("CrowdFace server stopped");
    
    Ok(())
}
