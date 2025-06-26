use sgrmath_core::{WgpuContext, ValueBuffer};
use std::error::Error;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the GPU context
    println!("Initializing GPU context...");
    let context = WgpuContext::new_async().await;
    println!("GPU context initialized successfully!");

    // Create a simple value buffer
    println!("\nCreating a value buffer...");
    let value = 42.0f32;
    let _buffer = ValueBuffer::init(&context, &value);
    println!("Created buffer with value: {}", value);

    Ok(())
} 