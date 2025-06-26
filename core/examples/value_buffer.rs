use sgrmath_core::{WgpuContext, ValueBuffer};
use std::error::Error;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the GPU context
    println!("Initializing GPU context...");
    let context = WgpuContext::new_async().await;
    println!("GPU context initialized successfully!");

    // Create a value buffer with initial value
    println!("\nCreating a value buffer...");
    let value = 42.0f32;
    let buffer = ValueBuffer::init(&context, &value);
    println!("Created buffer with value: {}", value);

    // Update the buffer value
    println!("\nUpdating buffer value...");
    let new_value = 100.0f32;
    buffer.set(&context, &new_value);
    println!("Updated buffer with new value: {}", new_value);

    Ok(())
} 