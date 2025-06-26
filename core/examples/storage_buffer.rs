use sgrmath_core::{WgpuContext, StorageBuffer, Size};
use std::error::Error;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the GPU context
    println!("Initializing GPU context...");
    let context = WgpuContext::new_async().await;
    println!("GPU context initialized successfully!");

    // Create a storage buffer with initial data
    println!("\nCreating a storage buffer...");
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buffer = StorageBuffer::new::<f32, _>(&context, Size::from(data.len()));
    buffer.update_buffer_range::<f32>(&context, &data, 0);
    println!("Created buffer with data: {:?}", data);

    // Update a portion of the buffer
    println!("\nUpdating buffer data...");
    let new_data = vec![5.0f32, 2.0];
    buffer.update_buffer_range::<f32>(&context, &new_data, 1);
    println!("Updated buffer with new data: {:?}", new_data);

    Ok(())
} 