use sgrmath_core::{WgpuContext, StorageBuffer, ReadbackBuffer, Size};
use std::error::Error;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the GPU context
    println!("Initializing GPU context...");
    let context = WgpuContext::new_async().await;
    println!("GPU context initialized successfully!");

    // Create a storage buffer with some data
    println!("\nCreating a storage buffer...");
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let storage_buffer = StorageBuffer::new::<f32, _>(&context, Size::from(data.len()));
    storage_buffer.update_buffer_range::<f32>(&context, &data, 0);
    println!("Created storage buffer with data: {:?}", data);

    // Create a readback buffer and read data from storage buffer
    println!("\nReading data back from GPU...");
    let mut readback_buffer = ReadbackBuffer::new::<f32, _>(&context, Size::from(data.len()));
    readback_buffer.scale::<f32, _>(&context, data.len());
    
    let read_data = readback_buffer.read::<f32>(&context, &storage_buffer, 0, data.len());
    println!("Read data from GPU: {:?}", read_data);

    Ok(())
} 