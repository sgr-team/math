use sgrmath_core::{StorageBuffer, ReadbackBuffer, WgpuContext, Shader};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Initialize GPU context
    let context = WgpuContext::new_async().await;

    // Create buffers
    let source = StorageBuffer::new::<i32, _>(&context, 10);
    let target = StorageBuffer::new::<i32, _>(&context, 10);
    let readback = ReadbackBuffer::new::<i32, _>(&context, 10);
    
    // Initialize input buffer with data
    source.update_buffer_range(&context, &(0_i32..10).collect::<Vec<_>>(), 0);
    target.update_buffer_range(&context, &(0_i32..10).collect::<Vec<_>>(), 0);

    // Create compute shader
    let shader = Shader::new(
        &context,
        "multiply_by_2",
        r#"
        @group(0) @binding(0) var<storage, read> input: array<i32>;
        @group(0) @binding(1) var<storage, read_write> output: array<i32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            output[global_id.x] = input[global_id.x] * 2;
        }
        "#
    );

    // Execute the shader
    shader.execute_with_params(&context, 10, &[&source, &target]);

    println!("Output data: {:?}", readback.read::<i32>(&context, &target, 0, 10));
} 