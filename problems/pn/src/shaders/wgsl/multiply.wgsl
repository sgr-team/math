@group(0) @binding(0) var<uniform> options: PNPOptions;
@group(0) @binding(1) var<storage, read> examples: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_start = global_id.y * options.vector_length;
    let example_start = global_id.x * options.vector_length;

    var sum: f32 = 0.0;
    for (var i = 0u; i < options.vector_length; i = i + 1u) {
        sum += vectors[vector_start + i] * examples[example_start + i];
    }
    
    output[global_id.y * options.examples_count + global_id.x] = sum;
}
