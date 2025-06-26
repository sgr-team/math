@group(0) @binding(0) var<storage, read> s_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_buffer: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < 5 { 
        t_buffer[global_id.x] = 2.0 * s_buffer[global_id.x] + 1.0; 
    } else { 
        t_buffer[global_id.x] = 10.0;
    };
}