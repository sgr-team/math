struct BlxAlphaOptions {
    generation_offset: u32,
    vector_length: u32,
    parents_count: u32,
    min: f32,
    max: f32,
}

@group(0) @binding(0) var<storage, read> options: BlxAlphaOptions;
@group(0) @binding(1) var<storage, read> population: array<f32>;
@group(0) @binding(2) var<storage, read> parents: array<u32>;
@group(0) @binding(3) var<storage, read> random: array<f32>;
@group(0) @binding(4) var<storage, read_write> generation: array<f32>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_index = global_id.y * options.vector_length + global_id.x;
    let parents_start = global_id.y * options.parents_count;
    let parents_end = parents_start + options.parents_count;

    var min = population[parents[parents_start] * options.vector_length + global_id.x]; 
    var max = min;
    var sum = min;
    for (var i = parents_start + 1u; i < parents_end; i = i + 1u) {
        let value = population[parents[i] * options.vector_length + global_id.x]; 
        sum += value;
        
        if (value < min) { min = value; }
        if (value > max) { max = value; }
    }

    let center = sum / f32(options.parents_count);
    let delta = max - min;
    
    var value = center + delta * random[global_index];
    if (value > options.max) { value = options.max; }
    if (value < options.min) { value = options.min; }
    
    generation[options.generation_offset + global_index] = value;
}
