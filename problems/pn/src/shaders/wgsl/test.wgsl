@group(0) @binding(0) var<uniform> options: PNPOptions;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> permutations: array<u32>;
@group(0) @binding(3) var<storage, read_write> permutation_labels: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let offset = global_id.x * options.examples_count;
    var result = 0.0;
    for (var e = 0u; e < options.examples_count; e++) {
        if (permutation_labels[permutations[offset + e]] == labels[e]) {
            result += 1.0;
        }
    }

    output[global_id.x] = result;
}
