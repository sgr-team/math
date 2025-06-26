@group(0) @binding(0) var<uniform> options: PNPOptions;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> permutations: array<u32>;
@group(0) @binding(3) var<storage, read_write> permutation_labels: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let offset = global_id.x * options.examples_count;
    let permutation_size = options.permutations_count * options.outputs_count;
    let permutation_labels_start = global_id.x * permutation_size;
    let permutation_labels_end = permutation_labels_start + permutation_size;

    for (var i = permutation_labels_start; i < permutation_labels_end; i = i + 1u) {
        permutation_labels[i] = 0u;
    }

    // collect results
    for (var i = 0u; i < options.examples_count; i++) {
        let example_label = labels[i];
        let permutation = permutations[offset + i];

        permutation_labels[permutation_labels_start + permutation * options.outputs_count + example_label]++;
    }

    var result = 0u;
    for (var p = 0u; p < options.permutations_count; p++) {
        let p_start = permutation_labels_start + p * options.outputs_count;
        let p_end = p_start + options.outputs_count;

        var max = 0u;
        for (var o = p_start; o < p_end; o++) {
            if (permutation_labels[o] > max) {
                max = permutation_labels[o];
            }
        }

        result += max;
    }

    output[global_id.x] = f32(result);
}
