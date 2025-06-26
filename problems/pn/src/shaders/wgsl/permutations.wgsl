const FACTORIALS: array<u32, 8> = array<u32, 8>(1u, 2u, 6u, 24u, 120u, 720u, 5040u, 40320u);

@group(0) @binding(0) var<uniform> options: PNPOptions;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let solution_start = global_id.x * options.vectors_count * options.examples_count + global_id.y;
    let solution_end = solution_start + options.vectors_count * options.examples_count;
    
    // Create array for indices
    var indices: array<u32, 8>;
    // Initialize indices
    for (var i = 0u; i <= options.vectors_count; i = i + 1u) {
        indices[i] = solution_start + i * options.examples_count;
    }
    
    // ToDo rewrite sorting to merge
    while (true) {
        var changed = false;
        for (var i = 0u; i < options.vectors_count; i = i + 1u) {
            var a = 0.0; if (indices[i] < solution_end) { a = input[indices[i]]; }
            var b = 0.0; if (indices[i + 1] < solution_end) { b = input[indices[i + 1]]; }

            if (b > a) {
                let temp = indices[i];
                indices[i] = indices[i + 1];
                indices[i + 1] = temp;
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
    }

    // Lehmer code implementation
    var result = 0u;
    for (var i = 0u; i <= options.vectors_count; i = i + 1u) {
        var count = 0u;
        let current = indices[i];
        for (var j = i + 1u; j <= options.vectors_count; j = j + 1u) {
            if (indices[j] < current) {
                count = count + 1u;
            }
        }
        
        // Get factorial directly from array
        result = result + count * FACTORIALS[options.vectors_count - i - 1];
    }

    for (var i = 0u; i <= options.vectors_count; i = i + 1u) {
        indices[i] = (indices[i] - solution_start) / options.examples_count;
    }
    output[global_id.x * options.examples_count + global_id.y] = result;
}
