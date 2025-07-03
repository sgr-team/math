use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sgrmath_core::{ProblemParams, ReadbackBuffer, StorageBuffer};

use crate::{shaders::ShaderOptions, PNP};

#[derive(Serialize, Deserialize)]
pub struct Solution {
    /// Number of vectors in the solution
    pub vectors_count: usize,
    /// Number of outputs in the solution
    pub outputs_count: usize,
    /// Vectors in the solution
    pub vectors: Vec<f32>,
    /// Permutations in the solution (vectors_count! * outputs_count)
    pub permutations: HashMap<usize, u32>,
    /// Training result
    pub result: usize
}

impl Solution {
    /// Create a new solution with the given number of vectors and outputs
    pub fn new(
        vectors_count: usize, 
        outputs_count: usize, 
        vectors: Vec<f32>, 
        permutations: Vec<u32>, 
        result: usize
    ) -> Self {
        Self { 
            vectors_count, 
            outputs_count, 
            vectors, 
            permutations: Self::permutations_from_vec(permutations), 
            result 
        }
    }

    /// Init
    pub fn init(pnp: &PNP, vectors: Vec<f32>) -> Self {
        let permutations_size = pnp.permutations_count * pnp.outputs_count;
        let params = ProblemParams {
            context: pnp.wgpu.clone(),
            solutions_offset: 0,
            solutions_count: 1,
            vector_length: pnp.vector_length,
            solutions: StorageBuffer::init(&pnp.wgpu, vectors.as_slice()),
            results: StorageBuffer::new::<f32, _>(&pnp.wgpu, 1),
        };
        let (options, multiply, permutations, permutation_labels) = pnp.create_buffers(&pnp.wgpu, &ShaderOptions::new(pnp, 1));
        let reader = ReadbackBuffer::new::<f32, _>(&pnp.wgpu, permutations_size);
        pnp.evaluate_with_buffers(&params, (&options, &multiply, &permutations, &permutation_labels));

        Self {
            vectors_count: pnp.vectors_count,
            outputs_count: pnp.outputs_count,
            vectors,
            permutations: Self::permutations_from_vec(reader.read(&pnp.wgpu, &permutation_labels, 0, permutations_size)),
            result: reader.read::<f32>(&pnp.wgpu, &params.results, 0, 1)[0] as usize,
        }
    }

    // Init with permutations
    // Use this method to test trained solution
    pub fn init_with_permutations(pnp: &PNP, vectors: Vec<f32>, permutations: Vec<u32>) -> Self {
        let (options, multiply, permutations_buffer, permutation_labels) = pnp.create_buffers(
            &pnp.wgpu, 
            &ShaderOptions::new(pnp, 1)
        );
        let params = ProblemParams {
            context: pnp.wgpu.clone(),
            solutions_offset: 0,
            solutions_count: 1,
            vector_length: pnp.vector_length,
            solutions: StorageBuffer::init(&pnp.wgpu, vectors.as_slice()),
            results: StorageBuffer::new::<f32, _>(&pnp.wgpu, 1),
        };
        let mut permutation_outputs = vec![0u32; pnp.permutations_count];
        for i in 0..pnp.permutations_count {
            let start = i * pnp.outputs_count;
            let mut max = (0, permutations[start]);
            for j in 1..pnp.outputs_count {
                if max.1 < permutations[start + j] {
                    max = (j, permutations[start + j]);
                }
            }

            permutation_outputs[i] = max.0 as u32;
        }
        permutation_labels.update_buffer_range(&pnp.wgpu, &permutation_outputs, 0);

        pnp.shaders.multiply.execute_with_params(
            &params.context, 
            (pnp.examples_count, pnp.vectors_count * params.solutions_count),
            &[ &options, &pnp.examples, &params.solutions, &multiply ]
        );
        pnp.shaders.permutations.execute_with_params(
            &params.context, 
            (params.solutions_count, pnp.examples_count),
            &[ &options, &multiply, &permutations_buffer ]
        );
        pnp.shaders.test.execute_with_params(
            &params.context, 
            params.solutions_count,
            &[ &options, &pnp.labels, &permutations_buffer, &permutation_labels, &params.results ]
        );

        Self {
            vectors_count: pnp.vectors_count,
            outputs_count: pnp.outputs_count,
            vectors,
            permutations: Self::permutations_from_vec(permutations),
            result: ReadbackBuffer::new::<f32, _>(&pnp.wgpu, 1)
                .read::<f32>(&pnp.wgpu, &params.results, 0, 1)[0] as usize,
        }
    }

    pub fn permutations_from_vec(permutations: Vec<u32>) -> HashMap<usize, u32> {
        let mut result = HashMap::new();
        for i in 0..permutations.len() {
            if permutations[i] == 0 {
                continue;
            }

            result.insert(i, permutations[i]);
        }

        result
    }

    pub fn permutations_to_vec(pnp: &PNP, permutations: &HashMap<usize, u32>) -> Vec<u32> {
        let mut result = vec![0; pnp.permutations_count * pnp.outputs_count];
        for (i, value) in permutations.iter() {
            result[*i] = *value;
        }

        result
    }
}
