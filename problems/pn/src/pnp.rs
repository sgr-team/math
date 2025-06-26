use std::collections::HashSet;

use sgrmath_core::{Iteration, ProblemParams, StorageBuffer, ValueBuffer, WgpuContext};
use wgpu::CommandBuffer;

use crate::shaders::{ShaderOptions, Shaders};

#[derive(Debug, Clone)]
pub struct PNP {
    /// Wgpu context
    pub wgpu: WgpuContext,
    /// Number of examples
    pub examples_count: usize,
    /// Permutations count
    pub permutations_count: usize,
    /// Number of vectors
    pub vectors_count: usize,
    /// Length of each vector
    pub vector_length: usize,
    /// Number of outputs
    pub outputs_count: usize,
    /// Examples vectors
    pub examples: StorageBuffer,
    /// Examples labels
    pub labels: StorageBuffer,
    /// Shaders
    pub shaders: Shaders,
    /// Binded params
    pub params: Option<ProblemParams>
}

impl PNP {
    /// Create a new PN instance
    pub fn new(
        wgpu: &WgpuContext, 
        examples_count: usize,
        vectors_count: usize, 
        vector_length: usize,
        outputs_count: usize
    ) -> Self {
        Self {
            wgpu: wgpu.clone(),
            examples_count,
            permutations_count: Self::permutations_count(vectors_count),
            vectors_count,
            vector_length,
            outputs_count,
            examples: StorageBuffer::new::<f32, _>(wgpu, examples_count * vector_length),
            labels: StorageBuffer::new::<u32, _>(wgpu, examples_count),
            shaders: Shaders::new(wgpu),
            params: None
        }
    }

    /// Initialize a new PN instance
    /// from existing vectors and labels
    pub fn init(
        wgpu: &WgpuContext,
        examples_count: usize,
        vectors_count: usize, 
        vector_length: usize,
        outputs_count: usize,
        vectors: Vec<f32>, 
        labels: Vec<u32>
    ) -> Self {
        let buffer_examples = StorageBuffer::init(wgpu, &vectors);
        let buffer_labels = StorageBuffer::init(wgpu, &labels);

        Self { 
            wgpu: wgpu.clone(), 
            examples_count,
            permutations_count: Self::permutations_count(vectors_count),
            vectors_count, 
            vector_length, 
            outputs_count,
            examples: buffer_examples, 
            labels: buffer_labels,
            shaders: Shaders::new(wgpu),
            params: None
        }
    }

    /// Initialize a new PN instance
    /// from a CSV String
    pub fn from_csv(
        wgpu: &WgpuContext, 
        vectors_count: usize, 
        vector_length: usize,
        data: String, 
        delimiter: char,
    ) -> Self {
        let lines = data
            .split("\n")
            .enumerate()
            .filter(|(_, line)| !line.is_empty());

        let mut vectors = Vec::new();
        let mut labels = Vec::new();

        let mut examples_count = 0;
        let mut unique_labels = HashSet::new();
        'line_loop: for (_, line) in lines {
            let mut example_length = 0;
            for (index, value) in line.split(delimiter).enumerate() {
                if index == 0 {
                    labels.push(
                        match value.trim().parse::<u32>() {
                            Ok(label) => {
                                unique_labels.insert(label);
                                label
                            },
                            Err(_) => {
                                continue 'line_loop; // skip the line (titles)
                            },
                        }
                    );
                    examples_count += 1;
                    continue;
                }

                example_length += 1;
                vectors.push(
                    match value.trim().parse::<u32>() {
                        Ok(value) => value as f32,
                        Err(_) => panic!("Error parsing value: {}", value),
                    }
                );
            }

            assert_eq!(
                example_length, 
                vector_length, 
                "Vector length is not consistent {example_length} != {vector_length}"
            );
        }

        assert_eq!(
            examples_count * vector_length, 
            vectors.len(), 
            "Examples count is not consistent {examples_count} * {vector_length} != {}", 
            vectors.len()
        );

        Self::init(
            wgpu, 
            examples_count, 
            vectors_count, 
            vector_length, 
            unique_labels.len(), 
            vectors, 
            labels
        )
    }

    fn permutations_count(count: usize) -> usize {
        let mut result = 1;
        for i in 2..=(count + 1) {
            result *= i;
        }

        result
    }
    
    pub fn create_buffers(
        &self, 
        ctx: &WgpuContext, 
        options: &ShaderOptions
    ) -> (ValueBuffer, StorageBuffer, StorageBuffer, StorageBuffer) {
        let buffer_options = ValueBuffer::init::<ShaderOptions>(ctx, options);
        let buffer_multiply = StorageBuffer::new::<f32, _>(
            ctx, 
            (
                (options.vectors_count * options.solutions_count) as usize,
                options.examples_count as usize,
            )
        );
        let buffer_permutations = StorageBuffer::new::<u32, _>(
            ctx, 
            (
                options.solutions_count as usize,
                options.examples_count as usize,
            )
        );
        let buffer_permutation_labels = StorageBuffer::new::<u32, _>(
            ctx, 
            (
                options.solutions_count as usize,
                options.permutations_count as usize,
                options.outputs_count as usize
            )
        );

        (buffer_options, buffer_multiply, buffer_permutations, buffer_permutation_labels)
    }

    pub fn evaluate_with_buffers(
        &self, 
        params: &ProblemParams, 
        (options, multiply, permutations, permutation_labels): (&ValueBuffer, &StorageBuffer, &StorageBuffer, &StorageBuffer)
    ) {
        self.shaders.multiply.execute_with_params(
            &params.context, 
            (self.examples_count, self.vectors_count * params.solutions_count),
            &[ &options, &self.examples, &params.solutions, &multiply ]
        );
        self.shaders.permutations.execute_with_params(
            &params.context, 
            (params.solutions_count, self.examples_count),
            &[ &options, &multiply, &permutations ]
        );
        self.shaders.results.execute_with_params(
            &params.context, 
            params.solutions_count,
            &[ &options, &self.labels, &permutations, &permutation_labels, &params.results ]
        );
    }
}

impl Iteration<ProblemParams> for PNP {
    fn bind(&mut self, params: &ProblemParams) {
        self.params = Some(params.clone());
        let shader_options = ShaderOptions::new(&self, params.solutions_count);
        let (options, multiply, permutations, permutation_labels) = self.create_buffers(&self.wgpu, &shader_options);

        self.shaders.multiply.bind(
            &params.context, 
            &[ &options, &self.examples, &params.solutions, &multiply ]
        );
        self.shaders.permutations.bind(
            &params.context, 
            &[ &options, &multiply, &permutations ]
        );
        self.shaders.results.bind(
            &params.context, 
            &[ &options, &self.labels, &permutations, &permutation_labels, &params.results ]
        );
    }

    fn evaluate(&mut self) {
        let params = self.params.as_ref().expect("evaluate called before bind");

        self.shaders.multiply.execute(
            &params.context, 
            (self.examples_count, self.vectors_count * params.solutions_count)
        );
        self.shaders.permutations.execute(
            &params.context, 
            (params.solutions_count, self.examples_count)
        );
        self.shaders.results.execute(
            &params.context, 
            params.solutions_count
        );
    }

    fn evaluate_async(&mut self) -> Vec<CommandBuffer> {
        let params = self.params.as_ref().expect("evaluate called before bind");

        self.shaders.multiply.execute(
            &params.context, 
            (self.examples_count, self.vectors_count * params.solutions_count)
        );
        self.shaders.permutations.execute(
            &params.context, 
            (params.solutions_count, self.examples_count)
        );
        
        vec![
            self.shaders.results.execute_async(
                &params.context, 
                params.solutions_count
            )
        ]
    }

    fn evaluate_with_params(&mut self, params: &ProblemParams) {
        let shader_options = ShaderOptions::new(&self, params.solutions_count);
        let (options, multiply, permutations, permutation_labels) = self.create_buffers(&self.wgpu, &shader_options);

        self.evaluate_with_buffers(params, (&options, &multiply, &permutations, &permutation_labels));
    }

    fn evaluate_with_params_async(&mut self, params: &ProblemParams) -> Vec<CommandBuffer> {
        let shader_options = ShaderOptions::new(&self, params.solutions_count);
        let (options, multiply, permutations, permutation_labels) = self.create_buffers(&self.wgpu, &shader_options);

        self.shaders.multiply.execute_with_params(
            &params.context, 
            (self.examples_count, self.vectors_count * params.solutions_count),
            &[ &options, &self.examples, &params.solutions, &multiply ]
        );
        self.shaders.permutations.execute_with_params(
            &params.context, 
            (params.solutions_count, self.examples_count),
            &[ &options, &multiply, &permutations ]
        );
        
        vec![
            self.shaders.results.execute_with_params_async(
                &params.context, 
                params.solutions_count,
                &[ &options, &self.labels, &permutations, &permutation_labels, &params.results ]
            )
        ]
    }
}
