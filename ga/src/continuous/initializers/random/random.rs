use rand::distr::{Distribution, Uniform};
use sgrmath_core::{Compiled, CompiledIteration, Iteration};

use crate::IterationParams;

#[derive(Clone, Debug)]
pub struct Random;

pub struct RandomIteration {
    params: IterationParams<f32>,
}

impl Random {
    pub fn new() -> CompiledIteration<Self, RandomIteration, IterationParams<f32>> {
        CompiledIteration::new(Self)
    }
}

impl Compiled<IterationParams<f32>, RandomIteration> for Random {
    fn compile(&self, params: &IterationParams<f32>) -> RandomIteration {
        RandomIteration::new(params)
    }
}

impl RandomIteration {
    pub fn new(params: &IterationParams<f32>) -> Self {
        Self { params: params.clone() }
    }

    pub fn execute(&self, params: &IterationParams<f32>) {
        let (wgpu, min, max, vector_length) = {
            let context = params.context.borrow();

            (context.wgpu.clone(), context.options.min_value, context.options.max_value, context.options.vector_length)
        };
        let mut context = params.context.borrow_mut();
        let data = params.data.borrow();
        
        let uniform = Uniform::new(min, max).unwrap();
        data.population.update_buffer_range(
            &wgpu, 
            &uniform
                .sample_iter(&mut context.rng)
                .take(params.solutions_count * vector_length)
                .collect::<Vec<f32>>(),
            params.solutions_offset * vector_length,
        );
    }
}

impl Iteration<IterationParams<f32>> for RandomIteration {
    fn bind(&mut self, params: &IterationParams<f32>) {
        self.params = params.clone();
    }

    fn evaluate(&mut self) {
        self.execute(&self.params);
    }

    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        self.execute(&self.params);
        vec![]
    }

    fn evaluate_with_params(&mut self, params: &IterationParams<f32>) {
        self.execute(params);
    }

    fn evaluate_with_params_async(&mut self, params: &IterationParams<f32>) -> Vec<wgpu::CommandBuffer> {
        self.execute(params);
        vec![]
    }
}
