use rand::distr::{Distribution, Uniform};
use rand_distr::Binomial;
use sgrmath_core::{Compiled, CompiledIteration, Iteration, WgpuContext};

use crate::IterationParams;

#[derive(Clone, Debug)]
pub struct Random {
    pub probability: f32,
}

pub struct RandomIteration {
    probability: f32,
    params: IterationParams<f32>,
}

impl Random {
    pub fn new(probability: f32) -> CompiledIteration<Self, RandomIteration, IterationParams<f32>> {
        CompiledIteration::new(Self { probability })
    }
}

impl Compiled<IterationParams<f32>, RandomIteration> for Random {
    fn compile(&self, params: &IterationParams<f32>) -> RandomIteration {
        RandomIteration::new(self.probability, params)
    }
}

impl RandomIteration {
    pub fn new(probability: f32, params: &IterationParams<f32>) -> Self {
        Self { probability, params: params.clone() }
    }

    pub fn execute(&self, params: &IterationParams<f32>) {
        let context = self.execute_async(params);
        context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    }

    pub fn execute_async(&self, params: &IterationParams<f32>) -> WgpuContext {
        let mut context = params.context.borrow_mut();
        let wgpu = context.wgpu.clone();
        let data = params.data.borrow();
        let binomial = Binomial::new(
            (context.options.vector_length * context.options.generation_size) as u64, 
            self.probability as f64
        ).unwrap();
        
        let max_index = params.solutions_count * context.options.vector_length;
        let mutations_count = binomial.sample(&mut context.rng) as usize;
        let uniform = Uniform::new(context.options.min_value, context.options.max_value).unwrap();
        let indexes_offset = params.solutions_offset * context.options.vector_length;

        let indexes = rand::seq::index::sample(&mut context.rng, max_index, mutations_count)
            .into_iter()
            .map(|i| i + indexes_offset);
        let values = uniform
            .sample_iter(&mut context.rng)
            .take(mutations_count);

        for (i, v) in indexes.zip(values) {
            data.next.update_buffer_range_async(&wgpu, &[v], i);
        }

        wgpu
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
        self.execute_async(&self.params);
        vec![]
    }

    fn evaluate_with_params(&mut self, params: &IterationParams<f32>) {
        self.execute(params);
    }

    fn evaluate_with_params_async(&mut self, params: &IterationParams<f32>) -> Vec<wgpu::CommandBuffer> {
        self.execute_async(params);
        vec![]
    }
}
