use rand::distr::{Distribution, Uniform};
use sgrmath_core::{Compiled, Iteration};
use bytemuck::Pod;

use crate::IterationParams;

#[derive(Clone)]
pub struct Random;

pub struct RandomIteration<T> 
where
    T: Pod
{
    params: IterationParams<T>,
}

impl Random {
    pub fn new() -> Self {
        Self { }
    }
}

impl<T> Compiled<IterationParams<T>, RandomIteration<T>> for Random
where
    T: Pod
{
    fn compile(&self, params: &IterationParams<T>) -> RandomIteration<T> {
        RandomIteration::new(params)
    }
}

impl<T> RandomIteration<T> 
where
    T: Pod
{
    pub fn new(params: &IterationParams<T>) -> Self {
        Self { params: params.clone() }
    }

    pub fn execute(&self, params: &IterationParams<T>) {
        let (wgpu, min, max, parents_count) = {
            let context = params.context.borrow();

            (context.wgpu.clone(), 0u32, context.options.population_size as u32 - 1, context.options.parents_count)
        };
        let mut context = params.context.borrow_mut();
        let data = params.data.borrow();
        
        let uniform = Uniform::new(min, max).unwrap();
        data.parents.update_buffer_range(
            &wgpu, 
            &uniform
                .sample_iter(&mut context.rng)
                .take(params.solutions_count * parents_count)
                .collect::<Vec<u32>>(),
            params.solutions_offset * parents_count,
        );
    }
}

impl<T> Iteration<IterationParams<T>> for RandomIteration<T> 
where
    T: Pod
{
    fn bind(&mut self, params: &IterationParams<T>) {
        self.params = params.clone();
    }

    fn evaluate(&mut self) {
        self.execute(&self.params);
    }

    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        self.execute(&self.params);
        vec![]
    }

    fn evaluate_with_params(&mut self, params: &IterationParams<T>) {
        self.execute(params);
    }

    fn evaluate_with_params_async(&mut self, params: &IterationParams<T>) -> Vec<wgpu::CommandBuffer> {
        self.execute(params);
        vec![]
    }
}
