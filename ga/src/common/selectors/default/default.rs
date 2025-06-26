use std::collections::HashSet;

use sgrmath_core::{Compiled, Iteration};
use bytemuck::Pod;

use crate::IterationParams;

#[derive(Clone)]
pub struct Default;

pub struct DefaultIteration<T> 
where
    T: Pod
{
    params: IterationParams<T>,
}

impl Default {
    pub fn new() -> Self {
        Self { }
    }
}

impl<T> Compiled<IterationParams<T>, DefaultIteration<T>> for Default
where
    T: Pod
{
    fn compile(&self, params: &IterationParams<T>) -> DefaultIteration<T> {
        DefaultIteration::new(params)
    }
}

impl<T> DefaultIteration<T> 
where
    T: Pod
{
    pub fn new(params: &IterationParams<T>) -> Self {
        Self { params: params.clone() }
    }

    pub fn execute(&self, params: &IterationParams<T>) {
        let mut context = params.context.borrow_mut();
        let mut data = params.data.borrow_mut();

        let next = data.read_generation(&mut context);
        let (population_size, generation_size) = (context.options.population_size as usize, context.options.generation_size as usize);

        let mut order = (0..(population_size + generation_size)).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            let a_value = if a < population_size { 
                data.individuals[a].result
            } else { 
                next[a - population_size].result
            };
            let b_value = if b < population_size { 
                data.individuals[b].result
            } else { 
                next[b - context.options.population_size].result
            };

            context.options.optimization_direction.compare(&a_value, &b_value)
        });

        let mut deleted = (0..context.options.population_size).collect::<HashSet<_>>();
        let mut new = vec![];
        for index in order.iter().take(context.options.population_size) {
            match index < &population_size {
                true => { deleted.remove(index); },
                false => { new.push(index - population_size); }
            }
        }

        data.update_population(
            &mut context, 
            deleted
                .into_iter()
                .zip(new.into_iter())
                .map(|(index, new_index)| (index, next[new_index].clone()))
                .collect::<Vec<_>>()
        );
    }
}

impl<T> Iteration<IterationParams<T>> for DefaultIteration<T> 
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
