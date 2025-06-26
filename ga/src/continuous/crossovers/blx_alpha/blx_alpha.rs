use std::ops::DerefMut;

use rand_distr::{Distribution, Uniform};
use sgrmath_core::{Compiled, CompiledIteration, Iteration, Shader, Size, StorageBuffer, ValueBuffer};

use crate::{continuous::crossovers::blx_alpha::ShaderOptions, Context, IterationParams};

pub struct BLXAlpha {
    pub k: f32
}

pub struct BLXAlphaIteration {
    k: f32,
    shader: Shader,
    bind: Option<IterationParams<f32>>,
    buffer_options: ValueBuffer,
    buffer_random: StorageBuffer,
}

impl BLXAlpha {
    pub fn new(k: f32) -> CompiledIteration<Self, BLXAlphaIteration, IterationParams<f32>> {
        CompiledIteration::new(Self { k })
    }
}

impl Compiled<IterationParams<f32>, BLXAlphaIteration> for BLXAlpha {
    fn compile(&self, params: &IterationParams<f32>) -> BLXAlphaIteration {
        BLXAlphaIteration::new(self.k, params)
    }
}

impl Iteration<IterationParams<f32>> for BLXAlphaIteration {
    fn bind(&mut self, params: &IterationParams<f32>) {
        let context = params.context.borrow();
        let data = params.data.borrow();
        self.bind = Some(params.clone());

        self.shader.bind(
            &context.wgpu, 
            &[
                &self.buffer_options,
                &data.population,
                &data.parents,
                &self.buffer_random,
                &data.next,
            ]
        );
    }

    fn evaluate(&mut self) {
        let params = self.bind.as_ref().expect("evaluate called without bind");
        let size = self.size(params);
        let mut context = params.context.borrow_mut();
        self.fill_random(&size, context.deref_mut());

        self.shader.execute(&context.wgpu, size);
    }

    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        let params = self.bind.as_ref().expect("evaluate called without bind");
        let mut context = params.context.borrow_mut();
        let size = self.size(params);
        self.fill_random(&size, context.deref_mut());

        vec![ self.shader.execute_async(&context.wgpu, size) ]
    }

    fn evaluate_with_params(&mut self, params: &IterationParams<f32>) {
        let size = self.size(params);
        let mut context = params.context.borrow_mut();
        let data = params.data.borrow();
        self.fill_random(&size, context.deref_mut());

        self.shader.execute_with_params(
            &context.wgpu, 
            size,
            &[
                &self.buffer_options,
                &data.population,
                &data.parents,
                &self.buffer_random,
                &data.next,
            ]
        );
    }

    fn evaluate_with_params_async(&mut self, params: &IterationParams<f32>) -> Vec<wgpu::CommandBuffer> {
        let size = self.size(params);
        let mut context = params.context.borrow_mut();
        let data = params.data.borrow();
        self.fill_random(&size, context.deref_mut());

        vec![ 
            self.shader.execute_with_params_async(
                &context.wgpu, 
                size,
                &[
                    &self.buffer_options,
                    &data.population,
                    &data.parents,
                    &self.buffer_random,
                    &data.next,
                ]
            )
        ]
    }
}

impl BLXAlphaIteration {
    pub fn new(k: f32, params: &IterationParams<f32>) -> Self {
        let context = params.context.borrow();
        Self { 
            k, 
            shader: Shader::new(&context.wgpu, "blx_alpha", include_str!("blx_alpha.wgsl")),
            bind: None,
            buffer_options: ValueBuffer::init(
                &context.wgpu, 
                &ShaderOptions {
                    generation_offset: (params.solutions_offset * context.options.vector_length) as u32,
                    vector_length: context.options.vector_length as u32,
                    parents_count: context.options.parents_count as u32,
                    min: context.options.min_value,
                    max: context.options.max_value,
                }
            ),
            buffer_random: StorageBuffer::new::<f32, _>(&context.wgpu, (params.solutions_count, context.options.vector_length)),
        }
    }

    fn fill_random(&self, size: &Size, context: &mut Context) {
        self.buffer_random.update_buffer_range::<f32>(
            &context.wgpu, 
            &Uniform::new(-self.k / 2.0, self.k / 2.0)
                .unwrap()
                .sample_iter(&mut context.rng)
                .take(size.len())
                .collect::<Vec<_>>(),
            0
        );
    }

    fn size(&self, params: &IterationParams<f32>) -> Size {
        let context = params.context.borrow();
        (context.options.vector_length, params.solutions_count).into()
    }
}
