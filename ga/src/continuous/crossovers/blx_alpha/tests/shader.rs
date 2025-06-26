use core::f32;
use std::{cell::RefCell, rc::Rc};

use sgrmath_core::{OptimizationDirection, ReadbackBuffer, Shader, StorageBuffer, ValueBuffer, WgpuContext};

use crate::{Context, Data, IterationParams, Options};
use super::super::ShaderOptions;

#[test]
fn execute() {
    assert_eq!(
        execute_shader(
            1,
            3,
            // population
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 2.0, 3.0, 4.0, 5.0,
                2.0, 4.0, 8.0, 16.0, 32.0,
            ],
            // parents
            vec![
                0, 1,
                0, 2,
                1, 2,
            ],
            // random
            vec![
                0.5, 1.0, -0.75, 0.25, 1.0,
                0.0, 0.25, -0.25, 100.0, -100.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        vec![
            42.2, 42.2, 42.2, 42.2, 42.2,
            1.0, 3.0, -0.75, 3.0, 7.5,
            1.0, 3.0, 2.0, 100.0, -100.0,
            1.5, 3.0, 5.5, 10.0, 18.5,
            42.2, 42.2, 42.2, 42.2, 42.2,
        ]
    )
}

fn execute_shader(
    offset: usize,
    count: usize,
    population: Vec<f32>,
    parents: Vec<u32>,
    random: Vec<f32>
) -> Vec<f32> {
    let options = options();
    let params = params(&options, offset, count);
    let context = params.context.borrow();
    let wgpu = context.wgpu.clone();
    let generation_size = options.generation_size * options.vector_length;

    let shader = Shader::new(&wgpu, "blx_alpha", include_str!("../blx_alpha.wgsl"));

    let buffer_options = ValueBuffer::init(
        &wgpu, 
        &ShaderOptions { 
            generation_offset: (offset * options.vector_length) as u32,
            vector_length: options.vector_length as u32,
            parents_count: options.parents_count as u32,
            min: options.min_value,
            max: options.max_value,
        }
    );
    let buffer_population = StorageBuffer::init::<f32>(&wgpu, &population);
    let buffer_parents = StorageBuffer::init::<u32>(&wgpu, &parents);
    let buffer_random = StorageBuffer::init::<f32>(&wgpu, &random);
    let buffer_generation = StorageBuffer::init::<f32>(&wgpu, &vec![42.2; generation_size]);

    shader.execute_with_params(
        &wgpu, 
        (options.vector_length, count),
        &[ &buffer_options, &buffer_population, &buffer_parents, &buffer_random, &buffer_generation ]
    );

    return ReadbackBuffer::new::<f32, _>(&wgpu, generation_size).read(&wgpu, &buffer_generation, 0, generation_size);
}

fn options() -> Options {
    Options {
        optimization_direction: OptimizationDirection::Minimize,
        population_size: 3,
        generation_size: 5,
        parents_count: 2,
        vector_length: 5,
        min_value: -100.0,
        max_value: 100.0,
    }
}

fn params(options: &Options, offset: usize, count: usize) -> IterationParams<f32> {
    let wgpu = WgpuContext::new();

    IterationParams {
        context: Rc::new(RefCell::new(Context::new(&wgpu, &options))),
        data: Rc::new(RefCell::new(Data::new(&wgpu, &options))),
        solutions_count: count,
        solutions_offset: offset,
    }
}