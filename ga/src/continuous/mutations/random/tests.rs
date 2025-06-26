use std::{cell::RefCell, rc::Rc};

use sgrmath_core::{OptimizationDirection, ReadbackBuffer, WgpuContext};
use crate::{Context, Data, IterationParams, Options};

use super::RandomIteration;

#[test]
fn initialize() {
    let data = execute(0, 50);

    let mut count = 0;
    for i in 0..data.len() {
        count += if data[i] == 42.5 { 1 } else { 0 };
    }

    assert!(count > 100, "Mutations count is too low ({} mutations)", count);
}

#[test]
fn offset() {
    let data = execute(20, 30);

    for i in 0..data.len() {
        if i < 2000 || i >= 5000 {
            assert_eq!(data[i], 42.5, "Value at index {} is not default ({} != 42.5)", i, data[i]);
            continue;
        }
    }
}

fn execute(offset: usize, count: usize) -> Vec<f32> {
    let options = options();
    let params = params(&options, offset, count);

    let (wgpu, next_buffer) = { 
        let (context, data) = (params.context.borrow(), params.data.borrow());
        data.next.update_buffer_range::<f32>(
            &context.wgpu, 
            &vec![42.5; options.population_size * options.vector_length], 
            0
        );

        (context.wgpu.clone(), data.next.clone())
    };

    RandomIteration::new(0.05, &params).execute(&params);

    let reader = ReadbackBuffer::new::<f32, _>(&wgpu, (options.population_size, options.vector_length));
    reader.read(&wgpu, &next_buffer, 0, options.population_size * options.vector_length)
}

pub fn options() -> Options {
    Options {
        optimization_direction: OptimizationDirection::Minimize,
        population_size: 50,
        generation_size: 100,
        parents_count: 2,
        vector_length: 100,
        min_value: -0.5,
        max_value: 0.5,
    }
}

pub fn params(options: &Options, offset: usize, count: usize) -> IterationParams<f32> {
    let wgpu = WgpuContext::new();
    
    IterationParams {
        context: Rc::new(RefCell::new(Context::new(&wgpu, &options))),
        data: Rc::new(RefCell::new(Data::new(&wgpu, &options))),
        solutions_count: count,
        solutions_offset: offset,
    }
}