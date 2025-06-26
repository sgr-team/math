use std::{cell::RefCell, rc::Rc};

use sgrmath_core::{OptimizationDirection, ReadbackBuffer, WgpuContext};
use crate::{continuous::initializers::RandomIteration, Context, Data, IterationParams, Options};

#[test]
fn initialize() {
    let data = execute(0, 50);

    for i in 0..data.len() {
        assert!(data[i] >= -1.0 && data[i] <= 1.0, "Value at index {} is out of range", i);
        assert!(data[i] != 0.0, "Value at index {} is 0", i);
    }
}

#[test]
fn offset() {
    let data = execute(20, 30);

    for i in 0..data.len() {
        if i < 200 || i >= 500 {
            assert_eq!(data[i], 0.0, "Value at index {} is not 0", i);
            continue;
        }

        assert!(data[i] != 0.0, "Value at index {} is 0", i);
    }
}

fn execute(offset: usize, count: usize) -> Vec<f32> {
    let options = options();
    let params = params(&options, offset, count);

    let (wgpu, result_buffer) = { 
        let (context, data) = (params.context.borrow(), params.data.borrow());
        data.population.update_buffer_range::<f32>(
            &context.wgpu, 
            &vec![0.0; options.population_size * options.vector_length], 
            0
        );

        (context.wgpu.clone(), data.population.clone())
    };

    RandomIteration::new(&params).execute(&params);

    let reader = ReadbackBuffer::new::<f32, _>(&wgpu, (options.population_size, options.vector_length));
    reader.read(&wgpu, &result_buffer, 0, options.population_size * options.vector_length)
}

pub fn options() -> Options {
    Options {
        optimization_direction: OptimizationDirection::Minimize,
        population_size: 50,
        generation_size: 100,
        parents_count: 2,
        vector_length: 10,
        min_value: -1.0,
        max_value: 1.0,
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