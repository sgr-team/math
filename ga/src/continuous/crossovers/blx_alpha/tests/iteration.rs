use core::f32;
use std::{cell::RefCell, rc::Rc};

use sgrmath_core::{Iteration, OptimizationDirection, ReadbackBuffer, WgpuContext};

use crate::{Context, Data, IterationParams, Options};
use super::super::BLXAlphaIteration;

#[test]
fn evaluate_with_params() {
    let result = execute_iteration(
        0.5,
        1,
        3,
        |iteration, params| iteration.evaluate_with_params(&params)
    );

    for i in 0..result.len() {
        if i < 5 || i >= 20 {
            assert!(result[i] == 42.5, "invalid value at index {} ({})", i, result[i]);
            continue;
        }

        assert!(result[i] >= -100.0 && result[i] <= 100.0, "invalid value at index {} ({})", i, result[i]);
        assert!(result[i] != 42.5, "default value at index {} ({})", i, result[i]);
    }
}

fn execute_iteration<F>(
    k: f32,
    offset: usize,
    count: usize,
    f: F
) -> Vec<f32> 
where 
    F: FnOnce(&mut BLXAlphaIteration, &IterationParams<f32>) -> ()
{
    let options = options();
    let params = params(&options, offset, count);
    {
        let context = params.context.borrow();
        let data = params.data.borrow();

        data.population.update_buffer_range::<f32>(
            &context.wgpu, 
            &vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 2.0, 3.0, 4.0, 5.0,
                2.0, 4.0, 8.0, 16.0, 32.0,
            ],
            0
        );
        data.parents.update_buffer_range::<u32>(
            &context.wgpu, 
            &vec![
                0, 1,
                0, 2,
                1, 2,
            ],
            0
        );
        data.next.update_buffer_range::<f32>(
            &context.wgpu, 
            &vec![
                42.5, 42.5, 42.5, 42.5, 42.5,
                42.5, 42.5, 42.5, 42.5, 42.5,
                42.5, 42.5, 42.5, 42.5, 42.5,
                42.5, 42.5, 42.5, 42.5, 42.5,
                42.5, 42.5, 42.5, 42.5, 42.5,
            ],
            0
        );
    }
    
    f(&mut BLXAlphaIteration::new(k, &params), &params);
    
    let context = params.context.borrow();
    let data = params.data.borrow();
    
    return ReadbackBuffer::new::<f32, _>(
        &context.wgpu, 
        options.generation_size * options.vector_length
    ).read(
        &context.wgpu, 
        &data.next, 
        0, 
        options.generation_size * options.vector_length
    );
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