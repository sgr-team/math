use std::{cell::RefCell, collections::HashSet, rc::Rc};

use sgrmath_core::{OptimizationDirection, WgpuContext};
use crate::{Context, Data, Individual, IterationParams, Options};

use super::DefaultIteration;

#[test]
fn select() {
    assert_eq!(
        execute(
            OptimizationDirection::Minimize,
            0, 
            5, 
            vec![ 0.0, 1.0, 2.0, 3.0, 4.0 ],
            vec![ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 ]
        ),
        HashSet::from_iter(vec![ 0, 1, 2, 5, 6 ].into_iter())
    );
    assert_eq!(
        execute(
            OptimizationDirection::Maximize,
            0, 
            5, 
            vec![ 0.0, 1.0, 2.0, 3.0, 4.0 ],
            vec![ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 ]
        ),
        HashSet::from_iter(vec![ 8, 9, 10, 3, 4 ].into_iter())
    );
}

fn execute(
    direction: OptimizationDirection,
    offset: usize, 
    count: usize,
    population_results: Vec<f32>,
    next_results: Vec<f32>
) -> HashSet<usize> {
    let options = options(direction);
    let params = params(&options, offset, count);

    {
        let context = params.context.borrow();
        let mut data = params.data.borrow_mut();
        
        data.results.update_buffer_range::<f32>(&context.wgpu, &next_results, 0);
        data.individuals = population_results
            .into_iter()
            .enumerate()
            .map(|(index, value)| Individual {
                id: index,
                generation: 0,
                parents: vec![],
                result: value,
            })
            .collect();

        (context.wgpu.clone(), data.population.clone())
    };

    DefaultIteration::new(&params).execute(&params);

    let result = params.data
        .borrow()
        .individuals
        .iter()
        .map(|individual| individual.id)
        .collect();

    result
}

pub fn options(optimization_direction: OptimizationDirection) -> Options {
    Options {
        optimization_direction,
        population_size: 5,
        generation_size: 6,
        parents_count: 2,
        vector_length: 5,
        min_value: -1.0,
        max_value: 1.0,
    }
}

pub fn params(options: &Options, offset: usize, count: usize) -> IterationParams<f32> {
    let wgpu = WgpuContext::new();
    
    IterationParams {
        context: Rc::new(RefCell::new({
            let mut context = Context::new(&wgpu, &options);
            context.generation_index = 1;
            context.next_id = offset + count;

            context
        })),
        data: Rc::new(RefCell::new(Data::new(&wgpu, &options))),
        solutions_count: count,
        solutions_offset: offset,
    }
}
