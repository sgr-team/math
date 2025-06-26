use sgrmath_core::{OptimizationDirection, WgpuContext};
use crate::{Context, Data, Individual, Options};

#[test]
fn update_population() {
    let (mut data, mut context) = prepare();

    data.update_population(
        &mut context, 
        vec![
            (0, Individual { id: 51, generation: 0, parents: vec![], result: 0.0 }),
            (5, Individual { id: 52, generation: 0, parents: vec![], result: 0.0 }),
            (12, Individual { id: 149, generation: 0, parents: vec![], result: 0.0 })
        ]
    );

    data.reader.scale::<i32, _>(&context.wgpu, 500);

    assert_eq!(
        data.reader.read::<i32>(&context.wgpu, &data.population, 0, 500),
        (0..500)
            .map(|index| {
                let (mut row, mut k, column) = (index / 10, 100, index % 10);
                
                match row {
                    0 => { k = 1000; row = 1; },
                    5 => { k = 1000; row = 2; },
                    12 => { k = 1000; row = 99; },
                    _ => {}
                }

                return row * k + column;
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn read_generation() {
    let (mut data, mut context) = prepare();

    assert_eq!(
        data.read_generation(&mut context),
        (0..100)
            .map(|index| Individual { 
                id: 50 + index, 
                generation: 0, 
                parents: vec![ 0, index ], 
                result: 3.0 * index as f32 + 2.5 
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn read_individual() {
    let (data, mut context) = prepare();

    assert_eq!(
        data.read_individual(&mut context, 24),
        vec![ 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409 ]
    );
}

fn prepare() -> (Data<i32>, Context) {
    let options = Options {
        optimization_direction: OptimizationDirection::Minimize,
        population_size: 50,
        generation_size: 100,
        parents_count: 2,
        vector_length: 10,
        min_value: 0.0,
        max_value: 1.0,
    };
    let wgpu = WgpuContext::new();
    let mut context = Context::new(&wgpu, &options);
    context.next_id = 50;
    context.is_initialized = true;
    
    let mut data = Data::<i32>::new(&wgpu, &options);
    data.individuals = (0..50).map(|index| Individual { 
        id: index, 
        generation: 0, 
        parents: vec![], 
        result: 0.0 
    }).collect();
    data.population.update_buffer_range::<i32>(
        &wgpu,  
        (0..500)
            .map(|index| {
                let row = index / 10;
                let column = index % 10;

                return row * 100 + column;
            })
            .collect::<Vec<_>>()
            .as_slice(),
        0
    );
    data.next.update_buffer_range::<i32>(
        &wgpu, 
        (0..1000)
            .map(|index| {
                let row = index / 10;
                let column = index % 10;

                return row * 1000 + column;
            })
            .collect::<Vec<_>>()
            .as_slice(),
        0
    );
    data.parents.update_buffer_range::<u32>(
        &wgpu,
        (0..200)
            .map(|index| {
                let row = index / 2;
                let column = index % 2;

                return if column == 0 { 0 } else { row };
            })
            .collect::<Vec<_>>()
            .as_slice(),
        0
    );
    data.results.update_buffer_range::<f32>(
        &wgpu,
        (0..100)
            .map(|index| 3.0 * index as f32 + 2.5)
            .collect::<Vec<_>>()
            .as_slice(),
        0
    );

    (data, context)
}
