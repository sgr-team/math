use sgrmath_core::{Iteration, ProblemParams, ReadbackBuffer, StorageBuffer, WgpuContext};
use sgrmath_pn::PNP;

use crate::example;

#[test]
fn new() {
    let ctx = WgpuContext::new();
    let pn = PNP::new(&ctx, 100, 3, 10, 10);
    
    assert_eq!(pn.examples.size(), 4000);
    assert_eq!(pn.labels.size(), 400);
}

#[test]
fn init() {
    let ctx = WgpuContext::new();
    let pn = PNP::init(
        &ctx, 
        100,
        3,
        10,
        10,
        (0..1000).map(|i| i as f32).collect::<Vec<_>>(), 
        (0..100).map(|i| 99 - i).collect::<Vec<_>>()
    );
    let reader = ReadbackBuffer::new::<u32, _>(&ctx, 1000);

    assert_eq!(pn.examples.size(), 4000);
    assert_eq!(pn.labels.size(), 400);

    assert_eq!(
        reader.read::<f32>(&ctx, &pn.examples, 0, 1000), 
        (0..1000).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        reader.read::<u32>(&ctx, &pn.labels, 0, 100), 
        (0..100).map(|i| (99 - i) as u32).collect::<Vec<_>>()
    );
}

#[test]
fn from_csv_with_static_value() {
    let ctx = WgpuContext::new();
    let pn = PNP::from_csv(
        &ctx, 
        3, 
        5, 
        "0,0,1,2,3,4\n2,10,9,8,7,6".to_string(), 
        ',',
        Some(255.0),
        None
    );
    let reader = ReadbackBuffer::new::<u32, _>(&ctx, 12);

    assert_eq!(pn.examples_count, 2);
    assert_eq!(
        reader.read::<f32>(&ctx, &pn.examples, 0, 12), 
        vec![ 0.0, 1.0, 2.0, 3.0, 4.0, 255.0, 10.0, 9.0, 8.0, 7.0, 6.0, 255.0 ]
    );
    assert_eq!(
        reader.read::<u32>(&ctx, &pn.labels, 0, 2), 
        vec![ 0, 2 ]
    );
}

#[test]
fn from_csv_without_static_value() {
    let ctx = WgpuContext::new();
    let pn = PNP::from_csv(
        &ctx, 
        3, 
        5, 
        "0,0,1,2,3,4\n2,10,9,8,7,6".to_string(), 
        ',',
        None,
        None
    );
    let reader = ReadbackBuffer::new::<u32, _>(&ctx, 10);

    assert_eq!(pn.examples_count, 2);
    assert_eq!(
        reader.read::<f32>(&ctx, &pn.examples, 0, 10), 
        vec![ 0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 9.0, 8.0, 7.0, 6.0 ]
    );
    assert_eq!(
        reader.read::<u32>(&ctx, &pn.labels, 0, 2), 
        vec![ 0, 2 ]
    );
}

#[test]
fn bind_and_evaluate() {
    let ctx = WgpuContext::new();
    let solutions = StorageBuffer::new::<f32, _>(&ctx, 20);
    let results = StorageBuffer::new::<f32, _>(&ctx, 5);
    let params = ProblemParams {
        context: ctx.clone(),
        solutions_offset: 0,
        solutions_count: 5,
        vector_length: 2, 
        solutions: solutions.clone(), 
        results: results.clone(), 
    };
    let reader = ReadbackBuffer::new::<f32, _>(&ctx, 5);
    let mut pnp = example::pnp(&ctx);
    solutions.update_buffer_range(&ctx, &example::vectors(), 0);

    Iteration::<ProblemParams>::bind(&mut pnp, &params);
    Iteration::<ProblemParams>::evaluate(&mut pnp);

    assert_eq!(reader.read::<f32>(&ctx, &results, 0, 5), example::results());
}


#[test]
fn evaluate_with_params() {
    let ctx = WgpuContext::new();
    let solutions = StorageBuffer::new::<f32, _>(&ctx, 20);
    let results = StorageBuffer::new::<f32, _>(&ctx, 5);
    let params = ProblemParams {
        context: ctx.clone(),
        solutions_offset: 0,
        solutions_count: 5,
        vector_length: 2, 
        solutions: solutions.clone(), 
        results: results.clone(), 
    };
    let reader = ReadbackBuffer::new::<f32, _>(&ctx, 5);
    let mut pnp = example::pnp(&ctx);
    
    solutions.update_buffer_range(&ctx, &example::vectors(), 0);
    Iteration::<ProblemParams>::evaluate_with_params(&mut pnp, &params);

    assert_eq!(reader.read::<f32>(&ctx, &results, 0, 5), example::results());
}
