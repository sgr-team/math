use sgrmath_core::{ReadbackBuffer, StorageBuffer, ValueBuffer, WgpuContext};
use sgrmath_pn::shaders::{Shaders, ShaderOptions};

use crate::example;

#[test]
fn evaluate() {
    let ctx = WgpuContext::new();
    
    assert_eq!(
        calc(
            &ctx, 
            ShaderOptions { 
                examples_count: 2, 
                vector_length: 2, 
                vectors_count: 2, 
                solutions_count: 2, 
                outputs_count: 3, 
                permutations_count: 6
            }, 
            vec![ 1.0, 2.0, 7.0, 8.0 ], 
            vec![ 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0 ]
        ), 
        vec![ 11.0, 53.0, 17.0, 83.0, 50.0, 230.0, 110.0, 530.0 ]
    );
    assert_eq!(
        calc(&ctx, example::shader_options(), example::examples(), example::vectors()), 
        example::multiply()
    );
}

fn calc(
    ctx: &WgpuContext, 
    options: ShaderOptions, 
    examples: Vec<f32>, 
    vectors: Vec<f32>
) -> Vec<f32> {
    let examples_size = (options.examples_count * options.vector_length) as usize;
    let vectors_size = (options.solutions_count * options.vectors_count * options.vector_length) as usize;
    let output_size = (options.solutions_count * options.vectors_count * options.examples_count) as usize;

    let options_buffer = ValueBuffer::init(ctx, &options);
    let examples_buffer = StorageBuffer::new::<f32, _>(ctx, examples_size);
    let vectors_buffer = StorageBuffer::new::<f32, _>(ctx, vectors_size);
    let output_buffer = StorageBuffer::new::<f32, _>(ctx, output_size);
    let readback_buffer = ReadbackBuffer::new::<f32, _>(ctx, output_size);
    
    examples_buffer.update_buffer_range::<f32>(ctx, examples.as_slice(), 0);
    vectors_buffer.update_buffer_range::<f32>(ctx, vectors.as_slice(), 0);

    let shader = Shaders::new(ctx).multiply;
    shader.execute_with_params(
        ctx, 
        (
            options.examples_count as usize, 
            (options.vectors_count * options.solutions_count) as usize
        ), 
        &[ &options_buffer, &examples_buffer, &vectors_buffer, &output_buffer, ]
    );

    readback_buffer.read::<f32>(ctx, &output_buffer, 0, output_size)
}