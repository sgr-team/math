use sgrmath_core::{ReadbackBuffer, Size, StorageBuffer, ValueBuffer, WgpuContext};
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
            vec![
                1.0, -1.0, 
                2.0, 1.0, 
                1.0, -1.0,
                -1.0, -2.0
            ]
        ), 
        vec![ 2, 3, 1, 4 ]
    );
    assert_eq!(
        calc(&ctx, example::shader_options(), example::multiply()), 
        example::permutations()
    );
}

fn calc(ctx: &WgpuContext, options: ShaderOptions, input: Vec<f32>) -> Vec<u32> {
    let input_size = (options.examples_count * options.vectors_count * options.solutions_count) as usize;
    let output_size: Size = (options.solutions_count as usize, options.examples_count as usize).into();

    let options_buffer = ValueBuffer::init(ctx, &options);
    let input_buffer = StorageBuffer::new::<f32, _>(ctx, input_size);
    let output_buffer = StorageBuffer::new::<f32, _>(ctx, output_size.len());
    let readback_buffer = ReadbackBuffer::new::<f32, _>(ctx, output_size.len());
    
    input_buffer.update_buffer_range::<f32>(ctx, input.as_slice(), 0);

    let shader = Shaders::new(ctx).permutations;
    shader.execute_with_params(
        ctx, 
        output_size.clone(), 
        &[ &options_buffer, &input_buffer, &output_buffer, ]
    );

    readback_buffer.read::<u32>(ctx, &output_buffer, 0, output_size.len())
}
