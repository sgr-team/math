use sgrmath_core::{ReadbackBuffer, StorageBuffer, ValueBuffer, WgpuContext};
use sgrmath_pn::shaders::{Shaders, ShaderOptions};

use crate::example;

#[test]
fn evaluate() {
    let ctx = WgpuContext::new();
    
    assert_eq!(
        calc(
            &ctx, 
            example::shader_options(),
            example::labels(), 
            example::permutations(), 
        ), 
        example::results()
    );
}

fn calc(
    ctx: &WgpuContext, 
    options: ShaderOptions, 
    examples: Vec<u32>, 
    permutations: Vec<u32>
) -> Vec<f32> {
    let options_buffer = ValueBuffer::init(ctx, &options);
    let examples_buffer = StorageBuffer::new::<u32, _>(
        ctx, 
        options.examples_count as usize
    );
    let permutations_buffer = StorageBuffer::new::<u32, _>(
        ctx, 
        (options.permutations_count as usize, options.examples_count as usize)
    );
    let permutation_labels_buffer = StorageBuffer::new::<f32, _>(
        ctx, 
        (options.solutions_count * options.permutations_count * options.outputs_count) as usize
    );
    let output_buffer = StorageBuffer::new::<f32, _>(ctx, options.solutions_count as usize);
    let readback_buffer = ReadbackBuffer::new::<f32, _>(ctx, options.solutions_count as usize);
    
    examples_buffer.update_buffer_range(ctx, examples.as_slice(), 0);
    permutations_buffer.update_buffer_range(ctx, permutations.as_slice(), 0);

    let shader = Shaders::new(ctx).results;
    shader.execute_with_params(
        ctx, 
        options.solutions_count as usize, 
        &[ &options_buffer, &examples_buffer, &permutations_buffer, &permutation_labels_buffer, &output_buffer, ]
    );

    readback_buffer.read(
        ctx, 
        &output_buffer, 
        0, 
        options.solutions_count as usize
    )
}