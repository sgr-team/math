use sgrmath_core::{StorageBuffer, WgpuContext};
use sgrmath_pn::{shaders::{ShaderOptions, Shaders}, PNP};

pub fn pnp(wgpu: &WgpuContext) -> PNP {
    PNP {
        wgpu: wgpu.clone(),
        examples_count: 10,
        permutations_count: 6,
        vectors_count: 2,
        vector_length: 2,
        outputs_count: 4,
        examples: StorageBuffer::init(&wgpu, &examples()),
        labels: StorageBuffer::init(&wgpu, &labels()),
        shaders: Shaders::new(&wgpu),
        params: None,
    }
}

pub fn shader_options() -> ShaderOptions {
    ShaderOptions { 
        examples_count: 10, 
        vector_length: 2, 
        vectors_count: 2, 
        solutions_count: 5, 
        outputs_count: 4, 
        permutations_count: 6 
    }
}


pub fn examples() -> Vec<f32> {
    return vec![
        0.0, 9.0,
        1.0, 8.0,
        2.0, 7.0,
        3.0, 6.0,
        4.0, 5.0,
        5.0, 4.0,
        6.0, 3.0,
        7.0, 2.0,
        8.0, 1.0,
        9.0, 0.0,
    ];
}

pub fn labels() -> Vec<u32> {
    return vec![ 1, 1, 1, 2, 2, 2, 3, 3, 3, 0 ];
}

pub fn vectors() -> Vec<f32> {
    vec![
        1.0, -0.5, -0.25, 0.5,
        1.0, 0.5, 0.25, 1.5,
        -1.0, -0.5, -0.25, -1.5,
        1.0, 1.0, -1.0, -1.0,
        0.0, 0.0, 0.0, 0.0,
    ]
}

pub fn multiply() -> Vec<f32> {
    let mut result = vec![];
    let opt = shader_options();
    let vctrs = vectors();
    let exmpls = examples();
    
    for vector_index in 0..(opt.vectors_count * opt.solutions_count) as usize {
        let vector_start = vector_index * opt.vector_length as usize;
        for example_index in 0..opt.examples_count as usize {
            let mut sum = 0.0;
            let example_start = example_index * opt.vector_length as usize;

            for k_index in 0..opt.vector_length as usize {
                sum += exmpls[example_start + k_index] * vctrs[vector_start + k_index];
            }
            result.push(sum);
        }
    }

    result
}

pub fn permutations() -> Vec<u32> {
    vec![
        3, 3, 3, 2, 0, 0, 0, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
        4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
}

pub fn results() -> Vec<f32> {
    vec![ 8.0, 6.0, 6.0, 3.0, 3.0 ]
}