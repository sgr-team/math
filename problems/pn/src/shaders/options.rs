use crate::PNP;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct ShaderOptions {
    pub examples_count: u32,
    pub vector_length: u32,
    pub vectors_count: u32,
    pub solutions_count: u32,
    pub outputs_count: u32,
    pub permutations_count: u32,
}

impl ShaderOptions {
    pub fn new(pn: &PNP, solutions_count: usize) -> Self {
        Self {
            examples_count: pn.examples_count as u32,
            vector_length: pn.vector_length as u32,
            vectors_count: pn.vectors_count as u32,
            solutions_count: solutions_count as u32,
            outputs_count: pn.outputs_count as u32,
            permutations_count: pn.permutations_count as u32,
        }
    }
}

impl ShaderOptions {
    pub fn wgsl() -> String {
        format!(
            "
            struct PNPOptions {{
                examples_count: u32,
                vector_length: u32,
                vectors_count: u32,
                solutions_count: u32,
                outputs_count: u32,
                permutations_count: u32
            }}
            ",
        )
    }
}