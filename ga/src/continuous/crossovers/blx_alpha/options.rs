#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct ShaderOptions {
    pub generation_offset: u32,
    pub vector_length: u32,
    pub parents_count: u32,
    pub min: f32,
    pub max: f32
}