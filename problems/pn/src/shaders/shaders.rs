use sgrmath_core::{Shader, WgpuContext};

use crate::shaders::ShaderOptions;

#[derive(Debug, Clone)]
pub struct Shaders {
    pub multiply: Shader,
    pub permutations: Shader,
    pub results: Shader,
    pub test: Shader,
}

impl Shaders {
    pub fn new(ctx: &WgpuContext) -> Self {
        Self {
            multiply: create_shader(ctx, "pnp::multiply", include_str!("wgsl/multiply.wgsl")),
            permutations: create_shader(ctx, "pnp::permutations", include_str!("wgsl/permutations.wgsl")),
            results: create_shader(ctx, "pnp::results", include_str!("wgsl/results.wgsl")),
            test: create_shader(ctx, "pnp::test", include_str!("wgsl/test.wgsl")),
        }
    }
}

fn create_shader<S: Into<String>>(
    ctx: &WgpuContext, 
    name: &str, 
    shader: S,
) -> Shader {
    Shader::new(
        ctx, 
        name, 
        format!(
            "{}\n\n{}",
            ShaderOptions::wgsl(),
            shader.into()
        )
    )
}
