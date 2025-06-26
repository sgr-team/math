use std::borrow::Cow;

use wgpu::{BindGroup, Buffer, ComputePipeline, PipelineCompilationOptions};

use crate::{Size, WgpuContext};

/// A wrapper around a wgpu compute pipeline that manages an optional bind group for resource binding.
///
/// The `Shader` struct allows you to execute compute shaders with either a persistent bind group 
/// (set via `bind`) or with parameters provided per-dispatch.
///
/// - Use `bind` to set a bind group for repeated executions with the same resources.
/// - Use `execute_with_params` or `execute_with_params_async` to dispatch with custom resources each time.
/// - Use `is_bound` to check if a bind group is currently set.
///
/// This design provides flexibility for both reusable and dynamic resource binding scenarios.
#[derive(Debug, Clone)]
pub struct Shader(pub ComputePipeline, pub Option<BindGroup>);

impl Shader {
    /// Creates a new `Shader` from WGSL source code and an optional pipeline layout.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used to create the pipeline and shader module.
    /// * `label` - A label for debugging purposes.
    /// * `source` - The WGSL source code for the compute shader.
    ///
    /// # Returns
    /// A new `Shader` instance with no bind group set.
    pub fn new<'a, S>(
        context: &WgpuContext, 
        label: &str, 
        source: S
    ) -> Self  
    where
        S: Into<Cow<'a, str>>,
    {
        Self(
            context.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Compute Pipeline: {label}")),
                    layout: None,
                    module: &context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    }),
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None
                }
            ),
            None
        )
    }

    /// Sets the bind group for this shader, allowing repeated execution with the same resources.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used to create the bind group.
    /// * `params` - The buffer resources to bind.
    pub fn bind(&mut self, context: &WgpuContext, params: &[&Buffer]) {
        self.1 = Some(self.create_bind_group(context, params));
    }

    /// Removes the currently set bind group from this shader.
    pub fn unbind(&mut self) {
        self.1 = None;
    }

    /// Executes the compute shader using the currently bound bind group.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used for command submission.
    /// * `size` - The dispatch size (workgroup counts) for the compute shader.
    ///
    /// # Panics
    /// Panics if no bind group is currently set. Use `bind` to set one.
    pub fn execute<S>(&self, context: &WgpuContext, size: S) 
    where
        S: Into<Size>,
    {
        context.queue.submit(Some(self.execute_async(context, size)));
    }

    /// Executes the compute shader with a custom set of buffer parameters, creating a bind group on the fly.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used for command submission.
    /// * `size` - The dispatch size (workgroup counts) for the compute shader.
    /// * `params` - The buffer resources to bind for this execution.
    pub fn execute_with_params<S>(&self, context: &WgpuContext, size: S, params: &[&Buffer]) 
    where
        S: Into<Size>,
    {
        context.queue.submit(Some(self.execute_with_params_async(context, size, params)));
    }

    /// Returns a command buffer for executing the compute shader with the currently bound bind group.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used for command encoding.
    /// * `size` - The dispatch size (workgroup counts) for the compute shader.
    ///
    /// # Returns
    /// A `wgpu::CommandBuffer` ready for submission.
    ///
    /// # Panics
    /// Panics if no bind group is currently set. Use `bind` to set one.
    #[must_use]
    #[allow(clippy::panic)]
    pub fn execute_async<S>(
        &self, 
        context: &WgpuContext, 
        size: S
    ) -> wgpu::CommandBuffer
    where
        S: Into<Size>,
    {
        self.1
            .as_ref()
            .map_or_else(
                || panic!("No bind group found. Use `bind` to bind parameters to the shader."), 
                |bind_group| self.execute_with_bind_group(context, size, bind_group)
            )
    }
    
    /// Returns a command buffer for executing the compute shader with a custom set of buffer parameters.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used for command encoding.
    /// * `size` - The dispatch size (workgroup counts) for the compute shader.
    /// * `params` - The buffer resources to bind for this execution.
    ///
    /// # Returns
    /// A `wgpu::CommandBuffer` ready for submission.
    pub fn execute_with_params_async<S>(
        &self, 
        context: &WgpuContext, 
        size: S, 
        params: &[&Buffer]
    ) -> wgpu::CommandBuffer
    where
        S: Into<Size>,
    {
        self.execute_with_bind_group(context, size, &self.create_bind_group(context, params))
    }

    fn execute_with_bind_group<S>(&self, context: &WgpuContext, size: S, bind_group: &BindGroup) -> wgpu::CommandBuffer
    where
        S: Into<Size>,
    {
        let s = size.into();
        
        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.0);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(s.width as u32, s.height as u32, s.depth as u32);
        }

        encoder.finish()
    }

    fn create_bind_group(&self, context: &WgpuContext, params: &[&Buffer]) -> BindGroup {
        context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.0.get_bind_group_layout(0),
            entries: params
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry { binding: i as u32, resource: buffer.as_entire_binding() })
                .collect::<Vec<_>>()
                .as_slice(),
        })
    }

    /// Returns `true` if a bind group is currently set for this shader.
    #[must_use]
    pub const fn is_bound(&self) -> bool {
        self.1.is_some()
    }
}

impl From<wgpu::ComputePipeline> for Shader {
    fn from(pipeline: wgpu::ComputePipeline) -> Self {
        Self(pipeline, None)
    }
}