use wgpu::Buffer;

use crate::{Iteration, ProblemParams, Shader, WgpuContext};

/// A problem implementation that uses GPU shaders for computation.
/// 
/// This struct manages the lifecycle of a shader-based computation problem,
/// handling the binding of buffers and execution of the shader program.
/// 
/// # Example
/// 
/// ```rust
/// use sgrmath_core::{ProblemParams, Shader, ShaderProblem, StorageBuffer, WgpuContext};
/// 
/// // Create a shader that computes distance to target point
/// fn create_distance_problem() -> ShaderProblem {
///     let context = WgpuContext::new();
///     let target_buffer = StorageBuffer::new::<f32, _>(&context, 2);
///     target_buffer.update_buffer_range::<f32>(&context, &[ 42.0, 42.2 ], 0);
///     
///     ShaderProblem::new(
///         Shader::new(&context, "distance", "shader source"),
///         vec![ target_buffer.0 ]
///     )
/// }
/// ```
pub struct ShaderProblem {
    /// Current binding state containing the WGPU context and number of solutions.
    /// None when the problem is not bound to a context.
    binding_state: Option<(WgpuContext, usize)>,
    /// The shader program that performs the actual computation
    pub shader: Shader,
    /// Additional buffer parameters that will be passed to the shader
    pub additional_params: Vec<Buffer>,
}

impl ShaderProblem {
    /// Creates a new shader-based problem.
    /// 
    /// # Arguments
    /// 
    /// * `shader` - The shader program to use for computation
    /// * `additional_params` - Additional buffer parameters to pass to the shader
    #[must_use]
    pub const fn new(shader: Shader, additional_params: Vec<Buffer>) -> Self {
        Self { binding_state: None, shader, additional_params }
    }
}

impl Iteration<ProblemParams> for ShaderProblem {
    /// Binds the problem to a WGPU context and prepares the shader for execution.
    /// 
    /// This method sets up the binding state and configures the shader with
    /// the necessary buffer parameters.
    fn bind(&mut self, params: &ProblemParams) {
        self.binding_state = Some((params.context.clone(), params.solutions_count));
        let mut buffers = vec![ &params.solutions.0, &params.results ];
        buffers.extend(self.additional_params.iter());

        self.shader.bind(&params.context, &buffers);
    }

    /// Executes the shader program using the previously bound context and parameters.
    /// 
    /// # Panics
    /// 
    /// Panics if the problem has not been bound to a context using `bind()`.
    fn evaluate(&mut self) {
        let (context, solutions_count) = self.binding_state
            .as_ref()
            .expect("ShaderProblem must be bound before evaluate()");

        self.shader.execute(context, *solutions_count);
    }

    /// Executes the shader program with new parameters without changing the binding state.
    /// 
    /// This method allows for executing the shader with different input parameters
    /// while maintaining the same WGPU context binding.
    fn evaluate_with_params(&mut self, params: &ProblemParams) {
        let mut buffers = vec![ &params.solutions.0, &params.results ];
        buffers.extend(self.additional_params.iter());

        self.shader.execute_with_params(&params.context, params.solutions_count, &buffers);
    }

    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        let (context, solutions_count) = self.binding_state
            .as_ref()
            .expect("ShaderProblem must be bound before evaluate()");

        vec![ self.shader.execute_async(context, *solutions_count) ]
    }

    fn evaluate_with_params_async(&mut self, params: &ProblemParams) -> Vec<wgpu::CommandBuffer> {
        let mut buffers = vec![ &params.solutions.0, &params.results ];
        buffers.extend(self.additional_params.iter());

        vec![ self.shader.execute_with_params_async(&params.context, params.solutions_count, &buffers) ]
    }
}
