use crate::{Iteration, ReadbackBuffer};

use super::ProblemParams;

/// A CPU-based implementation of the `Iteration<ProblemParams>` trait.
///
/// ⚠️ **WARNING: Don't do this! Yes, you can, but don't do this!**
///
/// This implementation involves expensive data transfers between CPU and GPU:
/// 1. Reading solution data from GPU memory to CPU
/// 2. Processing it using the CPU-based solver
/// 3. Writing results back to GPU memory
///
/// This struct provides a way to evaluate optimization problems on the CPU
/// while maintaining compatibility with the GPU-based interface. It's useful
/// for problems that are better suited for CPU computation or for testing purposes.
///
/// # Type Parameters
/// * `'a` - The lifetime of the parameters.
/// * `T` - The type of the solution vector elements. Must implement `bytemuck::Pod` to allow safe memory operations.
/// 
///
/// # Performance Considerations
/// * High memory bandwidth usage due to CPU-GPU transfers
/// * Suitable for small problems or testing
/// * Not recommended for production use with large datasets
///
/// # Example
/// ```
/// use sgrmath_core::CpuProblem;
/// 
/// struct Options { pub target: [f32; 2] }
///
/// let problem = CpuProblem::<f32, Options>::new(
///     |solutions, options, params| {
///         (0..params.solutions_count).map(|i| {
///             (solutions[2 * i] - options.target[0]).abs() +
///             (solutions[2 * i + 1] - options.target[1]).abs()
///         }).collect()
///     },
///     Options { target: [ 42.0, 42.2 ] }
/// );
/// ```
pub struct CpuProblem<T, O> 
where
    T: bytemuck::Pod,
{
    /// A function that implements the actual problem evaluation logic.
    pub solver: Box<dyn Fn(Vec<T>, &O, &ProblemParams) -> Vec<f32>>,
    /// The problem options.
    pub options: O,
    /// A buffer used for reading data from GPU memory.
    /// Lazily initialized when first needed to minimize memory usage.
    reader: Option<ReadbackBuffer>,
    /// The parameters bound to this problem.
    binded_params: Option<ProblemParams>,
}

impl<T, O> CpuProblem<T, O> 
where
    T: bytemuck::Pod,
{
    /// Creates a new `CpuProblem` with the given solver function.
    ///
    /// # Arguments
    /// * `solver` - A function that implements the actual problem evaluation logic.
    ///
    /// # Returns
    /// A new `CpuProblem` instance.
    pub fn new(solver: impl Fn(Vec<T>, &O, &ProblemParams) -> Vec<f32> + 'static, options: O) -> Self {
        Self { solver: Box::new(solver), options, reader: None, binded_params: None }
    }

    fn evaluate_inner(&mut self, params: Option<&ProblemParams>) {
        let params = params
            .or_else(|| self.binded_params.as_ref())
            .expect("bind() must be called before evaluate()");

        let solutions_len = params.solutions_count * params.vector_length;
        let reader = self.reader.get_or_insert_with(|| ReadbackBuffer::new::<T, _>(
            &params.context, 
            solutions_len
        ));
        reader.scale::<T, _>(&params.context, solutions_len);

        let solutions = reader.read(&params.context, &params.solutions, 0, solutions_len);
        let results = (self.solver)(solutions, &self.options, &params);

        params.results.update_buffer_range(&params.context, &results, 0);
    }
}

impl<T, O> Iteration<ProblemParams> for CpuProblem<T, O> 
where
    T: bytemuck::Pod,
{
    fn bind(&mut self, params: &ProblemParams) {
        self.binded_params = Some(params.clone());
    }

    fn evaluate(&mut self) {
        self.evaluate_inner(None);
    }

    fn evaluate_with_params(&mut self, params: &ProblemParams) {
        self.evaluate_inner(Some(params));
    }

    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        self.evaluate_inner(None);
        vec![]
    }

    fn evaluate_with_params_async(&mut self, params: &ProblemParams) -> Vec<wgpu::CommandBuffer> {
        self.evaluate_inner(Some(params));
        vec![]
    }
}
