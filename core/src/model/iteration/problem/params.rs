use std::ops::Range;

use crate::{Sliced, StorageBuffer, WgpuContext};

/// A struct representing the parameters for a problem.
///
/// This struct contains the buffers and parameters needed for evaluating a problem.
/// It is used to pass parameters to the `Problem` trait methods.
///
/// # Fields
/// * `context` - The GPU context
/// * `solutions` - The solutions buffer
/// * `results` - The results buffer
/// * `solutions_offset` - The offset of the solutions buffer
/// * `solutions_count` - The number of solutions to evaluate
/// * `vector_length` - The size of each solution vector
///
/// # Examples
/// ```
/// use sgrmath_core::{ProblemParams, WgpuContext, StorageBuffer};
/// 
/// async fn create_problem_params(
///     context: WgpuContext,
///     solutions: StorageBuffer,
///     results: StorageBuffer,
///     solutions_offset: usize,
///     solutions_count: usize,
///     vector_length: usize
/// ) -> ProblemParams {
///     ProblemParams { context, solutions, results, solutions_offset, solutions_count, vector_length }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ProblemParams {
    /// The GPU context
    pub context: WgpuContext,
    /// The solutions buffer
    pub solutions: StorageBuffer,
    /// The results buffer
    pub results: StorageBuffer,
    /// The offset of the solutions buffer
    pub solutions_offset: usize,
    /// The number of solutions to evaluate
    pub solutions_count: usize,
    /// The size of each solution vector
    pub vector_length: usize,
}

impl Sliced for ProblemParams {
    fn range(&self) -> Range<usize> {
        self.solutions_offset..self.solutions_offset + self.solutions_count
    }

    fn set_range(&mut self, range: Range<usize>) {
        self.solutions_offset = range.start;
        self.solutions_count = range.end - range.start;
    }
}
