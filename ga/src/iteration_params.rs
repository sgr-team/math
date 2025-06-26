use std::{cell::RefCell, ops::Range, rc::Rc};

use bytemuck::Pod;
use sgrmath_core::Sliced;

use crate::{Context, Data};

/// Parameters for iteration iterations in genetic algorithm.
///
/// This struct contains all necessary parameters for performing iteration iterations
/// such as initialization, crossover, mutation, etc.
///
/// # Type Parameters
/// * `T` - The type of data being processed, must implement `Pod` for GPU compatibility
#[derive(Clone, Debug)]
pub struct IterationParams<T>
where
    T: Pod
{
    /// The GA context
    pub context: Rc<RefCell<Context>>,
    /// The GA data
    pub data: Rc<RefCell<Data<T>>>,
    /// Offset in the solutions array where this iteration should start
    pub solutions_offset: usize,
    /// Number of solutions to process in this iteration
    pub solutions_count: usize,
}

impl<T> IterationParams<T> 
where
    T: Pod
{
    /// Creates a new iteration parameters instance.
    ///
    /// # Arguments
    /// * `context` - The GA context
    /// * `data` - The GA data
    /// * `solutions_count` - The number of solutions to process in this iteration
    ///
    /// # Returns
    /// A new `IterationParams` instance
    pub fn new(context: Rc<RefCell<Context>>, data: Rc<RefCell<Data<T>>>, solutions_count: usize) -> Self {
        Self { context, data, solutions_offset: 0, solutions_count }
    }
}

impl<T> Sliced for IterationParams<T> 
where
    T: Pod
{
    /// Returns the range of solutions this iteration should process
    fn range(&self) -> Range<usize> {
        self.solutions_offset..self.solutions_offset + self.solutions_count
    }

    /// Sets the range of solutions this iteration should process
    fn set_range(&mut self, range: Range<usize>) {
        self.solutions_offset = range.start;
        self.solutions_count = range.end - range.start;
    }
}
