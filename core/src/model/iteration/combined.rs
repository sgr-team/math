use std::ops::Deref;

use crate::{Iteration, WgpuContext};

/// A container that combines multiple iterations to be executed in parallel.
/// 
/// CombinedIteration allows you to run multiple iterations concurrently, collecting
/// their command buffers and managing synchronization. All iterations are executed
/// asynchronously, and the container ensures proper synchronization when needed.
/// 
/// # Examples
/// ```
/// use sgrmath_core::{CombinedIteration, NotImplementedIteration, WgpuContext};
/// 
/// fn example(context: &WgpuContext) {
///     let _combined = CombinedIteration::<Params>::new(context)
///         .add(Box::new(NotImplementedIteration::new("First")))
///         .add(Box::new(NotImplementedIteration::new("Second")));
/// }
/// 
/// type Params = std::ops::Range<usize>;
/// ```
pub struct CombinedIteration<T: Clone> {
    iterations: Vec<Box<dyn Iteration<T>>>,
    context: WgpuContext,
}

impl<T> CombinedIteration<T> 
where
    T: Clone
{
    /// Creates a new empty CombinedIteration.
    /// 
    /// # Arguments
    /// * `context` - The WGPU context used for synchronization
    pub fn new(context: &WgpuContext) -> Self {
        Self {
            iterations: vec![],
            context: context.clone(),
        }
    }

    /// Adds a new iteration to be executed in parallel with others.
    /// 
    /// # Arguments
    /// * `iteration` - The iteration to add
    /// 
    /// # Returns
    /// Self for method chaining
    pub fn add(mut self, iteration: Box<dyn Iteration<T>>) -> Self {
        self.iterations.push(iteration);
        self
    }

    /// Removes an iteration at the specified index.
    /// 
    /// # Arguments
    /// * `index` - The index of the iteration to remove
    /// 
    /// # Returns
    /// Self for method chaining
    pub fn remove(mut self, index: usize) -> Self {
        self.iterations.remove(index);
        self
    }

    /// Removes all iterations.
    /// 
    /// # Returns
    /// Self for method chaining
    pub fn clear(mut self) -> Self {
        self.iterations.clear();
        self
    }

    /// Sets the iterations to the specified vector.
    /// 
    /// # Arguments
    /// * `iterations` - Vector of iterations to execute in parallel
    /// 
    /// # Returns
    /// &mut Self for method chaining
    pub fn set(&mut self, iterations: Vec<Box<dyn Iteration<T>>>) -> &mut Self {
        self.iterations = iterations;
        self
    }
}

impl<T> Iteration<T> for CombinedIteration<T> 
where
    T: Clone
{
    /// Binds parameters to all iterations.
    /// 
    /// Each iteration receives the same parameters, allowing them to work
    /// with the same data in parallel.
    fn bind(&mut self, params: &T) {
        for iteration in self.iterations.iter_mut() {
            iteration.bind(params);
        }
    }

    /// Evaluates all iterations synchronously.
    /// 
    /// This method:
    /// 1. Collects command buffers from all iterations asynchronously
    /// 2. Waits for all operations to complete
    /// 
    /// Note: While this method is synchronous, the underlying iterations
    /// are still executed asynchronously.
    fn evaluate(&mut self) {
        self.evaluate_async();
        self.context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    }
    
    /// Evaluates all iterations asynchronously.
    /// 
    /// This method:
    /// 1. Collects command buffers from all iterations
    /// 2. Waits for any pending operations to complete
    /// 3. Returns the collected command buffers
    /// 
    /// The iterations are executed in parallel, and their command buffers
    /// are combined into a single vector.
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        let mut result = vec![];
        for iteration in self.iterations.iter_mut() {
            result.extend(iteration.evaluate_async());
        }

        self.context.device.poll(wgpu::MaintainBase::Wait).unwrap();
        result
    }

    /// Evaluates all iterations with parameters synchronously.
    /// 
    /// This method:
    /// 1. Collects command buffers from all iterations asynchronously
    /// 2. Waits for all operations to complete
    /// 
    /// Note: While this method is synchronous, the underlying iterations
    /// are still executed asynchronously.
    fn evaluate_with_params(&mut self, params: &T) {
        self.evaluate_with_params_async(params);
        self.context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    }
    
    /// Evaluates all iterations with parameters asynchronously.
    /// 
    /// This method:
    /// 1. Collects command buffers from all iterations
    /// 2. Returns the collected command buffers
    /// 
    /// The iterations are executed in parallel, and their command buffers
    /// are combined into a single vector.
    fn evaluate_with_params_async(&mut self, params: &T) -> Vec<wgpu::CommandBuffer> {
        let mut result = vec![];
        for iteration in self.iterations.iter_mut() {
            result.extend(iteration.evaluate_with_params_async(params));
        }

        result
    }
}

impl<T> Deref for CombinedIteration<T> 
where
    T: Clone
{
    type Target = Vec<Box<dyn Iteration<T>>>;

    /// Returns a reference to the underlying vector of iterations.
    fn deref(&self) -> &Self::Target {
        &self.iterations
    }
}
