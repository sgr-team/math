use std::ops::Deref;

use crate::{Iteration, IterationSize, Sliced};

/// A container that manages multiple iterations with different size distributions.
/// 
/// SlicedIteration allows you to split a population into multiple segments and apply
/// different iteration strategies to each segment. The size of each segment can be
/// specified either as a fixed count or as a proportional value.
/// 
/// # Examples
/// ```
/// use std::ops::Range;
/// use sgrmath_core::{SlicedIteration, IterationSize, NotImplementedIteration, Sliced};
/// 
/// let iteration: SlicedIteration<Params> = SlicedIteration::new()
///     // Fixed size of 100 elements
///     .add(100, Box::new(NotImplementedIteration::new("First")))
///     // Gets twice as much space as the next proportional
///     .add(IterationSize::Proportional(2.0), Box::new(NotImplementedIteration::new("Second")));
/// 
/// type Params = Range<usize>;
/// ```
pub struct SlicedIteration<T: Sliced + Clone> (
    Vec<(IterationSize, Box<dyn Iteration<T>>)>,
    Option<Vec<usize>>
);

impl<T> SlicedIteration<T> 
where
    T: Sliced + Clone
{
    /// Creates a new empty SlicedIteration.
    pub fn new() -> Self { Self(vec![], None) }

    /// Adds a new iteration with the specified size.
    /// 
    /// # Arguments
    /// * `size` - The size of the segment, either as a fixed count or proportional value
    /// * `iteration` - The iteration to apply to this segment
    /// 
    /// # Returns
    /// Self for method chaining
    pub fn add<S>(mut self, size: S, iteration: Box<dyn Iteration<T>>) -> Self
    where
        S: Into<IterationSize>
    {
        self.0.push((size.into(), iteration));
        self.1 = None;
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
        self.0.remove(index);
        self.1 = None;
        self
    }

    /// Removes all iterations.
    /// 
    /// # Returns
    /// Self for method chaining
    pub fn clear(mut self) -> Self {
        self.0.clear();
        self.1 = None;
        self
    }

    /// Sets the iterations to the specified vector.
    /// 
    /// # Arguments
    /// * `slices` - Vector of (size, iteration) pairs
    /// 
    /// # Returns
    /// &mut Self for method chaining
    pub fn set(&mut self, slices: Vec<(IterationSize, Box<dyn Iteration<T>>)>) -> &mut Self {
        self.0 = slices;
        self.1 = None;
        self
    }

    /// Distributes the total size across all iterations according to their size specifications.
    /// 
    /// This method:
    /// 1. First allocates space for all fixed sizes
    /// 2. Then distributes the remaining space proportionally among proportional sizes
    /// 3. Ensures the last proportional size gets the remaining space to maintain the total
    /// 
    /// # Arguments
    /// * `total` - The total size to distribute
    /// 
    /// # Returns
    /// A reference to the vector of calculated sizes
    /// 
    /// # Panics
    /// If the total size is less than the sum of fixed sizes
    pub fn distribute(&mut self, total: usize) -> &Vec<usize> {
        match self.1 {
            Some(ref sizes) => sizes,
            None => {
                // First pass: validate proportional values and calculate fixed sum
                let mut fixed_sum = 0;
                let mut proportional_sum = 0.0;
                let mut proportional_indices = Vec::new();
                
                for (i, (size, _)) in self.0.iter().enumerate() {
                    match size {
                        IterationSize::Count(count) => fixed_sum += count,
                        IterationSize::Proportional(value) => {
                            proportional_sum += value;
                            proportional_indices.push(i);
                        }
                    }
                }

                // Check if we have enough space for fixed sizes
                if fixed_sum > total {
                    panic!("Total size is less than sum of fixed sizes, got {} < {}", total, fixed_sum);
                }

                // Calculate remaining space for proportional distribution
                let remaining = total - fixed_sum;
                
                // Second pass: calculate final sizes
                let mut result = vec![0; self.0.len()];
                let mut distributed = 0;

                // Handle all proportional values except the last one
                for &idx in proportional_indices.iter().take(proportional_indices.len().saturating_sub(1)) {
                    if let IterationSize::Proportional(value) = self.0[idx].0 {
                        let size = (remaining as f32 * (value / proportional_sum)) as usize;
                        result[idx] = size;
                        distributed += size;
                    }
                }

                // Handle the last proportional value to ensure total sum
                if let Some(&last_idx) = proportional_indices.last() {
                    result[last_idx] = remaining - distributed;
                }

                // Fill in fixed sizes
                for (i, (size, _)) in self.0.iter().enumerate() {
                    if let IterationSize::Count(count) = size {
                        result[i] = *count;
                    }
                }

                self.1 = Some(result);
                self.1.as_ref().unwrap()
            }
        }
    }
}

impl<T> Iteration<T> for SlicedIteration<T> 
where
    T: Sliced + Clone
{
    /// Binds parameters to all iterations, distributing the range according to their sizes.
    /// 
    /// For each iteration:
    /// 1. Calculates its portion of the range based on its size
    /// 2. Creates a new parameter instance with the appropriate range
    /// 3. Binds the parameters to the iteration
    fn bind(&mut self, params: &T) {
        let range = params.range();
        let sizes = self.distribute(range.len());
        let mut prev = range.start;
        let iteration_params: Vec<_> = sizes.iter()
            .map(|&size| {
                let mut params = params.clone();
                let end = prev + size;
                params.set_range(prev..end);
                prev = end;
                params
            })
            .collect();

        for (params, (_, iteration)) in iteration_params.into_iter().zip(self.0.iter_mut()) {
            iteration.bind(&params);
        }
    }

    /// Evaluates all iterations in sequence.
    fn evaluate(&mut self) {
        for (_, iteration) in self.0.iter_mut() {
            iteration.evaluate();
        }
    }

    /// Evaluates all iterations with their respective parameter ranges.
    /// 
    /// Similar to `bind` followed by `evaluate`, but combines them into a single operation.
    fn evaluate_with_params(&mut self, params: &T) {
        let range = params.range();
        let sizes = self.distribute(range.len());
        let mut prev = range.start;
        let iteration_params: Vec<_> = sizes.iter()
            .map(|&size| {
                let mut iteration_param = params.clone();
                let end = prev + size;
                iteration_param.set_range(prev..end);
                prev = end;
                iteration_param
            })
            .collect();

        for (iteration_param, (_, iteration)) in iteration_params.into_iter().zip(self.0.iter_mut()) {
            iteration.evaluate_with_params(&iteration_param);
        }
    }

    /// Evaluates all iterations asynchronously and collects their command buffers.
    /// 
    /// # Returns
    /// A vector of command buffers from all iterations
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        let mut result = vec![];
        for (_, iteration) in self.0.iter_mut() {
            result.extend(iteration.evaluate_async());
        }
        result
    }

    /// Evaluates all iterations asynchronously with their respective parameter ranges.
    /// 
    /// Similar to `bind` followed by `evaluate_async`, but combines them into a single operation.
    /// 
    /// # Returns
    /// A vector of command buffers from all iterations
    fn evaluate_with_params_async(&mut self, params: &T) -> Vec<wgpu::CommandBuffer> {
        let range = params.range();
        let sizes = self.distribute(range.len());
        let mut prev = range.start;
        let iteration_params: Vec<_> = sizes.iter()
            .map(|&size| {
                let mut iteration_param = params.clone();
                let end = prev + size;
                iteration_param.set_range(prev..end);
                prev = end;
                iteration_param
            })
            .collect();

        let mut result = vec![];
        for (iteration_param, (_, iteration)) in iteration_params.into_iter().zip(self.0.iter_mut()) {
            result.extend(iteration.evaluate_with_params_async(&iteration_param));
        }
        result
    }
}

impl<T> Deref for SlicedIteration<T> 
where
    T: Sliced + Clone
{
    type Target = Vec<(IterationSize, Box<dyn Iteration<T>>)>;

    /// Returns a reference to the underlying vector of iterations.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
