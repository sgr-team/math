use std::cmp::Ordering;

/// Direction of optimization for the genetic algorithm.
///
/// Determines whether the algorithm should try to minimize
/// or maximize the fitness function.
#[derive(Clone, Debug)]
pub enum OptimizationDirection {
    /// Minimize the fitness function (find the smallest possible value)
    Minimize,
    /// Maximize the fitness function (find the largest possible value)
    Maximize,
}

impl OptimizationDirection {
    /// Checks if the optimization direction is to minimize the fitness function
    /// 
    /// # Returns
    /// * `true` if the optimization direction is to minimize the fitness function
    /// * `false` if the optimization direction is to maximize the fitness function
    pub fn is_minimize(&self) -> bool {
        matches!(self, OptimizationDirection::Minimize)
    }

    /// Checks if the optimization direction is to maximize the fitness function
    /// 
    /// # Returns
    /// * `true` if the optimization direction is to maximize the fitness function
    /// * `false` if the optimization direction is to minimize the fitness function
    pub fn is_maximize(&self) -> bool {
        matches!(self, OptimizationDirection::Maximize)
    }

    /// Compares two fitness values and returns the ordering
    /// 
    /// # Arguments
    /// * `a` - The first fitness value
    /// * `b` - The second fitness value
    /// 
    /// # Returns
    /// * `Less` if `a` is "less" (with respect to the optimization direction) than `b`
    /// * `Equal` if `a` is "equal" to `b`
    /// * `Greater` if `a` is "greater" (with respect to the optimization direction) than `b`
    pub fn compare(&self, a: &f32, b: &f32) -> Ordering {
        match self {
            OptimizationDirection::Minimize => a.partial_cmp(b).unwrap(),
            OptimizationDirection::Maximize => b.partial_cmp(a).unwrap(),
        }
    }
}