use sgrmath_core::OptimizationDirection;

/// Configuration options for genetic algorithm.
///
/// This struct contains all the parameters needed to configure
/// the genetic algorithm's behavior and performance.
#[derive(Clone, Debug)]
pub struct Options {
    /// Direction of optimization - whether to minimize or maximize the fitness function
    pub optimization_direction: OptimizationDirection,
    /// Total size of the population
    pub population_size: usize,
    /// Number of individuals processed in each generation
    pub generation_size: usize,
    /// Number of parents selected for crossover in each generation
    pub parents_count: usize,
    /// Length of the solution vector for each individual
    pub vector_length: usize,
    /// Minimum possible value in the solution vector
    pub min_value: f32,
    /// Maximum possible value in the solution vector
    pub max_value: f32,
}
