/// Represents an individual in the genetic algorithm population.
///
/// Each individual has a unique ID, belongs to a specific generation,
/// has a list of parent IDs, and a fitness result.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    /// Unique identifier for the individual
    pub id: usize,
    /// Generation number this individual belongs to
    pub generation: usize,
    /// List of parent IDs that created this individual
    pub parents: Vec<usize>,
    /// Fitness result of this individual
    pub result: f32,
}
