mod context;
mod data;
mod ga;
mod individual;
mod iteration_params;
mod options;

pub use context::*;
pub use data::*;
pub use ga::*;
pub use individual::*;
pub use iteration_params::*;
pub use options::*;

/// Common module
/// 
/// This module contains the common functions and structures for the GA.
pub mod common;

/// Continuous optimization module
/// 
/// This module contains the implementation of the continuous optimization iterations for the GA.
pub mod continuous;