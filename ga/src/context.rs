use rand::rngs::ThreadRng;
use sgrmath_core::WgpuContext;

use crate::Options;

/// Context for genetic algorithm operations.
///
/// This struct holds the state and resources needed for genetic algorithm operations,
/// including GPU context, random number generator, and various counters.
#[derive(Debug)]
pub struct Context {
    /// The WGPU context used for GPU operations
    pub wgpu: WgpuContext,
    /// Configuration options for the genetic algorithm
    pub options: Options,
    /// Random number generator for genetic operations
    pub rng: ThreadRng,
    /// Next available ID for new individuals
    pub next_id: usize,
    /// Current generation index
    pub generation_index: usize,
    /// Whether the first generation has been initialized
    pub is_initialized: bool,
}

impl Context {
    /// Creates a new context instance.
    ///
    /// # Arguments
    /// * `wgpu` - The WGPU context used for GPU operations
    /// * `options` - Configuration options for the genetic algorithm
    ///
    /// # Returns
    /// A new `Context` instance
    pub fn new(wgpu: &WgpuContext, options: &Options) -> Self {
        Self { 
            wgpu: wgpu.clone(), 
            options: options.clone(),
            rng: rand::rng(),
            next_id: 0,
            generation_index: 0,
            is_initialized: false,
        }
    }
}
