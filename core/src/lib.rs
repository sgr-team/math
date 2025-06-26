#![deny(clippy::unwrap_used, clippy::panic, clippy::unimplemented, clippy::todo)]
#![warn(clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::missing_safety_doc, clippy::module_name_repetitions, clippy::multiple_crate_versions, clippy::too_many_arguments, clippy::type_complexity, clippy::upper_case_acronyms, clippy::module_inception, clippy::cast_possible_truncation)]

//! SGR Math Core - A foundational library providing GPU acceleration infrastructure
//! through WebGPU (wgpu) shaders for mathematical computations.
//!
//! This library provides the core infrastructure for GPU-accelerated mathematical computations,
//! including buffer management, shader compilation, and GPU context management.
//!
//! # Examples
//!
//! ```no_run
//! use sgrmath_core::WgpuContext;
//!
//! fn example() {
//!     // Create a new GPU context
//!     let context = WgpuContext::new();
//! }
//! ```

mod buffers;
mod model;
mod shader;
mod wgpu_context;

// Re-export main types
pub use wgpu_context::WgpuContext;

// Re-export buffer types
pub use buffers::{
    ReadbackBuffer,
    StorageBuffer,
    ValueBuffer,
};

// Re-export model types
pub use model::{
    Size,
    OptimizationDirection,
    ProblemParams,
    CpuProblem,
    ShaderProblem,
    Iteration,
    Compiled,
    CompiledIteration,
    NotImplementedIteration,
    CombinedIteration,
    IterationSize,
    Sliced,
    SlicedIteration,
};

// Re-export shader types
pub use shader::Shader;
