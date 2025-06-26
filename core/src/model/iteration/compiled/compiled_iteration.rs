use std::marker::PhantomData;

use crate::model::Iteration;
use super::Compiled;

pub enum CompiledIteration<O, I, P> 
where
    O: Compiled<P, I>,
    I: Iteration<P>,
{
    Options((O, PhantomData<P>)),
    Iteration(I),
}

impl<O, I, P> CompiledIteration<O, I, P>
where
    O: Compiled<P, I>,
    I: Iteration<P>,
{
    /// Creates a new `CompiledIteration` with the specified options.
    /// 
    /// The iteration will be initialized lazily when `bind` or `evaluate_with_params` is called.
    /// 
    /// # Arguments
    /// * `options` - The compiled options to use for creating the iteration
    /// 
    /// # Returns
    /// A new `CompiledIteration` instance
    pub fn new(options: O) -> Self {
        Self::Options((options, PhantomData))
    }
}

impl<O, I, P> Iteration<P> for CompiledIteration<O, I, P>
where
    O: Compiled<P, I>,
    I: Iteration<P>,
{
    /// Binds parameters to the iteration, initializing it if necessary.
    /// 
    /// If the iteration hasn't been initialized yet, it will be created using the compiled options.
    /// 
    /// # Arguments
    /// * `params` - The parameters to bind to the iteration
    fn bind(&mut self, params: &P) {
        match self {
            Self::Options((options, _)) => {
                let iteration = options.compile(params);
                *self = Self::Iteration(iteration);
                self.bind(params);
            }
            Self::Iteration(iteration) => iteration.bind(params),
        }
    }

    /// Evaluates the iteration using previously bound parameters.
    /// 
    /// # Panics
    /// Panics if called before `bind` or `evaluate_with_params`
    fn evaluate(&mut self) {
        match self {
            Self::Options((_, _)) => panic!("CompiledIteration::evaluate called before bind"),
            Self::Iteration(iteration) => iteration.evaluate(),
        }
    }

    /// Evaluates the iteration asynchronously using previously bound parameters.
    /// 
    /// # Returns
    /// A vector of command buffers from the iteration
    /// 
    /// # Panics
    /// Panics if called before `bind` or `evaluate_with_params`
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
        match self {
            Self::Options((_, _)) => panic!("CompiledIteration::evaluate_async called before bind"),
            Self::Iteration(iteration) => iteration.evaluate_async(),
        }
    }

    /// Evaluates the iteration with explicitly provided parameters.
    /// 
    /// If the iteration hasn't been initialized yet, it will be created using the compiled options.
    /// 
    /// # Arguments
    /// * `params` - The parameters to use for evaluation
    fn evaluate_with_params(&mut self, params: &P) {
        match self {
            Self::Options((options, _)) => {
                let iteration = options.compile(params);
                *self = Self::Iteration(iteration);
                self.evaluate_with_params(params);
            }
            Self::Iteration(iteration) => iteration.evaluate_with_params(params),
        }
    }

    /// Evaluates the iteration asynchronously with explicitly provided parameters.
    /// 
    /// If the iteration hasn't been initialized yet, it will be created using the compiled options.
    /// 
    /// # Arguments
    /// * `params` - The parameters to use for evaluation
    /// 
    /// # Returns
    /// A vector of command buffers from the iteration
    fn evaluate_with_params_async(&mut self, params: &P) -> Vec<wgpu::CommandBuffer> {
        match self {
            Self::Options((options, _)) => {
                let iteration = options.compile(params);
                *self = Self::Iteration(iteration);
                self.evaluate_with_params_async(params)
            },
            Self::Iteration(iteration) => iteration.evaluate_with_params_async(params),
        }
    }
}
