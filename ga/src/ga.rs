use std::{cell::RefCell, rc::Rc};

use bytemuck::Pod;
use sgrmath_core::{CompiledIteration, Iteration, NotImplementedIteration, ProblemParams, ReadbackBuffer, StorageBuffer, WgpuContext};

use crate::{Context, Data, Individual, IterationParams, Options};
use crate::common;

/// Genetic Algorithm implementation with GPU acceleration.
///
/// This struct represents a genetic algorithm that can be used to solve optimization problems.
/// It uses GPU acceleration through WGPU for improved performance.
pub struct GA<T> 
where
    T: Pod
{
    /// The GA context
    pub context: Rc<RefCell<Context>>,
    /// The GA data
    pub data: Rc<RefCell<Data<T>>>,
    /// The problem to be solved
    pub problem: Box<dyn Iteration<ProblemParams>>,
    /// The GA options
    pub options: Options,
    /// The initializer for the first generation
    pub initializer: Box<dyn Iteration<IterationParams<T>>>,
    /// Parents selection strategy
    pub parents: Box<dyn Iteration<IterationParams<T>>>,
    /// Crossover operation
    pub crossover: Box<dyn Iteration<IterationParams<T>>>,
    /// Mutation operation
    pub mutation: Box<dyn Iteration<IterationParams<T>>>,
    /// Selection strategy
    pub selector: Box<dyn Iteration<IterationParams<T>>>,
}

impl<T> GA<T> 
where
    T: Pod
{
    /// Creates a new genetic algorithm instance.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used for GPU operations
    /// * `options` - Configuration options for the genetic algorithm
    ///
    /// # Returns
    /// A new `GA` instance
    pub fn new(context: &WgpuContext, options: &Options) -> Self {
        Self {
            context: Rc::new(RefCell::new(Context::new(context, options))),
            data: Rc::new(RefCell::new(Data::new(context, options))),
            problem: Box::new(NotImplementedIteration::new("problem")),
            initializer: Box::new(NotImplementedIteration::new("initializer")),
            parents: Box::new(CompiledIteration::new(common::parents::Random::new())),
            crossover: Box::new(NotImplementedIteration::new("crossover")),
            mutation: Box::new(NotImplementedIteration::new("mutation")),
            selector: Box::new(CompiledIteration::new(common::selectors::Default::new())),
            options: options.clone(),
        }
    }

    /// Runs the genetic algorithm.
    ///
    /// # Arguments
    /// * `f` - A function that takes a reference to the GA and an index, and returns a boolean indicating whether to continue running
    pub fn run<F>(&mut self, f: F)
    where
        F: Fn(&mut Self, usize) -> bool
    {
        let mut index = 0;
        loop {
            self.generation();
            index += 1;

            if !f(self, index) {
                break;
            }
        }
    }

    /// Sets the problem to be solved.
    ///
    /// # Arguments
    /// * `problem` - The problem options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn problem<P>(mut self, problem: P) -> Self
    where
        P: Iteration<ProblemParams> + 'static,
    {
        self.problem = Box::new(problem);
        self
    }

    /// Sets the initializer for the first generation.
    ///
    /// # Arguments
    /// * `initializer` - The initializer options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn initializer<I>(mut self, initializer: I) -> Self
    where
        I: Iteration<IterationParams<T>> + 'static,
    {
        self.initializer = Box::new(initializer);
        self
    }

    /// Sets the parents selection strategy.
    ///
    /// # Arguments
    /// * `parents` - The parents selection options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn parents<P>(mut self, parents: P) -> Self
    where
        P: Iteration<IterationParams<T>> + 'static,
    {
        self.parents = Box::new(parents);
        self
    }

    /// Sets the crossover operation.
    ///
    /// # Arguments
    /// * `crossover` - The crossover options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn crossover<C>(mut self, crossover: C) -> Self
    where
        C: Iteration<IterationParams<T>> + 'static,
    {
        self.crossover = Box::new(crossover);
        self
    }

    /// Sets the mutation operation.
    ///
    /// # Arguments
    /// * `mutation` - The mutation options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn mutation<M>(mut self, mutation: M) -> Self
    where
        M: Iteration<IterationParams<T>> + 'static,
    {
        self.mutation = Box::new(mutation);
        self
    }

    /// Sets the selection strategy.
    ///
    /// # Arguments
    /// * `selector` - The selection options
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn selector<S>(mut self, selector: S) -> Self
    where
        S: Iteration<IterationParams<T>> + 'static,
    {
        self.selector = Box::new(selector);
        self
    }

    /// Returns the best individual.
    /// 
    /// # Panics
    /// Panics if no best individual exists (method generation was not called).
    /// The panic message will be "GA not initialized".
    /// 
    /// # Examples
    /// ```
    /// use sgrmath_ga::GA;
    /// 
    /// fn example(ga: &GA<f32>) {
    ///     let best = ga.best();
    ///     println!("Best individual ID: {}", best.id);
    /// }
    /// ```
    pub fn best(&self) -> Individual {
        self.best_safe().expect("best: GA not initialized")
    }

    /// Returns the best individual, if it exists.
    /// 
    /// This is a safe version of [`best()`] that returns `None` if no best individual exists.
    /// 
    /// # Examples
    /// ```
    /// use sgrmath_ga::GA;
    /// 
    /// fn example(ga: &GA<f32>) {
    ///     match ga.best_safe() {
    ///         Some(best) => println!("Best individual ID: {}", best.id),
    ///         None => println!("No best individual yet"),
    ///     }
    /// }
    /// ```
    pub fn best_safe<'a>(&self) -> Option<Individual> {
        let data = self.data.borrow();
        let context = self.context.borrow();
        
        match data.best(&context.options.optimization_direction) {
            Some((index, _)) => Some(data.individuals[index].clone()),
            None => None
        }
    }

    /// Returns the value of the best individual.
    /// 
    /// # Panics
    /// Panics if no best individual exists (which should never happen in normal operation).
    /// The panic message will be "Best individual not found".
    /// 
    /// # Examples
    /// ```
    /// use sgrmath_ga::GA;
    /// 
    /// fn example(ga: &GA<f32>) {
    ///     let best_value = ga.best_value();
    ///     println!("Best value: {:?}", best_value);
    /// }
    /// ```
    pub fn best_value(&self) -> Vec<T> {
        self.best_value_safe().expect("Best individual not found")
    }

    /// Returns the value of the best individual, if it exists.
    /// 
    /// This is a safe version of [`best_value()`] that returns `None` if no best individual exists.
    /// 
    /// # Examples
    /// ```
    /// use sgrmath_ga::GA;
    /// 
    /// fn example(ga: &GA<f32>) {
    ///     match ga.best_value_safe() {
    ///         Some(value) => println!("Best value: {:?}", value),
    ///         None => println!("No best value yet"),
    ///     }
    /// }
    /// ```
    pub fn best_value_safe(&self) -> Option<Vec<T>> {
        let data = self.data.borrow();
        let context = self.context.borrow();
        
        match data.best(&context.options.optimization_direction) {
            Some((index, _)) => Some(data.read_individual(&context, index)),
            None => None
        }
    }

    /// Compiles the genetic algorithm by binding all components to their parameters.
    ///
    /// This method should be called after setting up all components (initializer, parents, crossover, etc.).
    ///
    /// # Returns
    /// `&mut Self` for method chaining
    pub fn compile(mut self) -> Self {
        let (wgpu, options, next, results) = {
            let context = self.context.borrow();
            let data = self.data.borrow();

            (context.wgpu.clone(), context.options.clone(), data.next.clone(), data.results.clone())
        };

        let params = IterationParams::new(self.context.clone(), self.data.clone(), options.generation_size);
        let problem_params = ProblemParams { 
            context: wgpu.clone(), 
            solutions: next, 
            results: results, 
            solutions_offset: 0,
            solutions_count: options.generation_size, 
            vector_length: options.vector_length 
        };
        
        self.parents.bind(&params);
        self.crossover.bind(&params);
        self.mutation.bind(&params);
        self.problem.bind(&problem_params);
        self.selector.bind(&params);

        self
    }

    /// Runs a single generation of the genetic algorithm.
    pub fn generation(&mut self) {
        match self.is_initialized() {
            true => self.generation_next(),
            false => self.generation_init()
        }
    }

    fn generation_next(&mut self) {
        self.parents.evaluate();
        self.crossover.evaluate();
        self.mutation.evaluate();
        self.problem.evaluate();
        self.selector.evaluate();

        let mut context = self.context.borrow_mut();

        context.generation_index += 1;
        context.next_id += context.options.generation_size;
    }

    fn generation_init(&mut self) {
        let (wgpu, population, options) = {
            let context = self.context.borrow();
            let data = self.data.borrow();

            (context.wgpu.clone(), data.population.clone(), context.options.clone())
        };

        // Create result and readback buffer (population size can be more than generation size)
        let results_buffer = StorageBuffer::new::<T, _>(&wgpu, options.population_size);
        let readback_buffer = ReadbackBuffer::new::<f32, _>(&wgpu, options.population_size);

        let problem_params = ProblemParams { 
            context: wgpu.clone(), 
            solutions: population.clone(), 
            results: results_buffer.clone(), 
            solutions_offset: 0,
            solutions_count: options.generation_size, 
            vector_length: options.vector_length 
        };

        self.initializer.evaluate_with_params(&IterationParams::new(
            self.context.clone(), 
            self.data.clone(), 
            options.generation_size
        ));
        self.problem.evaluate_with_params(&problem_params);

        let mut context = self.context.borrow_mut();
        let mut data = self.data.borrow_mut();

        data.individuals = readback_buffer
            .read::<f32>(&wgpu, &results_buffer, 0, options.population_size)
            .into_iter()
            .enumerate()
            .map(|(i, result)| Individual { 
                id: context.next_id + i, 
                generation: 0, 
                parents: vec![], 
                result 
            })
            .collect();

        context.generation_index += 1;
        context.next_id += context.options.population_size;
        context.is_initialized = true;
    }

    fn is_initialized(&self) -> bool {
        let context = self.context.borrow();
        context.is_initialized
    }
}
