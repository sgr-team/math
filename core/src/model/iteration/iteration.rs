/// A trait representing a single iteration step in a computation process.
///
/// This trait defines the interface for iteration steps that can be bound to parameters
/// and evaluated either with bound parameters or with explicitly provided ones.
///
/// # Type Parameters
/// * `T` - The type of parameters used for binding and evaluation
///
/// # Examples
/// ```
/// use sgrmath_core::Iteration;
///
/// #[derive(Clone)]
/// struct Params {
///     value: f32,
/// }
///
/// #[derive(Default)]
/// struct MyOptimizer {
///     bound_params: Option<Params>,
/// }
///
/// impl MyOptimizer {
///     fn new() -> Self {
///         Self::default()
///     }
/// }
///
/// impl Iteration<Params> for MyOptimizer {
///     fn bind(&mut self, params: &Params) {
///         // Store parameters for later use
///     }
///
///     fn evaluate(&mut self) {
///         // Evaluate with bound parameters
///     }
///
///     fn evaluate_with_params(&mut self, params: &Params) {
///         // Evaluate with explicitly provided parameters
///     }
///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
///         // Evaluate asynchronously
///         vec![]
///     }
/// 
///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> {
///         // Evaluate asynchronously with explicitly provided parameters
///         vec![]
///     }
/// }
/// ```
pub trait Iteration<T> {
    /// Binds the parameters to this iteration step.
    ///
    /// This method stores the parameters for later use in `evaluate`.
    /// The bound parameters will be used when `evaluate` is called.
    ///
    /// # Arguments
    /// * `params` - The parameters to bind
    ///
    /// # Examples
    /// ```
    /// use sgrmath_core::Iteration;
    ///
    /// #[derive(Clone)]
    /// struct Params {
    ///     value: f32,
    /// }
    ///
    /// #[derive(Default)]
    /// struct MyOptimizer {
    ///     bound_params: Option<Params>,
    /// }
    ///
    /// impl MyOptimizer {
    ///     fn new() -> Self {
    ///         Self::default()
    ///     }
    /// }
    ///
    /// impl Iteration<Params> for MyOptimizer {
    ///     fn bind(&mut self, _params: &Params) {
    ///         // Store parameters for later use
    ///     }
    ///
    ///     fn evaluate(&mut self) {
    ///         // Evaluate with bound parameters
    ///     }
    ///
    ///     fn evaluate_with_params(&mut self, _params: &Params) {
    ///         // Evaluate with explicitly provided parameters
    ///     }
    /// 
    ///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> {
    ///         // Evaluate asynchronously
    ///         vec![]
    ///     }
    /// 
    ///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> {
    ///         // Evaluate asynchronously with explicitly provided parameters
    ///         vec![]
    ///     }
    /// }
    ///
    /// let mut optimizer = MyOptimizer::new();
    /// optimizer.bind(&Params { value: 42.0 });
    /// ```
    fn bind(&mut self, params: &T);

    /// Evaluates the iteration step using previously bound parameters.
    ///
    /// This method should be called after `bind` to perform the actual computation.
    /// It uses the parameters that were previously bound using the `bind` method.
    ///
    /// # Examples
    /// ```
    /// use sgrmath_core::Iteration;
    ///
    /// #[derive(Clone)]
    /// struct Params {
    ///     value: f32,
    /// }
    ///
    /// #[derive(Default)]
    /// struct MyOptimizer {
    ///     bound_params: Option<Params>,
    /// }
    ///
    /// impl MyOptimizer {
    ///     fn new() -> Self {
    ///         Self::default()
    ///     }
    /// }
    ///
    /// impl Iteration<Params> for MyOptimizer {
    ///     fn bind(&mut self, _params: &Params) {}
    ///     fn evaluate(&mut self) {}
    ///     fn evaluate_with_params(&mut self, _params: &Params) {}
    ///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { vec![] }
    ///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> { vec![] }
    /// }
    ///
    /// let mut optimizer = MyOptimizer::new();
    /// optimizer.bind(&Params { value: 42.0 });
    /// optimizer.evaluate();
    /// ```
    fn evaluate(&mut self);

    /// Evaluates asynchronously the iteration step using previously bound parameters.
    ///
    /// This method should be called after `bind` to perform the actual computation.
    /// It uses the parameters that were previously bound using the `bind` method.
    ///
    /// # Examples
    /// ```
    /// use sgrmath_core::{Iteration, WgpuContext};
    ///
    /// #[derive(Clone)]
    /// struct Params {
    ///     value: f32,
    /// }
    ///
    /// #[derive(Default)]
    /// struct MyOptimizer {
    ///     bound_params: Option<Params>,
    /// }
    ///
    /// impl MyOptimizer {
    ///     fn new() -> Self {
    ///         Self::default()
    ///     }
    /// }
    ///
    /// impl Iteration<Params> for MyOptimizer {
    ///     fn bind(&mut self, _params: &Params) {}
    ///     fn evaluate(&mut self) {}
    ///     fn evaluate_with_params(&mut self, _params: &Params) {}
    ///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { vec![] }
    ///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> { vec![] }
    /// }
    /// 
    /// fn evaluate_async_and_wait(context: &WgpuContext) {
    ///     let mut optimizer = MyOptimizer::new();
    ///     optimizer.bind(&Params { value: 42.0 });
    ///     let command_buffers = optimizer.evaluate_async();
    ///     context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    /// }
    /// ```
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer>;

    /// Evaluates the iteration step with explicitly provided parameters.
    ///
    /// This method performs the computation using the provided parameters,
    /// regardless of any previously bound parameters.
    ///
    /// # Arguments
    /// * `params` - The parameters to use for evaluation
    ///
    /// # Examples
    /// ```
    /// use sgrmath_core::Iteration;
    ///
    /// #[derive(Clone)]
    /// struct Params {
    ///     value: f32,
    /// }
    ///
    /// #[derive(Default)]
    /// struct MyOptimizer {
    ///     bound_params: Option<Params>,
    /// }
    ///
    /// impl MyOptimizer {
    ///     fn new() -> Self {
    ///         Self::default()
    ///     }
    /// }
    ///
    /// impl Iteration<Params> for MyOptimizer {
    ///     fn bind(&mut self, _params: &Params) {}
    ///     fn evaluate(&mut self) {}
    ///     fn evaluate_with_params(&mut self, _params: &Params) {}
    ///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { vec![] }
    ///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> { vec![] }
    /// }   
    ///
    /// let mut optimizer = MyOptimizer::new();
    /// optimizer.evaluate_with_params(&Params { value: 42.0 });
    /// ```
    fn evaluate_with_params(&mut self, params: &T);

    /// Evaluates the iteration step with explicitly provided parameters.
    ///
    /// This method performs the computation using the provided parameters,
    /// regardless of any previously bound parameters.
    ///
    /// # Arguments
    /// * `params` - The parameters to use for evaluation
    ///
    /// # Examples
    /// ```
    /// use sgrmath_core::{Iteration, WgpuContext};
    ///
    /// #[derive(Clone)]
    /// struct Params {
    ///     value: f32,
    /// }
    ///
    /// #[derive(Default)]
    /// struct MyOptimizer {
    ///     bound_params: Option<Params>,
    /// }
    ///
    /// impl MyOptimizer {
    ///     fn new() -> Self {
    ///         Self::default()
    ///     }
    /// }
    ///
    /// impl Iteration<Params> for MyOptimizer {
    ///     fn bind(&mut self, _params: &Params) {}
    ///     fn evaluate(&mut self) {}
    ///     fn evaluate_with_params(&mut self, _params: &Params) {}
    ///     fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { vec![] }
    ///     fn evaluate_with_params_async(&mut self, _params: &Params) -> Vec<wgpu::CommandBuffer> { vec![] }
    /// }   
    ///
    /// async fn evaluate_with_params_async_and_wait(context: &WgpuContext) {
    ///     let mut optimizer = MyOptimizer::new();
    ///     optimizer.bind(&Params { value: 42.0 });
    ///     context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    /// }
    /// ```
    fn evaluate_with_params_async(&mut self, params: &T) -> Vec<wgpu::CommandBuffer>;
}
