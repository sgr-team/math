use super::Iteration;

pub struct NotImplementedIteration(String);

impl NotImplementedIteration {
    pub fn new<S>(message: S) -> Self 
    where
        S: Into<String>
    {
        Self(message.into())
    }
}

impl<T> Iteration<T> for NotImplementedIteration {
    fn bind(&mut self, _params: &T) { panic!("{}", self.0); }
    fn evaluate(&mut self) { panic!("{}", self.0); }
    fn evaluate_with_params(&mut self, _params: &T) { panic!("{}", self.0); }
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { panic!("{}", self.0); }
    fn evaluate_with_params_async(&mut self, _params: &T) -> Vec<wgpu::CommandBuffer> { panic!("{}", self.0); }
}