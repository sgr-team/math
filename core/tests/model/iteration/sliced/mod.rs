use std::{ops::Range, rc::Rc, cell::RefCell};

use sgrmath_core::{Iteration, IterationSize, NotImplementedIteration, SlicedIteration, Sliced};

#[test]
fn distribute() {
    assert_eq!(
        SlicedIteration::<TestParams>::new()
            .add(1, Box::new(NotImplementedIteration::new("1")))
            .add(9, Box::new(NotImplementedIteration::new("2")))
            .add(IterationSize::Proportional(1.0), Box::new(NotImplementedIteration::new("3")))
            .add(IterationSize::Proportional(1.0), Box::new(NotImplementedIteration::new("4")))
            .distribute(20), 
        &[ 1, 9, 5, 5 ]
    );
    assert_eq!(
        SlicedIteration::<TestParams>::new()
            .add(1, Box::new(NotImplementedIteration::new("1")))
            .add(9, Box::new(NotImplementedIteration::new("2")))
            .add(IterationSize::Proportional(1.0), Box::new(NotImplementedIteration::new("3")))
            .add(IterationSize::Proportional(1.0), Box::new(NotImplementedIteration::new("4")))
            .distribute(17), 
        &[ 1, 9, 3, 4 ]
    );
    assert_eq!(
        SlicedIteration::<TestParams>::new()
            .add(1, Box::new(NotImplementedIteration::new("1")))
            .add(9, Box::new(NotImplementedIteration::new("2")))
            .add(IterationSize::Proportional(1.0), Box::new(NotImplementedIteration::new("3")))
            .add(IterationSize::Proportional(2.0), Box::new(NotImplementedIteration::new("4")))
            .distribute(17), 
        &[ 1, 9, 2, 5 ]
    );
}

#[test]
fn bind() {
    let (params, _, mut iteration) = prepare();
    iteration.bind(&TestParams { start: 0, end: 20 });
    
    assert_eq!(
        params.borrow().iter().map(|p| p.start..p.end).collect::<Vec<_>>(),
        vec![ 0..1, 1..10, 10..13, 13..20 ]
    );
}

#[test]
fn evaluate() {
    let (_, evaluated, mut iteration) = prepare();
    iteration.bind(&TestParams { start: 0, end: 20 });
    iteration.evaluate();
    
    assert_eq!(
        evaluated.borrow().iter().map(|s| s.clone()).collect::<Vec<_>>(),
        vec![ "1", "2", "3", "4" ]
    );
}

#[test]
fn evaluate_with_params() {
    let (params, evaluated, mut iteration) = prepare();
    iteration.evaluate_with_params(&TestParams { start: 11, end: 31 });
    
    assert_eq!(
        params.borrow().iter().map(|p| p.start..p.end).collect::<Vec<_>>(),
        vec![ 11..12, 12..21, 21..24, 24..31 ]
    );
    assert_eq!(
        evaluated.borrow().iter().map(|s| s.clone()).collect::<Vec<_>>(),
        vec![ "1", "2", "3", "4" ]
    );
}

#[test]
fn evaluate_async() {
    let (_, evaluated, mut iteration) = prepare();
    iteration.bind(&TestParams { start: 0, end: 20 });
    iteration.evaluate_async();
    
    assert_eq!(
        evaluated.borrow().iter().map(|s| s.clone()).collect::<Vec<_>>(),
        vec![ "1", "2", "3", "4" ]
    );
}

#[test]
fn evaluate_with_params_async() {
    let (params, evaluated, mut iteration) = prepare();
    iteration.evaluate_with_params_async(&TestParams { start: 11, end: 31 });
    
    assert_eq!(
        params.borrow().iter().map(|p| p.start..p.end).collect::<Vec<_>>(),
        vec![ 11..12, 12..21, 21..24, 24..31 ]
    );
    assert_eq!(
        evaluated.borrow().iter().map(|s| s.clone()).collect::<Vec<_>>(),
        vec![ "1", "2", "3", "4" ]
    );
}

fn prepare() -> (Rc<RefCell<Vec<TestParams>>>, Rc<RefCell<Vec<String>>>, SlicedIteration<TestParams>) {
    let params = Rc::new(RefCell::new(vec![]));
    let evaluated = Rc::new(RefCell::new(vec![]));
    let iteration = SlicedIteration::<TestParams>::new()
        .add(1, Box::new(TestIteration::new("1", params.clone(), evaluated.clone())))
        .add(9, Box::new(TestIteration::new("2", params.clone(), evaluated.clone())))
        .add(IterationSize::Proportional(1.0), Box::new(TestIteration::new("3", params.clone(), evaluated.clone())))
        .add(IterationSize::Proportional(2.0), Box::new(TestIteration::new("4", params.clone(), evaluated.clone())));

    (params, evaluated, iteration)
}

#[derive(Clone)]
struct TestParams {
    start: usize,
    end: usize,
}

struct TestIteration {
    name: String,
    params: Rc<RefCell<Vec<TestParams>>>,
    evaluated: Rc<RefCell<Vec<String>>>,
}

impl TestIteration {
    fn new<S>(
        name: S,
        params: Rc<RefCell<Vec<TestParams>>>, 
        evaluated: Rc<RefCell<Vec<String>>>) -> Self 
    where 
        S: Into<String>
    {
        Self { 
            name: name.into(), 
            params, 
            evaluated 
        }
    }
}

impl Iteration<TestParams> for TestIteration {
    fn bind(&mut self, params: &TestParams) { 
        self.params.borrow_mut().push(params.clone()); 
    }
    
    fn evaluate(&mut self) { 
        self.evaluated.borrow_mut().push(self.name.clone());
    }

    fn evaluate_with_params(&mut self, params: &TestParams) { 
        self.params.borrow_mut().push(params.clone()); 
        self.evaluated.borrow_mut().push(self.name.clone());
    }
    
    fn evaluate_async(&mut self) -> Vec<wgpu::CommandBuffer> { 
        self.evaluated.borrow_mut().push(self.name.clone());
        vec![] 
    }
    
    fn evaluate_with_params_async(&mut self, params: &TestParams) -> Vec<wgpu::CommandBuffer> { 
        self.params.borrow_mut().push(params.clone()); 
        self.evaluated.borrow_mut().push(self.name.clone());
        vec![] 
    }
}

impl Sliced for TestParams {
    fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    fn set_range(&mut self, range: Range<usize>) {
        self.start = range.start;
        self.end = range.end;
    }
}
