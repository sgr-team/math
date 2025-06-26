use sgrmath_core::{ReadbackBuffer, StorageBuffer, WgpuContext};

macro_rules! problem_tests {
    ($shader:expr) => {
        #[test]
        fn evaluate_bounded() {
            let (context, solutions, results, readback) = super::prepare();

            let mut problem: Box<dyn sgrmath_core::Iteration<sgrmath_core::ProblemParams>> = Box::new(($shader)(&context));
            problem.bind(&sgrmath_core::ProblemParams {
                context: context.clone(),
                solutions: solutions.clone(),
                results: results.clone(),
                solutions_offset: 0,
                solutions_count: 10,
                vector_length: 1,
            });

            problem.evaluate();

            assert_eq!(
                readback.read::<f32>(&context, &results, 0, 10), 
                (0_i32..10).map(|i| if i < 5 { 2.0 * i as f32 + 1.0 } else { 10.0 }).collect::<Vec<_>>()
            );
        }

        #[test]
        fn evaluate_unbounded() {
            let (context, _, _, _) = super::prepare();

            let mut problem: Box<dyn sgrmath_core::Iteration<sgrmath_core::ProblemParams>> = Box::new(($shader)(&context));

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || { problem.evaluate(); }
            ));

            assert!(result.is_err());
        }

        #[test]
        fn evaluate_with_params() {
            let (context, solutions, results, readback) = super::prepare();

            let mut problem: Box<dyn sgrmath_core::Iteration<sgrmath_core::ProblemParams>> = Box::new(($shader)(&context));

            problem.evaluate_with_params(&sgrmath_core::ProblemParams {
                context: context.clone(),
                solutions: solutions.clone(),
                results: results.clone(),
                solutions_offset: 0,
                solutions_count: 10,
                vector_length: 1,
            });

            assert_eq!(
                readback.read::<f32>(&context, &results, 0, 10), 
                (0_i32..10).map(|i| if i < 5 { 2.0 * i as f32 + 1.0 } else { 10.0 }).collect::<Vec<_>>()
            );
        }

    };
}

mod cpu {
    problem_tests!(
        |_| sgrmath_core::CpuProblem::new(
            |solutions: Vec<f32>, _, params: &sgrmath_core::ProblemParams| {
                let mut results = vec![0.0; params.solutions_count];
                for i in 0..results.len() {
                    results[i] = if i < 5 { solutions[i] * 2.0 + 1.0 } else { 10.0 };
                }
                results
            },
            ()
        )
    );
}

mod shader {
    problem_tests!(
        |context| sgrmath_core::ShaderProblem::new(
            sgrmath_core::Shader::new(context, "test", include_str!("results.wgsl")),
            vec![]
        )
    );
}

pub fn prepare() -> (WgpuContext, StorageBuffer, StorageBuffer, ReadbackBuffer) {
    let context = WgpuContext::new();
    let solutions = StorageBuffer::new::<i32, _>(&context, 10);
    let results = StorageBuffer::new::<i32, _>(&context, 10);
    let readback = ReadbackBuffer::new::<i32, _>(&context, 10);

    solutions.update_buffer_range(&context, &(0_i32..10).map(|i| i as f32).collect::<Vec<_>>(), 0);
    results.update_buffer_range(&context, &(0_i32..10).map(|i| 2.0 * i as f32).collect::<Vec<_>>(), 0);

    (context, solutions, results, readback)
}
