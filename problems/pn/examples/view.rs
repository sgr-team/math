use std::{collections::HashSet, env, fs};

use sgrmath_core::{WgpuContext};
use sgrmath_pn::{Solution, PNP};

const VECTOR_LENGTH: usize = 784;

fn main() {
    let args: Vec<_> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <solution_file>", args[0]);
        eprintln!("Example: {} ./.output/<best>.json", args[0]);
        std::process::exit(1);
    }
    let solution_path = env::current_dir().unwrap().join(&args[1]);
    let loaded_solution: Solution = serde_json::from_str(
        fs::read_to_string(&solution_path)
            .unwrap()
            .as_str()
    ).unwrap();

    let wgpu = WgpuContext::new();
    let pnp_train = PNP::from_csv(
        &wgpu, 
        loaded_solution.vectors_count, 
        VECTOR_LENGTH, 
        std::fs::read_to_string("./.data/train.csv").unwrap(), 
        ',',
        Some(255.0),
        None
    );
    let pnp_test = PNP::from_csv(
        &wgpu, 
        loaded_solution.vectors_count, 
        VECTOR_LENGTH, 
        std::fs::read_to_string("./.data/test.csv").unwrap(), 
        ',',
        Some(255.0),
        None
    );
    
    let solution_train = Solution::init_with_permutations(
        &pnp_train, 
        loaded_solution.vectors.clone(), 
        Solution::permutations_to_vec(&pnp_train, &loaded_solution.permutations)
    );
    let solution_test = Solution::init_with_permutations(
        &pnp_test, 
        loaded_solution.vectors, 
        Solution::permutations_to_vec(&pnp_test, &loaded_solution.permutations)
    );

    let mut used_permutations = 0;
    let mut used_outputs = HashSet::new();

    for p in 0..pnp_train.permutations_count {
        let start = p * pnp_train.outputs_count;

        let mut max = (0, 0);
        for o in 0..pnp_train.outputs_count {
            if let Some(value) = loaded_solution.permutations.get(&(start + o)) {
                if value > &max.1 {
                    max = (o, *value);
                }
            }
        }

        used_outputs.insert(max.0);
        if max.1 > 0 { used_permutations += 1; }
    }

    println!("Permutation Neuron Viewer");
    println!("  Used permutations: {}", used_permutations);
    println!("  Outputs ({}): {:?}", used_outputs.len(), used_outputs);
    println!("  Train result: {:.2}% ({})", solution_train.result as f32 / pnp_train.examples_count as f32 * 100.0, solution_train.result);
    println!("  Test result:  {:.2}% ({})", solution_test.result as f32 / pnp_test.examples_count as f32 * 100.0, solution_test.result);
}
