use std::{env, fs};

use sgrmath_core::{WgpuContext};
use sgrmath_pn::{Solution, PNP};

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
        784, 
        std::fs::read_to_string("./.data/train.csv").unwrap(), 
        ','
    );
    let pnp_test = PNP::from_csv(
        &wgpu, 
        loaded_solution.vectors_count, 
        784, 
        std::fs::read_to_string("./.data/test.csv").unwrap(), 
        ','
    );
    
    let solution_train = Solution::init_with_permutations(
        &pnp_train, 
        loaded_solution.vectors.clone(), 
        loaded_solution.permutations.clone()
    );
    let solution_test = Solution::init_with_permutations(
        &pnp_test, 
        loaded_solution.vectors, 
        loaded_solution.permutations
    );

    println!("Permutation Neuron Viewer");
    println!("Train result: {}", solution_train.result);
    println!("Test result: {}", solution_test.result);
}
