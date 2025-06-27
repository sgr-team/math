use std::{fs, path::Path};

use sgrmath_core::{OptimizationDirection, WgpuContext};
use sgrmath_ga::{GA, Options, continuous};
use sgrmath_pn::{Solution, PNP};

fn main() {
    let wgpu = WgpuContext::new();
    let pnp = PNP::from_csv(
        &wgpu, 
        3, 
        784, 
        std::fs::read_to_string("./.data/train.csv").unwrap(), 
        ','
    );

    println!("Permutation Neuron (genetic algorithm example)");
    println!("  examples_count: {}", pnp.examples_count);
    println!("  permutations_count: {}", pnp.permutations_count);
    println!("  vectors_count: {}", pnp.vectors_count);
    println!("  vector_length: {}", pnp.vector_length);
    println!("  outputs_count: {}", pnp.outputs_count);

    GA::new(
        &wgpu, 
        &Options {
            optimization_direction: OptimizationDirection::Maximize,
            population_size: 50,
            generation_size: 50,
            parents_count: 2,
            vector_length: 3 * 784,
            min_value: -255.0,
            max_value: 255.0,
        }
    )
        .problem(pnp.clone())
        .initializer(continuous::initializers::Random::new())
        .crossover(continuous::crossovers::BLXAlpha::new(2.0))
        .mutation(continuous::mutations::Random::new(0.02))
        .compile()
        .run(|ga, index| {
            let (mut min, mut max, mut med) = (f32::INFINITY, f32::NEG_INFINITY, 0.0);
            let (mut new_count, mut new_min, mut new_max, mut new_med) = (0, f32::INFINITY, f32::NEG_INFINITY, 0.0);
            let context = ga.context.borrow();
            let data = ga.data.borrow();
            let individuals = data.individuals.iter();
    
            for individual in individuals {
                let result = individual.result;
                if individual.generation == context.generation_index - 1 {
                    new_count += 1;
                    new_min = new_min.min(result);
                    new_max = new_max.max(result);
                    new_med += result;
                }
                
                min = min.min(result);
                max = max.max(result);
                med += result;
            }
            med = med / data.individuals.len() as f32;
            new_med = if new_count == 0 { 0.0 } else { new_med / new_count as f32 };
            let is_empty = new_count == 0;
    
            let best = ga.best();
            if best.generation == context.generation_index - 1 {
                save(&Solution::init(&pnp, ga.best_value()));
            }
    
            fn format_range(a: f32, b: f32, c: f32) -> String {
                format!("[{}..{}..{}]", a as u32, b as u32, c as u32)
            }
            println!(
                "Generation: {} \n    {}\n   {}",  
                index, 
                format_range(min, med, max), 
                if is_empty { 
                    " Empty".to_string() 
                } else { 
                    format!("+{} ({})", format_range(new_min, new_med, new_max), new_count) 
                }
            );
            true
        })
}

fn save(solution: &Solution) {
    let output_dir = Path::new(".output");
    if !output_dir.exists() {
        if let Err(e) = fs::create_dir_all(output_dir) {
            eprintln!("Error creating directory {:?}: {}", output_dir, e);
            return;
        }
    }
    
    let file_path = output_dir.join(format!("{}.json", solution.result));
    if file_path.exists() {
        return;
    }
    
    match serde_json::to_string_pretty(solution) {
        Ok(json_string) => fs::write(&file_path, json_string).unwrap(),
        Err(e) => panic!("Error serializing solution to JSON: {}", e),
    }
}
