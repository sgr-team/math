use std::{env, fs, io, process::Command};

use sgrmath_core::{ProblemParams, ReadbackBuffer, StorageBuffer, WgpuContext};
use sgrmath_pn::shaders::ShaderOptions;
use sgrmath_pn::{Solution, PNP};

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() < 2 {
        panic!("Usage: {} <solution_file>\nExample: {} ./.output/<best>.json", args[0], args[0]);
    }

    let solution_path = env::current_dir().unwrap().join(&args[1]);
    let solution: Solution = serde_json::from_str(
        fs::read_to_string(&solution_path)
            .unwrap()
            .as_str()
    ).unwrap();
    let temp_dir_buf = env::current_dir().unwrap().join(".temp");
    let temp_dir = temp_dir_buf.as_path();

    if fs::exists(temp_dir).unwrap() {
        if let Err(e) = fs::remove_dir_all(temp_dir) {
            panic!("Error removing temp directory {:?}: {}", temp_dir, e);
        }
    }
    if let Err(e) = fs::create_dir_all(temp_dir) {
        panic!("Error creating temp directory {:?}: {}", temp_dir, e);
    }

    process_csv(&WgpuContext::new(), "train.csv", &solution);
    process_csv(&WgpuContext::new(), "test.csv", &solution);

    let output = Command::new("python3")
        .arg("./examples/py_scripts/sln.py")
        .arg(".temp/train.csv")
        .arg(".temp/test.csv")
        .arg("--inputs")
        .arg("794")
        .output()?;
    println!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn process_csv(wgpu: &WgpuContext, filename: &str, solution: &Solution) {
    let file_path = env::current_dir().unwrap().join(".temp").join(filename);
    let pnp = PNP::from_csv(
        &wgpu, 
        3, 
        784, 
        std::fs::read_to_string(format!("./.data/{}", filename)).unwrap(), 
        ','
    );
    let (buffer_options, buffer_multiply, buffer_permutations, buffer_permutation_labels) = pnp.create_buffers(
        &pnp.wgpu, 
        &ShaderOptions::new(&pnp, 1)
    );
    let params = ProblemParams {
        context: wgpu.clone(),
        results: StorageBuffer::new::<f32, _>(&pnp.wgpu, 1),
        solutions_offset: 0,
        solutions_count: 1,
        vector_length: 784,
        solutions: StorageBuffer::new::<f32, _>(&pnp.wgpu, (pnp.vectors_count, pnp.vector_length)),
    };
    params.solutions.update_buffer_range(&pnp.wgpu, &solution.vectors, 0);

    pnp.evaluate_with_buffers(
        &params,
        (&buffer_options, &buffer_multiply, &buffer_permutations, &buffer_permutation_labels)
    );

    let reader = ReadbackBuffer::new::<f32, _>(&pnp.wgpu, (pnp.examples_count, params.vector_length));

    let examples = reader.read::<f32>(&pnp.wgpu, &pnp.examples, 0, pnp.examples_count * params.vector_length);
    let examples_labels = reader.read::<u32>(&pnp.wgpu, &pnp.labels, 0, pnp.examples_count);
    let permutation_labels = reader.read::<u32>(&pnp.wgpu, &buffer_permutation_labels, 0, pnp.permutations_count * pnp.outputs_count);
    let permutations = reader.read::<u32>(&pnp.wgpu, &buffer_permutations, 0, pnp.examples_count);

    let mut data = vec![];
    // add title
    let mut title = vec![ "label".to_string() ];
    for i in 0..10 {
        title.push(format!("O{i}"));
    }
    for i in 1..=28 {
        for j in 1..=28 {
            title.push(format!("{i}x{j}"));
        }
    }
    data.push(title.join(","));

    for e in 0..pnp.examples_count {
        let mut line = vec![ examples_labels[e].to_string() ];
        let permutation = permutations[e] as usize;

        let mut max_value = permutation_labels[permutation * pnp.outputs_count];
        for k in 0..pnp.outputs_count {
            max_value = max_value.max(permutation_labels[permutation * pnp.outputs_count + k]);
        }

        for k in 0..pnp.outputs_count {
            line.push(format!(
                "{}",
                (255.0 * (permutation_labels[permutation * pnp.outputs_count + k] as f32 / max_value as f32)) as u32
                // if permutation_labels[permutation * pnp.outputs_count + k] == max_value { 255 } else { 0 }
            ));
        }

        for i in (e * pnp.vector_length)..((e + 1) * pnp.vector_length) {
            line.push(format!("{}", examples[i]));
        }

        data.push(line.join(","));
    }

    fs::write(file_path, data.join("\n")).unwrap();
}