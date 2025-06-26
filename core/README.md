# SGR Math Core

## About

SGR Math Core is a foundational library that provides GPU acceleration infrastructure through WebGPU (wgpu) shaders. This library serves as a core component for implementing high-performance mathematical algorithms, particularly optimization algorithms like:

- Genetic Algorithms
- Ant Colony Optimization
- Simulated Annealing

The library is built in Rust and uses wgpu for GPU-accelerated computations, making it ideal for computationally intensive mathematical operations and optimization problems. By utilizing GPU shaders, the library can process large datasets and perform complex calculations with significantly improved performance compared to CPU-only implementations.

### Key Features

- GPU-accelerated mathematical computations
- Modern WebGPU (wgpu) implementation
- Rust-based core for performance and safety
- Modular architecture for easy extension
- Efficient buffer management for GPU operations
- Shader compilation and management
- GPU memory management utilities

### Technical Stack

- **Language**: Rust
- **GPU Framework**: wgpu (WebGPU implementation)
- **Async Runtime**: Tokio
- **Memory Management**: bytemuck for safe memory operations

This library is designed as a foundation for developers who need to implement high-performance mathematical algorithms with GPU acceleration. It provides the necessary infrastructure and utilities for GPU operations, allowing you to focus on implementing your specific algorithms while leveraging the power of GPU computing.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
sgrmath_core = "0.1.0"
```

## Documentation

The complete API documentation is available on [docs.rs](https://docs.rs/sgrmath_core).

## Usage

Here's a basic example of how to use the library:

```rust
use sgrmath_core::{WgpuContext, Model};

async fn example() {
    // Initialize the GPU context
    let context = WgpuContext::new().await;
    
    // Use the context to manage GPU resources
    // and implement your specific algorithms
}
```

## Features

The library provides several key features:

### GPU Infrastructure
- Parallel processing capabilities
- Efficient memory management for GPU buffers
- Shader compilation and management
- GPU resource handling
- Buffer management utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
