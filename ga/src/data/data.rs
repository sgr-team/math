use std::marker::PhantomData;

use bytemuck::Pod;
use sgrmath_core::{OptimizationDirection, ReadbackBuffer, StorageBuffer, WgpuContext};

use crate::{Context, Individual, Options};

/// Data structure for genetic algorithm.
///
/// This struct holds the data needed for genetic algorithm operations,
/// including the population buffer, individuals, and their results.
///
/// # Type Parameters
/// * `T` - The type of data being processed, must implement `Pod` for GPU compatibility
#[derive(Debug)]
pub struct Data<T> 
where
    T: Pod
{
    /// Phantom data to hold the type parameter
    _t: PhantomData<T>,
    /// Buffer containing the population data
    pub population: StorageBuffer,
    /// Buffer for storing intermediate results
    pub next: StorageBuffer,
    /// Buffer for storing parents
    pub parents: StorageBuffer,
    /// Buffer for storing results
    pub results: StorageBuffer,
    /// Readable buffer for reading data
    pub reader: ReadbackBuffer,
    /// Vector of individuals in the population
    pub individuals: Vec<Individual>,
}

impl<T> Data<T> 
where
    T: Pod
{
    /// Creates a new data instance.
    ///
    /// # Arguments
    /// * `wgpu` - The WGPU context used for GPU operations
    /// * `options` - Configuration options for the genetic algorithm
    ///
    /// # Returns
    /// A new `Data` instance
    pub fn new(wgpu: &WgpuContext, options: &Options) -> Self {
        Self { 
            _t: PhantomData,
            population: StorageBuffer::new::<T, _>(wgpu, (options.population_size, options.vector_length)),
            next: StorageBuffer::new::<T, _>(wgpu, (options.generation_size, options.vector_length)),
            parents: StorageBuffer::new::<T, _>(wgpu, (options.generation_size, options.parents_count)),
            results: StorageBuffer::new::<T, _>(wgpu, options.generation_size),
            reader: ReadbackBuffer::new::<T, _>(wgpu, (options.generation_size, options.parents_count)),
            individuals: Vec::with_capacity(options.population_size),
        }
    }

    /// Updates the population with new individuals.
    ///
    /// This method performs two main operations:
    /// 1. Copies new individuals from the next buffer to the population buffer
    /// 2. Updates the individuals vector with the new individuals' data
    ///
    /// # Buffer Mapping
    /// - Population buffer: individuals are stored at their population_index
    /// - Next buffer: individuals are stored at position ((individual.id - population_size) % generation_size)
    ///   The offset in id is due to the initial generation being generated with population_size
    ///
    /// # Arguments
    /// * `context` - The genetic algorithm context containing GPU resources
    /// * `options` - The genetic algorithm configuration options
    /// * `new_individuals` - Vector of tuples containing (population_index, individual) pairs
    pub fn update_population(
        &mut self, 
        context: &mut Context, 
        new_individuals: Vec<(usize,Individual)>
    ) {
        if new_individuals.len() == 0 {
            return;
        }

        let mut encoder = context.wgpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Update Population Encoder"),
        });
        let vector_length = context.options.vector_length;

        for (index, individual) in new_individuals.into_iter() {
            // The offset in id is due to the initial generation being generated with population_size
            let next_index = (individual.id - context.options.population_size) % context.options.generation_size;
            encoder.copy_buffer_to_buffer(
                &self.next,
                next_index as u64 * vector_length as u64 * std::mem::size_of::<T>() as u64,
                &self.population,
                index as u64 * vector_length as u64 * std::mem::size_of::<T>() as u64,
                vector_length as u64 * std::mem::size_of::<T>() as u64,
            );

            self.individuals[index] = individual;
        }
        
        context.wgpu.queue.submit(Some(encoder.finish()));
    }

    /// Reads the next generation from the results and parents buffers.
    ///
    /// # Arguments
    /// * `context` - The context of the genetic algorithm
    /// * `options` - The options of the genetic algorithm
    ///
    /// # Returns
    /// A vector of individuals
    pub fn read_generation(&mut self, context: &mut Context) -> Vec<Individual> {
        let parents_size = context.options.generation_size * context.options.parents_count;
        self.reader.scale::<u32, _>(&context.wgpu, parents_size);
        
        let parents = self.reader.read::<u32>(&context.wgpu, &self.parents, 0, parents_size);
        let results = self.reader.read::<f32>(&context.wgpu, &self.results, 0, context.options.generation_size);

        let mut individuals = Vec::with_capacity(context.options.generation_size);
        for (index, result) in results.into_iter().enumerate() {
            individuals.push(Individual { 
                id: context.next_id + index, 
                generation: context.generation_index, 
                parents: parents
                    .iter()
                    .skip(index * context.options.parents_count)
                    .take(context.options.parents_count)
                    .map(|x| *x as usize)
                    .collect(), 
                result
            });
        }

        individuals
    }

    /// Reads an individual from the population buffer.
    ///
    /// # Arguments
    /// * `context` - The context of the genetic algorithm
    /// * `options` - The options of the genetic algorithm
    /// * `index` - The index of the individual to read
    pub fn read_individual(&self, context: &Context, index: usize) -> Vec<T> {
        ReadbackBuffer::new::<T, _>(&context.wgpu, context.options.vector_length)
            .read::<T>(
                &context.wgpu, 
                &self.population, index * context.options.vector_length, 
                context.options.vector_length
            )
    }

    /// Finds the best individual in the population.
    ///
    /// # Arguments
    /// * `direction` - The direction of the optimization
    ///
    /// # Returns
    /// The index and result of the best individual
    pub fn best(&self, direction: &OptimizationDirection) -> Option<(usize, f32)> {
        self.individuals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| direction.compare(&a.result, &b.result))
            .map(|(index, individual)| (index, individual.result))
    }
}
