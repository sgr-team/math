use crate::{StorageBuffer, WgpuContext};

impl StorageBuffer {
    /// Initializes a new storage buffer with the given data
    /// 
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `data` - The data to write
    pub fn init<T>(context: &WgpuContext, data: &[T]) -> Self 
    where
        T: bytemuck::Pod,
    {
        let result = Self::new::<T, _>(context, data.len());
        result.update_buffer_range(context, data, 0);
        result
    }

    /// Updates a range of the buffer with new data
    /// 
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `data` - The data to write
    /// * `start` - The starting index in elements of type T
    /// 
    /// # Panics
    /// * If the data slice is empty
    /// * If the start index would cause an overflow
    /// * If the data would write beyond the buffer's bounds
    pub fn update_buffer_range<T>(&self, context: &WgpuContext, data: &[T], start: usize)
    where
        T: bytemuck::Pod,
    {
        self.update_buffer_range_async(context, data, start);
        context.device.poll(wgpu::MaintainBase::Wait).unwrap();
    }

    /// Updates a range of the buffer with new data asynchronously
    /// 
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `data` - The data to write
    /// * `start` - The starting index in elements of type T
    /// 
    /// # Panics
    /// * If the data slice is empty
    /// * If the start index would cause an overflow
    /// * If the data would write beyond the buffer's bounds
    pub fn update_buffer_range_async<T>(&self, context: &WgpuContext, data: &[T], start: usize)
    where
        T: bytemuck::Pod,
    {
        assert!(!data.is_empty(), "Cannot update buffer with empty data");

        let byte_offset = start.checked_mul(std::mem::size_of::<T>())
            .expect("Start index overflow");
        let byte_size = data.len().checked_mul(std::mem::size_of::<T>())
            .expect("Data size overflow");
        
        assert!(
            byte_offset + byte_size <= self.size(),
            "Data would write beyond buffer bounds"
        );

        context.queue.write_buffer(
            &self.0,
            byte_offset as u64,
            bytemuck::cast_slice(data)
        );
    }
}