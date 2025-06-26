use wgpu::Buffer;

use crate::{ReadbackBuffer, WgpuContext};

impl ReadbackBuffer {
    /// Scales the buffer to a new size if needed
    /// 
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `size` - The new size in elements of type T
    /// 
    /// # Returns
    /// * true if the buffer was scaled, false otherwise
    /// 
    /// # Panics
    /// * If the buffer size would overflow
    pub fn scale<T, S>(&mut self, context: &crate::WgpuContext, size: S) -> bool
    where
        T: bytemuck::Pod,
        S: Into<crate::Size>,
    {
        let new_len = size.into().len();
        let new_size = new_len.checked_mul(std::mem::size_of::<T>())
            .expect("Buffer size overflow");

        if new_size > self.size() {
            self.0 = Self::create_buffer::<T>(context, new_len as u64);
            return true;
        }

        false
    }

    /// Copies data from source buffer and reads it
    /// 
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `source` - The source buffer to copy from
    /// * `start` - The starting index in elements of type T
    /// * `len` - The number of elements to read
    /// 
    /// # Returns
    /// A vector containing the read data
    /// 
    /// # Panics
    /// * If start + len would cause an integer overflow
    /// * If start + len is beyond the end of the source buffer
    /// * If the buffer mapping fails
    #[must_use]
    pub fn read<T>(&self, context: &WgpuContext, source: &Buffer, start: usize, len: usize) -> Vec<T>
    where
        T: bytemuck::Pod,
    {
        let byte_start = start.checked_mul(std::mem::size_of::<T>())
            .map(|x| x as u64)
            .expect("Buffer size overflow");
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .map(|x| x as u64)
            .expect("Buffer size overflow");

        assert!(
            byte_start + byte_len <= source.size(), 
            "Read would go beyond source buffer bounds ({} + {} > {})", 
            byte_start, 
            byte_len, 
            source.size()
        );

        // First copy data from source to our buffer
        let mut encoder = context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("ReadbackBuffer Copy") }
        );
        encoder.copy_buffer_to_buffer(source, byte_start, &self.0, 0, byte_len);
        context.queue.submit(Some(encoder.finish()));

        // Now read from our buffer
        let buffer_slice = self.0.slice(0..byte_len as u64);
        
        // Create a oneshot channel for this operation
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("Failed to send mapping result");
        });

        // Wait for the mapping to complete
        context.device.poll(wgpu::MaintainBase::Wait)
            .expect("Failed to poll device");

        // Get the mapping result
        rx.recv()
            .expect("Failed to receive mapping result")
            .expect("Failed to map buffer");

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.0.unmap();

        result
    }
}