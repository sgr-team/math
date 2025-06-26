macro_rules! buffer {
    ($doc:expr, $name:ident, $usage:expr) => {
        #[doc = $doc]
        #[derive(Clone, Debug)]
        pub struct $name (pub wgpu::Buffer);

        impl $name {
            /// Creates a new buffer with the specified size
            /// 
            /// # Arguments
            /// * `context` - The WGPU context
            /// * `size` - The size of the buffer in elements of type T
            /// 
            /// # Returns
            /// A new buffer instance
            /// 
            /// # Panics
            /// Panics if the buffer size would overflow
            pub fn new<T, S>(context: &crate::WgpuContext, size: S) -> Self 
            where
                T: bytemuck::Pod,
                S: Into<crate::Size>,
            {
                let size = size.into();
                
                Self(Self::create_buffer::<T>(context, size.len() as u64))
            }

            /// Returns the size of the buffer in bytes
            #[must_use]
            pub fn size(&self) -> usize {
                self.0.size() as usize
            }

            /// Returns the size of the buffer in elements of type T
            #[must_use]
            pub fn len<T>(&self) -> usize 
            where
                T: bytemuck::Pod,
            {
                (self.0.size() / std::mem::size_of::<T>() as u64) as usize
            }

            /// Returns true if the buffer is empty
            #[must_use]
            pub fn is_empty(&self) -> bool {
                self.0.size() == 0
            }

            pub(crate) fn create_buffer<T>(context: &crate::WgpuContext, len: u64) -> wgpu::Buffer {
                context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(stringify!($name)),
                    size: len * std::mem::size_of::<T>() as u64,
                    usage: $usage,
                    mapped_at_creation: false,
                })
            }
        }

        impl std::ops::Deref for $name {
            type Target = wgpu::Buffer;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

buffer!(
    "
     StorageBuffer
     
     A buffer that can be used as a storage buffer
     
     # Usage
     * `wgpu::BufferUsages::STORAGE` - The buffer can be used as a storage buffer
     * `wgpu::BufferUsages::COPY_DST` - The buffer can be used as a destination for copying
     * `wgpu::BufferUsages::COPY_SRC` - The buffer can be used as a source for copying
    ",
    StorageBuffer, 
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
);
buffer!(
    "
     ReadbackBuffer
     
     A buffer that can be read back from the GPU
     
     # Usage
     * `wgpu::BufferUsages::MAP_READ` - The buffer can be mapped for reading
     * `wgpu::BufferUsages::COPY_DST` - The buffer can be used as a destination for copying
    ",
    ReadbackBuffer, 
    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
);

