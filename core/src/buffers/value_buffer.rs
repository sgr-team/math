use wgpu::Buffer;
use crate::WgpuContext;

/// A GPU buffer for storing a single value.
/// 
/// This buffer is useful for passing parameters or uniform data to shaders.
/// 
/// # Usage
/// * `wgpu::BufferUsages::STORAGE` - The buffer can be used as a storage buffer
/// * `wgpu::BufferUsages::UNIFORM` - The buffer can be used as a uniform buffer
/// * `wgpu::BufferUsages::COPY_DST` - The buffer can be used as a destination for copying
#[derive(Clone, Debug)]
pub struct ValueBuffer(pub Buffer);

impl ValueBuffer {
    /// Creates a new `ValueBuffer` with uninitialized contents.
    ///
    /// # Arguments
    /// * `context` - The WGPU context
    ///
    /// # Returns
    /// A new `ValueBuffer` instance
    #[must_use]
    pub fn new<T>(context: &WgpuContext) -> Self
    where
        T: bytemuck::Pod,
    {
        Self(
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ValueBuffer"),
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        )
    }

    /// Creates a new `ValueBuffer` and initializes it with the given value.
    ///
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `value` - The value to store in the buffer
    ///
    /// # Returns
    /// A new `ValueBuffer` instance containing the value
    pub fn init<T>(context: &WgpuContext, value: &T) -> Self
    where
        T: bytemuck::Pod,
    {
        let buf = Self::new::<T>(context);
        buf.set(context, value);
        buf
    }

    /// Updates the buffer with a new value.
    ///
    /// # Arguments
    /// * `context` - The WGPU context
    /// * `value` - The value to write to the buffer
    pub fn set<T>(&self, context: &WgpuContext, value: &T)
    where
        T: bytemuck::Pod,
    {
        context.queue.write_buffer(self, 0, bytemuck::cast_slice(std::slice::from_ref(value)));
    }
}

impl std::ops::Deref for ValueBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
