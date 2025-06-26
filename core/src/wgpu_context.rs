use std::ops::Deref;
use std::thread;
use tokio::runtime::Runtime;

use wgpu::{Device, Queue};

/// A wrapper around WGPU device and queue that provides GPU computation capabilities.
///
/// This struct encapsulates the core WGPU components needed for GPU computation:
/// - A [`Device`] for creating GPU resources and pipelines
/// - A [`Queue`] for submitting commands to the GPU
///
/// # Example
/// ```no_run
/// use sgrmath_core::WgpuContext;
///
/// fn example() {
///     // Create a new GPU context
///     let context = WgpuContext::new();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct WgpuContext {
    /// The WGPU device used for creating GPU resources.
    pub device: Device,
    /// The WGPU queue used for submitting commands to the GPU.
    pub queue: Queue,
}

impl WgpuContext {
    /// Creates a new WGPU context synchronously.
    ///
    /// This method will create a new WGPU context in a separate thread and return it.
    ///
    /// # Returns
    /// A new `WgpuContext` containing the initialized device and queue.
    #[must_use]
    pub fn new() -> Self {
        thread::spawn(|| Runtime::new().unwrap().block_on(Self::new_async()))
            .join()
            .expect("Failed to create WGPU context")
    }
    
    /// Creates a new WGPU context by initializing a GPU device and command queue.
    ///
    /// This method will:
    /// 1. Create a WGPU instance supporting all available backends
    /// 2. Request an adapter with default power preferences
    /// 3. Create a device and queue with default features and limits
    ///
    /// # Returns
    /// A new `WgpuContext` containing the initialized device and queue.
    ///
    /// # Panics
    /// This method will panic if:
    /// - No suitable GPU adapter is found
    /// - Device creation fails
    #[must_use]
    #[allow(clippy::unwrap_used)]
    pub async fn new_async() -> Self {
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = 1_000_000_000; // 1GB
        limits.max_storage_buffer_binding_size = 1_000_000_000; // 1GB
        limits.max_uniform_buffer_binding_size = 1_000_000_000; // 1GB


        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map(|(device, queue)| Self { device, queue })
            .unwrap()
    }
}

impl Deref for WgpuContext {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}