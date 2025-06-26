macro_rules! buffer_tests {
    ($mod_name:ident, $constructor:expr, $usage:expr) => {
        #[test]
        fn new() {
            let context = sgrmath_core::WgpuContext::new();
            let buffer = $constructor(&context, 100);

            assert_eq!(buffer.0.size(), 400);
            assert_eq!(buffer.0.usage(), $usage);
        }

        #[test]
        fn size() {
            let context = sgrmath_core::WgpuContext::new();
            
            assert_eq!($constructor(&context, 100).size(), 400);
            assert_eq!($constructor(&context, 89).size(), 356);
        }
        
        #[test]
        fn len() {
            let context = sgrmath_core::WgpuContext::new();
            
            assert_eq!($constructor(&context, 100).len::<f32>(), 100);
            assert_eq!($constructor(&context, 89).len::<f32>(), 89);
        }
    }
}

mod storage_buffer {
    buffer_tests!(
        storage_buffer,
        sgrmath_core::StorageBuffer::new::<f32, _>, 
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    );

    #[test]
    fn update_buffer_range() {
        let context = sgrmath_core::WgpuContext::new();
        let storage = sgrmath_core::StorageBuffer::new::<i32, _>(&context, 100);
        let readback = sgrmath_core::ReadbackBuffer::new::<i32, _>(&context, 100);

        storage.update_buffer_range::<i32>(&context, &[1, 2, 3, 4, 5], 0);
        storage.update_buffer_range::<i32>(&context, &[6, 7, 8], 20);

        assert_eq!(readback.read::<i32>(&context, &storage.0, 0, 10), vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0]);
        assert_eq!(readback.read::<i32>(&context, &storage.0, 20, 10), vec![6, 7, 8, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn update_buffer_range_async() {
        let context = sgrmath_core::WgpuContext::new();
        let storage = sgrmath_core::StorageBuffer::new::<i32, _>(&context, 100);
        let readback = sgrmath_core::ReadbackBuffer::new::<i32, _>(&context, 100);

        storage.update_buffer_range_async::<i32>(&context, &[1, 2, 3, 4, 5], 0);
        storage.update_buffer_range_async::<i32>(&context, &[6, 7, 8], 20);

        context.device.poll(wgpu::MaintainBase::Wait).unwrap();

        assert_eq!(readback.read::<i32>(&context, &storage.0, 0, 10), vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0]);
        assert_eq!(readback.read::<i32>(&context, &storage.0, 20, 10), vec![6, 7, 8, 0, 0, 0, 0, 0, 0, 0]);
    }
}

mod readback_buffer {
    buffer_tests!(
        readback_buffer,
        sgrmath_core::ReadbackBuffer::new::<f32, _>, 
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
    );

    #[test]
    fn scale() {
        let context = sgrmath_core::WgpuContext::new();
        let mut buffer = sgrmath_core::ReadbackBuffer::new::<f32, _>(&context, 100);

        buffer.scale::<f32, _>(&context, 20);
        buffer.scale::<f32, _>(&context, 99);

        assert_eq!(buffer.0.size(), 400);
        
        buffer.scale::<f32, _>(&context, 200);

        assert_eq!(buffer.0.size(), 800);
    }

    #[test]
    fn read() {
        let context = sgrmath_core::WgpuContext::new();
        let storage = sgrmath_core::StorageBuffer::new::<i32, _>(&context, 100);
        let readback = sgrmath_core::ReadbackBuffer::new::<i32, _>(&context, 100);

        storage.update_buffer_range::<i32>(
            &context, 
            (0_i32..100)
                .collect::<Vec<_>>()
                .as_slice(), 
            0
        );

        assert_eq!(
            readback.read::<i32>(&context, &storage.0, 0, 20), 
            (0_i32..20).collect::<Vec<_>>()
        );
        assert_eq!(
            readback.read::<i32>(&context, &storage.0, 42, 27), 
            (42_i32..69).collect::<Vec<_>>()
        );
    }
}

mod value_buffer {
    #[test]
    fn new() {
        let context = sgrmath_core::WgpuContext::new();
        let value = sgrmath_core::ValueBuffer::new::<Test>(&context);

        assert_eq!(value.size(), 8);
    }

    #[test]
    fn init() {
        let context = sgrmath_core::WgpuContext::new();
        let value = sgrmath_core::ValueBuffer::init::<Test>(&context, &Test(1.0, 2.0));
        let storage = sgrmath_core::StorageBuffer::new::<Test, _>(&context, 1);
        let readback = sgrmath_core::ReadbackBuffer::new::<Test, _>(&context, 1);
        let copy_shader = sgrmath_core::Shader::new(&context, "copy", include_str!("copy.wgsl"));

        assert_eq!(value.size(), 8);

        copy_shader.execute_with_params(&context, 2, &[ &value, &storage ]);
        assert_eq!(readback.read::<f32>(&context, &storage, 0, 2), vec![1.0, 2.0]);
    }

    #[test]
    fn set() {
        let context = sgrmath_core::WgpuContext::new();
        let value = sgrmath_core::ValueBuffer::init::<Test>(&context, &Test(1.0, 2.0));
        let storage = sgrmath_core::StorageBuffer::new::<Test, _>(&context, 1);
        let readback = sgrmath_core::ReadbackBuffer::new::<Test, _>(&context, 1);
        let copy_shader = sgrmath_core::Shader::new(&context, "copy", include_str!("copy.wgsl"));

        value.set(&context, &Test(3.0, 4.0));

        copy_shader.execute_with_params(&context, 2, &[ &value, &storage ]);
        assert_eq!(readback.read::<f32>(&context, &storage, 0, 2), vec![3.0, 4.0]);
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct Test(f32, f32);
}
