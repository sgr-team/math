use sgrmath_core::{ReadbackBuffer, Shader, StorageBuffer, WgpuContext};

#[test]
fn execute_bounded() {
    let (context, mut shader, source, target, readback) = prepare();

    shader.bind(&context, &[&source, &target]);
    shader.execute(&context, 5);

    assert_eq!(
        readback.read::<i32>(&context, &target, 0, 10), 
        (0_i32..10).map(|i| if i < 5 { 2 * i + 1 } else { 10 }).collect::<Vec<_>>()
    );
}

#[test]
fn execute_unbounded() {
    let (context, shader, _, _, _) = prepare();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || { shader.execute(&context, 5); }
    ));

    assert!(result.is_err());
}

#[test]
fn execute_with_params() {
    let (context, shader, source, target, readback) = prepare();

    shader.execute_with_params(&context, 5, &[&source, &target]);

    assert_eq!(
        readback.read::<i32>(&context, &target, 0, 10), 
        (0_i32..10).map(|i| if i < 5 { 2 * i + 1 } else { 10 }).collect::<Vec<_>>()
    );
}


fn prepare() -> (WgpuContext, Shader, StorageBuffer, StorageBuffer, ReadbackBuffer) {
    let context = WgpuContext::new();
    let source = StorageBuffer::new::<i32, _>(&context, 10);
    let target = StorageBuffer::new::<i32, _>(&context, 10);
    let readback = ReadbackBuffer::new::<i32, _>(&context, 10);

    let shader = Shader::new(&context, "test", include_str!("plus_one.wgsl"));
    target.update_buffer_range(&context, &[10_i32; 10], 0);
    source.update_buffer_range(&context, &(0_i32..10).map(|i| 2 * i).collect::<Vec<_>>(), 0);

    (context, shader, source, target, readback)
}