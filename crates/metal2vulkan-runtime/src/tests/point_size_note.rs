//! `PrecisionNote::DevicePointSize` is a render state's static note, decided by its vertex
//! function and the device alone (the translator's stage I/O plan): a state whose vertex function
//! writes `[[point_size]]` carries it where the device's `pointSizeRange` ends below Metal's
//! largest point, 511 pixels (lavapipe's 256, ANV's 255), and not where it reaches it; a state
//! whose vertex function writes none carries it on no device.
//!
//! A device test, under a device cfg on every device of the cfg's kind. The functions are those of
//! the owned fixture `validation/fixtures/public/engine/rt6_render_state.ll`: `rt6_point`, a point
//! of 4 pixels that writes `[[point_size]]`, `rt6_full`, Metal's fullscreen triangle, and
//! `rt6_white`; `llvm-as` assembles them, and without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, state};
    use crate::{
        ColorAttachmentDescriptor, FunctionDescriptor, PixelFormat, PrecisionNote,
        RenderPipelineDescriptor, StageSource,
    };
    use metal2vulkan_devtools::engine_dir;

    /// Metal's largest point, in pixels.
    const METAL_LARGEST_POINT: u32 = 511;

    /// The state of `rt6_point` carries the note exactly where the device's largest point is below
    /// Metal's, and the state of `rt6_full` on no device.
    #[test]
    fn every_device_notes_its_point_size_where_a_vertex_function_writes_one() {
        let Some(library) = library(&engine_dir().join("rt6_render_state.ll")) else {
            return;
        };
        let descriptor = |vertex| {
            let function = |name| {
                StageSource::Function(FunctionDescriptor::new(
                    library.function(name).expect("the function"),
                ))
            };
            let mut d = RenderPipelineDescriptor::new(function(vertex));
            d.fragment = Some(function("rt6_white"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            d
        };
        let (sized, plain) = (descriptor("rt6_point"), descriptor("rt6_full"));
        each_device("point_size_note", |name, _, device| {
            let largest = (device.core().translator.profile().limits()).max_point_size;
            let noted = |d| {
                (state(name, &device, d).precision_notes())
                    .contains(&PrecisionNote::DevicePointSize)
            };
            let below = largest < METAL_LARGEST_POINT;
            assert_eq!(
                noted(&sized),
                below,
                "{name}: a [[point_size]] on a device whose largest point is {largest}"
            );
            assert!(!noted(&plain), "{name}: no [[point_size]]");
            println!("{name}: largest point {largest}, the [[point_size]] state noted: {below}");
        });
    }
}
