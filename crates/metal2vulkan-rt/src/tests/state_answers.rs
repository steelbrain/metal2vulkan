//! What a pipeline state answers of Metal's beside its threads, under a device cfg on every device
//! of the cfg's kind. The harness's `block_scan` holds 256 words of static threadgroup memory,
//! 1024 bytes as Metal reports them; `sample_down` holds none. Both answer a thread execution
//! width of 32 and, declaring no imageblock, an imageblock memory length of 0. A render state of
//! `R8Unorm` and `R32Float` colors answers Metal's measured sample of 8 bytes (the float at its
//! alignment), and 8 by 8 pixels of one sample 512 bytes.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{descriptor, examples, library, state};
    use crate::tests::no_constexpr_samplers;
    use crate::{
        ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor, PixelFormat,
        StageSource,
    };
    use metal2vulkan_devtools::fixtures_dir;

    #[test]
    fn every_device_answers_a_state_s_metal_answers() {
        let Some(workloads) = library(&fixtures_dir().join("harness/workloads.ll")) else {
            return;
        };
        let kernel = |function: &str| {
            let function = workloads.function(function).expect("the kernel");
            ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)))
        };
        let kernels = [
            ("block_scan", kernel("block_scan"), 1024),
            ("sample_down", kernel("sample_down"), 0),
        ];
        let Some(examples) = examples() else {
            return;
        };
        let mut drawn = descriptor(&examples, "first_texel_fragment", PixelFormat::R8Unorm);
        drawn.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
        each_device("state_answers", |name, _, device| {
            for (function, descriptor, length) in &kernels {
                let compute = (device
                    .new_compute_pipeline_state(descriptor, no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
                assert_eq!(
                    compute.static_threadgroup_memory_length(),
                    *length,
                    "{name}: {function}"
                );
                assert_eq!(compute.thread_execution_width(), 32, "{name}: {function}");
                let imageblock = (compute.imageblock_memory_length_for_dimensions([8, 8, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
                assert_eq!(imageblock, 0, "{name}: {function}");
            }
            let render = state(name, &device, &drawn);
            let sample = (render.imageblock_sample_length())
                .unwrap_or_else(|error| panic!("{name}: the render state: {error}"));
            assert_eq!(sample, 8, "{name}: R8Unorm then R32Float");
            let pixels = (render.imageblock_memory_length_for_dimensions([8, 8, 1]))
                .unwrap_or_else(|error| panic!("{name}: the render state: {error}"));
            assert_eq!(pixels, 512, "{name}: 8 by 8 of one sample");
            println!("{name}: a state's static threadgroup memory, width and imageblock answered");
        });
    }
}
