//! What a compute state answers of Metal's beside its threads, under a device cfg on every device
//! of the cfg's kind. The harness's `block_scan` holds 256 words of static threadgroup memory,
//! 1024 bytes as Metal reports them; `sample_down` holds none. Both answer a thread execution
//! width of 32 and, declaring no imageblock, an imageblock memory length of 0.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::library;
    use crate::tests::no_constexpr_samplers;
    use crate::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
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
        each_device("state_answers", |name, _, device| {
            for (function, descriptor, length) in &kernels {
                let state = (device.new_compute_pipeline_state(descriptor, no_constexpr_samplers))
                    .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
                assert_eq!(
                    state.static_threadgroup_memory_length(),
                    *length,
                    "{name}: {function}"
                );
                assert_eq!(state.thread_execution_width(), 32, "{name}: {function}");
                let imageblock = (state.imageblock_memory_length_for_dimensions([8, 8, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
                assert_eq!(imageblock, 0, "{name}: {function}");
            }
            println!("{name}: a state's static threadgroup memory, width and imageblock answered");
        });
    }
}
