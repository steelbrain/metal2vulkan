//! The buffer slots' accesses a compute pipeline state answers, under a device cfg on every device
//! of the cfg's kind. The harness's `scale_offset` reads slot 0 (`device const float4 *`), writes
//! slot 1 (`device float4 *`, which Metal's reflection answers `ReadWrite`) and reads slot 2
//! (`constant float2 &`): `ReadOnly`, `WriteOnly` and `ReadOnly`. `lcg_step` reads and writes
//! slot 0 and reads slot 1: `ReadWrite` and `ReadOnly`. Each is the slot list a dispatch of the
//! state marks in its hazards.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::library;
    use crate::tests::no_constexpr_samplers;
    use crate::{BindingAccess, ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    use metal2vulkan_devtools::fixtures_dir;

    #[test]
    fn every_device_answers_a_state_s_buffer_accesses() {
        let Some(workloads) = library(&fixtures_dir().join("harness/workloads.ll")) else {
            return;
        };
        let cases = [
            (
                "scale_offset",
                vec![
                    (0, BindingAccess::ReadOnly),
                    (1, BindingAccess::WriteOnly),
                    (2, BindingAccess::ReadOnly),
                ],
            ),
            (
                "lcg_step",
                vec![(0, BindingAccess::ReadWrite), (1, BindingAccess::ReadOnly)],
            ),
        ];
        each_device("buffer_accesses", |name, _, device| {
            for (kernel, want) in &cases {
                let function = workloads.function(kernel).expect("the kernel");
                let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                let state = (device.new_compute_pipeline_state(&descriptor, no_constexpr_samplers))
                    .unwrap_or_else(|error| panic!("{name}: {kernel}: {error}"));
                assert_eq!(&state.buffer_accesses(), want, "{name}: {kernel}");
            }
            println!("{name}: two kernels' buffer accesses answered");
        });
    }
}
