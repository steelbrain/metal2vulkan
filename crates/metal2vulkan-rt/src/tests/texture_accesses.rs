//! The texture slots' accesses a pipeline state answers, under a device cfg on every device of
//! the cfg's kind. The harness's `sample_down` samples slot 0 and writes slot 1, whose size it
//! also queries: `ReadOnly` and `WriteOnly`. `first_texel_fragment` reads slot 0: `ReadOnly`,
//! beside `colored_vertex`, a vertex function of no texture, and no slot for a stage a render
//! state has no function of. Each is the slot list a dispatch or a draw of the state marks in its
//! hazards.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{descriptor, examples, library, state};
    use crate::tests::no_constexpr_samplers;
    use crate::{
        BindingAccess, ComputePipelineDescriptor, FunctionDescriptor, PixelFormat, Stage,
        StageSource,
    };
    use metal2vulkan_devtools::fixtures_dir;

    #[test]
    fn every_device_answers_a_state_s_texture_accesses() {
        let Some(workloads) = library(&fixtures_dir().join("harness/workloads.ll")) else {
            return;
        };
        let Some(examples) = examples() else {
            return;
        };
        let function = workloads.function("sample_down").expect("the kernel");
        let kernel = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let drawn = descriptor(&examples, "first_texel_fragment", PixelFormat::RGBA8Unorm);
        each_device("texture_accesses", |name, _, device| {
            let compute = (device.new_compute_pipeline_state(&kernel, no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: sample_down: {error}"));
            assert_eq!(
                compute.texture_accesses(),
                [(0, BindingAccess::ReadOnly), (1, BindingAccess::WriteOnly)],
                "{name}: sample_down"
            );
            let render = state(name, &device, &drawn);
            assert_eq!(
                render.texture_accesses(Stage::Fragment),
                [(0, BindingAccess::ReadOnly)],
                "{name}: first_texel_fragment"
            );
            for stage in [Stage::Vertex, Stage::Kernel] {
                let none = render.texture_accesses(stage);
                assert!(none.is_empty(), "{name}: {stage:?}: {none:?}");
            }
            println!("{name}: a kernel's and a fragment function's texture accesses answered");
        });
    }
}
