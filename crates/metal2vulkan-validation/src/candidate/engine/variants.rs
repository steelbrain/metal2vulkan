//! The kernel a case's dispatch runs, compiled as the runtime compiles it.
//!
//! A threadgroup with a partial simdgroup, which the kernel's flattened simd tier cannot run, runs
//! on its emulated-simdgroups variant ([`compile_kernel`]), as the runtime's encoder chooses the
//! module a dispatch runs. The authored compile census compiles a kernel so
//! (`acceptance::PipelineRequest::Kernel`), and a render or mesh pipeline as
//! `Device::compile_render_pipeline` and `Device::compile_mesh_pipeline` compile it. Everything here
//! is device-free.

use metal2vulkan::device::Device;
use metal2vulkan::error::Error;
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, KernelVariant, LoweringInputs,
    RuntimeFact, StageSource,
};
use metal2vulkan::Budget;

/// Compile `function` into a compute pipeline for `device` under `budget` as the runtime compiles
/// it for a dispatch of threadgroups of `threads_per_threadgroup`: its plain module
/// (`LoweringInputs::new`), or its emulated-simdgroups variant
/// (`KernelVariant::EmulatedSimdgroups`) where the threadgroup's threads are not a multiple of a
/// `RuntimeFact::LocalSizeConstraint` the plain module states (a partial simdgroup its
/// flattened simd tier cannot run), as the runtime's encoder chooses the module a dispatch runs.
pub fn compile_kernel(
    device: &Device,
    function: &FunctionDescriptor,
    threads_per_threadgroup: [u32; 3],
    budget: &Budget,
) -> Result<ComputePipeline, Error> {
    let compile = |inputs: &LoweringInputs| {
        let compiled = device.compile_function(function, inputs, budget)?;
        let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        device.compile_compute_pipeline(&descriptor, budget)
    };
    let plain = LoweringInputs::new();
    let pipeline = compile(&plain)?;
    let threads = threads_per_threadgroup
        .iter()
        .map(|&axis| u64::from(axis))
        .product::<u64>();
    let partial = pipeline.dependencies.iter().any(|dependency| {
        matches!(dependency.fact, RuntimeFact::LocalSizeConstraint { multiple }
            if !threads.is_multiple_of(u64::from(multiple.max(1))))
    });
    if partial {
        compile(&plain.kernel_variant(KernelVariant::EmulatedSimdgroups))
    } else {
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::device::{DeviceDescription, DevicePolicy};

    fn device(stem: &str) -> Device {
        let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).unwrap();
        let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        Device::from_queries(description, DevicePolicy::default()).unwrap()
    }

    /// A dispatch whose threadgroup has a partial simdgroup runs the kernel's emulated-simdgroups
    /// variant on a device whose subgroups run whole simdgroups (moltenvk-m3); a full threadgroup
    /// runs the plain module.
    #[test]
    fn a_partial_simdgroup_dispatch_compiles_the_emulated_variant() {
        use metal2vulkan::library::Library;
        use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

        let bitcode = match llvm_assemble(&engine_dir().join("simd_ops.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected simd_ops.ll: {why}"),
        };
        let device = device("moltenvk-m3");
        let library = Library::from_air(&bitcode).unwrap();
        let function = FunctionDescriptor::new(library.function("simd_sum_k").unwrap());
        let budget = Budget::standard();
        let under = |inputs: &LoweringInputs| {
            let compiled = device.compile_function(&function, inputs, &budget).unwrap();
            let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
            device
                .compile_compute_pipeline(&descriptor, &budget)
                .unwrap()
        };
        let plain = under(&LoweringInputs::new());
        let emulated =
            under(&LoweringInputs::new().kernel_variant(KernelVariant::EmulatedSimdgroups));
        assert_ne!(plain, emulated);
        for n in [1, 4, 16] {
            let kernel = compile_kernel(&device, &function, [n, 1, 1], &budget).unwrap();
            assert_eq!(kernel, emulated, "{n}");
        }
        let kernel = compile_kernel(&device, &function, [32, 1, 1], &budget).unwrap();
        assert_eq!(kernel, plain);
    }
}
