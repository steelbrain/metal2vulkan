//! A kernel's 16x16x16 simdgroup matrix product on the device the cfg selects, under the
//! validation layer.
//!
//! Metal makes the pipeline of such a kernel only on a GPU of the Apple10 family and later: an M3
//! Max under macOS 27.0.1 refuses every kernel of the owned fixture `simd_matrix16` (`float`,
//! `half` and widening signed 8-bit products, each with the four transpose-flag pairs) at its
//! creation, "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later"
//! (`crates/metal2vulkan-conformance/probes/simd_matrix16.m`). The runtime refuses it as Metal
//! does: `Unsupported` (`SimdgroupMatrix16x16x16`, `Needs::MetalGpuFamily` of Apple10) on a device
//! that does not claim Apple10, and `NotImplemented` on one that does, whose fragment layout is
//! not measured.

use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{ComputePipelineState, Error, GpuFamily, PipelineOptions};

/// The compute pipeline of the fixture's kernel `name` on `lane`'s device.
fn pipeline(lane: &Lane, name: &str) -> Result<ComputePipelineState, Error> {
    let ll = metal2vulkan_devtools::engine_dir().join("simd_matrix16.ll");
    let function = (lane.device)
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
        .new_function_with_name(name)
        .unwrap();
    lane.device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

/// Every kernel of the fixture fails its compute pipeline's creation as Metal's does below
/// Apple10.
#[test]
fn a_kernel_s_16x16x16_product_fails_its_pipeline_s_creation_as_metal_s() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let apple10 = GpuFamily::Apple10.raw();
    let needs = if lane.device.supports_family(apple10) {
        Needs::NotImplemented
    } else {
        Needs::MetalGpuFamily(apple10)
    };
    for form in ["f32_f32", "f16_f16", "s_s"] {
        for tt in ["00", "01", "10", "11"] {
            let kernel = format!("m16_{form}_{tt}");
            match pipeline(&lane, &kernel) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    (u.feature, u.needs),
                    (Feature::SimdgroupMatrix16x16x16, needs.clone()),
                    "{name} {kernel}"
                ),
                other => panic!("{name} {kernel}: {:?}", other.err()),
            }
        }
    }
    quiet(lane, &messages);
}
