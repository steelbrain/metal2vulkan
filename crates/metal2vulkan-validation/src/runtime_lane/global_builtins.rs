//! A fragment function's Metal 3.1 program-scope built-ins on the device the cfg selects, under the
//! validation layer.
//!
//! Apple's compiler takes a global `thread_index_in_quadgroup` and `threads_per_simdgroup` in a
//! fragment function. Metal makes and runs the render pipeline of the second (the owned fixture
//! `gsm_global_builtins_fragment`, 32 on an M4), and refuses the first's (`gsm_global_builtins_
//! fragment_quad`) on an M4 under macOS 27.0.1 and an M3 Max under macOS 26.7: "Encountered
//! unlowered function call to air.get_thread_index_in_quadgroup.i32" (the M4 adds "Verify device or
//! GPU architecture supports this feature"). The runtime refuses it as Metal does:
//! `Unsupported` (`GlobalBuiltinInFragment`, `Needs::MetalRefuses` naming those devices) on a
//! device that does not claim Apple10, and `NotImplemented` on one that does, which no
//! measurement covers.

use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{Error, Function, GpuFamily, PipelineOptions, RenderPipelineState};

/// The function `name` of the owned fixture `stem` on `lane`'s device.
fn function(lane: &Lane, stem: &str, name: &str) -> Function {
    let ll = metal2vulkan_devtools::engine_dir().join(format!("{stem}.ll"));
    lane.device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
        .new_function_with_name(name)
        .unwrap()
}

/// The render pipeline of the fixture `rt6_resolves`'s `rt6_resolve_half` vertex function and
/// `fragment` of `stem`, drawing an `R32Uint` target.
fn pipeline(lane: &Lane, stem: &str, fragment: &str) -> Result<RenderPipelineState, Error> {
    let vertex = function(lane, "rt6_resolves", "rt6_resolve_half");
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(
        function(lane, stem, fragment).function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A fragment function's global `thread_index_in_quadgroup` fails its render pipeline's
/// creation as Metal's does on an M4 and an M3 Max; the control, a global
/// `threads_per_simdgroup`, is made.
#[test]
fn a_fragment_s_global_quad_lane_fails_its_render_pipeline_s_creation() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    if let Err(e) = pipeline(
        &lane,
        "gsm_global_builtins_fragment",
        "global_builtins_in_a_fragment",
    ) {
        panic!("{name}: the control: {e:?}");
    }
    let needs = if lane.device.supports_family(GpuFamily::Apple10.raw()) {
        Needs::NotImplemented
    } else {
        Needs::MetalRefuses(&["Apple M3 Max, macOS 26.7", "Apple M4, macOS 27.0.1"])
    };
    match pipeline(
        &lane,
        "gsm_global_builtins_fragment_quad",
        "global_quad_lane_in_a_fragment",
    ) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::GlobalBuiltinInFragment, needs),
            "{name}"
        ),
        other => panic!("{name}: {:?}", other.err()),
    }
    quiet(lane, &messages);
}
