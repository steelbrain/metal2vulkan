//! Functions Metal's stitcher makes (`newLibraryWithStitchedDescriptor:`), linked into a kernel
//! that calls them, through the runtime on the device the cfg selects, under the validation
//! layer, against what the probe `stitched_function_linked.m` (the conformance crate's `probes/`)
//! measured on an Apple M3 Max under macOS 27.0.1: on the inputs 2, 3, 4 and 5, `fused` linked
//! in `privateFunctions` or in `functions` stores `(2 + 3) * 4 = 20`, and `fused4` stores
//! `(2 + 3) * 4 + 5 = 25`.
//!
//! The caller is the owned fixture `stitched_caller` (kernels `stitch_caller` and
//! `stitch_caller4` storing `fused(in[0], in[1], in[2])` and `fused4(in[0], .., in[3])` to word
//! 0). The stitched library is `metal2vulkan_devtools::stitched`'s, made from the library's bytes
//! (`newLibraryWithData:`), the form the stitched library takes when its bytes are handed over:
//! one module per stitched function, each carrying the stitching signature that describes the
//! graph it was made from.

use super::host_memory::{on_device, run};
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::stitched::{self, PROBE_GRAPHS};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{Device, Library, PipelineOptions, ResourceOptions, Size};

/// The kernel of the caller fixture that calls each of [`PROBE_GRAPHS`], and the word Metal
/// stored.
const KERNELS: [(&str, f32); 2] = [("stitch_caller", 20.0), ("stitch_caller4", 25.0)];

/// The caller's library and the stitched library.
fn libraries(device: &Device) -> (Library, Library) {
    let caller = llvm_assemble(&engine_dir().join("stitched_caller.ll")).unwrap();
    let modules: Vec<Vec<u8>> = PROBE_GRAPHS
        .iter()
        .map(|g| llvm_assemble_text(&stitched::module(g)).unwrap())
        .collect();
    let named: Vec<(&str, &[u8])> = PROBE_GRAPHS
        .iter()
        .zip(&modules)
        .map(|(g, m)| (g.name, m.as_slice()))
        .collect();
    (
        device.new_library_with_air(&caller).unwrap(),
        device
            .new_library_with_data(&stitched::library(&named))
            .unwrap(),
    )
}

/// Word 0 that `kernel` of `caller` stores linking `linked`, one thread on the inputs 2, 3, 4
/// and 5.
fn word(device: &Device, caller: &Library, kernel: &str, linked: LinkedFunctions) -> f32 {
    let kernel = caller.new_function_with_name(kernel).unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(
                kernel.function_descriptor().with_linked(linked),
            )),
            &PipelineOptions::default(),
        )
        .unwrap();
    let input: Vec<u8> = [2.0f32, 3.0, 4.0, 5.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input = device
        .new_buffer_with_bytes(&input, ResourceOptions::shared())
        .unwrap();
    let out = device.new_buffer(4, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    let bytes = read_buffer(&out, 0, 4).unwrap();
    f32::from_le_bytes(bytes[..4].try_into().unwrap())
}

/// Each stitched function, linked in `privateFunctions` and in `functions`, stores Metal's word:
/// the probe's `private` and `functions` (20) and `deep` (25).
#[test]
fn a_stitched_function_computes_its_graph() {
    on_device(|device| {
        let (caller, stitched) = libraries(device);
        for ((kernel, want), g) in KERNELS.iter().zip(&PROBE_GRAPHS) {
            let f = stitched.new_function_with_name(g.name).unwrap();
            let private = LinkedFunctions::new().with_private_function(f.linked_function());
            let plain = LinkedFunctions::new().with_function(f.linked_function());
            assert_eq!(
                word(device, &caller, kernel, private),
                *want,
                "{} private",
                g.name
            );
            assert_eq!(word(device, &caller, kernel, plain), *want, "{}", g.name);
        }
    });
}
