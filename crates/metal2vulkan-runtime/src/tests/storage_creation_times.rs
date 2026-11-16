//! What the creation of a state whose functions read storage views costs, as wall time of the
//! creation call on the venue it runs on: a render state of `colored_vertex` and
//! `colored_fragment` (`examples/runtime_examples.ll`, whose vertex function reads its positions
//! and colors through storage views) and a compute state of `buffer_end_byte_load`
//! (`engine/buffer_end_byte_load.ll`, a source and an output storage view) that requires its
//! threadgroup, so its creation makes its predicted region's pipeline. Each is created three
//! ways, [`ROUNDS`] times: cold (a new device over an empty shader cache directory), from the
//! device's memo (the same descriptor again on that device), and from the directory (a new device
//! over the cold one's directory). Each line also gives the pipelines the creation made
//! (`WorkCounts::creation_pipelines_made`): on a profile whose storage views declare
//! `SpecKey::RangeAtBindOffset` a creation makes two, the predicted key and its general form
//! (`pipeline/predicted.rs`, `pipeline/dispatch_making.rs`), together, and the cold figure says
//! whether it waits for them one after the other.
//!
//! Ignored: a timing run, in release mode with the drivers' disk caches off
//! (`MESA_SHADER_CACHE_DISABLE=true`), whose figures are the host's under its load, so it asserts
//! no time. A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on
//! the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Device, PipelineOptions};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::time::{Duration, Instant};

/// The rounds of each way.
const ROUNDS: usize = 7;

/// The two descriptors on `device`: the render state's and the compute state's.
fn descriptors(device: &Device) -> (RenderPipelineDescriptor, ComputePipelineDescriptor) {
    let library = super::encoder_holds::library(device, "examples/runtime_examples.ll");
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut render =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    render.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    render.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let kernel = super::encoder_holds::library(device, "engine/buffer_end_byte_load.ll")
        .new_function_with_name("buffer_end_byte_load")
        .unwrap();
    let mut compute =
        ComputePipelineDescriptor::new(StageSource::Function(kernel.function_descriptor()));
    compute.required_threads_per_threadgroup = Some([32, 1, 1]);
    (render, compute)
}

/// Create `descriptors`' two states on `device`: each one's wall time and the pipelines its
/// creation made.
fn create(
    device: &Device,
    (render, compute): &(RenderPipelineDescriptor, ComputePipelineDescriptor),
) -> [(Duration, u64); 2] {
    let options = PipelineOptions::default();
    let made = || device.work_counts().creation_pipelines_made;
    let before = made();
    let start = Instant::now();
    let render_state = device.new_render_pipeline_state(render, &options).unwrap();
    let render_time = start.elapsed();
    let between = made();
    let start = Instant::now();
    let compute_state = device
        .new_compute_pipeline_state(compute, &options)
        .unwrap();
    let compute_time = start.elapsed();
    let after = made();
    drop((render_state, compute_state));
    [
        (render_time, between - before),
        (compute_time, after - between),
    ]
}

/// The median of `times`, in microseconds, and each round's in order, with the pipelines each
/// made.
fn figures(times: &[(Duration, u64)]) -> String {
    let rounds: Vec<u128> = times.iter().map(|(t, _)| t.as_micros()).collect();
    let mut sorted = rounds.clone();
    sorted.sort_unstable();
    let made: Vec<u64> = times.iter().map(|&(_, n)| n).collect();
    format!(
        "median {} us, rounds {rounds:?} us, made {made:?}",
        sorted[sorted.len() / 2]
    )
}

#[test]
#[ignore = "a timing run: release mode, the drivers' disk caches off"]
fn what_a_storage_view_state_creation_costs() {
    // By way, render then compute.
    let mut ways: [[Vec<(Duration, u64)>; 2]; 3] = Default::default();
    for round in 0..ROUNDS {
        let scratch = Scratch::new(&format!("storage-creation-times-{round}"));
        {
            let device = device(&scratch.0);
            let descriptors = descriptors(&device);
            for (kind, took) in create(&device, &descriptors).into_iter().enumerate() {
                ways[0][kind].push(took);
            }
            for (kind, took) in create(&device, &descriptors).into_iter().enumerate() {
                ways[1][kind].push(took);
            }
        }
        let device = device(&scratch.0);
        let descriptors = descriptors(&device);
        for (kind, took) in create(&device, &descriptors).into_iter().enumerate() {
            ways[2][kind].push(took);
        }
    }
    for (way, name) in ways.iter().zip(["cold", "memo", "directory"]) {
        for (times, kind) in way.iter().zip(["render", "compute"]) {
            println!("STORAGE-CREATION {kind} {name}: {}", figures(times));
        }
    }
}
