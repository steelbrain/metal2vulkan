//! Twin makings take off one after the other (`pipeline/flight.rs`): two render pipeline states of
//! the same functions that differ only in the format of a color attachment the fragment function
//! does not write, made at once, each make a `VkPipeline` of their own, the second's driver call
//! only once the first's landed. The test hold (`flight::held`) keeps the first making in the air,
//! holding its turn, until the second waits for it.
//!
//! A device that links its pipelines from libraries takes no flight (`graphics.rs`), and the
//! test checks only that there; the flight's own unit test (`flight::tests`) runs everywhere.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The render pipeline is the owned
//! fixture `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`), as the
//! examples run it.

use crate::pipeline::flight::held;
use crate::{
    Device, DeviceKind, DeviceOptions, PipelineOptions, RenderPipelineState, ShaderCacheOptions,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-twin-flights-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`, empty.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The fixture's render pipeline state: `colored_vertex` and `colored_fragment`, writing an
/// `RGBA8Unorm` color 0, beside a color 1 of `second` it does not write.
fn state(device: &Device, second: PixelFormat) -> RenderPipelineState {
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.color_attachments[1] = Some(ColorAttachmentDescriptor::new(second));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Wait until `done` holds, at most a minute (a guard against a hang).
fn until(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Two twin states made at once: the first's making takes off and is held in the air; the
/// second's waits for it (one flight followed) and takes off only once it landed; both states are
/// made, each with its own `VkPipeline`.
#[test]
fn a_twin_making_waits_for_its_twin_to_land() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let before = device.work_counts();
    // A device that links its pipelines from libraries (lavapipe, RADV, ANV, NVIDIA) shares each
    // part's making already, and its makings take no flight: both are made, none followed.
    if device.shared().libraries.is_some() {
        std::thread::scope(|scope| {
            let first = scope.spawn(|| state(&device, PixelFormat::RGBA8Unorm));
            let second = scope.spawn(|| state(&device, PixelFormat::BGRA8Unorm));
            first.join().unwrap();
            second.join().unwrap();
        });
        assert_eq!(
            device.work_counts().flights_followed,
            before.flights_followed
        );
        return;
    }
    let hold = held::hold();
    std::thread::scope(|scope| {
        let first = scope.spawn(|| state(&device, PixelFormat::RGBA8Unorm));
        until(|| !held::events().is_empty());
        let second = scope.spawn(|| state(&device, PixelFormat::BGRA8Unorm));
        until(|| held::followers() == 1);
        drop(hold);
        first.join().unwrap();
        second.join().unwrap();
    });
    let events = held::events();
    let key = events[0].0;
    assert_eq!(
        events,
        [(key, true), (key, false), (key, true), (key, false)],
        "the twins' flights, one after the other"
    );
    let after = device.work_counts();
    assert_eq!(after.flights_followed - before.flights_followed, 1);
    assert_eq!(
        after.variant_pipelines_made - before.variant_pipelines_made,
        2,
        "each twin makes its own pipeline"
    );
}
