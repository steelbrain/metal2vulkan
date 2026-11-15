//! What a depth attachment's hidden layers cost the GPU through the runtime, on the device the cfg
//! selects, without the validation layer: report only, to weigh how the runtime makes and holds
//! depth render targets (`image_plan::hiz_candidate`, `render_target::depth_layout` and
//! `texture/held_depth.rs` in the runtime, `DriverFact::SampledDepthKeepsHiZ` and
//! `MultisampledDepthFasterOutOfGeneral`). Each frame is one command buffer of one pass into a
//! 1920x1080 `RGBA8Unorm` target and a `Depth32Float` attachment, both `RenderTarget` textures of
//! `Private` storage, cleared (the depth to 1.0) and stored; its draws are `rt6_render_state`'s
//! fullscreen triangle (`rt6_full`) shaded white (`rt6_white`) under `Less` with writes, front to
//! back from depth 0.1, so the first layer passes and every later one is hidden. The cases: one
//! layer and [`LAYERS`] layers, each single-sampled and with 4 samples, and each with the depth a
//! device's own texture and an automatic `Private` heap's (whose image takes no usage the texture
//! does not declare: `with_undeclared_usage` in the runtime gives a device's own texture the
//! sampled usage where it is free, and a heap's texture its declared usage alone). A device's own
//! multisampled depth runs twice more, its frames' encoders naming it otherwise too (made with
//! `ShaderRead` usage too and bound as a fragment texture its pipeline does not read): such an
//! encoder moves the depth into the attachment layout and back around its instance, where one that
//! only renders it holds it there from frame to frame. The two forms alternate, twice each.
//!
//! Each case times [`RUNS`] frames after [`WARM`] untimed ones: the command buffer's GPU start to
//! end and the host's wall time from its commit until its wait returns. Printed per case: the
//! medians and the fastest frames, whether the device has the driver facts, and the instances that
//! held the depth or moved it and the moves back the case ran (`WorkCounts`). Run at two commits
//! on one device, the cases compare how the runtime's depth images cost. The first run of a
//! session after the GPU idled ran some cases several times slower on the Iris Xe of RPL-P,
//! either way: run it twice and keep the second.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandQueue, DepthStencilState, Device, Heap, HeapDescriptor, HeapType, LoadAction,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
    Texture, TextureDescriptor,
};
use std::time::Instant;

/// The timed frames of each case, and the untimed ones before them.
const RUNS: usize = 40;
const WARM: usize = 8;

/// The target's size.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;

/// The layers a hiding case draws.
const LAYERS: usize = 32;

/// The bytes between two layers' depths in the depth buffer (`Rt6Depth`, two `float`s), as
/// wide as any device's constant buffer offset alignment.
const STRIDE: u64 = 256;

/// One sample count's pipeline and attachments, and the heap of a heap's depth.
struct Kit {
    pipeline: RenderPipelineState,
    target: Texture,
    depth: Texture,
    _heap: Option<Heap>,
}

impl Kit {
    /// The kit of `samples`, its depth a heap's where `in_heap`, and of `ShaderRead` usage too
    /// where `read` (so that binding it for reading takes no image of its own).
    fn new(device: &Device, samples: u64, in_heap: bool, read: bool) -> Kit {
        let ll = metal2vulkan_devtools::engine_dir().join("rt6_render_state.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("rt6_full"));
        descriptor.fragment = Some(function("rt6_white"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        descriptor.raster_sample_count = samples as u32;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let descriptor = |format: PixelFormat| {
            let mut d = TextureDescriptor::texture_2d(format, WIDTH, HEIGHT, false);
            if samples > 1 {
                d.texture_type = TextureType::Type2DMultisample;
                d.sample_count = samples;
            }
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            d
        };
        let mut depth = descriptor(PixelFormat::Depth32Float);
        if read {
            depth.usage |= TextureUsage::SHADER_READ;
        }
        let heap = in_heap.then(|| {
            let size = device.heap_texture_size_and_align(&depth).unwrap().size;
            let d = HeapDescriptor::new(HeapType::Automatic, size, ResourceOptions::private());
            device.new_heap(&d).unwrap()
        });
        Kit {
            pipeline,
            target: device
                .new_texture(&descriptor(PixelFormat::RGBA8Unorm))
                .unwrap(),
            depth: match &heap {
                Some(heap) => heap.new_texture(&depth).unwrap(),
                None => device.new_texture(&depth).unwrap(),
            },
            _heap: heap,
        }
    }
}

/// The depths of [`LAYERS`] layers, [`STRIDE`] bytes apart: layer `i` at `0.1 + 0.8 i / LAYERS`,
/// flat (slope 0).
fn depths(device: &Device) -> Buffer {
    let mut bytes = vec![0u8; STRIDE as usize * LAYERS];
    for i in 0..LAYERS {
        let z = 0.1f32 + 0.8 * i as f32 / LAYERS as f32;
        let at = i * STRIDE as usize;
        bytes[at..at + 4].copy_from_slice(&z.to_le_bytes());
    }
    buffer_with(device, &bytes).unwrap()
}

/// One timed frame's GPU and wall times, in microseconds.
struct Times {
    gpu: f64,
    wall: f64,
}

/// The times of one frame of `layers` layers into `kit`'s attachments, its encoder naming the
/// depth otherwise too where `named`.
fn frame(
    queue: &CommandQueue,
    kit: &Kit,
    less: &DepthStencilState,
    depths: &Buffer,
    (layers, named): (usize, bool),
) -> Times {
    let command_buffer = queue.command_buffer().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(kit.target.clone());
    target.load_action = LoadAction::Clear;
    target.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(kit.depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.clear_depth = 1.0;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    if named {
        (encoder.set_fragment_texture(Some(&kit.depth), 0)).unwrap();
    }
    encoder.set_render_pipeline_state(&kit.pipeline).unwrap();
    encoder.set_depth_stencil_state(less).unwrap();
    for layer in 0..layers {
        encoder
            .set_vertex_buffer(Some(depths), layer as u64 * STRIDE, 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wall = committed.elapsed().as_secs_f64() * 1e6;
    assert_eq!(command_buffer.error(), None);
    let start = command_buffer.gpu_start_time().unwrap();
    let end = command_buffer.gpu_end_time().unwrap();
    Times {
        gpu: (end - start) * 1e6,
        wall,
    }
}

/// The median and the smallest of `values`.
fn median_and_least(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2], values[0])
}

/// Every case's times, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn depth_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let name = device.name().to_owned();
    let facts = &device.properties().driver_facts;
    let hiz = facts.contains(&DriverFact::SampledDepthKeepsHiZ);
    let moves = facts.contains(&DriverFact::MultisampledDepthFasterOutOfGeneral);
    let queue = device.new_command_queue().unwrap();
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    let less = device.new_depth_stencil_state(&less).unwrap();
    let depths = depths(&device);
    let cases = [
        (1, false, false),
        (1, true, false),
        (4, false, false),
        (4, false, true),
        (4, true, false),
        (4, false, false),
        (4, false, true),
    ];
    for (samples, in_heap, named) in cases {
        let kit = Kit::new(&device, samples, in_heap, named);
        let home = match (in_heap, named) {
            (true, _) => "heap",
            (false, false) => "own",
            (false, true) => "own named",
        };
        for layers in [1, LAYERS] {
            for _ in 0..WARM {
                frame(&queue, &kit, &less, &depths, (layers, named));
            }
            let before = device.work_counts();
            let runs: Vec<Times> = (0..RUNS)
                .map(|_| frame(&queue, &kit, &less, &depths, (layers, named)))
                .collect();
            let after = device.work_counts();
            let held = after.depth_instances_held - before.depth_instances_held;
            let moved = after.depth_attachments_moved - before.depth_attachments_moved;
            let released = after.held_depths_released - before.held_depths_released;
            let (gpu, gpu_least) = median_and_least(runs.iter().map(|r| r.gpu).collect());
            let (wall, wall_least) = median_and_least(runs.iter().map(|r| r.wall).collect());
            eprintln!(
                "DEPTH-COST on {name} (SampledDepthKeepsHiZ {hiz}, \
                 MultisampledDepthFasterOutOfGeneral {moves}): {home} {samples}x, {layers} \
                 layers: GPU median {gpu:.1} us, fastest {gpu_least:.1} us; wall median \
                 {wall:.1} us, fastest {wall_least:.1} us; instances held {held}, moved {moved}, \
                 released {released}"
            );
        }
    }
}

/// The frames of blits [`held_depth_cpu`] times in each round, and the untimed ones before them.
const CPU_FRAMES: usize = 2000;
const CPU_WARM: usize = 200;

/// The blit encoders of each such frame.
const CPU_ENCODERS: usize = 8;

/// The CPU time so far of the process's threads, in microseconds (0 where Linux's `/proc` is
/// not).
fn process_cpu_us() -> f64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0.0;
    };
    let on_cpu_ns = |task: std::fs::DirEntry| {
        let stat = std::fs::read_to_string(task.path().join("schedstat")).ok()?;
        stat.split_whitespace().next()?.parse::<f64>().ok()
    };
    tasks.flatten().filter_map(on_cpu_ns).sum::<f64>() / 1e3
}

/// What a held depth alive on the device costs the CPU of batches that do not name it: frames of
/// [`CPU_ENCODERS`] blit fills of a small buffer, each frame one command buffer committed and
/// waited for, with no held depth on the device and with one (a device's own multisampled depth,
/// which every batch's plan then looks at), alternating, four times each. Printed per round: the
/// process's CPU time and the wall time a frame, averaged over its [`CPU_FRAMES`] frames. Ignored
/// by default, as [`depth_costs`] is.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn held_depth_cpu() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let name = device.name().to_owned();
    let moves = (device.properties().driver_facts)
        .contains(&DriverFact::MultisampledDepthFasterOutOfGeneral);
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let frame = || {
        let command_buffer = queue.command_buffer().unwrap();
        for value in 0..CPU_ENCODERS {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..4096, value as u8).unwrap();
            blit.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };
    for held in [false, true, false, true, false, true, false, true] {
        let depth = held.then(|| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 64, 64, false);
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = 4;
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        });
        for _ in 0..CPU_WARM {
            frame();
        }
        let cpu = process_cpu_us();
        let started = Instant::now();
        for _ in 0..CPU_FRAMES {
            frame();
        }
        let wall = started.elapsed().as_secs_f64() * 1e6 / CPU_FRAMES as f64;
        let cpu = (process_cpu_us() - cpu) / CPU_FRAMES as f64;
        eprintln!(
            "HELD-DEPTH-CPU on {name} (MultisampledDepthFasterOutOfGeneral {moves}): held depth \
             alive {held}: CPU {cpu:.2} us a frame of {CPU_ENCODERS} blits, wall {wall:.2} us"
        );
        drop(depth);
    }
}
