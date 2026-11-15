//! What framebuffer fetch through the pixel interlock costs a pass at a compositor's shape, on the
//! device the cfg selects, without the validation layer: report only, in release builds, to weigh
//! the per-pass work the interlock form adds (`metal2vulkan_runtime`'s `framebuffer_fetch` and
//! `attachment_stand_in`) before changing it.
//!
//! Every target is a private 1920x1080 `RGBA16Float` texture, drawn by the fullscreen triangle
//! (`rt6_resolves`' `rt6_resolve_full`):
//!
//! - **`STORAGE`.** `plain` (render target and shader read usage, which the runtime gives
//!   `STORAGE` only where it costs the image nothing) and `written` (shader write usage too, so
//!   `STORAGE` always): each one's size and alignment as a heap places it, a pass of 32 blended
//!   draws into it (`rt6_resolve_white`, `One` and `OneMinusSourceAlpha`), a draw sampling it
//!   into another target (`tob_sample`), and a copy of it into another texture and back.
//! - **The fetch.** A pass loading and storing the target with one draw that fetches it
//!   (`exf_fetch_blend`): `plain` through a scratch image the pass copies it into and back out of
//!   (where the device gives it no `STORAGE`), `written` directly.
//! - **The split.** On `written`: a pass of a blended draw then a fetching one (two render pass
//!   instances in the interlock form), beside passes of two blended draws and of two fetching
//!   draws (one instance each).
//! - **The memoryless backing.** A pass clearing a memoryless texture and fetching it once (the
//!   pass renders into a backing image made for it), beside the same pass over `written` stored
//!   `DontCare`.
//!
//! Each case times one command buffer [`RUNS`] times after one untimed run: its GPU start to end,
//! the host's wall time from its commit until its wait returns, and the host's encoding time
//! (the encoders' creation to their end). MoltenVK's GPU times may leave out work of the command
//! buffer (`mipmap_costs`), so the wall time bounds the GPU's from above. Printed per case: the
//! medians and the fastest runs, and the notices the case announced (each kind once per device,
//! so at its first case).
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::degrade_notices::counts;
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, CpuCacheMode, Device, HazardTrackingMode, Library,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use std::time::Instant;

/// The timed runs of each case.
const RUNS: usize = 9;

/// The targets' sides: a 1080p compositor's.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;

/// The targets' format.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// The fullscreen triangle's depth.
const Z: f32 = 0.5;

/// The engine fixture `stem`'s library on `device`.
fn library(device: &Device, stem: &str) -> Library {
    let ll = metal2vulkan_devtools::engine_dir().join(format!("{stem}.ll"));
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The pipelines and state the cases draw with.
struct Kit {
    device: Device,
    queue: CommandQueue,
    /// Blended white: `One`, `OneMinusSourceAlpha`.
    white: RenderPipelineState,
    /// `exf_fetch_blend`: half of what it reads plus the pixel's position.
    fetch: RenderPipelineState,
    /// `tob_sample`: the texture at index 0 sampled at the pixel.
    sample: RenderPipelineState,
    sampler: SamplerState,
    /// The fullscreen triangle's depth, at vertex buffer 0.
    depth: Buffer,
}

impl Kit {
    fn new(device: Device) -> Kit {
        let resolves = library(&device, "rt6_resolves");
        let blend = library(&device, "exf_fetch_blend");
        let renders = library(&device, "tob_renders");
        let function = |library: &Library, name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |library: &Library, name: &str, blended: bool| {
            let mut descriptor =
                RenderPipelineDescriptor::new(function(&resolves, "rt6_resolve_full"));
            descriptor.fragment = Some(function(library, name));
            let mut color = ColorAttachmentDescriptor::new(FORMAT);
            if blended {
                color.blending_enabled = true;
                color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            }
            descriptor.color_attachments[0] = Some(color);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        Kit {
            queue: device.new_command_queue().unwrap(),
            white: pipeline(&resolves, "rt6_resolve_white", true),
            fetch: pipeline(&blend, "exf_fetch_blend", false),
            sample: pipeline(&renders, "tob_sample", false),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            depth: buffer_with(&device, &Z.to_le_bytes()).unwrap(),
            device,
        }
    }

    /// A 1920x1080 texture of `usage` in `storage`.
    fn texture(&self, usage: TextureUsage, storage: StorageMode) -> Texture {
        self.device
            .new_texture(&descriptor(usage, storage))
            .unwrap()
    }
}

/// The descriptor of a 1920x1080 texture of `usage` in `storage`.
fn descriptor(usage: TextureUsage, storage: StorageMode) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(FORMAT, WIDTH, HEIGHT, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    d
}

/// A one-target pass of `texture`, loaded by `load` and stored by `store`.
fn pass(texture: &Texture, load: LoadAction, store: StoreAction) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.load_action = load;
    target.store_action = store;
    descriptor
}

/// What one case encodes into its command buffer.
type Encode<'a> = dyn Fn(&Kit, &CommandBuffer) + 'a;

/// One pass of `descriptor` drawing the fullscreen triangle once with each of `steps`.
fn draws<'a>(
    descriptor: RenderPassDescriptor,
    steps: Vec<&'a RenderPipelineState>,
) -> Box<Encode<'a>> {
    Box::new(move |kit: &Kit, command_buffer: &CommandBuffer| {
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_vertex_buffer(Some(&kit.depth), 0, 0).unwrap();
        for step in &steps {
            encoder.set_render_pipeline_state(step).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    })
}

/// One pass into `into` drawing `source` sampled at each pixel.
fn sampling<'a>(source: &'a Texture, into: &'a Texture) -> Box<Encode<'a>> {
    Box::new(move |kit: &Kit, command_buffer: &CommandBuffer| {
        let descriptor = pass(into, LoadAction::DontCare, StoreAction::Store);
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_render_pipeline_state(&kit.sample).unwrap();
        encoder.set_vertex_buffer(Some(&kit.depth), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(source), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&kit.sampler), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    })
}

/// One blit copying the whole of `source` into `destination`.
fn copy<'a>(source: &'a Texture, destination: &'a Texture) -> Box<Encode<'a>> {
    Box::new(move |_: &Kit, command_buffer: &CommandBuffer| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_texture(
            source,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            destination,
            0,
            0,
            Origin::new(0, 0, 0),
        )
        .unwrap();
        blit.end_encoding().unwrap();
    })
}

/// One timed run's GPU, wall and encoding times, in microseconds.
struct Times {
    gpu: f64,
    wall: f64,
    encoding: f64,
}

/// The times of one command buffer of `encode`.
fn run(kit: &Kit, encode: &Encode<'_>) -> Times {
    let command_buffer = kit.queue.command_buffer().unwrap();
    let encoding = Instant::now();
    encode(kit, &command_buffer);
    let encoded = encoding.elapsed().as_secs_f64() * 1e6;
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
        encoding: encoded,
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
fn interlocked_fetch_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let name = device.name().to_owned();
    let kit = Kit::new(device);
    let target = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let written_usage = target | TextureUsage::SHADER_WRITE;
    for (label, usage) in [("plain", target), ("written", written_usage)] {
        let placed = kit
            .device
            .heap_texture_size_and_align(&descriptor(usage, StorageMode::Private))
            .unwrap();
        eprintln!(
            "FETCH-COST on {name}: {label} {WIDTH}x{HEIGHT} {}: {} bytes, aligned {}",
            FORMAT.name(),
            placed.size,
            placed.align
        );
    }
    let plain = kit.texture(target, StorageMode::Private);
    let written = kit.texture(written_usage, StorageMode::Private);
    let other = kit.texture(target, StorageMode::Private);
    let memoryless = kit.texture(TextureUsage::RENDER_TARGET, StorageMode::Memoryless);
    let loaded = |t: &Texture| pass(t, LoadAction::Load, StoreAction::Store);
    let cleared = |t: &Texture| pass(t, LoadAction::Clear, StoreAction::DontCare);
    let cases: Vec<(String, Box<Encode<'_>>)> = vec![
        (
            "plain: 32 blended draws".into(),
            draws(loaded(&plain), vec![&kit.white; 32]),
        ),
        (
            "written: 32 blended draws".into(),
            draws(loaded(&written), vec![&kit.white; 32]),
        ),
        ("plain: sampled".into(), sampling(&plain, &other)),
        ("written: sampled".into(), sampling(&written, &other)),
        ("plain: copied from".into(), copy(&plain, &other)),
        ("written: copied from".into(), copy(&written, &other)),
        ("plain: copied into".into(), copy(&other, &plain)),
        ("written: copied into".into(), copy(&other, &written)),
        (
            "plain: one fetching draw, loaded and stored".into(),
            draws(loaded(&plain), vec![&kit.fetch]),
        ),
        (
            "written: one fetching draw, loaded and stored".into(),
            draws(loaded(&written), vec![&kit.fetch]),
        ),
        (
            "written: blended, blended".into(),
            draws(loaded(&written), vec![&kit.white, &kit.white]),
        ),
        (
            "written: blended, fetching".into(),
            draws(loaded(&written), vec![&kit.white, &kit.fetch]),
        ),
        (
            "written: fetching, fetching".into(),
            draws(loaded(&written), vec![&kit.fetch, &kit.fetch]),
        ),
        (
            "memoryless: cleared, one fetching draw".into(),
            draws(cleared(&memoryless), vec![&kit.fetch]),
        ),
        (
            "written: cleared, one fetching draw, stored DontCare".into(),
            draws(cleared(&written), vec![&kit.fetch]),
        ),
    ];
    for (case, encode) in &cases {
        let _ = run(&kit, encode.as_ref());
        let notices = counts(&messages.take_notices());
        let runs: Vec<Times> = (0..RUNS).map(|_| run(&kit, encode.as_ref())).collect();
        let (gpu, gpu_least) = median_and_least(runs.iter().map(|r| r.gpu).collect());
        let (wall, wall_least) = median_and_least(runs.iter().map(|r| r.wall).collect());
        let (encoding, _) = median_and_least(runs.iter().map(|r| r.encoding).collect());
        eprintln!(
            "FETCH-COST on {name}: {case}: GPU median {gpu:.1} us, fastest {gpu_least:.1} us; \
             wall median {wall:.1} us, fastest {wall_least:.1} us; encoding median \
             {encoding:.1} us; announced {notices:?}"
        );
    }
}
