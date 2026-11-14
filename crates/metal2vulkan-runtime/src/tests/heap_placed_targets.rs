//! A frame's render targets placed once in one tracked `Private` placement heap, each at the
//! offset the device's answers lay it out at (`Device::heap_texture_size_and_align`), as a port of
//! placed resources makes them, against the same textures made by the device. The frame: a
//! depth-only pass into a shadow map; a pass into a 4-sample color and depth, both cleared and
//! neither stored, that resolves the color and samples the shadow map; a kernel that reads the
//! resolve and one that writes a texture; a pass into a second-format view of another texture;
//! and a pass into an sRGB view of the final target, of no load, whose draws sample those two.
//!
//! Every texture of the frame lies in its own bytes of the heap, so the heap's frame reads and
//! writes what the device's does (the final target's texels and the kernel's reads of the
//! resolve alike) and needs no copy the device's frame does not make: no draw reads a copy of a
//! texture it does not render (`WorkCounts::attached_read_slots`), none announces a use of its
//! attachments' memory (`Degradation::AttachmentReadInPlace`), and a texture whose image lies in
//! the heap's memory is no shadow of it, so none of the shadows' copies, sample moves or image
//! copies run (`WorkCounts::shadow_copies`, `heap_sample_move_images_lent`, `vkCmdCopyImage`). A
//! texture held as a shadow (an image the heap's memory takes at no answer the device gives) is
//! printed; under the default `HeapAnswerPolicy::Device` neither multisample target is one, each
//! in its image at its device's answer (`texture/heap.rs`'s `device_heap_answer`; an RX 7900 XTX
//! aligns a 4-sample image past 128 texels a side to 256 KiB, past every Mac's answer), and under
//! `HeapAnswerPolicy::Metal` the frame clears them whole, so it moves none of their samples in
//! (`render/shadows.rs`). Each lane prints its counted frames' mean wall time, a figure of
//! release builds alone.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures'
//! `tob_rect`, `tob_solid` and `tob_sample` (`engine/tob_renders.ll`), `pdc_vertex` and
//! `pdc_constexpr` (`engine/pdc_compare_clamp.ll`), and `a8r_write` and `a8r_read`
//! (`engine/a8r_targets.ll`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::Heard;
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Degradation, DepthStencilState, Device, DeviceKind, DeviceOptions, HazardTrackingMode, Heap,
    HeapDescriptor, HeapType, Library, LoadAction, MessageSink, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, ShaderCacheOptions,
    Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::HeapAnswerPolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The frame's extent: past 128 texels a side, where an RX 7900 XTX aligns a multisample image to
/// 256 KiB.
const WIDTH: u64 = 320;
const HEIGHT: u64 = 192;

/// The shadow map's side.
const SHADOW: u64 = 256;

/// The frames each lane runs, and the first one counted and timed: the first makes the pipelines'
/// first uses and the shadows' first copies.
const FRAMES: u64 = 12;
const COUNTED_FROM: u64 = 2;

/// The kernel `a8r_read` reads the resolve at 8 x 8 texels.
const READ_SIDE: u64 = 8;

/// The textures of the frame, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    MsaaColor,
    MsaaDepth,
    Hdr,
    Final,
    Ui,
    Post,
    Shadow,
}

const TARGETS: [Target; 7] = [
    Target::MsaaColor,
    Target::MsaaDepth,
    Target::Hdr,
    Target::Final,
    Target::Ui,
    Target::Post,
    Target::Shadow,
];

/// A tracked `Private` heap's options, its textures' too.
fn tracked() -> ResourceOptions {
    ResourceOptions::private().with_hazard_tracking_mode(HazardTrackingMode::Tracked)
}

/// The descriptor of `target`.
fn descriptor(target: Target) -> TextureDescriptor {
    let read_target = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let (pf, side, usage) = match target {
        Target::MsaaColor => (PixelFormat::RGBA16Float, None, TextureUsage::RENDER_TARGET),
        Target::MsaaDepth => (PixelFormat::Depth32Float, None, TextureUsage::RENDER_TARGET),
        Target::Hdr => (PixelFormat::RGBA16Float, None, read_target),
        Target::Final => (PixelFormat::RGBA8Unorm, None, read_target),
        Target::Ui => (
            PixelFormat::RGBA8Unorm,
            None,
            read_target | TextureUsage::PIXEL_FORMAT_VIEW,
        ),
        Target::Post => (
            PixelFormat::RGBA8Unorm,
            None,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ),
        Target::Shadow => (PixelFormat::Depth32Float, Some(SHADOW), read_target),
    };
    let (width, height) = side.map_or((WIDTH, HEIGHT), |side| (side, side));
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = tracked();
    if matches!(target, Target::MsaaColor | Target::MsaaDepth) {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = 4;
    }
    descriptor
}

/// A device of the kind the cfg selects, its heap textures answered under `answers`, that
/// announces `AttachmentReadInPlace` at every occurrence, its notices into `heard`.
fn device(cache: &Path, answers: HeapAnswerPolicy, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
    let mut options = DeviceOptions {
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        announce_each: [Degradation::AttachmentReadInPlace].into_iter().collect(),
        ..DeviceOptions::default()
    };
    options.policy.heap_answers = answers;
    Device::system_default_with(&options).unwrap()
}

/// The frame's textures, placed in one heap (`heap`) or made by the device.
struct Targets {
    textures: Vec<Texture>,
    /// The heap's size, where they lie in one.
    heap: Option<(Heap, u64)>,
}

impl Targets {
    /// Each of [`TARGETS`] placed one after another at its answered alignment in one heap, as a
    /// program places them by the answers.
    fn placed(device: &Device) -> Targets {
        let mut offsets = Vec::new();
        let mut end: u64 = 0;
        for target in TARGETS {
            let answer = device.heap_texture_size_and_align(&descriptor(target));
            let answer = answer.unwrap();
            let offset = end.next_multiple_of(answer.align);
            offsets.push(offset);
            end = offset + answer.size;
        }
        let heap = HeapDescriptor::new(HeapType::Placement, end, tracked());
        let heap = device.new_heap(&heap).unwrap();
        let textures = (TARGETS.iter().zip(offsets))
            .map(|(target, offset)| heap.new_texture_at_offset(&descriptor(*target), offset))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        Targets {
            textures,
            heap: Some((heap, end)),
        }
    }

    /// Each of [`TARGETS`] made by the device.
    fn made(device: &Device) -> Targets {
        let textures = TARGETS
            .iter()
            .map(|target| device.new_texture(&descriptor(*target)).unwrap())
            .collect();
        Targets {
            textures,
            heap: None,
        }
    }

    fn get(&self, target: Target) -> &Texture {
        &self.textures[target as usize]
    }
}

/// The device's pipelines and states.
struct Tools {
    queue: CommandQueue,
    depth_only: RenderPipelineState,
    scene: RenderPipelineState,
    ui: RenderPipelineState,
    composite: RenderPipelineState,
    write: ComputePipelineState,
    read: ComputePipelineState,
    depth_less: DepthStencilState,
    sampler: SamplerState,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let renders = library(device, "engine/tob_renders.ll");
        let compares = library(device, "engine/pdc_compare_clamp.ll");
        let targets = library(device, "engine/a8r_targets.ll");
        let function = |library: &Library, name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let render = |vertex, fragment: Option<StageSource>, colors: Option<PixelFormat>| {
            let mut descriptor = RenderPipelineDescriptor::new(vertex);
            descriptor.fragment = fragment;
            descriptor.color_attachments[0] = colors.map(ColorAttachmentDescriptor::new);
            descriptor
        };
        let options = PipelineOptions::default();
        let made = |descriptor: &RenderPipelineDescriptor| {
            (device.new_render_pipeline_state(descriptor, &options)).unwrap()
        };
        let mut depth_only = render(function(&renders, "tob_rect"), None, None);
        depth_only.depth_format = Some(PixelFormat::Depth32Float);
        let mut scene = render(
            function(&compares, "pdc_vertex"),
            Some(function(&compares, "pdc_constexpr")),
            Some(PixelFormat::RGBA16Float),
        );
        scene.depth_format = Some(PixelFormat::Depth32Float);
        scene.raster_sample_count = 4;
        let ui = render(
            function(&renders, "tob_rect"),
            Some(function(&renders, "tob_solid")),
            Some(PixelFormat::BGRA8Unorm),
        );
        let composite = render(
            function(&renders, "tob_rect"),
            Some(function(&renders, "tob_sample")),
            Some(PixelFormat::RGBA8UnormSRGB),
        );
        let kernel = |name| {
            let descriptor = ComputePipelineDescriptor::new(function(&targets, name));
            (device.new_compute_pipeline_state(&descriptor, &options)).unwrap()
        };
        let mut less = DepthStencilDescriptor::default();
        less.depth_compare_function = CompareFunction::Less;
        less.depth_write_enabled = true;
        Tools {
            queue: device.new_command_queue().unwrap(),
            depth_only: made(&depth_only),
            scene: made(&scene),
            ui: made(&ui),
            composite: made(&composite),
            write: kernel("a8r_write"),
            read: kernel("a8r_read"),
            depth_less: device.new_depth_stencil_state(&less).unwrap(),
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
        }
    }
}

/// The bytes of a rectangle from corner `(x0, y0)` to `(x1, y1)` (`tob_rect`'s `r`).
fn rect(r: [f32; 4]) -> Vec<u8> {
    bytes_of(&r, f32::to_le_bytes)
}

/// One frame into `targets`, its final target's texels blitted into `back` and the kernel's
/// reads of the resolve into `read`.
fn frame(tools: &Tools, targets: &Targets, (back, read): (&Buffer, &Buffer)) {
    let final_view = (targets.get(Target::Final))
        .new_texture_view(PixelFormat::RGBA8UnormSRGB)
        .unwrap();
    let ui_view = (targets.get(Target::Ui))
        .new_texture_view(PixelFormat::BGRA8Unorm)
        .unwrap();
    let command_buffer = tools.queue.command_buffer().unwrap();
    // The shadow pass: the left half of the map at depth 0.
    let mut pass = RenderPassDescriptor::new();
    let depth = &mut pass.depth_attachment;
    depth.texture = Some(targets.get(Target::Shadow).clone());
    depth.load_action = LoadAction::Clear;
    depth.clear_depth = 1.0;
    depth.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder
        .set_render_pipeline_state(&tools.depth_only)
        .unwrap();
    encoder.set_depth_stencil_state(&tools.depth_less).unwrap();
    encoder
        .set_vertex_bytes(&rect([-1.0, -1.0, 0.0, 1.0]), 0)
        .unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    // The scene: 4 samples cleared, the color resolved, neither stored; the draw compares the
    // shadow map's centre.
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(targets.get(Target::MsaaColor).clone());
    color.resolve_texture = Some(targets.get(Target::Hdr).clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.05, 0.07, 0.1, 1.0);
    color.store_action = StoreAction::MultisampleResolve;
    let depth = &mut pass.depth_attachment;
    depth.texture = Some(targets.get(Target::MsaaDepth).clone());
    depth.load_action = LoadAction::Clear;
    depth.clear_depth = 1.0;
    depth.store_action = StoreAction::DontCare;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&tools.scene).unwrap();
    encoder.set_depth_stencil_state(&tools.depth_less).unwrap();
    let refs: [f32; 4] = [0.5, 0.5, 0.0, 0.0];
    (encoder.set_fragment_bytes(&bytes_of(&refs, f32::to_le_bytes), 0)).unwrap();
    (encoder.set_fragment_texture(Some(targets.get(Target::Shadow)), 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    // The kernels: one reads the resolve, one writes the post target.
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&tools.read).unwrap();
    encoder
        .set_texture(Some(targets.get(Target::Hdr)), 0)
        .unwrap();
    encoder.set_buffer(Some(read), 0, 1).unwrap();
    let grid = Size::new(READ_SIDE, READ_SIDE, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.set_compute_pipeline_state(&tools.write).unwrap();
    encoder
        .set_texture(Some(targets.get(Target::Post)), 0)
        .unwrap();
    let post: [f32; 4] = [0.75, 0.25, 0.5, 1.0];
    (encoder.set_bytes(&bytes_of(&post, f32::to_le_bytes), 0)).unwrap();
    let all = Size::new(WIDTH, HEIGHT, 1);
    encoder.dispatch_threads(all, Size::new(8, 8, 1)).unwrap();
    encoder.end_encoding().unwrap();
    // The UI: its second-format view cleared, a rectangle drawn.
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(ui_view);
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&tools.ui).unwrap();
    encoder
        .set_vertex_bytes(&rect([-0.5, -0.5, 0.5, 0.5]), 0)
        .unwrap();
    let solid: [f32; 4] = [0.2, 0.4, 0.6, 1.0];
    (encoder.set_fragment_bytes(&bytes_of(&solid, f32::to_le_bytes), 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    // The composite into the final target's sRGB view: the post target sampled on the left, the
    // UI on the right.
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(final_view);
    color.load_action = LoadAction::DontCare;
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&tools.composite).unwrap();
    (encoder.set_fragment_sampler_state(Some(&tools.sampler), 0)).unwrap();
    for (half, read) in [
        ([-1.0, -1.0, 0.0, 1.0], Target::Post),
        ([0.0, -1.0, 1.0, 1.0], Target::Ui),
    ] {
        encoder.set_vertex_bytes(&rect(half), 0).unwrap();
        (encoder.set_fragment_texture(Some(targets.get(read)), 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    }
    encoder.end_encoding().unwrap();
    // The read-back.
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let row = WIDTH * 4;
    (blit.copy_from_texture_to_buffer(
        targets.get(Target::Final),
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(WIDTH, HEIGHT, 1),
        back,
        0,
        row,
        row * HEIGHT,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// What the counted frames did, each a frame's mean: the counts that differ between a heap's
/// frame and the device's.
#[derive(Debug, Default, PartialEq, Eq)]
struct Done {
    attached_read_slots: u64,
    attachment_reads_through_copies: u64,
    shadow_copies: u64,
    shadow_attachment_textures: u64,
    shadow_attachment_clears_in_part: u64,
    heap_sample_move_images_lent: u64,
    image_copies: u64,
    buffer_image_copies: u64,
}

impl Done {
    fn of(before: &WorkCounts, after: &WorkCounts, frames: u64) -> Done {
        let delta = |count: fn(&WorkCounts) -> u64| (count(after) - count(before)) / frames;
        let calls = |counts: &WorkCounts, command: &str| -> u64 {
            (counts.copy_sites.iter())
                .filter(|site| site.command == command)
                .map(|site| site.calls)
                .sum()
        };
        let copies = |command: &str| (calls(after, command) - calls(before, command)) / frames;
        Done {
            attached_read_slots: delta(|c| c.attached_read_slots),
            attachment_reads_through_copies: delta(|c| c.attachment_reads_through_copies),
            shadow_copies: delta(|c| c.shadow_copies),
            shadow_attachment_textures: delta(|c| c.shadow_attachment_textures),
            shadow_attachment_clears_in_part: delta(|c| c.shadow_attachment_clears_in_part),
            heap_sample_move_images_lent: delta(|c| c.heap_sample_move_images_lent),
            image_copies: copies("vkCmdCopyImage"),
            buffer_image_copies: copies("vkCmdCopyBufferToImage"),
        }
    }
}

/// What a lane of [`FRAMES`] frames read and did.
struct Ran {
    texels: Vec<u8>,
    read: Vec<u8>,
    done: Done,
    announced: usize,
    /// The textures held as shadows of the heap's bytes, by name.
    shadows: Vec<Target>,
    /// The counted frames' mean wall time, each committed and waited for: a figure of release
    /// builds alone, printed, never checked.
    frame: Duration,
}

/// [`FRAMES`] frames into `targets`.
fn run(device: &Device, heard: &Heard, tools: &Tools, targets: &Targets) -> Ran {
    let shared = ResourceOptions::shared();
    let back = device.new_buffer(WIDTH * 4 * HEIGHT, shared).unwrap();
    let read = (device.new_buffer(READ_SIDE * READ_SIDE * 16, shared)).unwrap();
    let mut before = None;
    for at in 0..FRAMES {
        if at == COUNTED_FROM {
            before = Some((device.work_counts(), Instant::now()));
        }
        frame(tools, targets, (&back, &read));
    }
    let (before, started) = before.unwrap();
    let frame = started.elapsed() / (FRAMES - COUNTED_FROM) as u32;
    let done = Done::of(&before, &device.work_counts(), FRAMES - COUNTED_FROM);
    let announced = (heard.lock().unwrap().iter())
        .filter(|notice| notice.degradation == Degradation::AttachmentReadInPlace)
        .count();
    let shadows = (TARGETS.iter().zip(&targets.textures))
        .filter(|(_, texture)| texture.shadow().is_some())
        .map(|(target, _)| *target)
        .collect();
    let bytes = |buffer: &Buffer| {
        let contents = buffer.contents().unwrap();
        let length = buffer.length() as usize;
        // SAFETY: the buffer's bytes, which no command buffer writes now.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
    };
    Ran {
        texels: bytes(&back),
        read: bytes(&read),
        done,
        announced,
        shadows,
        frame,
    }
}

/// A heap's frames and the device's, under `answers`.
fn lanes(leg: &str, answers: HeapAnswerPolicy) -> (Ran, Ran) {
    let scratch = Scratch::new(leg);
    let heard = Heard::default();
    let device = device(&scratch.0, answers, &heard);
    let tools = Tools::new(&device);
    let made = run(&device, &heard, &tools, &Targets::made(&device));
    heard.lock().unwrap().clear();
    let targets = Targets::placed(&device);
    let placed = run(&device, &heard, &tools, &targets);
    let size = targets.heap.as_ref().map(|(_, size)| *size);
    let answers = TARGETS.map(|target| device.heap_texture_size_and_align(&descriptor(target)));
    eprintln!(
        "{leg}: {:?} heap of {size:?} bytes, answers {answers:?}, shadows {:?}; a frame: placed \
         {:?} in {:.3} ms, made {:?} in {:.3} ms",
        device.name(),
        placed.shadows,
        placed.done,
        placed.frame.as_secs_f64() * 1e3,
        made.done,
        made.frame.as_secs_f64() * 1e3,
    );
    assert_eq!(
        placed.texels, made.texels,
        "{leg}: the final target's texels"
    );
    assert_eq!(
        placed.read, made.read,
        "{leg}: the kernel's reads of the resolve"
    );
    (placed, made)
}

#[test]
fn a_frame_of_targets_placed_apart_in_a_heap_copies_nothing_the_device_s_does_not() {
    let (placed, made) = lanes("heap-placed-targets", HeapAnswerPolicy::Device);
    assert_eq!(
        placed.announced, 0,
        "reads of an attachment's memory announced"
    );
    assert_eq!(placed.done.attached_read_slots, 0, "{:?}", placed.done);
    assert_eq!(made.done, Done::default(), "the device's frame");
    for target in [Target::MsaaColor, Target::MsaaDepth] {
        assert!(!placed.shadows.contains(&target), "{target:?} a shadow");
    }
    if placed.shadows.is_empty() {
        assert_eq!(placed.done, made.done, "a frame of no shadow");
    }
}

#[test]
fn a_frame_under_metal_s_answers_moves_no_cleared_multisample_target_s_samples_in() {
    let (placed, _) = lanes("heap-placed-targets-metal", HeapAnswerPolicy::Metal);
    assert_eq!(
        placed.announced, 0,
        "reads of an attachment's memory announced"
    );
    assert_eq!(
        placed.done.shadow_attachment_clears_in_part, 0,
        "{:?}",
        placed.done
    );
    assert_eq!(
        placed.done.heap_sample_move_images_lent, 0,
        "{:?}",
        placed.done
    );
}
