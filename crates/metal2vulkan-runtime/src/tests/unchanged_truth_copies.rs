//! Frames over a 2618 x 1024 `BGRA8Unorm` texture whose texels change only every
//! [`EVERY`]th frame, read through a shadow (`texture/usage_shadow.rs`): (a) by a draw that
//! samples it in the render pass that renders it (`texture/attached_reads.rs`), and (b) by a draw
//! that samples it beyond its declared usage. The texture is a `Private` texture of its own, or a
//! surface plane in a `Private` buffer the device made. Each frame is a command buffer of its own;
//! a change is a pass that clears the texture to a color of its own, in a command buffer before
//! the frame's.
//!
//! Each frame counts its copies in (`WorkCounts::shadow_copies`), the texels they moved
//! (`shadow_texels_copied_in` and `shadow_texels_copied_in_images`), those a standing took off
//! (`shadow_texels_skipped_by_standing`) and its GPU time, and checks the texel the frame's draw
//! wrote: the last change's color.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_sample` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, IoSurfacePlane, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ImageUsageFlags;
use std::time::Duration;

/// The texture: a guest's tile.
const WIDTH: u64 = 2618;
const HEIGHT: u64 = 1024;
/// A surface plane's rows: `WIDTH` texels of 4 bytes, padded to 64 bytes.
const ROW: u64 = (WIDTH * 4).div_ceil(64) * 64;
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The frames, and every how many frames the texture's texels change.
const FRAMES: u32 = 12;
const EVERY: u32 = 4;

/// The scissor a small attached read draws through.
const SMALL: ScissorRect = ScissorRect {
    x: 512,
    y: 256,
    width: 64,
    height: 64,
};

/// The texel each frame's draw writes that the frame checks: inside [`SMALL`].
const PROBE: (u64, u64) = (540, 280);

/// What holds the texture's texels.
#[derive(Clone, Copy, Debug)]
enum Truth {
    /// A `Private` texture of its own.
    Own,
    /// A surface plane in a `Private` buffer the device made.
    Plane,
}

/// What reads the texture through its shadow.
#[derive(Clone, Copy, Debug)]
enum Need {
    /// A draw of the pass that renders the texture samples it, over every pixel.
    AttachedWhole,
    /// The same through [`SMALL`].
    AttachedSmall,
    /// A draw into another target samples it, its usage without `ShaderRead`.
    BeyondUsage,
}

/// The red the change at frame `frame` clears the texture to.
fn red_of(frame: u32) -> u8 {
    40 + 15 * (frame / EVERY) as u8
}

struct Tools {
    queue: CommandQueue,
    sample: RenderPipelineState,
    sampler: SamplerState,
    back: Buffer,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let renders = library(device, "engine/tob_renders.ll");
        let vertex = renders.new_function_with_name("tob_rect").unwrap();
        let fragment = renders.new_function_with_name("tob_sample").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
        let sample = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        Tools {
            queue: device.new_command_queue().unwrap(),
            sample,
            sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            back: device.new_buffer(4, ResourceOptions::shared()).unwrap(),
        }
    }

    /// The texture of `truth` with `usage`.
    fn texture(&self, truth: Truth, usage: TextureUsage) -> (Texture, Option<Buffer>) {
        let device = self.queue.device();
        let mut descriptor = TextureDescriptor::texture_2d(FORMAT, WIDTH, HEIGHT, false);
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        match truth {
            Truth::Own => (device.new_texture(&descriptor).unwrap(), None),
            Truth::Plane => {
                let buffer = device
                    .new_buffer(ROW * HEIGHT, ResourceOptions::private())
                    .unwrap();
                let plane = IoSurfacePlane::new(0, ROW, true);
                let texture = buffer
                    .new_texture_with_iosurface_plane(&descriptor, plane)
                    .unwrap();
                (texture, Some(buffer))
            }
        }
    }

    /// A pass that clears `texture` to `red` and stores it, in a command buffer of its own.
    fn change(&self, texture: &Texture, red: u8) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(f64::from(red) / 255.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// A frame: a pass loading and storing `target`, a draw of `tob_sample` over it sampling
    /// `texture`, through [`SMALL`] where `small`. Its work counts before and after, and its GPU
    /// time in microseconds.
    fn frame(
        &self,
        target: &Texture,
        texture: &Texture,
        small: bool,
    ) -> (WorkCounts, WorkCounts, f64) {
        let device = self.queue.device();
        let before = device.work_counts();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.sample).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        (encoder.set_fragment_sampler_state(Some(&self.sampler), 0)).unwrap();
        if small {
            encoder.set_scissor_rect(SMALL).unwrap();
        }
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let gpu = match (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        ) {
            (Ok(start), Ok(end)) => (end - start) * 1e6,
            _ => f64::NAN,
        };
        (before, device.work_counts(), gpu)
    }

    /// The red of `texture`'s texel at [`PROBE`].
    fn red_at_probe(&self, texture: &Texture) -> u8 {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (x, y) = PROBE;
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(x, y, 0),
            Size::new(1, 1, 1),
            &self.back,
            0,
            4,
            4,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's 4 bytes, which no command writes now.
        let texel = unsafe { std::slice::from_raw_parts(bytes, 4) };
        // `BGRA8Unorm`'s red is its third byte.
        texel[2]
    }
}

/// One frame's counts.
#[derive(Debug, Default, Clone, Copy)]
struct Frame {
    copies: u64,
    texels: u64,
    skipped: u64,
    through_copies: u64,
    own_pixel: u64,
    snapshotted: u64,
    gpu_us: f64,
}

impl Frame {
    fn of((before, after, gpu_us): (WorkCounts, WorkCounts, f64)) -> Frame {
        let delta = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
        Frame {
            copies: delta(|c| c.shadow_copies),
            texels: delta(|c| c.shadow_texels_copied_in)
                + delta(|c| c.shadow_texels_copied_in_images),
            skipped: delta(|c| c.shadow_texels_skipped_by_standing),
            through_copies: delta(|c| c.attachment_reads_through_copies),
            own_pixel: delta(|c| c.own_pixel_reads),
            snapshotted: delta(|c| c.reading_draws_snapshotted),
            gpu_us,
        }
    }
}

/// [`FRAMES`] frames of `need` over a texture of `truth`: each frame's counts, the frames with a
/// change and those without apart.
fn frames(device: &Device, tools: &Tools, truth: Truth, need: Need) -> (Vec<Frame>, Vec<Frame>) {
    let usage = match need {
        Need::BeyondUsage => TextureUsage::RENDER_TARGET,
        Need::AttachedWhole | Need::AttachedSmall => {
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ
        }
    };
    let (texture, _buffer) = tools.texture(truth, usage);
    let target = match need {
        Need::BeyondUsage => tools.texture(Truth::Own, TextureUsage::RENDER_TARGET).0,
        Need::AttachedWhole | Need::AttachedSmall => texture.clone(),
    };
    let small = matches!(need, Need::AttachedSmall);
    let (mut changed, mut unchanged) = (Vec::new(), Vec::new());
    for frame in 0..FRAMES {
        let change = frame % EVERY == 0;
        if change {
            tools.change(&texture, red_of(frame));
        }
        let counts = Frame::of(tools.frame(&target, &texture, small));
        let red = tools.red_at_probe(&target);
        let plan = texture.image_plan().ok();
        let feedback = plan.is_some_and(|p| {
            p.usage
                .contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT)
        });
        eprintln!(
            "UNCHANGED-TRUTH {} {truth:?} {need:?} frame {frame} change {change}: {counts:?}; \
             red {red}; own image shadowed {}, usage shadow {}, feedback usage {feedback}",
            device.name(),
            texture.shadow().is_some(),
            texture.usage_shadow_copies().is_some(),
        );
        assert_eq!(
            red,
            red_of(frame),
            "frame {frame}'s draw read the last change"
        );
        match change {
            true => changed.push(counts),
            false => unchanged.push(counts),
        }
    }
    (changed, unchanged)
}

/// Every pair of truth and need: the copies in of frames that changed the texture and of those
/// that did not, summed.
#[test]
fn shadows_of_an_unchanged_truth_copy_in_per_frame() {
    let scratch = Scratch::new("unchanged-truth");
    let device = device(&scratch.0);
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    let tools = Tools::new(&device);
    eprintln!(
        "UNCHANGED-TRUTH {}: in-place checks settled {settled}, snapshot path {}",
        device.name(),
        device.shared().color_write_masks.is_some(),
    );
    for truth in [Truth::Own, Truth::Plane] {
        for need in [Need::AttachedWhole, Need::AttachedSmall, Need::BeyondUsage] {
            let (changed, unchanged) = frames(&device, &tools, truth, need);
            let sum = |frames: &[Frame]| {
                let n = frames.len().max(1) as f64;
                let texels: u64 = frames.iter().map(|f| f.texels).sum();
                let copies: u64 = frames.iter().map(|f| f.copies).sum();
                let gpu: f64 = frames.iter().map(|f| f.gpu_us).sum::<f64>() / n;
                (copies, texels, gpu)
            };
            eprintln!(
                "UNCHANGED-TRUTH-SUM {} {truth:?} {need:?}: changed frames {} (copies, texels, \
                 mean gpu us) {:?}; unchanged frames {} {:?}",
                device.name(),
                changed.len(),
                sum(&changed),
                unchanged.len(),
                sum(&unchanged),
            );
        }
    }
}
