//! Textures made over one `Private` buffer's bytes apart (`Buffer::new_texture` at offsets whose
//! texels do not meet) share no bytes, as on Metal: a draw that reads one while its pass renders
//! another reads the texture itself, not a copy or a snapshot of it (`texture/attached_reads.rs`,
//! `WorkCounts::attached_read_slots`, `reading_draws_snapshotted`, `reading_draws_in_place`), and
//! announces no use of its attachment's memory (`Degradation::AttachmentReadInPlace`). Each is an
//! image over the buffer's memory, whose uses are the buffer's bytes (`Texels::bound_use`), one
//! key for both, so where each image lies is its range of the buffer's bytes
//! (`tracker::ImagePlace`). A texture made over the attachment's own bytes shares its memory: the
//! draw that reads it takes the attached read's route.
//!
//! After a memory barrier over the render targets before the pass's first draw, a draw that
//! samples the texture of a color the pass clears reads the clear (`texture/attached_reads.rs`:
//! its copy cleared to the pass's clear value, `WorkCounts::attachment_read_clears`, or a split,
//! `attachment_read_splits`). A texture over the attachment's buffer apart from it is no such
//! texture: the draw reads its own texels there too, and the texture over the attachment's own
//! bytes reads the clear.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of `tob_renders.ll`, an owned fixture of `validation/fixtures/public/engine/`
//! (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The side of each texture: 256 bytes a row, at which every driver lays a linear image's rows
/// with no byte between them.
const SIDE: u64 = 64;
const ROW: u64 = SIDE * 4;

/// The color the read texture is cleared to, and its texels' bytes: each channel a whole 8-bit
/// code, n / 255, which a device rounding to nearest (as Vulkan's conversion to normalized
/// fixed-point asks) stores as n. A channel scaled onto a midpoint (0.5 is 127.5) may store either
/// code beside it (announced `device-color-conversion`): an RTX 3090 stores 0.5 as `0x7f`, where
/// Metal stores `0x80`.
const READ_COLOR: [f64; 4] = [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0];
const READ_BYTES: [u8; 4] = [0x40, 0x80, 0xbf, 0xff];

/// An `RGBA8Unorm` render target the shaders read, over a `Private` buffer.
fn target() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// A pass of `texture` that loads it `load`, clearing to `clear`, and stores it.
fn pass_of(texture: &Texture, load: LoadAction, clear: [f64; 4]) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(clear[0], clear[1], clear[2], clear[3]);
    pass
}

/// What one reading pass counted and announced.
#[derive(Debug, PartialEq, Eq)]
struct Read {
    /// The draw's slots that read a copy (`WorkCounts::attached_read_slots`), and its draws that
    /// read the attachment through a snapshot or in place (`reading_draws_snapshotted`,
    /// `reading_draws_in_place`), or after a barrier before the first draw read a copy cleared to
    /// the clear or split the pass (`attachment_read_clears`, `attachment_read_splits`): every
    /// route of an attached read, summed.
    attached: u64,
    /// `Degradation::AttachmentReadInPlace` notices.
    announced: usize,
}

/// The device, its tools and the notices it heard.
struct Lane {
    device: Device,
    heard: Heard,
    queue: CommandQueue,
    /// `tob_rect` and `tob_read`: each pixel the texel under it of the texture at slot 0.
    reads: RenderPipelineState,
}

impl Lane {
    /// One pass that clears `target` to zero and draws over all of it once, reading `read`, after
    /// a memory barrier over the render targets from and to the fragment stage where `barrier`.
    fn read(&self, target: &Texture, read: &Texture, barrier: bool) -> Read {
        let before = self.device.work_counts();
        let heard = self.heard.lock().unwrap().len();
        let command_buffer = self.queue.command_buffer().unwrap();
        let pass = pass_of(target, LoadAction::Clear, [0.0; 4]);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.reads).unwrap();
        let rect: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
        let rect = bytes_of(&rect, f32::to_le_bytes);
        encoder.set_vertex_bytes(&rect, 0).unwrap();
        encoder.set_fragment_texture(Some(read), 0).unwrap();
        if barrier {
            let fragment = RenderStages::FRAGMENT;
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
        }
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = self.device.work_counts();
        let announced = (self.heard.lock().unwrap()[heard..].iter())
            .filter(|notice| notice.degradation == Degradation::AttachmentReadInPlace)
            .count();
        let attached = (after.attached_read_slots - before.attached_read_slots)
            + (after.reading_draws_snapshotted - before.reading_draws_snapshotted)
            + (after.reading_draws_in_place - before.reading_draws_in_place)
            + (after.attachment_read_clears - before.attachment_read_clears)
            + (after.attachment_read_splits - before.attachment_read_splits);
        Read {
            attached,
            announced,
        }
    }

    /// Clear `texture` to `color` in a pass of no draw.
    fn clear(&self, texture: &Texture, color: [f64; 4]) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let pass = pass_of(texture, LoadAction::Clear, color);
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// The bytes of `texture`'s texels, blitted into a buffer.
    fn texels(&self, texture: &Texture) -> Vec<u8> {
        let back: Buffer = (self.device)
            .new_buffer(ROW * SIDE, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &back,
            0,
            ROW,
            ROW * SIDE,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let contents = back.contents().unwrap();
        // SAFETY: the buffer's bytes, which no command buffer writes now.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), (ROW * SIDE) as usize) }.to_vec()
    }
}

/// The lane on a device that announces each `AttachmentReadInPlace`, and three textures over one
/// `Private` buffer: the attachment over its first bytes, the texture read at the next 64 KiB,
/// apart, cleared to [`READ_COLOR`], and a texture over the attachment's own bytes.
fn lane(scratch: &Scratch) -> (Lane, [Texture; 3]) {
    let heard = Heard::default();
    let device = device(&scratch.0, &[Degradation::AttachmentReadInPlace], &heard);
    let apart = (ROW * SIDE).next_multiple_of(1 << 16);
    let buffer = (device.new_buffer(apart + ROW * SIDE, ResourceOptions::private())).unwrap();
    let textures = [0, apart, 0].map(|offset| buffer.new_texture(&target(), offset, ROW).unwrap());
    for texture in &textures {
        assert!(
            texture.aliases_buffer(),
            "each texture is an image over the buffer's bytes, whose key is the buffer's"
        );
    }
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        reads: {
            let function = |name: &str| {
                let library = library(&device, "engine/tob_renders.ll");
                let function = library.new_function_with_name(name).unwrap();
                StageSource::Function(function.function_descriptor())
            };
            let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
            descriptor.fragment = Some(function("tob_read"));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
        },
        device,
        heard,
    };
    lane.clear(&textures[1], READ_COLOR);
    (lane, textures)
}

/// No attached read's route, and no notice.
const APART: Read = Read {
    attached: 0,
    announced: 0,
};

/// How many of `texture`'s texels are not `expected`, and the bytes of the first that is not.
fn wrong_texels(lane: &Lane, texture: &Texture, expected: [u8; 4]) -> (usize, Option<Vec<u8>>) {
    let texels = lane.texels(texture);
    let others: Vec<&[u8]> = texels.chunks(4).filter(|t| *t != expected).collect();
    (others.len(), others.first().map(|t| t.to_vec()))
}

#[test]
fn a_draw_reads_a_texture_over_its_attachments_buffer_apart_in_place() {
    let scratch = Scratch::new("buffer-carved-reads");
    let (lane, [attachment, beside, over]) = lane(&scratch);
    let read = lane.read(&attachment, &beside, false);
    assert_eq!(
        read, APART,
        "a texture over the buffer apart from the attachment"
    );
    let (wrong, first) = wrong_texels(&lane, &attachment, READ_BYTES);
    assert_eq!(
        wrong, 0,
        "attachment texels other than the read texture's bytes {READ_BYTES:02x?}: the first \
         is {first:02x?}"
    );
    // Over the attachment's own bytes: the read takes the attached read's route.
    let read = lane.read(&attachment, &over, false);
    assert!(
        read.attached > 0,
        "a texture over the attachment's bytes: {read:?}"
    );
}

#[test]
fn after_a_barrier_before_the_first_draw_a_texture_apart_reads_its_own_texels() {
    let scratch = Scratch::new("buffer-carved-cleared");
    let (lane, [attachment, beside, over]) = lane(&scratch);
    let read = lane.read(&attachment, &beside, true);
    // The texels first: a copy of the texture cleared to the attachment's clear is a wrong value.
    let (wrong, first) = wrong_texels(&lane, &attachment, READ_BYTES);
    assert_eq!(
        wrong, 0,
        "attachment texels other than the read texture's bytes {READ_BYTES:02x?}: the first \
         is {first:02x?}"
    );
    assert_eq!(
        read, APART,
        "a texture over the buffer apart from the attachment"
    );
    // Over the attachment's own bytes: the read takes the attached read's route, and reads the
    // clear.
    let read = lane.read(&attachment, &over, true);
    assert!(
        read.attached > 0,
        "a texture over the attachment's bytes: {read:?}"
    );
    let (wrong, first) = wrong_texels(&lane, &attachment, [0; 4]);
    assert_eq!(
        wrong, 0,
        "attachment texels other than the clear's bytes [00, 00, 00, 00]: the first is \
         {first:02x?}"
    );
}
