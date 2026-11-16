//! Textures placed apart in one tracked `Private` placement heap, each at an offset the device's
//! answers lay it out at (`Device::heap_texture_size_and_align`), share no bytes, as on Metal: a
//! draw that reads one while its pass renders another reads the texture itself, not a copy of it
//! (`texture/attached_reads.rs`, `WorkCounts::attachment_reads_through_copies`), and announces no
//! use of its attachment's memory (`Degradation::AttachmentReadInPlace`,
//! `render_target::shares_attachment`).
//! Every image bound in the heap's memory is the heap buffer's resource to hazard tracking
//! (`texture/heap.rs`), one key for all of them, so the memory each presents is its range of the
//! heap's bytes. A texture placed over the attachment's own bytes shares its memory, and so does a
//! heap buffer over them: the draw that reads that texture reads its copy, and the draw that reads
//! either announces it; a heap buffer placed apart shares none.
//!
//! A pass of five colors on a device that draws four at once draws in groups
//! (`render/color_groups.rs`) and refuses a draw that reads what a later group renders
//! (`NotImplemented(ColorAttachmentGroups)`): of its attachments in one heap, the draw that reads
//! a heap buffer placed apart from them draws, and the draw that reads one over its fifth
//! attachment's bytes is refused.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of `tob_renders.ll`, `cg_cover` of `color_groups.ll`, and `hpr_one` and
//! `hpr_five` of `hpr_reads.ll`, owned fixtures of `validation/fixtures/public/engine/`
//! (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::draw_again::{device, Heard};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error,
    HazardTrackingMode, HeapDescriptor, HeapType, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap, Size, StoreAction,
    Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The side of each texture.
const SIDE: u64 = 64;

/// The color the read texture is cleared to, and its texels' bytes: each channel a whole 8-bit
/// code, n / 255, which a device rounding to nearest (as Vulkan's conversion to normalized
/// fixed-point asks) stores as n. A channel scaled onto a midpoint (0.5 is 127.5) may store either
/// code beside it (announced `device-color-conversion`): an RTX 3090 stores 0.5 as `0x7f`, where
/// Metal stores `0x80`.
const READ_COLOR: [f64; 4] = [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0];
const READ_BYTES: [u8; 4] = [0x40, 0x80, 0xbf, 0xff];

/// A tracked `Private` heap's options.
fn tracked() -> ResourceOptions {
    ResourceOptions::private().with_hazard_tracking_mode(HazardTrackingMode::Tracked)
}

/// An `RGBA8Unorm` render target the shaders read, of the heap's options.
fn target() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = tracked();
    descriptor
}

/// The state that draws `vertex` with `fragment`, each a fixture's function by its path under the
/// fixtures and its name, into `colors`.
fn state(
    device: &Device,
    vertex: (&str, &str),
    fragment: (&str, &str),
    colors: &[PixelFormat],
) -> RenderPipelineState {
    let function = |(path, name): (&str, &str)| {
        let function = library(device, path).new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    for (color, &format) in descriptor.color_attachments.iter_mut().zip(colors) {
        *color = Some(ColorAttachmentDescriptor::new(format));
    }
    (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
}

/// What a draw reads: a texture at fragment texture slot 0, or a buffer at fragment buffer slot 0.
#[derive(Clone, Copy)]
enum Bound<'a> {
    Texture(&'a Texture),
    Buffer(&'a Buffer),
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
    /// `WorkCounts::attachment_reads_through_copies`: the draws that read a texture's copy, on
    /// every path a draw takes. A draw that takes every step counts its slots in
    /// `WorkCounts::attached_read_slots` too, but a draw rebound does not
    /// (`render/draw_again/rebound.rs`). The read of the texture over the attachment's bytes is a
    /// draw of the first read's state, whose kept draw the queue carries into its encoder
    /// (`render/carried.rs`), so it is rebound, except in builds with debug assertions, where the
    /// first carried draw rebound takes every step to check the rebind.
    through_copies: u64,
    /// `Degradation::AttachmentReadInPlace` notices.
    announced: usize,
}

/// The heap, its device's tools and the notices it heard.
struct Lane {
    device: Device,
    heard: Heard,
    queue: CommandQueue,
    /// `tob_rect` and `tob_read`: each pixel the texel under it of the texture at slot 0.
    reads_texture: RenderPipelineState,
    /// `cg_cover` and `hpr_one`: each pixel the buffer's first `float4` plus a tint.
    reads_buffer: RenderPipelineState,
}

impl Lane {
    /// One pass that clears `target` and draws over all of it once, reading `bound`.
    fn read(&self, target: &Texture, bound: Bound<'_>) -> Read {
        let before = self.device.work_counts();
        let heard = self.heard.lock().unwrap().len();
        let command_buffer = self.queue.command_buffer().unwrap();
        let pass = pass_of(target, LoadAction::Clear, [0.0; 4]);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let vertices = match bound {
            Bound::Texture(texture) => {
                encoder
                    .set_render_pipeline_state(&self.reads_texture)
                    .unwrap();
                let rect: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
                let rect = bytes_of(&rect, f32::to_le_bytes);
                encoder.set_vertex_bytes(&rect, 0).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                6
            }
            Bound::Buffer(buffer) => {
                encoder
                    .set_render_pipeline_state(&self.reads_buffer)
                    .unwrap();
                encoder.set_fragment_buffer(Some(buffer), 0, 0).unwrap();
                3
            }
        };
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, vertices, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = self.device.work_counts();
        let announced = (self.heard.lock().unwrap()[heard..].iter())
            .filter(|notice| notice.degradation == Degradation::AttachmentReadInPlace)
            .count();
        Read {
            through_copies: after.attachment_reads_through_copies
                - before.attachment_reads_through_copies,
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
        let row = SIDE * 4;
        let back: Buffer = (self.device)
            .new_buffer(row * SIDE, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let size = Size::new(SIDE, SIDE, 1);
        (blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            size,
            &back,
            0,
            row,
            row * SIDE,
            BlitOption::None,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let contents = back.contents().unwrap();
        // SAFETY: the buffer's bytes, which no command buffer writes now.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * SIDE) as usize) }.to_vec()
    }
}

#[test]
fn a_draw_reads_a_texture_placed_apart_from_its_attachment_in_place() {
    let scratch = Scratch::new("heap-placed-reads");
    let heard = Heard::default();
    let device = device(&scratch.0, &[Degradation::AttachmentReadInPlace], &heard);
    let answer = device.heap_texture_size_and_align(&target()).unwrap();
    // The attachment at 0, the texture read at the next offset its answer takes, apart.
    let apart = answer.size.next_multiple_of(answer.align);
    let descriptor = HeapDescriptor::new(HeapType::Placement, apart + answer.size, tracked());
    let heap = device.new_heap(&descriptor).unwrap();
    let attachment = heap.new_texture_at_offset(&target(), 0).unwrap();
    let beside = heap.new_texture_at_offset(&target(), apart).unwrap();
    // Over the attachment's bytes: a texture, and a buffer of the heap.
    let over = heap.new_texture_at_offset(&target(), 0).unwrap();
    let bytes = heap.new_buffer_at_offset(256, tracked(), 0).unwrap();
    // A heap buffer's offset is a multiple of 256 bytes: the first inside the texture's range.
    let elsewhere = heap.new_buffer_at_offset(256, tracked(), apart.next_multiple_of(256));
    let elsewhere = elsewhere.unwrap();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        reads_texture: state(
            &device,
            ("engine/tob_renders.ll", "tob_rect"),
            ("engine/tob_renders.ll", "tob_read"),
            &[PixelFormat::RGBA8Unorm],
        ),
        reads_buffer: state(
            &device,
            ("engine/color_groups.ll", "cg_cover"),
            ("engine/hpr_reads.ll", "hpr_one"),
            &[PixelFormat::RGBA8Unorm],
        ),
        device,
        heard,
    };
    lane.clear(&beside, READ_COLOR);
    let apart_read = Read {
        through_copies: 0,
        announced: 0,
    };
    let read = lane.read(&attachment, Bound::Texture(&beside));
    assert_eq!(
        read, apart_read,
        "a texture placed apart from the attachment"
    );
    let texels = lane.texels(&attachment);
    let others: Vec<&[u8]> = texels.chunks(4).filter(|t| *t != READ_BYTES).collect();
    let first = others.first().map(|t| t.to_vec());
    assert_eq!(
        others.len(),
        0,
        "attachment texels other than the read texture's bytes {READ_BYTES:02x?}: the first \
         is {first:02x?}"
    );
    let read = lane.read(&attachment, Bound::Buffer(&elsewhere));
    let what = "a heap buffer placed apart from the attachment";
    assert_eq!(read, apart_read, "{what}");
    // Over the attachment's own bytes: the texture's slot reads its copy, and reading it or a
    // heap buffer over those bytes announces the read.
    let read = lane.read(&attachment, Bound::Texture(&over));
    let shared = Read {
        through_copies: 1,
        announced: 1,
    };
    assert_eq!(read, shared, "a texture placed over the attachment's bytes");
    let read = lane.read(&attachment, Bound::Buffer(&bytes));
    let bytes_read = Read {
        through_copies: 0,
        announced: 1,
    };
    assert_eq!(
        read, bytes_read,
        "a heap buffer over the attachment's bytes"
    );
}

#[test]
fn a_pass_drawn_in_groups_draws_reading_a_heap_buffer_placed_apart_from_its_attachments() {
    let scratch = Scratch::new("heap-placed-groups");
    let mut options = options(&scratch.0);
    options.color_attachment_limit = Some(4);
    let device = Device::system_default_with(&options).unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 16, 16, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = tracked();
    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
    let apart = answer.size.next_multiple_of(answer.align);
    let heap = HeapDescriptor::new(HeapType::Placement, 6 * apart, tracked());
    let heap = device.new_heap(&heap).unwrap();
    // The five attachments, each at its own offset; a heap buffer past them, and one over the
    // fifth's first bytes (a heap buffer's offset is a multiple of 256 bytes).
    let attachments: Vec<Texture> = (0..5)
        .map(|at| heap.new_texture_at_offset(&descriptor, at * apart).unwrap())
        .collect();
    let buffer_at = |at: u64| {
        let offset = (at * apart).next_multiple_of(256);
        heap.new_buffer_at_offset(16, tracked(), offset).unwrap()
    };
    let (past, over) = (buffer_at(5), buffer_at(4));
    let five = [PixelFormat::RGBA16Float; 5];
    let state = state(
        &device,
        ("engine/color_groups.ll", "cg_cover"),
        ("engine/hpr_reads.ll", "hpr_five"),
        &five,
    );
    let queue = device.new_command_queue().unwrap();
    // One pass clearing the attachments and drawing over them, reading `read`.
    let draw = |read: &Buffer| -> Result<(), Error> {
        let mut pass = RenderPassDescriptor::new();
        for (color, texture) in pass.color_attachments.iter_mut().zip(&attachments) {
            color.texture = Some(texture.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
        }
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_fragment_buffer(Some(read), 0, 0).unwrap();
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drawn
    };
    let before = device.work_counts().color_group_passes;
    draw(&past).expect("a draw reading a heap buffer placed apart from the attachments");
    let grouped = device.work_counts().color_group_passes - before;
    assert!(grouped > 0, "the pass of five colors drew in groups");
    let refused = draw(&over);
    assert!(
        matches!(
            refused,
            Err(Error::NotImplemented(RuntimeGap::ColorAttachmentGroups))
        ),
        "a draw reading a heap buffer over the fifth attachment's bytes: {refused:?}"
    );
}
