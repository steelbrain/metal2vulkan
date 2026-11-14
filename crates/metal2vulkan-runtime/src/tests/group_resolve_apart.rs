//! A pass drawn in color groups (`render/color_groups.rs`) refuses a resolve target set in its
//! first group's encoder after a draw of the pass read that target's memory
//! (`NotImplemented(ColorAttachmentGroups)`): the later groups run the pass's log again, and the
//! target's resolve is the first group's. Distinct resources that share one resource key, at bytes
//! apart, share no memory: a texture placed in a heap apart from a heap buffer a draw read
//! (`tracker::ImagePlace`), and a texture made over a buffer's bytes apart from the bytes a draw
//! read of that buffer. Set late as such a pass's resolve target, either resolves, as on Metal;
//! over the bytes the draw read, it is refused.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `cg_cover` of
//! `color_groups.ll` and `hpr_five` of `hpr_reads.ll`, owned fixtures of
//! `validation/fixtures/public/engine/` (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Error, HazardTrackingMode,
    HeapDescriptor, HeapType, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResolveTarget, ResourceOptions, RuntimeGap, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The side of each texture: 512 bytes a row of the format, at which every driver lays a linear
/// image's rows with no byte between them.
const SIDE: u64 = 64;

/// The color format of the pass's five attachments and of the resolve target.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// The color the late target resolves: `hpr_five`'s color 1, the varying tint, whose alpha is
/// 0.25 at every pixel (`0x3400` as a half).
const RESOLVED: u32 = 1;
const ALPHA: u16 = 0x3400;

/// A tracked `Private` resource's options.
fn tracked() -> ResourceOptions {
    ResourceOptions::private().with_hazard_tracking_mode(HazardTrackingMode::Tracked)
}

/// A single-sampled render target of the pass's format, of `options`: the late resolve target.
fn target_descriptor(options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    descriptor
}

/// A device that draws four colors at once, the state that draws five four-sample colors with
/// `hpr_five` (reading the buffer at fragment slot 0), its five attachments and a queue.
struct Lane {
    device: Device,
    queue: CommandQueue,
    state: RenderPipelineState,
    attachments: Vec<Texture>,
}

impl Lane {
    fn new(scratch: &Scratch) -> Lane {
        let mut options = options(&scratch.0);
        options.color_attachment_limit = Some(4);
        let device = Device::system_default_with(&options).unwrap();
        let function = |path: &str, name: &str| {
            let function = library(&device, path).new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(function("engine/color_groups.ll", "cg_cover"));
        descriptor.fragment = Some(function("engine/hpr_reads.ll", "hpr_five"));
        descriptor.raster_sample_count = 4;
        for color in descriptor.color_attachments.iter_mut().take(5) {
            *color = Some(ColorAttachmentDescriptor::new(FORMAT));
        }
        let state =
            (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap();
        let mut msaa = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
        msaa.texture_type = TextureType::Type2DMultisample;
        msaa.sample_count = 4;
        msaa.usage = TextureUsage::RENDER_TARGET;
        msaa.resource_options = ResourceOptions::private();
        let attachments = (0..5).map(|_| device.new_texture(&msaa).unwrap()).collect();
        Lane {
            queue: device.new_command_queue().unwrap(),
            device,
            state,
            attachments,
        }
    }

    /// One pass of the five attachments, color [`RESOLVED`] stored `Unknown`, that draws once
    /// reading `read` at `offset`, then sets `target` as that color's resolve target and its store
    /// action to `MultisampleResolve`: the setter's answer.
    fn resolve_after_read(
        &self,
        read: &Buffer,
        offset: u64,
        target: &Texture,
    ) -> Result<(), Error> {
        let mut pass = RenderPassDescriptor::new();
        for (index, (color, texture)) in
            (pass.color_attachments.iter_mut().zip(&self.attachments)).enumerate()
        {
            color.texture = Some(texture.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = match index as u32 == RESOLVED {
                true => StoreAction::Unknown,
                false => StoreAction::DontCare,
            };
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.state).unwrap();
        encoder.set_fragment_buffer(Some(read), offset, 0).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        let set =
            encoder.set_color_resolve_target(Some(ResolveTarget::new(target.clone())), RESOLVED);
        (encoder.set_color_store_action(StoreAction::MultisampleResolve, RESOLVED)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        set
    }

    /// Whether the resolve wrote `target`: every texel's alpha the tint's.
    fn resolved(&self, target: &Texture) -> bool {
        let row = SIDE * 8;
        let back: Buffer = (self.device)
            .new_buffer(row * SIDE, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_texture_to_buffer(
            target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
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
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * SIDE) as usize) };
        (bytes.chunks(8)).all(|texel| u16::from_le_bytes([texel[6], texel[7]]) == ALPHA)
    }

    /// Whether the pass drew in groups since `before` (`WorkCounts::color_group_passes`).
    fn grouped_since(&self, before: u64) -> bool {
        self.device.work_counts().color_group_passes > before
    }
}

fn refused(answer: &Result<(), Error>) -> bool {
    matches!(
        answer,
        Err(Error::NotImplemented(RuntimeGap::ColorAttachmentGroups))
    )
}

#[test]
fn a_grouped_pass_resolves_into_a_heap_texture_placed_apart_from_the_buffer_it_read() {
    let scratch = Scratch::new("group-resolve-heap");
    let lane = Lane::new(&scratch);
    let descriptor = target_descriptor(tracked());
    let answer = lane
        .device
        .heap_texture_size_and_align(&descriptor)
        .unwrap();
    // The target at 0; a heap buffer at the first 256-byte offset past it, and one over its first
    // bytes.
    let apart = answer
        .size
        .next_multiple_of(answer.align)
        .next_multiple_of(256);
    let heap = HeapDescriptor::new(HeapType::Placement, apart + 256, tracked());
    let heap = lane.device.new_heap(&heap).unwrap();
    let target = heap.new_texture_at_offset(&descriptor, 0).unwrap();
    let elsewhere = heap.new_buffer_at_offset(16, tracked(), apart).unwrap();
    let over = heap.new_buffer_at_offset(16, tracked(), 0).unwrap();
    let before = lane.device.work_counts().color_group_passes;
    let set = lane.resolve_after_read(&elsewhere, 0, &target);
    assert!(
        lane.grouped_since(before),
        "the pass of five colors drew in groups"
    );
    set.expect("a target placed apart from the heap buffer the draw read");
    assert!(lane.resolved(&target), "the target holds the resolve");
    let set = lane.resolve_after_read(&over, 0, &target);
    assert!(
        refused(&set),
        "a target over the heap buffer the draw read: {set:?}"
    );
}

#[test]
fn a_grouped_pass_resolves_into_a_buffer_texture_apart_from_the_bytes_it_read() {
    let scratch = Scratch::new("group-resolve-buffer");
    let lane = Lane::new(&scratch);
    let descriptor = target_descriptor(ResourceOptions::private());
    let row = SIDE * 8;
    // The target over the buffer's first bytes; the draw reads the buffer at the first 64 KiB
    // offset past them, or at 0.
    let apart = (row * SIDE).next_multiple_of(1 << 16);
    let buffer = (lane.device)
        .new_buffer(apart + 256, ResourceOptions::private())
        .unwrap();
    let target = buffer.new_texture(&descriptor, 0, row).unwrap();
    assert!(
        target.aliases_buffer(),
        "the target is an image over the buffer's bytes, whose key is the buffer's"
    );
    let before = lane.device.work_counts().color_group_passes;
    let set = lane.resolve_after_read(&buffer, apart, &target);
    assert!(
        lane.grouped_since(before),
        "the pass of five colors drew in groups"
    );
    set.expect("a target apart from the bytes of its buffer the draw read");
    assert!(lane.resolved(&target), "the target holds the resolve");
    let set = lane.resolve_after_read(&buffer, 0, &target);
    assert!(
        refused(&set),
        "a target over the bytes the draw read: {set:?}"
    );
}
