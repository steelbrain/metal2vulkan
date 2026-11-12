//! Device tests of a draw that samples a mipmapped texture its own render pass renders at another
//! level, under the validation layer: a slot every read of which reaches only levels past the
//! highest attached one reads the texture's own image in place of its copy, through a view of the
//! levels from the first up to the highest it reads, which holds the attached level too, unread
//! (`render/attached_levels.rs`, `WorkCounts::attached_read_slots_past`). The pass records its one
//! instance inline in a primary command buffer (`render/inline_first.rs`), where the layer sees
//! its attachment.
//!
//! The texture is a `Private` [`SIDE`]-square `RGBA8Unorm` of five levels, each level `l` holding
//! [`COLORS`]`[l]` in every byte, so a read's level shows in the color it writes. A pass loads and
//! stores one level of it and draws once over it with a function of the owned fixture
//! `fixtures/public/engine/alv_levels.ll` (`alv_level1` and `alv_level2` sample at those explicit
//! levels of detail, `alv_implicit` at the one its derivatives give, the attached level's own),
//! through a runtime sampler of nearest filters and mip filter.

use super::attachment_reads::function;
use super::degrade_notices::on_device;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The texture's level-0 side, and its levels.
const SIDE: u64 = 16;
const LEVELS: usize = 5;

/// Every byte of level `l`'s texels.
const COLORS: [u8; LEVELS] = [0x10, 0x40, 0x80, 0xc0, 0xf0];

/// The side of level `level`.
fn side(level: u64) -> u64 {
    (SIDE >> level).max(1)
}

/// The `alv_rect` state with the fragment function `name` of `alv_levels.ll`, into an unblended
/// `RGBA8Unorm` color.
fn state(device: &Device, name: &str) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "alv_rect", "alv_levels.ll"));
    d.fragment = Some(function(device, name, "alv_levels.ll"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// A sampler of nearest filters and mip filter, its levels of detail clamped into
/// `[min_lod, max_lod]`.
fn sampler(device: &Device, (min_lod, max_lod): (f32, f32)) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.mip_filter = SamplerMipFilter::Nearest;
    descriptor.lod_min_clamp = min_lod;
    descriptor.lod_max_clamp = max_lod;
    device.new_sampler_state(&descriptor).unwrap()
}

/// The `Private` texture, each level holding its color.
fn texture(queue: &CommandQueue) -> Texture {
    let device = queue.device();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, true);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    assert_eq!(texture.mipmap_level_count(), LEVELS as u64);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let mut sources = Vec::new();
    for (level, color) in COLORS.iter().enumerate() {
        let side = side(level as u64);
        let bytes = vec![*color; (side * side * 4) as usize];
        let source = device
            .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
            .unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            side * 4,
            side * side * 4,
            Size::new(side, side, 1),
            &texture,
            0,
            level as u64,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        sources.push(source);
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}

/// What one pass counted: `WorkCounts::attached_read_slots_past`, `attached_read_slots_reaching`
/// and whether it copied anything in (`shadow_copies`, `shadow_texels_first_copied_in_images`).
fn counted(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, bool) {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    (
        delta(|c| c.attached_read_slots_past),
        delta(|c| c.attached_read_slots_reaching),
        delta(|c| c.shadow_copies) > 0 || delta(|c| c.shadow_texels_first_copied_in_images) > 0,
    )
}

/// A pass loading and storing `level` of `texture`, drawing `pipeline` once over it through
/// `sampler` with the texture bound at fragment slot 0, then its level read back into `back`: what
/// it counted, and the level's bytes.
fn pass(
    queue: &CommandQueue,
    (texture, level): (&Texture, u64),
    (pipeline, sampler): (&RenderPipelineState, &SamplerState),
    back: &Buffer,
) -> ((u64, u64, bool), Vec<u8>) {
    let device = queue.device();
    let before = device.work_counts();
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(texture.clone());
    color.level = level;
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(sampler), 0)
        .unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    let side = side(level);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        level,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        back,
        0,
        side * 4,
        side * side * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let counts = counted(&before, &device.work_counts());
    let bytes = back.contents().unwrap().as_ptr();
    // SAFETY: the read-back buffer's bytes, which no command writes now.
    let bytes = unsafe { std::slice::from_raw_parts(bytes, (side * side * 4) as usize) };
    (counts, bytes.to_vec())
}

/// With level 0 attached, an explicit read of level 1; with level 1 attached, one of level 2;
/// with level 2 attached, a read at the level of detail its derivatives give (2) through a sampler
/// that clamps it to level 3 and above: each reads the texture's own image in place, with no copy,
/// and writes the color of the level it reads, the copy's texels there, with the layer reporting
/// nothing. An explicit read of the attached level 1 reads the copy, and writes level 1's color
/// from before the pass.
#[test]
fn reads_of_the_levels_past_the_attached_one_read_them_in_place() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let back = device
            .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
            .unwrap();
        let unclamped = sampler(device, (0.0, f32::MAX));
        let from_three = sampler(device, (3.0, f32::MAX));
        let in_place = (1, 0, false);
        let through_copy = (0, 1, true);
        let cases = [
            (
                "level 1, level 0 attached",
                0,
                "alv_level1",
                &unclamped,
                1,
                in_place,
            ),
            (
                "level 2, level 1 attached",
                1,
                "alv_level2",
                &unclamped,
                2,
                in_place,
            ),
            (
                "implicit from 3, level 2 attached",
                2,
                "alv_implicit",
                &from_three,
                3,
                in_place,
            ),
            (
                "level 1, level 1 attached",
                1,
                "alv_level1",
                &unclamped,
                1,
                through_copy,
            ),
        ];
        for (case, attached, function, sampler, read, counts) in cases {
            // Each case over a texture of its own, as it was made.
            let texture = texture(&queue);
            let pipeline = state(device, function);
            let (seen, bytes) = pass(&queue, (&texture, attached), (&pipeline, sampler), &back);
            assert_eq!(seen, counts, "{case}: past, reaching, copied");
            let wrong = bytes.iter().position(|&byte| byte != COLORS[read]);
            assert_eq!(
                wrong.map(|at| (at, bytes[at])),
                None,
                "{case}: every byte of level {attached} is level {read}'s color {:#x}",
                COLORS[read]
            );
        }
    });
}
