//! Draws that sample a mipmapped texture their own render pass renders at another level
//! (`render/attached_levels.rs`): a slot every read of which reaches only levels before the
//! attached one reads the texture's own image through a view of those levels, with no copy; any
//! other reads the copy, as before. Either way the draw reads the texels from before the pass.
//!
//! The texture is a private 16 x 16 `RGBA8Unorm` of five levels, each level `l` holding
//! [`COLORS`]`[l]` in every channel of every texel, so a read's level shows in the color it
//! writes. A pass loads and stores one level of it and draws once over it with a function of
//! the owned fixture `validation/fixtures/public/engine/alv_levels.ll`, through a runtime sampler
//! of nearest filters and mip filter.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The texture's level-0 side, and its levels.
const SIDE: u64 = 16;
const LEVELS: usize = 5;

/// Every byte of level `l`'s texels.
const COLORS: [u8; LEVELS] = [0x10, 0x40, 0x80, 0xc0, 0xf0];

/// The side of level `level`.
fn side(level: u64) -> u64 {
    (SIDE >> level).max(1)
}

/// What one pass moved, between its counts.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// `WorkCounts::attached_read_slots_narrowed`, `attached_read_slots_reaching` and
    /// `attached_read_slots_first_level`.
    narrowed: u64,
    reaching: u64,
    first_level: u64,
    /// Whether the pass copied anything in (`WorkCounts::shadow_copies`,
    /// `shadow_texels_first_copied_in_images`).
    copied: bool,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
        Moved {
            narrowed: delta(|c| c.attached_read_slots_narrowed),
            reaching: delta(|c| c.attached_read_slots_reaching),
            first_level: delta(|c| c.attached_read_slots_first_level),
            copied: delta(|c| c.shadow_copies) > 0
                || delta(|c| c.shadow_texels_first_copied_in_images) > 0,
        }
    }
}

/// The device's queue, the pipelines of each fragment function, and a read-back buffer.
struct Tools {
    queue: CommandQueue,
    pipelines: Vec<(&'static str, RenderPipelineState)>,
    back: Buffer,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let levels = library(device, "engine/alv_levels.ll");
        let vertex = levels.new_function_with_name("alv_rect").unwrap();
        let pipelines = ["alv_level1", "alv_level2", "alv_implicit"]
            .into_iter()
            .map(|name| {
                let fragment = levels.new_function_with_name(name).unwrap();
                let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                    vertex.function_descriptor(),
                ));
                descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
                let options = PipelineOptions::default();
                let state = device.new_render_pipeline_state(&descriptor, &options);
                (name, state.unwrap())
            })
            .collect();
        Tools {
            queue: device.new_command_queue().unwrap(),
            pipelines,
            back: device
                .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// The private texture, each level holding its color.
    fn texture(&self) -> Texture {
        let device = self.queue.device();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, true);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&descriptor).unwrap();
        assert_eq!(texture.mipmap_level_count(), LEVELS as u64);
        let command_buffer = self.queue.command_buffer().unwrap();
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

    /// A pass loading and storing `level` of `texture`, drawing `function` once over it through
    /// `sampler` with the texture bound at fragment slot 0, in a command buffer of its own: what
    /// it moved, and the level's bytes after it.
    fn pass(
        &self,
        (texture, level): (&Texture, u64),
        function: &str,
        sampler: &SamplerState,
    ) -> (Moved, Vec<u8>) {
        let device = self.queue.device();
        let before = device.work_counts();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.level = level;
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let state = (self.pipelines.iter()).find(|(name, _)| *name == function);
        encoder
            .set_render_pipeline_state(&state.unwrap().1)
            .unwrap();
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
            &self.back,
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
        let moved = Moved::of(&before, &device.work_counts());
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's bytes, which no command writes now.
        let bytes = unsafe { std::slice::from_raw_parts(bytes, (side * side * 4) as usize) };
        (moved, bytes.to_vec())
    }
}

/// A sampler of nearest filters and mip filter, its levels of detail clamped at `max_lod`.
fn sampler(device: &Device, max_lod: f32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.mip_filter = SamplerMipFilter::Nearest;
    descriptor.lod_max_clamp = max_lod;
    device.new_sampler_state(&descriptor).unwrap()
}

/// With level 2 attached: an explicit read of level 1 reads the texture's own image through a
/// view of levels 0 and 1, with no copy, and writes level 1's color, the copy's texels there;
/// one of level 2 reads the copy, and writes level 2's color from before the pass; a read at
/// the level of detail its derivatives give (2 here: a pixel of level 2 spans a texel of it)
/// reads in place where the bound sampler clamps it to level 1, and through the copy where it
/// does not. With level 0 attached, the read of level 1 reads the copy.
#[test]
fn reads_of_the_levels_before_the_attached_one_read_them_in_place() {
    let scratch = Scratch::new("attached-levels");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let unclamped = sampler(&device, f32::MAX);
    let clamped = sampler(&device, 1.0);
    let in_place = Moved {
        narrowed: 1,
        reaching: 0,
        first_level: 0,
        copied: false,
    };
    let through_copy = Moved {
        narrowed: 0,
        reaching: 1,
        first_level: 0,
        copied: true,
    };
    let first_level = Moved {
        narrowed: 0,
        reaching: 0,
        first_level: 1,
        copied: true,
    };
    let cases = [
        (
            "level 1, level 2 attached",
            2,
            "alv_level1",
            &unclamped,
            1,
            &in_place,
        ),
        (
            "level 2, level 2 attached",
            2,
            "alv_level2",
            &unclamped,
            2,
            &through_copy,
        ),
        (
            "implicit, clamped at 1",
            2,
            "alv_implicit",
            &clamped,
            1,
            &in_place,
        ),
        (
            "implicit, unclamped",
            2,
            "alv_implicit",
            &unclamped,
            2,
            &through_copy,
        ),
        (
            "level 1, level 0 attached",
            0,
            "alv_level1",
            &unclamped,
            1,
            &first_level,
        ),
    ];
    for (case, attached, function, sampler, read, moved) in cases {
        // Each case over a texture of its own, as it was made.
        let texture = tools.texture();
        let (counted, bytes) = tools.pass((&texture, attached), function, sampler);
        assert_eq!(&counted, moved, "{case}: the counts");
        let wrong = bytes.iter().position(|&byte| byte != COLORS[read]);
        assert_eq!(
            wrong.map(|at| (at, bytes[at])),
            None,
            "{case}: every byte of level {attached} is level {read}'s color {:#x}",
            COLORS[read]
        );
    }
}
