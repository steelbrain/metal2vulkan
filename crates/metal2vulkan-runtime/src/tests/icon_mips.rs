//! An icon's levels after the events that give its texture another image: a 128x128
//! `RGBA16Float` texture of every level, its left half one color and its right half another at
//! each level, written as a program writes an icon's levels, then sampled at levels 0 to 2
//! through a draw into a target of each level's size (its implicit level of detail selecting the
//! level) and read back by blits. Every texel read must be the one written: a level an event left
//! behind, never generated, or kept from another texture would read zero or another color.
//!
//! The writes:
//! - **Rendered, in place.** Level 0 cleared to the right color, its left half drawn in the left
//!   one; level 1 drawn from level 0 and level 2 from level 1, each sampling the texture itself
//!   in its own level's pass, reads of levels before the attached one, in place
//!   (`render/attached_levels.rs`).
//! - **Rendered, through the copy.** The same, each later level's pass sampling level 0 at the
//!   pixel's position over level 0's size, which the slot reads through the texture's copy
//!   (`texture/attached_reads.rs`): each later level holds level 0's top left corner, the left
//!   color alone.
//! - **Blitted.** Level 0 rendered, then `generate_mipmaps` (`blit_texture.rs`).
//! - **Replaced.** A `Shared` texture, each level written by `replace_region`.
//!
//! The events, after the writes completed:
//! - **None.**
//! - **A storage form at the commit.** A pass over level 0 that fetches its color and halves it
//!   (`fetch_ordered`), in a command buffer committed alone: a device whose fetch needs `STORAGE`
//!   the image lacks (the M3) renders it through a scratch and gives the texture its storage form
//!   at that commit, a copy of every level and slice of the old image (`texture/storage_form.rs`),
//!   whose old image then leaves the recycler's book (`recycle.rs`,
//!   `ChurnCounts::swapped_out_images_forgotten`). Level 0 then holds half of each color; the
//!   later levels what they held.
//! - **A storage form at a later commit.** The same, another command buffer uncommitted as it
//!   commits, committed after it with nothing in it: the switch rides on that one.
//! - **A recycled image.** Before the icons, as many textures of the same making written with the
//!   colors swapped, completed and dropped: their images park, and the icons take them
//!   (`ChurnCounts::images_recycled`), so a level they kept would read the swapped colors.
//!
//! Icons are made in rounds, and each round's go as it ends, so a later round's textures are made
//! after the images an earlier round's storage forms swapped out went.
//!
//! And icons whose levels only the CPU writes, as a window server's are: `Managed` (or `Shared`)
//! 128x128 `RGBA16Float` textures of five levels and `ShaderRead` usage alone, each level of its
//! own colors written by `replace_region`, made before the command buffer that samples them or
//! while it is open, 21 a round. That command buffer draws every level of each through a sampler
//! pinned to the level into a target of its size: each target, and each level read by a blit,
//! must hold what was uploaded.
//!
//! The functions are `tob_rect`, `tob_solid` and `tob_sample` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll`, `alv_rect`, `alv_implicit` and
//! `alv_level1` of `validation/fixtures/public/engine/alv_levels.ll`, and `fetch_ordered` of
//! `validation/fixtures/public/engine/fragment_fetch.ll`.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, CpuCacheMode, Device,
    HazardTrackingMode, LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, Size, StorageMode, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The icon's level-0 side, its levels (every one down to 1x1), and the levels checked.
const SIDE: u64 = 128;
const LEVELS: u64 = 8;
const CHECKED: u64 = 3;

/// An `RGBA16Float` texel's bytes.
const TEXEL: u64 = 8;

/// The colors of the left and right halves.
const LEFT: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
const RIGHT: [f32; 4] = [1.0, 0.75, 0.5, 0.25];

/// The icons each round makes, and the rounds.
const ICONS: usize = 8;
const ROUNDS: usize = 3;

/// The side of level `level`.
fn side(level: u64) -> u64 {
    (SIDE >> level).max(1)
}

/// The bits of a half that holds `value` exactly, `value` normal and positive.
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// How the icon's levels are written.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Way {
    InPlace,
    ThroughCopy,
    Blitted,
    Replaced,
}

/// What happens to the icon after its levels are written.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    None,
    FormAtCommit,
    FormLater,
    Recycled,
}

/// What a draw's fragment function takes at slot 0.
#[derive(Clone, Copy)]
enum Fragment<'a> {
    /// Bytes at buffer 0.
    Bytes(&'a [u8]),
    /// A texture and sampler at texture and sampler 0.
    Sampled(&'a Texture, &'a SamplerState),
    /// Nothing.
    Nothing,
}

/// A draw of a pass: its state, the vertex bytes it takes, and what its fragment function takes.
type Draw<'a> = (&'a RenderPipelineState, Option<&'a [u8]>, Fragment<'a>);

/// The queue, the states and samplers, and a read-back buffer.
struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    sample: RenderPipelineState,
    implicit: RenderPipelineState,
    level1: RenderPipelineState,
    /// `fetch_ordered`, where the device fetches.
    fetch: Option<RenderPipelineState>,
    /// Nearest filters and mip filter, unclamped, and clamped to level 0.
    sampler: SamplerState,
    level0: SamplerState,
    /// `tob_rect`'s corners covering the whole target, and its left half.
    whole: Vec<u8>,
    left: Vec<u8>,
    back: Buffer,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let function = |fixture: &str, name: &str| {
            let function = library(device, fixture)
                .new_function_with_name(name)
                .unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let state = |(vertex_fixture, vertex): (&str, &str), (fixture, fragment): (&str, &str)| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex_fixture, vertex));
            descriptor.fragment = Some(function(fixture, fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        let sampler = |max_lod: f32| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.mip_filter = SamplerMipFilter::Nearest;
            descriptor.lod_max_clamp = max_lod;
            device.new_sampler_state(&descriptor).unwrap()
        };
        let (tob, alv) = ("engine/tob_renders.ll", "engine/alv_levels.ll");
        let fetch = ("engine/fragment_fetch.ll", "fetch_ordered");
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: state((tob, "tob_rect"), (tob, "tob_solid")).unwrap(),
            sample: state((tob, "tob_rect"), (tob, "tob_sample")).unwrap(),
            implicit: state((alv, "alv_rect"), (alv, "alv_implicit")).unwrap(),
            level1: state((alv, "alv_rect"), (alv, "alv_level1")).unwrap(),
            fetch: state((tob, "tob_rect"), fetch).ok(),
            sampler: sampler(f32::MAX),
            level0: sampler(0.0),
            whole: bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes),
            left: bytes_of(&[-1.0f32, -1.0, 0.0, 1.0], f32::to_le_bytes),
            back: device
                .new_buffer(SIDE * SIDE * TEXEL * 2, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A pass over `level` of `target`, cleared to `clear` (loaded where `None`) and stored,
    /// drawing each of `draws` once.
    fn pass(
        &self,
        command_buffer: &CommandBuffer,
        (target, level): (&Texture, u64),
        clear: Option<[f32; 4]>,
        draws: &[Draw],
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.level = level;
        match clear {
            Some(clear) => {
                let [r, g, b, a] = clear.map(f64::from);
                color.load_action = LoadAction::Clear;
                color.clear_color = ClearColor::new(r, g, b, a);
            }
            None => color.load_action = LoadAction::Load,
        }
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        for &(state, vertex, fragment) in draws {
            encoder.set_render_pipeline_state(state).unwrap();
            if let Some(bytes) = vertex {
                encoder.set_vertex_bytes(bytes, 0).unwrap();
            }
            match fragment {
                Fragment::Sampled(texture, sampler) => {
                    encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    (encoder.set_fragment_sampler_state(Some(sampler), 0)).unwrap();
                }
                Fragment::Bytes(bytes) => encoder.set_fragment_bytes(bytes, 0).unwrap(),
                Fragment::Nothing => {}
            }
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// A new icon of the colors `(left, right)`, its levels written `way`, the GPU's writes
    /// encoded in `command_buffer`.
    fn icon(&self, command_buffer: &CommandBuffer, way: Way, colors: [[f32; 4]; 2]) -> Texture {
        let device = self.queue.device();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, true);
        d.mipmap_level_count = LEVELS;
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = match way {
            Way::Replaced => ResourceOptions::shared(),
            _ => ResourceOptions::private(),
        };
        let icon = device.new_texture(&d).unwrap();
        if way == Way::Replaced {
            for level in 0..LEVELS {
                let side = side(level);
                let bytes: Vec<u8> = (0..side * side)
                    .flat_map(|i| expected(way, colors, level, i % side).map(u16::to_le_bytes))
                    .flatten()
                    .collect();
                let region = Region::new_2d(0, 0, side, side);
                (icon.replace_region(region, level, 0, &bytes, side * TEXEL, 0)).unwrap();
            }
            return icon;
        }
        let left = bytes_of(&colors[0], f32::to_le_bytes);
        let draw: Draw = (&self.solid, Some(&self.left), Fragment::Bytes(&left));
        self.pass(command_buffer, (&icon, 0), Some(colors[1]), &[draw]);
        let whole = Some(self.whole.as_slice());
        match way {
            Way::InPlace => {
                let draw = (&self.implicit, None, Fragment::Sampled(&icon, &self.level0));
                self.pass(command_buffer, (&icon, 1), Some([0.0; 4]), &[draw]);
                let draw = (&self.level1, None, Fragment::Sampled(&icon, &self.sampler));
                self.pass(command_buffer, (&icon, 2), Some([0.0; 4]), &[draw]);
            }
            Way::ThroughCopy => {
                for level in 1..CHECKED {
                    let draw = (&self.sample, whole, Fragment::Sampled(&icon, &self.sampler));
                    self.pass(command_buffer, (&icon, level), Some([0.0; 4]), &[draw]);
                }
            }
            Way::Blitted => {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.generate_mipmaps(&icon).unwrap();
                blit.end_encoding().unwrap();
            }
            Way::Replaced => {}
        }
        icon
    }

    /// A pass over level 0 of `icon` that loads it and halves it through a fetch.
    fn fetched(&self, command_buffer: &CommandBuffer, icon: &Texture, fetch: &RenderPipelineState) {
        let draw: Draw = (fetch, Some(&self.whole), Fragment::Nothing);
        self.pass(command_buffer, (icon, 0), None, &[draw]);
    }

    /// Copy `level` of `texture` into the read-back buffer at `offset`.
    fn read(&self, command_buffer: &CommandBuffer, (texture, level): (&Texture, u64), offset: u64) {
        let side = side(level);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &self.back,
            offset,
            side * TEXEL,
            side * side * TEXEL,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }

    /// The texels of the read-back buffer from `offset` of a square of level `level`'s side that
    /// are not `want`'s by more than a unit of their last place, how many and the first: its
    /// position and texel.
    fn wrong(
        &self,
        offset: u64,
        level: u64,
        want: impl Fn(u64) -> [u16; 4],
    ) -> (usize, Option<(u64, u64, [u16; 4])>) {
        let side = side(level);
        let contents = self.back.contents().unwrap();
        // SAFETY: bytes within the read-back buffer, which no command buffer writes now.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                contents.as_ptr().add(offset as usize),
                (side * side * TEXEL) as usize,
            )
        };
        let wrong: Vec<(u64, u64, [u16; 4])> = (bytes.as_chunks::<8>().0.iter().enumerate())
            .map(|(i, t)| {
                let h = |c: usize| u16::from_le_bytes([t[2 * c], t[2 * c + 1]]);
                (i as u64 % side, i as u64 / side, [0, 1, 2, 3].map(h))
            })
            .filter(|&(x, _, texel)| {
                let want = want(x);
                (0..4).any(|c| texel[c].abs_diff(want[c]) > 1)
            })
            .collect();
        (wrong.len(), wrong.first().copied())
    }
}

/// The texel at column `x` of level `level` of an icon of `colors` written `way`, as halves.
fn expected(way: Way, [left, right]: [[f32; 4]; 2], level: u64, x: u64) -> [u16; 4] {
    let color = match (way, level) {
        // Each later level holds level 0's top left corner.
        (Way::ThroughCopy, 1..) => left,
        _ if x < side(level) / 2 => left,
        _ => right,
    };
    color.map(half)
}

/// The counts that say which paths ran: (storage forms taken, images recycled, swapped-out
/// images forgotten, reads through copies, slots narrowed).
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    let churn = (&before.object_churn, &after.object_churn);
    [
        after.storage_forms_taken - before.storage_forms_taken,
        churn.1.images_recycled - churn.0.images_recycled,
        churn.1.swapped_out_images_forgotten - churn.0.swapped_out_images_forgotten,
        after.attachment_reads_through_copies - before.attachment_reads_through_copies,
        after.attached_read_slots_narrowed - before.attached_read_slots_narrowed,
    ]
}

/// Commit `command_buffer` and wait for it, which must not fail.
fn finish(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

fn run(case: &str, way: Way, event: Event) {
    let scratch = Scratch::new(&format!("icon-mips-{case}"));
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let forms = matches!(event, Event::FormAtCommit | Event::FormLater);
    let fetch = match (forms, &tools.fetch) {
        (true, None) => {
            eprintln!(
                "ICON-MIPS {case} on {}: no framebuffer fetch: nothing to check",
                device.name()
            );
            return;
        }
        (_, fetch) => fetch.as_ref(),
    };
    let colors = [LEFT, RIGHT];
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        if event == Event::Recycled {
            // Textures of the same making, of the colors swapped, completed and gone.
            let command_buffer = tools.queue.command_buffer().unwrap();
            let swapped: Vec<Texture> = (0..ICONS)
                .map(|_| tools.icon(&command_buffer, way, [RIGHT, LEFT]))
                .collect();
            finish(&command_buffer);
            drop(swapped);
        }
        let command_buffer = tools.queue.command_buffer().unwrap();
        let icons: Vec<Texture> = (0..ICONS)
            .map(|_| tools.icon(&command_buffer, way, colors))
            .collect();
        finish(&command_buffer);
        if let Some(fetch) = fetch.filter(|_| forms) {
            let command_buffer = tools.queue.command_buffer().unwrap();
            let later = (event == Event::FormLater).then(|| tools.queue.command_buffer().unwrap());
            for icon in &icons {
                tools.fetched(&command_buffer, icon, fetch);
            }
            command_buffer.commit().unwrap();
            if let Some(later) = &later {
                later.commit().unwrap();
            }
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            if let Some(later) = later {
                later.wait_until_completed_for(WAIT).unwrap();
            }
        }
        counts.push(moved(&before, &device.work_counts()));
        let want = |level: u64, x: u64| {
            let texel = expected(way, colors, level, x);
            // Halved, a normal half's exponent less one.
            match (forms, level) {
                (true, 0) => texel.map(|h| h - 0x0400),
                _ => texel,
            }
        };
        for (i, icon) in icons.iter().enumerate() {
            let mut wrong = Vec::new();
            for level in 0..CHECKED {
                // The level sampled through a draw into a target of its size, and read by a blit.
                let mut d = TextureDescriptor::texture_2d(
                    PixelFormat::RGBA16Float,
                    side(level),
                    side(level),
                    false,
                );
                d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::private();
                let target = device.new_texture(&d).unwrap();
                let command_buffer = tools.queue.command_buffer().unwrap();
                let draw: Draw = (
                    &tools.implicit,
                    None,
                    Fragment::Sampled(icon, &tools.sampler),
                );
                tools.pass(&command_buffer, (&target, 0), Some([0.0; 4]), &[draw]);
                tools.read(&command_buffer, (&target, 0), 0);
                tools.read(&command_buffer, (icon, level), SIDE * SIDE * TEXEL);
                finish(&command_buffer);
                let sampled = tools.wrong(0, level, |x| want(level, x));
                let read = tools.wrong(SIDE * SIDE * TEXEL, level, |x| want(level, x));
                if sampled.0 != 0 || read.0 != 0 {
                    wrong.push(format!("level {level}: sampled {sampled:?}, read {read:?}"));
                }
            }
            if !wrong.is_empty() {
                failures.push(format!("round {round} icon {i}: {wrong:?}"));
            }
        }
    }
    eprintln!(
        "ICON-MIPS {case} on {}: {} of {} icons wrong; per round (storage forms taken, images \
         recycled, swapped-out images forgotten, reads through copies, slots narrowed) \
         {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

/// The levels of a texture of uploaded levels, and the textures each round makes of it.
const UPLOADED_LEVELS: u64 = 5;
const UPLOADED_ICONS: usize = 21;

/// The colors of the left and right halves of level `level` of a texture of uploaded levels,
/// each level's its own.
fn uploaded_colors(level: u64) -> [[f32; 4]; 2] {
    let n = (level + 1) as f32 * 0.125;
    [[n, 0.5, 0.25, 1.0], [0.5, n, 0.75, 0.5]]
}

/// The uploaded run: each round [`UPLOADED_ICONS`] textures of `storage`, 128x128
/// `RGBA16Float` of [`UPLOADED_LEVELS`] levels and `ShaderRead` usage alone, each level written by
/// `replace_region` and no GPU work, made before the command buffer that samples them or, where
/// `open`, while it is open; that command buffer draws each level of each into a target of the
/// level's size through a sampler pinned to the level (its least and greatest level of detail
/// the level's), and each level and target is then read back by blits. The round's textures go
/// as it ends.
fn run_uploaded(case: &str, storage: StorageMode, open: bool) {
    let scratch = Scratch::new(&format!("icon-mips-{case}"));
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let pinned: Vec<SamplerState> = (0..UPLOADED_LEVELS)
        .map(|level| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.mip_filter = SamplerMipFilter::Nearest;
            descriptor.lod_min_clamp = level as f32;
            descriptor.lod_max_clamp = level as f32;
            device.new_sampler_state(&descriptor).unwrap()
        })
        .collect();
    let options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let icon = || {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, true);
        d.mipmap_level_count = UPLOADED_LEVELS;
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = options;
        let icon = device.new_texture(&d).unwrap();
        for level in 0..UPLOADED_LEVELS {
            let side = side(level);
            let [left, right] = uploaded_colors(level).map(|color| color.map(half));
            let bytes: Vec<u8> = (0..side * side)
                .flat_map(|i| match i % side < side / 2 {
                    true => left,
                    false => right,
                })
                .flat_map(u16::to_le_bytes)
                .collect();
            let region = Region::new_2d(0, 0, side, side);
            (icon.replace_region(region, level, 0, &bytes, side * TEXEL, 0)).unwrap();
        }
        icon
    };
    let want = |level: u64, x: u64| {
        let [left, right] = uploaded_colors(level);
        match x < side(level) / 2 {
            true => left.map(half),
            false => right.map(half),
        }
    };
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        let made_first = (!open).then(|| (0..UPLOADED_ICONS).map(|_| icon()).collect());
        let command_buffer = tools.queue.command_buffer().unwrap();
        let icons: Vec<Texture> =
            made_first.unwrap_or_else(|| (0..UPLOADED_ICONS).map(|_| icon()).collect());
        let targets: Vec<Vec<Texture>> = icons
            .iter()
            .map(|icon| {
                (0..UPLOADED_LEVELS)
                    .map(|level| {
                        let side = side(level);
                        let mut d = TextureDescriptor::texture_2d(
                            PixelFormat::RGBA16Float,
                            side,
                            side,
                            false,
                        );
                        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                        d.resource_options = ResourceOptions::private();
                        let target = device.new_texture(&d).unwrap();
                        let sampler = &pinned[level as usize];
                        let draw: Draw = (&tools.implicit, None, Fragment::Sampled(icon, sampler));
                        tools.pass(&command_buffer, (&target, 0), Some([0.0; 4]), &[draw]);
                        target
                    })
                    .collect()
            })
            .collect();
        finish(&command_buffer);
        counts.push(moved(&before, &device.work_counts()));
        for (i, (icon, targets)) in icons.iter().zip(&targets).enumerate() {
            let mut wrong = Vec::new();
            for (level, target) in (0..UPLOADED_LEVELS).zip(targets) {
                let command_buffer = tools.queue.command_buffer().unwrap();
                tools.read(&command_buffer, (target, 0), 0);
                tools.read(&command_buffer, (icon, level), SIDE * SIDE * TEXEL);
                finish(&command_buffer);
                let sampled = tools.wrong(0, level, |x| want(level, x));
                let read = tools.wrong(SIDE * SIDE * TEXEL, level, |x| want(level, x));
                if sampled.0 != 0 || read.0 != 0 {
                    wrong.push(format!("level {level}: sampled {sampled:?}, read {read:?}"));
                }
            }
            if !wrong.is_empty() {
                failures.push(format!("round {round} icon {i}: {wrong:?}"));
            }
        }
    }
    eprintln!(
        "ICON-MIPS {case} on {}: {} of {} icons wrong; per round (storage forms taken, images \
         recycled, swapped-out images forgotten, reads through copies, slots narrowed) \
         {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * UPLOADED_ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

#[test]
fn uploaded_managed_levels_sampled_through_pinned_samplers_read_what_was_uploaded() {
    run_uploaded("managed", StorageMode::Managed, false);
}

#[test]
fn managed_levels_uploaded_while_the_sampling_command_buffer_is_open_read_what_was_uploaded() {
    run_uploaded("managed-open", StorageMode::Managed, true);
}

#[test]
fn uploaded_shared_levels_sampled_through_pinned_samplers_read_what_was_uploaded() {
    run_uploaded("shared", StorageMode::Shared, false);
}

#[test]
fn levels_rendered_in_place_read_back() {
    run("in-place", Way::InPlace, Event::None);
}

#[test]
fn levels_rendered_in_place_survive_a_storage_form_at_the_commit() {
    run("in-place-form", Way::InPlace, Event::FormAtCommit);
}

#[test]
fn levels_rendered_in_place_survive_a_storage_form_at_a_later_commit() {
    run("in-place-form-later", Way::InPlace, Event::FormLater);
}

#[test]
fn levels_rendered_in_place_over_a_recycled_image_read_their_own() {
    run("in-place-recycled", Way::InPlace, Event::Recycled);
}

#[test]
fn levels_rendered_through_the_copy_read_back() {
    run("through-copy", Way::ThroughCopy, Event::None);
}

#[test]
fn levels_rendered_through_the_copy_survive_a_storage_form_at_the_commit() {
    run("through-copy-form", Way::ThroughCopy, Event::FormAtCommit);
}

#[test]
fn levels_rendered_through_the_copy_survive_a_storage_form_at_a_later_commit() {
    run(
        "through-copy-form-later",
        Way::ThroughCopy,
        Event::FormLater,
    );
}

#[test]
fn levels_rendered_through_the_copy_over_a_recycled_image_read_their_own() {
    run("through-copy-recycled", Way::ThroughCopy, Event::Recycled);
}

#[test]
fn generated_mipmaps_read_back() {
    run("blitted", Way::Blitted, Event::None);
}

#[test]
fn generated_mipmaps_survive_a_storage_form_at_the_commit() {
    run("blitted-form", Way::Blitted, Event::FormAtCommit);
}

#[test]
fn generated_mipmaps_survive_a_storage_form_at_a_later_commit() {
    run("blitted-form-later", Way::Blitted, Event::FormLater);
}

#[test]
fn generated_mipmaps_over_a_recycled_image_read_their_own() {
    run("blitted-recycled", Way::Blitted, Event::Recycled);
}

#[test]
fn replaced_levels_read_back() {
    run("replaced", Way::Replaced, Event::None);
}

#[test]
fn replaced_levels_survive_a_fetch_at_the_commit() {
    run("replaced-form", Way::Replaced, Event::FormAtCommit);
}

#[test]
fn replaced_levels_survive_a_fetch_before_a_later_commit() {
    run("replaced-form-later", Way::Replaced, Event::FormLater);
}

#[test]
fn replaced_levels_over_a_recycled_image_read_their_own() {
    run("replaced-recycled", Way::Replaced, Event::Recycled);
}
