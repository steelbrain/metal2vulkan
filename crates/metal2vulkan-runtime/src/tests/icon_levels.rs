//! An icon's levels rendered as a window server renders them: a `Private` `RGBA16Float` render
//! target of three levels, each level in a pass of its own that clears it and stores it, all in
//! one command buffer: level 0's pass draws the fill, and each later level's pass draws from the
//! level before it, sampling the texture itself, bound whole (`texture/attached_reads.rs`). Then,
//! in a later command buffer, the icon is sampled into targets of its levels' sizes, so each reads
//! one level. Every level, and every pixel sampled from one, holds the fill. Icons are made and
//! released in rounds, as the server makes one per icon each time it draws the dock again.
//!
//! Two forms of the later levels' reads:
//! - **Through the copy.** `tob_sample` reads the texture at the pixel's position over level 0's
//!   size, so its level of detail is 0 at any level's pass; the runtime cannot bound it below the
//!   attached level, so the slot reads the texture's copy, which must hold what level 0's pass
//!   stored.
//! - **In place.** `alv_implicit` through a sampler clamped to level 0 draws level 1, and
//!   `alv_level1` level 2: both read only levels before the attached one, through a view of them
//!   (`render/attached_levels.rs`).
//!
//!
//! Where level 0's pass also fetches its attachment in a raster order group (the fill drawn at
//! twice its value, then halved), a device whose fetch needs `STORAGE` the image lacks (the M3)
//! renders that pass through a scratch, and the icon takes its storage form at a commit
//! (`texture/storage_form.rs`): the icons' own, or, where another command buffer was uncommitted
//! as they committed, that one's.
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
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The icon's level-0 side, and its levels.
const SIDE: u64 = 64;
const LEVELS: u64 = 3;

/// An `RGBA16Float` texel's bytes.
const TEXEL: u64 = 8;

/// The fill, and its bits as halves.
const FILL: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
const FILL_BITS: [u16; 4] = [0x3400, 0x3800, 0x3a00, 0x3c00];

/// The icons each round makes, and the rounds.
const ICONS: usize = 16;
const ROUNDS: usize = 3;

/// The side of level `level`.
fn side(level: u64) -> u64 {
    (SIDE >> level).max(1)
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

/// How a run renders its icons.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// The later levels read through the copy, else in place.
    through_copy: bool,
    /// Level 0's pass fetches its attachment after the fill (`fetch_ordered`, halving the fill
    /// drawn at twice its value): the M3 renders it through a scratch and the texture takes its
    /// storage form at a commit (`texture/storage_form.rs`).
    fetch: bool,
    /// The icons' command buffer commits while another is uncommitted, which commits after it
    /// with nothing in it: a storage form rides on that later commit.
    later_commit: bool,
}

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
    /// `tob_rect`'s corners covering the whole target, and `tob_solid`'s colors: the fill, and
    /// twice it.
    whole: Vec<u8>,
    fill: Vec<u8>,
    double: Vec<u8>,
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
        // The vertex function of `fixture`, or of `tob_renders.ll` for `tob_rect`, and the
        // fragment function of `fixture`.
        let state = |fixture: &str, vertex: &str, fragment: &str| {
            let vertex_fixture = match vertex {
                "tob_rect" => "engine/tob_renders.ll",
                _ => fixture,
            };
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
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: state(tob, "tob_rect", "tob_solid").unwrap(),
            sample: state(tob, "tob_rect", "tob_sample").unwrap(),
            implicit: state(alv, "alv_rect", "alv_implicit").unwrap(),
            level1: state(alv, "alv_rect", "alv_level1").unwrap(),
            fetch: state("engine/fragment_fetch.ll", "tob_rect", "fetch_ordered").ok(),
            sampler: sampler(f32::MAX),
            level0: sampler(0.0),
            whole: bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes),
            fill: bytes_of(&FILL, f32::to_le_bytes),
            double: bytes_of(&FILL.map(|c| c * 2.0), f32::to_le_bytes),
            back: device
                .new_buffer(SIDE * SIDE * TEXEL * LEVELS, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A pass over `level` of `target`, cleared to zero and stored, drawing each of `draws` once.
    fn pass(
        &self,
        command_buffer: &CommandBuffer,
        (target, level): (&Texture, u64),
        draws: &[Draw],
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.level = level;
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
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

    /// A new icon, its levels' passes encoded in `command_buffer`, the later levels read through
    /// the copy or in place.
    fn icon(&self, command_buffer: &CommandBuffer, shape: Shape) -> Texture {
        let device = self.queue.device();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, true);
        d.mipmap_level_count = LEVELS;
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let icon = device.new_texture(&d).unwrap();
        let whole = Some(self.whole.as_slice());
        let first: Vec<Draw> = match (shape.fetch, &self.fetch) {
            (true, Some(fetch)) => vec![
                (&self.solid, whole, Fragment::Bytes(&self.double)),
                (fetch, whole, Fragment::Nothing),
            ],
            _ => vec![(&self.solid, whole, Fragment::Bytes(&self.fill))],
        };
        self.pass(command_buffer, (&icon, 0), &first);
        for level in 1..LEVELS {
            let draw: Draw = match (shape.through_copy, level) {
                (true, _) => (&self.sample, whole, Fragment::Sampled(&icon, &self.sampler)),
                (false, 1) => (&self.implicit, None, Fragment::Sampled(&icon, &self.level0)),
                (false, _) => (&self.level1, None, Fragment::Sampled(&icon, &self.sampler)),
            };
            self.pass(command_buffer, (&icon, level), &[draw]);
        }
        icon
    }

    /// A target of `side` that `icon` is sampled into, so its draw reads the level of that side.
    fn sampled(&self, command_buffer: &CommandBuffer, icon: &Texture, side: u64) -> Texture {
        let device = self.queue.device();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, side, side, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let target = device.new_texture(&d).unwrap();
        let draw: Draw = (&self.implicit, None, Fragment::Sampled(icon, &self.sampler));
        self.pass(command_buffer, (&target, 0), &[draw]);
        target
    }

    /// Copy `level` of `texture`, of `side` texels a side, into the read-back buffer at `offset`.
    fn read(
        &self,
        command_buffer: &CommandBuffer,
        (texture, level, side): (&Texture, u64, u64),
        offset: u64,
    ) {
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

    /// How many texels of the read-back buffer from `offset`, `count` of them, are not the fill,
    /// and the first such.
    fn wrong(&self, offset: u64, count: u64) -> (usize, Option<[u16; 4]>) {
        let contents = self.back.contents().unwrap();
        // SAFETY: bytes within the read-back buffer, which no command buffer writes now.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                contents.as_ptr().add(offset as usize),
                (count * TEXEL) as usize,
            )
        };
        let texels = bytes.as_chunks::<8>().0.iter().map(|t| {
            let h = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
            [h(0), h(2), h(4), h(6)]
        });
        let wrong: Vec<[u16; 4]> = texels.filter(|&texel| texel != FILL_BITS).collect();
        (wrong.len(), wrong.first().copied())
    }

    /// The wrong texels of each level of the read-back buffer.
    fn wrong_levels(&self) -> Vec<(usize, Option<[u16; 4]>)> {
        (0..LEVELS)
            .map(|level| self.wrong(offset(level), side(level) * side(level)))
            .collect()
    }
}

/// The read-back offsets of the levels, one after another.
fn offset(level: u64) -> u64 {
    (0..level).map(|l| side(l) * side(l) * TEXEL).sum()
}

/// The counts that say which form the later levels' reads took, and the storage forms taken.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.attachment_reads_through_copies - before.attachment_reads_through_copies,
        after.attached_read_slots_narrowed - before.attached_read_slots_narrowed,
        after.shadow_copies - before.shadow_copies,
        after.storage_forms_taken - before.storage_forms_taken,
    ]
}

fn run(case: &str, shape: Shape) {
    let scratch = Scratch::new(&format!("icon-levels-{case}"));
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    if shape.fetch && tools.fetch.is_none() {
        eprintln!(
            "ICON-LEVELS {case} on {}: no framebuffer fetch: nothing to check",
            device.name()
        );
        return;
    }
    let mut kept: Vec<Texture> = Vec::new();
    let mut failures = Vec::new();
    let mut counts = [0; 4];
    for round in 0..ROUNDS {
        // The round's icons, in one command buffer, as the server renders them.
        let before = device.work_counts();
        let command_buffer = tools.queue.command_buffer().unwrap();
        let later = (shape.later_commit).then(|| tools.queue.command_buffer().unwrap());
        let icons: Vec<Texture> = (0..ICONS)
            .map(|_| tools.icon(&command_buffer, shape))
            .collect();
        command_buffer.commit().unwrap();
        if let Some(later) = &later {
            later.commit().unwrap();
        }
        // The last round's icons go as the new ones are rendered.
        kept.clear();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        if let Some(later) = later {
            later.wait_until_completed_for(WAIT).unwrap();
        }
        let round_counts = moved(&before, &device.work_counts());
        counts = [0, 1, 2, 3].map(|i| counts[i] + round_counts[i]);
        for (i, icon) in icons.iter().enumerate() {
            // Each icon drawn at each level's size in a later command buffer, then every level
            // of it read back; then the sampled targets.
            let command_buffer = tools.queue.command_buffer().unwrap();
            let targets: Vec<Texture> = (0..LEVELS)
                .map(|level| tools.sampled(&command_buffer, icon, side(level)))
                .collect();
            for level in 0..LEVELS {
                tools.read(&command_buffer, (icon, level, side(level)), offset(level));
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let levels = tools.wrong_levels();
            let command_buffer = tools.queue.command_buffer().unwrap();
            for (level, target) in targets.iter().enumerate() {
                tools.read(
                    &command_buffer,
                    (target, 0, side(level as u64)),
                    offset(level as u64),
                );
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let sampled = tools.wrong_levels();
            if levels.iter().chain(&sampled).any(|&(n, _)| n != 0) {
                failures.push(format!(
                    "round {round} icon {i}: wrong texels by level {levels:x?}, sampled \
                     {sampled:x?}"
                ));
            }
        }
        kept.extend(icons);
    }
    eprintln!(
        "ICON-LEVELS {case} on {}: {} of {} icons wrong; (reads through copies, slots narrowed, \
         shadow copies, storage forms) {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * ICONS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

/// A shape of the icons' render.
fn shape(through_copy: bool, fetch: bool, later_commit: bool) -> Shape {
    Shape {
        through_copy,
        fetch,
        later_commit,
    }
}

#[test]
fn an_icons_levels_drawn_through_the_copy_hold_its_fill_when_sampled_later() {
    run("through-copy", shape(true, false, false));
}

#[test]
fn an_icons_levels_drawn_in_place_hold_its_fill_when_sampled_later() {
    run("in-place", shape(false, false, false));
}

#[test]
fn a_fetched_icons_levels_drawn_through_the_copy_hold_its_fill_when_sampled_later() {
    run("fetched-through-copy", shape(true, true, false));
}

#[test]
fn a_fetched_icons_levels_hold_its_fill_where_its_storage_form_rides_a_later_commit() {
    run("fetched-later-commit", shape(true, true, true));
}
