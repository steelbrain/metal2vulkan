//! Passes of one color whose fetching draws render it alone: no other color, no depth or stencil.
//!
//! A device that fetches through the pixel interlock leaves a color out of the subpass of the
//! draws that read and write it so (`render_target.rs`), so such a draw's instance has no
//! attachment left in its subpass at all; where a draw that does not fetch the color comes before
//! it, the pass runs as two instances. Where the color's image lacks `STORAGE` (the M3), the pass
//! renders it through a scratch copied in and back (`attachment_stand_in.rs`), and its texture
//! takes its storage form at a later commit (`texture/storage_form.rs`).
//!
//! Each round, every target new: a pass clears it and draws a flag over its left half
//! (`dcs_flag`); a second loads it, draws the flag over its top rows and then halves every pixel
//! through a fetch (`fetch_ordered`); a third loads it and halves every pixel again. Each pixel
//! then holds a quarter of the flag (the left half and the top rows) or of the clear. Two
//! formats: `RG16Float`, a distance filter's, whose halves are exact, and `BGRA8Unorm`, a color
//! filter's, within one code of the rounded quarter.
//!
//! A second pass shape: a bitmap's pass, a cleared and stored `RGBA16Float` beside a new
//! `RG16Float` second color cleared and stored `DontCare` and a `Stencil8`, whose one draw fetches
//! the second color and writes nothing to it (`dcs_read`): the first color holds the clear's
//! `(x, y, 1, 1)`.
//!
//! A third: an accumulation's pass, the same attachments but the second color cleared to
//! `(1, 0)`, whose one draw lays [`LAYERS`] copies of a rectangle over the whole target, each
//! fragment fetching the second color and writing `w = c1.x / 2 + c1.y` into the first and
//! `(w, c1.y / 4)` into the second (`con_fetch_coverage`): the interlock orders the overlapping
//! fragments of the draw, so the first color holds `2^-LAYERS` in every channel.
//!
//! The functions are `dcs_rect`, `dcs_flag` and `dcs_read` of the owned fixture
//! `validation/fixtures/public/engine/dcs_dont_care_split.ll`, `fetch_ordered` of
//! `fragment_fetch.ll` and `con_fetch_coverage` of `con_fetch_coverage.ll` there.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The targets' side, the targets each round renders, and the rounds.
const SIDE: u64 = 192;
const TARGETS: usize = 24;
const ROUNDS: usize = 3;

/// The draws' rectangles, in pixels (left, top, right, bottom).
const LEFT: [u64; 4] = [0, 0, SIDE / 2, SIDE];
const TOP: [u64; 4] = [0, 0, SIDE, SIDE / 4];
const WHOLE: [u64; 4] = [0, 0, SIDE, SIDE];

/// The first pass's clear, and what `dcs_flag` writes.
const CLEAR: [f32; 4] = [0.25, 0.75, 0.0, 1.0];
const FLAG: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The bitmap pass's second color's clear, exact in a half.
const SECOND: [f32; 2] = [0.375, 0.625];

/// The accumulation's layers in its one draw: each halves the second color's red, so the last
/// leaves `2^-LAYERS`, a normal half.
const LAYERS: u32 = 12;

/// A half's 1.0.
const ONE: u16 = 0x3c00;

/// The pass shapes a run renders.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// The three passes of the color alone.
    Lone,
    /// The bitmap's pass, whose draw reads the second color.
    Bitmap,
    /// The accumulation's pass, one draw of [`LAYERS`] overlapping rectangles.
    Layers,
}

/// A pass's draws: each state, its rectangle and its instances.
type Draws<'a> = &'a [(&'a RenderPipelineState, [u64; 4], u32)];

/// The queue, the states, the target's size as `dcs_rect` takes it, the coverage's tint, and a
/// read-back buffer.
struct Tools {
    queue: CommandQueue,
    flag: RenderPipelineState,
    halve: RenderPipelineState,
    read: RenderPipelineState,
    coverage: RenderPipelineState,
    size: Vec<u8>,
    tint: Vec<u8>,
    back: Buffer,
}

impl Tools {
    /// The tools for targets of `format`, or `None` where the device makes no state that fetches.
    fn new(device: &Device, format: PixelFormat) -> Option<Tools> {
        let dcs = library(device, "engine/dcs_dont_care_split.ll");
        let fetch = library(device, "engine/fragment_fetch.ll");
        let con = library(device, "engine/con_fetch_coverage.ll");
        let function = |library: &crate::Library, name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let state = |fragment: StageSource, second: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function(&dcs, "dcs_rect"));
            descriptor.fragment = Some(fragment);
            if second {
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
                descriptor.color_attachments[1] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
                descriptor.stencil_format = Some(PixelFormat::Stencil8);
            } else {
                descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
            }
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        Some(Tools {
            queue: device.new_command_queue().unwrap(),
            flag: state(function(&dcs, "dcs_flag"), false).unwrap(),
            halve: state(function(&fetch, "fetch_ordered"), false).ok()?,
            read: state(function(&dcs, "dcs_read"), true).ok()?,
            coverage: state(function(&con, "con_fetch_coverage"), true).ok()?,
            size: bytes_of(&[SIDE as f32, SIDE as f32], f32::to_le_bytes),
            tint: bytes_of(&[ONE; 4], u16::to_le_bytes),
            back: device
                .new_buffer(SIDE * SIDE * 8 * TARGETS as u64, ResourceOptions::shared())
                .unwrap(),
        })
    }

    /// A new `Private` texture of `format` at the side, of `usage`.
    fn target(&self, format: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        self.queue.device().new_texture(&d).unwrap()
    }

    /// `draws` encoded into `command_buffer` in a pass of `pass`.
    fn pass(&self, command_buffer: &CommandBuffer, pass: &RenderPassDescriptor, draws: Draws) {
        let mut encoder = command_buffer.render_command_encoder(pass).unwrap();
        encoder.set_fragment_bytes(&self.tint, 0).unwrap();
        for &(state, rect, instances) in draws {
            encoder.set_render_pipeline_state(state).unwrap();
            let rect = bytes_of(&rect.map(|p| p as f32), f32::to_le_bytes);
            encoder.set_vertex_bytes(&rect, 0).unwrap();
            encoder.set_vertex_bytes(&self.size, 1).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, instances, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// The three passes over `target`, the color alone.
    fn render(&self, command_buffer: &CommandBuffer, target: &Texture) {
        let lone = |load: LoadAction| {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = load;
            let [r, g, b, a] = CLEAR.map(f64::from);
            color.clear_color = ClearColor::new(r, g, b, a);
            color.store_action = StoreAction::Store;
            pass
        };
        let pass = lone(LoadAction::Clear);
        self.pass(command_buffer, &pass, &[(&self.flag, LEFT, 1)]);
        let draws = [(&self.flag, TOP, 1), (&self.halve, WHOLE, 1)];
        self.pass(command_buffer, &lone(LoadAction::Load), &draws);
        let pass = lone(LoadAction::Load);
        self.pass(command_buffer, &pass, &[(&self.halve, WHOLE, 1)]);
    }

    /// A pass over `first`, cleared and stored, beside a new second color cleared to `second`
    /// and not stored and a new stencil, of `draws`.
    fn beside(
        &self,
        command_buffer: &CommandBuffer,
        first: &Texture,
        second: [f32; 2],
        draws: Draws,
    ) {
        let coverage = self.target(PixelFormat::RG16Float, TextureUsage::RENDER_TARGET);
        let stencil = self.target(PixelFormat::Stencil8, TextureUsage::RENDER_TARGET);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(first.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
        let color = &mut pass.color_attachments[1];
        color.texture = Some(coverage);
        color.load_action = LoadAction::Clear;
        let [x, y] = second.map(f64::from);
        color.clear_color = ClearColor::new(x, y, 0.0, 0.0);
        color.store_action = StoreAction::DontCare;
        let depth_stencil = &mut pass.stencil_attachment;
        depth_stencil.texture = Some(stencil);
        depth_stencil.load_action = LoadAction::DontCare;
        depth_stencil.store_action = StoreAction::DontCare;
        self.pass(command_buffer, &pass, draws);
    }

    /// Every target copied into the read-back buffer, `texel` bytes a pixel.
    fn read_back(&self, targets: &[Texture], texel: u64) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (index, target) in targets.iter().enumerate() {
            let bytes = SIDE * SIDE * texel;
            blit.copy_from_texture_to_buffer(
                target,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &self.back,
                index as u64 * bytes,
                SIDE * texel,
                bytes,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
}

/// The bits of a half that holds `value` exactly, `value` zero or normal.
fn half(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exponent as u16) << 10) | ((bits & 0x7f_ffff) >> 13) as u16
}

/// What the three passes leave at `(x, y)`, red, green, blue, alpha: a quarter of the flag or
/// of the clear.
fn expected((x, y): (u64, u64)) -> [f32; 4] {
    let flagged = x < LEFT[2] || y < TOP[3];
    (if flagged { FLAG } else { CLEAR }).map(|c| c / 4.0)
}

/// Whether target `index`'s texels in the read-back buffer, of `format`, are each `expected` of
/// its pixel; else the first wrong pixel and its bytes.
fn first_wrong(
    back: &Buffer,
    format: PixelFormat,
    index: usize,
    expected: impl Fn((u64, u64)) -> [f32; 4],
) -> Option<((u64, u64), Vec<u8>)> {
    let texel = texel(format);
    let span = (SIDE * SIDE * texel) as usize;
    let contents = back.contents().unwrap();
    // SAFETY: bytes within the buffer, which no command buffer writes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr().add(index * span), span) };
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .find_map(|(x, y)| {
            let at = ((y * SIDE + x) * texel) as usize;
            let got = &bytes[at..at + texel as usize];
            let want = expected((x, y));
            let h = |i: usize| u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
            let near = |code: u8, value: f32| (f32::from(code) - value * 255.0).abs() <= 1.0;
            let right = match format {
                PixelFormat::RG16Float => (0..2).all(|i| h(i) == half(want[i])),
                PixelFormat::RGBA16Float => (0..4).all(|i| h(i) == half(want[i])),
                // Blue, green, red, alpha.
                _ => [2, 1, 0, 3]
                    .iter()
                    .zip(got)
                    .all(|(&i, &code)| near(code, want[i])),
            };
            (!right).then(|| ((x, y), got.to_vec()))
        })
}

/// The bytes of a pixel of `format`.
fn texel(format: PixelFormat) -> u64 {
    match format {
        PixelFormat::RGBA16Float => 8,
        _ => 4,
    }
}

/// The counts that say how the passes ran: render pass instances, splits for the interlocked
/// writes, storage forms taken, variant pipelines made.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.render_passes_begun_instances - before.render_passes_begun_instances,
        after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
        after.storage_forms_taken - before.storage_forms_taken,
        after.variant_pipelines_made - before.variant_pipelines_made,
    ]
}

/// The run of `shape` over targets of `format` (the bitmap's and the accumulation's
/// `RGBA16Float`).
fn run(case: &str, format: PixelFormat, shape: Shape) {
    let scratch = Scratch::new(&format!("lone-fetch-{case}"));
    let device = device(&scratch.0);
    let Some(tools) = Tools::new(&device, format) else {
        eprintln!(
            "LONE-FETCH {case} on {}: no framebuffer fetch: nothing to check",
            device.name()
        );
        return;
    };
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for round in 0..ROUNDS {
        let before = device.work_counts();
        let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let targets: Vec<Texture> = (0..TARGETS).map(|_| tools.target(format, usage)).collect();
        let command_buffer = tools.queue.command_buffer().unwrap();
        for target in &targets {
            match shape {
                Shape::Lone => tools.render(&command_buffer, target),
                Shape::Bitmap => {
                    let draws = [(&tools.read, WHOLE, 1)];
                    tools.beside(&command_buffer, target, SECOND, &draws);
                }
                Shape::Layers => {
                    let draws = [(&tools.coverage, WHOLE, LAYERS)];
                    tools.beside(&command_buffer, target, [1.0, 0.0], &draws);
                }
            }
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        counts.push(moved(&before, &device.work_counts()));
        tools.read_back(&targets, texel(format));
        for index in 0..TARGETS {
            let wrong = match shape {
                Shape::Lone => first_wrong(&tools.back, format, index, expected),
                Shape::Bitmap => first_wrong(&tools.back, format, index, |_| {
                    [SECOND[0], SECOND[1], 1.0, 1.0]
                }),
                Shape::Layers => first_wrong(&tools.back, format, index, |_| {
                    [(-(LAYERS as f32)).exp2(); 4]
                }),
            };
            if let Some(first) = wrong {
                failures.push(format!(
                    "round {round} target {index}: first wrong {first:x?}"
                ));
            }
        }
    }
    eprintln!(
        "LONE-FETCH {case} on {}: {} of {} targets wrong; per round (instances, interlocked \
         splits, storage forms, variants made) {counts:?}; first {:?}",
        device.name(),
        failures.len(),
        ROUNDS * TARGETS,
        failures.first()
    );
    assert!(failures.is_empty(), "{case}: {failures:?}");
}

#[test]
fn a_lone_fetched_rg16float_color_keeps_its_values() {
    run("rg16float", PixelFormat::RG16Float, Shape::Lone);
}

#[test]
fn a_lone_fetched_bgra8unorm_color_keeps_its_values() {
    run("bgra8unorm", PixelFormat::BGRA8Unorm, Shape::Lone);
}

#[test]
fn a_second_color_fetched_and_not_written_reads_its_clear() {
    run("bitmap", PixelFormat::RGBA16Float, Shape::Bitmap);
}

#[test]
fn the_overlapping_layers_of_one_fetching_draw_accumulate_in_order() {
    run("layers", PixelFormat::RGBA16Float, Shape::Layers);
}
