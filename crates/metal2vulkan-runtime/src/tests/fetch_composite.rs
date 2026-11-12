//! A composite pass that both samples icon textures and fetches its target, as a window server's
//! compositor draws a dock: icons sampled into the target, and glass drawn over them reading the
//! target's own pixels (framebuffer fetch in a raster order group). Every icon draw's pixels hold
//! the icon's color, and every fetch halves what was under it, in draw order, whether the pass
//! renders the target in place, splits into instances, or renders it through a scratch image
//! (`attachment_stand_in.rs`).
//!
//! An `RGBA16Float` icon is filled by a pass clearing it to [`ICON`]; the composite pass clears
//! its `RGBA16Float` target to [`BACKGROUND`] (or loads it, the CPU having written it), then draws
//! the icon at [`A`], fetches [`F`], draws the icon at [`A2`] and fetches [`F2`], which covers half
//! of [`A`]. The target is a `Private` render target, a texture over a `Shared` buffer's bytes (a
//! linear image, as a surface's back buffer is), or a texture over a no-copy buffer's host pages,
//! loaded, as a guest's surface is. Over host pages in place, where the device's check passed for
//! a shader's writes of its texels, its image takes the storage usage the fetch reads through
//! (`texture/in_place_image.rs`), so the pass fetches it in place: with the feedback loop usage or
//! without it (MoltenVK, which has no `VK_EXT_attachment_feedback_loop_layout`). Each target is
//! composited twice: in the icon's command buffer, then in a later one with a new icon, the first
//! released.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device without a framebuffer fetch in
//! the pixel interlock says so and checks nothing.
//!
//! An `RGBA8Unorm_sRGB` target over host pages, loaded, renders through a scratch on a device that
//! fetches through storage, whatever its image: a fetching draw reads and writes it through an
//! unsigned-integer view no texture's image is made mutable to (`framebuffer_fetch.rs`). Where
//! the pass's first instance (the first icon) recorded inline into the target itself, the scratch
//! arrives with the first fetch and takes the target's texels after that instance, and the words
//! the later instances changed are merged back into the host pages (`linear_merge.rs`): kept
//! after that copy, not before the pass, where the scratch held what it held before (in the
//! second round, the image of the first round's scratch, recycled, with that round's pixels).
//! Its codes are the device's own sRGB conversions, compared within one code.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::through_storage;
use crate::in_place::{Access, Padding, PagesClass, RowEnds, Verdict};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// Where the composite's target lies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Home {
    /// A `Private` render target.
    Private,
    /// A texture over a `Shared` buffer's bytes.
    Buffer,
    /// A texture over a no-copy buffer's host pages.
    HostPages,
}

/// The target's pixel format.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// `RGBA16Float`, a texel's channels the bits of halves.
    Half,
    /// `RGBA8Unorm_sRGB`, a texel's channels 8-bit codes.
    Srgb,
}

impl Format {
    fn pixel_format(self) -> PixelFormat {
        match self {
            Format::Half => PixelFormat::RGBA16Float,
            Format::Srgb => PixelFormat::RGBA8UnormSRGB,
        }
    }

    /// A texel's bytes.
    fn texel(self) -> u64 {
        match self {
            Format::Half => 8,
            Format::Srgb => 4,
        }
    }
}

/// The target's side, in pixels.
const SIDE: u64 = 128;

/// The icon's side, in texels.
const ICON_SIDE: u64 = 64;

/// The icon's color and the target's clear color.
const ICON: [f64; 4] = [0.25, 0.5, 0.75, 1.0];
const BACKGROUND: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

/// The draws' rectangles, `[x0, y0, x1, y1]` in pixels.
const A: [u64; 4] = [8, 8, 40, 40];
const F: [u64; 4] = [48, 8, 80, 40];
const A2: [u64; 4] = [88, 8, 120, 40];
const F2: [u64; 4] = [8, 24, 40, 56];

/// A draw of the composite pass.
#[derive(Clone, Copy)]
enum Draw {
    Icon([u64; 4]),
    Fetch([u64; 4]),
}

const DRAWS: [Draw; 4] = [
    Draw::Icon(A),
    Draw::Fetch(F),
    Draw::Icon(A2),
    Draw::Fetch(F2),
];

/// The state of `tob_rect` and `fixture`'s `fragment`, into a color of `format`.
fn state(
    device: &Device,
    (fixture, fragment): (&str, &str),
    format: Format,
) -> Result<RenderPipelineState, crate::Error> {
    let function = |path: &str, name: &str| {
        let function = library(device, path).new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(function("engine/tob_renders.ll", "tob_rect"));
    descriptor.fragment = Some(function(fixture, fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format.pixel_format()));
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    bytes_of(&[x(x0), y(y1), x(x1), y(y0)], f32::to_le_bytes)
}

/// The bits of `value` as a half, for the values the test writes (exact in a half).
fn half(value: f64) -> u16 {
    match value {
        1.0 => 0x3c00,
        0.75 => 0x3a00,
        0.5 => 0x3800,
        0.375 => 0x3600,
        0.25 => 0x3400,
        0.125 => 0x3000,
        v => panic!("no half for {v}"),
    }
}

/// The 8-bit sRGB code of `value` in channel `channel` (alpha linear).
fn code(channel: usize, value: f64) -> u16 {
    let encoded = match channel {
        3 => value,
        _ if value <= 0.0031308 => value * 12.92,
        _ => 1.055 * value.powf(1.0 / 2.4) - 0.055,
    };
    (encoded * 255.0).round() as u16
}

/// The value of the 8-bit sRGB code `code` in channel `channel` (alpha linear).
fn value(channel: usize, code: u16) -> f64 {
    let encoded = f64::from(code) / 255.0;
    match channel {
        3 => encoded,
        _ if encoded <= 0.04045 => encoded / 12.92,
        _ => ((encoded + 0.055) / 1.055).powf(2.4),
    }
}

/// The texel `color` stores in `format`.
fn stored(format: Format, color: [f64; 4]) -> [u16; 4] {
    match format {
        Format::Half => color.map(half),
        Format::Srgb => std::array::from_fn(|channel| code(channel, color[channel])),
    }
}

/// The texel the composite leaves at pixel `(x, y)` of a target of `format`: each draw's over
/// what the target stores, a fetch reading the stored texel back.
fn expected(format: Format, x: u64, y: u64) -> [u16; 4] {
    let within = |[x0, y0, x1, y1]: [u64; 4]| x >= x0 && x < x1 && y >= y0 && y < y1;
    let mut texel = stored(format, BACKGROUND);
    for draw in DRAWS {
        match draw {
            Draw::Icon(rect) if within(rect) => texel = stored(format, ICON),
            Draw::Fetch(rect) if within(rect) => {
                let read: [f64; 4] = match format {
                    // Every half the test writes, halved, is exact.
                    Format::Half => std::array::from_fn(|channel| half_value(texel[channel]) * 0.5),
                    Format::Srgb => {
                        std::array::from_fn(|channel| value(channel, texel[channel]) * 0.5)
                    }
                };
                texel = stored(format, read);
            }
            _ => {}
        }
    }
    texel
}

/// The value of the half bits `bits`, for the values the test writes.
fn half_value(bits: u16) -> f64 {
    [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]
        .into_iter()
        .find(|&v| half(v) == bits)
        .unwrap_or_else(|| panic!("no value for half {bits:#x}"))
}

/// Whether `got` is the composite's `want` in `format`: the same bits of halves, or 8-bit sRGB
/// codes within one of it (the device's own conversions).
fn matches(format: Format, got: [u16; 4], want: [u16; 4]) -> bool {
    match format {
        Format::Half => got == want,
        Format::Srgb => (got.iter().zip(want)).all(|(&g, w)| g.abs_diff(w) <= 1),
    }
}

/// The tools: the queue, the two states, the icon's sampler, and the target's format.
struct Tools {
    queue: CommandQueue,
    sample: RenderPipelineState,
    fetch: RenderPipelineState,
    sampler: SamplerState,
    format: Format,
}

impl Tools {
    /// An icon, its pass encoded in `command_buffer`: cleared to [`ICON`] and stored.
    fn icon(&self, command_buffer: &CommandBuffer) -> Texture {
        let device = self.queue.device();
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, ICON_SIDE, ICON_SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let icon = device.new_texture(&d).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(icon.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(ICON[0], ICON[1], ICON[2], ICON[3]);
        color.store_action = StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        icon
    }

    /// The composite pass over `target`, sampling `icon`, encoded in `command_buffer`: loading
    /// the target where `load`, else clearing it to [`BACKGROUND`].
    fn composite(
        &self,
        command_buffer: &CommandBuffer,
        (target, load): (&Texture, bool),
        icon: &Texture,
    ) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = match load {
            true => LoadAction::Load,
            false => LoadAction::Clear,
        };
        let [r, g, b, a] = BACKGROUND;
        color.clear_color = ClearColor::new(r, g, b, a);
        color.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        for draw in DRAWS {
            let rect = match draw {
                Draw::Icon(rect) => {
                    encoder.set_render_pipeline_state(&self.sample).unwrap();
                    encoder.set_fragment_texture(Some(icon), 0).unwrap();
                    (encoder.set_fragment_sampler_state(Some(&self.sampler), 0)).unwrap();
                    rect
                }
                Draw::Fetch(rect) => {
                    encoder.set_render_pipeline_state(&self.fetch).unwrap();
                    rect
                }
            };
            encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// `target`'s texels, copied into `back` by a blit in `command_buffer`.
    fn read(&self, command_buffer: &CommandBuffer, target: &Texture, back: &Buffer) {
        let texel = self.format.texel();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            back,
            0,
            SIDE * texel,
            SIDE * SIDE * texel,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
}

/// `buffer`'s texels of `format`, a host-visible buffer no command writes now.
fn texels(buffer: &Buffer, format: Format) -> Vec<[u16; 4]> {
    let contents = buffer.contents().unwrap();
    let length = (SIDE * SIDE * format.texel()) as usize;
    // SAFETY: the buffer's first `length` bytes, which no command buffer writes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) };
    match format {
        Format::Half => (bytes.as_chunks::<8>().0.iter())
            .map(|t| {
                let h = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
                [h(0), h(2), h(4), h(6)]
            })
            .collect(),
        Format::Srgb => (bytes.as_chunks::<4>().0.iter())
            .map(|t| t.map(u16::from))
            .collect(),
    }
}

/// The pixels of `texels` (of `format`) that differ from the composite's answer, with their
/// texels, at most the first 8, and how many differ.
fn wrong(texels: &[[u16; 4]], format: Format) -> (usize, Vec<((u64, u64), [u16; 4], [u16; 4])>) {
    let all: Vec<_> = (texels.iter().enumerate())
        .map(|(i, &texel)| ((i as u64 % SIDE, i as u64 / SIDE), texel))
        .map(|((x, y), texel)| ((x, y), texel, expected(format, x, y)))
        .filter(|&(_, texel, want)| !matches(format, texel, want))
        .collect();
    (all.len(), all.into_iter().take(8).collect())
}

/// Write [`BACKGROUND`] into every texel of the target of `format` over `buffer`'s first bytes,
/// a host-visible buffer no command uses now.
fn fill_background(buffer: &Buffer, format: Format) {
    let contents = buffer.contents().unwrap();
    let background = stored(format, BACKGROUND);
    let texel: Vec<u8> = match format {
        Format::Half => background.iter().flat_map(|h| h.to_le_bytes()).collect(),
        Format::Srgb => background.iter().map(|&c| c as u8).collect(),
    };
    let bytes = texel.repeat((SIDE * SIDE) as usize);
    // SAFETY: the buffer holds at least the target's `SIDE * SIDE` texels, which no command
    // buffer uses now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

fn run(case: &str, home: Home, format: Format) {
    // The host pages outlive the buffer over them, made later and dropped first.
    let host = Host::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let scratch = Scratch::new(&format!("fetch-composite-{case}"));
    let device = device(&scratch.0);
    let fetch = match state(
        &device,
        ("engine/fragment_fetch.ll", "fetch_ordered"),
        format,
    ) {
        Ok(fetch) => fetch,
        Err(error) => {
            eprintln!(
                "FETCH-COMPOSITE {case} on {}: no framebuffer fetch: {error}",
                device.name()
            );
            return;
        }
    };
    let tools = Tools {
        queue: device.new_command_queue().unwrap(),
        sample: state(&device, ("engine/tob_renders.ll", "tob_sample"), format).unwrap(),
        fetch,
        sampler: (device.new_sampler_state(&SamplerDescriptor::new())).unwrap(),
        format,
    };
    let texel = format.texel();
    let length = SIDE * SIDE * texel;
    let mut d = TextureDescriptor::texture_2d(format.pixel_format(), SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let (target, backing) = match home {
        Home::Buffer => {
            d.resource_options = ResourceOptions::shared();
            let backing = device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap();
            let target = backing.new_texture(&d, 0, SIDE * texel).unwrap();
            (target, Some(backing))
        }
        Home::HostPages => {
            let Some(backing) = over_host(&device, &host, &runs) else {
                return;
            };
            // The target is made once the check's verdicts are known, so its pages' class has
            // them: an image in place, not a shadow waiting for them.
            device.settle_in_place_checks(Duration::from_secs(120));
            d.resource_options = backing.resource_options();
            let target = backing.new_texture(&d, 0, SIDE * texel).unwrap();
            (target, Some(backing))
        }
        Home::Private => {
            d.resource_options = ResourceOptions::private();
            (device.new_texture(&d).unwrap(), None)
        }
    };
    // Over host pages, the pass loads what the CPU wrote there before each round.
    let load = home == Home::HostPages;
    let mut failures = Vec::new();
    let fetches = through_storage(device.compile_device().profile());
    let in_place =
        home == Home::HostPages && target.shadow().is_none() && target.linear_image().unwrap();
    // In place over host pages where the device's check passed for a shader's writes of its
    // texels, on a device that fetches through storage, the image takes the storage usage, and
    // the pass fetches it in place (`texture/in_place_image.rs`); but an 8-bit sRGB one's, which
    // renders through a scratch.
    if home == Home::HostPages && format == Format::Half {
        let backing = backing.as_ref().unwrap();
        let layout = crate::linear_plan::BufferLayout {
            offset: 0,
            row_pitch: SIDE * texel,
            slice_pitch: length,
            row_bytes: SIDE * texel,
            rows: SIDE,
            slices: 1,
        };
        let start = backing.contents().map_or(0, |at| at.as_ptr().addr() as u64);
        let checks = &device.shared().in_place;
        let rows = RowEnds::of(start, &layout, checks.line());
        let writes = Access::written(texel, Padding::Kept)
            .is_some_and(|w| checks.verdict(PagesClass::Pointer, w, rows) == Verdict::Passed);
        let usage = target.image_plan().unwrap().usage;
        let storage = usage.contains(ImageUsageFlags::STORAGE);
        eprintln!(
            "FETCH-COMPOSITE {case} on {}: in place {in_place}, writes passed {writes}, fetched \
             through storage {fetches}, feedback loop usage {}, storage usage {storage}",
            device.name(),
            usage.contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT),
        );
        if in_place && writes && fetches && !storage {
            failures.push(
                "in place, the check passed for writes and the device fetches through storage, \
                 but the image took no storage usage: its pass fetches it through a scratch"
                    .to_owned(),
            );
        }
    }
    let back = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    // The icon's pass and the composite in one command buffer, then in a later one over a new
    // icon, the first released.
    let mut icon = None;
    for round in 0..2 {
        if load {
            fill_background(backing.as_ref().unwrap(), format);
        }
        let before = device.work_counts();
        let command_buffer = tools.queue.command_buffer().unwrap();
        let new = tools.icon(&command_buffer);
        if round == 1 {
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        let command_buffer = match round {
            0 => command_buffer,
            _ => tools.queue.command_buffer().unwrap(),
        };
        drop(icon.replace(new));
        tools.composite(&command_buffer, (&target, load), icon.as_ref().unwrap());
        tools.read(&command_buffer, &target, &back);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let after = device.work_counts();
        let (count, first) = wrong(&texels(&back, format), format);
        let merges = after.linear_merges_recorded - before.linear_merges_recorded;
        eprintln!(
            "FETCH-COMPOSITE {case} round {round} on {}: wrong pixels {count} of {}, instances \
             {}, interlocked-write splits {}, storage forms {}, inline passes {}, linear merges \
             {merges}; first wrong (pixel, got, want) {first:x?}",
            device.name(),
            SIDE * SIDE,
            after.render_passes_begun_instances - before.render_passes_begun_instances,
            after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
            after.storage_forms_taken - before.storage_forms_taken,
            after.inline_passes - before.inline_passes,
        );
        if count != 0 {
            failures.push(format!("round {round}: {count} pixels, first {first:x?}"));
        }
        // The 8-bit sRGB target in place over host pages, on a device that fetches through
        // storage, rendered through a scratch merged back into the pages: the route checked.
        if format == Format::Srgb && home == Home::HostPages && in_place && fetches && merges == 0 {
            failures.push(format!(
                "round {round}: no linear merge: not the scratch's route"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{case}: the composite's pixels: {failures:?}"
    );
}

#[test]
fn a_private_targets_composite_of_icons_and_fetches_keeps_every_draw() {
    run("private", Home::Private, Format::Half);
}

#[test]
fn a_composite_over_a_buffers_bytes_of_icons_and_fetches_keeps_every_draw() {
    run("over-buffer", Home::Buffer, Format::Half);
}

#[test]
fn a_loaded_composite_over_host_pages_fetches_them_in_place_and_keeps_every_draw() {
    run("over-host-pages", Home::HostPages, Format::Half);
}

#[test]
fn a_loaded_srgb_composite_over_host_pages_through_a_scratch_keeps_every_draw() {
    run("srgb-over-host-pages", Home::HostPages, Format::Srgb);
}
