//! A composite pass that both samples icon textures and fetches its target, as a window server's
//! compositor draws a dock: icons sampled into the target, and glass drawn over them reading the
//! target's own pixels (framebuffer fetch in a raster order group). Every icon draw's pixels hold
//! the icon's color, and every fetch halves what was under it, in draw order, whether the pass
//! renders the target in place, splits into instances, or renders it through a scratch image
//! (`attachment_stand_in.rs`).
//!
//! An `RGBA16Float` icon is filled by a pass clearing it to [`ICON`]; the composite pass clears
//! its `RGBA16Float` target to [`BACKGROUND`], then draws the icon at [`A`], fetches [`F`], draws
//! the icon at [`A2`] and fetches [`F2`], which covers half of [`A`]. The target is a `Private`
//! render target, or a texture over a `Shared` buffer's bytes (a linear image, as a surface's
//! back buffer is). Each target is composited twice: in the icon's command buffer, then in a later
//! one with a new icon, the first released.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device without a framebuffer fetch in
//! the pixel interlock says so and checks nothing.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The target's side, in pixels.
const SIDE: u64 = 128;

/// The icon's side, in texels.
const ICON_SIDE: u64 = 64;

/// An `RGBA16Float` texel's bytes.
const TEXEL: u64 = 8;

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

/// The state of `tob_rect` and `fixture`'s `fragment`, into an `RGBA16Float` color.
fn state(
    device: &Device,
    fixture: &str,
    fragment: &str,
) -> Result<RenderPipelineState, crate::Error> {
    let function = |path: &str, name: &str| {
        let function = library(device, path).new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(function("engine/tob_renders.ll", "tob_rect"));
    descriptor.fragment = Some(function(fixture, fragment));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
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
        v if v == 1.0 => 0x3c00,
        v if v == 0.75 => 0x3a00,
        v if v == 0.5 => 0x3800,
        v if v == 0.375 => 0x3600,
        v if v == 0.25 => 0x3400,
        v if v == 0.125 => 0x3000,
        v => panic!("no half for {v}"),
    }
}

/// The color the composite leaves at pixel `(x, y)`.
fn expected(x: u64, y: u64) -> [u16; 4] {
    let within = |[x0, y0, x1, y1]: [u64; 4]| x >= x0 && x < x1 && y >= y0 && y < y1;
    let mut color = BACKGROUND;
    for draw in DRAWS {
        match draw {
            Draw::Icon(rect) if within(rect) => color = ICON,
            Draw::Fetch(rect) if within(rect) => color = color.map(|c| c * 0.5),
            _ => {}
        }
    }
    color.map(half)
}

/// The tools: the queue, the two states, and the icon's sampler.
struct Tools {
    queue: CommandQueue,
    sample: RenderPipelineState,
    fetch: RenderPipelineState,
    sampler: SamplerState,
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

    /// The composite pass over `target`, sampling `icon`, encoded in `command_buffer`.
    fn composite(&self, command_buffer: &CommandBuffer, target: &Texture, icon: &Texture) {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
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
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            back,
            0,
            SIDE * TEXEL,
            SIDE * SIDE * TEXEL,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
}

/// `buffer`'s texels, a host-visible buffer no command writes now.
fn texels(buffer: &Buffer) -> Vec<[u16; 4]> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's first `SIDE * SIDE * TEXEL` bytes, which no command buffer writes now.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), (SIDE * SIDE * TEXEL) as usize) };
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|t| {
            let h = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
            [h(0), h(2), h(4), h(6)]
        })
        .collect()
}

/// The pixels of `texels` that differ from the composite's answer, with their texels, at most
/// the first 8, and how many differ.
fn wrong(texels: &[[u16; 4]]) -> (usize, Vec<((u64, u64), [u16; 4], [u16; 4])>) {
    let all: Vec<_> = (texels.iter().enumerate())
        .map(|(i, &texel)| ((i as u64 % SIDE, i as u64 / SIDE), texel))
        .filter(|&((x, y), texel)| texel != expected(x, y))
        .map(|((x, y), texel)| ((x, y), texel, expected(x, y)))
        .collect();
    (all.len(), all.into_iter().take(8).collect())
}

fn run(case: &str, over_buffer: bool) {
    let scratch = Scratch::new(&format!("fetch-composite-{case}"));
    let device = device(&scratch.0);
    let fetch = match state(&device, "engine/fragment_fetch.ll", "fetch_ordered") {
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
        sample: state(&device, "engine/tob_renders.ll", "tob_sample").unwrap(),
        fetch,
        sampler: (device.new_sampler_state(&SamplerDescriptor::new())).unwrap(),
    };
    let length = SIDE * SIDE * TEXEL;
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let (target, _backing) = match over_buffer {
        true => {
            d.resource_options = ResourceOptions::shared();
            let backing = device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap();
            let target = backing.new_texture(&d, 0, SIDE * TEXEL).unwrap();
            (target, Some(backing))
        }
        false => {
            d.resource_options = ResourceOptions::private();
            (device.new_texture(&d).unwrap(), None)
        }
    };
    let back = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let mut failures = Vec::new();
    // The icon's pass and the composite in one command buffer, then in a later one over a new
    // icon, the first released.
    let mut icon = None;
    for round in 0..2 {
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
        tools.composite(&command_buffer, &target, icon.as_ref().unwrap());
        tools.read(&command_buffer, &target, &back);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let after = device.work_counts();
        let (count, first) = wrong(&texels(&back));
        eprintln!(
            "FETCH-COMPOSITE {case} round {round} on {}: wrong pixels {count} of {}, instances \
             {}, interlocked-write splits {}, storage forms {}; first wrong (pixel, got, want) \
             {first:x?}",
            device.name(),
            SIDE * SIDE,
            after.render_passes_begun_instances - before.render_passes_begun_instances,
            after.instance_splits_interlocked_writes - before.instance_splits_interlocked_writes,
            after.storage_forms_taken - before.storage_forms_taken,
        );
        if count != 0 {
            failures.push(format!("round {round}: {count} pixels, first {first:x?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{case}: the composite's pixels: {failures:?}"
    );
}

#[test]
fn a_private_targets_composite_of_icons_and_fetches_keeps_every_draw() {
    run("private", false);
}

#[test]
fn a_composite_over_a_buffers_bytes_of_icons_and_fetches_keeps_every_draw() {
    run("over-buffer", true);
}
