//! Side effects after a discard on the device the cfg selects, under the validation layer: each
//! lands where the fragment was kept and nowhere else.
//!
//! The owned fixture `rch_guarded_effects` draws one triangle (`rch_guarded_vertex`, its corners
//! [`CORNERS`] in pixels, no pixel centre within [`MARGIN`] of an edge) over a 32x16
//! `RGBA32Float` target cleared to [`CLEAR`]: its edges cut quads, whose helper invocations run
//! beside the covered pixels. Each fragment samples an 8x8 `RGBA32Float` texture repeated over
//! the target at LOD 0, the texel at `(x mod 8, y mod 8)`: `(u / 8, v / 8, 0.25, a)`, with `a` 0
//! where `(u + 2v) mod 5` is 0 and 1 elsewhere, discards where `a` is below 0.5, and returns the
//! texel. A kept pixel is a covered one whose texel's `a` is 1. After the discard:
//!
//! - `rch_guarded_store` stores the texel times 0.5 at the pixel's index (`y * 64 + x`) in a
//!   buffer of [`SENTINEL`] words;
//! - `rch_guarded_atomic` adds 1 to a counter and stores the pixel's index at the old value's
//!   slot: the counter ends at the kept pixels' number, the slots below it hold their indices once
//!   each, in any order, and the rest stay [`SENTINEL`] (a discarded invocation's atomic, whose
//!   old value Metal does not define, is never performed);
//! - `rch_guarded_write` writes the texel times 0.25 at the pixel in a 32x16 `RGBA32Float`
//!   texture of [`SENTINEL`] words;
//! - `rch_guarded_copy` copies the pair of vectors at index `u / 2` of four ([`pair`]) to the
//!   pixel's index in a buffer of [`SENTINEL`] words.
//!
//! Metal's answer is that: a discarded fragment's stores, atomics, texture writes and copies do
//! not land (an M4), and a helper invocation's never do. Every value is a
//! dyadic fraction an `f32` holds exactly. A predicated discard (MoltenVK on an Apple GPU defers
//! each of these past the stores; a device without demote predicates them all) runs a discarded
//! fragment on to its end as a live invocation, so a side effect it did not guard would land as an
//! extra word; a store a simdgroup's driver dropped would be a lost one. The colour target keeps
//! the texel at a kept pixel and [`CLEAR`] elsewhere.

use super::resolves::{lane, quiet, Lane};
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};

/// The target's width and height.
pub(super) const W: u64 = 32;
pub(super) const H: u64 = 16;

/// A pixel's index is `y * ROW + x`.
pub(super) const ROW: u64 = 64;

/// The texture's side.
const TILE: u64 = 8;

/// The triangle's corners, in pixels.
pub(super) const CORNERS: [[f64; 2]; 3] = [[1.3, 0.81], [30.7, 3.2], [6.4, 15.09]];

/// The least distance, in pixels, of a pixel centre from an edge's line: none is near enough to
/// one for a rasterizer's rounding to decide it.
const MARGIN: f64 = 0.01;

/// The colour target's clear value.
const CLEAR: f32 = -1.0;

/// Every word of a buffer or texture the fragments write, before the draw.
pub(super) const SENTINEL: u32 = 0xDEAD_BEEF;

/// The texel at `(u, v)`.
fn texel(u: u64, v: u64) -> [f32; 4] {
    let a = if (u + 2 * v).is_multiple_of(5) {
        0.0
    } else {
        1.0
    };
    [u as f32 / 8.0, v as f32 / 8.0, 0.25, a]
}

/// The pair `rch_guarded_copy` copies from slot `k`.
fn pair(k: u64) -> [f32; 8] {
    let k = k as f32;
    [k, k + 0.5, k + 0.25, 1.0, -k, 2.0 * k, 0.125, 3.0]
}

/// The signed doubled area of `(a, b, p)`.
fn edge(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Whether the triangle covers the centre of pixel `(x, y)`.
pub(super) fn covered(x: u64, y: u64) -> bool {
    let p = [x as f64 + 0.5, y as f64 + 0.5];
    let [a, b, c] = CORNERS;
    let e = [edge(a, b, p), edge(b, c, p), edge(c, a, p)];
    for (i, (from, to)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
        let length = ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt();
        assert!(
            (e[i] / length).abs() > MARGIN,
            "pixel ({x}, {y}) is within {MARGIN} of an edge"
        );
    }
    e.iter().all(|v| *v > 0.0) || e.iter().all(|v| *v < 0.0)
}

/// The kept pixels, by `(x, y)`, row by row.
fn kept() -> Vec<(u64, u64)> {
    (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| covered(x, y) && texel(x % TILE, y % TILE)[3] == 1.0)
        .collect()
}

pub(super) fn words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

pub(super) fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub(super) fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// The fixture's pipeline of fragment function `fragment`.
fn pipeline(lane: &Lane, library: &Library, fragment: &str) -> RenderPipelineState {
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("rch_guarded_vertex")));
    descriptor.fragment = Some(StageSource::Function(function(fragment)));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared `RGBA32Float` texture of `(width, height)` and `usage`, holding `texels`.
pub(super) fn texture(
    lane: &Lane,
    (width, height): (u64, u64),
    usage: TextureUsage,
    texels: &[u32],
) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(
            Region::new_2d(0, 0, width, height),
            0,
            0,
            &bytes(texels),
            16 * width,
            0,
        )
        .unwrap();
    texture
}

/// What a draw left: the colour target's words, the fragment buffers' words, and the written
/// texture's.
struct Drawn {
    color: Vec<u32>,
    buffers: Vec<Vec<u32>>,
    written: Vec<u32>,
}

/// One draw of `fragment` with `buffers` (each its initial words) at `buffer(0)` onwards.
fn draw(lane: &Lane, library: &Library, fragment: &str, buffers: &[Vec<u32>]) -> Drawn {
    let pipeline = pipeline(lane, library, fragment);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = lane.device.new_texture(&descriptor).unwrap();
    let sampled: Vec<u32> = (0..TILE)
        .flat_map(|v| (0..TILE).flat_map(move |u| words(&texel(u, v))))
        .collect();
    let sampled = texture(lane, (TILE, TILE), TextureUsage::SHADER_READ, &sampled);
    let written = texture(
        lane,
        (W, H),
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        &vec![SENTINEL; (4 * W * H) as usize],
    );
    let corners: Vec<f32> = CORNERS
        .iter()
        .flat_map(|[x, y]| {
            [
                (x / W as f64 * 2.0 - 1.0) as f32,
                (1.0 - y / H as f64 * 2.0) as f32,
            ]
        })
        .collect();
    let corners = buffer_with(&lane.device, &bytes(&words(&corners))).unwrap();
    let held: Vec<Buffer> = buffers
        .iter()
        .map(|w| buffer_with(&lane.device, &bytes(w)).unwrap())
        .collect();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    let clear = f64::from(CLEAR);
    attachment.clear_color = ClearColor::new(clear, clear, clear, clear);
    attachment.store_action = StoreAction::Store;
    lane.run(&descriptor, |_, encoder, _| {
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&sampled), 0).unwrap();
        encoder.set_fragment_texture(Some(&written), 1).unwrap();
        for (index, buffer) in held.iter().enumerate() {
            encoder
                .set_fragment_buffer(Some(buffer), 0, index as u32)
                .unwrap();
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    Drawn {
        color: unpack(&lane.read(&target, (0, 0), 16, BlitOption::None)),
        buffers: held
            .iter()
            .zip(buffers)
            .map(|(b, w)| unpack(&read_buffer(b, 0, 4 * w.len() as u64).unwrap()))
            .collect(),
        written: unpack(&lane.read(&written, (0, 0), 16, BlitOption::None)),
    }
}

/// The words lost (a kept pixel's that kept `before`) and extra (another's that did not), as
/// `(lost, extra, first difference)`, of `got` against `want` by pixel, `n` words a pixel at
/// `stride` pixels a row.
fn compare(
    got: &[u32],
    want: impl Fn(u64, u64) -> Option<Vec<u32>>,
    before: &[u32],
    (n, stride): (usize, u64),
) -> (usize, usize, Option<String>) {
    let (mut lost, mut extra, mut first) = (0, 0, None);
    for y in 0..H {
        for x in 0..W {
            let at = ((y * stride + x) as usize) * n;
            let here = &got[at..at + n];
            let (expected, kept) = match want(x, y) {
                Some(w) => (w, true),
                None => (before.to_vec(), false),
            };
            if here != expected.as_slice() {
                if kept {
                    lost += 1;
                } else {
                    extra += 1;
                }
                first.get_or_insert_with(|| {
                    format!("({x}, {y}) kept {kept}: {here:x?}, expected {expected:x?}")
                });
            }
        }
    }
    (lost, extra, first)
}

/// Each side effect after the discard lands at the kept pixels, and nowhere else, with the
/// colour the kept pixels return; the atomic counts the kept pixels once each.
#[test]
fn side_effects_after_a_discard_land_at_the_kept_pixels_alone() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let ll = metal2vulkan_devtools::engine_dir().join("rch_guarded_effects.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let kept = kept();
    let is_kept = |x: u64, y: u64| kept.contains(&(x, y));
    let tex = |x: u64, y: u64| texel(x % TILE, y % TILE);
    let scaled = |x: u64, y: u64, s: f32| words(&tex(x, y).map(|v| v * s));
    let covered_count = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| covered(x, y))
        .count();
    eprintln!(
        "guarded effects on {name}: {} kept of {covered_count} covered pixels",
        kept.len()
    );
    assert!(kept.len() > 16 && kept.len() + 16 < covered_count);
    let slots = (ROW * H) as usize;
    let mut failures = Vec::new();
    let mut check = |what: &str, (lost, extra, first): (usize, usize, Option<String>)| {
        eprintln!("guarded effects {what} on {name}: {lost} lost, {extra} extra {first:?}");
        if lost + extra > 0 {
            failures.push(format!(
                "{what}: {lost} lost, {extra} extra, first {first:?}"
            ));
        }
    };
    let color = |d: &Drawn| {
        compare(
            &d.color,
            |x, y| is_kept(x, y).then(|| words(&tex(x, y))),
            &[CLEAR.to_bits(); 4],
            (4, W),
        )
    };

    let d = draw(
        &lane,
        &library,
        "rch_guarded_store",
        &[vec![SENTINEL; 4 * slots]],
    );
    check("store: colour", color(&d));
    check(
        "store: buffer",
        compare(
            &d.buffers[0],
            |x, y| is_kept(x, y).then(|| scaled(x, y, 0.5)),
            &[SENTINEL; 4],
            (4, ROW),
        ),
    );

    let d = draw(
        &lane,
        &library,
        "rch_guarded_atomic",
        &[vec![0], vec![SENTINEL; slots]],
    );
    check("atomic: colour", color(&d));
    let count = d.buffers[0][0] as usize;
    let out = &d.buffers[1];
    let mut indices: Vec<u32> = out[..count.min(slots)].to_vec();
    indices.sort_unstable();
    let mut want: Vec<u32> = kept.iter().map(|&(x, y)| (y * ROW + x) as u32).collect();
    want.sort_unstable();
    let extra_slots = out[count.min(slots)..]
        .iter()
        .filter(|w| **w != SENTINEL)
        .count();
    let lost = want.iter().filter(|w| !indices.contains(w)).count();
    let extra = count.saturating_sub(want.len()) + extra_slots;
    check(
        "atomic: counter and slots",
        (
            lost,
            extra,
            (indices != want || extra_slots > 0)
                .then(|| format!("counter {count}, {} kept", want.len())),
        ),
    );

    let d = draw(&lane, &library, "rch_guarded_write", &[]);
    check("write: colour", color(&d));
    check(
        "write: texture",
        compare(
            &d.written,
            |x, y| is_kept(x, y).then(|| scaled(x, y, 0.25)),
            &[SENTINEL; 4],
            (4, W),
        ),
    );

    let pairs: Vec<u32> = (0..4).flat_map(|k| words(&pair(k))).collect();
    let d = draw(
        &lane,
        &library,
        "rch_guarded_copy",
        &[vec![SENTINEL; 8 * slots], pairs],
    );
    check("copy: colour", color(&d));
    check(
        "copy: buffer",
        compare(
            &d.buffers[0],
            |x, y| is_kept(x, y).then(|| words(&pair((x % TILE) / 2))),
            &[SENTINEL; 8],
            (8, ROW),
        ),
    );
    assert!(
        failures.is_empty(),
        "on {name}:\n  {}",
        failures.join("\n  ")
    );
    quiet(lane, &messages);
}
