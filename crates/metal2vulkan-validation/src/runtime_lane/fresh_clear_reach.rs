//! Device tests of a `Clear` past the render area of a never-written `Private` color target,
//! by Metal's compression (the runtime's `clear_reach.rs`), on the device the cfg selects under
//! the validation layer, presenting an Apple silicon Mac (`DeviceOptions::mac_class`), whose
//! textures Metal's rule compresses.
//!
//! **Metal**, our probes on the M3 Max (macOS 27.0.1): a new `Private` render target
//! (`RenderTarget | ShaderRead`) of `BGRA8Unorm`, `RGBA16Float` or `RG16Float`, at each size
//! below, with `allowGPUOptimizedContents` on (compressed: every size's shorter side is 16 or
//! more) and off (uncompressed), compared texel by texel with an untouched twin of the same
//! descriptor:
//!
//! - **A `Clear` over a smaller render area** (`renderTargetWidth` and `Height`) without a draw
//!   call clears the whole texture where it is compressed, and the render area rounded up to
//!   32x32 tiles where it is not; with a draw call (here one over the whole target), the tiles
//!   either way, the draw's texels the render area's alone.
//! - **A `Clear` with only a scissor** (the render area the whole texture) clears the whole
//!   texture, the draw's texels the scissor's.
//! - **A `Load`** with a draw over a smaller render area, or within a scissor, changes the render
//!   area's (or the scissor's) texels alone.
//!
//! Every texel past what the pass changes reads what the twin reads (Metal's fresh pattern, or
//! zero, or the device's zero where it cannot write the pattern). A `Load` pass records no clear
//! of its own beside its pass (the cost check).

use super::clear_reach::{lane_with, quiet, Lane};
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, MacClass, Origin, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The clear color: exact in every format here, no half step of a unorm channel.
const GREEN: [f64; 4] = [0.0, 1.0, 0.0, 1.0];

/// A format: its texel's bytes cleared to [`GREEN`] and drawn by the fixture's fragment
/// function, `(1/255, 0, 0, 1)`.
struct Format {
    pf: PixelFormat,
    cleared: &'static [u8],
    drawn: &'static [u8],
}

const FORMATS: [Format; 3] = [
    Format {
        pf: PixelFormat::BGRA8Unorm,
        cleared: &[0x00, 0xff, 0x00, 0xff],
        drawn: &[0x00, 0x00, 0x01, 0xff],
    },
    Format {
        pf: PixelFormat::RGBA16Float,
        cleared: &[0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c],
        drawn: &[0x04, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c],
    },
    Format {
        pf: PixelFormat::RG16Float,
        cleared: &[0x00, 0x00, 0x00, 0x3c],
        drawn: &[0x04, 0x1c, 0x00, 0x00],
    },
];

/// The probes' sizes: the texture's, and the render area's (or the scissor's).
const SHAPES: [([u64; 2], [u64; 2]); 5] = [
    ([224, 32], [210, 26]),
    ([1984, 64], [1931, 40]),
    ([100, 100], [37, 21]),
    ([128, 128], [64, 64]),
    ([256, 256], [17, 100]),
];

/// A probe's pass.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// `Clear` and store over the smaller render area, without a draw call.
    ClearArea,
    /// The same with a draw over the whole target.
    ClearAreaDrawn,
    /// `Load` and store over the smaller render area, with that draw.
    LoadArea,
    /// `Load` and store, the render area the whole texture, the draw within the scissor.
    LoadScissor,
    /// `Clear` and store, the render area the whole texture, the draw within the scissor.
    ClearScissor,
}

const MODES: [Mode; 5] = [
    Mode::ClearArea,
    Mode::ClearAreaDrawn,
    Mode::LoadArea,
    Mode::LoadScissor,
    Mode::ClearScissor,
];

/// What Metal changes in a `size` texture through `mode` over `area`, compressed or not: the
/// extent from the origin its draw writes, and the one its clear writes (the module docs).
fn metal_s(mode: Mode, size: [u64; 2], area: [u64; 2], compressed: bool) -> [[u64; 2]; 2] {
    let tiles = [0, 1].map(|axis| area[axis].div_ceil(32).saturating_mul(32).min(size[axis]));
    match mode {
        Mode::ClearArea if compressed => [[0, 0], size],
        Mode::ClearArea => [[0, 0], tiles],
        Mode::ClearAreaDrawn => [area, tiles],
        Mode::LoadArea | Mode::LoadScissor => [area, [0, 0]],
        Mode::ClearScissor => [area, size],
    }
}

/// A new `Private` render target of `pf` and `size`, its GPU optimization `optimized`.
fn target(lane: &Lane, pf: PixelFormat, size: [u64; 2], optimized: bool) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, size[0], size[1], false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor.allow_gpu_optimized_contents = optimized;
    lane.device.new_texture(&descriptor).unwrap()
}

/// `texture`'s texels, `bytes` each, through a blit into a shared buffer.
fn read(lane: &Lane, texture: &Texture, bytes: u64) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let length = bytes * width * height;
    let buffer = buffer_with(&lane.device, &vec![0; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &buffer,
        0,
        bytes * width,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&buffer, 0, length).unwrap()
}

/// Encode `mode`'s pass over `texture` and `area` (a draw through `pipeline` of the triangle over
/// the whole target where it has one), then commit and wait.
fn encode(
    lane: &Lane,
    texture: &Texture,
    mode: Mode,
    area: [u64; 2],
    pipeline: &RenderPipelineState,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = match mode {
        Mode::LoadArea | Mode::LoadScissor => LoadAction::Load,
        _ => LoadAction::Clear,
    };
    attachment.clear_color = ClearColor::new(GREEN[0], GREEN[1], GREEN[2], GREEN[3]);
    attachment.store_action = StoreAction::Store;
    if matches!(
        mode,
        Mode::ClearArea | Mode::ClearAreaDrawn | Mode::LoadArea
    ) {
        descriptor.render_target_width = area[0];
        descriptor.render_target_height = area[1];
    }
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let mut held = Vec::new();
    if mode != Mode::ClearArea {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        if matches!(mode, Mode::LoadScissor | Mode::ClearScissor) {
            let scissor = ScissorRect::new(0, 0, area[0], area[1]);
            encoder.set_scissor_rect(scissor).unwrap();
        }
        let corners: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let bytes: Vec<u8> = corners.iter().flat_map(|v| v.to_le_bytes()).collect();
        let corners = buffer_with(&lane.device, &bytes).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.push(corners);
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
}

/// How `observed` differs from `expected` (texels of `bytes` in rows of `width`) within the
/// extent `area`, within the 32x32 tiles it reaches, and past them, or `None` where it does not.
fn differences(
    observed: &[u8],
    expected: &[u8],
    (bytes, width): (usize, u64),
    area: [u64; 2],
) -> Option<String> {
    let tiles = area.map(|side| side.div_ceil(32) * 32);
    let mut counts = [0u64; 3];
    let mut first = None;
    for (at, (seen, want)) in observed
        .chunks(bytes)
        .zip(expected.chunks(bytes))
        .enumerate()
    {
        if seen == want {
            continue;
        }
        let (x, y) = (at as u64 % width, at as u64 / width);
        let region = match (x < area[0] && y < area[1], x < tiles[0] && y < tiles[1]) {
            (true, _) => 0,
            (false, true) => 1,
            (false, false) => 2,
        };
        counts[region] += 1;
        first.get_or_insert(format!("({x}, {y}) {seen:02x?} where {want:02x?}"));
    }
    first.map(|first| {
        format!(
            "{} texels differ inside the area, {} in the tiles past it, {} beyond; the first {first}",
            counts[0], counts[1], counts[2]
        )
    })
}

/// A `Clear` past the render area of a never-written `Private` target reaches what Metal's
/// reaches, by its compression: the whole texture compressed and the 32x32 tiles uncompressed
/// without a draw call, the tiles with one, the whole texture with only a scissor; a `Load`
/// changes the render area's or the scissor's texels alone, and records no clear but its pass;
/// a draw writes Metal's area alone. Every texel past those reads what an untouched twin reads.
#[test]
fn a_fresh_target_s_clear_reaches_as_metal_s_by_its_compression() {
    let (messages, _cache, lane) =
        lane_with(|options| options.mac_class = Some(MacClass::AppleSilicon));
    assert!(lane.device.properties().lossless_compression);
    let mut failures = Vec::new();
    let mut cases = 0;
    for format in &FORMATS {
        let bytes = format.cleared.len();
        assert_ne!(format.cleared, format.drawn);
        let pipeline = lane.pipeline(&[format.pf], (None, None), 1);
        for (size, area) in SHAPES {
            for optimized in [true, false] {
                for mode in MODES {
                    let texture = target(&lane, format.pf, size, optimized);
                    let twin = target(&lane, format.pf, size, optimized);
                    let before = lane.device.work_counts().commands_recorded;
                    encode(&lane, &texture, mode, area, &pipeline);
                    let after = lane.device.work_counts().commands_recorded;
                    let observed = read(&lane, &texture, bytes as u64);
                    let untouched = read(&lane, &twin, bytes as u64);
                    // The pass's own texels differ from the twin's, so the case shows where they
                    // land (a device whose fresh texels read zero reads neither).
                    assert!(untouched
                        .chunks(bytes)
                        .all(|texel| texel != format.cleared && texel != format.drawn));
                    let [drawn, cleared] = metal_s(mode, size, area, optimized);
                    let mut expected = untouched.clone();
                    for (at, texel) in expected.chunks_mut(bytes).enumerate() {
                        let (x, y) = (at as u64 % size[0], at as u64 / size[0]);
                        if x < drawn[0] && y < drawn[1] {
                            texel.copy_from_slice(format.drawn);
                        } else if x < cleared[0] && y < cleared[1] {
                            texel.copy_from_slice(format.cleared);
                        }
                    }
                    let case = format!(
                        "{:?} {}x{} {:?} {}x{} optimized {optimized}",
                        format.pf, size[0], size[1], mode, area[0], area[1]
                    );
                    cases += 1;
                    if let Some(why) = differences(&observed, &expected, (bytes, size[0]), area) {
                        failures.push(format!("{case}: {why}"));
                    }
                    let passes = after.cmd_begin_render_pass - before.cmd_begin_render_pass;
                    let clears = after.cmd_clear_attachments - before.cmd_clear_attachments;
                    if matches!(mode, Mode::LoadArea | Mode::LoadScissor)
                        && (passes, clears) != (1, 0)
                    {
                        failures.push(format!(
                            "{case}: {passes} render passes and {clears} attachment clears recorded"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} cases differ from Metal's:\n{}",
        failures.len(),
        failures.join("\n")
    );
    quiet(lane, &messages);
}
