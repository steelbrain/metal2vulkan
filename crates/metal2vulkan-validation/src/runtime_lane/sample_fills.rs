//! Metal's private texture fills of multisample textures (`BlitCommandEncoder::fill_texture` and
//! `fill_texture_color`), on the device the cfg selects, under the validation layer, against
//! Metal's samples.
//!
//! Each case replays one of our probe `ms.m`'s (an Apple M3 Max under macOS 27.0.1): a private
//! multisample texture of 5 x 3 texels (a 2-slice array for one), every slice cleared by a pass
//! (color `(0.25, 0.5, 0.75, 1)`, depth 0.25, stencil `0x5a`), one fill, and every sample read
//! back by a dispatch: a color format's bits through an unsigned view of its components
//! (`candidate::runtime_samples`), a depth's bits, a stencil's value (`Depth32Float_Stencil8`'s
//! through its `X32_Stencil8` view). Metal's answer per case is its read-back lines,
//! `after  s<slice> y<row>:` and each texel's samples as the probe printed them, committed in
//! [`METAL_LINES`] beside the first 16 hex digits of their SHA-256, which the probe printed. A
//! differing case names its first differing line. A sample whose bytes `n · b` of a texel's are 16 or fewer takes its own bytes
//! of the pattern, past 16 every sample the pattern cut at the texel (`tests::blit_fill` checks
//! the rule on every measured case without a device). The depths outside `[0, 1]` (`d32-wide-4`)
//! are not compared where the device's depth copies go by a draw that announced the values it
//! changes (`drawn-depth-copy`), whose samples it prints on a `SAMPLE-FILLS-DRAWN` line; a
//! differing case prints a `SAMPLE-FILLS-DIFFERS` line.

use super::degrade_notices::loses_depth_denormals;
use super::resolves::{lane, quiet, Lane};
use super::WAIT;
use crate::candidate::runtime_samples::encode_through;
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use metal2vulkan::error::Needs;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Degradation, Error, LoadAction, Region, RenderPassDescriptor,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};

/// Metal's lines per case, each case's block headed `[<case>]` (the `d32-wide-4` case apart,
/// whose depths out of `[0, 1]` a drawn depth copy changes on the devices that copy depth by a
/// draw, so only its digest is Metal's).
const METAL_LINES: &str = include_str!("../../fixtures/public/sample_fills/metal.txt");

/// The SHA-256 of [`METAL_LINES`].
const METAL_LINES_SHA: &str = "ca208ca294348632d67305825a0fe256d368d466d1634707b821d989e55ead00";

/// The probe's 16-byte pattern.
const P16: &str = "00112233445566778899aabbccddeeff";

/// The probe's 32-byte pattern.
const P32: &str = "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f";

/// Depths in `[0, 1]`, one a sample: a denormal, 1, 0 and 0.25.
const DEPTHS: &str = "010000000000803f000000000000803e";

/// What a case fills: bytes, or a color packed in the texture's format.
#[derive(Clone, Copy, Debug)]
enum Fill {
    Bytes(&'static str),
    Color([f64; 4]),
}

/// One case: its name, the format's raw value, samples, the array's slices (0 for a 2D
/// multisample texture), the fill's slice, its region's origin and size, what it fills, and
/// Metal's answer.
type Case = (
    &'static str,
    u32,
    u64,
    u64,
    u64,
    [u64; 3],
    [u64; 3],
    Fill,
    &'static str,
);

/// The cases, as the probe ran them.
const CASES: [Case; 17] = [
    (
        "r8-4",
        10,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "6a3cb9659495a014",
    ),
    (
        "rg8-4",
        30,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "7ae02f8d059eb534",
    ),
    (
        "rgba8-4",
        70,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "d2933cdc42077ec8",
    ),
    (
        "rgba8-2",
        70,
        2,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P32),
        "030f2a7c3939ede7",
    ),
    (
        "bgra8srgb-4",
        81,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "fb44c4e915f05b98",
    ),
    (
        "rgba16f-4",
        115,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "5d2db7b16bf04af2",
    ),
    (
        "rgba16f-2",
        115,
        2,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P32),
        "a729da908613bd7e",
    ),
    (
        "rgba32f-2",
        125,
        2,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P32),
        "93c2217f450a212d",
    ),
    (
        "r32f-4",
        55,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "9c5b7be066d1c9fc",
    ),
    (
        "d32-4",
        252,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(DEPTHS),
        "2098b9445cd43b96",
    ),
    (
        "d32-2",
        252,
        2,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(DEPTHS),
        "869ba839ce907836",
    ),
    (
        "d32-wide-4",
        252,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "9c5b7be066d1c9fc",
    ),
    (
        "s8-4",
        253,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(P16),
        "7f346b34bb6eabfb",
    ),
    (
        "d32s8-4",
        260,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Bytes(DEPTHS),
        "51cb2e517b1b1dd0",
    ),
    (
        "array-4",
        70,
        4,
        2,
        1,
        [1, 0, 0],
        [3, 2, 1],
        Fill::Bytes(P16),
        "a626fded296f9cfd",
    ),
    (
        "color-4",
        70,
        4,
        0,
        0,
        [1, 1, 0],
        [2, 1, 1],
        Fill::Color([0.5, 0.25, 1.0, 0.75]),
        "ecfad086c3dc487a",
    ),
    (
        "whole-2",
        70,
        2,
        0,
        3,
        [0, 0, 1],
        [9, 9, 2],
        Fill::Bytes("a1b2c3d4"),
        "98c79dda44dbd307",
    ),
];

/// The texture's side.
const WIDTH: u32 = 5;
const HEIGHT: u32 = 3;

/// How a texture of `pf` is read back, as the probe printed it: the kernel, its view's format,
/// what the readback reads, and the bytes of each printed value (a depth or stencil aspect's
/// word, else the texel's).
fn reading(pf: PixelFormat) -> (&'static str, Option<PixelFormat>, SampleRead, usize) {
    use PixelFormat as P;
    use ReadbackView as V;
    match pf {
        P::R8Unorm => (
            "rt6_read_words1",
            Some(P::R8Uint),
            SampleRead::Color(V::R8Uint),
            1,
        ),
        P::RG8Unorm => (
            "rt6_read_bytes4",
            Some(P::RG8Uint),
            SampleRead::Color(V::R16Uint),
            2,
        ),
        P::RGBA8Unorm | P::BGRA8UnormSRGB => (
            "rt6_read_bytes4",
            Some(P::RGBA8Uint),
            SampleRead::Color(V::R32Uint),
            4,
        ),
        P::RGBA16Float => (
            "rt6_read_halves4",
            Some(P::RGBA16Uint),
            SampleRead::Color(V::Rg32Uint),
            8,
        ),
        P::RGBA32Float => (
            "rt6_read_words4",
            Some(P::RGBA32Uint),
            SampleRead::Color(V::Rgba32Uint),
            16,
        ),
        P::R32Float => (
            "rt6_read_words1",
            Some(P::R32Uint),
            SampleRead::Color(V::R32Uint),
            4,
        ),
        P::Stencil8 => ("rt6_read_stencil", None, SampleRead::Stencil, 1),
        _ => ("rt6_read_depth", None, SampleRead::Depth, 4),
    }
}

/// One value as the probe printed it: a texel of 1 or 2 bytes as one hex number, else its 32-bit
/// words in order, each `%08x`.
fn printed(bytes: &[u8]) -> String {
    match bytes.len() {
        1 => format!("{:02x}", bytes[0]),
        2 => format!("{:04x}", u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => bytes
            .chunks(4)
            .map(|word| format!("{:08x}", u32::from_le_bytes(word.try_into().unwrap())))
            .collect(),
    }
}

/// The probe's lines of every sample `bytes` holds (the readback's layout: texels in rows, each
/// texel's samples, `unit` bytes each) of slice `slice`, labelled `label`.
fn lines(label: &str, slice: u64, bytes: &[u8], samples: usize, unit: usize) -> Vec<String> {
    // The lines print every sample of every texel: a readback longer than that would hold bytes
    // the digest never saw.
    assert_eq!(
        bytes.len(),
        HEIGHT as usize * WIDTH as usize * samples * unit,
        "{label}s{slice}: the printed samples do not cover every byte read back"
    );
    (0..HEIGHT as usize)
        .map(|y| {
            let mut line = format!("{label} s{slice} y{y}:");
            for x in 0..WIDTH as usize {
                line.push_str(" |");
                for s in 0..samples {
                    let at = ((y * WIDTH as usize + x) * samples + s) * unit;
                    line.push(' ');
                    line.push_str(&printed(&bytes[at..at + unit]));
                }
            }
            line
        })
        .collect()
}

/// Bytes from hex.
fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// The first 16 hex digits of the SHA-256 of `lines`, joined by newlines.
fn digest(lines: &[String]) -> String {
    let sum = Sha256::digest(lines.join("\n").as_bytes());
    sum.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// A 2D multisample view of `texture`'s `slice` in `pf`, or `texture` itself.
fn slice_of(texture: &Texture, pf: PixelFormat, slice: u64, array: bool) -> Texture {
    if !array && pf == texture.pixel_format() {
        return texture.clone();
    }
    let kind = if array {
        TextureType::Type2DMultisample
    } else {
        texture.texture_type()
    };
    texture
        .new_texture_view_with_levels_slices(pf, kind, 0..1, slice..slice + 1)
        .unwrap()
}

/// `error`, refusing `what` of `case`, printed as a `SAMPLE-FILLS-SKIPPED` line where it is the
/// device's own refusal: a typed `Unsupported` naming a capability the device lacks (a sample
/// count it does not render, 2 on lavapipe, which Vulkan leaves optional). Any other error fails.
fn refused_by_device(case: &str, what: &str, error: Error) {
    match error {
        Error::Unsupported(u) if u.needs != Needs::NotImplemented => {
            println!("SAMPLE-FILLS-SKIPPED {case}: {what} refused by the device: {u}");
        }
        error => panic!("{case}: {what}: {error}"),
    }
}

/// Run `case` on `lane`: the probe's lines of every slice's samples after the fill, or `None`
/// where the device refuses the texture or the fill for a capability it lacks (printed).
fn run(lane: &Lane, case: &Case) -> Option<Vec<String>> {
    let &(name, raw, samples, slices, slice, origin, size, fill, _) = case;
    let pf = PixelFormat::try_from(raw).unwrap();
    let device = &lane.device;
    let array = slices > 0;
    let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH.into(), HEIGHT.into(), false);
    descriptor.texture_type = if array {
        TextureType::Type2DMultisampleArray
    } else {
        TextureType::Type2DMultisample
    };
    descriptor.array_length = slices.max(1);
    descriptor.sample_count = samples;
    descriptor.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::private();
    let texture = match device.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(error) => {
            refused_by_device(name, "the texture", error);
            return None;
        }
    };
    let depth = matches!(
        pf,
        PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8
    );
    let stencil = matches!(
        pf,
        PixelFormat::Stencil8 | PixelFormat::Depth32FloatStencil8
    );
    let command_buffer = lane.queue.command_buffer().unwrap();
    for layer in 0..slices.max(1) {
        let mut pass = RenderPassDescriptor::new();
        if depth {
            pass.depth_attachment.texture = Some(texture.clone());
            pass.depth_attachment.slice = layer;
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 0.25;
            pass.depth_attachment.store_action = StoreAction::Store;
        }
        if stencil {
            pass.stencil_attachment.texture = Some(texture.clone());
            pass.stencil_attachment.slice = layer;
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.clear_stencil = 0x5a;
            pass.stencil_attachment.store_action = StoreAction::Store;
        }
        if !depth && !stencil {
            let color = &mut pass.color_attachments[0];
            color.texture = Some(texture.clone());
            color.slice = layer;
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
            color.store_action = StoreAction::Store;
        }
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let region = Region::new_3d(origin[0], origin[1], origin[2], size[0], size[1], size[2]);
    let filled = match fill {
        Fill::Bytes(pattern) => blit.fill_texture(&texture, 0, slice, region, &hex(pattern)),
        Fill::Color([r, g, b, a]) => blit.fill_texture_color(
            &texture,
            0,
            slice,
            region,
            ClearColor::new(r, g, b, a),
            None,
        ),
    };
    blit.end_encoding().unwrap();
    if let Err(error) = filled {
        refused_by_device(name, "the fill", error);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        return None;
    }
    let (kernel, view, read, unit) = reading(pf);
    let extent = [WIDTH, HEIGHT];
    let read_slice = |pf: PixelFormat, kernel, read, layer| {
        let source = slice_of(&texture, pf, layer, array);
        encode_through(
            device,
            &command_buffer,
            &source,
            kernel,
            read,
            extent,
            samples as u32,
        )
        .unwrap()
    };
    let readbacks: Vec<_> = (0..slices.max(1))
        .map(|layer| {
            let color = read_slice(view.unwrap_or(pf), kernel, read, layer);
            let stencil = (pf == PixelFormat::Depth32FloatStencil8).then(|| {
                read_slice(
                    PixelFormat::X32Stencil8,
                    "rt6_read_stencil",
                    SampleRead::Stencil,
                    layer,
                )
            });
            (color, stencil)
        })
        .collect();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{name}"
    );
    let mut out = Vec::new();
    let mut stencils = Vec::new();
    for (layer, (color, stencil)) in (0..).zip(&readbacks) {
        out.extend(lines(
            "after ",
            layer,
            &color.bytes().unwrap(),
            samples as usize,
            unit,
        ));
        if let Some(stencil) = stencil {
            // `Depth32Float_Stencil8`'s stencil, printed as a word.
            let words: Vec<u8> = stencil
                .bytes()
                .unwrap()
                .iter()
                .flat_map(|&s| u32::from(s).to_le_bytes())
                .collect();
            stencils.extend(lines("afterS ", layer, &words, samples as usize, 4));
        }
    }
    out.extend(stencils);
    Some(out)
}

/// Metal's lines of `case`, or `None` where none are recorded.
fn metal_lines(case: &str) -> Option<Vec<String>> {
    let head = format!("[{case}]\n");
    let at = METAL_LINES.find(&head)? + head.len();
    let rest = &METAL_LINES[at..];
    let end = rest.find("\n[").map_or(rest.len(), |n| n + 1);
    Some(rest[..end].lines().map(str::to_owned).collect())
}

/// How `lines` differ from Metal's, or `None` where they are alike: the first line they part at.
fn unlike_metal(lines: &[String], metal: &[String]) -> Option<String> {
    if lines == metal {
        return None;
    }
    match lines.iter().zip(metal).position(|(got, want)| got != want) {
        Some(i) => Some(format!(
            "line {i} is `{}`, Metal's `{}`",
            lines[i], metal[i]
        )),
        None => Some(format!("{} lines, Metal's {}", lines.len(), metal.len())),
    }
}

/// The head of a read-back line and its hex words: `after  s0 y1: | 3e800000 3e800000 | ...`.
fn line_words(line: &str) -> Option<(&str, Vec<&str>)> {
    let (head, words) = line.split_once(':')?;
    let words = words.split(['|', ' ']).filter(|w| !w.is_empty()).collect();
    Some((head, words))
}

/// Whether Metal's word `want` is a denormal 32-bit float and `got` is its zero, the sign kept.
fn flushed(got: &str, want: &str) -> bool {
    let (Ok(got), Ok(want)) = (u32::from_str_radix(got, 16), u32::from_str_radix(want, 16)) else {
        return false;
    };
    want & 0x7f80_0000 == 0 && want & 0x007f_ffff != 0 && got == want & 0x8000_0000
}

/// Whether `lines` differ from Metal's `metal` only where Metal holds a denormal 32-bit float
/// depth and this device holds its zero ([`flushed`]).
fn only_flushed_denormals(lines: &[String], metal: &[String]) -> bool {
    lines.len() == metal.len()
        && lines.iter().zip(metal).all(|(got, want)| {
            got == want
                || match (line_words(got), line_words(want)) {
                    (Some((got_head, got)), Some((want_head, want))) => {
                        got_head == want_head
                            && got.len() == want.len()
                            && (got.iter().zip(&want)).all(|(g, w)| g == w || flushed(g, w))
                    }
                    _ => false,
                }
        })
}

/// Whether `case` fills a 32-bit float depth aspect with [`DEPTHS`], whose first word is a
/// denormal: the only case shape whose answer a device that flushes depth denormals may differ in.
fn fills_a_depth_denormal(case: &Case) -> bool {
    let pf = PixelFormat::try_from(case.1).unwrap();
    matches!(
        pf,
        PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8
    ) && matches!(case.7, Fill::Bytes(bytes) if bytes == DEPTHS)
}

/// The recorded lines are the probe's: their own SHA-256, and each case's digest the probe
/// printed of its lines.
#[test]
fn the_recorded_lines_are_the_probes() {
    assert_eq!(
        crate::hash::hex_encode(Sha256::digest(METAL_LINES)),
        METAL_LINES_SHA,
        "the recorded lines"
    );
    for case in &CASES {
        let Some(lines) = metal_lines(case.0) else {
            assert_eq!(case.0, "d32-wide-4", "a case without recorded lines");
            continue;
        };
        assert_eq!(digest(&lines), case.8, "{}", case.0);
    }
}

/// Every case's samples are Metal's, but a depth whose drawn copy announced (once a device, at
/// the first depth fill) that it changes values out of `[0, 1]`, and a denormal depth on a device
/// that does not keep one a GPU write stores (`loses_depth_denormals`, announced
/// `depth-denormals`): a multisampled image takes no CPU store, so Metal's `0x00000001` reads
/// back as `0x00000000` there, and every other word is still compared.
#[test]
fn multisample_fills_write_metal_s_samples() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let mut wrong = Vec::new();
    let mut drawn = false;
    // Whether the device keeps no denormal a GPU write stores in a 32-bit float depth image, how
    // many cases differ in those words alone, and whether it announced the loss.
    let flushes = loses_depth_denormals(&lane.device, false);
    let mut flushed_cases = 0;
    let mut announced_denormals = false;
    for case in &CASES {
        let Some(lines) = run(&lane, case) else {
            continue;
        };
        let notices = messages.take_notices();
        let kinds: Vec<Degradation> = notices
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation)
            .collect();
        drawn |= kinds.contains(&Degradation::DrawnDepthCopy);
        announced_denormals |= kinds.contains(&Degradation::DepthDenormals);
        let case_name = case.0;
        let unlike = match metal_lines(case_name) {
            Some(metal) => unlike_metal(&lines, &metal),
            None => (digest(&lines) != case.8)
                .then(|| format!("{}, Metal's digest {}", digest(&lines), case.8)),
        };
        let Some(unlike) = unlike else {
            continue;
        };
        if drawn && case_name == "d32-wide-4" {
            println!(
                "SAMPLE-FILLS-DRAWN {name}: {case_name}: the drawn depth copy announced its \
                 changes: {}",
                lines[1]
            );
            continue;
        }
        let denormals = flushes
            && fills_a_depth_denormal(case)
            && metal_lines(case_name).is_some_and(|metal| only_flushed_denormals(&lines, &metal));
        if denormals {
            flushed_cases += 1;
            println!(
                "SAMPLE-FILLS-DENORMALS {name}: {case_name}: Metal's depth denormal reads back as \
                 its zero here, every other word Metal's: {}",
                lines[1]
            );
            continue;
        }
        wrong.push(format!("{case_name}: {unlike}"));
    }
    for line in &wrong {
        println!("SAMPLE-FILLS-DIFFERS {name}: {line}");
    }
    assert!(
        wrong.is_empty(),
        "{} cases differ from Metal's",
        wrong.len()
    );
    assert!(
        flushed_cases == 0 || announced_denormals,
        "{name}: {flushed_cases} case(s) hold Metal's depth denormal as its zero with no \
         `depth-denormals` notice"
    );
    quiet(lane, &messages);
}
