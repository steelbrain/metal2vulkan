//! A fetch of a blended 8-bit unorm attachment on the device the cfg selects, under the validation
//! layer: the owned fixtures `fbl_fetch_blend8` (fast math) and `fbl_fetch_blend8_safe` (without),
//! in the shape of a CoreAnimation composite of the x86 macOS guest. Color 0 is a `SIDE`² private
//! target of each 8-bit unorm format (`BGRA8Unorm`, the guest's, `RGBA8Unorm`, `RG8Unorm`,
//! `R8Unorm`) blended one over one minus the source alpha for color and alpha, color 1 an
//! `RGBA16Float` target of mask none. Each function reads color 0 and returns, at pixel
//! `i = SIDE · y + x`, a loaded source chosen by the fetched alpha (`src[i + SIDE²]` where it is
//! zero, else `src[i]`): `*_loaded` a `half4`, `*_loaded_float` a `float4`; or `*_product` the
//! loaded `half4` times the fetched color. One draw of the fullscreen triangle.
//!
//! Metal's answer (plain Metal on the M3 Max under macOS 27.0.1, our own MSL of this shape, every
//! `half` alpha in `[0, 1]` against every destination code of each channel, four source draws per
//! pair, each format, `half4` and `float4` outputs, fast and safe math, every channel: products and
//! sums of the output, the fetched color among them, fold nothing into the blend): the output
//! clamped to `[0, 1]`, a NaN to 0; the destination's code `c` read as `h(c / 255)` (`f(c / 255)`
//! for a `float4` output); `h(fma(d, h(1 - a), s))` under fast math and `h(s + h(d · h(1 - a)))`
//! without (in `float` for a `float4` output, likewise); the result's code by Metal's unorm write,
//! the exact product with 255 rounded to nearest. The texels are codes covering every value in each
//! channel, a zero alpha at times; the sources random `half`s or `float`s, mostly in `[0, 1]`, some
//! out of range, infinite or NaN.
//!
//! Where the draw announces no `half-output-fast-math-blend`, `half-output-precision` or
//! `device-color-conversion`, every channel is Metal's. On a device that fetches through the pixel
//! interlock (RADV, ANV, NVIDIA, MoltenVK) the function computes Metal's blend itself, exactly,
//! announcing none of the three, which is asserted; a format whose storage the device lacks there
//! is refused by that capability, which is printed. Through input attachments (lavapipe) the device
//! blends; what it announces and how many channels are Metal's is printed. A format the device
//! renders through another (an emulation or a word view) has no storage form at all and is refused
//! by name, printed too: NVIDIA renders no `R8Unorm_sRGB` or `RG8Unorm_sRGB` of its own, their
//! Vulkan formats being sampled images there with no color-attachment bit. Each draw prints a
//! `FETCH-BLEND8` line naming its function and format.
//!
//! Other blends, write masks and the blend color, into `BGRA8Unorm` through each fixture's
//! loaded `half4` and `float4`: source alpha over, the blend color's lerps with one channel of the
//! encoder's color past each end, blue alone stored, and alpha alone stored, against the
//! arithmetic `Blend::unorm8` names for each, each channel the mask leaves out its code; checked as
//! above.
//!
//! The four-channel 8-bit sRGB formats (`BGRA8Unorm_sRGB`, `RGBA8Unorm_sRGB`) under premultiplied
//! over and those blends: the same arithmetic over the texel as a fetch decodes it, encoded by
//! Metal's table (`Blend::srgb8`, measured on the M3 Max); asserted on a device that fetches
//! through the pixel interlock.

use super::{form, library, through_inputs};
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane_with, quiet, Lane};
use crate::runtime_lane::WAIT;
use crate::runtime_lane::{buffer_with, Messages};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::Needs;
use metal2vulkan::metal::format::{vk_format, FormatEmulation, WordAttachment};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, Error, Library, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeSet;

/// The target's side, as the fixtures'.
const SIDE: u64 = 64;

/// The target's texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The notes under which a channel may be another than Metal's.
const NOTES: [PrecisionNote; 3] = [
    PrecisionNote::HalfOutputFastMathBlend,
    PrecisionNote::HalfOutputPrecision,
    PrecisionNote::DeviceColorConversion,
];

/// Each format, its bytes a texel, and the channel (red 0 to alpha 3) of each byte.
const FORMATS: [(PixelFormat, &[usize]); 4] = [
    (PixelFormat::BGRA8Unorm, &[2, 1, 0, 3]),
    (PixelFormat::RGBA8Unorm, &[0, 1, 2, 3]),
    (PixelFormat::RG8Unorm, &[0, 1]),
    (PixelFormat::R8Unorm, &[0]),
];

/// What a function returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Output {
    /// A loaded `half4`.
    Half,
    /// A loaded `float4`.
    Float,
    /// The loaded `half4` times the fetched color.
    Product,
}

/// A fixture's functions: its vertex function, each fragment function and its output, and
/// whether the module computes without fast math.
struct Fixture {
    stem: &'static str,
    vertex: &'static str,
    cases: [(&'static str, Output); 3],
    safe: bool,
}

const FIXTURES: [Fixture; 2] = [
    Fixture {
        stem: "fbl_fetch_blend8",
        vertex: "fbl8_vertex",
        cases: [
            ("fbl8_over_loaded", Output::Half),
            ("fbl8_over_loaded_float", Output::Float),
            ("fbl8_over_product", Output::Product),
        ],
        safe: false,
    },
    Fixture {
        stem: "fbl_fetch_blend8_safe",
        vertex: "fbl8_safe_vertex",
        cases: [
            ("fbl8_safe_loaded", Output::Half),
            ("fbl8_safe_loaded_float", Output::Float),
            ("fbl8_safe_product", Output::Product),
        ],
        safe: true,
    },
];

/// A deterministic stream of 64-bit words (xorshift).
struct Words(u64);

impl Words {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A source lane: one time in 16 out of range (`specials`: or infinite, or NaN), else in
    /// `[0, 1]`.
    fn source(&mut self, specials: bool) -> f64 {
        const SPECIALS: [f64; 8] = [
            -0.0,
            -1.0,
            2.0,
            f64::INFINITY,
            -f64::INFINITY,
            f64::NAN,
            1.5,
            -0.25,
        ];
        let w = self.next();
        let k = ((w >> 4) % 8) as usize;
        match w.is_multiple_of(16) {
            true if specials || SPECIALS[k].is_finite() => SPECIALS[k],
            _ => ((w >> 16) & 0xFF_FFFF) as f64 / f64::from(0xFF_FFFFu32),
        }
    }
}

/// `x` rounded to `half`, nearest even (a NaN `0x7e00`).
fn h(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let ulp = 2f64.powi(e.max(-14) - 10);
    let r = (x / ulp).round_ties_even() * ulp;
    if r.abs() > 65504.0 {
        return f64::INFINITY.copysign(x);
    }
    r
}

/// The bits of `half` `x` (a value [`h`] gives).
fn half_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7c00;
    }
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    if e < -14 {
        return sign | (a / 2f64.powi(-24)) as u16;
    }
    sign | (((e + 15) as u16) << 10) | ((a / 2f64.powi(e - 10)) as u16 - 1024)
}

/// `x` clamped to `[0, 1]`, a NaN to 0.
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Metal's code of source lane `s` (alpha `a`) over destination code `c`, in `float` where
/// `float`, without fast math where `safe` (the module docs).
fn metal_code(float: bool, safe: bool, (s, a): (f64, f64), c: u8) -> u8 {
    let (s, a) = (clamp01(s), clamp01(a));
    let v = match float {
        false => {
            let d = h(f64::from(c) / 255.0);
            let m = h(1.0 - a);
            match safe {
                false => h(d * m + s),
                true => h(s + h(d * m)),
            }
        }
        true => {
            let d = f32::from(c) / 255.0;
            let (s, a) = (s as f32, a as f32);
            let m = 1.0 - a;
            f64::from(match safe {
                false => d.mul_add(m, s),
                true => s + d * m,
            })
        }
    };
    (clamp01(v) * 255.0 + 0.5).floor() as u8
}

/// The texels' codes, red to alpha, and the sources (two a pixel), as the output holds them:
/// each channel's code runs through every value, alpha zero at times. A product's sources are
/// finite: under fast math a product of an infinity or a NaN is undefined to Metal's compiler.
fn data(output: Output) -> (Vec<[u8; 4]>, Vec<[f64; 4]>) {
    let mut words = Words(0x51ab_1e8b_17c0_de55);
    let texels = (0..TEXELS)
        .map(|i| std::array::from_fn(|c| ((i * (2 * c + 1) + 37 * c) % 256) as u8))
        .collect();
    let sources = (0..2 * TEXELS)
        .map(|_| {
            [0; 4].map(|_: u8| {
                let x = words.source(output != Output::Product);
                match output {
                    Output::Float => f64::from(x as f32),
                    Output::Half | Output::Product => h(x),
                }
            })
        })
        .collect();
    (texels, sources)
}

/// A private `SIDE`² texture of `pf` holding `bytes`.
fn filled(lane: &Lane, pf: PixelFormat, bytes: &[u8], bpp: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let source = buffer_with(&lane.device, bytes).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        bpp * SIDE,
        bytes.len() as u64,
        Size::new(SIDE, SIDE, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}

/// A pipeline of `fixture`'s `entry` blending color 0 of `pf` by `blended` (with color 1 of mask
/// none), or `None` where the device cannot render the attachment as a fetch's storage form needs
/// (printed): a capability it lacks -- the format's storage through the pixel interlock, or its
/// rendering or blend through input attachments -- or a format it renders through another
/// (`NotImplemented`, below).
fn pipeline(
    lane: &Lane,
    (library, fixture): (&Library, &Fixture),
    entry: &str,
    blended: ColorAttachmentDescriptor,
) -> Option<RenderPipelineState> {
    let device = lane.device.name();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let pf = blended.pixel_format;
    // Whether the device holds the attachment in the Metal format's own Vulkan format. No storage
    // form of a fetch is built over an emulated attachment (`lower/fragment/fetch.rs`'s
    // `not_modelled`), so such a format's fetch is refused `NotImplemented`, by design and the
    // same for every device that cannot render it: NVIDIA's `R8_SRGB` and `R8G8_SRGB` are sampled
    // images with no color-attachment bit, so the one- and two-channel sRGB formats reach it there
    // where the four-channel ones do not.
    let usages = BTreeSet::from([FormatUsage::ColorAttachment]);
    let profile = lane.device.compile_device().profile();
    let held_directly = vk_format(pf, &usages, profile)
        .is_ok_and(|mapping| mapping.emulation == FormatEmulation::Direct);
    let mut descriptor = RenderPipelineDescriptor::new(function(fixture.vertex));
    descriptor.fragment = Some(function(entry));
    let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    unwritten.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(blended);
    descriptor.color_attachments[1] = Some(unwritten);
    match lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    {
        Ok(state) => Some(state),
        // A format whose storage, or blend, the device lacks is refused by that capability
        // (lavapipe renders and blends no `R8Unorm_sRGB` and `RG8Unorm_sRGB`, through input
        // attachments); one the device renders through another format, by name.
        Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented || !held_directly => {
            let held = match held_directly {
                true => "the format's own image",
                false => "rendered through another format",
            };
            println!("FETCH-BLEND8 {entry} {pf:?} | {device} | {held} | refused: {u}");
            None
        }
        Err(error) => panic!("{entry} into {pf:?} on {device}: {error}"),
    }
}

/// One draw of `state` (whose function returns `output`) over a target of `pf` (`order` the
/// channel of each byte) holding `texels`, the encoder's blend color `color`: the target's bytes
/// after it, and which of [`NOTES`] it announced.
fn draw(
    (lane, messages): (&Lane, &Messages),
    state: &RenderPipelineState,
    (pf, order): (PixelFormat, &[usize]),
    (output, texels, sources): (Output, &[[u8; 4]], &[[f64; 4]]),
    (color, kinds): ([f32; 4], &[Degradation]),
) -> (Vec<u8>, Vec<Degradation>) {
    let bpp = order.len();
    let bytes: Vec<u8> = texels
        .iter()
        .flat_map(|t| order.iter().map(|&c| t[c]))
        .collect();
    let target = filled(lane, pf, &bytes, bpp as u64);
    let mut second = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    second.usage = TextureUsage::RENDER_TARGET;
    second.resource_options = ResourceOptions::private();
    let second = lane.device.new_texture(&second).unwrap();
    let src: Vec<u8> = match output {
        Output::Float => sources
            .iter()
            .flat_map(|s| s.map(|v| v as f32))
            .flat_map(f32::to_le_bytes)
            .collect(),
        Output::Half | Output::Product => sources
            .iter()
            .flat_map(|s| s.map(half_bits))
            .flat_map(u16::to_le_bytes)
            .collect(),
    };
    let src = buffer_with(&lane.device, &src).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let attachment = &mut pass.color_attachments[1];
    attachment.texture = Some(second);
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    attachment.store_action = StoreAction::DontCare;
    let _ = messages.take_notices();
    lane.run(&pass, |_, encoder, _| {
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_blend_color(color[0], color[1], color[2], color[3]);
        encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    let announced = counts(&messages.take_notices());
    let noted = kinds
        .iter()
        .copied()
        .filter(|k| announced.contains_key(k))
        .collect();
    (
        lane.read(&target, (0, 0), bpp as u64, BlitOption::None),
        noted,
    )
}

/// The source each pixel's function returns: a load chosen by the fetched alpha, or the first
/// load times the fetched color (each channel's code, a missing color channel 0, a missing alpha
/// one).
fn returned(
    output: Output,
    order: &[usize],
    (i, codes): (usize, [u8; 4]),
    sources: &[[f64; 4]],
) -> [f64; 4] {
    let fetched: [f64; 4] = std::array::from_fn(|c| match order.contains(&c) {
        true => f64::from(codes[c]) / 255.0,
        false => f64::from(u8::from(c == 3)),
    });
    let chosen = match order.contains(&3) && codes[3] == 0 {
        true => i + TEXELS,
        false => i,
    };
    match output {
        Output::Product => std::array::from_fn(|c| h(sources[i][c] * h(fetched[c]))),
        _ => sources[chosen],
    }
}

/// Check one draw (the module docs): print its line, assert that a device fetching through the
/// pixel interlock announces none of [`NOTES`], and that every texel is Metal's where none is
/// announced. `wrong` holds each texel that is not: its index, the bytes read and Metal's.
fn checked(
    lane: &Lane,
    what: (&str, PixelFormat),
    noted: &[Degradation],
    wrong: &[(usize, Vec<u8>, Vec<u8>)],
) {
    let device = lane.device.name();
    let (entry, pf) = what;
    println!(
        "FETCH-BLEND8 {entry} {pf:?} | {device} | through {} | Metal's {}/{} texels | announced \
         {noted:?}",
        form(lane),
        TEXELS - wrong.len(),
        TEXELS,
    );
    let what = format!("{entry} into {pf:?} on {device}");
    if !through_inputs(lane) {
        assert!(noted.is_empty(), "{what}: announced {noted:?}");
    }
    if noted.is_empty() {
        let shown = &wrong[..wrong.len().min(8)];
        assert!(
            wrong.is_empty(),
            "{what}: {} of {TEXELS} texels differ from Metal's, first (index, read, Metal's) \
             {shown:?}",
            wrong.len()
        );
    }
}

/// Each draw's channels are Metal's where it announces none of [`NOTES`], and on a device that
/// fetches through the pixel interlock it announces none (the module docs).
#[test]
fn a_blended_eight_bit_unorm_fetch_stores_metals_blend() {
    let kinds = NOTES.map(Degradation::from_precision_note);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.into_iter().collect();
    });
    let device = lane.device.name();
    let mut drawn = 0;
    for fixture in &FIXTURES {
        let library = library(&lane, fixture.stem);
        for (pf, order) in FORMATS {
            for (entry, output) in fixture.cases {
                let mut blended = ColorAttachmentDescriptor::new(pf);
                blended.blending_enabled = true;
                blended.source_rgb_blend_factor = BlendFactor::One;
                blended.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                blended.rgb_blend_operation = BlendOperation::Add;
                blended.source_alpha_blend_factor = BlendFactor::One;
                blended.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                blended.alpha_blend_operation = BlendOperation::Add;
                let Some(state) = pipeline(&lane, (&library, fixture), entry, blended) else {
                    continue;
                };
                let (texels, sources) = data(output);
                let data = (output, texels.as_slice(), sources.as_slice());
                let (read, noted) = draw(
                    (&lane, &messages),
                    &state,
                    (pf, order),
                    data,
                    ([0.0; 4], &kinds),
                );
                let bpp = order.len();
                let mut wrong = Vec::new();
                for (i, &codes) in texels.iter().enumerate() {
                    let got = &read[bpp * i..bpp * i + bpp];
                    let s = returned(output, order, (i, codes), &sources);
                    let float = output == Output::Float;
                    let want: Vec<u8> = order
                        .iter()
                        .map(|&c| metal_code(float, fixture.safe, (s[c], s[3]), codes[c]))
                        .collect();
                    if got != want.as_slice() {
                        wrong.push((i, got.to_vec(), want));
                    }
                }
                checked(&lane, (entry, pf), &noted, &wrong);
                drawn += 1;
            }
        }
    }
    assert!(drawn > 0, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}

/// One lane's operands as Metal's blend reads them, each rounded to the arithmetic's type: the
/// source lane `s` and alpha `sa` clamped, the destination lane `d` and alpha `da` decoded, and
/// the blend color's lane `c` and alpha `ca` clamped.
#[derive(Clone, Copy)]
struct Operands {
    s: f64,
    sa: f64,
    d: f64,
    da: f64,
    c: f64,
    ca: f64,
}

/// A lane's Metal arithmetic, in `float` where the first flag says so and without fast math
/// where the second does.
type Form = fn((bool, bool), Operands) -> f64;

/// `x` rounded to `float` where `float`, else to `half`.
fn rounded(float: bool, x: f64) -> f64 {
    match float {
        true => f64::from(x as f32),
        false => h(x),
    }
}

/// `a b + c` rounded once, in `float` where `float`, else in `half` (exact in `f64` over a blend's
/// `half` operands).
fn fused(float: bool, (a, b, c): (f64, f64, f64)) -> f64 {
    match float {
        true => f64::from((a as f32).mul_add(b as f32, c as f32)),
        false => h(a * b + c),
    }
}

/// `x y + z w`, each product rounded, then their sum, in `float` where `float`, else `half`.
fn sum(float: bool, (x, y): (f64, f64), (z, w): (f64, f64)) -> f64 {
    rounded(float, rounded(float, x * y) + rounded(float, z * w))
}

/// The states [`other_blends_masks_and_the_blend_color_store_metals_blend`] draws: the blend
/// (color factors and operation, then alpha's), the write mask, the encoder's blend color, and
/// the color and alpha lanes' Metal arithmetic (`None` where the mask keeps the lane).
type State = (
    [(BlendFactor, BlendFactor, BlendOperation); 2],
    ColorWriteMask,
    [f32; 4],
    Option<Form>,
    Option<Form>,
);

/// The blends, write masks and blend colors [`other_blends_masks_and_the_blend_color_store_metals_blend`]
/// and [`a_blended_four_channel_srgb_fetch_stores_metals_blend`] draw, each with Metal's arithmetic
/// for its color and alpha lanes (`Blend::unorm8`'s forms).
fn states() -> [State; 4] {
    use BlendFactor as F;
    use BlendOperation as O;
    [
        // Source alpha over: the color's source product rounded first, alpha a lerp of the
        // source alpha from the destination's.
        (
            [
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
                (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
            ],
            ColorWriteMask::ALL,
            [0.0; 4],
            Some(|(float, safe), l| {
                let m = rounded(float, 1.0 - l.sa);
                match safe {
                    false => fused(float, (l.d, m, rounded(float, l.s * l.sa))),
                    true => sum(float, (l.s, l.sa), (l.d, m)),
                }
            }),
            Some(|(float, safe), l| match safe {
                false => fused(float, (l.sa, rounded(float, l.sa - l.da), l.da)),
                true => sum(float, (l.sa, l.sa), (l.da, rounded(float, 1.0 - l.sa))),
            }),
        ),
        // The blend color, one channel past each end: lerps of it under fast math.
        (
            [
                (F::BlendColor, F::OneMinusBlendColor, O::Add),
                (F::BlendAlpha, F::OneMinusBlendAlpha, O::Add),
            ],
            ColorWriteMask::ALL,
            [0.3, -0.25, 1.5, 0.45],
            Some(|(float, safe), l| match safe {
                false => fused(float, (l.c, rounded(float, l.s - l.d), l.d)),
                true => sum(float, (l.s, l.c), (l.d, rounded(float, 1.0 - l.c))),
            }),
            Some(|(float, safe), l| match safe {
                false => fused(float, (l.ca, rounded(float, l.sa - l.da), l.da)),
                true => sum(float, (l.sa, l.ca), (l.da, rounded(float, 1.0 - l.ca))),
            }),
        ),
        // Blue alone stored: a lerp of the source alpha a vector does not take.
        (
            [
                (F::OneMinusSourceAlpha, F::SourceAlpha, O::Add),
                (F::One, F::Zero, O::Add),
            ],
            ColorWriteMask::BLUE,
            [0.0; 4],
            Some(|(float, safe), l| match safe {
                false => fused(float, (l.sa, rounded(float, l.d - l.s), l.s)),
                true => sum(float, (l.s, rounded(float, 1.0 - l.sa)), (l.d, l.sa)),
            }),
            None,
        ),
        // Alpha alone stored: the color channels' destination factor computes one minus the
        // destination alpha still, so alpha takes no lerp of it.
        (
            [
                (F::Zero, F::OneMinusDestinationAlpha, O::Add),
                (F::OneMinusDestinationColor, F::DestinationColor, O::Add),
            ],
            ColorWriteMask::ALPHA,
            [0.0; 4],
            None,
            Some(|(float, safe), l| {
                let m = rounded(float, 1.0 - l.da);
                match safe {
                    false => fused(float, (l.sa, m, rounded(float, l.da * l.da))),
                    true => sum(float, (l.sa, m), (l.da, l.da)),
                }
            }),
        ),
    ]
}

/// Other blends, write masks and the blend color, into `BGRA8Unorm`: Metal's arithmetic (plain
/// Metal on the M3 Max under macOS 27.0.1) is each color lane's and alpha's form `Blend::unorm8`
/// names; a channel the mask leaves out keeps its code. Each draw's channels are Metal's where it
/// announces none of [`NOTES`], and on a device that fetches through the pixel interlock it
/// announces none.
#[test]
fn other_blends_masks_and_the_blend_color_store_metals_blend() {
    let states = states();
    let kinds = NOTES.map(Degradation::from_precision_note);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.into_iter().collect();
    });
    let device = lane.device.name();
    let (pf, order) = FORMATS[0];
    let mut drawn = 0;
    for fixture in &FIXTURES {
        let library = library(&lane, fixture.stem);
        for &(entry, output) in &fixture.cases[..2] {
            for ([color, alpha], mask, bc, color_form, alpha_form) in states {
                let mut blended = ColorAttachmentDescriptor::new(pf);
                blended.blending_enabled = true;
                (
                    blended.source_rgb_blend_factor,
                    blended.destination_rgb_blend_factor,
                    blended.rgb_blend_operation,
                ) = color;
                (
                    blended.source_alpha_blend_factor,
                    blended.destination_alpha_blend_factor,
                    blended.alpha_blend_operation,
                ) = alpha;
                blended.write_mask = mask;
                let Some(state) = pipeline(&lane, (&library, fixture), entry, blended) else {
                    continue;
                };
                let (texels, sources) = data(output);
                let data = (output, texels.as_slice(), sources.as_slice());
                let (read, noted) =
                    draw((&lane, &messages), &state, (pf, order), data, (bc, &kinds));
                let float = output == Output::Float;
                let read_as = |x: f64| rounded(float, clamp01(x));
                let mut wrong = Vec::new();
                for (i, &codes) in texels.iter().enumerate() {
                    let got = &read[4 * i..4 * i + 4];
                    let s = returned(output, order, (i, codes), &sources);
                    let operands = |c: usize| Operands {
                        s: read_as(s[c]),
                        sa: read_as(s[3]),
                        d: rounded(float, f64::from(codes[c]) / 255.0),
                        da: rounded(float, f64::from(codes[3]) / 255.0),
                        c: read_as(f64::from(bc[c])),
                        ca: read_as(f64::from(bc[3])),
                    };
                    let want: Vec<u8> = order
                        .iter()
                        .map(|&c| {
                            let form = if c < 3 { color_form } else { alpha_form };
                            let bit = [
                                ColorWriteMask::RED,
                                ColorWriteMask::GREEN,
                                ColorWriteMask::BLUE,
                                ColorWriteMask::ALPHA,
                            ][c];
                            match form.filter(|_| mask.contains(bit)) {
                                Some(form) => {
                                    let v = form((float, fixture.safe), operands(c));
                                    (clamp01(v) * 255.0 + 0.5).floor() as u8
                                }
                                None => codes[c],
                            }
                        })
                        .collect();
                    if got != want.as_slice() {
                        wrong.push((i, got.to_vec(), want));
                    }
                }
                let what = format!("{entry} {color:?} {alpha:?} {mask:?} {bc:?}");
                checked(&lane, (&what, pf), &noted, &wrong);
                drawn += 1;
            }
        }
    }
    assert!(drawn > 0, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}

/// Metal's sRGB byte of each twelve-bit index (the first indices of `bind::color_write_table`,
/// Metal's encode thresholds).
pub(super) static SRGB_BYTES: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
    let first: Vec<u32> = metal2vulkan::bind::color_write_table()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|e| u32::from_le_bytes([e[0], e[1], e[2], e[3]]))
        .collect();
    (0..4096u32)
        .map(|i| (1..256).rev().find(|&b| first[b] <= i).unwrap_or(0) as u8)
        .collect()
});

/// A blended 8-bit four-channel sRGB attachment (`BGRA8Unorm_sRGB`, `RGBA8Unorm_sRGB`) fetched,
/// on the device the cfg selects, through each fixture's loaded `half4` and `float4`: premultiplied
/// over and [`states`]' blends, masks and blend colors. Metal's answer (plain Metal on the M3 Max
/// under macOS 27.0.1, our own MSL of this shape, random sources and texel bytes, every one of
/// these states, both outputs, fast and safe math, both formats: every channel) is the 8-bit unorm
/// blend's arithmetic over the texel as a fetch reads it, each color byte's twelve-bit step over
/// 4095 and alpha's byte over 255 rounded to the output's type, the result's color bytes by
/// Metal's sRGB encode (the exact product with 4095 rounded to nearest, through its thresholds)
/// and alpha's by its unorm write. The texels' codes cover every byte in each channel. On a device
/// that fetches through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the function computes
/// that blend itself, exactly, announcing none of [`NOTES`], which is asserted with every channel;
/// through input attachments (lavapipe) the device blends, and what it announces and how many
/// texels are Metal's is printed. Each draw prints a `FETCH-BLEND8` line.
#[test]
fn a_blended_four_channel_srgb_fetch_stores_metals_blend() {
    srgb_blends(&[
        (PixelFormat::BGRA8UnormSRGB, &[2, 1, 0, 3]),
        (PixelFormat::RGBA8UnormSRGB, &[0, 1, 2, 3]),
    ]);
}

/// A blended 8-bit one- or two-channel sRGB attachment (`R8Unorm_sRGB`, `RG8Unorm_sRGB`) fetched,
/// as [`a_blended_four_channel_srgb_fetch_stores_metals_blend`] draws it, its alpha read as 1 and
/// a missing color channel as 0. Metal's answer (the M3 Max under macOS 27.0.1, our probe `sb.m`
/// of the same shape over both formats) is the same arithmetic in every channel of every state,
/// both outputs and both maths, but `R8Unorm_sRGB`'s source-alpha forms under fast math with a
/// `half4`, which the function computes in the rounded forms and notes (`UnmeasuredFetchBlend`,
/// beside the pipeline's `HalfOutputFastMathBlend`): those draws print their count of texels the
/// model gives and assert nothing more. On a
/// device that renders the format through a word view (`WordAttachment`) the pipeline is
/// refused, printed.
#[test]
fn a_blended_one_or_two_channel_srgb_fetch_stores_metals_blend() {
    srgb_blends(&[
        (PixelFormat::R8UnormSRGB, &[0]),
        (PixelFormat::RG8UnormSRGB, &[0, 1]),
    ]);
}

/// The draws of the sRGB blend tests over `formats`, each with its bytes' channels.
fn srgb_blends(formats: &[(PixelFormat, &[usize])]) {
    use BlendFactor as F;
    use BlendOperation as O;
    // Premultiplied over: `fma(d, 1 - a, s)` under fast math, `s + d (1 - a)` without.
    let over_form: Form = |(float, safe), l| {
        let m = rounded(float, 1.0 - l.sa);
        match safe {
            false => fused(float, (l.d, m, l.s)),
            true => rounded(float, l.s + rounded(float, l.d * m)),
        }
    };
    let over: State = (
        [(F::One, F::OneMinusSourceAlpha, O::Add); 2],
        ColorWriteMask::ALL,
        [0.0; 4],
        Some(over_form),
        Some(over_form),
    );
    let unmeasured = Degradation::from_precision_note(PrecisionNote::UnmeasuredFetchBlend);
    let kinds: Vec<Degradation> = (NOTES.map(Degradation::from_precision_note).into_iter())
        .chain([unmeasured])
        .collect();
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.iter().copied().collect();
    });
    let device = lane.device.name();
    let bytes = &*SRGB_BYTES;
    let mut drawn = 0;
    for fixture in &FIXTURES {
        let library = library(&lane, fixture.stem);
        for &(pf, order) in formats {
            for &(entry, output) in &fixture.cases[..2] {
                for ([color, alpha], mask, bc, color_form, alpha_form) in
                    std::iter::once(over).chain(states())
                {
                    let mut blended = ColorAttachmentDescriptor::new(pf);
                    blended.blending_enabled = true;
                    (
                        blended.source_rgb_blend_factor,
                        blended.destination_rgb_blend_factor,
                        blended.rgb_blend_operation,
                    ) = color;
                    (
                        blended.source_alpha_blend_factor,
                        blended.destination_alpha_blend_factor,
                        blended.alpha_blend_operation,
                    ) = alpha;
                    blended.write_mask = mask;
                    let Some(state) = pipeline(&lane, (&library, fixture), entry, blended) else {
                        continue;
                    };
                    let (texels, sources) = data(output);
                    let data = (output, texels.as_slice(), sources.as_slice());
                    let (read, noted) =
                        draw((&lane, &messages), &state, (pf, order), data, (bc, &kinds));
                    let bpp = order.len();
                    let float = output == Output::Float;
                    let read_as = |x: f64| rounded(float, clamp01(x));
                    // A code as Metal's blend reads it: a color byte's step over 4095, alpha's
                    // byte over 255, rounded to the arithmetic's type.
                    let decoded = |c: usize, code: u8| match c < 3 {
                        true => {
                            let step = crate::runtime_lane::srgb_step(u32::from(code));
                            rounded(float, f64::from(step) / 4095.0)
                        }
                        false => rounded(float, f64::from(code) / 255.0),
                    };
                    let mut wrong = Vec::new();
                    for (i, &codes) in texels.iter().enumerate() {
                        let got = &read[bpp * i..bpp * i + bpp];
                        let s = returned(output, order, (i, codes), &sources);
                        // A format without alpha reads it as 1.
                        let da = match order.contains(&3) {
                            true => decoded(3, codes[3]),
                            false => 1.0,
                        };
                        let operands = |c: usize| Operands {
                            s: read_as(s[c]),
                            sa: read_as(s[3]),
                            d: decoded(c, codes[c]),
                            da,
                            c: read_as(f64::from(bc[c])),
                            ca: read_as(f64::from(bc[3])),
                        };
                        let want: Vec<u8> = order
                            .iter()
                            .map(|&c| {
                                let form = if c < 3 { color_form } else { alpha_form };
                                let bit = [
                                    ColorWriteMask::RED,
                                    ColorWriteMask::GREEN,
                                    ColorWriteMask::BLUE,
                                    ColorWriteMask::ALPHA,
                                ][c];
                                match form.filter(|_| mask.contains(bit)) {
                                    Some(form) => {
                                        let v = clamp01(form((float, fixture.safe), operands(c)));
                                        match c < 3 {
                                            true => bytes[(v * 4095.0).round_ties_even() as usize],
                                            false => (v * 255.0).round_ties_even() as u8,
                                        }
                                    }
                                    None => codes[c],
                                }
                            })
                            .collect();
                        if got != want.as_slice() {
                            wrong.push((i, got.to_vec(), want));
                        }
                    }
                    let what = format!("{entry} {color:?} {alpha:?} {mask:?} {bc:?}");
                    if noted.contains(&unmeasured) {
                        println!(
                            "FETCH-BLEND8 {what} {pf:?} | {device} | through {} | the model's \
                             {}/{} texels | announced {noted:?}",
                            form(&lane),
                            TEXELS - wrong.len(),
                            TEXELS,
                        );
                    } else if through_inputs(&lane) {
                        println!(
                            "FETCH-BLEND8 {what} {pf:?} | {device} | through {} | Metal's {}/{} \
                             texels | announced {noted:?}",
                            form(&lane),
                            TEXELS - wrong.len(),
                            TEXELS,
                        );
                    } else {
                        checked(&lane, (&what, pf), &noted, &wrong);
                    }
                    drawn += 1;
                }
            }
        }
    }
    // A device that renders none of the formats but through its word view refuses each pipeline
    // (printed); one that renders any draws.
    let profile = lane.device.compile_device().profile();
    let renders = (formats.iter()).any(|&(pf, _)| WordAttachment::of(pf, profile).is_none());
    assert!(drawn > 0 || !renders, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}
