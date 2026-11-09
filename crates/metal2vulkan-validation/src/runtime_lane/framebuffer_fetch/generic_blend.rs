//! A fetch of a blended attachment of a format whose Metal blend arithmetic the translation does
//! not build, on the device the cfg selects, under the validation layer: the owned fixtures
//! `fbl_fetch_blend8` (fast math) and `fbl_fetch_blend8_safe` (without) into `RGBA32Float`,
//! `RGBA16Unorm`, `RGBA8Snorm`, `RGB10A2Unorm` and `A8Unorm` color 0 targets of `SIDE`², beside an
//! `RGBA16Float` color 1 of mask none. Each function reads color 0 and returns, at pixel
//! `i = SIDE · y + x`, a loaded source chosen by the fetched alpha (`src[i + SIDE²]` where it is
//! zero, else `src[i]`): `*_loaded` a `half4`, `*_loaded_float` a `float4`.
//! Two blends: one over one minus the source alpha, and the blend color over one minus the blend
//! alpha reversed beside the destination alpha over one minus the source color subtracted (the
//! encoder's color `(0.25, 1.5, -0.5, 0.75)`). One draw of the fullscreen triangle.
//!
//! On a device that fetches through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the function
//! computes the generic float blend (`StorageView::generic_blend`), and each draw announces
//! `unmeasured-fetch-blend`, which is asserted. Its answer is the model's: the source (and the
//! blend color) clamped to a normalized format's range, a NaN to its low end; the texel decoded;
//! each factor, product and sum rounded to `float`; the device's conversion storing the result. An
//! `RGBA32Float` target holds the model's `float`s exactly, which is asserted; another format's
//! channels lie within one code of the model's rounding of them (the device's decode and
//! conversion are its own), which is asserted, and how many are equal is printed. A format whose
//! storage the device lacks there is refused by that capability, which is printed. Through input
//! attachments (lavapipe) the device blends; how many channels are the model's is printed. Each
//! draw prints a `FETCH-GENERIC` line.
//!
//! Plain Metal on the M3 Max under macOS 27.0.1 (our own MSL of this shape, these formats, both
//! blends, `half4` and `float4` outputs, fast and safe math): the model holds every channel of
//! `RGBA32Float` without fast math, and of `RGB10A2Unorm` and `A8Unorm` of a `float4` output; under
//! fast math Metal fuses the source-over product into its sum; a `half4` output's 10-bit and 8-bit
//! channels lie within one code (Metal reads them in `half`), the 16-bit normalized and 8-bit snorm
//! ones too.
//!
//! A blend of the second source (the owned fixture `fbl_fetch_dual`, fast math: `half4` and
//! `float4` outputs and second outputs, each loaded from its own buffer) into `RGBA32Float`,
//! `RGBA16Float`, `RGBA16Unorm` and `RGBA8Unorm`: the second output's color over one minus it, and
//! its alpha over one minus it, the second source clamped as the source; checked as above. Metal
//! makes and draws such a pipeline; against our probe of it (both outputs `half4` or `float4`,
//! fast and safe math) the model holds every channel of `RGBA32Float` without fast math and of
//! `RGBA8Unorm` of `float4` outputs, and `RGBA16Unorm`'s within one code; Metal blends
//! `RGBA16Float` in `half` and writes it toward zero, so there the model is a code off in about
//! half the channels, and further where a `half` step overflows. Into `RGBA8Unorm_sRGB` too: the
//! function computes the 8-bit unorm blend's rounded forms over the decoded texel in the output's
//! type, encoded by Metal's table (`Blend::srgb8`), which the model's `float` blend of the decoded
//! texel, encoded by that table, meets within one code; Metal's without fast math, and within one
//! code of the fast-math fixture's on our probe.

use super::{fetches, form, library, through_inputs};
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane_with, quiet, Lane};
use crate::runtime_lane::WAIT;
use crate::runtime_lane::{buffer_with, Messages};
use metal2vulkan::error::Needs;
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

/// The target's side, as the fixture's.
const SIDE: u64 = 64;

/// The target's texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The encoder's blend color.
const COLOR: [f32; 4] = [0.25, 1.5, -0.5, 0.75];

/// How a format holds its channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Float32,
    Half16,
    Unorm16,
    Snorm8,
    Unorm8,
    Rgb10a2,
    A8,
    /// `RGBA8Unorm_sRGB`: each color byte Metal's twelve-bit step over 4095, alpha's over 255.
    Srgb8,
}

impl Layout {
    /// The bytes a texel.
    fn bytes(self) -> usize {
        match self {
            Layout::Float32 => 16,
            Layout::Half16 | Layout::Unorm16 => 8,
            Layout::Snorm8 | Layout::Unorm8 | Layout::Rgb10a2 | Layout::Srgb8 => 4,
            Layout::A8 => 1,
        }
    }

    /// The low end of the range a normalized format clamps the source to.
    fn low(self) -> Option<f32> {
        match self {
            Layout::Float32 | Layout::Half16 => None,
            Layout::Snorm8 => Some(-1.0),
            Layout::Unorm16 | Layout::Unorm8 | Layout::Rgb10a2 | Layout::A8 | Layout::Srgb8 => {
                Some(0.0)
            }
        }
    }

    /// The texel `b` as the storage image reads it, red to alpha (a channel the format lacks 0).
    fn decode(self, b: &[u8]) -> [f32; 4] {
        match self {
            Layout::Float32 => {
                std::array::from_fn(|c| f32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap()))
            }
            Layout::Half16 => {
                std::array::from_fn(|c| from_half(u16::from_le_bytes([b[2 * c], b[2 * c + 1]])))
            }
            Layout::Unorm16 => std::array::from_fn(|c| {
                f32::from(u16::from_le_bytes([b[2 * c], b[2 * c + 1]])) / 65535.0
            }),
            Layout::Unorm8 => std::array::from_fn(|c| f32::from(b[c]) / 255.0),
            Layout::Snorm8 => std::array::from_fn(|c| (f32::from(b[c] as i8) / 127.0).max(-1.0)),
            Layout::Rgb10a2 => {
                let w = u32::from_le_bytes(b.try_into().unwrap());
                std::array::from_fn(|c| match c {
                    3 => (w >> 30) as f32 / 3.0,
                    _ => ((w >> (10 * c)) & 1023) as f32 / 1023.0,
                })
            }
            Layout::A8 => [0.0, 0.0, 0.0, f32::from(b[0]) / 255.0],
            Layout::Srgb8 => std::array::from_fn(|c| match c {
                3 => f32::from(b[3]) / 255.0,
                _ => crate::runtime_lane::srgb_step(u32::from(b[c])) as f32 / 4095.0,
            }),
        }
    }

    /// Each stored channel's code of texel `b` (a `float`'s bits).
    fn codes(self, b: &[u8]) -> Vec<i64> {
        match self {
            Layout::Float32 => (0..4)
                .map(|c| i64::from(u32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap())))
                .collect(),
            Layout::Half16 | Layout::Unorm16 => (0..4)
                .map(|c| i64::from(u16::from_le_bytes([b[2 * c], b[2 * c + 1]])))
                .collect(),
            Layout::Snorm8 => b.iter().map(|&x| i64::from(x as i8)).collect(),
            Layout::Unorm8 | Layout::Srgb8 => b.iter().map(|&x| i64::from(x)).collect(),
            Layout::Rgb10a2 => {
                let w = u32::from_le_bytes(b.try_into().unwrap());
                vec![
                    i64::from(w & 1023),
                    i64::from((w >> 10) & 1023),
                    i64::from((w >> 20) & 1023),
                    i64::from(w >> 30),
                ]
            }
            Layout::A8 => vec![i64::from(b[0])],
        }
    }

    /// The codes of value `v` stored by a conversion rounding to nearest.
    fn encode(self, v: [f32; 4]) -> Vec<i64> {
        let code = |x: f32, low: f32, max: f32| {
            let x = if x.is_nan() { low } else { x.clamp(low, 1.0) };
            (f64::from(x) * f64::from(max)).round() as i64
        };
        match self {
            Layout::Float32 => v.iter().map(|x| i64::from(x.to_bits())).collect(),
            Layout::Half16 => v.iter().map(|&x| i64::from(to_half(x))).collect(),
            Layout::Unorm16 => v.iter().map(|&x| code(x, 0.0, 65535.0)).collect(),
            Layout::Snorm8 => v.iter().map(|&x| code(x, -1.0, 127.0)).collect(),
            Layout::Unorm8 => v.iter().map(|&x| code(x, 0.0, 255.0)).collect(),
            Layout::Rgb10a2 => vec![
                code(v[0], 0.0, 1023.0),
                code(v[1], 0.0, 1023.0),
                code(v[2], 0.0, 1023.0),
                code(v[3], 0.0, 3.0),
            ],
            Layout::A8 => vec![code(v[3], 0.0, 255.0)],
            Layout::Srgb8 => (0..4)
                .map(|c| match c {
                    3 => code(v[3], 0.0, 255.0),
                    _ => {
                        let index = code(v[c], 0.0, 4095.0) as usize;
                        i64::from(super::blend8::SRGB_BYTES[index])
                    }
                })
                .collect(),
        }
    }
}

/// The formats.
const FORMATS: [(PixelFormat, Layout); 5] = [
    (PixelFormat::RGBA32Float, Layout::Float32),
    (PixelFormat::RGBA16Unorm, Layout::Unorm16),
    (PixelFormat::RGBA8Snorm, Layout::Snorm8),
    (PixelFormat::RGB10A2Unorm, Layout::Rgb10a2),
    (PixelFormat::A8Unorm, Layout::A8),
];

/// The blends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blend {
    /// One over one minus the source alpha, added, for color and alpha.
    Over,
    /// Color: the blend color over one minus the blend alpha, reverse subtracted; alpha: the
    /// destination alpha over one minus the source color, subtracted.
    Constant,
    /// The second source's color over one minus it, added, and its alpha over one minus it,
    /// added: the function's second output weighs the first against the destination.
    Dual,
}

impl Blend {
    fn attachment(self, pf: PixelFormat) -> ColorAttachmentDescriptor {
        use BlendFactor as F;
        use BlendOperation as O;
        let mut a = ColorAttachmentDescriptor::new(pf);
        a.blending_enabled = true;
        let (sc, dc, oc, sa, da, oa) = match self {
            Blend::Over => (
                F::One,
                F::OneMinusSourceAlpha,
                O::Add,
                F::One,
                F::OneMinusSourceAlpha,
                O::Add,
            ),
            Blend::Constant => (
                F::BlendColor,
                F::OneMinusBlendAlpha,
                O::ReverseSubtract,
                F::DestinationAlpha,
                F::OneMinusSourceColor,
                O::Subtract,
            ),
            Blend::Dual => (
                F::Source1Color,
                F::OneMinusSource1Color,
                O::Add,
                F::Source1Alpha,
                F::OneMinusSource1Alpha,
                O::Add,
            ),
        };
        a.source_rgb_blend_factor = sc;
        a.destination_rgb_blend_factor = dc;
        a.rgb_blend_operation = oc;
        a.source_alpha_blend_factor = sa;
        a.destination_alpha_blend_factor = da;
        a.alpha_blend_operation = oa;
        a
    }

    /// The model's blend of source `s` (and second source `s1`) over destination `d` into a
    /// format of `layout` (the module docs).
    fn model(self, layout: Layout, (s, s1): ([f32; 4], [f32; 4]), d: [f32; 4]) -> [f32; 4] {
        let clamp = |x: f32| match layout.low() {
            Some(low) if x.is_nan() => low,
            Some(low) => x.clamp(low, 1.0),
            None => x,
        };
        let s = s.map(clamp);
        let s1 = s1.map(clamp);
        let c = COLOR.map(clamp);
        match self {
            Blend::Over => {
                let m = 1.0 - s[3];
                std::array::from_fn(|k| s[k] * 1.0 + d[k] * m)
            }
            Blend::Constant => {
                let m = 1.0 - c[3];
                let n = 1.0 - s[3];
                std::array::from_fn(|k| match k {
                    3 => s[3] * d[3] - d[3] * n,
                    _ => d[k] * m - s[k] * c[k],
                })
            }
            Blend::Dual => std::array::from_fn(|k| s[k] * s1[k] + d[k] * (1.0 - s1[k])),
        }
    }
}

/// A deterministic stream of 64-bit words (xorshift).
struct Words(u64);

impl Words {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A lane: one time in 16 out of `[0, 1]` (`specials`: or infinite, or NaN), else in it.
    fn lane(&mut self, specials: bool) -> f32 {
        const SPECIALS: [f32; 8] = [
            -0.0,
            -1.0,
            2.0,
            f32::INFINITY,
            -f32::INFINITY,
            f32::NAN,
            1.5,
            -0.25,
        ];
        let w = self.next();
        let k = ((w >> 4) % 8) as usize;
        match w.is_multiple_of(16) {
            true if specials || SPECIALS[k].is_finite() => SPECIALS[k],
            _ => ((w >> 16) & 0xFF_FFFF) as f32 / 0xFF_FFFF as f32,
        }
    }
}

/// The bits of `x` rounded to `half`, nearest even (a NaN `0x7e00`).
fn to_half(x: f32) -> u16 {
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = f64::from(x.abs());
    if x.is_nan() {
        return 0x7e00;
    }
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() || a >= 65520.0 {
        return sign | 0x7c00;
    }
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let ulp = 2f64.powi(e.max(-14) - 10);
    let q = (a / ulp).round_ties_even();
    // `q · ulp`, as the bits of a `half` (a carry into the next binade included).
    match e < -14 {
        true => sign | q as u16,
        false => sign | ((((e + 15) as u16) << 10) + (q as u16 - 1024)),
    }
}

/// The `half` of bits `b`.
fn from_half(b: u16) -> f32 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 10) & 0x1F);
    let m = f32::from(b & 0x3FF);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The target's bytes before the draw, and the sources and second sources (two of each a pixel),
/// as `half`s where `half`.
fn data(layout: Layout, half: bool) -> (Vec<u8>, [Vec<[f32; 4]>; 2]) {
    let mut words = Words(0x51ab_1e8b_17c0_de55);
    let bytes = match layout {
        Layout::Float32 => (0..4 * TEXELS)
            .flat_map(|_| words.lane(false).to_le_bytes())
            .collect(),
        _ => (0..layout.bytes() * TEXELS)
            .map(|_| (words.next() >> 24) as u8)
            .collect(),
    };
    let mut sources = || {
        (0..2 * TEXELS)
            .map(|_| {
                [0; 4].map(|_: u8| {
                    let x = words.lane(true);
                    match half {
                        true => from_half(to_half(x)),
                        false => x,
                    }
                })
            })
            .collect()
    };
    let first = sources();
    (bytes, [first, sources()])
}

/// A private `SIDE`² texture of `pf` holding `bytes`, `bpp` a texel.
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

/// A pipeline of `library`'s `vertex` and `entry` blending color 0 by `blended` (with color 1 of
/// mask none), or `None` where the device lacks a capability the pipeline needs: the format's
/// storage through the pixel interlock, or its blend through input attachments (printed).
fn pipeline(
    lane: &Lane,
    (library, vertex): (&Library, &str),
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
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
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
        // (lavapipe blends no `RGBA8Snorm`, through input attachments).
        Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {
            println!("FETCH-GENERIC {entry} {pf:?} | {device} | refused: {u}");
            None
        }
        Err(error) => panic!("{entry} into {pf:?} on {device}: {error}"),
    }
}

/// One draw of `state` over a target of `pf` holding `bytes`, the sources and second sources
/// `sources` (`half`s where `half`) in buffers 0 and 1: the target's bytes after it, and whether it
/// announced `unmeasured-fetch-blend`.
fn draw(
    (lane, messages): (&Lane, &Messages),
    state: &RenderPipelineState,
    (pf, layout): (PixelFormat, Layout),
    (half, bytes, sources): (bool, &[u8], &[Vec<[f32; 4]>; 2]),
) -> (Vec<u8>, bool) {
    let kind = Degradation::from_precision_note(PrecisionNote::UnmeasuredFetchBlend);
    let bpp = layout.bytes() as u64;
    let target = filled(lane, pf, bytes, bpp);
    let mut second = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    second.usage = TextureUsage::RENDER_TARGET;
    second.resource_options = ResourceOptions::private();
    let second = lane.device.new_texture(&second).unwrap();
    let buffers = sources.each_ref().map(|sources| {
        let bytes: Vec<u8> = match half {
            true => sources
                .iter()
                .flat_map(|s| s.map(to_half))
                .flat_map(u16::to_le_bytes)
                .collect(),
            false => sources
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        buffer_with(&lane.device, &bytes).unwrap()
    });
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
        encoder.set_blend_color(COLOR[0], COLOR[1], COLOR[2], COLOR[3]);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder
                .set_fragment_buffer(Some(buffer), 0, index as u32)
                .unwrap();
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    let announced = counts(&messages.take_notices()).contains_key(&kind);
    (lane.read(&target, (0, 0), bpp, BlitOption::None), announced)
}

/// Draw `entry` of `library` (whose vertex function is `vertex`, whose outputs are `half`s where
/// `half`) into a target of `pf` blended by `blend`, and check it (the module docs): print its line,
/// and on a device that fetches through the pixel interlock assert that it announced
/// `unmeasured-fetch-blend` and that every channel lies within one code of the model's (a `float`
/// channel's exactly). Whether it drew.
fn checked(
    (lane, messages): (&Lane, &Messages),
    (library, vertex): (&Library, &str),
    (entry, half): (&str, bool),
    (pf, layout): (PixelFormat, Layout),
    blend: Blend,
) -> bool {
    let device = lane.device.name();
    let Some(state) = pipeline(lane, (library, vertex), entry, blend.attachment(pf)) else {
        return false;
    };
    let (bytes, sources) = data(layout, half);
    let data = (half, bytes.as_slice(), &sources);
    let (read, announced) = draw((lane, messages), &state, (pf, layout), data);
    let bpp = layout.bytes();
    let nan = |code: i64| match layout {
        Layout::Float32 => f32::from_bits(code as u32).is_nan(),
        Layout::Half16 => code & 0x7c00 == 0x7c00 && code & 0x3ff != 0,
        _ => false,
    };
    let (mut equal, mut channels, mut worst) = (0, 0, 0);
    for i in 0..TEXELS {
        let d = layout.decode(&bytes[bpp * i..bpp * i + bpp]);
        let chosen = if d[3] == 0.0 { i + TEXELS } else { i };
        let s = (sources[0][chosen], sources[1][chosen]);
        let want = layout.encode(blend.model(layout, s, d));
        let got = layout.codes(&read[bpp * i..bpp * i + bpp]);
        for (&w, &g) in want.iter().zip(&got) {
            channels += 1;
            match w == g || (nan(w) && nan(g)) {
                true => equal += 1,
                false if layout == Layout::Float32 => worst = i64::MAX,
                false => worst = worst.max((w - g).abs()),
            }
        }
    }
    println!(
        "FETCH-GENERIC {entry} {pf:?} {blend:?} | {device} | through {} | the model's \
         {equal}/{channels} channels, worst {worst} | announced {announced}",
        form(lane),
    );
    if !through_inputs(lane) {
        let what = format!("{entry} {pf:?} {blend:?} on {device}");
        assert!(announced, "{what}: no unmeasured-fetch-blend");
        assert!(worst <= 1, "{what}: {equal}/{channels}, worst {worst}");
    }
    true
}

/// Each draw announces `unmeasured-fetch-blend` and stores the model's channels, a 32-bit float
/// target's exactly, another's within one code, on a device that fetches through the pixel
/// interlock (the module docs).
#[test]
fn a_blended_fetch_of_another_format_stores_the_generic_blend() {
    let kind = Degradation::from_precision_note(PrecisionNote::UnmeasuredFetchBlend);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = [kind].into_iter().collect();
    });
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name();
    let mut drawn = 0;
    for safe in [false, true] {
        let (library, vertex) = match safe {
            false => (library(&lane, "fbl_fetch_blend8"), "fbl8_vertex"),
            true => (library(&lane, "fbl_fetch_blend8_safe"), "fbl8_safe_vertex"),
        };
        let entries = match safe {
            false => [
                ("fbl8_over_loaded", true),
                ("fbl8_over_loaded_float", false),
            ],
            true => [
                ("fbl8_safe_loaded", true),
                ("fbl8_safe_loaded_float", false),
            ],
        };
        for format in FORMATS {
            for blend in [Blend::Over, Blend::Constant] {
                for entry in entries {
                    let fixture = (&library, vertex);
                    drawn +=
                        usize::from(checked((&lane, &messages), fixture, entry, format, blend));
                }
            }
        }
    }
    assert!(drawn > 0, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}

/// A blend of the second source, read from the function's second output (the owned fixture
/// `fbl_fetch_dual`), into a 32-bit float, a 16-bit float, a 16-bit unorm, an 8-bit unorm and an
/// 8-bit sRGB target: each draw announces `unmeasured-fetch-blend` and stores the model's channels,
/// as above.
#[test]
fn a_second_source_blend_of_a_fetched_attachment_stores_the_generic_blend() {
    let kind = Degradation::from_precision_note(PrecisionNote::UnmeasuredFetchBlend);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = [kind].into_iter().collect();
    });
    if !fetches(&lane.device) {
        return;
    }
    let device = lane.device.name();
    let library = library(&lane, "fbl_fetch_dual");
    let formats = [
        (PixelFormat::RGBA32Float, Layout::Float32),
        (PixelFormat::RGBA16Float, Layout::Half16),
        (PixelFormat::RGBA16Unorm, Layout::Unorm16),
        (PixelFormat::RGBA8Unorm, Layout::Unorm8),
        (PixelFormat::RGBA8UnormSRGB, Layout::Srgb8),
    ];
    let mut drawn = 0;
    for format in formats {
        for entry in [("fbd_dual_half", true), ("fbd_dual_float", false)] {
            let fixture = (&library, "fbd_vertex");
            drawn += usize::from(checked(
                (&lane, &messages),
                fixture,
                entry,
                format,
                Blend::Dual,
            ));
        }
    }
    assert!(drawn > 0, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}
