//! A wide-gamut client's render pipelines into `BGRA10_XR` and its `_sRGB` twin: what a draw
//! stores, compared code for code.
//!
//! A `CAMetalLayer` that asks for wide gamut hands its client `BGRA10_XR` drawables, and a client
//! such as Impeller then makes every pipeline over that format: four samples resolved into the
//! drawable, a stencil (or depth and stencil) attachment, and blending on by default. Three
//! families hold those shapes against Metal:
//!
//! - **The blended draws** (`xr-gamut-*`): a 32 x 32 target, cleared, then two full-screen draws of
//!   the owned fixture `rt6_samples`'s `half4` or `float4` fill, each blended over what the one
//!   before left, under premultiplied source-over (one, one minus the source alpha, for color and
//!   alpha) or straight alpha-over (the source alpha for color, one for alpha, each over one minus
//!   the source alpha). At one sample the target is read back; at four samples (the fill writes
//!   each sample its own value) the resolve texture is, which the pass resolves into at its store
//!   and does not keep the samples of, as a wide-gamut client's onscreen pass does. One more row
//!   adds a `Depth32Float_Stencil8` attachment the pass clears, under the four-sample source-over
//!   `half4` draw.
//! - **The unwritten attachment** (`xr-unwritten-*`): a pass whose first draw fills the target
//!   unblended and whose second is a pipeline that blends the attachment but writes no color: one
//!   without a fragment function, and one whose function writes depth alone. Metal leaves the
//!   attachment as the first draw left it. The padded `RGB10A2Unorm_sRGB` takes the pipeline
//!   without a fragment function too (not `R10Unorm_X6` or `RG10Unorm_X12`, whose texels Metal
//!   leaves in a fixed pattern after the pass, nor the bfloat formats, whose blend Metal's
//!   validation layer refuses: `OTHER_FORMATS`).
//! - **The masked write** (`xr-masked-*`): a first draw fills the target, and a second writes the
//!   other values under a mask of some of the channels (red; alpha; red, green and blue; blue and
//!   green), unblended, and under the mask of red, green and blue with source-over blending.
//!   `BGR10_XR` and its twin, whose three codes share one word, take red and blue-and-green
//!   unblended and red under source-over.
//!
//! The values are the family's own: each a `half` (so the `half4` and `float4` fills carry the
//! same value), an alpha on a grid of thirty-seconds in `[0, 1]` and each color its alpha times a
//! factor in `[-0.5, 1.5]` on a grid of 256ths, rounded to `2^-10`: premultiplied colors that
//! reach past the format's unit range on both sides, as extended-range content does. The first
//! draw's values come from seed 1, the second's from seed 2.
//!
//! Nothing here can fault a GPU: small targets of the case's own extent, full-screen triangles,
//! no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Handle, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, PassColor, PassDepth, PassStencil, Plane, ReadAt, RenderPass,
    RenderPipelineSpec, RenderStep, RenderWork, StageFunction, TargetSpec,
};
use crate::transcript::{Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use metal2vulkan_runtime::{Degradation, StoreAction};

use super::extended_range_targets::{
    fixture, fixture_inputs, half_bits, not_read, program, shader_refs, shape, texel_layout,
    Format, FILLS, FILL_FLOAT, FILL_HALF, FORMATS, VERTEX,
};

/// The fixture's fill that writes depth alone (`[[depth(any)]]`), reading a `float` a sample.
const FILL_DEPTH: &str = "rt6_fill_depth32float";

/// The target's side.
const SIDE: u32 = 32;

/// The clear color: within the unit range, alpha one half.
const CLEAR: [f64; 4] = [0.25, 0.5, 0.75, 0.5];

/// The two formats a wide-gamut layer asks for: `BGRA10_XR` and its `_sRGB` twin.
fn formats() -> [Format; 2] {
    [FORMATS[0], FORMATS[1]]
}

/// Premultiplied source-over, the client's default blend.
const OVER: Blend = Blend::added(F::One, F::OneMinusSourceAlpha);

/// Straight alpha-over.
const STRAIGHT: Blend = Blend::separate(
    (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
    (F::One, F::OneMinusSourceAlpha, O::Add),
);

// ---- The values ---------------------------------------------------------------------------------

/// A splitmix64 stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// `count` premultiplied values from `seed` (the module docs), red, green, blue, alpha.
fn values(seed: u64, count: usize) -> Vec<[f32; 4]> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let alpha = rng.below(33) as f32 / 32.0;
            let color = |rng: &mut Rng| {
                let factor = rng.below(513) as f32 / 256.0 - 0.5;
                (alpha * factor * 1024.0).round() / 1024.0
            };
            let (r, g, b) = (color(&mut rng), color(&mut rng), color(&mut rng));
            [r, g, b, alpha]
        })
        .collect()
}

/// The fill buffer's bytes of `values`: `half4`s where `half`, else `float4`s.
fn value_bytes(values: &[[f32; 4]], half: bool) -> Vec<u8> {
    match half {
        true => values
            .iter()
            .flatten()
            .flat_map(|&v| half_bits(v).to_le_bytes())
            .collect(),
        false => values
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

/// The fill steps of one draw: `pipeline`, its values buffer at fragment buffer 0 and the shape at
/// 1, and the full-screen triangle.
fn fill(pipeline: Handle, values: Handle, extent: Handle) -> [RenderStep; 4] {
    [
        RenderStep::Pipeline(pipeline),
        RenderStep::FragmentBuffer {
            buffer: values,
            offset: 0,
            index: 0,
        },
        RenderStep::FragmentBuffer {
            buffer: extent,
            offset: 0,
            index: 1,
        },
        RenderStep::triangle(),
    ]
}

/// The notices every case of the family may announce, each with the reason it cannot loosen a
/// code: `shared-in-system-memory` (the read-back buffer is `Shared`; a device without unified
/// memory announces it), the kept encoding of a blit through a `BGRA10_XR` texture Metal keeps
/// compressed (a draw stores canonical codes, so the blit moves the same bytes either way),
/// `half-output-fast-math-blend` (the fixture is a fast-math module whose `half4` fill a pipeline
/// stores), and `extended-range-bits-kept` (a draw in a pass over such a color; every texel here is
/// written by a Metal write first, so its bytes are canonical). Every value still compares
/// exactly.
fn notices(format: &Format) -> Vec<Allowance> {
    let mut named = vec![crate::replay::shared_memory_notice()];
    named.extend(super::compressed_blits::kept_encoding_notice(
        format.format.as_raw(),
    ));
    named.extend(
        [
            Degradation::HalfOutputFastMathBlend,
            Degradation::ExtendedRangeBitsKept,
        ]
        .into_iter()
        .map(|kind| crate::replay::notice_allowance(kind.code())),
    );
    named
}

/// Read `target` (one sample, `format`) back into a buffer after `work`, and record its texels.
fn read_back(
    side: &mut dyn Side,
    sites: &mut Sites,
    format: &Format,
    target: Handle,
    mut work: Vec<RenderWork>,
) -> Result<(), Abort> {
    let bytes = (SIDE * SIDE * format.texel) as usize;
    let out = made(sites, "buffer/texels", side.zeroed_buffer(bytes))?;
    work.push(RenderWork::ReadTexture {
        texture: target,
        into: out,
        offset: 0,
        row_pitch: SIDE * format.texel,
        size: [SIDE, SIDE],
        plane: Plane::Whole,
        at: ReadAt::BASE,
    });
    let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
    program(sites, "program", &outcome, &work);
    let texels = call(sites, "read-back", side.read_buffer(out, 0, bytes), |_| {
        Value::Absent
    })?;
    sites.push(Node::leaf(
        "texels",
        Observed::bytes(texels, texel_layout(format, [SIDE, SIDE])),
    ));
    Ok(())
}

// ---- The blended draws --------------------------------------------------------------------------

/// One case: two blended draws into one format at one sample count.
pub struct GamutDraw {
    name: &'static str,
    format: Format,
    blend: Blend,
    /// Whether the fill returns `half4` rather than `float4`.
    half: bool,
    samples: u32,
    /// Whether the pass and the pipeline have a `Depth32Float_Stencil8` attachment.
    depth_stencil: bool,
}

/// Every blended draw: each format under each blend, `half4` and `float4`, one sample and four,
/// and the four-sample source-over `half4` draw beside a depth and stencil attachment.
pub fn draws() -> Vec<GamutDraw> {
    const NAMES: [[&str; 9]; 2] = [
        [
            "xr-gamut-bgra10xr-over-half-x1",
            "xr-gamut-bgra10xr-over-float-x1",
            "xr-gamut-bgra10xr-over-half-x4",
            "xr-gamut-bgra10xr-over-float-x4",
            "xr-gamut-bgra10xr-straight-half-x1",
            "xr-gamut-bgra10xr-straight-float-x1",
            "xr-gamut-bgra10xr-straight-half-x4",
            "xr-gamut-bgra10xr-straight-float-x4",
            "xr-gamut-bgra10xr-over-half-x4-stencil",
        ],
        [
            "xr-gamut-bgra10xr-srgb-over-half-x1",
            "xr-gamut-bgra10xr-srgb-over-float-x1",
            "xr-gamut-bgra10xr-srgb-over-half-x4",
            "xr-gamut-bgra10xr-srgb-over-float-x4",
            "xr-gamut-bgra10xr-srgb-straight-half-x1",
            "xr-gamut-bgra10xr-srgb-straight-float-x1",
            "xr-gamut-bgra10xr-srgb-straight-half-x4",
            "xr-gamut-bgra10xr-srgb-straight-float-x4",
            "xr-gamut-bgra10xr-srgb-over-half-x4-stencil",
        ],
    ];
    let mut cases = Vec::new();
    for (format, names) in formats().into_iter().zip(NAMES) {
        let mut names = names.into_iter();
        for blend in [OVER, STRAIGHT] {
            for samples in [1, 4] {
                for half in [true, false] {
                    cases.push(GamutDraw {
                        name: names.next().expect("a name"),
                        format,
                        blend,
                        half,
                        samples,
                        depth_stencil: false,
                    });
                }
            }
        }
        cases.push(GamutDraw {
            name: names.next().expect("a name"),
            format,
            blend: OVER,
            half: true,
            samples: 4,
            depth_stencil: true,
        });
    }
    cases
}

impl GamutDraw {
    fn fill(&self) -> &'static str {
        match self.half {
            true => FILL_HALF,
            false => FILL_FLOAT,
        }
    }

    fn colors(&self) -> [Option<ColorState>; 1] {
        [Some(
            ColorState::new(self.format.format).blended(self.blend),
        )]
    }

    fn pipeline<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        let ds = self
            .depth_stencil
            .then_some(PixelFormat::Depth32FloatStencil8);
        RenderPipelineSpec::new(VERTEX, self.fill(), colors)
            .multisampled(self.samples)
            .depth_stencil(ds, ds)
    }

    fn target(&self) -> TargetSpec {
        match self.samples {
            1 => TargetSpec::read(self.format.format, [SIDE, SIDE]),
            n => TargetSpec::plain(self.format.format, [SIDE, SIDE]).multisampled(n),
        }
    }

    fn resolve(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [SIDE, SIDE])
    }

    fn depth(&self) -> TargetSpec {
        TargetSpec::plain(PixelFormat::Depth32FloatStencil8, [SIDE, SIDE])
            .multisampled(self.samples)
    }

    /// Draw `index`'s fill buffer.
    fn values(&self, index: u64) -> Vec<u8> {
        let count = (SIDE * SIDE * self.samples) as usize;
        value_bytes(&values(1 + index, count), self.half)
    }
}

impl Case for GamutDraw {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// Beside the family's notices, `extended-range-blends-stored-codes` (the second draw blends
    /// over the codes the first stored, where Metal's tile holds `half`s through the pass, on a
    /// device whose passes hold no `half` shadow, `framebuffer_fetch::holds_extended_shadows`:
    /// each venue of the suite holds it, in the pixel interlock or, on lavapipe, beside the input
    /// attachment), and at four samples what the pass's shape announces: `shader-resolve` and
    /// `extended-range-resolve` (no device's resolve averages the codes, so the runtime's own
    /// resolve does, `BGRA10_XR`'s alpha and an sRGB twin's color from the codes where Metal
    /// averages finer values), `dont-care-store` (the multisampled attachment of
    /// `MultisampleResolve`, and the depth and stencil stored `DontCare`, on a driver measured to
    /// keep one of Metal's answers), and on a device without multisampled storage images
    /// `sample-planes` and `fetch-through-scratch` (the interlocked blend reads and writes the
    /// samples as planes, through a scratch image the target's usage needs). And
    /// `render-pass-split` (`crate::replay::interlocked_split_notice`): the second draw blends in
    /// the pixel interlock over what the first wrote there. None loosens a code: every texel still
    /// compares exactly.
    fn allowances(&self) -> Vec<Allowance> {
        let mut named = notices(&self.format);
        named.push(crate::replay::notice_allowance(
            Degradation::ExtendedRangeBlendsStoredCodes.code(),
        ));
        named.push(crate::replay::interlocked_split_notice());
        if self.samples > 1 {
            named.extend(
                [
                    Degradation::ShaderResolve,
                    Degradation::ExtendedRangeResolve,
                    Degradation::DontCareStore,
                    Degradation::SamplePlanes,
                    Degradation::FetchThroughScratch,
                ]
                .into_iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
            );
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, self.fill()])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        let colors = self.colors();
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("blend", &self.blend.words()),
            Input::bytes("pipeline", &self.pipeline(&colors).words()),
            Input::bytes("target", &self.target().words()),
            Input::bytes(
                "clear",
                &CLEAR
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Input::bytes("buffer:values-0", &self.values(0)),
            Input::bytes("buffer:values-1", &self.values(1)),
            Input::bytes("buffer:shape", &shape(SIDE, self.samples)),
        ]);
        if self.samples > 1 {
            inputs.push(Input::bytes("resolve", &self.resolve().words()));
        }
        if self.depth_stencil {
            inputs.push(Input::bytes("depth-stencil", &self.depth().words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let colors = self.colors();
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &self.pipeline(&colors)),
        )?;
        let first = made(sites, "buffer/values-0", side.buffer(&self.values(0)))?;
        let second = made(sites, "buffer/values-1", side.buffer(&self.values(1)))?;
        let extent = made(
            sites,
            "buffer/shape",
            side.buffer(&shape(SIDE, self.samples)),
        )?;
        let target = made(sites, "target", side.target(&self.target()))?;
        let mut steps = Vec::new();
        steps.extend(fill(pipeline, first, extent));
        steps.extend(fill(pipeline, second, extent));
        let (color, read) = match self.samples {
            1 => (PassColor::cleared(target, CLEAR), target),
            _ => {
                let resolve = made(sites, "resolve", side.target(&self.resolve()))?;
                let color = PassColor::cleared(target, CLEAR)
                    .resolving(resolve)
                    .stored(StoreAction::MultisampleResolve);
                (color, resolve)
            }
        };
        let mut pass = RenderPass::clearing(vec![color]).drawing(steps);
        if self.depth_stencil {
            let depth = made(sites, "depth-stencil", side.target(&self.depth()))?;
            pass = pass
                .with_depth(PassDepth::cleared(depth, 1.0).stored(StoreAction::DontCare))
                .with_stencil(PassStencil::cleared(depth, 0).stored(StoreAction::DontCare));
        }
        read_back(
            side,
            sites,
            &self.format,
            read,
            vec![RenderWork::Pass(pass)],
        )
    }
}

// ---- The unwritten attachment -------------------------------------------------------------------

/// What the second pipeline of an unwritten case is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unwritten {
    /// No fragment function.
    NoFragment,
    /// A fragment function that writes depth alone.
    DepthOnly,
}

/// One case: a fill, then a blended pipeline that writes no color.
pub struct UnwrittenDraw {
    name: &'static str,
    format: Format,
    shape: Unwritten,
}

/// The padded 10-bit format an unwritten case also takes: Metal leaves an unwritten attachment as it
/// is whatever its format, this among them, where the runtime refused its blended attachment
/// written or not. Not `R10Unorm_X6` or `RG10Unorm_X12`: after this pass Metal's texels of either
/// are a fixed pattern on the M3 Max under macOS 27.0.1 (each row's first sixteen words `0xffff`,
/// the rest zero, in every run and under validation), neither the clear's nor the first draw's
/// codes, an answer no case can compare with. Not the bfloat formats: Metal's validation layer
/// refuses a pipeline that blends one ("the pixelformat MTLPixelFormatR16Bfloat for this render
/// target is not blendable"), which ends the recording's process.
const OTHER_FORMATS: [Format; 1] = [Format {
    slug: "rgb10a2unorm-srgb",
    format: PixelFormat::RGB10A2UnormSRGB,
    texel: 4,
    word: Element::U32,
    words: 1,
    copies: true,
}];

/// Each format under each shape, and the padded format without a fragment function.
pub fn unwritten() -> Vec<UnwrittenDraw> {
    let names = [
        (
            "xr-unwritten-bgra10xr-no-fragment",
            Unwritten::NoFragment,
            0,
        ),
        ("xr-unwritten-bgra10xr-depth-only", Unwritten::DepthOnly, 0),
        (
            "xr-unwritten-bgra10xr-srgb-no-fragment",
            Unwritten::NoFragment,
            1,
        ),
        (
            "xr-unwritten-bgra10xr-srgb-depth-only",
            Unwritten::DepthOnly,
            1,
        ),
    ];
    let others = ["xr-unwritten-rgb10a2unorm-srgb-no-fragment"];
    let mut cases: Vec<UnwrittenDraw> = names
        .into_iter()
        .map(|(name, shape, format)| UnwrittenDraw {
            name,
            format: formats()[format],
            shape,
        })
        .collect();
    cases.extend(
        others
            .into_iter()
            .zip(OTHER_FORMATS)
            .map(|(name, format)| UnwrittenDraw {
                name,
                format,
                shape: Unwritten::NoFragment,
            }),
    );
    cases
}

impl UnwrittenDraw {
    fn depth_format(&self) -> Option<PixelFormat> {
        (self.shape == Unwritten::DepthOnly).then_some(PixelFormat::Depth32Float)
    }

    fn plain_colors(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(self.format.format))]
    }

    fn blended_colors(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(self.format.format).blended(OVER))]
    }

    fn first<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FILL_HALF, colors).depth_stencil(self.depth_format(), None)
    }

    fn second<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        let fragment = match self.shape {
            Unwritten::NoFragment => None,
            Unwritten::DepthOnly => Some(StageFunction::named(FILL_DEPTH)),
        };
        RenderPipelineSpec::of(StageFunction::named(VERTEX), fragment, colors)
            .depth_stencil(self.depth_format(), None)
    }

    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [SIDE, SIDE])
    }

    fn depth(&self) -> TargetSpec {
        TargetSpec::plain(PixelFormat::Depth32Float, [SIDE, SIDE])
    }

    fn values(&self) -> Vec<u8> {
        value_bytes(&values(1, (SIDE * SIDE) as usize), true)
    }

    /// The depth fill's `float`s: one half a pixel.
    fn depths() -> Vec<u8> {
        (0..SIDE * SIDE)
            .flat_map(|_| 0.5f32.to_le_bytes())
            .collect()
    }
}

impl Case for UnwrittenDraw {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// Beside the family's notices, for the pipeline that writes depth alone, `dont-care-store`:
    /// the pass stores its depth `DontCare`, which a driver measured to keep one of Metal's
    /// answers (lavapipe) leaves to the device. It loosens no code.
    fn allowances(&self) -> Vec<Allowance> {
        let mut named = notices(&self.format);
        if self.shape == Unwritten::DepthOnly {
            named.push(crate::replay::notice_allowance(
                Degradation::DontCareStore.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut entries = vec![VERTEX, FILL_HALF];
        if self.shape == Unwritten::DepthOnly {
            entries.push(FILL_DEPTH);
        }
        shader_refs(FILLS, &entries)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        let (plain, blended) = (self.plain_colors(), self.blended_colors());
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("pipeline:first", &self.first(&plain).words()),
            Input::bytes("pipeline:second", &self.second(&blended).words()),
            Input::bytes("target", &self.target().words()),
            Input::bytes(
                "clear",
                &CLEAR
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Input::bytes("buffer:values", &self.values()),
            Input::bytes("buffer:shape", &shape(SIDE, 1)),
        ]);
        if self.shape == Unwritten::DepthOnly {
            inputs.push(Input::bytes("depth", &self.depth().words()));
            inputs.push(Input::bytes("buffer:depths", &Self::depths()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let (plain, blended) = (self.plain_colors(), self.blended_colors());
        let first = made(
            sites,
            "pipeline/first",
            side.render_pipeline(library, &self.first(&plain)),
        )?;
        let second = made(
            sites,
            "pipeline/second",
            side.render_pipeline(library, &self.second(&blended)),
        )?;
        let values = made(sites, "buffer/values", side.buffer(&self.values()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(SIDE, 1)))?;
        let target = made(sites, "target", side.target(&self.target()))?;
        let mut steps = Vec::new();
        steps.extend(fill(first, values, extent));
        let mut pass = RenderPass::clearing(vec![PassColor::cleared(target, CLEAR)]);
        match self.shape {
            Unwritten::NoFragment => {
                steps.extend([RenderStep::Pipeline(second), RenderStep::triangle()]);
            }
            Unwritten::DepthOnly => {
                let depths = made(sites, "buffer/depths", side.buffer(&Self::depths()))?;
                steps.extend(fill(second, depths, extent));
                let depth = made(sites, "depth", side.target(&self.depth()))?;
                pass =
                    pass.with_depth(PassDepth::cleared(depth, 1.0).stored(StoreAction::DontCare));
            }
        }
        let pass = pass.drawing(steps);
        read_back(
            side,
            sites,
            &self.format,
            target,
            vec![RenderWork::Pass(pass)],
        )
    }
}

// ---- The masked write ---------------------------------------------------------------------------

/// One case: a fill, then a fill of other values under a mask of some channels.
pub struct MaskedDraw {
    name: &'static str,
    format: Format,
    mask: ColorWriteMask,
    /// The second draw's blend, where it blends.
    blend: Option<Blend>,
}

/// Each format under each mask, and under the mask of the color channels with source-over.
pub fn masked() -> Vec<MaskedDraw> {
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    let shapes = [
        ("red", ColorWriteMask::RED, None),
        ("alpha", ColorWriteMask::ALPHA, None),
        ("rgb", rgb, None),
        (
            "blue-green",
            ColorWriteMask::BLUE.union(ColorWriteMask::GREEN),
            None,
        ),
        ("rgb-over", rgb, Some(OVER)),
    ];
    const NAMES: [[&str; 5]; 2] = [
        [
            "xr-masked-bgra10xr-red",
            "xr-masked-bgra10xr-alpha",
            "xr-masked-bgra10xr-rgb",
            "xr-masked-bgra10xr-blue-green",
            "xr-masked-bgra10xr-rgb-over",
        ],
        [
            "xr-masked-bgra10xr-srgb-red",
            "xr-masked-bgra10xr-srgb-alpha",
            "xr-masked-bgra10xr-srgb-rgb",
            "xr-masked-bgra10xr-srgb-blue-green",
            "xr-masked-bgra10xr-srgb-rgb-over",
        ],
    ];
    let mut cases = Vec::new();
    for (format, names) in formats().into_iter().zip(NAMES) {
        for ((_, mask, blend), name) in shapes.into_iter().zip(names) {
            cases.push(MaskedDraw {
                name,
                format,
                mask,
                blend,
            });
        }
    }
    // `BGR10_XR` and its twin hold the three color codes in one word: a mask of some of them.
    let word_shapes = [
        ("red", ColorWriteMask::RED, None),
        (
            "blue-green",
            ColorWriteMask::BLUE.union(ColorWriteMask::GREEN),
            None,
        ),
        ("red-over", ColorWriteMask::RED, Some(OVER)),
    ];
    const WORD_NAMES: [[&str; 3]; 2] = [
        [
            "xr-masked-bgr10xr-red",
            "xr-masked-bgr10xr-blue-green",
            "xr-masked-bgr10xr-red-over",
        ],
        [
            "xr-masked-bgr10xr-srgb-red",
            "xr-masked-bgr10xr-srgb-blue-green",
            "xr-masked-bgr10xr-srgb-red-over",
        ],
    ];
    for (format, names) in [FORMATS[2], FORMATS[3]].into_iter().zip(WORD_NAMES) {
        for ((_, mask, blend), name) in word_shapes.into_iter().zip(names) {
            cases.push(MaskedDraw {
                name,
                format,
                mask,
                blend,
            });
        }
    }
    cases
}

impl MaskedDraw {
    fn plain_colors(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(self.format.format))]
    }

    fn masked_colors(&self) -> [Option<ColorState>; 1] {
        let state = ColorState::new(self.format.format).masked(self.mask);
        [Some(match self.blend {
            Some(blend) => state.blended(blend),
            None => state,
        })]
    }

    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [SIDE, SIDE])
    }

    fn values(index: u64) -> Vec<u8> {
        value_bytes(&values(1 + index, (SIDE * SIDE) as usize), true)
    }
}

impl Case for MaskedDraw {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// Beside the family's notices, for the blended write, `extended-range-blends-stored-codes`:
    /// it blends over the codes the first draw stored on a device whose passes hold no `half`
    /// shadow (`framebuffer_fetch::holds_extended_shadows`); on lavapipe the shadow is held beside
    /// the input attachment. And `render-pass-split` (`crate::replay::interlocked_split_notice`):
    /// a blended write, and a masked one into `BGR10_XR`'s shared word, reads and writes in the
    /// pixel interlock what the first draw wrote there. It loosens no code.
    fn allowances(&self) -> Vec<Allowance> {
        let mut named = notices(&self.format);
        if self.blend.is_some() {
            named.push(crate::replay::notice_allowance(
                Degradation::ExtendedRangeBlendsStoredCodes.code(),
            ));
        }
        let shared_word = matches!(
            self.format.format,
            PixelFormat::BGR10XR | PixelFormat::BGR10XRSRGB
        );
        if self.blend.is_some() || shared_word {
            named.push(crate::replay::interlocked_split_notice());
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, FILL_HALF])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        let (plain, masked) = (self.plain_colors(), self.masked_colors());
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes(
                "pipeline:first",
                &RenderPipelineSpec::new(VERTEX, FILL_HALF, &plain).words(),
            ),
            Input::bytes(
                "pipeline:second",
                &RenderPipelineSpec::new(VERTEX, FILL_HALF, &masked).words(),
            ),
            Input::bytes("target", &self.target().words()),
            Input::bytes(
                "clear",
                &CLEAR
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Input::bytes("buffer:values-0", &Self::values(0)),
            Input::bytes("buffer:values-1", &Self::values(1)),
            Input::bytes("buffer:shape", &shape(SIDE, 1)),
        ]);
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let (plain, masked) = (self.plain_colors(), self.masked_colors());
        let first = made(
            sites,
            "pipeline/first",
            side.render_pipeline(library, &RenderPipelineSpec::new(VERTEX, FILL_HALF, &plain)),
        )?;
        let second = made(
            sites,
            "pipeline/second",
            side.render_pipeline(
                library,
                &RenderPipelineSpec::new(VERTEX, FILL_HALF, &masked),
            ),
        )?;
        let first_values = made(sites, "buffer/values-0", side.buffer(&Self::values(0)))?;
        let second_values = made(sites, "buffer/values-1", side.buffer(&Self::values(1)))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(SIDE, 1)))?;
        let target = made(sites, "target", side.target(&self.target()))?;
        let mut steps = Vec::new();
        steps.extend(fill(first, first_values, extent));
        steps.extend(fill(second, second_values, extent));
        let pass = RenderPass::clearing(vec![PassColor::cleared(target, CLEAR)]).drawing(steps);
        read_back(
            side,
            sites,
            &self.format,
            target,
            vec![RenderWork::Pass(pass)],
        )
    }
}
