//! What a pass sees when it **samples a texture an earlier pass of the same command buffer drew**.
//!
//! Every other render family in the suite asks about one pass: it clears, it draws, and the host
//! reads the target back. A program does not look like that. It renders into a texture, and then a
//! later pass reads that texture through a sampler without the host ever seeing the bytes in
//! between. Three things stand between the two passes, and **no case in the suite asks Metal about
//! any of them**:
//!
//! - the target's move from "a pass is writing it" to "a fragment function is sampling it", which
//!   on our side is an image-layout transition and on Metal's side is nothing a program states;
//! - the ordering between the two passes, which Metal tracks for a resource a command buffer uses
//!   twice and which our side must turn into a barrier;
//! - what the sampler sees of an image the same command buffer has just rendered.
//!
//! So each case here draws a known pattern into a **private** target, samples that target in a
//! second pass of the **same command buffer**, and reads both targets back at the end. Reading the
//! first target as well as the second is what makes a difference diagnosable: if the sampled
//! target is wrong the case says whether the first pass drew the wrong thing or the second pass
//! sampled the right thing wrongly.
//!
//! # The pattern, and why it is bands
//!
//! The first pass clears the target to a distinctive code and then draws **four vertical bands**,
//! one draw each, each band a colour of its own out of one buffer at four offsets, and then a
//! **fifth draw of a fifth colour over half the clip-space height**. Vertical bands because the
//! mapping from a clip-space `x` to a pixel column is unambiguous, so their places are a fact about
//! the geometry rather than about which way Metal hands its `y` axis; five different colours
//! because a sample that reaches the wrong texel then reads a different value rather than passing
//! by coincidence; and a clear nothing covers nowhere, so a draw that did not happen at all shows
//! as the clear code rather than as a colour.
//!
//! The fifth draw is what makes the pattern **distinguishable by position on both axes**, and it is
//! not decoration. Four column bands alone are the same in every row, so a sample that read the
//! right column of the right texture at the wrong row would be invisible -- which is exactly the
//! bug a chain of passes over shared handles exists to catch. With the half-height overdraw, the
//! exact row's "the second target equals the first" and the magnified row's "each source texel is
//! doubled on both axes" are checks in `y` as well as in `x`. Which half of the target the overdraw
//! lands in is Metal's answer and this module does not state it: the cross-check asserts that the
//! rows fall into two contiguous halves, one of them the overdraw's, and tests the rest relatively.
//!
//! Every colour channel is a whole code over 255, so `RGBA8Unorm` stores exactly what the fragment
//! function returned and the texels are exact on this venue.
//!
//! # Three rows, three different questions of the sampler
//!
//! - **`render-then-sample-exact`** samples at the texel centres: `tob_sample` divides the pixel's
//!   own position by the sampled texture's extent, so with the two targets of one extent every
//!   sample lands on a texel centre and the second target must come back **equal to the first**.
//!   The filter cannot change that answer, which is the point: it is the barrier and the layout,
//!   and nothing else, that this row can fail on.
//! - **`render-then-sample-magnified-nearest`** samples an 8 by 8 target over a 16 by 16 one
//!   through `tgp_layer`, whose texture coordinates run across the drawn rectangle rather than
//!   being derived from the sampled texture, so each source texel covers four pixels. Nearest
//!   filtering makes that an exact doubling.
//! - **`render-then-sample-magnified-linear`** is the same geometry with a linear filter, so the
//!   hardware's filter really runs between the texels of an image the same command buffer drew.
//!   Its answer is Metal's and is not computed here; what this module's tests hold is that the
//!   filter ran at all (the row differs from the nearest row) and that every channel lies between
//!   the source texels it is a blend of.
//!
//! # What the rows announce, and on which venue
//!
//! Every texel of every row compared exactly on both venues measured here, so the only difference
//! any row has ever shown is a degrade notice of ours, announced where Metal announces nothing.
//! Measured:
//!
//! - MoltenVK on this M3 Max: the two `tgp_composite` rows announce
//!   `half-output-fast-math-blend`, for the `half` colour output reaching a storing attachment.
//!   The exact row announces nothing and reads exact.
//! - lavapipe: every row announces `shader-color-conversion`, for the normalized store our side
//!   rounds in the shader where the device's own conversion is not Metal's. Beside it, the
//!   `tgp_composite` rows announce `half-output-fast-math-blend` and `device-interpolation`, the
//!   second for the texture coordinate `tgp_rect` hands across the rectangle on a device not
//!   measured to interpolate as Metal does, and the exact row announces `device-arithmetic` for
//!   the division `tob_sample` does.
//!
//! Each row **declares every code it can announce**, one declaration a code, because the
//! comparison reads a declaration per code: a code a row announces and declares nowhere is a
//! divergence for an undeclared notice, so two rows declare three codes between the venues and the
//! exact row declares two. A declaration loosens no value; what it says is that the row's shape
//! can reach that note, which this table is the measurement of. The interpolation note is worth
//! reading twice over: lavapipe interpolates a varying otherwise than Metal, and these rows'
//! sampled texels still came back exactly Metal's, so what the row declares is the note and not a
//! difference in any value.
//!
//! Nothing here can fault a GPU: two passes, four and one draws, targets of at most 16 by 16, and
//! no loop in any function either pass runs.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The format both targets of every row carry. Its codes survive a store unchanged, so every texel
/// of every row is exact.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 4;

/// The bands the first pass draws, which is also how many colours the buffer holds.
const BANDS: u32 = 4;

/// The clear the first pass loads its target with: a code no band's colour holds, so a band that
/// was never drawn reads as this rather than as another band's colour.
const SOURCE_CLEAR: u8 = 7;

/// The clear the second pass loads its target with, told apart from the first pass's so a sampled
/// target that kept its clear cannot be read as the source's.
const TARGET_CLEAR: u8 = 3;

/// The four colours the first pass draws, a whole code a channel so `RGBA8Unorm` stores exactly
/// what the fragment function returned, and every channel of every band different from every
/// other.
const COLORS: [[u8; 4]; BANDS as usize] = [
    [16, 32, 48, 200],
    [64, 80, 96, 210],
    [112, 128, 144, 220],
    [160, 176, 192, 230],
];

/// The fifth colour, which the fifth draw puts over half the target's height, so the pattern is
/// distinguishable by **row** as well as by column and a sample that read the wrong row cannot
/// pass. Its channels differ from every band's and from both clears, as the bands' do from each
/// other.
const OVERDRAW: [u8; 4] = [208, 224, 240, 240];

/// The rectangle the fifth draw covers: half the clip-space height, over every column.
///
/// Which half of the target that is -- the first rows or the last -- is Metal's answer, and nothing
/// here states it ([`tests::metal_drew_the_bands_and_then_the_half_overdraw`]).
const HALF: [f32; 4] = [-1.0, -1.0, 1.0, 0.0];

/// The owned fixture whose `tob_rect` draws a rectangle from a buffer's two corners, whose
/// `tob_solid` returns a colour from a buffer, and whose `tob_sample` samples texture 0 at the
/// pixel's own position over the sampled texture's extent.
const TOB: &str = "tob_renders";

/// The owned fixture whose `tgp_rect` draws a rectangle with its texture coordinates **across the
/// rectangle**, and whose `tgp_layer` writes the sample of texture 0 into colour 0.
const TGP: &str = "tgp_composite";

/// The vertex and fragment functions that fill the source target: the rectangle and the solid
/// colour, both of [`TOB`].
const FILL: Functions = Functions {
    fixture: TOB,
    vertex: "tob_rect",
    fragment: "tob_solid",
    divides_a_float: false,
    interpolates_a_varying: false,
    half_output: false,
};

/// A pipeline's two functions and the fixture they come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Functions {
    fixture: &'static str,
    vertex: &'static str,
    fragment: &'static str,
    /// Whether the fragment function divides a `float`: `tob_sample` divides the pixel's position
    /// by the sampled texture's extent, and nothing else here does.
    ///
    /// This is the shape that decides the row's `device-arithmetic` declaration, and nothing else.
    divides_a_float: bool,
    /// Whether the fragment function reads a varying the device interpolates: `tgp_layer` reads
    /// the texture coordinate `tgp_rect` hands across the rectangle, while `tob_sample` derives
    /// its coordinate from the pixel's own position and reads no interpolated varying.
    ///
    /// This is the shape that decides the row's `device-interpolation` declaration, and nothing
    /// else.
    interpolates_a_varying: bool,
    /// Whether the fragment function returns a `half` colour output rather than a `float` one.
    ///
    /// This is the shape that decides one of the row's notice declarations and nothing else: our
    /// side records `half-output-fast-math-blend` where a `half` colour output reaches an
    /// attachment that stores a channel out of a module naming no math mode
    /// (`metal2vulkan::pipeline::color_writes::pipeline_writes`), which is a fact about the
    /// function and the attachment and not about any device.
    half_output: bool,
}

/// What one row asks: the two extents, the functions the sampling pass runs, and its filter.
#[derive(Clone, Copy)]
struct Subject {
    name: &'static str,
    /// The side, in texels, of the target the first pass draws into and the second pass samples.
    source_side: u32,
    /// The side, in texels, of the target the second pass draws into.
    target_side: u32,
    /// The functions the sampling pass's pipeline is made of.
    sample: Functions,
    /// Both of the sampler's filters.
    filter: SamplerMinMagFilter,
}

/// The three rows, in the order the report lists them.
const SUBJECTS: &[Subject] = &[
    Subject {
        name: "render-then-sample-exact",
        source_side: 16,
        target_side: 16,
        sample: Functions {
            fixture: TOB,
            vertex: "tob_rect",
            fragment: "tob_sample",
            divides_a_float: true,
            interpolates_a_varying: false,
            half_output: false,
        },
        filter: SamplerMinMagFilter::Nearest,
    },
    Subject {
        name: "render-then-sample-magnified-nearest",
        source_side: 8,
        target_side: 16,
        sample: Functions {
            fixture: TGP,
            vertex: "tgp_rect",
            fragment: "tgp_layer",
            divides_a_float: false,
            interpolates_a_varying: true,
            half_output: true,
        },
        filter: SamplerMinMagFilter::Nearest,
    },
    Subject {
        name: "render-then-sample-magnified-linear",
        source_side: 8,
        target_side: 16,
        sample: Functions {
            fixture: TGP,
            vertex: "tgp_rect",
            fragment: "tgp_layer",
            divides_a_float: false,
            interpolates_a_varying: true,
            half_output: true,
        },
        filter: SamplerMinMagFilter::Linear,
    },
];

/// One row of the family.
pub struct RenderChain {
    subject: Subject,
}

/// Every row of the family, in the order the report lists them.
pub fn all() -> Vec<RenderChain> {
    SUBJECTS
        .iter()
        .map(|subject| RenderChain { subject: *subject })
        .collect()
}

/// Every row as the case set holds it, so the one line this module adds to
/// [`crate::cases::all`] stays one line.
pub fn boxed() -> Vec<Box<dyn Case>> {
    all()
        .into_iter()
        .map(|case| Box::new(case) as Box<dyn Case>)
        .collect()
}

/// Band `at` of `BANDS`, as `tob_rect` and `tgp_rect` take a rectangle: the two clip-space corners
/// `(x, y)` and `(z, w)`, over the whole height.
///
/// A band's columns are the pixels whose centre lies inside it, which for a side that `BANDS`
/// divides is exactly `side / BANDS` of them and never a column of two bands: the band edge falls
/// between two pixel centres rather than on one.
fn band(at: u32) -> [f32; 4] {
    let edge = |step: u32| 2.0 * step as f32 / BANDS as f32 - 1.0;
    [edge(at), -1.0, edge(at + 1), 1.0]
}

/// The whole target, as a rectangle.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// A rectangle's bytes, as the vertex function's `constant float4 &` binding takes them.
fn rect_bytes(rect: [f32; 4]) -> Vec<u8> {
    rect.iter().flat_map(|edge| edge.to_le_bytes()).collect()
}

/// The buffer the solid fragment function reads its colour out of: the four band colours and the
/// overdraw's, as `float4`s, each channel a whole code over 255.
fn colors_buffer() -> Vec<u8> {
    COLORS
        .iter()
        .chain(std::iter::once(&OVERDRAW))
        .flat_map(|color| {
            color
                .iter()
                .flat_map(|code| (f32::from(*code) / 255.0).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The one colour attachment both pipelines of a row carry.
const COLOR_STATE: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The draw both rectangle functions take: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The layout a target of `side` texels a side is read under: one row of the layout a row of
/// texels and one element a channel, so a diff names the pixel and the channel.
fn layout(side: u32) -> ByteLayout {
    let row_pitch = TEXEL * side;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [side, side, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row_pitch,
        row_pitch,
        rows: side,
        image_pitch: row_pitch * side,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl Subject {
    /// Bytes the source target's whole level 0 takes, which is also where the sampled target's
    /// bytes start in the read-back buffer.
    fn source_bytes(&self) -> u32 {
        TEXEL * self.source_side * self.source_side
    }

    /// Bytes the sampled target's whole level 0 takes.
    fn target_bytes(&self) -> u32 {
        TEXEL * self.target_side * self.target_side
    }

    /// The fixtures the row loads, in the order it loads them: one where both pipelines come from
    /// the same fixture, two where they do not.
    fn fixtures(&self) -> Vec<&'static str> {
        let mut named = vec![FILL.fixture];
        if self.sample.fixture != FILL.fixture {
            named.push(self.sample.fixture);
        }
        named
    }

    /// The sampler state the second pass binds: the row's filter on both axes, clamped to the
    /// edge, which is Metal's own default address mode.
    fn sampler(&self) -> SamplerSpec {
        SamplerSpec::filtered(self.filter)
    }

    /// The pipeline of `functions` over the one colour attachment.
    fn pipeline(functions: Functions) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::new(functions.vertex, functions.fragment, &COLOR_STATE)
    }
}

impl Case for RenderChain {
    fn name(&self) -> &'static str {
        self.subject.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No texel is loosened: whole codes over 255 in `RGBA8Unorm` are an exact answer on every
        // device, and a difference in one is what this family exists to report. Every entry is a
        // **notice declaration** on the notices site, which loosens no value; Metal announces
        // nothing, so each notice of ours has to be declared or the row is divergent at the notices
        // site for an undeclared code.
        //
        // **One declaration a code the row can announce, not one a row.** A declaration fires on
        // its own code, and a code the case declares nowhere is a divergence however much prose
        // describes it, so a row that can announce two codes names both. Which codes those are
        // follows the row's shape, and each is one a run was measured announcing for this very row:
        //
        // - `shader-color-conversion`, for every row, because every row's fragment stage stores a
        //   normalized channel, which our side rounds in the shader on a device whose own
        //   conversion is not Metal's. Measured announced by all three rows on lavapipe, and by
        //   none of them on MoltenVK on this M3 Max, where Metal's own conversion runs.
        // - `half-output-fast-math-blend`, for the two rows whose fragment stage returns a `half`
        //   colour output ([`Functions::half_output`]). Our side records it where such an output
        //   reaches an attachment that stores a channel out of a module naming no math mode, which
        //   `tgp_composite` does. Measured announced by those two rows on both venues.
        // - `device-arithmetic`, for the row whose fragment stage divides a `float`
        //   ([`Functions::divides_a_float`]): `tob_sample` divides the pixel's position by the
        //   sampled texture's extent, and our side records the note where such arithmetic takes
        //   Vulkan's precision bounds rather than correct rounding. Measured announced by that row
        //   on lavapipe.
        // - `device-interpolation`, for the two rows whose fragment stage reads a varying the
        //   device interpolates ([`Functions::interpolates_a_varying`]): `tgp_rect` hands the
        //   texture coordinate across the rectangle, and our side records the note on a device not
        //   measured to interpolate as Metal does. Measured announced by those two rows on
        //   lavapipe, where their texels still compared exactly: the interpolation the note covers
        //   did not move a sampled texel of this geometry.
        //
        // Every value of every row compared **exactly** on both venues, which is the evidence that
        // these declarations loosen nothing: the whole difference they cover is the announcement.
        // - `shared-in-system-memory`, for every row, because the buffer the chain reads its
        //   target back through is `Shared`, which a device with no unified memory keeps in
        //   system memory because that is the only memory both sides can reach. It belongs to
        //   the venue's memory model rather than to a chain's shape.
        let mut named = vec![
            crate::replay::notice_allowance(Degradation::ShaderColorConversion.code()),
            crate::replay::shared_memory_notice(),
        ];
        if self.subject.sample.half_output {
            named.push(crate::replay::notice_allowance(
                Degradation::HalfOutputFastMathBlend.code(),
            ));
        }
        if self.subject.sample.divides_a_float {
            named.push(crate::replay::notice_allowance(
                Degradation::DeviceArithmetic.code(),
            ));
        }
        if self.subject.sample.interpolates_a_varying {
            named.push(crate::replay::notice_allowance(
                Degradation::DeviceInterpolation.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for name in self.subject.fixtures() {
            let fixture = Fixture::new(name);
            let entry = match name {
                _ if name == FILL.fixture && name == self.subject.sample.fixture => format!(
                    "{}+{}+{}",
                    FILL.vertex, FILL.fragment, self.subject.sample.fragment
                ),
                _ if name == FILL.fixture => format!("{}+{}", FILL.vertex, FILL.fragment),
                _ => format!(
                    "{}+{}",
                    self.subject.sample.vertex, self.subject.sample.fragment
                ),
            };
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{name}{extension}"),
                    entry: entry.clone(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let subject = &self.subject;
        let mut inputs = Vec::new();
        for name in subject.fixtures() {
            let fixture = Fixture::new(name);
            inputs.push(Input {
                name: format!("air:{name}"),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            });
            inputs.push(Input {
                name: format!("air-typed:{name}"),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            });
        }
        inputs.push(Input::bytes(
            "fill-pipeline",
            &Subject::pipeline(FILL).words(),
        ));
        inputs.push(Input::bytes(
            "sample-pipeline",
            &Subject::pipeline(subject.sample).words(),
        ));
        inputs.push(Input::bytes(
            "source-target",
            &TargetSpec::read(FORMAT, [subject.source_side; 2]).words(),
        ));
        inputs.push(Input::bytes(
            "target",
            &TargetSpec::read(FORMAT, [subject.target_side; 2]).words(),
        ));
        inputs.push(Input::bytes("sampler", &subject.sampler().words()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        let mut rects: Vec<u8> = (0..BANDS).flat_map(|at| rect_bytes(band(at))).collect();
        rects.extend(rect_bytes(HALF));
        rects.extend(rect_bytes(WHOLE));
        inputs.push(Input::bytes("rects", &rects));
        inputs.push(Input::bytes("clears", &[SOURCE_CLEAR, TARGET_CLEAR, 0, 0]));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let subject = &self.subject;
        let mut libraries: Vec<(&'static str, Handle)> = Vec::new();
        for name in subject.fixtures() {
            let loaded = Fixture::new(name).loaded().map_err(|why| Abort {
                at: format!("library:{name}"),
                failure: Failure::named("FixtureNotRead", why),
            })?;
            let library = made(
                sites,
                &format!("library:{name}"),
                side.library(&loaded.source()),
            )?;
            libraries.push((name, library));
        }
        let library_of = |named: &'static str| -> Handle {
            libraries
                .iter()
                .find(|(name, _)| *name == named)
                .map(|(_, library)| *library)
                .expect("the row loads every fixture its pipelines name")
        };
        let fill_state = made(
            sites,
            "fill-pipeline",
            side.render_pipeline(library_of(FILL.fixture), &Subject::pipeline(FILL)),
        )?;
        let sample_state = made(
            sites,
            "sample-pipeline",
            side.render_pipeline(
                library_of(subject.sample.fixture),
                &Subject::pipeline(subject.sample),
            ),
        )?;
        let source = made(
            sites,
            "source-target",
            side.target(&TargetSpec::read(FORMAT, [subject.source_side; 2])),
        )?;
        let target = made(
            sites,
            "target",
            side.target(&TargetSpec::read(FORMAT, [subject.target_side; 2])),
        )?;
        let sampler = made(sites, "sampler", side.sampler(&subject.sampler()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer((subject.source_bytes() + subject.target_bytes()) as usize),
        )?;

        // The first pass fills the source target, band by band: one pipeline, then a rectangle and
        // a colour offset a draw.
        let mut fill = vec![RenderStep::Pipeline(fill_state)];
        for at in 0..=BANDS {
            // The bands, and then the overdraw over half the height, whose colour is the one after
            // the bands' in the same buffer.
            let rect = if at < BANDS { band(at) } else { HALF };
            fill.push(RenderStep::VertexBytes {
                bytes: rect_bytes(rect),
                index: 0,
            });
            fill.push(RenderStep::FragmentBuffer {
                buffer: colors,
                offset: u64::from(at) * 16,
                index: 0,
            });
            fill.push(RECT_DRAW);
        }
        // The second pass samples the target the first one drew, in the same command buffer and
        // with nothing of the host in between: the whole target, one draw.
        let sample = vec![
            RenderStep::Pipeline(sample_state),
            RenderStep::FragmentTexture {
                texture: source,
                index: 0,
            },
            RenderStep::FragmentSampler { sampler, index: 0 },
            RenderStep::VertexBytes {
                bytes: rect_bytes(WHOLE),
                index: 0,
            },
            RECT_DRAW,
        ];
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(source, cleared(SOURCE_CLEAR))])
                    .drawing(fill),
            ),
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, cleared(TARGET_CLEAR))])
                    .drawing(sample),
            ),
            RenderWork::ReadTexture {
                texture: source,
                into: read_back,
                offset: 0,
                row_pitch: TEXEL * subject.source_side,
                size: [subject.source_side; 2],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
            RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: u64::from(subject.source_bytes()),
                row_pitch: TEXEL * subject.target_side,
                size: [subject.target_side; 2],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            },
        ];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        // What each piece of the command buffer answered. Metal's encoder calls answer nothing, so
        // a piece of ours that refused compares against Metal answering and reads as a refusal
        // rather than as a divergence in the texels.
        let names = ["fill-pass", "sample-pass", "read-source", "read-target"];
        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, what) in names.iter().enumerate() {
            pieces = pieces.with(Node::leaf(
                *what,
                Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        sites.push(pieces);
        if let Some((at, failure)) = names
            .iter()
            .enumerate()
            .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
        {
            return Err(Abort {
                at: format!("pieces/{at}"),
                failure: failure.clone(),
            });
        }
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(
                read_back,
                0,
                (subject.source_bytes() + subject.target_bytes()) as usize,
            ),
            |_| Value::Absent,
        )?;
        let split = subject.source_bytes() as usize;
        sites.push(Node::branch("drawn", Vec::new()).with(Node::leaf(
            "texels",
            Observed::bytes(bytes[..split].to_vec(), layout(subject.source_side)),
        )));
        sites.push(Node::branch("sampled", Vec::new()).with(Node::leaf(
            "texels",
            Observed::bytes(bytes[split..].to_vec(), layout(subject.target_side)),
        )));
        Ok(())
    }
}

/// The clear colour whose every channel is the code `code` over 255, as Metal's clear takes it.
fn cleared(code: u8) -> [f64; 4] {
    [f64::from(code) / 255.0; 4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `name`'s recorded Metal answer, or `None` where nothing has recorded it.
    ///
    /// A row of this family is settled -- two passes, a fixed geometry, no race -- so its runs fold
    /// into one outcome and a recording with more than one is itself a finding.
    fn only_outcome(name: &str) -> Option<Transcript> {
        let recording = crate::recording::read(name).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The bytes the site at `path` of `transcript` holds.
    fn bytes(transcript: &Transcript, path: &str) -> Vec<u8> {
        let mut node = &transcript.root;
        for name in path.split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("no site {path} (stopped at {name})"));
        }
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{path} holds no bytes");
        };
        bytes.clone()
    }

    /// Texel `(x, y)` of a `side`-wide target's bytes.
    fn texel(bytes: &[u8], side: u32, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * side + x) * TEXEL) as usize;
        [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
    }

    /// The bookkeeping of the family holds with no device: distinct names, a geometry whose bands
    /// tile each source extent, and layouts that cover exactly the bytes each read answers.
    #[test]
    fn the_rows_name_distinct_questions_over_a_geometry_that_tiles() {
        let named: BTreeSet<&str> = SUBJECTS.iter().map(|subject| subject.name).collect();
        assert_eq!(named.len(), SUBJECTS.len());
        // Every drawn colour, the overdraw's included, differs from every other's in **every
        // channel**, and from both clears, so a sample that reached the wrong texel cannot read as
        // the right one: a copy or a sample that moved the right number of bytes from the wrong
        // place is invisible unless the bytes differ by position, which is why this is channel by
        // channel rather than whole-value.
        let drawn: Vec<[u8; 4]> = COLORS.iter().copied().chain([OVERDRAW]).collect();
        for (at, color) in drawn.iter().enumerate() {
            for channel in color {
                assert_ne!(*channel, SOURCE_CLEAR, "colour {at}");
                assert_ne!(*channel, TARGET_CLEAR, "colour {at}");
            }
            for other in &drawn[at + 1..] {
                for (channel, (ours, theirs)) in color.iter().zip(other).enumerate() {
                    assert_ne!(ours, theirs, "colour {at}, channel {channel}");
                }
            }
        }
        assert_ne!(SOURCE_CLEAR, TARGET_CLEAR);
        // And the overdraw covers exactly half the clip-space height over every column, which is
        // what makes the pattern vary by row.
        assert_eq!(HALF[0], -1.0);
        assert_eq!(HALF[2], 1.0);
        assert_eq!(HALF[3] - HALF[1], 1.0);
        for subject in SUBJECTS {
            assert_eq!(
                subject.source_side % BANDS,
                0,
                "{}: the bands do not tile the source",
                subject.name
            );
            assert_eq!(
                layout(subject.source_side).extent(),
                subject.source_bytes() as usize,
                "{}",
                subject.name
            );
            assert_eq!(
                layout(subject.target_side).extent(),
                subject.target_bytes() as usize,
                "{}",
                subject.name
            );
            // Each row's two pipelines are two different pipelines, so a recording cannot be read
            // for the other one.
            assert_ne!(
                Subject::pipeline(FILL).words(),
                Subject::pipeline(subject.sample).words(),
                "{}",
                subject.name
            );
        }
        // The two clip-space bands at the ends reach the ends of the target.
        assert_eq!(band(0)[0], -1.0);
        assert_eq!(band(BANDS - 1)[2], 1.0);
    }

    /// Every allowance of every row is a notice declaration on the notices site, and the codes a
    /// row declares are exactly the ones its own shape can announce.
    ///
    /// Both halves matter. That each is a notice declaration on the notices site is the family's
    /// whole tolerance story: a row that ever loosened a texel would fail here, and the texels are
    /// the field the family exists to compare. That the set follows the shape is the comparison's
    /// own rule read in both directions -- a code the row announces and declares nowhere is a
    /// divergence for an undeclared notice, and a code it declares but cannot announce is an
    /// allowance standing behind nothing.
    #[test]
    fn every_allowance_is_a_notice_declaration_chosen_from_the_rows_shape() {
        use crate::allowance::Justification;
        for case in all() {
            let codes: Vec<String> = case
                .allowances()
                .iter()
                .map(|allowance| {
                    assert_eq!(
                        allowance.path,
                        format!("case/{}/*", crate::replay::NOTICES_SITE),
                        "{}: an allowance outside the notices site",
                        case.name()
                    );
                    match &allowance.justified_by {
                        Justification::Degradation(code) => code.clone(),
                        other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                    }
                })
                .collect();
            // The whole set, one declaration a code the row can announce: a code the row
            // announces and declares nowhere is a divergence, and a code it declares but cannot
            // announce is an allowance standing behind nothing, so the set is stated exactly.
            let mut wanted = vec![
                Degradation::ShaderColorConversion.code().to_owned(),
                Degradation::SharedInSystemMemory.code().to_owned(),
            ];
            if case.subject.sample.half_output {
                wanted.push(Degradation::HalfOutputFastMathBlend.code().to_owned());
            }
            if case.subject.sample.divides_a_float {
                wanted.push(Degradation::DeviceArithmetic.code().to_owned());
            }
            if case.subject.sample.interpolates_a_varying {
                wanted.push(Degradation::DeviceInterpolation.code().to_owned());
            }
            assert_eq!(
                codes,
                wanted,
                "{}: not the declarations the row's shape asks for",
                case.name()
            );
        }
    }

    /// Metal's recorded first target holds the four bands where the geometry puts them and the
    /// overdraw over exactly half the rows, with the clear nowhere.
    ///
    /// This is the cross-check the first pass owes, and it is written so that **which** half the
    /// overdraw landed in is Metal's answer rather than this module's: every row is either all
    /// overdraw or the four bands, the two groups are half the rows each, and each group is
    /// contiguous. A row that is neither, a band in the wrong column, a dropped draw (the clear) or
    /// an overdraw that covered the wrong number of rows all fail here rather than being carried
    /// into the sampled target as "Metal's answer".
    #[test]
    fn metal_drew_the_bands_and_then_the_half_overdraw() {
        let mut checked = 0usize;
        for subject in SUBJECTS {
            let Some(recorded) = only_outcome(subject.name) else {
                println!("{}: no recording in the tree yet", subject.name);
                continue;
            };
            let drawn = bytes(&recorded, "drawn/texels");
            let side = subject.source_side;
            assert_eq!(drawn.len(), subject.source_bytes() as usize);
            // Which rows the overdraw took, read off the bytes rather than assumed.
            let mut overdrawn = Vec::new();
            for y in 0..side {
                let all_overdraw = (0..side).all(|x| texel(&drawn, side, x, y) == OVERDRAW);
                let banded = (0..side)
                    .all(|x| texel(&drawn, side, x, y) == COLORS[(x * BANDS / side) as usize]);
                assert!(
                    all_overdraw ^ banded,
                    "{}: row {y} is neither the overdraw nor the bands: {:?}",
                    subject.name,
                    (0..side)
                        .map(|x| texel(&drawn, side, x, y))
                        .collect::<Vec<_>>()
                );
                if all_overdraw {
                    overdrawn.push(y);
                }
            }
            assert_eq!(
                overdrawn.len() as u32,
                side / 2,
                "{}: the overdraw covered {} of {side} rows",
                subject.name,
                overdrawn.len()
            );
            // Contiguous, so it is a half and not a scatter; which half is Metal's to say.
            assert!(
                overdrawn.windows(2).all(|pair| pair[1] == pair[0] + 1),
                "{}: the overdrawn rows are not contiguous: {overdrawn:?}",
                subject.name
            );
            assert!(
                overdrawn[0] == 0 || overdrawn[0] == side / 2,
                "{}: the overdrawn rows start at {}",
                subject.name,
                overdrawn[0]
            );
            checked += 1;
        }
        println!("metal_drew_the_bands_and_then_the_half_overdraw: {checked} rows checked");
    }

    /// The exact row's sampled target is its drawn target, texel for texel.
    ///
    /// `tob_sample` divides the pixel's own position by the sampled texture's extent, so with the
    /// two targets of one extent every sample lands on a texel centre and the answer cannot depend
    /// on the filter. What is left for the row to fail on is the barrier between the two passes and
    /// the layout the sampled image is in -- which is what the family is for.
    #[test]
    fn the_exact_rows_sample_reproduces_what_the_first_pass_drew() {
        let subject = SUBJECTS[0];
        assert_eq!(subject.source_side, subject.target_side);
        let Some(recorded) = only_outcome(subject.name) else {
            println!("{}: no recording in the tree yet", subject.name);
            return;
        };
        assert_eq!(
            bytes(&recorded, "sampled/texels"),
            bytes(&recorded, "drawn/texels"),
            "{}",
            subject.name
        );
    }

    /// The magnified nearest row's sampled target is its drawn target doubled on both axes, and
    /// **mirrored in `y`**.
    ///
    /// `tgp_rect` runs the texture coordinates across the drawn rectangle: `uv` is the unit-square
    /// parameter of the rectangle's corners, so `uv.x` is `(x + 0.5) / 16` across a 16-wide target
    /// and the nearest texel of an 8-wide source is `x / 2`. In `y` it is not: a rectangle's
    /// `uv.y` is 0 at its clip-space `y` of -1, Metal's viewport puts clip `y` of **+1** at pixel
    /// row 0, and a sampled `v` of 0 is the texture's **first** row -- so `uv.y` at pixel row `y`
    /// is `1 - (y + 0.5) / 16`, the sampled source row is `(side - 1) - y / 2`, and the sample sees
    /// the source's rows in the order opposite to the one a blit reads them in.
    ///
    /// The exact row does not mirror, because `tob_sample` divides the **pixel's own position** by
    /// the sampled texture's extent and so carries no clip-space orientation at all. Two sampling
    /// conventions, two answers, and the pattern varying by row is what makes the difference
    /// visible: with four column bands alone, both rows would have passed an identity in `y`.
    ///
    /// All of it is arithmetic the case states and Metal was not asked, so a disagreement is a
    /// finding about the geometry rather than a number to adopt.
    #[test]
    fn the_magnified_nearest_row_doubles_each_source_texel() {
        let subject = SUBJECTS[1];
        assert_eq!(subject.filter, SamplerMinMagFilter::Nearest);
        assert_eq!(subject.target_side, 2 * subject.source_side);
        let Some(recorded) = only_outcome(subject.name) else {
            println!("{}: no recording in the tree yet", subject.name);
            return;
        };
        let (drawn, sampled) = (
            bytes(&recorded, "drawn/texels"),
            bytes(&recorded, "sampled/texels"),
        );
        for y in 0..subject.target_side {
            for x in 0..subject.target_side {
                assert_eq!(
                    texel(&sampled, subject.target_side, x, y),
                    texel(
                        &drawn,
                        subject.source_side,
                        x / 2,
                        subject.source_side - 1 - y / 2
                    ),
                    "{}: ({x}, {y})",
                    subject.name
                );
            }
        }
    }

    /// The linear row's filter really ran over the image the first pass drew, and every channel it
    /// answered lies between the source texels it is a blend of.
    ///
    /// Metal's filter is Metal's answer and is not recomputed here: a bound read off a formula
    /// would be this suite deciding what the hardware does. What the row can hold without
    /// deciding that is two things -- that the answer is **not** the nearest row's, so the filter
    /// was not quietly dropped, and that each channel is inside the range of the four source texels
    /// that surround the sample, which no blend of them can leave.
    #[test]
    fn the_magnified_linear_row_blends_between_the_source_texels() {
        let (nearest, linear) = (SUBJECTS[1], SUBJECTS[2]);
        assert_eq!(linear.filter, SamplerMinMagFilter::Linear);
        assert_eq!(linear.source_side, nearest.source_side);
        assert_eq!(linear.target_side, nearest.target_side);
        let (Some(near), Some(lin)) = (only_outcome(nearest.name), only_outcome(linear.name))
        else {
            println!("{}: no recording in the tree yet", linear.name);
            return;
        };
        let drawn = bytes(&lin, "drawn/texels");
        let sampled = bytes(&lin, "sampled/texels");
        assert_eq!(
            drawn,
            bytes(&near, "drawn/texels"),
            "the two rows draw the same source"
        );
        let mut moved = 0usize;
        for y in 0..linear.target_side {
            for x in 0..linear.target_side {
                let here = texel(&sampled, linear.target_side, x, y);
                if here != texel(&bytes(&near, "sampled/texels"), linear.target_side, x, y) {
                    moved += 1;
                }
                // The four source texels the sample can be a blend of, with the edge clamp folding
                // a neighbour outside the source back onto the edge. `u` runs with the pixel
                // column and `v` against the pixel row, for the reason
                // `the_magnified_nearest_row_doubles_each_source_texel` sets out: a rectangle's
                // `uv.y` carries Metal's clip-space orientation and the pixel row does not, so the
                // sample's `v` is `(side - 1) - y / 2` and its second neighbour lies on the other
                // side of the texel from `u`'s.
                let clamp = |at: i32| at.clamp(0, linear.source_side as i32 - 1) as u32;
                let (u, v) = (x as i32 / 2, linear.source_side as i32 - 1 - y as i32 / 2);
                let (du, dv) = (
                    if x % 2 == 0 { -1 } else { 1 },
                    if y % 2 == 0 { 1 } else { -1 },
                );
                let neighbours = [
                    texel(&drawn, linear.source_side, clamp(u), clamp(v)),
                    texel(&drawn, linear.source_side, clamp(u + du), clamp(v)),
                    texel(&drawn, linear.source_side, clamp(u), clamp(v + dv)),
                    texel(&drawn, linear.source_side, clamp(u + du), clamp(v + dv)),
                ];
                for channel in 0..4 {
                    let low = neighbours.iter().map(|n| n[channel]).min().unwrap();
                    let high = neighbours.iter().map(|n| n[channel]).max().unwrap();
                    assert!(
                        (low..=high).contains(&here[channel]),
                        "{}: ({x}, {y}) channel {channel} answered {} outside {low}..={high}",
                        linear.name,
                        here[channel]
                    );
                }
            }
        }
        assert!(
            moved > 0,
            "{}: the linear row answered exactly what the nearest row did, so no filter ran",
            linear.name
        );
    }
}
