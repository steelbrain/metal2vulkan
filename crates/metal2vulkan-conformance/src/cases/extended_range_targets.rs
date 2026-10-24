//! Render targets of Apple's extended-range formats: what a draw and a clear store.
//!
//! This is the family where Metal's float rules are least like anyone else's. A clear of a
//! `BGR10_XR` target goes **straight from the `float`** for a plain private target in a pass with
//! no draw call, and **through `half`** for a smaller one, for one of `ShaderWrite` usage, and for
//! one whose pass draws into another attachment — so the same clear color stores two different
//! codes depending on facts that have nothing to do with the color. The codes are integers, a
//! signed zero and a NaN each take their own one, and the sRGB twins round two neighbouring
//! `double`s to different sides of an encode threshold. Every site here therefore records the
//! stored **codes**, as unsigned words, and compares them exactly.
//!
//! Three families, each a case per format so a format a venue honestly lacks is that venue's gap at
//! its own row rather than a hole in the others:
//!
//! - **The draws** ([`draws`]): a 128 x 64 target of each format, one full-screen draw of the owned
//!   fixture `rt6_samples`'s `half4` and `float4` fills over 8,192 outputs from the probe's
//!   generator — values across and past each format's range, code boundaries a few `f32` steps
//!   either side, NaN, the infinities and both zeros. A single-plane target's texels are blitted
//!   out; a two-plane target cannot be copied, so the case records what Metal's own
//!   `xr_two_planes`'s `xr2p_read_float` reads back from it, which is a Metal answer rather than a
//!   model.
//! - **The clears** ([`clears`]): 484 clear colors into a 1 x 1 target through a `Clear` load
//!   action, each recorded both as the stored texel (where the format can be copied) and as the
//!   `float` a kernel reads back.
//! - **The clear routes** ([`routes`]): the four shapes whose route differs, over four colors
//!   chosen because their two routes disagree — ties between codes, NaN, and sRGB values either
//!   side of an encode threshold.
//!
//! The generator is the probe's own, rebuilt here, and `the_generator_reproduces_the_probes_outputs`
//! checks the rebuild against the two digests the probe printed, so a case whose inputs drifted
//! from the probe's fails rather than comparing other bytes against Metal's answer.
//!
//! Nothing here can fault a GPU: single-sampled targets of a case's own small extent, one
//! full-screen triangle a pass, no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, Dispatch, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture whose full-screen triangle and fills every draw here runs.
pub(super) const FILLS: &str = "rt6_samples";

/// Its vertex function: the full-screen triangle (-1, -1), (3, -1), (-1, 3).
pub(super) const VERTEX: &str = "rt6_fill_vertex";

/// Its `half4` fill and its `float4` fill.
pub(super) const FILL_HALF: &str = "rt6_fill_rgba16float";
pub(super) const FILL_FLOAT: &str = "rt6_fill_rgba32float";

/// The fixture whose kernel reads a texel of a two-plane extended-range texture as a `float4`,
/// which is the only way such a target's texels reach the host: Metal copies no two-plane
/// extended-range texture.
pub(super) const PLANES: &str = "xr_two_planes";

/// Its reading kernel.
pub(super) const READ_FLOAT: &str = "xr2p_read_float";

/// The draws' target: 128 x 64 texels, one output each.
pub(super) const WIDTH: u32 = 128;
pub(super) const HEIGHT: u32 = 64;

/// The outputs one draw covers.
pub(super) const COUNT: usize = (WIDTH * HEIGHT) as usize;

/// The clear colors a sweep runs, in batches of this many passes to one command buffer, as the
/// probe batched them: one command buffer a clear would be hundreds of round trips on both sides.
const BATCH: usize = 64;

/// The bytes a `float4` read-back takes.
const READ_BYTES: u32 = 16;

// ---- The probe's generator ----------------------------------------------------------------------

/// The probe's xorshift64* generator, rebuilt here so the inputs are the probe's own and a host
/// without the probe can produce them.
struct Rng(u64);

impl Rng {
    fn seeded(seed: u64) -> Rng {
        let state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        let mut generator = Rng(if state == 0 { 1 } else { state });
        for _ in 0..8 {
            generator.next();
        }
        generator
    }

    fn next(&mut self) -> u64 {
        let mut word = self.0;
        word ^= word >> 12;
        word ^= word << 25;
        word ^= word >> 27;
        self.0 = word;
        word.wrapping_mul(2_685_821_657_736_338_717)
    }

    /// A uniform `f64` in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A uniform integer below `n`.
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
}

/// `value` moved `steps` `f32` steps, up for a positive count.
fn step(mut value: f32, steps: i32) -> f32 {
    for _ in 0..steps.max(0) {
        value = value.next_up();
    }
    for _ in 0..(-steps).max(0) {
        value = value.next_down();
    }
    value
}

/// A color channel's output: mostly inside the format's range, then code boundaries and their
/// `f32` neighbours, then values well past the range, then Metal's specials.
fn color(generator: &mut Rng) -> f32 {
    let kind = generator.below(1000);
    if kind < 650 {
        return (-0.8 + 2.1 * generator.unit()) as f32;
    }
    if kind < 750 {
        let code = generator.below(1024) - 384;
        let steps = generator.below(5) - 2;
        return step((f64::from(code) / 510.0) as f32, steps);
    }
    if kind < 820 {
        let code = generator.below(1023);
        let steps = generator.below(5) - 2;
        return step(((f64::from(code) + 0.5 - 384.0) / 510.0) as f32, steps);
    }
    if kind < 900 {
        let grid = generator.below(22552) - 6168;
        let offset = if generator.below(2) != 0 { 0.5 } else { 0.0 };
        let steps = generator.below(3) - 1;
        return step(((f64::from(grid) + offset) / 8192.0) as f32, steps);
    }
    if kind < 995 {
        return (-3.0 + 6.0 * generator.unit()) as f32;
    }
    match kind {
        995 => f32::NAN,
        996 => f32::INFINITY,
        997 => f32::NEG_INFINITY,
        998 => -0.0,
        _ => 0.0,
    }
}

/// An alpha channel's output, over the alpha byte's own boundaries.
fn alpha(generator: &mut Rng) -> f32 {
    let kind = generator.below(1000);
    if kind < 650 {
        return generator.unit() as f32;
    }
    if kind < 700 {
        return 0.0;
    }
    if kind < 750 {
        return 1.0;
    }
    if kind < 800 {
        let byte = generator.below(256);
        let steps = generator.below(3) - 1;
        return step((f64::from(byte) / 255.0) as f32, steps);
    }
    if kind < 850 {
        let byte = generator.below(255);
        let steps = generator.below(3) - 1;
        return step(((f64::from(byte) + 0.5) / 255.0) as f32, steps);
    }
    if kind < 900 {
        let code = generator.below(511);
        let steps = generator.below(3) - 1;
        return step((f64::from(code) / 510.0) as f32, steps);
    }
    if kind < 995 {
        return (-0.5 + 2.0 * generator.unit()) as f32;
    }
    match kind {
        995 => f32::NAN,
        996 => f32::INFINITY,
        997 => -0.0,
        _ => 2.0,
    }
}

/// The draws' outputs: the probe's seed 77, red, green, blue then alpha per texel.
pub(super) fn outputs() -> Vec<[f32; 4]> {
    let mut generator = Rng::seeded(77);
    (0..COUNT)
        .map(|_| {
            let rgb = [
                color(&mut generator),
                color(&mut generator),
                color(&mut generator),
            ];
            [rgb[0], rgb[1], rgb[2], alpha(&mut generator)]
        })
        .collect()
}

/// `value` rounded to `half` to nearest, ties to even, as the probe's `(_Float16)` conversion
/// rounds it. Written out rather than taken from a library so the bits are the probe's own,
/// NaN payloads included.
pub(super) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return sign | 0x7e00 | ((magnitude >> 13) & 0x1ff) as u16;
    }
    if magnitude < 0x3880_0000 {
        // Below 2^-14: a `half` denormal, in units of 2^-24.
        if magnitude < 0x3300_0000 {
            return sign;
        }
        let mantissa = (magnitude & 0x7f_ffff) | 0x80_0000;
        let shift = 126 - (magnitude >> 23);
        let (quotient, rest, tie) = (
            mantissa >> shift,
            mantissa & ((1 << shift) - 1),
            1 << (shift - 1),
        );
        let quotient = quotient + u32::from(rest > tie || (rest == tie && quotient & 1 == 1));
        return sign | quotient as u16;
    }
    let mut rounded = (magnitude >> 13) - ((127 - 15) << 10);
    let rest = magnitude & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && rounded & 1 == 1) {
        rounded += 1;
    }
    if rounded >= 0x7c00 {
        return sign | 0x7c00;
    }
    sign | rounded as u16
}

/// The draws' `float4` bytes.
pub(super) fn float_bytes() -> Vec<u8> {
    outputs()
        .iter()
        .flat_map(|texel| texel.iter().flat_map(|channel| channel.to_le_bytes()))
        .collect()
}

/// The draws' `half4` bytes: each output rounded to `half` as the probe rounded it.
pub(super) fn half_bytes() -> Vec<u8> {
    outputs()
        .iter()
        .flat_map(|texel| texel.map(half_bits))
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// The buffer the fills read their extent from: the row's width and the samples of a pixel, as the
/// fixture's `Rt6Shape`.
pub(super) fn shape(width: u32, samples: u32) -> Vec<u8> {
    let mut bytes = width.to_le_bytes().to_vec();
    bytes.extend(samples.to_le_bytes());
    bytes
}

/// The clear colors: Metal's specials, every sixteenth code's value, and code and alpha-byte
/// boundaries with their `f64` and `f32` neighbours — the subset of the probe's the lane kept, in
/// the probe's order.
fn clear_values() -> Vec<f64> {
    let mut values = vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        -2.0,
        1e-40,
        3.0,
        -1.0,
        1.0,
        0.5,
        1.252_941_176_470_588_2,
        1.3,
        1e30,
        -0.752_941_176_470_588_2,
        0.1,
        0.2,
        0.25,
        0.75,
    ];
    values.extend(
        (0..1024)
            .step_by(16)
            .map(|code| f64::from(code - 384) / 510.0),
    );
    let boundaries = (8..1023)
        .step_by(16)
        .map(|code| (f64::from(code) + 0.5 - 384.0) / 510.0)
        .chain(
            (4..255)
                .step_by(16)
                .map(|byte| (f64::from(byte) + 0.5) / 255.0),
        );
    for boundary in boundaries {
        let single = boundary as f32;
        values.extend([
            boundary,
            boundary.next_down(),
            boundary.next_up(),
            f64::from(single.next_down()),
            f64::from(single.next_up()),
        ]);
    }
    values
}

/// The colors a sweep fingerprints as: every clear value's bits, so a value that moved by one
/// `f64` step is a change to the case's inputs rather than a quiet pass.
fn clear_words() -> Vec<u8> {
    clear_values()
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

// ---- The formats -------------------------------------------------------------------------------

/// One extended-range format a target takes, and how its texels reach the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Format {
    /// The short word the case names itself with.
    pub(super) slug: &'static str,
    pub(super) format: PixelFormat,
    /// The bytes one texel takes, for a format whose texels Metal copies.
    pub(super) texel: u32,
    /// What one copied word is: `BGRA10_XR`'s four 16-bit fields, or one 32-bit word.
    pub(super) word: Element,
    /// The words one texel holds.
    pub(super) words: u32,
    /// Whether Metal copies the format at all. Metal copies no two-plane extended-range texture,
    /// so such a target's answer is the `float` a kernel reads back rather than its bytes.
    pub(super) copies: bool,
}

/// Every extended-range render-target format, as the lane sweeps them.
pub(super) const FORMATS: [Format; 6] = [
    Format {
        slug: "bgra10xr",
        format: PixelFormat::BGRA10XR,
        texel: 8,
        word: Element::U16,
        words: 4,
        copies: true,
    },
    Format {
        slug: "bgra10xr-srgb",
        format: PixelFormat::BGRA10XRSRGB,
        texel: 8,
        word: Element::U16,
        words: 4,
        copies: true,
    },
    Format {
        slug: "bgr10xr",
        format: PixelFormat::BGR10XR,
        texel: 4,
        word: Element::U32,
        words: 1,
        copies: true,
    },
    Format {
        slug: "bgr10xr-srgb",
        format: PixelFormat::BGR10XRSRGB,
        texel: 4,
        word: Element::U32,
        words: 1,
        copies: true,
    },
    Format {
        slug: "planes",
        format: PixelFormat::RGB10A82PXR10,
        texel: 4,
        word: Element::U32,
        words: 1,
        copies: false,
    },
    Format {
        slug: "planes-srgb",
        format: PixelFormat::RGB10A82PXR10SRGB,
        texel: 4,
        word: Element::U32,
        words: 1,
        copies: false,
    },
];

/// The layout a copied target's texels are read under: one row of the layout per row of texels, so
/// a diff names the texel by its position and prints the code it stored.
pub(super) fn texel_layout(format: &Format, size: [u32; 2]) -> ByteLayout {
    let pitch = size[0] * format.texel;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
        },
        element: format.word,
        row_offset: 0,
        row_elements: size[0] * format.words,
        row_pitch: pitch,
        rows: size[1],
        image_pitch: pitch * size[1],
        images: 1,
        // The rows are packed, so the sites cover every byte and there is no gap.
        gaps: Gaps::Ignored,
    }
}

/// The layout a `float4` read-back of `rows` rows of `per_row` texels is read under.
pub(super) fn read_layout(per_row: u32, rows: u32) -> ByteLayout {
    let pitch = per_row * READ_BYTES;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(pitch) * u64::from(rows),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: per_row * 4,
        row_pitch: pitch,
        rows,
        image_pitch: pitch * rows,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The two fixtures, under the entry points a case runs.
pub(super) fn fixture(name: &'static str, entry: &'static str) -> Fixture {
    Fixture { name, entry }
}

/// Both committed texts of each of `entries`, as a case's shader list.
pub(super) fn shader_refs(
    name: &'static str,
    entries: &[&'static str],
) -> Result<Vec<ShaderRef>, String> {
    let mut shaders = Vec::new();
    for entry in entries {
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{name}{extension}"),
                entry: (*entry).to_owned(),
                source_sha256: crate::sha256_hex(fixture(name, entry).text(extension)?.as_bytes()),
            });
        }
    }
    Ok(shaders)
}

/// The two digests a fixture contributes to a case's input fingerprint.
pub(super) fn fixture_inputs(label: &str, name: &'static str) -> Result<Vec<Input>, String> {
    let held = fixture(name, VERTEX);
    Ok(vec![
        Input {
            name: format!("{label}:air"),
            digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: format!("{label}:air-typed"),
            digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// Stop the run because a fixture could not be read.
pub(super) fn not_read(at: &str, why: String) -> Abort {
    Abort {
        at: at.to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    }
}

/// The site a render program's outcome is emitted at, so a refused draw reads as the call we
/// refused rather than as bytes that moved.
pub(super) fn program(
    sites: &mut Sites,
    at: &str,
    outcome: &crate::case::PassOutcome,
    work: &[RenderWork],
) {
    let mut node = Node::branch(at, Vec::new()).with(Node::leaf(
        "status",
        Observed::value(outcome.status.clone()),
    ));
    for (piece, refused) in work.iter().zip(&outcome.refused) {
        if let Some(failure) = refused {
            node = node.with(Node::leaf(
                piece.call(),
                Observed::value(failure.recorded()),
            ));
        }
    }
    if let Some(error) = &outcome.error {
        node = node.with(Node::leaf("error", Observed::value(error.clone())));
    }
    sites.push(node);
}

// ---- The draws ---------------------------------------------------------------------------------

/// One case: a full-screen draw of one fill into a target of one format.
pub struct TargetDraw {
    name: &'static str,
    format: Format,
    /// Whether the fill returns `half4` rather than `float4`.
    half: bool,
}

/// Every draw the lane runs: each format from `half` and from `float`.
pub fn draws() -> Vec<TargetDraw> {
    let names: [[&str; 2]; 6] = [
        ["xr-draw-bgra10xr-half", "xr-draw-bgra10xr-float"],
        ["xr-draw-bgra10xr-srgb-half", "xr-draw-bgra10xr-srgb-float"],
        ["xr-draw-bgr10xr-half", "xr-draw-bgr10xr-float"],
        ["xr-draw-bgr10xr-srgb-half", "xr-draw-bgr10xr-srgb-float"],
        ["xr-draw-planes-half", "xr-draw-planes-float"],
        ["xr-draw-planes-srgb-half", "xr-draw-planes-srgb-float"],
    ];
    let mut cases = Vec::new();
    for (format, pair) in FORMATS.iter().zip(names) {
        for (half, name) in [(true, pair[0]), (false, pair[1])] {
            cases.push(TargetDraw {
                name,
                format: *format,
                half,
            });
        }
    }
    cases
}

impl TargetDraw {
    /// The fill the case draws.
    fn fill(&self) -> &'static str {
        if self.half {
            FILL_HALF
        } else {
            FILL_FLOAT
        }
    }

    /// The outputs the fill reads.
    fn values(&self) -> Vec<u8> {
        if self.half {
            half_bytes()
        } else {
            float_bytes()
        }
    }

    /// The pipeline the case makes: the fill over one attachment of the target's format.
    fn pipeline<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, self.fill(), colors)
    }
}

impl Case for TargetDraw {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        if !self.half {
            return Vec::new();
        }
        // `rt6_samples` is a fast-math module, so a pipeline that stores its `half4` fill into a
        // color attachment announces that the half output's rounding into the attachment may not be
        // Metal's (`color_writes.rs`: the kind fires for a stored fast-math half output, whether or
        // not the attachment blends). Only the notice set is loosened: every stored code still
        // compares exactly, so a device that rounds as Metal does shows as announced with nothing
        // differing — which is what this M3 Max shows — and one that does not shows the code. The
        // allowance fails the case if the notice stops firing, so it cannot outlive its reason. The
        // `float4` fill of the same outputs announces nothing and carries no allowance, which is
        // what makes the pairing worth recording.
        vec![crate::replay::notice_allowance(
            Degradation::HalfOutputFastMathBlend.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = shader_refs(FILLS, &[VERTEX, FILL_HALF, FILL_FLOAT])?;
        if !self.format.copies {
            shaders.extend(shader_refs(PLANES, &[READ_FLOAT])?);
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        if !self.format.copies {
            inputs.extend(fixture_inputs("planes", PLANES)?);
        }
        let colors = [Some(ColorState::new(self.format.format))];
        inputs.extend([
            // The format and the fill are the question the case asks.
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("pipeline", &self.pipeline(&colors).words()),
            Input::bytes(
                "target",
                &TargetSpec::read(self.format.format, [WIDTH, HEIGHT]).words(),
            ),
            Input::bytes("buffer:values", &self.values()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
        ]);
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let colors = [Some(ColorState::new(self.format.format))];
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &self.pipeline(&colors)),
        )?;
        let values = made(sites, "buffer/values", side.buffer(&self.values()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let target = made(
            sites,
            "target",
            side.target(&TargetSpec::read(self.format.format, [WIDTH, HEIGHT])),
        )?;

        // One command buffer: the pass that draws, then whichever read-back the format allows.
        let pass = RenderWork::Pass(
            RenderPass::clearing(vec![PassColor::cleared(target, [0.0, 0.0, 0.0, 1.0])]).drawing(
                vec![
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
                ],
            ),
        );
        if self.format.copies {
            let bytes = (WIDTH * HEIGHT * self.format.texel) as usize;
            let out = made(sites, "buffer/texels", side.zeroed_buffer(bytes))?;
            let work = vec![
                pass,
                RenderWork::ReadTexture {
                    texture: target,
                    into: out,
                    offset: 0,
                    row_pitch: WIDTH * self.format.texel,
                    size: [WIDTH, HEIGHT],
                    plane: Plane::Whole,
                    at: ReadAt::BASE,
                },
            ];
            let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
            program(sites, "program", &outcome, &work);
            let texels = call(sites, "read-back", side.read_buffer(out, 0, bytes), |_| {
                Value::Absent
            })?;
            sites.push(Node::leaf(
                "texels",
                Observed::bytes(texels, texel_layout(&self.format, [WIDTH, HEIGHT])),
            ));
            return Ok(());
        }

        // A two-plane target: Metal copies none, so the answer is what Metal's own kernel reads.
        let planes = fixture(PLANES, READ_FLOAT)
            .loaded()
            .map_err(|why| not_read("library/planes", why))?;
        let plane_library = made(sites, "library/planes", side.library(&planes.source()))?;
        let read = made(
            sites,
            "pipeline/read",
            side.compute_pipeline(plane_library, READ_FLOAT),
        )?;
        let bytes = (WIDTH * HEIGHT * READ_BYTES) as usize;
        let out = made(sites, "buffer/reads", side.zeroed_buffer(bytes))?;
        let work = vec![
            pass,
            RenderWork::Dispatch(Dispatch {
                pipeline: read,
                buffers: vec![(out, 0), (extent, 0)],
                textures: vec![target],
                grid: [WIDTH, HEIGHT, 1],
                threads: [32, 32, 1],
            }),
        ];
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);
        let reads = call(sites, "read-back", side.read_buffer(out, 0, bytes), |_| {
            Value::Absent
        })?;
        sites.push(Node::leaf(
            "reads",
            Observed::bytes(reads, read_layout(WIDTH, HEIGHT)),
        ));
        Ok(())
    }
}

// ---- The clears --------------------------------------------------------------------------------

/// One case: every clear color into a 1 x 1 target of one format.
pub struct ClearTexels {
    name: &'static str,
    format: Format,
}

/// Every format's clears.
pub fn clears() -> Vec<ClearTexels> {
    let names = [
        "xr-clears-bgra10xr",
        "xr-clears-bgra10xr-srgb",
        "xr-clears-bgr10xr",
        "xr-clears-bgr10xr-srgb",
        "xr-clears-planes",
        "xr-clears-planes-srgb",
    ];
    FORMATS
        .iter()
        .zip(names)
        .map(|(format, name)| ClearTexels {
            name,
            format: *format,
        })
        .collect()
}

impl Case for ClearTexels {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(PLANES, &[READ_FLOAT])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("planes", PLANES)?;
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes(
                "target",
                &TargetSpec::read(self.format.format, [1, 1]).words(),
            ),
            // Every clear color's bits, in the order the sweep clears them.
            Input::bytes("clear-colors", &clear_words()),
            Input::bytes("buffer:shape", &shape(1, 1)),
        ]);
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let planes = fixture(PLANES, READ_FLOAT)
            .loaded()
            .map_err(|why| not_read("library/planes", why))?;
        let library = made(sites, "library/planes", side.library(&planes.source()))?;
        let read = made(
            sites,
            "pipeline/read",
            side.compute_pipeline(library, READ_FLOAT),
        )?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(1, 1)))?;
        let target = made(
            sites,
            "target",
            side.target(&TargetSpec::read(self.format.format, [1, 1])),
        )?;
        let values = clear_values();
        let texel_bytes = values.len() * self.format.texel as usize;
        let read_bytes = values.len() * READ_BYTES as usize;
        let texels = made(sites, "buffer/texels", side.zeroed_buffer(texel_bytes))?;
        let reads = made(sites, "buffer/reads", side.zeroed_buffer(read_bytes))?;

        for (batch, colors) in values.chunks(BATCH).enumerate() {
            let first = batch * BATCH;
            let mut work = Vec::new();
            for (offset, &value) in colors.iter().enumerate() {
                let at = (first + offset) as u64;
                // The clear is the whole pass: a `Clear` load action, stored, with no draw call.
                work.push(RenderWork::Pass(RenderPass::clearing(vec![
                    PassColor::cleared(target, [value; 4]),
                ])));
                if self.format.copies {
                    work.push(RenderWork::ReadTexture {
                        texture: target,
                        into: texels,
                        offset: at * u64::from(self.format.texel),
                        row_pitch: self.format.texel,
                        size: [1, 1],
                        plane: Plane::Whole,
                        at: ReadAt::BASE,
                    });
                }
                work.push(RenderWork::Dispatch(Dispatch {
                    pipeline: read,
                    buffers: vec![(reads, at * u64::from(READ_BYTES)), (extent, 0)],
                    textures: vec![target],
                    grid: [1, 1, 1],
                    threads: [1, 1, 1],
                }));
            }
            let at = format!("render/{batch}");
            let outcome = call(sites, &at, side.render(&work), |_| Value::Absent)?;
            program(sites, &format!("program/{batch}"), &outcome, &work);
        }

        if self.format.copies {
            let stored = call(
                sites,
                "read-back/texels",
                side.read_buffer(texels, 0, texel_bytes),
                |_| Value::Absent,
            )?;
            // One row of the layout a clear color, so a diff names the color that moved.
            let mut layout = texel_layout(&self.format, [1, 1]);
            layout.region = Region::Buffer {
                offset: 0,
                length: texel_bytes as u64,
            };
            layout.rows = values.len() as u32;
            layout.image_pitch = texel_bytes as u32;
            sites.push(Node::leaf("texels", Observed::bytes(stored, layout)));
        }
        let got = call(
            sites,
            "read-back/reads",
            side.read_buffer(reads, 0, read_bytes),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf(
            "reads",
            Observed::bytes(got, read_layout(1, values.len() as u32)),
        ));
        Ok(())
    }
}

// ---- The clear routes --------------------------------------------------------------------------

/// Colors whose texels differ between Metal's two clear routes: ties between codes, which the
/// straight route takes up (`0.25` there `float` steps below a tie, down), NaN (code 0 straight,
/// 384 through `half`; alpha 384 on both), and `_sRGB` values each side of an encode threshold
/// whose route rounds them apart (`0.0254486893...` straight below its threshold, where an exact
/// sRGB curve puts it above).
const ROUTE_COLORS: [[f64; 4]; 4] = [
    [0.25, -0.25, f64::NAN, 0.75],
    [
        0.000_980_392_156_862_745,
        0.500_980_392_156_862_7,
        0.999,
        f64::NAN,
    ],
    [
        0.035_294_117_647_058_82,
        -0.522_549_019_607_843_2,
        0.025_448_689_342_018_27,
        0.25,
    ],
    [
        -0.343_137_254_901_960_8,
        1.175_490_196_078_431_4,
        f64::INFINITY,
        -1.0,
    ],
];

/// One shape whose clear route the lane measured: the target's side, whether it is of
/// `ShaderWrite` usage, and whether the pass draws into the other attachment.
#[derive(Clone, Copy, Debug)]
struct Route {
    name: &'static str,
    side: u32,
    shader_write: bool,
    draws: bool,
}

/// The four shapes: the first clears straight from the `float` and the other three through `half`.
const ROUTES: [Route; 4] = [
    Route {
        name: "plain-16",
        side: 16,
        shader_write: false,
        draws: false,
    },
    Route {
        name: "plain-8",
        side: 8,
        shader_write: false,
        draws: false,
    },
    Route {
        name: "written-16",
        side: 16,
        shader_write: true,
        draws: false,
    },
    Route {
        name: "drawn-16",
        side: 16,
        shader_write: false,
        draws: true,
    },
];

/// The format the attachment a routes case draws into takes, so the pass has a draw call that
/// writes something other than the target under test.
const OTHER: PixelFormat = PixelFormat::RGBA32Float;

/// One case: every shape and every color, for one single-plane format.
pub struct ClearRoutes {
    name: &'static str,
    format: Format,
}

/// Every single-plane format's clear routes. A two-plane target is left out because Metal copies
/// none, and the whole point of this family is the stored texel.
pub fn routes() -> Vec<ClearRoutes> {
    let names = [
        "xr-clear-routes-bgra10xr",
        "xr-clear-routes-bgra10xr-srgb",
        "xr-clear-routes-bgr10xr",
        "xr-clear-routes-bgr10xr-srgb",
    ];
    FORMATS
        .iter()
        .filter(|format| format.copies)
        .zip(names)
        .map(|(format, name)| ClearRoutes {
            name,
            format: *format,
        })
        .collect()
}

impl ClearRoutes {
    /// The target one shape makes.
    fn target(&self, route: &Route) -> TargetSpec {
        let spec = TargetSpec::read(self.format.format, [route.side, route.side]);
        if route.shader_write {
            spec.written()
        } else {
            spec
        }
    }

    /// The pipeline the drawing shape sets: the `float4` fill, writing attachment 0 of [`OTHER`],
    /// with the target under test at attachment 1.
    fn pipeline<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FILL_FLOAT, colors)
    }

    /// The outputs the drawing shape's fill reads, one `float4` a texel of a `side` x `side`
    /// target.
    fn drawn_values(side: u32) -> Vec<u8> {
        (0..side * side)
            .flat_map(|at| [(at % 7) as f32 / 7.0, 0.25, 0.5, 1.0])
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}

impl Case for ClearRoutes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, FILL_FLOAT])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        let colors = [
            Some(ColorState::new(OTHER)),
            Some(ColorState::new(self.format.format)),
        ];
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("pipeline", &self.pipeline(&colors).words()),
            Input::bytes(
                "clear-colors",
                &ROUTE_COLORS
                    .iter()
                    .flat_map(|color| color.iter().flat_map(|value| value.to_bits().to_le_bytes()))
                    .collect::<Vec<u8>>(),
            ),
        ]);
        for route in &ROUTES {
            let mut words = self.target(route).words();
            words.push(u8::from(route.draws));
            words.extend(shape(route.side, 1));
            words.extend(Self::drawn_values(route.side));
            inputs.push(Input::bytes(format!("route:{}", route.name), &words));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/fills", why))?;
        let library = made(sites, "library/fills", side.library(&fills.source()))?;
        let colors = [
            Some(ColorState::new(OTHER)),
            Some(ColorState::new(self.format.format)),
        ];
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &self.pipeline(&colors)),
        )?;

        for route in &ROUTES {
            let at = |what: &str| format!("{}/{what}", route.name);
            let values = made(
                sites,
                &at("buffer/values"),
                side.buffer(&Self::drawn_values(route.side)),
            )?;
            let extent = made(
                sites,
                &at("buffer/shape"),
                side.buffer(&shape(route.side, 1)),
            )?;
            let bytes = (route.side * route.side * self.format.texel) as usize;
            let out = made(sites, &at("buffer/texels"), side.zeroed_buffer(bytes))?;
            for (index, color) in ROUTE_COLORS.iter().enumerate() {
                // A fresh target a color, as the probe made one: a target the earlier color left
                // texels in would answer a different question.
                let target = made(
                    sites,
                    &at(&format!("target/{index}")),
                    side.target(&self.target(route)),
                )?;
                let other = made(
                    sites,
                    &at(&format!("other/{index}")),
                    side.target(&TargetSpec::read(OTHER, [route.side, route.side])),
                )?;
                let steps = if route.draws {
                    vec![
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
                } else {
                    Vec::new()
                };
                let work = vec![
                    RenderWork::Pass(
                        RenderPass::clearing(vec![
                            PassColor::discarded(other),
                            PassColor::cleared(target, *color),
                        ])
                        .drawing(steps),
                    ),
                    RenderWork::ReadTexture {
                        texture: target,
                        into: out,
                        offset: 0,
                        row_pitch: route.side * self.format.texel,
                        size: [route.side, route.side],
                        plane: Plane::Whole,
                        at: ReadAt::BASE,
                    },
                ];
                let outcome = call(
                    sites,
                    &at(&format!("render/{index}")),
                    side.render(&work),
                    |_| Value::Absent,
                )?;
                program(sites, &at(&format!("program/{index}")), &outcome, &work);
                let texels = call(
                    sites,
                    &at(&format!("read-back/{index}")),
                    side.read_buffer(out, 0, bytes),
                    |_| Value::Absent,
                )?;
                sites.push(Node::leaf(
                    at(&format!("texels/{index}")),
                    Observed::bytes(texels, texel_layout(&self.format, [route.side, route.side])),
                ));
            }
        }
        Ok(())
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for case in draws() {
        cases.push(Box::new(case));
    }
    for case in clears() {
        cases.push(Box::new(case));
    }
    for case in routes() {
        cases.push(Box::new(case));
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The SHA-256 of the probe's `float4` outputs (its `rt.src.f`), which the rebuilt generator
    /// must reproduce.
    const SOURCE_FLOAT: &str = "aa40c5c5e485af466c71b3a8877a9be51f2a8e3005a19fc950c104feede45238";

    /// The SHA-256 of the probe's `half4` outputs (its `rt.src.h`).
    const SOURCE_HALF: &str = "e1e257f0dca558bae1d82053b821eea9a78ad59af1379462600590815d0f366a";

    /// The rebuilt generator is the probe's: both digests the probe printed. Without this the
    /// cases would be comparing Metal's answer against other bytes the moment the generator
    /// drifted, which is exactly what the input fingerprint exists to stop.
    #[test]
    fn the_generator_reproduces_the_probes_outputs() {
        assert_eq!(outputs().len(), COUNT);
        assert_eq!(crate::sha256_hex(&float_bytes()), SOURCE_FLOAT);
        assert_eq!(crate::sha256_hex(&half_bytes()), SOURCE_HALF);
    }

    /// The clear sweep is the probe's 484 colors, and they hold the values the family turns on: a
    /// signed zero, a NaN, both infinities and a code tie.
    #[test]
    fn the_clear_sweep_is_the_probes() {
        let values = clear_values();
        assert_eq!(values.len(), 484);
        assert!(values.iter().any(|value| value.is_nan()));
        assert!(values.contains(&f64::INFINITY));
        assert!(values.contains(&f64::NEG_INFINITY));
        assert!(values
            .iter()
            .any(|value| *value == 0.0 && value.is_sign_negative()));
    }

    /// Every case names a distinct row, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_rows() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
    }

    /// Each format's texel size and word count agree, so a layout covers every byte a copy writes
    /// and none past it.
    #[test]
    fn every_formats_words_cover_its_texel() {
        for format in &FORMATS {
            assert_eq!(
                format.words * format.word.size(),
                format.texel,
                "{}: {} words of {} bytes is not a texel of {}",
                format.slug,
                format.words,
                format.word.size(),
                format.texel
            );
            assert_eq!(
                texel_layout(format, [WIDTH, HEIGHT]).extent(),
                (WIDTH * HEIGHT * format.texel) as usize,
                "{}",
                format.slug
            );
        }
    }

    /// The route colors are four distinct colors, and each holds a value whose two routes differ:
    /// a sweep that lost them would compare the routes without asking what tells them apart.
    #[test]
    fn the_route_colors_separate_the_two_routes() {
        let words: BTreeSet<[u64; 4]> = ROUTE_COLORS
            .iter()
            .map(|color| color.map(f64::to_bits))
            .collect();
        assert_eq!(words.len(), ROUTE_COLORS.len());
        assert!(ROUTE_COLORS
            .iter()
            .flatten()
            .any(|value| value.is_nan() || value.is_infinite()));
    }

    /// The four shapes are distinct in the facts Metal's route reads: the extent, the usage and
    /// whether the pass draws.
    #[test]
    fn the_routes_differ_in_what_metal_reads() {
        let shapes: BTreeSet<(u32, bool, bool)> = ROUTES
            .iter()
            .map(|route| (route.side, route.shader_write, route.draws))
            .collect();
        assert_eq!(shapes.len(), ROUTES.len());
    }
}
