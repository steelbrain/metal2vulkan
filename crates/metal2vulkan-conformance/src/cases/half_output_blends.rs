//! What a device's blend stores for a fragment function's `half` output, into a 16- and a 32-bit
//! float attachment.
//!
//! `runtime_lane::half_output_blends` is a **measurement** lane rather than a check: it prints
//! `HALF-BLEND` lines, fits a rule to them, and asserts only that each draw completes and that a
//! pipeline announcing nothing stores the fitted rule's value. The rule it fitted is worth having —
//! into a 16-bit float format Metal blends the `half` the function returns with each factor (`1 - x`
//! included), each product and the sum rounded to `half`, nearest even, and `Min` and `Max` read no
//! factor; into a 32-bit float format it is a multiply-add in `float` — but the numbers behind it
//! were printed to a terminal and are gone.
//!
//! **So the recorded answer here is the blended attachment's stored codes, compared exactly**, and
//! the fitted rule becomes a cross-check over Metal's own measurement instead of the thing the
//! suite believes. That is the design call this family needed, stated rather than guessed: a rule a
//! fit pins is only as good as the fit, and a release that changes Metal's blend should show as the
//! pixel that moved.
//!
//! One case per attachment format, with each blend a site of its own, so a format a venue honestly
//! lacks as a blendable render target is that venue's gap
//! ([`crate::verdict::Verdict::LacksFeature`]) at one row rather than seventeen.
//!
//! **What each case runs.** A 16 by 16 target of the format, of shared storage because the
//! destination codes are arbitrary and reach a texture only through `replaceRegion:`. Seventeen
//! render pipelines over the owned fixture `rt6_samples` — its full-screen triangle and its
//! `rt6_fill_rgba16float`, which returns the `half4` of buffer 0 at the pixel — one per blend, and
//! seventeen passes under a `Load` load action, each into its own target and each blitted out, all
//! in one command buffer.
//!
//! The `one-zero` site is also the **source** read back: a source factor of one and a destination
//! factor of zero writes the `half4` the function returned through the attachment's own conversion
//! and nothing else, so the case does not need a separate unblended draw to know what the device
//! computed for the source — which is the property the lane's own extra draw was there for.
//!
//! **The five blends of the lane's twenty-two that are not here** are the ones whose factors read
//! the blend color (`BlendColor`, `OneMinusBlendColor`, `BlendAlpha`, `OneMinusBlendAlpha`). The
//! reason used to be that `setBlendColorRed:green:blue:alpha:` had no step in the render
//! vocabulary; **that is no longer true** -- [`crate::render::RenderStep::BlendColor`] is one, and
//! [`super::blend_sweeps`] and [`super::rotated_targets`] both set it -- so nothing is in the way
//! and these five are simply **not ported**. They are the blends behind the lane's finding that
//! Metal reads the blend color's `half` **before** its one minus (`h(1 - h(c))`), which this
//! family's `half` outputs would compare and which [`super::blend_sweeps`] does not reach: that
//! sweep blends `float` outputs into a 32-bit float target, where there is no `half` rounding to
//! find. The module docs of `runtime_lane::half_output_blends` keep the finding until somebody
//! ports the five rows.
//!
//! **Where our side differs from Metal, measured over every word.** On this round's MoltenVK
//! (Apple M3 Max, API 1.2) `half-output-blends-r32f` is `announced` -- every code of all seventeen
//! blends identical to Metal's -- and the other five are `divergent`. Measured with a temporary
//! tolerance wide enough to pass, then read off the per-site counts and the tolerance dropped, so
//! these are counts over every byte of the site rather than the first mismatch:
//!
//! | case | site | differing words of the site | largest raw-code distance |
//! |---|---|---|---|
//! | `r16f` | `sc-dc` | 21 of 256 | 1 |
//! | `rg16f` | `sc-dc` | 40 of 512 | 1 |
//! | `rg16f` | `sas-1msa` | 47 of 512 | 80 |
//! | `rgba16f` | `sc-dc` | 90 of 1024 | 1 |
//! | `rgba16f` | `da-1mda` | 256 of 1024 | 432 |
//! | `rg32f` | `sc-dc` | 41 of 512 | 1 |
//! | `rgba32f` | `sc-dc` | 77 of 1024 | 1 |
//! | `rgba32f` | `da-1mda` | 146 of 1024 | 97 |
//!
//! Three of the seventeen blends move at all, and the other fourteen -- including every one whose
//! factor is `One`, `Zero`, a source factor, `Min` or `Max` -- are code-identical in every format.
//! `sc-dc` is one code at its worst. The two large distances are colour channels whose result sits
//! near zero, where a raw-code distance is many steps of a small value; the alpha channel of
//! `da-1mda` does not move at all, in either format.
//!
//! **What the difference is**, read off those eight sites rather than off the notice: Metal rounds
//! the **source** product to the attachment's precision and carries the **destination** product
//! into the sum unrounded -- `round(round(Sf * S) + Df * D)`, one fused multiply-add on the
//! destination term -- and our side is the mirror image, `round(Sf * S + round(Df * D))`. Each rule
//! holds at all 5,376 compared words of the eight sites, and this module's own
//! `metals_recorded_codes_hold_the_rounding_the_divergent_sites_measure` keeps Metal's
//! half of that against the committed recordings. So the gap is not half-precision rounding but
//! which of the two products the arithmetic keeps unrounded, which is why the two 32-bit float
//! formats differ too and why `sc-dc`, whose source product `S * S` is exact in the attachment,
//! still moves by a single code. Metal's rule is its compiler's fast-math fold of the blend into
//! the fragment function; ours is the device's fixed-function blend, which the pipeline leaves the
//! blend to. `half-output-blends-r32f` is where the two rules agree at every word, which is why it
//! is the one of the six that is exact. The alpha channel of `da-1mda`, whose source factor and
//! destination value are both the destination alpha, follows neither rule on either side and is
//! identical on both, so nothing here asserts a rule for it.
//!
//! That stays `divergent` rather than being loosened: `half-output-fast-math-blend`'s own
//! definition says validation derives no tolerance from it. `known-gaps.tsv` carries the eight
//! rows.
//!
//! Nothing here can fault a GPU: single-sampled 16 by 16 targets, one full-screen triangle a pass,
//! no loop, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, TargetSpec,
};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O, PixelFormat};
use metal2vulkan_runtime::Degradation;

use super::extended_range_targets::{
    fixture, fixture_inputs, not_read, program, shader_refs, shape, FILLS, FILL_HALF, VERTEX,
};

/// The target's extent: 256 pixels, one per source value of the sweep.
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;

/// The pixels one draw covers.
const PIXELS: usize = (WIDTH * HEIGHT) as usize;

/// The source halves, as exact codes: `k / 16 - 2` for `k` in 0 to 63, which is every sixteenth
/// between -2 and 2 and is exact in `half` at every one of them.
///
/// It holds both of the lane's sweeps between them: its -0.25 to 1.25 and its -2 to 2, with 0, both
/// unit values and the values past one that a `1 - x` factor turns negative.
const SOURCES: usize = 64;

/// How far apart two channels of one pixel are in the sweep, so each source value lands at every
/// channel position across the pixels and a blend that treated one channel differently from another
/// shows as a pixel rather than as a hash.
const STRIDE: usize = 7;

/// Source value `index` of the sweep, as an `f32`: exact in `half`, so the function's `half4`
/// carries it unrounded and the blend's own rounding is the only rounding in the answer.
fn source(index: usize) -> f32 {
    index as f32 / 16.0 - 2.0
}

/// The `half` code of `value`, which is exact for every value the sweep and the destinations hold.
fn half_code(value: f32) -> u16 {
    let sign = u16::from(value.is_sign_negative()) << 15;
    let size = value.abs();
    if size == 0.0 {
        return sign;
    }
    let bits = size.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    // Every value here is a normal half, so there is no denormal or overflow arm to get wrong; the
    // guard below is what says so.
    assert!(
        (-14..=15).contains(&exponent),
        "{value} is not a normal half"
    );
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    assert_eq!(bits & 0x1fff, 0, "{value} is not exact in half");
    sign | (((exponent + 15) as u16) << 10) | mantissa
}

/// Buffer 0's bytes: one `half4` a pixel, each channel its own value of the sweep.
fn source_bytes() -> Vec<u8> {
    (0..PIXELS)
        .flat_map(|pixel| (0..4).map(move |channel| (pixel, channel)))
        .flat_map(|(pixel, channel)| {
            half_code(source((pixel + STRIDE * channel) % SOURCES)).to_le_bytes()
        })
        .collect()
}

/// The generator the destination codes come from: xorshift64\*, seeded from the format so each
/// format's destination is its own and none of them is all one value.
///
/// Written out here rather than taken from a probe, because this family's probe printed its
/// destinations as lines and kept none: the generator **is** the statement of what the destinations
/// are, and the case's input fingerprint covers the bytes it produces.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut state = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x632b_e59b_d9b4_e019);
        if state == 0 {
            state = 1;
        }
        let mut generator = Self(state);
        for _ in 0..8 {
            generator.next();
        }
        generator
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// A blend the case sweeps: the word its site is named by, and the six fields.
struct Sweep {
    name: &'static str,
    blend: Blend,
}

/// The seventeen blends of the lane's twenty-two whose factors read no blend color.
fn blends() -> Vec<Sweep> {
    let added = |name, source, destination| Sweep {
        name,
        blend: Blend::added(source, destination),
    };
    let under = |name, source, destination, operation| Sweep {
        name,
        blend: Blend::separate(
            (source, destination, operation),
            (source, destination, operation),
        ),
    };
    vec![
        added("one-zero", F::One, F::Zero),
        added("one-one", F::One, F::One),
        added("one-1msa", F::One, F::OneMinusSourceAlpha),
        added("sa-1msa", F::SourceAlpha, F::OneMinusSourceAlpha),
        added("zero-1msa", F::Zero, F::OneMinusSourceAlpha),
        added("dc-1mdc", F::DestinationColor, F::OneMinusDestinationColor),
        added("1msc-1msc", F::OneMinusSourceColor, F::OneMinusSourceColor),
        added("sc-dc", F::SourceColor, F::DestinationColor),
        added("da-1mda", F::DestinationAlpha, F::OneMinusDestinationAlpha),
        added("sas-1msa", F::SourceAlphaSaturated, F::OneMinusSourceAlpha),
        added("1mdc-one", F::OneMinusDestinationColor, F::One),
        added("dc-zero", F::DestinationColor, F::Zero),
        under("min", F::SourceAlpha, F::OneMinusSourceAlpha, O::Min),
        under("max", F::SourceAlpha, F::OneMinusSourceAlpha, O::Max),
        under("one-1msa-sub", F::One, F::OneMinusSourceAlpha, O::Subtract),
        under(
            "sa-1msa-rsub",
            F::SourceAlpha,
            F::OneMinusSourceAlpha,
            O::ReverseSubtract,
        ),
        under("one-one-sub", F::One, F::One, O::Subtract),
    ]
}

/// The case: one attachment format, blended seventeen ways.
pub struct HalfOutputBlends {
    name: &'static str,
    format: PixelFormat,
    /// The channels the format's texel holds, which is how many of the `half4` reach it.
    channels: u32,
    /// Whether a channel is a 32-bit float, where the lane fitted a multiply-add in `float`
    /// rather than a blend in `half`.
    wide: bool,
}

/// Every format the lane blends into.
pub fn all() -> Vec<HalfOutputBlends> {
    let case = |name, format, channels, wide| HalfOutputBlends {
        name,
        format,
        channels,
        wide,
    };
    vec![
        case("half-output-blends-r16f", PixelFormat::R16Float, 1, false),
        case("half-output-blends-rg16f", PixelFormat::RG16Float, 2, false),
        case(
            "half-output-blends-rgba16f",
            PixelFormat::RGBA16Float,
            4,
            false,
        ),
        case("half-output-blends-r32f", PixelFormat::R32Float, 1, true),
        case("half-output-blends-rg32f", PixelFormat::RG32Float, 2, true),
        case(
            "half-output-blends-rgba32f",
            PixelFormat::RGBA32Float,
            4,
            true,
        ),
    ]
}

impl HalfOutputBlends {
    /// The bytes one texel takes.
    fn texel_bytes(&self) -> u32 {
        self.channels * if self.wide { 4 } else { 2 }
    }

    /// The bytes one target's level takes.
    fn level_bytes(&self) -> usize {
        PIXELS * self.texel_bytes() as usize
    }

    /// The destination value of channel `channel` of pixel `pixel`, as the format stores it: a
    /// value in `[0, 1]` off the generator, exact in the format's own channel.
    ///
    /// The lane's destinations were random halves and random `float`s in `[0, 1]`; these are the
    /// same shape from a generator the case states, so a host with no Metal builds the same bytes.
    fn destination_bytes(&self) -> Vec<u8> {
        let mut generator = Rng::new(u64::from(self.format.as_raw()));
        let mut bytes = Vec::with_capacity(self.level_bytes());
        for _ in 0..PIXELS {
            for _ in 0..self.channels {
                match self.wide {
                    // A multiple of 2^-20, exact in `f32`.
                    true => {
                        let value = generator.below(1 << 20) as f32 / (1u32 << 20) as f32;
                        bytes.extend(value.to_bits().to_le_bytes());
                    }
                    // A multiple of 2^-10, exact in `half`.
                    false => {
                        let value = generator.below(1025) as f32 / 1024.0;
                        bytes.extend(half_code(value).to_le_bytes());
                    }
                }
            }
        }
        bytes
    }

    /// The target each pass blends into: shared, so the destination codes go in through
    /// `replaceRegion:`, and shader-read beside its render-target usage, as the lane's were.
    fn target(&self) -> TargetSpec {
        TargetSpec::read(self.format, [WIDTH, HEIGHT]).shared()
    }

    /// The attachment state of one blend.
    fn colors(&self, blend: Blend) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(self.format).blended(blend))]
    }

    /// The layout a stored target is compared under: one row a pixel, a channel an element, so the
    /// element a diff names is the pixel and the channel inside it.
    fn layout(&self) -> ByteLayout {
        let pitch = self.texel_bytes();
        let length = pitch * PIXELS as u32;
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [WIDTH, HEIGHT, 1],
            },
            element: if self.wide {
                Element::U32
            } else {
                Element::U16
            },
            row_offset: 0,
            row_elements: self.channels,
            row_pitch: pitch,
            rows: PIXELS as u32,
            image_pitch: length,
            images: 1,
            // A row is one pixel's channels and the pixels abut, so there is no byte outside one.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for HalfOutputBlends {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // `rt6_samples` is a fast-math module, so a pipeline that stores its `half4` fill into a
        // color attachment announces that the half output's rounding into the attachment may not be
        // Metal's. This declares the notice so the case can be read at all beside Metal, which
        // announces none; it loosens **no stored code**. A device that blends as Metal does reads as
        // announced with nothing differing, and one that does not shows the pixel. The declaration
        // fails the case if the notice stops firing, so it cannot outlive its reason, and it is
        // chosen from the case's own shape -- a stored `half` output under fast math -- never from
        // the device.
        //
        // Nothing loosens a stored code and nothing may: the note's own definition in
        // `pipeline/artifact.rs` ends "Validation derives no tolerance from it", so a tolerance
        // written from it would be the suite overruling the contract it exists to test. Five of the
        // six formats are therefore left **divergent** on this round's MoltenVK, with the measured
        // counts in the module docs above.
        // A half-float attachment also takes the stage's `half` through a 16-bit output on a
        // device that does not store a 32-bit output's NaN as Metal does, announced as
        // `shader-half-color-bits` on the three 16-bit formats of this family. It is the case's own
        // shape again -- a stored `half` output into a half-float attachment -- so its code is
        // declared, and it loosens no stored code either: the three rows stay divergent on the
        // texels the module docs above measure, which is the finding and must not be laundered.
        //
        // Two more of the same shape are declared from the run that shows them: a device whose own
        // conversion into the attachment is not Metal's (`device-color-conversion`, lavapipe) and
        // one whose NaN encoding in a half-float target is its own (`device-nan-encoding`). Both
        // are declarations at the notices site and neither loosens a stored code -- which matters
        // most for `device-color-conversion`, whose own definition says validation derives no
        // tolerance and no explanation from it and a difference stays a mismatch for review. The
        // rows stay divergent on their texels, on both venues.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 6 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                [
                    Degradation::HalfOutputFastMathBlend,
                    Degradation::ShaderHalfColorBits,
                    Degradation::DeviceColorConversion,
                    Degradation::DeviceNanEncoding,
                ]
                .into_iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
            )
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, FILL_HALF])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("fills", FILLS)?;
        inputs.extend([
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("target", &self.target().words()),
            Input::bytes("buffer:values", &source_bytes()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
            // The destination the blends read, which is half of what every answer depends on.
            Input::bytes("texture:destination", &self.destination_bytes()),
        ]);
        // Each blend's own six fields, keyed by its name, so a case whose blend table moved is
        // unrecorded rather than compared against the old table's codes.
        for sweep in blends() {
            let mut bytes = sweep.name.as_bytes().to_vec();
            bytes.push(0);
            bytes.extend(self.colors(sweep.blend)[0].unwrap().words());
            inputs.push(Input::bytes(format!("blend:{}", sweep.name), &bytes));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fills = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library", why))?;
        let library = made(sites, "library", side.library(&fills.source()))?;
        let values = made(sites, "buffer/values", side.buffer(&source_bytes()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let destination = self.destination_bytes();
        let pitch = WIDTH * self.texel_bytes();

        let sweeps = blends();
        let mut work = Vec::new();
        let mut reads = Vec::new();
        for sweep in &sweeps {
            let colors = self.colors(sweep.blend);
            let pipeline = made(
                sites,
                &format!("pipeline/{}", sweep.name),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::new(VERTEX, FILL_HALF, &colors),
                ),
            )?;
            let target = made(
                sites,
                &format!("target/{}", sweep.name),
                side.target(&self.target()),
            )?;
            // The destination codes go in before the pass, as the lane's probe put them there: a
            // blend's answer depends on them, and no clear or kernel write can set an arbitrary
            // code.
            call(
                sites,
                &format!("replace-region/{}", sweep.name),
                side.replace_region(target, [WIDTH, HEIGHT], pitch, &destination),
                |_| Value::Absent,
            )?;
            let into = made(
                sites,
                &format!("buffer/texels/{}", sweep.name),
                side.zeroed_buffer(self.level_bytes()),
            )?;
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(vec![
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
                ]),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into,
                offset: 0,
                row_pitch: pitch,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            reads.push((sweep.name, into));
        }
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);

        let mut stored = Node::branch("blended", Vec::new());
        for (name, buffer) in reads {
            let bytes = call(
                sites,
                &format!("read-back/{name}"),
                side.read_buffer(buffer, 0, self.level_bytes()),
                |_| Value::Absent,
            )?;
            stored = stored.with(Node::leaf(name, Observed::bytes(bytes, self.layout())));
        }
        sites.push(stored);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The `half` value of code `code`, for the cross-check's arithmetic.
    fn half_value(code: u16) -> f32 {
        let sign = if code >> 15 == 1 { -1.0f32 } else { 1.0 };
        let exponent = i32::from((code >> 10) & 0x1f);
        let mantissa = f32::from(code & 0x3ff);
        sign * match exponent {
            0 => mantissa * 2f32.powi(-24),
            31 => f32::INFINITY,
            raised => (1024.0 + mantissa) * 2f32.powi(raised - 25),
        }
    }

    /// Every source value is exact in `half` and the sweep spans both of the lane's ranges, so the
    /// function's `half4` carries the case's own numbers and the blend's rounding is the only
    /// rounding in the answer.
    #[test]
    fn the_sweep_is_exact_in_half_and_spans_the_lanes_ranges() {
        for index in 0..SOURCES {
            let value = source(index);
            // Round-trips through the code, which `half_code` asserts is exact anyway.
            assert_eq!(half_value(half_code(value)), value, "source {index}");
        }
        let values: Vec<f32> = (0..SOURCES).map(source).collect();
        assert!(values.contains(&-2.0) && values.contains(&0.0));
        assert!(values.contains(&-0.25) && values.contains(&1.25));
        assert!(values.contains(&1.0) && values.contains(&-1.0));
        // Values past one, which a `1 - x` factor turns negative: the inputs that separate a
        // clamped blend from an unclamped one.
        assert!(values.iter().any(|value| *value > 1.0));
        assert_eq!(source_bytes().len(), PIXELS * 8);
    }

    /// Every case's destination covers its whole level, is its own, and holds values in `[0, 1]`
    /// exactly as the format stores them.
    #[test]
    fn every_destination_is_its_own_and_covers_its_level() {
        let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
        for case in all() {
            let bytes = case.destination_bytes();
            assert_eq!(bytes.len(), case.level_bytes(), "{}", case.name);
            assert_eq!(case.layout().extent(), case.level_bytes(), "{}", case.name);
            assert!(seen.insert(bytes.clone()), "{} repeats", case.name);
            // Every channel decodes to a value inside the unit range.
            match case.wide {
                true => {
                    for word in bytes.as_chunks::<4>().0 {
                        let value = f32::from_bits(u32::from_le_bytes(*word));
                        assert!((0.0..=1.0).contains(&value), "{value}");
                    }
                }
                false => {
                    for code in bytes.as_chunks::<2>().0 {
                        let value = half_value(u16::from_le_bytes(*code));
                        assert!((0.0..=1.0).contains(&value), "{value}");
                    }
                }
            }
        }
    }

    /// The blends name distinct sites and distinct six-field states: a table with two names for one
    /// blend, or one name for two, would record two answers to one question.
    #[test]
    fn the_blends_are_distinct_by_name_and_by_state() {
        let sweeps = blends();
        assert_eq!(sweeps.len(), 17);
        let named: BTreeSet<&str> = sweeps.iter().map(|sweep| sweep.name).collect();
        assert_eq!(named.len(), sweeps.len());
        let states: BTreeSet<Vec<u8>> = sweeps.iter().map(|sweep| sweep.blend.words()).collect();
        assert_eq!(states.len(), sweeps.len());
        // No blend here reads the blend color. That is now a property of the table rather than of
        // the vocabulary -- `RenderStep::BlendColor` exists -- and it is what the module docs name
        // as the five rows nobody has ported.
        for sweep in &sweeps {
            for factor in [
                sweep.blend.source_rgb,
                sweep.blend.destination_rgb,
                sweep.blend.source_alpha,
                sweep.blend.destination_alpha,
            ] {
                assert!(
                    !matches!(
                        factor,
                        F::BlendColor
                            | F::OneMinusBlendColor
                            | F::BlendAlpha
                            | F::OneMinusBlendAlpha
                    ),
                    "{} reads the blend color",
                    sweep.name
                );
            }
        }
    }

    /// The cases name distinct rows and distinct formats.
    #[test]
    fn the_cases_name_distinct_formats() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        let formats: BTreeSet<u32> = all().iter().map(|case| case.format.as_raw()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(formats.len(), all().len());
    }

    /// The three blends whose answer the lane's rule pins with no rounding to argue about, held
    /// against Metal's own recorded codes for every pixel and every channel of every format:
    ///
    /// - `one-zero` stores the source through the attachment's conversion and nothing else;
    /// - `min` and `max` read **no factor**, so the stored channel is the smaller or the larger of
    ///   the source's and the destination's own values.
    ///
    /// The other fourteen blends involve a product and a sum whose rounding is the subject, so
    /// their answer is Metal's recording and not a model. This is the part of the fit that can be
    /// checked rather than believed, and it is the strongest thing the port can say without
    /// restating the lane's whole model.
    #[test]
    fn metals_recorded_codes_hold_the_three_blends_the_rule_pins() {
        use metal2vulkan_runtime::MetalLevel;
        for case in all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                // Recorded in a later commit of this batch.
                continue;
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: one outcome", case.name));
            let blended = taken
                .root
                .child("blended")
                .unwrap_or_else(|| panic!("{}: no blended site", case.name));
            let layout = case.layout();
            let destination = case.destination_bytes();
            let read = |name: &str| -> Vec<u8> {
                let site = blended
                    .child(name)
                    .unwrap_or_else(|| panic!("{}: no {name} site", case.name));
                match &site.observed.value {
                    Value::Bytes(bytes) => bytes.clone(),
                    other => panic!("{}: {name} holds {}", case.name, other.kind()),
                }
            };
            // A channel's own value, out of the format's stored bytes.
            let channel_of = |bytes: &[u8], pixel: usize, channel: u32| -> f32 {
                let at = layout.at(0, pixel as u32, channel);
                match case.wide {
                    true => {
                        f32::from_bits(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
                    }
                    false => half_value(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())),
                }
            };
            let one_zero = read("one-zero");
            let smallest = read("min");
            let largest = read("max");
            for bytes in [&one_zero, &smallest, &largest] {
                assert_eq!(bytes.len(), case.level_bytes(), "{}", case.name);
            }
            for pixel in 0..PIXELS {
                for channel in 0..case.channels {
                    let stored = channel_of(&one_zero, pixel, channel);
                    let destination_value = channel_of(&destination, pixel, channel);
                    // The source the device computed, read out of the `one-zero` site.
                    let source_value = stored;
                    // A `half` source stored into a 16-bit float channel keeps its own value; a
                    // 32-bit float channel holds the widening, which is exact.
                    let want = half_value(half_code(source(
                        (pixel + STRIDE * channel as usize) % SOURCES,
                    )));
                    assert_eq!(
                        stored, want,
                        "{} one-zero pixel {pixel} channel {channel}: Metal stored {stored}, the \
                         source is {want}",
                        case.name
                    );
                    assert_eq!(
                        channel_of(&smallest, pixel, channel),
                        source_value.min(destination_value),
                        "{} min pixel {pixel} channel {channel}",
                        case.name
                    );
                    assert_eq!(
                        channel_of(&largest, pixel, channel),
                        source_value.max(destination_value),
                        "{} max pixel {pixel} channel {channel}",
                        case.name
                    );
                }
            }
        }
    }

    /// The grid the exact arithmetic below counts in: every operand of a blend here is a multiple
    /// of `2^-20` (a source is a multiple of `2^-4`, a destination of `2^-10` or `2^-20`, and a
    /// factor of either or of one minus either), so every product is a multiple of `2^-40` and the
    /// whole rule is exact in it. A rounded value is a multiple of `2^-24` (`half`, down to its
    /// subnormals) or of an `f32` unit in the last place at these magnitudes, both of which the
    /// grid holds, so no step of the rule leaves it.
    const GRID: u32 = 40;

    /// `value` on the [`GRID`], which every operand of this family sits on exactly.
    fn exact(value: f32) -> i128 {
        let scaled = f64::from(value) * 2f64.powi(GRID as i32);
        assert_eq!(scaled.fract(), 0.0, "{value} is not a multiple of 2^-40");
        scaled as i128
    }

    /// The product of two [`GRID`] values, on the grid: exact, because both are multiples of
    /// `2^-20`.
    fn product(left: i128, right: i128) -> i128 {
        let full = left * right;
        assert_eq!(full % (1 << GRID), 0, "a product left the grid");
        full / (1 << GRID)
    }

    /// `value`, a [`GRID`] value, rounded to nearest even onto a binary float format of `mantissa`
    /// fraction bits whose smallest normal exponent is `least`, as a grid value.
    ///
    /// Written out rather than taken from a cast so the rule the test holds is the rule in the
    /// test: a cast to `half` would have to go through a code, and a cast to `f32` of a value this
    /// one computes in `f64` would round twice.
    fn rounded(value: i128, mantissa: u32, least: i32) -> i128 {
        if value == 0 {
            return 0;
        }
        let size = value.unsigned_abs();
        // The exponent `e` with `2^e <= |value| < 2^(e+1)`, from the top set bit.
        let top = 127 - size.leading_zeros() as i32;
        let exponent = (top - GRID as i32).max(least);
        // The format's unit in the last place there, counted in grid steps. One finer than a grid
        // step means the grid already holds every value of the format, so nothing rounds.
        let steps = exponent - mantissa as i32 + GRID as i32;
        if steps <= 0 {
            return value;
        }
        let unit = 1i128 << steps;
        let (whole, part) = ((size / unit as u128) as i128, (size % unit as u128) as i128);
        // Away from zero on a part past the half, and to the even neighbour on a tie. A carry into
        // the next binade needs no arm: the value is then a power of two, which that binade holds.
        let up = part * 2 > unit || (part * 2 == unit && whole % 2 == 1);
        let size = (whole + i128::from(up)) * unit;
        if value < 0 {
            -size
        } else {
            size
        }
    }

    /// A [`GRID`] value as an `f64`, which is exact for every value the rule produces.
    fn value_of(grid: i128) -> f64 {
        grid as f64 / 2f64.powi(GRID as i32)
    }

    /// The eight sites this family's rounding was measured over: every site of it that is
    /// `divergent` on this round's MoltenVK, which is where Metal's rounding and ours part.
    ///
    /// `sas-1msa` appears only for `RG16Float`, which holds no alpha channel, so
    /// `SourceAlphaSaturated` is the smaller of the source alpha and one less the destination's
    /// at every channel here and the alpha channel's own value of one never arises.
    const MEASURED: [(&str, &str); 8] = [
        ("half-output-blends-r16f", "sc-dc"),
        ("half-output-blends-rg16f", "sc-dc"),
        ("half-output-blends-rg16f", "sas-1msa"),
        ("half-output-blends-rgba16f", "sc-dc"),
        ("half-output-blends-rgba16f", "da-1mda"),
        ("half-output-blends-rg32f", "sc-dc"),
        ("half-output-blends-rgba32f", "sc-dc"),
        ("half-output-blends-rgba32f", "da-1mda"),
    ];

    /// Metal's own rounding in a blend of a `half` colour output, held against its recorded codes
    /// at every word of every site of this family that differs from ours.
    ///
    /// **The rule, measured over this family's own recordings:** Metal rounds the **source**
    /// product to the attachment's precision and carries the **destination** product into the sum
    /// unrounded -- `round(round(Sf * S) + Df * D)`, one fused multiply-add on the destination
    /// term. It holds at all 5,376 compared words of the eight sites of [`MEASURED`], in 16- and
    /// 32-bit float formats alike.
    ///
    /// That is what makes this family's divergences more than a half-precision story: the two
    /// 32-bit float formats diverge as well, and `sc-dc`, whose source product `S * S` is exact in
    /// the attachment, still moves by a single code. The runtime's own answer is the mirror image
    /// of the rule -- `round(Sf * S + round(Df * D))`, the destination product rounded and the
    /// source product fused -- which is the gap `known-gaps.tsv` carries for these eight sites.
    ///
    /// **Only these eight sites are held, because only they were measured.** Metal's rounding at
    /// the family's other sites is ours, which is why they are exact, and at two of them
    /// (`sas-1msa` into the two formats that hold an alpha channel) it is neither of the two rules
    /// written here: both sides agree there, so no gap turns on it and no rule is asserted for it.
    /// `one-zero`, `min` and `max` are the test above.
    #[test]
    fn metals_recorded_codes_hold_the_rounding_the_divergent_sites_measure() {
        use metal2vulkan_runtime::MetalLevel;
        let mut checked = 0usize;
        for case in all() {
            let sites: Vec<&str> = MEASURED
                .iter()
                .filter(|(named, _)| *named == case.name())
                .map(|(_, site)| *site)
                .collect();
            if sites.is_empty() {
                continue;
            }
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: one outcome", case.name));
            let blended = taken
                .root
                .child("blended")
                .unwrap_or_else(|| panic!("{}: no blended site", case.name));
            let layout = case.layout();
            let destination = case.destination_bytes();
            // The attachment's own precision, which every rounding of the rule is onto.
            let (mantissa, least) = match case.wide {
                true => (23, -126),
                false => (10, -14),
            };
            let round = |grid: i128| rounded(grid, mantissa, least);
            let stored = |bytes: &[u8], pixel: usize, channel: u32| -> f64 {
                let at = layout.at(0, pixel as u32, channel);
                match case.wide {
                    true => f64::from(f32::from_bits(u32::from_le_bytes(
                        bytes[at..at + 4].try_into().unwrap(),
                    ))),
                    false => f64::from(half_value(u16::from_le_bytes(
                        bytes[at..at + 2].try_into().unwrap(),
                    ))),
                }
            };
            let source_of =
                |pixel: usize, channel: u32| source((pixel + STRIDE * channel as usize) % SOURCES);
            // A destination channel the attachment does not hold reads one, as Metal's and
            // Vulkan's blends both take a missing alpha.
            let destination_of = |pixel: usize, channel: u32| -> f32 {
                if channel >= case.channels {
                    return 1.0;
                }
                let at = layout.at(0, pixel as u32, channel);
                match case.wide {
                    true => f32::from_bits(u32::from_le_bytes(
                        destination[at..at + 4].try_into().unwrap(),
                    )),
                    false => half_value(u16::from_le_bytes(
                        destination[at..at + 2].try_into().unwrap(),
                    )),
                }
            };
            for site in sites {
                let Some(node) = blended.child(site) else {
                    panic!("{}: no {site} site", case.name);
                };
                let Value::Bytes(bytes) = &node.observed.value else {
                    panic!("{}: {site} holds {}", case.name, node.observed.value.kind());
                };
                assert_eq!(bytes.len(), case.level_bytes(), "{} {site}", case.name);
                for pixel in 0..PIXELS {
                    let source_alpha = exact(source_of(pixel, 3));
                    let destination_alpha = exact(destination_of(pixel, 3));
                    let one = exact(1.0);
                    for channel in 0..case.channels {
                        // The alpha channel of `da-1mda`, whose source factor and destination value
                        // are both the destination alpha: neither rule fits Metal there or here,
                        // the two sides agree at every one of those words, and a rule nobody has
                        // measured is not asserted.
                        if site == "da-1mda" && case.channels == 4 && channel == 3 {
                            continue;
                        }
                        let value = exact(source_of(pixel, channel));
                        let held = exact(destination_of(pixel, channel));
                        let (source_factor, destination_factor) = match site {
                            "sc-dc" => (value, held),
                            "da-1mda" => (destination_alpha, one - destination_alpha),
                            "sas-1msa" => (
                                source_alpha.min(one - destination_alpha),
                                one - source_alpha,
                            ),
                            other => panic!("{other}"),
                        };
                        let want = value_of(round(
                            round(product(source_factor, value))
                                + product(destination_factor, held),
                        ));
                        let got = stored(bytes, pixel, channel);
                        assert_eq!(
                            got, want,
                            "{} {site} pixel {pixel} channel {channel}: Metal stored {got}, the \
                             rounding its divergent sites measure gives {want}",
                            case.name
                        );
                        checked += 1;
                    }
                }
            }
        }
        // The eight sites' compared words, less the held-out alpha channel of the two `da-1mda`
        // sites: the count the rule was measured over, so a site losing its recording is loud.
        assert_eq!(checked, 5376, "every compared word of the eight sites");
    }
}
