//! `alphaToCoverageEnabled` and `alphaToOneEnabled`, and the state each of them crosses.
//!
//! Before this family **neither field was reachable**: `crate::render::RenderPipelineSpec` carried
//! no `alpha_to_coverage` and no `alpha_to_one`, so `super::render_descriptors` could state them on
//! a descriptor and read them back, and nothing could put either into effect in a draw. "Blending
//! crossed with alpha-to-coverage" was therefore not an unwritten cross but an unreachable one.
//! The two fields are on the spec now, and this family is what makes them mean something.
//!
//! # The observables, and why there are two
//!
//! Alpha-to-coverage narrows the coverage a pixel's samples were rasterized with, and two
//! different things can be asked about that, so the fixture `a2c_coverage` reports both:
//!
//! - **The coverage the fragment was given**, as an integer, written into a buffer by
//!   `a2c_report`'s `[[sample_mask]]` input, ORed over the fragment's runs (which it counts
//!   beside it). Exact on every device: no resolve and no average is in the way. This turned out
//!   **not** to be the alpha's coverage, which is the first thing this family measured and the
//!   reason the second observable is here — see below.
//! - **What the draw stored**, as the multisample target's **resolve**. A resolve of this family's
//!   one-pixel target averages `k` covered samples against `4 - k` uncovered ones, and it is
//!   compared **exactly**, which is `super::sample_positions`' own ruling for the same shape: a
//!   venue whose resolve rounds differently is a difference worth seeing rather than one worth
//!   tolerating.
//!
//! The two are not the same question, and the gap between them is a measured result rather than a
//! design note:
//!
//! **Alpha-to-coverage is invisible to the fragment's own `[[sample_mask]]` input.** Every
//! four-sample row of this family reports the same coverage word, `0b1111`, and the same four runs,
//! with the field on at any alpha and with it off — an alpha of 0 included, where **nothing** was
//! stored. So the field acts after the fragment, on the fragment's output, as blending does, and a
//! case that tried to read the alpha's coverage from the shader would have measured the rasterizer
//! instead. The report is kept because that independence is worth comparing, and because the run
//! count says Metal shades this fragment **once per sample** at four samples (four runs, not one),
//! which is the shape our own `GraphicsState::sample_shading` claims for a fragment that writes
//! memory.
//!
//! What the alpha's coverage *is* read from is therefore the stored texel, and for the same reason a
//! sample mask the **fragment writes** can only be seen there.
//!
//! # What this measured about Metal
//!
//! On this Apple M3 Max, macOS 27.0.1, with a destination of 1 in every channel and a fragment of
//! `(0.25, 0.5, 0.75, alpha)`:
//!
//! 1. **The coverage is `floor(alpha * rasterSampleCount)` samples.** The alpha sweep is what says
//!    so and it is why the sweep is here rather than one alpha: at four samples an alpha of 0.5
//!    covers two, 0.25 covers one, 0.75 covers three, 0 covers none and 1 covers four, read off the
//!    resolve as five values 0x30 or more apart. The cell that pins `floor` against
//!    round-to-nearest is the **one-sample** row — an alpha of 0.5 at one sample covers **no**
//!    sample and the destination survives untouched, where rounding to nearest would have stored
//!    the fragment. Neither the four-sample rows nor the one-sample row alone says which rule it
//!    is; the pair does, and that is the sample-count cross.
//! 2. **The alpha is read before the attachment's write mask.** A write mask that drops alpha
//!    leaves the coverage exactly as the same alpha gives with the mask absent, and only the stored
//!    alpha moves. A mask sweep cannot ask this and an alpha sweep cannot either.
//! 3. **The alpha's coverage and a sample mask the fragment writes are ANDed.** At alpha 0.5 the
//!    alpha covers samples 0 and 3; under a fragment mask of `0b0101` one sample is stored and
//!    under `0b1010` one is stored, where a coverage derived from the alpha *after* the mask
//!    applied would have stored two in each. This is the family's pair-only cell: with the mask
//!    alone the answer is the mask, with the alpha alone it is the alpha, and only together do they
//!    separate the two orders.
//! 4. **A first output with no alpha component leaves the coverage as rasterized.** The `float2`
//!    fragment into an `RG8Unorm` attachment stores its own value on every sample with
//!    alpha-to-coverage on, so the field is inert where there is no alpha to read. This is an
//!    **independent** cell by Metal's own definition and it is here as the control that the other
//!    rows' coverage came from the alpha.
//! 5. **Alpha-to-one replaces the alpha before the blend reads it.** Under a blend whose RGB
//!    factors are `SourceAlpha` and `OneMinusSourceAlpha`, the stored colour with alpha-to-one off
//!    is the source blended at 0.5 and with it on is the source alone — so the blend read 1 and not
//!    0.5. The field alone only moves the stored alpha; the blend alone gives the 0.5 answer. A
//!    write mask that drops alpha still lets alpha-to-one feed the blend, and only stops the 1
//!    from being stored.
//!
//! # Which rows can detect the field at all, by falsification
//!
//! Measured, not reasoned: with `reads_alpha_to_coverage` forced to `false` in our own leg
//! (`crates/metal2vulkan/src/pipeline/link/graphics.rs`) and nothing else changed, **eight** of
//! this family's rows diverge and the rest stay exact. The eight are every `a2c-on` row of a
//! four-component output below full coverage, the one-sample row included. Two `a2c-on` rows do
//! **not** move, and both are deliberate:
//!
//! - **`a2c-on-4-alpha-100`** — an alpha of 1 covers `floor(1 * 4) = 4` samples, which is every
//!   sample the rasterizer gave, so the field on and off store the same texel. It is the sweep's
//!   endpoint and the proof that the top of the range is full coverage rather than a count the
//!   suite never reached; it is **not** evidence that the field was read, and the rows below it
//!   are what carry that.
//! - **`a2c-on-4-no-alpha-output`** — no alpha component, so there is nothing to narrow.
//!
//! A second falsification separates the pair from its axes. Dropping the alpha's coverage **only
//! where the fragment writes a sample mask** — the "coverage taken after the mask" order — moves
//! exactly **two** rows, `a2c-on-4-alpha-50-shadermask-5` and `-shadermask-a`, by 47 codes, which
//! is the resolve's distance between one covered sample and two. Every single-axis row stays
//! exact: the mask-alone control, the alpha-alone row, the whole sweep and the write-mask cross.
//! That is a divergence only the pair can see, which is the property the cross was authored for.
//!
//! # What turned out independent
//!
//! Said here because a pair that shows nothing is a result:
//!
//! - **`maxVertexAmplificationCount` above 1 with no amplified draw** changes nothing: a pipeline
//!   at a count of 2 drawn as a plain draw stores what a count of 1 stores. Measured in the probe,
//!   not authored, because the cell's answer is one of its axes' alone.
//! - **`rasterSampleCount` disagreeing with the attachment's sample count** is not authored at all:
//!   plain Metal creates the pipeline and draws something — a 1-sample pipeline into a 4-sample
//!   attachment left samples 1 to 3 at neither the clear nor the fragment — and Metal's **API
//!   validation layer ends the process** at `setRenderPipelineState:`
//!   (`the texture sample count (4) does not match the renderPipelineState colorSampleCount (1)`).
//!   A case of it would end a recording run, and a plain-Metal answer that looks undefined is not
//!   one to pin as a conformance expectation. It is in `docs/FAST-PATHS.md`'s known gaps instead.
//!
//! # Nothing here can fault a GPU
//!
//! One one-pixel target a row, four samples at most, one triangle a pass, no loop, and a report
//! buffer of eight bytes indexed by the one pixel's own coordinate.

use crate::allowance::Allowance;
use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};

/// The owned fixture written for this family: a fullscreen triangle, a fragment whose colour comes
/// from a buffer, one that reports the coverage it was given beside that colour, one that reports
/// it and writes a sample mask of its own, and one whose output has no alpha component.
const FIXTURE: &str = "a2c_coverage";

const VERTEX: &str = "a2c_full";
/// Reports its runs and the coverage they were given, and stores the buffer's colour.
const REPORT: &str = "a2c_report";
/// The same, and a sample mask of its own beside the colour.
const REPORT_MASKED: &str = "a2c_report_masked";
/// A `float2` output: no alpha component for alpha-to-coverage to read.
const RG: &str = "a2c_rg";

/// The four-component attachment's format, whose codes Metal's store leaves alone.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;
/// The two-component attachment's format, for the row whose output has no alpha.
const RG_FORMAT: PixelFormat = PixelFormat::RG8Unorm;

/// The fragment's three colour channels, held apart from each other and from the destination, so a
/// channel stored from a neighbour's reads wrong.
const COLOR_RGB: [f32; 3] = [0.25, 0.5, 0.75];

/// The destination every row draws over: the value an **uncovered** sample keeps, so the resolve
/// says how many samples were covered.
///
/// 1 in the three colour channels, which is what makes the resolve readable as a count, and
/// **0.25 in alpha**. The alpha is not 1 for a reason the first recording of this family found: with
/// a destination alpha of 1, `a2one-on-1-mask-no-alpha` stored `0xff` because the mask kept the
/// destination's own 1, and `a2one-on-1-blend-source-alpha` stored `0xff` because alpha-to-one
/// wrote a 1 — two different mechanisms, one byte, and a cell that cannot tell them apart. At 0.25
/// the masked row stores `0x40` and the unmasked one `0xff`.
const DESTINATION: [f64; 4] = [1.0, 1.0, 1.0, 0.25];

/// Which fragment function a row runs, and so which observables it has.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Report {
    /// `a2c_report`: the coverage it was given, and the stored colour.
    Plain,
    /// `a2c_report_masked`: the same, and the row's own sample mask beside the colour.
    Masked(u32),
    /// `a2c_rg`: a `float2` output, no coverage report and no alpha.
    NoAlpha,
}

/// One row of the family.
#[derive(Clone, Copy)]
struct Row {
    name: &'static str,
    /// `alphaToCoverageEnabled`.
    alpha_to_coverage: bool,
    /// `alphaToOneEnabled`.
    alpha_to_one: bool,
    /// `rasterSampleCount`, which is the attachment's sample count too.
    samples: u32,
    /// The fragment's alpha, which is what alpha-to-coverage reads.
    alpha: f32,
    /// The attachment's `writeMask`.
    write_mask: ColorWriteMask,
    /// Which fragment function runs.
    report: Report,
    /// `blendingEnabled` with RGB factors that read the **source alpha**, which is how a row sees
    /// what alpha-to-one fed the blend.
    blended: bool,
}

impl Row {
    /// A four-sample row of `a2c_report` at `alpha`, every channel written and no blend: the shape
    /// every alpha of the sweep takes.
    const fn sweep(name: &'static str, alpha_to_coverage: bool, alpha: f32) -> Self {
        Self {
            name,
            alpha_to_coverage,
            alpha_to_one: false,
            samples: 4,
            alpha,
            write_mask: ColorWriteMask::ALL,
            report: Report::Plain,
            blended: false,
        }
    }
}

/// Every row, grouped by the question it answers. Each group's prose is in the module header.
const ROWS: &[Row] = &[
    // The control: the same alpha with the field off, so every coverage below is the field's doing.
    Row::sweep("a2c-off-4-alpha-50", false, 0.5),
    // The sweep that gives `floor(alpha * samples)`.
    Row::sweep("a2c-on-4-alpha-00", true, 0.0),
    Row::sweep("a2c-on-4-alpha-25", true, 0.25),
    Row::sweep("a2c-on-4-alpha-50", true, 0.5),
    Row::sweep("a2c-on-4-alpha-75", true, 0.75),
    Row::sweep("a2c-on-4-alpha-100", true, 1.0),
    // The sample-count cross: the cell that pins `floor` against round-to-nearest.
    Row {
        name: "a2c-on-1-alpha-50",
        samples: 1,
        ..Row::sweep("a2c-on-1-alpha-50", true, 0.5)
    },
    Row {
        name: "a2c-off-1-alpha-50",
        samples: 1,
        ..Row::sweep("a2c-off-1-alpha-50", false, 0.5)
    },
    // The write-mask cross: the coverage must not move, and only the stored alpha may.
    Row {
        name: "a2c-on-4-alpha-50-mask-no-alpha",
        write_mask: ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE),
        ..Row::sweep("a2c-on-4-alpha-50-mask-no-alpha", true, 0.5)
    },
    // The sample-mask cross, with its own control: the AND is only visible in the stored texel.
    Row {
        name: "a2c-off-4-alpha-50-shadermask-5",
        report: Report::Masked(0b0101),
        ..Row::sweep("a2c-off-4-alpha-50-shadermask-5", false, 0.5)
    },
    Row {
        name: "a2c-on-4-alpha-50-shadermask-5",
        report: Report::Masked(0b0101),
        ..Row::sweep("a2c-on-4-alpha-50-shadermask-5", true, 0.5)
    },
    Row {
        name: "a2c-on-4-alpha-50-shadermask-a",
        report: Report::Masked(0b1010),
        ..Row::sweep("a2c-on-4-alpha-50-shadermask-a", true, 0.5)
    },
    // The independent cell: no alpha component, so nothing for the field to read.
    Row {
        name: "a2c-on-4-no-alpha-output",
        report: Report::NoAlpha,
        ..Row::sweep("a2c-on-4-no-alpha-output", true, 0.5)
    },
    Row {
        name: "a2c-off-4-no-alpha-output",
        report: Report::NoAlpha,
        ..Row::sweep("a2c-off-4-no-alpha-output", false, 0.5)
    },
    // Alpha-to-one, and the blend that says what it fed.
    Row {
        name: "a2one-off-1-blend-source-alpha",
        samples: 1,
        blended: true,
        ..Row::sweep("a2one-off-1-blend-source-alpha", false, 0.5)
    },
    Row {
        name: "a2one-on-1-blend-source-alpha",
        samples: 1,
        alpha_to_one: true,
        blended: true,
        ..Row::sweep("a2one-on-1-blend-source-alpha", false, 0.5)
    },
    Row {
        name: "a2one-on-1-mask-no-alpha",
        samples: 1,
        alpha_to_one: true,
        write_mask: ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE),
        ..Row::sweep("a2one-on-1-mask-no-alpha", false, 0.5)
    },
];

/// One case of the family.
pub struct AlphaCoverageCase {
    row: Row,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<AlphaCoverageCase> {
    ROWS.iter()
        .map(|row| AlphaCoverageCase { row: *row })
        .collect()
}

/// The blend whose RGB factors read the **source alpha**: what alpha-to-one fed it is readable in
/// the stored colour, and nothing else of the pipeline can say it.
const SOURCE_ALPHA_BLEND: Blend = Blend {
    source_rgb: BlendFactor::SourceAlpha,
    destination_rgb: BlendFactor::OneMinusSourceAlpha,
    rgb: BlendOperation::Add,
    source_alpha: BlendFactor::One,
    destination_alpha: BlendFactor::Zero,
    alpha: BlendOperation::Add,
};

impl AlphaCoverageCase {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The fragment function the row runs.
    fn fragment(&self) -> &'static str {
        match self.row.report {
            Report::Plain => REPORT,
            Report::Masked(_) => REPORT_MASKED,
            Report::NoAlpha => RG,
        }
    }

    /// The attachment's format: two components for the row whose output has no alpha.
    fn format(&self) -> PixelFormat {
        match self.row.report {
            Report::NoAlpha => RG_FORMAT,
            _ => FORMAT,
        }
    }

    /// The bytes of one texel of the attachment's format.
    fn texel_bytes(&self) -> u32 {
        match self.row.report {
            Report::NoAlpha => 2,
            _ => 4,
        }
    }

    /// The one color state the pipeline holds.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        let mut state = ColorState::new(self.format()).masked(self.row.write_mask);
        if self.row.blended {
            state = state.blended(SOURCE_ALPHA_BLEND);
        }
        [Some(state)]
    }

    /// The pipeline the row makes.
    fn pipeline_spec<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        let mut spec =
            RenderPipelineSpec::new(VERTEX, self.fragment(), colors).multisampled(self.row.samples);
        if self.row.alpha_to_coverage {
            spec = spec.covering_by_alpha();
        }
        if self.row.alpha_to_one {
            spec = spec.alpha_to_one();
        }
        spec
    }

    /// The fragment's colour, as the four floats buffer 0 holds.
    fn color_bytes(&self) -> Vec<u8> {
        [COLOR_RGB[0], COLOR_RGB[1], COLOR_RGB[2], self.row.alpha]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect()
    }

    /// The multisample target, and the single-sampled texture its resolve lands in. A row of one
    /// sample has no resolve: its own texels are what it reads.
    fn target_spec(&self) -> TargetSpec {
        let spec = TargetSpec::plain(self.format(), [1, 1]).multisampled(self.row.samples);
        // A one-sample row reads this texture itself, so it is of shared storage: Metal's own API
        // validation stops `getBytes:` on a private texture, and a case that asks Metal something
        // its validation layer refuses records a stop it has no use for. A multisample row reads
        // its resolve instead and leaves this one private, which is what a render target is.
        match self.row.samples > 1 {
            true => spec,
            false => spec.shared(),
        }
    }

    /// The layout the one texel is read under: one element a channel.
    fn texel_layout(&self) -> ByteLayout {
        let row_pitch = self.texel_bytes();
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [1, 1, 1],
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: row_pitch,
            row_pitch,
            rows: 1,
            image_pitch: row_pitch,
            images: 1,
            gaps: Gaps::Compared,
        }
    }

    /// The layout the report buffer's two words are read under: the runs, then the coverage.
    fn report_layout() -> ByteLayout {
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: 8,
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: 2,
            row_pitch: 2,
            rows: 1,
            image_pitch: 2,
            images: 1,
            gaps: Gaps::Compared,
        }
    }
}

impl Case for AlphaCoverageCase {
    fn name(&self) -> &'static str {
        self.row.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened. The coverage report is an integer, and the resolve of `k` covered
        // samples against `4 - k` uncovered is compared exactly, which is this suite's ruling for
        // the same shape in `super::sample_positions`.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        let fragment = self.fragment();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: format!("{VERTEX}+{fragment}"),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let colors = self.color_state();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &self.pipeline_spec(&colors).words()),
            Input::bytes("target", &self.target_spec().words()),
            Input::bytes("buffer:0", &self.color_bytes()),
        ];
        if self.row.report != Report::NoAlpha {
            // The width the fragment indexes its report by, and the report's own zeroed words.
            inputs.push(Input::bytes("buffer:1", &1u32.to_le_bytes()));
            inputs.push(Input::bytes("buffer:2", &[0u8; 8]));
        }
        if let Report::Masked(mask) = self.row.report {
            inputs.push(Input::bytes("buffer:3", &mask.to_le_bytes()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, failure: Failure| Abort {
            at: at.to_owned(),
            failure,
        };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let colors = self.color_state();
        let state = crate::case::made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &self.pipeline_spec(&colors)),
        )?;
        let target = crate::case::made(sites, "target", side.target(&self.target_spec()))?;
        let color = crate::case::made(sites, "buffer:0", side.buffer(&self.color_bytes()))?;

        let mut steps = vec![
            RenderStep::Pipeline(state),
            RenderStep::FragmentBuffer {
                buffer: color,
                offset: 0,
                index: 0,
            },
        ];
        let mut report = None;
        if self.row.report != Report::NoAlpha {
            let width = crate::case::made(sites, "buffer:1", side.buffer(&1u32.to_le_bytes()))?;
            let held = crate::case::made(sites, "buffer:2", side.buffer(&[0u8; 8]))?;
            report = Some(held);
            steps.push(RenderStep::FragmentBuffer {
                buffer: width,
                offset: 0,
                index: 1,
            });
            steps.push(RenderStep::FragmentBuffer {
                buffer: held,
                offset: 0,
                index: 2,
            });
        }
        if let Report::Masked(mask) = self.row.report {
            let held = crate::case::made(sites, "buffer:3", side.buffer(&mask.to_le_bytes()))?;
            steps.push(RenderStep::FragmentBuffer {
                buffer: held,
                offset: 0,
                index: 3,
            });
        }
        steps.push(RenderStep::triangle());

        // A multisampled row resolves into a single-sampled texture of its own, which is what the
        // host reads; a one-sample row reads its own target.
        let resolve = match self.row.samples > 1 {
            true => Some(crate::case::made(
                sites,
                "resolve",
                side.target(&TargetSpec::plain(self.format(), [1, 1]).shared()),
            )?),
            false => None,
        };
        let mut attachment = PassColor::cleared(target, DESTINATION);
        if let Some(into) = resolve {
            attachment = attachment.resolving(into);
        }
        let outcome = side
            .render(&[RenderWork::Pass(
                RenderPass::clearing(vec![attachment]).drawing(steps),
            )])
            .map_err(|failure| stop("drawn/pass", failure))?;

        let mut node = Node::branch("drawn", Vec::new());
        let mut calls = Node::branch("calls", Vec::new());
        for (index, refused) in outcome.refused.iter().enumerate() {
            calls = calls.with(Node::leaf(
                format!("{index}-render"),
                Observed::value(match refused {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(calls);
        if let Some(failure) = outcome.refused.iter().flatten().next() {
            sites.push(node);
            return Err(Abort {
                at: "drawn/calls/0-render".to_owned(),
                failure: failure.clone(),
            });
        }
        node = node.with(Node::leaf(
            "status",
            Observed::value(Value::ok(outcome.status.clone())),
        ));
        // What the draw stored.
        let read = resolve.unwrap_or(target);
        node = node.with(Node::leaf(
            "texel",
            Observed::bytes(
                side.read_texture(read, [1, 1], self.texel_bytes())
                    .map_err(|failure| stop("drawn/texel", failure))?,
                self.texel_layout(),
            ),
        ));
        // And the coverage the fragment was given, with the runs that read it.
        if let Some(held) = report {
            node = node.with(Node::leaf(
                "coverage",
                Observed::bytes(
                    side.read_buffer(held, 0, 8)
                        .map_err(|failure| stop("drawn/coverage", failure))?,
                    Self::report_layout(),
                ),
            ));
        }
        sites.push(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The model each row's answer is held against, and the guards that keep the crosses honest.
    //!
    //! The coverage model is Metal's, measured: `floor(alpha * rasterSampleCount)` samples, ANDed
    //! with a sample mask the fragment writes. The tests derive each row's expected coverage from
    //! that one rule rather than listing answers, so a row that agrees for the wrong reason fails.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// How many samples `alpha` covers at `samples`, by the measured rule.
    fn covered(alpha: f32, samples: u32) -> u32 {
        (alpha * samples as f32).floor() as u32
    }

    /// The red code a resolve of `k` covered samples out of `samples` leaves: the fragment's own
    /// red against the destination's, averaged, as Metal's fixed-function resolve averages them.
    ///
    /// Derived rather than tabulated, so the count a recording is read as is a consequence of the
    /// two inputs the case states and not a number anybody typed.
    fn resolved_red(k: u32, samples: u32) -> f64 {
        let source = f64::from(COLOR_RGB[0]);
        let destination = DESTINATION[0];
        (f64::from(k) * source + f64::from(samples - k) * destination) / f64::from(samples)
    }

    /// How many samples the recorded red code says were covered, of `samples`.
    ///
    /// The candidates are at least 0x30 apart at four samples, so the nearest one is unambiguous
    /// and a venue whose resolve rounds a code differently is still read as the same count. The
    /// recording itself is compared byte for byte by the suite; this only interprets it.
    fn covered_from_red(red: u8, samples: u32) -> u32 {
        let (mut best, mut distance) = (0, f64::INFINITY);
        for k in 0..=samples {
            let want = resolved_red(k, samples) * 255.0;
            let found = (want - f64::from(red)).abs();
            if found < distance {
                (best, distance) = (k, found);
            }
        }
        assert!(
            distance <= 1.5,
            "a red code of {red} is no resolve of 0..={samples} covered samples"
        );
        best
    }

    /// The one outcome of `name`'s recorded Metal answer.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The bytes the site at `path` holds.
    fn bytes(transcript: &Transcript, path: &str) -> Vec<u8> {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("no site {path} (stopped at {name})"));
        }
        let Value::Bytes(found) = &node.observed.value else {
            panic!("{path} holds no bytes");
        };
        found.clone()
    }

    /// The texel `name`'s recording stored.
    fn stored(name: &str) -> Vec<u8> {
        bytes(&only_outcome(name), "case/drawn/texel")
    }

    /// The row named `name`.
    fn row(name: &str) -> Row {
        *ROWS
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("no row {name}"))
    }

    /// How many samples `name`'s recording says were covered.
    fn covered_in(name: &str) -> u32 {
        covered_from_red(stored(name)[0], row(name).samples)
    }

    /// The runs and the coverage word `name`'s report buffer holds.
    fn report_of(name: &str) -> (u32, u32) {
        let found = bytes(&only_outcome(name), "case/drawn/coverage");
        assert_eq!(found.len(), 8, "{name}: the report is two words");
        let word = |at: usize| u32::from_le_bytes(found[at..at + 4].try_into().unwrap());
        (word(0), word(4))
    }

    /// Metal covers `floor(alpha * rasterSampleCount)` samples in every row of the sweep, and the
    /// one-sample row is what separates that rule from rounding to nearest.
    #[test]
    fn metals_coverage_is_the_floor_of_the_alpha_times_the_samples() {
        for held in ROWS {
            // The coverage rows only: a blended row's red is the blend's result rather than an
            // average of covered samples, and an alpha-to-one row's is its own question.
            if held.report != Report::Plain || held.blended || held.alpha_to_one {
                continue;
            }
            let wanted = match held.alpha_to_coverage {
                // The field off: every rasterized sample is covered, whatever the alpha.
                false => held.samples,
                true => covered(held.alpha, held.samples),
            };
            assert_eq!(
                covered_in(held.name),
                wanted,
                "{}: not {wanted} covered samples",
                held.name
            );
        }
        // The cell that pins `floor`: the same alpha covers two of four samples and none of one,
        // where rounding to nearest would have covered the one.
        assert_eq!(covered(0.5, 4), 2);
        assert_eq!(covered(0.5, 1), 0);
        assert_eq!(covered_in("a2c-on-4-alpha-50"), 2);
        assert_eq!(covered_in("a2c-on-1-alpha-50"), 0);
        assert_eq!(covered_in("a2c-off-1-alpha-50"), 1);
        // And the sweep is a sweep: the five alphas give five different counts, so no two rows of
        // it could be passed by each other's answer.
        let counts: Vec<u32> = ["00", "25", "50", "75", "100"]
            .iter()
            .map(|a| covered_in(&format!("a2c-on-4-alpha-{a}")))
            .collect();
        assert_eq!(counts, vec![0, 1, 2, 3, 4]);
    }

    /// Alpha-to-coverage is invisible to the fragment's own `[[sample_mask]]` input: every
    /// four-sample row reports the same coverage word and the same runs with the field on and off.
    ///
    /// An independence, measured, and the family's reason for keeping the report: a case that read
    /// the alpha's coverage from the shader would have been reading the rasterizer.
    #[test]
    fn the_field_does_not_reach_the_fragments_own_coverage_input() {
        let off = report_of("a2c-off-4-alpha-50");
        assert_eq!(
            off,
            (4, 0b1111),
            "the control's own report is not four full runs"
        );
        for held in ROWS {
            if held.report == Report::NoAlpha || held.samples != 4 {
                continue;
            }
            assert_eq!(
                report_of(held.name),
                off,
                "{}: the field moved the coverage the fragment was given",
                held.name
            );
        }
        // Including the row that stored nothing at all, which is the strongest form of it.
        assert_eq!(covered_in("a2c-on-4-alpha-00"), 0);
        assert_eq!(report_of("a2c-on-4-alpha-00"), off);
        // At one sample the fragment runs once and reads one sample, with the field on and off.
        assert_eq!(report_of("a2c-on-1-alpha-50"), (1, 0b1));
        assert_eq!(report_of("a2c-off-1-alpha-50"), (1, 0b1));
    }

    /// The write mask does not move the coverage the alpha gives: Metal reads the alpha before it.
    ///
    /// A pair-only claim, and the assertion is the pair: the three colour channels are **byte for
    /// byte** the unmasked row's, so the same samples were covered, while the stored alpha is the
    /// destination's instead of the fragment's.
    #[test]
    fn the_write_mask_does_not_move_the_coverage_the_alpha_gives() {
        let plain = stored("a2c-on-4-alpha-50");
        let masked = stored("a2c-on-4-alpha-50-mask-no-alpha");
        assert_eq!(
            plain[..3],
            masked[..3],
            "a write mask that drops alpha moved the samples the alpha covered"
        );
        assert_ne!(
            plain[3], masked[3],
            "the write mask changed no stored channel, so the row proves nothing"
        );
        // And what the masked row kept is the destination's own alpha, which is the only thing a
        // dropped channel can hold.
        let destination = (DESTINATION[3] * 255.0).round() as u8;
        assert_eq!(
            masked[3], destination,
            "the dropped channel is not the destination's"
        );
    }

    /// The alpha's coverage and the fragment's own sample mask AND: together they store **fewer**
    /// samples than either does alone.
    ///
    /// This is the family's pair-only cell and the falsification is in the arithmetic. A coverage
    /// derived from the alpha **after** the mask applied would cover `floor(0.5 * 4) = 2` of the
    /// masked samples and store two — exactly what the mask-alone control stores. It stores one.
    #[test]
    fn the_alpha_coverage_and_the_fragments_sample_mask_are_anded() {
        let alpha_alone = covered_in("a2c-on-4-alpha-50");
        let mask_alone = covered_in("a2c-off-4-alpha-50-shadermask-5");
        assert_eq!(alpha_alone, 2, "the alpha alone covers two samples");
        assert_eq!(mask_alone, 2, "the mask alone covers two samples");
        for name in [
            "a2c-on-4-alpha-50-shadermask-5",
            "a2c-on-4-alpha-50-shadermask-a",
        ] {
            let both = covered_in(name);
            assert_eq!(both, 1, "{name}: the pair must narrow to one sample");
            assert!(
                both < alpha_alone && both < mask_alone,
                "{name}: the pair stored no fewer than one of its axes, so it sees no order"
            );
            // And the stored texel differs from the mask-alone control's, which is the model the
            // pair rules out.
            assert_ne!(
                stored(name),
                stored("a2c-off-4-alpha-50-shadermask-5"),
                "{name}: the pair stored what a coverage taken after the mask would store"
            );
        }
        // The two masks pick disjoint halves and both narrow to one, which is what a coverage of
        // samples 0 and 3 gives: `0b1001 & 0b0101` and `0b1001 & 0b1010` are one sample each.
        assert_eq!((0b1001u32 & 0b0101).count_ones(), 1);
        assert_eq!((0b1001u32 & 0b1010).count_ones(), 1);
    }

    /// A first output with no alpha component leaves the coverage as rasterized, so the field is
    /// inert there: the two `no-alpha-output` rows store the identical texel.
    ///
    /// The family's one indistinguishable-by-construction cell, named rather than dropped: there is
    /// no alpha for alpha-to-coverage to read, so Metal has nothing to narrow. The texel is the
    /// fragment's own two channels, so the rows are a draw that happened rather than two empty
    /// targets agreeing.
    #[test]
    fn a_fragment_with_no_alpha_component_is_untouched_by_the_field() {
        let on = stored("a2c-on-4-no-alpha-output");
        assert_eq!(
            on,
            stored("a2c-off-4-no-alpha-output"),
            "alpha-to-coverage moved a fragment that writes no alpha"
        );
        let code = |c: f32| (c * 255.0).round() as u8;
        assert_eq!(
            on,
            vec![code(COLOR_RGB[0]), code(COLOR_RGB[1])],
            "the row stored neither the fragment's channels nor the destination's"
        );
    }

    /// Alpha-to-one replaces the source alpha before the blend reads it, which neither the field
    /// nor the blend can say alone.
    #[test]
    fn alpha_to_one_feeds_the_blend_before_it_reads_the_source_alpha() {
        let off = stored("a2one-off-1-blend-source-alpha");
        let on = stored("a2one-on-1-blend-source-alpha");
        assert_ne!(
            off, on,
            "alpha-to-one changed nothing a blend of the source alpha stored"
        );
        let code = |c: f64| (c * 255.0).round() as u8;
        // With the field on the RGB factors are 1 and 0, so the stored colour is the source's own
        // three channels, and the stored alpha is the 1 the field wrote.
        for (channel, source) in COLOR_RGB.iter().enumerate() {
            assert_eq!(
                on[channel],
                code(f64::from(*source)),
                "channel {channel}: the blend did not read an alpha of 1"
            );
        }
        assert_eq!(on[3], 0xff, "the field stored no alpha of 1");
        // With it off the same blend reads the fragment's own 0.5 against a destination of 1, which
        // is a different colour in every channel.
        for (channel, source) in COLOR_RGB.iter().enumerate() {
            let alpha = f64::from(row("a2one-off-1-blend-source-alpha").alpha);
            let want = f64::from(*source) * alpha + DESTINATION[channel] * (1.0 - alpha);
            assert!(
                off[channel].abs_diff(code(want)) <= 1,
                "channel {channel}: {} is no blend of the source at its own alpha",
                off[channel]
            );
        }
    }

    /// A write mask that drops alpha stops alpha-to-one's 1 from being **stored** while the blend
    /// still reads it, which is the three-way cell the destination's own alpha makes readable.
    #[test]
    fn the_write_mask_stops_alpha_to_ones_store_and_not_its_read() {
        let masked = stored("a2one-on-1-mask-no-alpha");
        let unmasked = stored("a2one-on-1-blend-source-alpha");
        // The colour channels agree: the field reached the pipeline in both.
        assert_eq!(
            masked[..3],
            unmasked[..3],
            "the two rows stored different colours"
        );
        // The stored alpha does not: the masked row kept the destination's.
        let destination = (DESTINATION[3] * 255.0).round() as u8;
        assert_eq!(
            masked[3], destination,
            "the mask did not stop the field's store"
        );
        assert_ne!(
            masked[3], unmasked[3],
            "the destination's alpha is 1, so this cell cannot tell the two apart"
        );
    }

    /// Every row fingerprints its own inputs, so no recording can be read for another row, and no
    /// row is loosened.
    #[test]
    fn every_row_fingerprints_apart_and_nothing_is_loosened() {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for case in all() {
            assert!(
                case.allowances().is_empty(),
                "{}: a loosened site",
                case.name()
            );
            case.shaders().unwrap();
            let colors = case.color_state();
            let mut words = case.pipeline_spec(&colors).words();
            // The colour buffer and the fragment's own mask are inputs rather than pipeline state,
            // so the key a row is read by is the pipeline's words beside them.
            for input in case.inputs().unwrap() {
                words.extend(input.name.as_bytes());
                words.extend(input.digest.hex.as_bytes());
            }
            assert!(
                !seen.contains(&words),
                "{}: fingerprints exactly as another row",
                case.name()
            );
            seen.push(words);
        }
        assert_eq!(seen.len(), ROWS.len());
    }
}
