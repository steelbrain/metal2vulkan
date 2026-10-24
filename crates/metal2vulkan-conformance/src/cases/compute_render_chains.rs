//! What crosses the boundary between a **compute dispatch and a render pass** of one command
//! buffer, in both directions.
//!
//! Programs cross that boundary constantly: a kernel builds a texture and a pass samples it; a
//! pass renders a target and a kernel reads it. Each direction is a hazard our side has to turn
//! into a barrier and an image-layout transition, and Metal states neither.
//!
//! **What the suite already asked, and what it did not.** One direction is covered:
//! [`crate::cases::extended_range_targets`] and [`crate::cases::extended_range_blends`] put a
//! dispatch after a pass in one command buffer and read the just-rendered target with a kernel, so
//! a colour attachment's store reaching a kernel's texture read is not new here. The other
//! direction -- a kernel's writes reaching a **fragment stage's sampled read** -- was asked by
//! nothing, and neither was a fragment function's own shader-write reaching a kernel.
//! `pass-writes-then-kernel-reads` is therefore the row that adds least, and it is kept for a
//! reason it earns rather than for symmetry: it is the one of the three whose answer is a
//! **blit's** too, so it says the kernel read what a blit reads of the same texture, which is the
//! control the other two rows' equalities are measured against.
//!
//! Three rows, one for each way the two kinds of work hand a texture over:
//!
//! - **`kernel-writes-then-fragment-samples`**: a kernel writes every texel of a private texture,
//!   and a pass in the same command buffer samples that texture through a sampler. The sampled
//!   target must come back equal to the written one, which it can only do if the dispatch's writes
//!   are visible to the fragment stage and the image reached the layout a sampled read needs. This
//!   is the direction nothing asked.
//! - **`pass-writes-then-kernel-reads`**: a pass draws four bands into a private render target,
//!   and a kernel in the same command buffer reads every texel of it into a buffer, which must
//!   hold what a blit of the same target holds. The hazard is the one the extended-range families
//!   already reach; what this row adds is the blit beside it.
//! - **`fragment-writes-then-kernel-reads`**: a **fragment** function writes a texture through a
//!   shader-write binding rather than through a colour attachment, and a kernel then reads it. The
//!   write leaves the pass by a different door from the attachment's store, so it is its own
//!   hazard and not a second spelling of the row above.
//!
//! # Why every target here is `RGBA32Float`
//!
//! `tob_solid` returns the `float4` its buffer holds and changes nothing about it, and a
//! `RGBA32Float` target stores a `float` as it is. So the colours a band holds are **exactly the
//! bytes this module wrote into the buffer**, with no normalized conversion and no rounding rule
//! of anybody's in the way, and the rows' cross-checks are arithmetic rather than a measurement
//! adopted from the run. It also lets a band state a channel outside `[0, 1]`, which a normalized
//! target could not carry, and two of the four do.
//!
//! The one value in the family that a shader computes is `tob_write`'s, which scales a pixel's
//! coordinate by `0.125` -- a power of two, so it is exact whatever math mode the fixture is
//! compiled under. `scout_write_f`'s values are **not** cross-checked arithmetically for the
//! opposite reason: they divide by 7 and by 5, and whether a device's compiler turns that into a
//! reciprocal multiply is not this suite's business. What the row holds about them is that the
//! sample reproduced them, which is the question.
//!
//! Nothing here can fault a GPU: 16 by 16 textures, one dispatch of 256 threads, one or four
//! draws, no loop in any function.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, Dispatch, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The format every texture of the family carries: a `float` stored as it is, so nothing of a
/// normalized conversion stands between a value and its cross-check.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The side, in texels, of every texture of the family.
///
/// Sixteen because `scout_read_f` writes texel `(x, y)` into word `y * 16 + x` of its buffer: the
/// stride is the fixture's own and a case at another extent would be reading a strided layout
/// while claiming to read a texture.
const SIDE: u32 = 16;

/// The bands a filling pass draws.
const BANDS: u32 = 4;

/// Bytes one texture's whole level 0 takes, and one kernel's buffer of one `float4` a texel.
const PLANE: u32 = TEXEL * SIDE * SIDE;

/// The four colours a filling pass draws, each channel an exact binary fraction so a `float` holds
/// it without rounding, and two of them outside `[0, 1]` because a `RGBA32Float` target carries
/// what a normalized one could not.
const COLORS: [[f32; 4]; BANDS as usize] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.125, 0.375, 0.625, 0.875],
    [2.5, -1.25, 0.0, 1.5],
    [-0.5, 3.75, 0.5, 0.625],
];

/// The clear a filling pass loads its target with: outside every band's every channel, so a band
/// that was never drawn reads as this.
const CLEAR: f64 = -7.5;

/// The owned fixture of the rectangle, the solid colour, the sample and the fragment's own texture
/// write.
const TOB: &str = "tob_renders";

/// The owned fixture of the kernels that write and read a texture texel by texel.
const SCOUT: &str = "format_scout";

/// `tob_rect`: the rectangle both of this family's passes draw.
const RECT: &str = "tob_rect";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// Which way one row hands a texture between the two kinds of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Crossing {
    /// A kernel writes the texture; a pass samples it.
    KernelToFragment,
    /// A pass's colour attachment holds the texture; a kernel reads it.
    AttachmentToKernel,
    /// A fragment function writes the texture through a shader-write binding; a kernel reads it.
    FragmentWriteToKernel,
}

/// What one row asks.
#[derive(Clone, Copy)]
struct Subject {
    name: &'static str,
    crossing: Crossing,
}

/// The three rows, in the order the report lists them.
const SUBJECTS: &[Subject] = &[
    Subject {
        name: "kernel-writes-then-fragment-samples",
        crossing: Crossing::KernelToFragment,
    },
    Subject {
        name: "pass-writes-then-kernel-reads",
        crossing: Crossing::AttachmentToKernel,
    },
    Subject {
        name: "fragment-writes-then-kernel-reads",
        crossing: Crossing::FragmentWriteToKernel,
    },
];

/// One row of the family.
pub struct ComputeRenderChain {
    subject: Subject,
}

/// Every row of the family, in the order the report lists them.
pub fn all() -> Vec<ComputeRenderChain> {
    SUBJECTS
        .iter()
        .map(|subject| ComputeRenderChain { subject: *subject })
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

/// Band `at` of [`BANDS`], as `tob_rect` takes a rectangle: the two clip-space corners.
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

/// The buffer the solid fragment function reads its colour out of: the four colours as `float4`s.
fn colors_buffer() -> Vec<u8> {
    COLORS
        .iter()
        .flat_map(|color| {
            color
                .iter()
                .flat_map(|channel| channel.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The one colour attachment every pipeline of the family carries.
const COLOR_STATE: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The sampler the sampling row binds: Metal's own defaults, nearest and clamped to the edge.
///
/// Nearest rather than linear, and not for want of a filtering question: Metal filters no 32-bit
/// float texture, so a linear sampler over a `RGBA32Float` texture asks a different thing
/// altogether. The filtering question belongs to the family that asks it
/// ([`crate::cases::render_chains`]), and is not what this one is about.
fn sampler() -> SamplerSpec {
    SamplerSpec::new()
}

/// A texture of the family: the usage and storage one of its roles needs.
fn texture_spec(crossing: Crossing) -> TargetSpec {
    match crossing {
        // Sampled by a fragment stage and written by a kernel.
        Crossing::KernelToFragment => TargetSpec::read(FORMAT, [SIDE; 2]).written(),
        // Rendered into and read by a kernel.
        Crossing::AttachmentToKernel => TargetSpec::read(FORMAT, [SIDE; 2]),
        // Written by a fragment stage and read by a kernel.
        Crossing::FragmentWriteToKernel => TargetSpec::read(FORMAT, [SIDE; 2]).written(),
    }
}

/// The layout a texture's or a kernel buffer's plane of [`SIDE`] by [`SIDE`] `float4`s is read
/// under: one row of the layout a row of texels, one element a channel.
fn plane_layout(region: Region) -> ByteLayout {
    let row_pitch = TEXEL * SIDE;
    ByteLayout {
        region,
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The region a texture's whole level 0 is.
fn texture_region() -> Region {
    Region::Texture {
        level: 0,
        slice: 0,
        aspect: Aspect::Color,
        sample: 0,
        origin: [0, 0, 0],
        size: [SIDE, SIDE, 1],
    }
}

/// The region a kernel's whole output buffer is.
fn buffer_region() -> Region {
    Region::Buffer {
        offset: 0,
        length: u64::from(PLANE),
    }
}

impl Subject {
    /// The fragment function the row's pass runs, and the kernel its dispatch runs.
    fn functions(&self) -> (&'static str, &'static str) {
        match self.crossing {
            Crossing::KernelToFragment => ("tob_sample", "scout_write_f"),
            Crossing::AttachmentToKernel => ("tob_solid", "scout_read_f"),
            Crossing::FragmentWriteToKernel => ("tob_write", "scout_read_f"),
        }
    }

    /// Whether either function the row runs divides a `float`, which is the shape that decides
    /// its `device-arithmetic` declaration.
    ///
    /// `tob_sample` divides the pixel's position by the sampled texture's extent and
    /// `scout_write_f` divides a coordinate by 7 and by 5; `tob_solid`, `tob_write` and
    /// `scout_read_f` divide nothing, and the two rows made of those compare exact on both venues
    /// measured.
    fn divides_a_float(&self) -> bool {
        matches!(self.crossing, Crossing::KernelToFragment)
    }

    /// The render pipeline the row makes.
    fn pipeline(&self) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::new(RECT, self.functions().0, &COLOR_STATE)
    }
}

impl Case for ComputeRenderChain {
    fn name(&self) -> &'static str {
        self.subject.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. A `RGBA32Float` texel is the `float` the stage produced, so every
        // value of every row is exact or it is a bug, and the one notice a stored colour could
        // carry -- the normalized conversion -- cannot apply, because nothing here stores a
        // normalized channel.
        // `device-arithmetic` is declared by the rows whose functions do the arithmetic the note
        // covers: our side records it where a float division, remainder, square root, fused
        // multiply-add or dot product takes Vulkan's precision bounds rather than correct rounding,
        // which happens on a device whose float functions are not Metal's own
        // (`metal2vulkan::pipeline::artifact`'s `PrecisionNote::DeviceArithmetic`). Which device
        // that is, is the device's answer; which rows can announce it is the case's, and it is the
        // ones whose functions divide a float.
        //
        // Measured: it fires on lavapipe and not on MoltenVK on this M3 Max, and on lavapipe every
        // value compared **exactly** -- the whole difference was the announcement, which is what
        // makes declaring it a declaration and not a tolerance. Metal announces nothing, so a
        // notice of ours has to be declared or the comparison cannot be made at all.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 3 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![crate::replay::shared_memory_notice()];
        if self.subject.divides_a_float() {
            named.push(crate::replay::notice_allowance(
                Degradation::DeviceArithmetic.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let (fragment, kernel) = self.subject.functions();
        let mut shaders = Vec::new();
        for (name, entry) in [
            (TOB, format!("{RECT}+{fragment}")),
            (SCOUT, kernel.to_owned()),
        ] {
            let fixture = Fixture::new(name);
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
        for name in [TOB, SCOUT] {
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
        inputs.push(Input::bytes("render-pipeline", &subject.pipeline().words()));
        inputs.push(Input::bytes("kernel", subject.functions().1.as_bytes()));
        inputs.push(Input::bytes(
            "texture:handed-over",
            &texture_spec(subject.crossing).words(),
        ));
        inputs.push(Input::bytes(
            "texture:attachment",
            &TargetSpec::read(FORMAT, [SIDE; 2]).words(),
        ));
        inputs.push(Input::bytes("sampler", &sampler().words()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        let mut rects: Vec<u8> = (0..BANDS).flat_map(|at| rect_bytes(band(at))).collect();
        rects.extend(rect_bytes(WHOLE));
        inputs.push(Input::bytes("rects", &rects));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("grid", &grid_words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let subject = &self.subject;
        let kernel = subject.functions().1;
        let mut library_of: Vec<(&'static str, Handle)> = Vec::new();
        for name in [TOB, SCOUT] {
            let loaded = Fixture::new(name).loaded().map_err(|why| Abort {
                at: format!("library:{name}"),
                failure: Failure::named("FixtureNotRead", why),
            })?;
            let library = made(
                sites,
                &format!("library:{name}"),
                side.library(&loaded.source()),
            )?;
            library_of.push((name, library));
        }
        let library = |named: &'static str| -> Handle {
            library_of
                .iter()
                .find(|(name, _)| *name == named)
                .map(|(_, held)| *held)
                .expect("the row loads both of its fixtures")
        };
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library(TOB), &subject.pipeline()),
        )?;
        let compute = made(
            sites,
            "compute-pipeline",
            side.compute_pipeline(library(SCOUT), kernel),
        )?;
        // The texture the two kinds of work hand over, and the colour attachment the pass needs
        // whether or not the row is about it.
        let handed = made(
            sites,
            "texture:handed-over",
            side.target(&texture_spec(subject.crossing)),
        )?;
        let attachment = made(
            sites,
            "texture:attachment",
            side.target(&TargetSpec::read(FORMAT, [SIDE; 2])),
        )?;
        let sampled = made(sites, "sampler", side.sampler(&sampler()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let out = made(sites, "buffer:kernel", side.zeroed_buffer(PLANE as usize))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;

        // The pass: the bands where the row fills its own target, the whole rectangle where it
        // samples or writes a texture.
        let mut steps = vec![RenderStep::Pipeline(state)];
        match subject.crossing {
            Crossing::KernelToFragment => {
                steps.push(RenderStep::FragmentTexture {
                    texture: handed,
                    index: 0,
                });
                steps.push(RenderStep::FragmentSampler {
                    sampler: sampled,
                    index: 0,
                });
                steps.push(RenderStep::VertexBytes {
                    bytes: rect_bytes(WHOLE),
                    index: 0,
                });
                steps.push(RECT_DRAW);
            }
            Crossing::AttachmentToKernel => {
                for at in 0..BANDS {
                    steps.push(RenderStep::VertexBytes {
                        bytes: rect_bytes(band(at)),
                        index: 0,
                    });
                    steps.push(RenderStep::FragmentBuffer {
                        buffer: colors,
                        offset: u64::from(at) * 16,
                        index: 0,
                    });
                    steps.push(RECT_DRAW);
                }
            }
            Crossing::FragmentWriteToKernel => {
                steps.push(RenderStep::FragmentTexture {
                    texture: handed,
                    index: 0,
                });
                steps.push(RenderStep::VertexBytes {
                    bytes: rect_bytes(WHOLE),
                    index: 0,
                });
                steps.push(RECT_DRAW);
            }
        }
        // Which texture the pass renders into, and which one it hands over. Where the row's
        // subject is the attachment itself, they are one texture.
        let into = match subject.crossing {
            Crossing::AttachmentToKernel => handed,
            _ => attachment,
        };
        let pass = RenderWork::Pass(
            RenderPass::clearing(vec![PassColor::cleared(into, [CLEAR; 4])]).drawing(steps),
        );
        let dispatch = RenderWork::Dispatch(Dispatch {
            pipeline: compute,
            buffers: match subject.crossing {
                // `scout_write_f` binds no buffer; the readers write theirs at buffer 0.
                Crossing::KernelToFragment => Vec::new(),
                _ => vec![(out, 0)],
            },
            textures: vec![handed],
            grid: [SIDE, SIDE, 1],
            threads: [8, 8, 1],
        });
        // The order is the row's whole subject: the dispatch first where the kernel is the
        // producer, the pass first where it is.
        let read = RenderWork::ReadTexture {
            texture: handed,
            into: read_back,
            offset: 0,
            row_pitch: TEXEL * SIDE,
            size: [SIDE; 2],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        };
        let (work, names): (Vec<RenderWork>, Vec<&str>) = match subject.crossing {
            Crossing::KernelToFragment => (
                vec![
                    dispatch,
                    pass,
                    read,
                    // The sampled target lands in the kernel's own buffer, which this row's kernel
                    // writes nothing into.
                    RenderWork::ReadTexture {
                        texture: attachment,
                        into: out,
                        offset: 0,
                        row_pitch: TEXEL * SIDE,
                        size: [SIDE; 2],
                        plane: Plane::Whole,
                        at: ReadAt::BASE,
                    },
                ],
                vec!["dispatch", "pass", "read-handed-over", "read-sampled"],
            ),
            _ => (
                vec![pass, dispatch, read],
                vec!["pass", "dispatch", "read-handed-over"],
            ),
        };
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
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
        let handed_bytes = call(
            sites,
            "read-handed-over",
            side.read_buffer(read_back, 0, PLANE as usize),
            |_| Value::Absent,
        )?;
        let other_bytes = call(
            sites,
            "read-other",
            side.read_buffer(out, 0, PLANE as usize),
            |_| Value::Absent,
        )?;
        sites.push(Node::branch("handed-over", Vec::new()).with(Node::leaf(
            "texels",
            Observed::bytes(handed_bytes, plane_layout(texture_region())),
        )));
        // What the consumer of the handed-over texture answered: the sampled target where a pass
        // consumes it, the kernel's own buffer where a kernel does.
        let (what, region) = match subject.crossing {
            Crossing::KernelToFragment => ("sampled", texture_region()),
            _ => ("read-by-the-kernel", buffer_region()),
        };
        sites.push(Node::branch(what, Vec::new()).with(Node::leaf(
            "values",
            Observed::bytes(other_bytes, plane_layout(region)),
        )));
        Ok(())
    }
}

/// The dispatch's grid and threadgroup, as an input fingerprint covers them.
fn grid_words() -> Vec<u8> {
    [SIDE, SIDE, 1, 8, 8, 1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `name`'s recorded Metal answer, or `None` where nothing has recorded it.
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

    /// The `float4` at texel `(x, y)` of a plane of [`SIDE`] by [`SIDE`].
    fn value(bytes: &[u8], x: u32, y: u32) -> [f32; 4] {
        let at = ((y * SIDE + x) * TEXEL) as usize;
        let mut channels = [0f32; 4];
        for (channel, held) in channels.iter_mut().enumerate() {
            let word = at + 4 * channel;
            *held = f32::from_le_bytes([
                bytes[word],
                bytes[word + 1],
                bytes[word + 2],
                bytes[word + 3],
            ]);
        }
        channels
    }

    /// The bookkeeping holds with no device: three distinct rows, three distinct crossings, a
    /// geometry whose bands tile the extent, and layouts that cover exactly one plane.
    #[test]
    fn the_rows_name_three_distinct_crossings() {
        let named: BTreeSet<&str> = SUBJECTS.iter().map(|subject| subject.name).collect();
        let crossings: BTreeSet<Crossing> = SUBJECTS.iter().map(|s| s.crossing).collect();
        assert_eq!(named.len(), SUBJECTS.len());
        assert_eq!(crossings.len(), SUBJECTS.len());
        assert_eq!(SIDE % BANDS, 0);
        assert_eq!(plane_layout(texture_region()).extent(), PLANE as usize);
        assert_eq!(plane_layout(buffer_region()).extent(), PLANE as usize);
        // Every band's colour differs from every other's in every channel, and none of them is
        // the clear, so a band read in the wrong place cannot pass.
        for (at, color) in COLORS.iter().enumerate() {
            for channel in color {
                assert_ne!(f64::from(*channel), CLEAR);
            }
            for other in &COLORS[at + 1..] {
                for (channel, (ours, theirs)) in color.iter().zip(other).enumerate() {
                    assert_ne!(ours, theirs, "band {at}, channel {channel}");
                }
            }
        }
        // Two bands state a channel outside [0, 1], which is what the float target is for.
        assert!(COLORS
            .iter()
            .flatten()
            .any(|channel| *channel < 0.0 || *channel > 1.0));
        // Each row names its own pipeline, so no recording can be read for another row's.
        let pipelines: Vec<Vec<u8>> = SUBJECTS
            .iter()
            .map(|subject| subject.pipeline().words())
            .collect();
        for (at, ours) in pipelines.iter().enumerate() {
            for theirs in &pipelines[at + 1..] {
                assert_ne!(ours, theirs);
            }
        }
    }

    /// No node of any of the family's recorded transcripts holds two children of one name.
    ///
    /// Written because it happened: the texture the two kinds of work hand over was made at the
    /// site `handed-over` and its texels were emitted under a branch of the same name, so a walk
    /// down `handed-over/texels` found the **call** site, which has no children, and the three
    /// cross-checks below read as "no such site" rather than as an answer. The make sites are
    /// `texture:`-prefixed now.
    ///
    /// It is over this family's own recordings rather than over the whole case set because a case
    /// module is the wrong place to hold a rule about every case; the whole-set form belongs in
    /// the suite's own tests.
    #[test]
    fn no_recorded_node_holds_two_children_of_one_name() {
        fn walk(node: &crate::transcript::Node, path: &str, found: &mut Vec<String>) {
            let mut seen = BTreeSet::new();
            for child in &node.children {
                if !seen.insert(child.name.clone()) {
                    found.push(format!("{path}/{}", child.name));
                }
                walk(child, &format!("{path}/{}", child.name), found);
            }
        }
        let mut checked = 0usize;
        for subject in SUBJECTS {
            let Some(recorded) = only_outcome(subject.name) else {
                println!("{}: no recording in the tree yet", subject.name);
                continue;
            };
            let mut found = Vec::new();
            walk(&recorded.root, "case", &mut found);
            assert!(found.is_empty(), "{}: {found:?}", subject.name);
            checked += 1;
        }
        println!("no_recorded_node_holds_two_children_of_one_name: {checked} rows checked");
    }

    /// The sampling row's sampled target is the texture the kernel wrote, texel for texel, and
    /// that texture is not one value repeated -- so the check is not one a blank texture passes.
    #[test]
    fn the_sample_reproduced_what_the_kernel_wrote() {
        let subject = SUBJECTS[0];
        assert_eq!(subject.crossing, Crossing::KernelToFragment);
        let Some(recorded) = only_outcome(subject.name) else {
            println!("{}: no recording in the tree yet", subject.name);
            return;
        };
        let written = bytes(&recorded, "handed-over/texels");
        assert_eq!(
            bytes(&recorded, "sampled/values"),
            written,
            "{}",
            subject.name
        );
        let distinct: BTreeSet<[u32; 4]> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .map(|(x, y)| value(&written, x, y).map(f32::to_bits))
            .collect();
        assert!(
            distinct.len() > 1,
            "{}: the kernel wrote one value everywhere, so the sample proves nothing",
            subject.name
        );
    }

    /// The attachment row's kernel read exactly what a blit of the same target reads, and the
    /// bands are where the geometry puts them.
    ///
    /// The second half is the arithmetic the case states: `tob_solid` returns the `float4` its
    /// buffer holds unchanged and a `RGBA32Float` target stores it as it is, so the colour of
    /// column `x` is `COLORS[x * BANDS / SIDE]` and nothing was measured to say so.
    #[test]
    fn the_kernel_read_the_render_target_the_pass_stored() {
        let subject = SUBJECTS[1];
        assert_eq!(subject.crossing, Crossing::AttachmentToKernel);
        let Some(recorded) = only_outcome(subject.name) else {
            println!("{}: no recording in the tree yet", subject.name);
            return;
        };
        let stored = bytes(&recorded, "handed-over/texels");
        assert_eq!(
            bytes(&recorded, "read-by-the-kernel/values"),
            stored,
            "{}",
            subject.name
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                assert_eq!(
                    value(&stored, x, y),
                    COLORS[(x * BANDS / SIDE) as usize],
                    "{}: ({x}, {y})",
                    subject.name
                );
            }
        }
    }

    /// The fragment-write row's kernel read exactly what a blit reads, and both hold what
    /// `tob_write` writes: `(0.125 x, 0.125 y, 0.75, 1)`.
    ///
    /// Every one of those is exact under any math mode -- a scale by a power of two and two
    /// constants -- so this is arithmetic rather than a number adopted from the run.
    #[test]
    fn the_kernel_read_what_the_fragment_stage_wrote() {
        let subject = SUBJECTS[2];
        assert_eq!(subject.crossing, Crossing::FragmentWriteToKernel);
        let Some(recorded) = only_outcome(subject.name) else {
            println!("{}: no recording in the tree yet", subject.name);
            return;
        };
        let written = bytes(&recorded, "handed-over/texels");
        assert_eq!(
            bytes(&recorded, "read-by-the-kernel/values"),
            written,
            "{}",
            subject.name
        );
        for y in 0..SIDE {
            for x in 0..SIDE {
                assert_eq!(
                    value(&written, x, y),
                    [0.125 * x as f32, 0.125 * y as f32, 0.75, 1.0],
                    "{}: ({x}, {y})",
                    subject.name
                );
            }
        }
    }

    /// No row of the family loosens a value: every allowance is a notice declaration on the
    /// notices site, and the one notice declared is the one the row's own functions imply.
    #[test]
    fn no_row_loosens_a_value() {
        use crate::allowance::Justification;
        for case in all() {
            // Every allowance is a notice declaration on the notices site, so no row loosens a
            // value, and the one notice declared is the one the row's own functions imply.
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
            assert_eq!(
                codes.contains(&Degradation::DeviceArithmetic.code().to_owned()),
                case.subject.divides_a_float(),
                "{}: the device-arithmetic notice is declared by the wrong row",
                case.name()
            );
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            let inputs = case.inputs().unwrap();
            for wanted in ["render-pipeline", "kernel", "texture:handed-over", "grid"] {
                assert!(
                    inputs.iter().any(|input| input.name == wanted),
                    "{}: no {wanted} in its inputs",
                    case.name()
                );
            }
        }
    }
}
