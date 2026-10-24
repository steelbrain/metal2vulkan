//! A chain long enough to be a frame: four pieces of work over two shared textures, **one read at
//! the end**.
//!
//! The other program families in the suite are two or three pieces long and read what each piece
//! produced, which is what makes a difference in them easy to place
//! ([`crate::cases::render_chains`], [`crate::cases::compute_render_chains`]). This one is the
//! other half of the question: a single recording covering a **sequence** rather than a step, so
//! that what it compares is the composition and not any one hand-over.
//!
//! One command buffer, four pieces, in this order:
//!
//! 1. a render pass draws four bands of known colour into texture `A`;
//! 2. a dispatch reads and writes **every texel of `A` in place**, swapping its channels
//!    (`scout_read_write_f` writes `read(g).yxwz`);
//! 3. a render pass samples `A` into texture `B`;
//! 4. a dispatch reads every texel of `B` into a buffer.
//!
//! Then the host reads that buffer, and nothing else. Three hand-overs sit inside the chain and
//! none of them is visible from outside it: a colour attachment's store to a kernel's read-write
//! access, a kernel's writes to a fragment stage's sampled read, and a colour attachment's store
//! to a kernel's read. Our side turns each into a barrier and an image-layout transition; Metal
//! states none of them.
//!
//! # The whole answer is arithmetic
//!
//! Each link is chosen so that the final buffer is a value this module states rather than a
//! measurement it adopts. `tob_solid` returns the `float4` its buffer holds unchanged; a
//! `RGBA32Float` texture stores a `float` as it is; `scout_read_write_f` swaps channels and
//! computes nothing; `tob_sample` samples at the pixel's position over the sampled texture's
//! extent, which for two textures of one extent is a texel centre; and `scout_read_f` reads a
//! texel into a word. So word `y * 16 + x` of the buffer must hold the swapped colour of the band
//! column `x` belongs to -- and if it does not, the chain broke somewhere, which is the finding.
//!
//! A chain with one read cannot say **which** link broke, and that is the trade this case makes
//! deliberately: the step-level families say which, and this one says whether the composition
//! holds. What it does carry per piece is each piece's own answer, so a refusal inside the chain is
//! named at its piece rather than read as a difference in the bytes.
//!
//! Nothing here can fault a GPU: 16 by 16 textures, two dispatches of 256 threads, five draws, no
//! loop in any function.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, Dispatch, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The format both textures carry: a `float` stored as it is, so the chain's answer is arithmetic.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// Both textures' side, in texels. Sixteen because `scout_read_f` writes texel `(x, y)` into word
/// `y * 16 + x` of its buffer: the stride is the fixture's own.
const SIDE: u32 = 16;

/// Bytes one plane of [`SIDE`] by [`SIDE`] `float4`s takes.
const PLANE: u32 = TEXEL * SIDE * SIDE;

/// The bands the first pass draws.
const BANDS: u32 = 4;

/// The clear the first pass loads `A` with: outside every band's every channel, so a band that was
/// never drawn is seen at the end of the chain rather than inferred.
const CLEAR: f32 = -7.5;

/// The four colours the first pass draws, each channel an exact binary fraction, and no channel of
/// one equal to the same channel of another -- so a swap that did not happen, or happened twice,
/// changes the answer.
const COLORS: [[f32; 4]; BANDS as usize] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.125, 0.375, 0.625, 0.875],
    [2.5, -1.25, 0.0, 1.5],
    [-0.5, 3.75, 0.25, 0.625],
];

/// The owned fixture of the rectangle, the solid colour and the sample.
const TOB: &str = "tob_renders";

/// The owned fixture of the kernels that read, write and read-and-write a texture texel by texel.
const SCOUT: &str = "format_scout";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners.
const RECT: &str = "tob_rect";

/// `tob_solid`: the `float4` a buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// `tob_sample`: texture 0 sampled at the pixel's position over the texture's extent.
const SAMPLE: &str = "tob_sample";

/// `scout_read_write_f`: texel `g` becomes its own channels as `yxwz`.
const SWAP: &str = "scout_read_write_f";

/// `scout_read_f`: texel `(x, y)` into word `y * 16 + x` of buffer 0.
const READ: &str = "scout_read_f";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The one colour attachment both pipelines carry.
const COLOR_STATE: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// Band `at` of [`BANDS`], as `tob_rect` takes a rectangle.
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

/// The buffer the solid draws read their colours out of.
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

/// The pipeline of `fragment` over the one colour attachment.
fn pipeline(fragment: &str) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(RECT, fragment, &COLOR_STATE)
}

/// `A`: drawn into, read and written by a kernel in place, then sampled.
fn first_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE; 2]).written()
}

/// `B`: drawn into and read by a kernel.
fn second_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE; 2])
}

/// The layout the final buffer is read under: one row of the layout a row of texels, one element a
/// channel, so a diff names the pixel and the channel.
fn layout() -> ByteLayout {
    let row_pitch = TEXEL * SIDE;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(PLANE),
        },
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

/// The dispatch's grid and threadgroup, as an input fingerprint covers them.
fn grid_words() -> Vec<u8> {
    [SIDE, SIDE, 1, 8, 8, 1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// The case.
pub struct FrameChain;

/// The case the suite holds.
pub fn boxed() -> Vec<Box<dyn Case>> {
    vec![Box::new(FrameChain)]
}

impl Case for FrameChain {
    fn name(&self) -> &'static str {
        "frame-length-chain"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened: every link stores or moves a `float` and computes nothing, so the
        // chain's answer is exact or a link broke. The normalized-conversion notice cannot apply,
        // because nothing here stores a normalized channel.
        // The one entry is a **notice declaration** on the notices site, which loosens no value.
        // Metal announces nothing, so a notice of ours has to be declared or the comparison cannot
        // be made at all, and `device-arithmetic` is the one this case's own functions imply:
        // `tob_sample` divides the pixel's position by the sampled texture's extent, and our side
        // records the note where a float division takes Vulkan's precision bounds rather than
        // correct rounding (`metal2vulkan::pipeline::artifact`'s
        // `PrecisionNote::DeviceArithmetic`). Which device does that is the device's answer; that
        // this case can announce it is the case's own shape.
        //
        // Measured: it fires on lavapipe and not on MoltenVK on this M3 Max, and on lavapipe every
        // word of this case's chain compared **exactly** -- the whole difference was the announcement.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for this case and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::DeviceArithmetic.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (name, entry) in [
            (TOB, format!("{RECT}+{SOLID}+{SAMPLE}")),
            (SCOUT, format!("{SWAP}+{READ}")),
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
        inputs.push(Input::bytes("fill-pipeline", &pipeline(SOLID).words()));
        inputs.push(Input::bytes("sample-pipeline", &pipeline(SAMPLE).words()));
        inputs.push(Input::bytes("kernels", format!("{SWAP},{READ}").as_bytes()));
        inputs.push(Input::bytes("first-texture", &first_spec().words()));
        inputs.push(Input::bytes("second-texture", &second_spec().words()));
        inputs.push(Input::bytes("sampler", &SamplerSpec::new().words()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        let mut rects: Vec<u8> = (0..BANDS).flat_map(|at| rect_bytes(band(at))).collect();
        rects.extend(rect_bytes(WHOLE));
        inputs.push(Input::bytes("rects", &rects));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("grid", &grid_words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let mut held: Vec<(&'static str, Handle)> = Vec::new();
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
            held.push((name, library));
        }
        let library = |named: &'static str| -> Handle {
            held.iter()
                .find(|(name, _)| *name == named)
                .map(|(_, handle)| *handle)
                .expect("the case loads both of its fixtures")
        };
        let fill = made(
            sites,
            "fill-pipeline",
            side.render_pipeline(library(TOB), &pipeline(SOLID)),
        )?;
        let sample = made(
            sites,
            "sample-pipeline",
            side.render_pipeline(library(TOB), &pipeline(SAMPLE)),
        )?;
        let swap = made(
            sites,
            "kernel:swap",
            side.compute_pipeline(library(SCOUT), SWAP),
        )?;
        let read = made(
            sites,
            "kernel:read",
            side.compute_pipeline(library(SCOUT), READ),
        )?;
        let first = made(sites, "first-texture", side.target(&first_spec()))?;
        let second = made(sites, "second-texture", side.target(&second_spec()))?;
        let sampler = made(sites, "sampler", side.sampler(&SamplerSpec::new()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let out = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;

        let mut bands = vec![RenderStep::Pipeline(fill)];
        for at in 0..BANDS {
            bands.push(RenderStep::VertexBytes {
                bytes: rect_bytes(band(at)),
                index: 0,
            });
            bands.push(RenderStep::FragmentBuffer {
                buffer: colors,
                offset: u64::from(at) * 16,
                index: 0,
            });
            bands.push(RECT_DRAW);
        }
        let sampling = vec![
            RenderStep::Pipeline(sample),
            RenderStep::FragmentTexture {
                texture: first,
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
                RenderPass::clearing(vec![PassColor::cleared(first, [f64::from(CLEAR); 4])])
                    .drawing(bands),
            ),
            RenderWork::Dispatch(Dispatch {
                pipeline: swap,
                buffers: Vec::new(),
                textures: vec![first],
                grid: [SIDE, SIDE, 1],
                threads: [8, 8, 1],
            }),
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(second, [f64::from(CLEAR); 4])])
                    .drawing(sampling),
            ),
            RenderWork::Dispatch(Dispatch {
                pipeline: read,
                buffers: vec![(out, 0)],
                textures: vec![second],
                grid: [SIDE, SIDE, 1],
                threads: [8, 8, 1],
            }),
        ];
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        // Each piece's own answer, so a refusal inside the chain is named at its piece rather than
        // read as a difference in the bytes the chain ends with.
        let names = ["fill-pass", "swap-dispatch", "sample-pass", "read-dispatch"];
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
            side.read_buffer(out, 0, PLANE as usize),
            |_| Value::Absent,
        )?;
        sites.push(
            Node::branch("chain", Vec::new())
                .with(Node::leaf("values", Observed::bytes(bytes, layout()))),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// What the chain says word `y * 16 + x` of the final buffer holds: the colour of the band column
    /// `x` belongs to, its channels swapped as `scout_read_write_f` swaps them.
    ///
    /// Stated as the composition of the four links, from what each function does and nothing Metal was
    /// asked.
    fn expected(x: u32) -> [f32; 4] {
        let color = COLORS[(x * BANDS / SIDE) as usize];
        [color[1], color[0], color[3], color[2]]
    }

    /// The chain's expectation is a composition of four links and none of its steps is a no-op.
    ///
    /// Written over the rule rather than over a recording, so it holds with no device: a swap that
    /// did nothing, or a colour whose channels happen to be symmetric, would make the chain's
    /// answer indistinguishable from a chain that skipped the dispatch -- and this is what stops
    /// that.
    #[test]
    fn the_chain_states_a_composition_whose_every_link_moves_something() {
        assert_eq!(SIDE % BANDS, 0);
        assert_eq!(layout().extent(), PLANE as usize);
        for (at, color) in COLORS.iter().enumerate() {
            // The swap really changes this colour, so the dispatch in the middle of the chain
            // cannot be skipped without the answer moving.
            assert_ne!(
                expected(at as u32 * SIDE / BANDS),
                *color,
                "band {at}: its channels survive the swap unchanged"
            );
            for channel in color {
                assert_ne!(*channel, CLEAR);
            }
            for other in &COLORS[at + 1..] {
                for (channel, (ours, theirs)) in color.iter().zip(other).enumerate() {
                    assert_ne!(ours, theirs, "band {at}, channel {channel}");
                }
            }
        }
        // The four bands answer four different things, so a sample or a read that reached the
        // wrong column is caught.
        let answers: BTreeSet<[u32; 4]> = (0..BANDS)
            .map(|at| expected(at * SIDE / BANDS).map(f32::to_bits))
            .collect();
        assert_eq!(answers.len(), BANDS as usize);
        // The two pipelines are two different pipelines, and the two textures two different
        // descriptors: `A` is written by a kernel and `B` is not.
        assert_ne!(pipeline(SOLID).words(), pipeline(SAMPLE).words());
        assert_ne!(first_spec().words(), second_spec().words());
    }

    /// Metal's recorded chain holds what the composition of its four links says, in every word.
    ///
    /// The one thing this case can tell is whether the whole chain holds, which is what it is for.
    /// Where it differs, the step-level families say which hand-over broke.
    #[test]
    fn metals_recorded_chain_is_the_composition_of_its_links() {
        let Ok(recording) = crate::recording::read(FrameChain.name()) else {
            println!("frame-length-chain: no recording in the tree yet");
            return;
        };
        let answer = recording.answers.first().expect("no answer");
        assert_eq!(answer.outcomes.len(), 1, "more than one outcome");
        let node = answer.outcomes[0]
            .transcript
            .root
            .child("chain")
            .and_then(|node| node.child("values"))
            .expect("no chain/values site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("chain/values holds no bytes");
        };
        assert_eq!(bytes.len(), PLANE as usize);
        let mut differ = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let at = ((y * SIDE + x) * TEXEL) as usize;
                let mut got = [0f32; 4];
                for (channel, held) in got.iter_mut().enumerate() {
                    let word = at + 4 * channel;
                    *held = f32::from_le_bytes([
                        bytes[word],
                        bytes[word + 1],
                        bytes[word + 2],
                        bytes[word + 3],
                    ]);
                }
                let want = expected(x);
                if got != want {
                    differ.push(format!(
                        "({x}, {y}): Metal {got:?}, the chain says {want:?}"
                    ));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("FRAME-CHAIN-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} words differ from the composition of the chain's links",
            differ.len(),
            SIDE * SIDE
        );
    }
}
