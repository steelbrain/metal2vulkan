//! Where a simdgroup sum before a discard of whole quads counts from.
//!
//! The owned fixture `fragment_discard_hoist` draws Metal's fullscreen triangle over an 8 by 4
//! `RGBA32Sint` target cleared to -7, keeps the quad of pixels `(0..2, 0..2)` and discards every
//! other quad whole. Each kept thread writes
//! `(before > after, after - kept, (before - after) & 3, before < after)` from its simdgroup's lane
//! counts before and after the discard: four numbers that hold however a device packs quads into
//! simdgroups, so `(0, 0, 0, 0)` means the sum counted the survivors alone — Metal's compiler moved
//! it past the discard — and `(1, 0, 0, 0)` means it also counted the dropped quads that share the
//! simdgroup.
//!
//! Thirteen fragment functions, one row each: the shapes the lane measured for where Metal's
//! compiler places the sum. Record mode runs every one of them on native Metal, so the recording
//! holds Metal's own target rather than the lane's two-valued table, and the lane's table becomes a
//! cross-check this module's tests apply to the recorded targets.
//!
//! A row's site is the whole target under a layout whose row is a row of texels and whose element
//! is one 32-bit signed word, so a diff names the pixel and the channel: "row 1, index 0" of the
//! kept quad is the answer, and a texel outside the kept quad that is not the clear value is a draw
//! that kept a quad it should have discarded. The clear value is part of the answer and compares
//! like any other word.
//!
//! The one sampled texture and the sampler state are bound because three of the shapes read them
//! (`hoisted_sample`, `hoisted_texture_read`); the texture holds one texel of zeros.
//!
//! Nothing here can fault a GPU: one render pass a row, one triangle, one 8 by 4 target, and the
//! fixture's one loop is bounded by a buffer word the case sets to 0.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side, TextureRole,
    TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture every function of the case lives in.
const FIXTURE: &str = "fragment_discard_hoist";

/// The vertex function: Metal's fullscreen triangle.
const VERTEX: &str = "discard_hoist_full";

/// The target's extent in texels.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;

/// The target's format: four 32-bit signed words a texel, so what the fragment function returns is
/// read back as the numbers it returned.
const FORMAT: PixelFormat = PixelFormat::RGBA32Sint;

/// Bytes one row of the target takes.
const ROW_PITCH: u32 = WIDTH * 16;

/// Bytes one row's whole target takes, which is its slice of the read-back buffer.
const TARGET_BYTES: u32 = HEIGHT * ROW_PITCH;

/// Every channel of the clear value. A texel outside the kept quad holds this, so "the draw
/// discarded that quad whole" is a compared fact.
const CLEAR: i32 = -7;

/// The quads the draw keeps: bit `(y / 2) * 4 + x / 2`, which is the quad of pixels
/// `(0..2, 0..2)` alone.
const KEPT_QUADS: u32 = 1;

/// The thirteen fragment functions, in the order the lane lists them.
const SHAPES: [&str; 13] = [
    "hoisted_known",
    "counted_dependent",
    "hoisted_const",
    "hoisted_device_read",
    "hoisted_written_read",
    "hoisted_sample",
    "hoisted_texture_read",
    "hoisted_loop",
    "hoisted_call",
    "counted_call_use",
    "hoisted_nested",
    "counted_store",
    "counted_atomic",
];

/// The fragment buffers every shape binds, at indices 0 up: the kept-quad mask, the zero the
/// fixture's conditions read, and three scratch buffers the storing and atomic shapes write.
fn fragment_buffers() -> Vec<Vec<u8>> {
    vec![
        KEPT_QUADS.to_le_bytes().to_vec(),
        0i32.to_le_bytes().to_vec(),
        vec![0; 256],
        vec![0; 1024],
        vec![0; 64],
    ]
}

/// The sampled texture's spec: one `RGBA32Float` texel the host writes with zeros.
fn sampled_spec() -> TextureSpec {
    TextureSpec::new(PixelFormat::RGBA32Float, [1, 1], 16, TextureRole::Read)
}

/// The sampled texture's bytes.
fn sampled_texels() -> Vec<u8> {
    vec![0; 16]
}

/// The one colour attachment every shape's pipeline writes.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The target every row draws into: private, since the blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [WIDTH, HEIGHT])
}

/// The pipeline every shape's row creates: the one vertex function and the shape's own fragment
/// function, both out of the library the `render_pipeline` call names.
fn pipeline_spec(shape: &str) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::of(
        StageFunction::named(VERTEX),
        Some(StageFunction::named(shape)),
        &COLORS,
    )
}

/// The layout a row's target is read under: one row of the layout per row of texels, one element
/// per channel, so a diff names the pixel and the channel.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::I32,
        row_offset: 0,
        row_elements: WIDTH * 4,
        row_pitch: ROW_PITCH,
        rows: HEIGHT,
        image_pitch: HEIGHT * ROW_PITCH,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The case.
pub struct DiscardHoist;

/// The case the suite holds.
pub fn all() -> Vec<DiscardHoist> {
    vec![DiscardHoist]
}

impl DiscardHoist {
    /// The fixture under the entry point `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }
}

impl Case for DiscardHoist {
    fn name(&self) -> &'static str {
        "discard-quad-sum-placement"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The two shapes whose placement by Metal's compiler is not known are the ones where the
        // module keeps the sum ahead of the discard and says so. The allowance loosens the targets
        // of those two rows and nothing else, and it fails if the notice did not fire, so a row
        // that stops needing it drops its line.
        SHAPES
            .iter()
            .filter(|shape| NOTED.contains(shape))
            .map(|shape| {
                Allowance::new(
                    format!("case/{shape}/*"),
                    crate::allowance::Loosening::Unchecked,
                    crate::allowance::Justification::notice(Degradation::QuadsCountedBeforeDiscard),
                )
            })
            // Two more kinds our side announces on lavapipe, declared so the notices site stays
            // readable there: it splits a fragment quad at a clipped edge where Metal keeps it
            // whole (`quads-split-by-clipping`) and groups a fragment simdgroup its own way
            // (`fragment-simdgroup-grouping`). Declarations only -- `discard-quad-sum-placement`
            // stays divergent on the sums it measures, which is the finding.
            .chain(
                [
                    Degradation::QuadsCountedBeforeDiscard,
                    Degradation::QuadsSplitByClipping,
                    Degradation::FragmentSimdgroupGrouping,
                ]
                .into_iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
            )
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in std::iter::once(VERTEX).chain(SHAPES) {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(VERTEX).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(VERTEX);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("target", &target_spec().words()),
            Input::bytes(
                "pipelines",
                &SHAPES
                    .iter()
                    .flat_map(|shape| pipeline_spec(shape).words())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes("texture:sampled", &sampled_texels()),
            Input::bytes("sampler", &SamplerSpec::new().words()),
        ];
        for (index, bytes) in fragment_buffers().iter().enumerate() {
            inputs.push(Input::bytes(format!("buffer:{index}"), bytes));
        }
        // The clear value is part of the answer: a texel outside the kept quad holds it, and the
        // comparison would read a clear that moved as the draw keeping a quad it should have
        // discarded.
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("shapes", SHAPES.join(",").as_bytes()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture(VERTEX).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        // One target for every row: each pass clears it, and the pieces of one command buffer run
        // in the order the case lists them, so a row sees only its own draw.
        let target = made(sites, "target", side.target(&target_spec()))?;
        let sampled = made(
            sites,
            "texture/sampled",
            side.texture_2d_in_role(&sampled_spec(), &sampled_texels()),
        )?;
        let sampler = made(sites, "sampler", side.sampler(&SamplerSpec::new()))?;
        let buffers: Vec<Handle> = fragment_buffers()
            .iter()
            .enumerate()
            .map(|(index, bytes)| made(sites, &format!("buffer/{index}"), side.buffer(bytes)))
            .collect::<Result<Vec<Handle>, Abort>>()?;
        let read_back = made(
            sites,
            "buffer/read-back",
            side.zeroed_buffer(SHAPES.len() * TARGET_BYTES as usize),
        )?;

        // All thirteen rows in one command buffer: a pass and a blit each, the blits writing one
        // slice of the read-back buffer apiece, so the case costs one commit and one wait.
        let mut pipelines = Vec::with_capacity(SHAPES.len());
        for shape in SHAPES {
            pipelines.push(made(
                sites,
                &format!("{shape}/pipeline"),
                side.render_pipeline(library, &pipeline_spec(shape)),
            )?);
        }
        let mut work = Vec::with_capacity(SHAPES.len() * 2);
        for (at, pipeline) in pipelines.iter().enumerate() {
            let mut steps = vec![RenderStep::Pipeline(*pipeline)];
            for (index, buffer) in buffers.iter().enumerate() {
                steps.push(RenderStep::FragmentBuffer {
                    buffer: *buffer,
                    offset: 0,
                    index: index as u32,
                });
            }
            steps.push(RenderStep::FragmentTexture {
                texture: sampled,
                index: 0,
            });
            steps.push(RenderStep::FragmentSampler { sampler, index: 0 });
            steps.push(RenderStep::triangle());
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(TARGET_BYTES),
                row_pitch: ROW_PITCH,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, SHAPES.len() * TARGET_BYTES as usize),
            |_| Value::Absent,
        )?;

        for (at, shape) in SHAPES.iter().enumerate() {
            let first = at * TARGET_BYTES as usize;
            let target_bytes = bytes[first..first + TARGET_BYTES as usize].to_vec();
            let mut row = Node::branch(*shape, Vec::new()).with(Node::leaf(
                "texels",
                Observed::bytes(target_bytes, layout()),
            ));
            // What each of the row's two pieces of work answered, which is `None` on Metal's side:
            // a piece of ours that refuses compares against Metal answering and reads as a refusal
            // rather than a divergence.
            for (piece, what) in [(2 * at, "pass"), (2 * at + 1, "read")] {
                if let Some(Some(failure)) = outcome.refused.get(piece) {
                    row = row.with(Node::leaf(what, Observed::value(failure.recorded())));
                }
            }
            sites.push(row);
        }
        Ok(())
    }
}

/// The shapes whose placement by Metal's compiler the module does not know: it keeps the sum ahead
/// of the discard and announces `quads-counted-before-discard`.
const NOTED: [&str; 2] = ["counted_call_use", "hoisted_nested"];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// What a kept thread's simdgroup counted ahead of the discard, as the lane names it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Answer {
        /// `(0, 0, 0, 0)`: the lanes that survive the discard alone, so the sum moved past it.
        Survivors,
        /// `(1, 0, 0, 0)`: the dropped quads that share the simdgroup too.
        DroppedQuads,
    }

    /// Metal's answer for each shape, copied from `runtime_lane::discard_hoist::SHAPES` to be
    /// checked against the recording, never to stand in for it.
    const LANE_ANSWERS: [(&str, Answer); 13] = [
        ("hoisted_known", Answer::Survivors),
        ("counted_dependent", Answer::DroppedQuads),
        ("hoisted_const", Answer::Survivors),
        ("hoisted_device_read", Answer::Survivors),
        ("hoisted_written_read", Answer::Survivors),
        ("hoisted_sample", Answer::Survivors),
        ("hoisted_texture_read", Answer::Survivors),
        ("hoisted_loop", Answer::Survivors),
        ("hoisted_call", Answer::Survivors),
        ("counted_call_use", Answer::DroppedQuads),
        ("hoisted_nested", Answer::Survivors),
        ("counted_store", Answer::DroppedQuads),
        ("counted_atomic", Answer::DroppedQuads),
    ];

    /// Whether the quad of pixel `(x, y)` is kept.
    fn kept(x: u32, y: u32) -> bool {
        (KEPT_QUADS >> ((y / 2) * 4 + x / 2)) & 1 != 0
    }

    /// The answer a target's bytes hold: the kept quad's four texels agree on one, and every other
    /// texel holds the clear value. `Err` names a texel outside both.
    fn answer(bytes: &[u8]) -> Result<Answer, String> {
        let mut found: Option<Answer> = None;
        for (at, texel) in bytes.chunks(16).enumerate() {
            let (x, y) = (at as u32 % WIDTH, at as u32 / WIDTH);
            let word = |channel: usize| {
                i32::from_le_bytes([
                    texel[4 * channel],
                    texel[4 * channel + 1],
                    texel[4 * channel + 2],
                    texel[4 * channel + 3],
                ])
            };
            let words = [word(0), word(1), word(2), word(3)];
            let here = match words {
                [0, 0, 0, 0] if kept(x, y) => Answer::Survivors,
                [1, 0, 0, 0] if kept(x, y) => Answer::DroppedQuads,
                [CLEAR, CLEAR, CLEAR, CLEAR] if !kept(x, y) => continue,
                _ => return Err(format!("({x}, {y}) holds {words:?}")),
            };
            if found.is_some_and(|held| held != here) {
                return Err(format!(
                    "({x}, {y}) answers {here:?}, another kept texel {found:?}"
                ));
            }
            found = Some(here);
        }
        found.ok_or_else(|| "no kept texel was written".to_owned())
    }

    /// The case's bookkeeping holds: each shape names a distinct site, the layout covers the
    /// target exactly, and the kept quad is four texels of the 32.
    #[test]
    fn the_shapes_and_the_layout_agree_with_the_target() {
        let named: BTreeSet<&str> = SHAPES.iter().copied().collect();
        assert_eq!(named.len(), SHAPES.len());
        assert_eq!(layout().extent(), (HEIGHT * ROW_PITCH) as usize);
        let kept_texels = (0..WIDTH * HEIGHT)
            .filter(|at| kept(at % WIDTH, at / WIDTH))
            .count();
        assert_eq!(kept_texels, 4);
        // Every shape the allowance names is a shape the case runs: an allowance over a site no
        // row emits would be an excuse nothing holds to account.
        for shape in NOTED {
            assert!(SHAPES.contains(&shape), "{shape}");
        }
    }

    /// Every target Metal recorded answers as the lane's table says Metal answers, shape for
    /// shape, and every texel outside the kept quad holds the clear value.
    ///
    /// This is the cross-check the port owes the lane: the lane carried a two-valued table, the
    /// case carries Metal's targets, and this says whether the two agree. It reads the recording,
    /// so it runs where there is no GPU.
    #[test]
    fn metal_s_recorded_targets_answer_as_the_lane_s_table() {
        let case = DiscardHoist;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for recorded in &recording.answers {
            for outcome in &recorded.outcomes {
                for (shape, expected) in LANE_ANSWERS {
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(shape)
                        .and_then(|node| node.child("texels"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    checked += 1;
                    match answer(bytes) {
                        Ok(got) if got == expected => {}
                        Ok(got) => differ.push(format!(
                            "{shape}: Metal {got:?}, the lane's table {expected:?}"
                        )),
                        Err(why) => differ.push(format!("{shape}: {why}")),
                    }
                }
            }
        }
        assert!(checked > 0, "the recording holds no shape");
        for line in differ.iter().take(20) {
            println!("DISCARD-HOIST-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {checked} recorded targets differ from the lane's table",
            differ.len()
        );
    }
}
