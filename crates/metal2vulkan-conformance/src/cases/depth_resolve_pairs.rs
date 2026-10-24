//! `MTLMultisampleDepthResolveFilter` crossed with the attachment's **store action** and with the
//! samples the pass left: what a multisampled depth attachment's resolve texture holds.
//!
//! Nothing in the suite compared a depth or a stencil resolve at all. The option-space axis reads
//! `MTLMultisampleDepthResolveFilter` at **0 of its 3 values** and
//! `MTLMultisampleStencilResolveFilter` at 0 of 2, with one value each named only by the suite's
//! shared sources ([`crate::pass_descriptors`]'s `pass-attachment-filters`, which **sets the field
//! and reads it back** — a descriptor round trip, not a resolve). The runtime, meanwhile, carries a
//! whole subsystem for it (`resolve.rs`, `depth_stencil_resolve.rs`): three filters, a store action
//! that keeps the samples beside one that does not, a scratch image and a copy where a device
//! resolves one aspect at a time, and a draw after the pass where a device has no `MIN` or `MAX`
//! resolve mode. Every rule in it was measured by a probe and none of it was ever put beside Metal
//! by a case.
//!
//! A resolve is the shape where a per-value sweep says least: **the answer is a function of the
//! filter, the samples and the store action together**, and a case that fixes two of the three and
//! sweeps the last proves nothing about the other cells. So this family is a grid.
//!
//! | case | the grid |
//! |---|---|
//! | `resolve-depth-filters-4` | three filters by two resolving store actions by seven per-sample depth patterns, at 4 samples |
//! | `resolve-depth-filters-2` | the same at 2 samples, which Vulkan leaves optional and lavapipe does not render |
//!
//! # How a pass puts a different depth in each sample
//!
//! Not by partial coverage: which samples an edge covers is the rasterizer's own grid, so a case
//! built on it would compare a device's sample pattern and not its resolve. The pass states the
//! samples outright instead. The owned fixture `rt6_resolves` has a vertex function whose
//! fullscreen triangle sits at the depth buffer 0 holds (`rt6_resolve_full`) and a fragment
//! function whose `[[sample_mask]]` output is the mask buffer 0 holds (`rt6_resolve_masked`), and
//! the two buffer 0s are different stages' and so different bindings. One draw a sample, each at
//! its own depth and masked to its own sample, puts a stated depth in a stated sample and leaves
//! the rest of the pixel at the pass's clear — a pattern no sample grid can change.
//!
//! The attachment's depth comparison is `Always` and its depth write is on, so what a draw writes
//! is what it carried, and a sample no draw's mask named keeps `CLEAR_DEPTH`.
//!
//! # The seven patterns, and what each tells apart
//!
//! Each pattern is one cell of the target, and every depth is an exact binary fraction, so no cell's
//! answer turns on rounding. `-` is a sample no draw covered, which holds the clear of 1.
//!
//! | pattern | samples | `Sample0` | `Min` | `Max` |
//! |---|---|---|---|---|
//! | ascending | 1/8, 2/8, 3/8, 4/8 | 1/8 | 1/8 | 4/8 |
//! | descending | 4/8, 3/8, 2/8, 1/8 | 4/8 | 1/8 | 4/8 |
//! | sample 0 between | 3/8, 1/8, 5/8, 2/8 | 3/8 | 1/8 | 5/8 |
//! | a sample uncovered | 3/8, 1/8, -, 2/8 | 3/8 | 1/8 | 1 |
//! | every sample alike | 2/8 ×4 | 2/8 | 2/8 | 2/8 |
//! | a tie at the smallest | 6/8, 1/8, 1/8, 4/8 | 6/8 | 1/8 | 6/8 |
//! | nothing covered | - ×4 | 1 | 1 | 1 |
//!
//! Read down the three columns: the three filters answer three **distinct seven-tuples**, and no
//! two of them agree in more than four of the seven cells, which is what
//! `tests::the_three_filters_answer_three_distinct_tuples` holds with no device. Read across,
//! two rows are **indistinguishable by construction** and are named rather than left to be
//! noticed: *every sample alike*, where the three filters must agree because there is one value to
//! pick, and *nothing covered*, where they must agree on the clear. They are in the grid because
//! they are what the other five are read against — a device that resolved the clear where a sample
//! was written, or a sample where nothing was, is a device those two rows catch and the others
//! cannot.
//!
//! *A sample uncovered* is the one that crosses the resolve with the **load action**: its `Max` is
//! the pass's `clearDepth` and nothing else in the cell is, so a resolve that reads only the
//! samples a draw touched answers 3/8 there instead of 1.
//!
//! # The store action's half of the cross
//!
//! Metal defines two store actions that resolve, and they differ in what becomes of the
//! multisampled attachment: `MultisampleResolve` leaves its texels undefined and
//! `StoreAndMultisampleResolve` keeps the samples for a later pass. **Neither is read here**: what
//! a `DontCare` store leaves is undefined on both sides, so a case that compared it would be
//! comparing two devices' choices rather than Metal's rule.
//!
//! What the grid asks of the store action is the other question, and it has an answer: **is the
//! resolve itself the same under both?** Twenty-one cells say so under each, so a side whose
//! resolve depends on the store action — one that resolves from the stored copy under one and from
//! the tile under the other — fails. That is a claim about independence, and a grid is the only
//! shape that can make it: the per-value axis would read `MultisampleResolve` and
//! `StoreAndMultisampleResolve` as two values compared and say nothing about the pair.
//!
//! # What is not asked, and why
//!
//! - **A non-resolving store action beside a resolve texture.** `Store` or `DontCare` with
//!   `resolveTexture` set is measured to resolve nothing and leave the texture untouched
//!   (`metal2vulkan_runtime`'s `resolve.rs`, a probe on an M4), and it is also a shape **Metal's
//!   API validation stops**. The suite's record mode runs a validated leg beside the plain one, so
//!   a case here would be one keystroke from an assertion that ends the process, and this family
//!   does not find out by trying. It stays a runtime-lane fact.
//! - **`MTLMultisampleStencilResolveFilterDepthResolvedSample`** beside these filters. Metal takes
//!   it only where the depth and the stencil are **one** attachment, which this family's
//!   depth-only pass is not. It is [`crate::cases::stencil_resolved_sample`]'s, which crosses it
//!   with every one of these three filters on a combined `Depth32Float_Stencil8` attachment.
//! - **A layered pass, a resolve slice or level past the first, and a memoryless resolve
//!   texture.** Each is a separate rule of the runtime's resolve path with its own measured
//!   answer, and each is a cross of its own rather than this one's.
//!
//! # Why this family is safe
//!
//! One command buffer a case: six render passes over an 8 by 8 target and one blit each. Twenty-six
//! draws of three vertices a pass, no loop in either stage, every scissor rectangle inside the
//! attachments, every vertex inside the clip volume, every depth inside `[0, 1]`, and no sample
//! count past what the venue answers. A venue that does not render the case's sample count refuses
//! the target and the suite counts the case
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving it uncompared.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, Plane, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, ScissorBox, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{Degradation, MultisampleDepthResolveFilter, StoreAction};

/// The owned fixture: the fullscreen triangle at buffer 0's depth, and the fragment whose sample
/// mask is buffer 0's.
const FIXTURE: &str = "rt6_resolves";

/// `rt6_resolve_full`: `float4(4 * a - 1, z, 1)` at each of three vertex ids, where `z` is the
/// `float` vertex buffer 0 holds.
const VERTEX: &str = "rt6_resolve_full";

/// `rt6_resolve_masked`: white, and the `uint` fragment buffer 0 holds as `[[sample_mask]]`.
const FRAGMENT: &str = "rt6_resolve_masked";

/// The side of every attachment, in texels.
const SIDE: u32 = 8;

/// The side of one cell: one pattern's own pixels.
const CELL: u32 = 2;

/// Cells along one axis.
const GRID: u32 = SIDE / CELL;

/// The depth attachment's and the resolve texture's format: one `float` a texel, which a blit reads
/// back as itself and which holds every depth the patterns state exactly.
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// Bytes one texel of [`DEPTH_FORMAT`] takes.
const DEPTH_TEXEL: u32 = 4;

/// Bytes a resolve texture's whole level 0 takes.
const DEPTH_PLANE: u32 = DEPTH_TEXEL * SIDE * SIDE;

/// The colour attachment's format. The fragment function returns a `float4` at colour 0, so the
/// pass must attach one; nothing reads it and it is stored `DontCare`.
const COLOR_FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// `clearDepth`: the far plane, which no pattern's drawn depth equals, so a sample no draw covered
/// is read rather than inferred.
const CLEAR_DEPTH: f64 = 1.0;

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How many runs record mode takes a case of this family over: one command buffer, one queue and
/// nothing a device may reorder, so the runs are expected to fold into one outcome and a case whose
/// answer moved between runs is the finding rather than one run being read as the answer.
const RUNS: u32 = 8;

/// The two store actions that resolve, in the SDK's declaration order.
const RESOLVING: [StoreAction; 2] = [
    StoreAction::MultisampleResolve,
    StoreAction::StoreAndMultisampleResolve,
];

/// Every `MTLMultisampleDepthResolveFilter`, named value by value so the suite's option-space axis
/// reads them, and held against the runtime's own list by
/// [`tests::the_swept_filters_are_every_value_metal_defines`].
const FILTERS: [MultisampleDepthResolveFilter; 3] = [
    MultisampleDepthResolveFilter::Sample0,
    MultisampleDepthResolveFilter::Min,
    MultisampleDepthResolveFilter::Max,
];

/// One cell of the grid: the depth each sample is left holding, `None` for a sample no draw's mask
/// named, which keeps the pass's clear.
///
/// Four entries whatever the case's sample count; a case of `n` samples reads the first `n`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pattern {
    name: &'static str,
    samples: [Option<f32>; 4],
    /// Whether the three filters must agree on this pattern, which two of the seven do by
    /// construction.
    agrees: bool,
}

/// The depth a draw writes, as an exact eighth.
const fn eighth(of: u32) -> Option<f32> {
    Some(of as f32 / 8.0)
}

/// The seven patterns, in the order the module's table states them.
const PATTERNS: [Pattern; 7] = [
    Pattern {
        name: "ascending",
        samples: [eighth(1), eighth(2), eighth(3), eighth(4)],
        agrees: false,
    },
    Pattern {
        name: "descending",
        samples: [eighth(4), eighth(3), eighth(2), eighth(1)],
        agrees: false,
    },
    Pattern {
        name: "sample-0-between",
        samples: [eighth(3), eighth(1), eighth(5), eighth(2)],
        agrees: false,
    },
    Pattern {
        name: "a-sample-uncovered",
        samples: [eighth(3), eighth(1), None, eighth(2)],
        agrees: false,
    },
    Pattern {
        name: "every-sample-alike",
        samples: [eighth(2), eighth(2), eighth(2), eighth(2)],
        agrees: true,
    },
    Pattern {
        name: "a-tie-at-the-smallest",
        samples: [eighth(6), eighth(1), eighth(1), eighth(4)],
        agrees: false,
    },
    Pattern {
        name: "nothing-covered",
        samples: [None, None, None, None],
        agrees: true,
    },
];

/// The cell at index `at`, laid out row major.
fn cell(at: usize) -> ScissorBox {
    let column = (at as u32 % GRID) * CELL;
    let row = (at as u32 / GRID) * CELL;
    ScissorBox::new(
        u64::from(column),
        u64::from(row),
        u64::from(CELL),
        u64::from(CELL),
    )
}

/// The multisampled depth attachment at `samples` samples.
///
/// `MTLTextureUsageShaderRead` beside the render-target usage, and not a detail: a device with no
/// `MIN` or `MAX` depth resolve mode, and one that resolves a combined format's aspects one at a
/// time, reads the attachment's samples through a sampled view
/// (`metal2vulkan_runtime`'s `depth_stencil_resolve.rs`), and an attachment Metal made without the
/// usage is a texture that path cannot take. The usage is therefore part of what the case means on
/// every venue, not something the side may choose.
fn depth_target(samples: u32) -> TargetSpec {
    TargetSpec::read(DEPTH_FORMAT, [SIDE; 2]).multisampled(samples)
}

/// The single-sampled texture one cell's pass resolves into, which the blit reads back.
fn resolve_target() -> TargetSpec {
    TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2])
}

/// The multisampled colour attachment, which nothing reads.
fn color_target(samples: u32) -> TargetSpec {
    TargetSpec::plain(COLOR_FORMAT, [SIDE; 2]).multisampled(samples)
}

/// The layout a resolve texture's whole level 0 is read under: one `float` an element, so a diff
/// names the texel whose resolved depth moved.
fn depth_layout() -> ByteLayout {
    let row_pitch = DEPTH_TEXEL * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Depth,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The one colour state the pipeline holds.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(COLOR_FORMAT))]
}

/// The pipeline at `samples` samples, naming the depth format the pass attaches.
fn pipeline_spec<'a>(samples: u32, states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
        .multisampled(samples)
        .depth_stencil(Some(DEPTH_FORMAT), None)
}

/// The depth/stencil state every draw takes: write the depth it carries, whatever is there.
fn depth_state() -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state
}

/// The name a `(filter, store)` cell's site takes.
fn cell_name(filter: MultisampleDepthResolveFilter, store: StoreAction) -> String {
    format!("{}/{}", filter_name(filter), store_name(store))
}

/// The filter's enumerator name, which is also what the recording's site names hold.
fn filter_name(filter: MultisampleDepthResolveFilter) -> &'static str {
    match filter {
        MultisampleDepthResolveFilter::Sample0 => "MTLMultisampleDepthResolveFilterSample0",
        MultisampleDepthResolveFilter::Min => "MTLMultisampleDepthResolveFilterMin",
        MultisampleDepthResolveFilter::Max => "MTLMultisampleDepthResolveFilterMax",
        _ => "MTLMultisampleDepthResolveFilterUnknown",
    }
}

/// The store action's enumerator name.
fn store_name(store: StoreAction) -> &'static str {
    match store {
        StoreAction::MultisampleResolve => "MTLStoreActionMultisampleResolve",
        StoreAction::StoreAndMultisampleResolve => "MTLStoreActionStoreAndMultisampleResolve",
        _ => "MTLStoreActionUnknown",
    }
}

/// The steps one pass encodes: a draw a covered sample of every pattern, each at its own depth,
/// masked to its own sample and confined to its pattern's cell.
///
/// Every setting is stated at every draw, so no draw's answer rests on what the draw before it left
/// behind.
fn steps(samples: u32, pipeline: Handle, state: Handle) -> Vec<RenderStep> {
    let mut steps = Vec::new();
    for (at, pattern) in PATTERNS.iter().enumerate() {
        for (sample, held) in pattern.samples[..samples as usize].iter().enumerate() {
            let Some(depth) = held else { continue };
            steps.push(RenderStep::Pipeline(pipeline));
            steps.push(RenderStep::DepthStencilState(state));
            steps.push(RenderStep::VertexBytes {
                bytes: depth.to_le_bytes().to_vec(),
                index: 0,
            });
            steps.push(RenderStep::FragmentBytes {
                bytes: (1u32 << sample).to_le_bytes().to_vec(),
                index: 0,
            });
            steps.push(RenderStep::Scissor(cell(at)));
            steps.push(TRIANGLE);
        }
    }
    steps
}

/// The handles one pass names. A side's handle numbers are its own and none reaches a fingerprint,
/// so the zeroed set states the same pass.
#[derive(Clone, Copy)]
struct Handles {
    color: Handle,
    depth: Handle,
    resolve: Handle,
    pipeline: Handle,
    state: Handle,
}

impl Handles {
    const fn fingerprint() -> Self {
        let zero = Handle(0);
        Self {
            color: zero,
            depth: zero,
            resolve: zero,
            pipeline: zero,
            state: zero,
        }
    }
}

/// One cell's pass: the colour attachment stored `DontCare`, the depth attachment cleared and
/// resolved into `resolve` through `filter` under `store`, and the patterns' draws.
fn pass(
    samples: u32,
    filter: MultisampleDepthResolveFilter,
    store: StoreAction,
    handles: Handles,
) -> RenderPass {
    let mut depth = PassDepth::cleared(handles.depth, CLEAR_DEPTH);
    depth.resolve = Some(handles.resolve);
    depth.filter = filter;
    depth.store = store;
    RenderPass::clearing(vec![
        PassColor::cleared(handles.color, [0.0; 4]).stored(StoreAction::DontCare)
    ])
    .with_depth(depth)
    .drawing(steps(samples, handles.pipeline, handles.state))
}

/// The shader reference both texts of the fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: format!("{VERTEX}+{FRAGMENT}"),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(DepthFilters { samples: 4 }),
        Box::new(DepthFilters { samples: 2 }),
    ]
}

/// The grid at one sample count: three filters by two resolving store actions by seven per-sample
/// depth patterns.
///
/// **One case a sample count**, as the multisample families beside this one are built: a venue that
/// does not render the count refuses the target and the suite counts that case
/// [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving the whole grid
/// uncompared. Vulkan leaves 2 samples optional and lavapipe does not render them.
pub struct DepthFilters {
    samples: u32,
}

impl DepthFilters {
    /// The name the case's sample count gives it.
    fn case_name(&self) -> &'static str {
        match self.samples {
            2 => "resolve-depth-filters-2",
            _ => "resolve-depth-filters-4",
        }
    }

    /// The `(filter, store)` cells, in the order the case runs them.
    fn cells() -> Vec<(MultisampleDepthResolveFilter, StoreAction)> {
        let mut cells = Vec::new();
        for filter in FILTERS {
            for store in RESOLVING {
                cells.push((filter, store));
            }
        }
        cells
    }
}

impl Case for DepthFilters {
    fn name(&self) -> &'static str {
        self.case_name()
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened: every resolved depth Metal leaves is a depth we must leave. What is
        // declared is the notice set, because a device without a `MIN` or `MAX` depth resolve mode
        // reaches Metal's answer through a draw after the pass rather than through the store, and
        // the runtime announces that. On MoltenVK the modes are the device's and nothing fires; on
        // a device that lacks them the notice is what says the exact path ran.
        vec![
            crate::replay::notice_allowance(Degradation::ShaderResolve.code()),
            crate::replay::notice_allowance(Degradation::DontCareStore.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let states = colors();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes(
                "render-pipeline",
                &pipeline_spec(self.samples, &states).words(),
            ),
            Input::bytes("target:color", &color_target(self.samples).words()),
            Input::bytes("target:depth", &depth_target(self.samples).words()),
            Input::bytes("target:resolve", &resolve_target().words()),
            Input::bytes("clear:depth", &CLEAR_DEPTH.to_le_bytes()),
        ];
        for (filter, store) in Self::cells() {
            inputs.push(Input::bytes(
                format!("pass:{}", cell_name(filter, store)),
                &pass(self.samples, filter, store, Handles::fingerprint()).words(),
            ));
        }
        // The patterns by their own numbers, so a pattern that moved reads as a change to the
        // case's inputs rather than a quiet pass against another grid's answer.
        let mut rows = Vec::new();
        for pattern in PATTERNS {
            for held in pattern.samples {
                rows.extend(held.unwrap_or(f32::NAN).to_le_bytes());
            }
        }
        inputs.push(Input::bytes("patterns", &rows));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let states = colors();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &pipeline_spec(self.samples, &states)),
        )?;
        let color = made(
            sites,
            "target:color",
            side.target(&color_target(self.samples)),
        )?;
        let depth = made(
            sites,
            "target:depth",
            side.target(&depth_target(self.samples)),
        )?;
        let state = made(sites, "state", side.depth_stencil_state(&depth_state()))?;

        let cells = Self::cells();
        let mut resolves = Vec::new();
        let mut reads = Vec::new();
        for (filter, store) in &cells {
            let name = cell_name(*filter, *store);
            resolves.push(made(
                sites,
                &format!("target:resolve:{name}"),
                side.target(&resolve_target()),
            )?);
            reads.push(made(
                sites,
                &format!("buffer:read:{name}"),
                side.zeroed_buffer(DEPTH_PLANE as usize),
            )?);
        }

        // One command buffer: a pass a cell, each followed by the blit that reads its resolve
        // texture, so the whole grid is one submission on both sides.
        let mut work = Vec::new();
        let mut names = Vec::new();
        for (at, (filter, store)) in cells.iter().enumerate() {
            let name = cell_name(*filter, *store);
            work.push(RenderWork::Pass(pass(
                self.samples,
                *filter,
                *store,
                Handles {
                    color,
                    depth,
                    resolve: resolves[at],
                    pipeline,
                    state,
                },
            )));
            names.push(format!("pass:{name}"));
            work.push(RenderWork::ReadTexture {
                texture: resolves[at],
                into: reads[at],
                offset: 0,
                row_pitch: DEPTH_TEXEL * SIDE,
                size: [SIDE; 2],
                plane: Plane::Whole,
            });
            names.push(format!("read:{name}"));
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, what) in names.iter().enumerate() {
            pieces = pieces.with(Node::leaf(
                what.clone(),
                Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        sites.push(pieces);
        if let Some((what, failure)) = names
            .iter()
            .enumerate()
            .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (what, f)))
        {
            return Err(Abort {
                at: format!("pieces/{what}"),
                failure: failure.clone(),
            });
        }

        let mut resolved = Node::branch("resolved", Vec::new());
        for (at, (filter, store)) in cells.iter().enumerate() {
            let name = cell_name(*filter, *store);
            let bytes = call(
                sites,
                &format!("read-back/{name}"),
                side.read_buffer(reads[at], 0, DEPTH_PLANE as usize),
                |_| Value::Absent,
            )?;
            resolved = resolved.with(Node::leaf(name, Observed::bytes(bytes, depth_layout())));
        }
        sites.push(resolved);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// Cells a target holds.
    const CELLS: usize = (GRID * GRID) as usize;

    /// The depths the pattern's `samples` come to at `samples` samples: a sample the pattern leaves
    /// uncovered, and every sample past the pass's count, is the clear.
    fn depths(pattern: &Pattern, samples: u32) -> Vec<f32> {
        pattern.samples[..samples as usize]
            .iter()
            .map(|held| held.unwrap_or(CLEAR_DEPTH as f32))
            .collect()
    }

    /// What `filter` resolves the pattern's samples to at `samples` samples, which is Metal's rule
    /// applied here rather than read back from a recording.
    fn resolved(filter: MultisampleDepthResolveFilter, pattern: &Pattern, samples: u32) -> f32 {
        let held = depths(pattern, samples);
        match filter {
            MultisampleDepthResolveFilter::Min => {
                held.iter().copied().fold(f32::INFINITY, f32::min)
            }
            MultisampleDepthResolveFilter::Max => {
                held.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            }
            // `Sample0` and anything Apple adds: sample 0's own depth, which is what the name says and
            // the only reading a case may give a value it does not know.
            _ => held[0],
        }
    }

    /// **Every filter this family sweeps is a value Metal defines**, in the runtime's own order and
    /// with nothing left out.
    ///
    /// The denominator has to come from the vocabulary rather than from a list written here, or a
    /// value Apple adds lowers the family's ratio with nothing failing.
    #[test]
    fn the_swept_filters_are_every_value_metal_defines() {
        let raws: Vec<u64> = FILTERS.iter().map(|filter| filter.raw()).collect();
        assert_eq!(raws, vec![0, 1, 2], "the filters are not Metal's three");
        let names: BTreeSet<&str> = FILTERS.iter().map(|f| filter_name(*f)).collect();
        assert_eq!(names.len(), FILTERS.len(), "two filters share a name");
        assert!(
            !names.contains("MTLMultisampleDepthResolveFilterUnknown"),
            "a filter this module states no name for is swept"
        );
        // The two resolving store actions, and only those: a store that does not resolve is not in
        // the grid, for the reason the module states.
        let stores: Vec<u64> = RESOLVING.iter().map(|store| store.raw()).collect();
        assert_eq!(stores.len(), 2);
        assert!(stores.iter().all(|raw| *raw != 0));
        let store_names: BTreeSet<&str> = RESOLVING.iter().map(|s| store_name(*s)).collect();
        assert_eq!(store_names.len(), RESOLVING.len());
        assert!(!store_names.contains("MTLStoreActionUnknown"));
    }

    /// **The three filters answer three distinct seven-tuples**, which is what makes the grid
    /// non-vacuous: changing the filter changes at least one cell's resolved depth, at both sample
    /// counts.
    ///
    /// And the two patterns the module names as agreeing really do agree, while the other five
    /// separate at least two of the three filters: a sweep that could not fire is a sweep that
    /// compares nothing many times.
    #[test]
    fn the_three_filters_answer_three_distinct_tuples() {
        for samples in [2, 4] {
            let tuple = |filter| -> Vec<u32> {
                PATTERNS
                    .iter()
                    .map(|p| resolved(filter, p, samples).to_bits())
                    .collect()
            };
            let tuples: Vec<Vec<u32>> = FILTERS.iter().map(|f| tuple(*f)).collect();
            let distinct: BTreeSet<&Vec<u32>> = tuples.iter().collect();
            assert_eq!(
                distinct.len(),
                FILTERS.len(),
                "two filters answer alike at {samples} samples"
            );
            for pattern in PATTERNS {
                let answers: BTreeSet<u32> = FILTERS
                    .iter()
                    .map(|f| resolved(*f, &pattern, samples).to_bits())
                    .collect();
                match pattern.agrees {
                    true => assert_eq!(
                        answers.len(),
                        1,
                        "{} is named as agreeing and does not at {samples} samples",
                        pattern.name
                    ),
                    false => assert!(
                        answers.len() > 1,
                        "{} separates no filter at {samples} samples",
                        pattern.name
                    ),
                }
            }
        }
        // At four samples every pattern but the two named ones separates the filters, and the
        // `a-sample-uncovered` row's `Max` is the clear: the one cell that crosses the resolve with
        // the load action.
        let uncovered = PATTERNS
            .iter()
            .find(|p| p.name == "a-sample-uncovered")
            .expect("the uncovered pattern");
        assert_eq!(
            resolved(MultisampleDepthResolveFilter::Max, uncovered, 4),
            CLEAR_DEPTH as f32
        );
        assert_ne!(
            resolved(MultisampleDepthResolveFilter::Sample0, uncovered, 4),
            CLEAR_DEPTH as f32
        );
    }

    /// **Every depth the patterns state is an exact binary fraction inside the clip volume**, and
    /// none of them is the clear, so no cell's answer turns on rounding and a sample no draw
    /// covered is read rather than inferred.
    #[test]
    fn every_stated_depth_is_exact_and_inside_the_clip_volume() {
        for pattern in PATTERNS {
            for held in pattern.samples.iter().flatten() {
                assert!(
                    (0.0..1.0).contains(held),
                    "{} states the depth {held}, which is not inside [0, 1)",
                    pattern.name
                );
                // An eighth is exact in a `float` and in a `Depth32Float` texel alike, so the
                // round trip through the attachment changes nothing.
                assert_eq!(*held * 8.0, (*held * 8.0).round());
                assert_ne!(f64::from(*held), CLEAR_DEPTH);
            }
        }
        assert_eq!(PATTERNS.len(), 7);
        assert!(PATTERNS.len() <= CELLS, "the patterns do not fit the grid");
        let names: BTreeSet<&str> = PATTERNS.iter().map(|p| p.name).collect();
        assert_eq!(names.len(), PATTERNS.len(), "two patterns share a name");
    }

    /// **Every pass of every case stays inside its attachments and names what it holds**, and each
    /// draw states all six of its settings.
    #[test]
    fn every_pass_is_inside_the_target_and_states_each_draws_settings() {
        // The check sees a rectangle that reaches past the target, so a green run means the cells
        // are inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, u64::from(SIDE) + 1, 1).past([SIDE; 2]));
        for case in all() {
            let samples = match case.name() {
                "resolve-depth-filters-2" => 2,
                _ => 4,
            };
            let cells = DepthFilters::cells();
            assert_eq!(cells.len(), FILTERS.len() * RESOLVING.len());
            let distinct: BTreeSet<String> = cells
                .iter()
                .map(|(filter, store)| cell_name(*filter, *store))
                .collect();
            assert_eq!(distinct.len(), cells.len(), "two cells share a name");
            let drawn: usize = PATTERNS
                .iter()
                .map(|p| p.samples[..samples as usize].iter().flatten().count())
                .sum();
            let encoded = steps(samples, Handle(0), Handle(0));
            assert_eq!(encoded.len(), 6 * drawn, "{}", case.name());
            for step in &encoded {
                if let RenderStep::Scissor(rect) = step {
                    assert!(
                        !rect.past([SIDE; 2]),
                        "{}: {rect:?} reaches past the target",
                        case.name()
                    );
                }
            }
            // The resolve texture is single-sampled and the attachments are not, which is the one
            // shape Metal resolves at all.
            assert_eq!(resolve_target().samples, 1);
            assert_eq!(depth_target(samples).samples, samples);
            assert_eq!(color_target(samples).samples, samples);
            // The attachment a resolve draw may have to read is of shader-read usage.
            assert!(depth_target(samples).shader_read);
            assert_eq!(depth_layout().extent(), DEPTH_PLANE as usize);
        }
    }

    /// **The family names its two cases apart**, under the settings the module states.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 2);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // The declared allowances loosen no value: each names a notice and nothing else.
            assert_eq!(case.allowances().len(), 2);
        }
    }

    /// The one outcome of `case`'s recording, where the tree holds one. A case whose answer moved
    /// between runs is the finding, so that is a failure here rather than one run read as the
    /// answer.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so the case's answer moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The `length` bytes of `case`'s `resolved/<what>` site.
    fn recorded(case: &str, what: &str, length: usize) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("resolved")
            .and_then(|node| node.child(what))
            .unwrap_or_else(|| panic!("{case}: no resolved/{what} site"));
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: resolved/{what} holds no bytes");
        };
        assert_eq!(bytes.len(), length);
        Some(bytes.clone())
    }

    /// The `float` the texel at `(x, y)` of a read-back plane holds.
    fn texel(bytes: &[u8], x: u32, y: u32) -> f32 {
        let at = (y * SIDE + x) as usize * DEPTH_TEXEL as usize;
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[at..at + 4]);
        f32::from_le_bytes(word)
    }

    /// The depth every texel of cell `at` holds, which must be one value: the cell's pass resolved
    /// the same samples at each of its pixels.
    fn cell_depth(bytes: &[u8], at: usize) -> f32 {
        let rect = cell(at);
        let first = texel(bytes, rect.x as u32, rect.y as u32);
        for y in rect.y as u32..(rect.y + rect.height) as u32 {
            for x in rect.x as u32..(rect.x + rect.width) as u32 {
                assert_eq!(
                    texel(bytes, x, y).to_bits(),
                    first.to_bits(),
                    "cell {at} is not one depth: ({x}, {y}) holds {}",
                    texel(bytes, x, y)
                );
            }
        }
        first
    }

    /// **Metal's recorded resolve holds what each filter's rule says, at every cell of the grid,
    /// and the two store actions resolve alike.**
    #[test]
    fn metal_resolved_each_pattern_as_its_filter_says() {
        for case in all() {
            let samples = match case.name() {
                "resolve-depth-filters-2" => 2,
                _ => 4,
            };
            let mut planes = Vec::new();
            for (filter, store) in DepthFilters::cells() {
                let name = cell_name(filter, store);
                let Some(bytes) = recorded(case.name(), &name, DEPTH_PLANE as usize) else {
                    println!("{}: no recording of {name} in the tree yet", case.name());
                    return;
                };
                for (at, pattern) in PATTERNS.iter().enumerate() {
                    assert_eq!(
                        cell_depth(&bytes, at).to_bits(),
                        resolved(filter, pattern, samples).to_bits(),
                        "{}: {name} did not resolve {} as its filter says (it holds {})",
                        case.name(),
                        pattern.name,
                        cell_depth(&bytes, at)
                    );
                }
                // A cell no pattern took is the clear, resolved: what says no draw escaped its
                // rectangle.
                for at in PATTERNS.len()..CELLS {
                    assert_eq!(
                        f64::from(cell_depth(&bytes, at)),
                        CLEAR_DEPTH,
                        "{}: {name} cell {at} holds no pattern and is not the clear",
                        case.name()
                    );
                }
                planes.push((filter, bytes));
            }
            // The resolve is the same under both store actions, cell for cell: the pair's own
            // claim, which no per-value sweep of either member can make.
            for (at, (filter, bytes)) in planes.iter().enumerate() {
                if at % RESOLVING.len() == 0 {
                    let (_, other) = &planes[at + 1];
                    assert_eq!(
                        bytes,
                        other,
                        "{}: {} resolves differently under the two store actions",
                        case.name(),
                        filter_name(*filter)
                    );
                }
            }
        }
    }
}
