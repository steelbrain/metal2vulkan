//! `MTLMultisampleStencilResolveFilter` crossed with the per-sample stencil a pass left, and with
//! the **depth attachment's own resolve filter** beside it: what a multisampled stencil
//! attachment's resolve texture holds.
//!
//! `conf/dspairs` closed `MTLMultisampleDepthResolveFilter` at 3 of 3 behaviourally and left the
//! stencil filter at **0 of 2**. One of its two values can be stated on a pass whose stencil is a
//! texture of its own, and one cannot; this module takes the first, and
//! [`crate::cases::stencil_resolved_sample`] takes the second on a combined attachment:
//!
//! | case | what it crosses |
//! |---|---|
//! | `resolve-stencil-sample0` | `Sample0` by two resolving store actions by seven per-sample stencil patterns, on a stencil-only pass |
//! | `resolve-stencil-under-depth-filters` | the same stencil filter by **every** `MTLMultisampleDepthResolveFilter`, both aspects resolving out of one pass into two textures |
//!
//! # Why the second case exists, and what only it can see
//!
//! `Sample0` is one value, so a case over it alone has no filter axis to sweep: it can show that
//! sample 0's stencil is what the resolve holds, and no more. The question a *pair* asks is the one
//! the runtime's own code makes answerable and no case has ever put beside Metal — the stencil
//! resolve and the depth resolve are chosen together
//! (`metal2vulkan_runtime`'s `resolve.rs`, one `match` over `(stencil_resolves, stencil_filter)`
//! whose other arm reads `depth_filter`), so a side that let the depth attachment's filter decide
//! the stencil's would be **right on one of the three depth filters and wrong on the other two**.
//! No sweep of either filter on its own can see that. `resolve-stencil-under-depth-filters` states
//! both in one pass, three times, and reads both resolve textures:
//!
//! - the **depth** column moves with the depth filter, which is what says the depth filter reached
//!   the device at all;
//! - the **stencil** column is the same byte in all three cells, which is the claim — and the
//!   patterns are chosen so that a stencil that followed the depth filter would answer a
//!   *different* byte under `Min` and under `Max` while still agreeing under `Sample0`.
//!
//! Two patterns carry that weight. `sample-0-between` holds its smallest depth at sample 1 and its
//! largest at sample 2, so the stencil of the depth-picked sample is `0x11` under `Min` and `0x55`
//! under `Max` where sample 0's own is `0x33`. `sample-0-uncovered` has no draw at sample 0 at all,
//! so its stencil is the **clear** while its resolved depth is not: a side that resolved the
//! stencil of whichever sample it had a value for answers `0x11` or `0x55` there instead of
//! `0x40`.
//!
//! # How a pass puts a different stencil in each sample
//!
//! As `conf/dspairs`' depth grid puts a different depth in each sample, by the same owned fixture
//! and for the same reason: which samples an edge covers is the rasterizer's own grid, so a case
//! built on partial coverage would compare a device's sample pattern rather than its resolve. The
//! pass states the samples outright. `rt6_resolves`' `rt6_resolve_masked` returns the `uint` its
//! fragment buffer 0 holds as `[[sample_mask]]`, and `rt6_resolve_full` puts the fullscreen
//! triangle at the depth its vertex buffer 0 holds. One draw a sample, each masked to its own
//! sample, under a depth/stencil state whose two faces compare `Always` and whose pass operation
//! is `Replace` with every mask bit set, writes `setStencilReferenceValue:`'s byte into that
//! sample's stencil alone and leaves every other sample at the pass's `clearStencil`.
//!
//! The stencil reference is the draw's own, stated at every draw, so no draw's answer rests on
//! what the draw before it left set.
//!
//! # The seven patterns, and what each tells apart
//!
//! `-` is a sample no draw's mask named, which keeps the clear `CLEAR_STENCIL`.
//!
//! | pattern | sample 0 | 1 | 2 | 3 | `Sample0` resolves |
//! |---|---|---|---|---|---|
//! | ascending | `0x11` | `0x22` | `0x33` | `0x44` | `0x11` |
//! | descending | `0x44` | `0x33` | `0x22` | `0x11` | `0x44` |
//! | sample-0-between | `0x33` | `0x11` | `0x55` | `0x22` | `0x33` |
//! | sample-0-uncovered | `-` | `0x11` | `0x55` | `0x22` | `0x40` |
//! | only-sample-0 | `0x66` | `-` | `-` | `-` | `0x66` |
//! | every-sample-alike | `0x77` | `0x77` | `0x77` | `0x77` | `0x77` |
//! | nothing-covered | `-` | `-` | `-` | `-` | `0x40` |
//!
//! Six distinct answers over seven patterns, and the set separates `Sample0` from every nearby
//! rule a side could have implemented instead: from the **last** sample (descending and
//! only-sample-0 disagree), from the smallest or largest **byte** (sample-0-between disagrees
//! both ways), from *any covered* sample (sample-0-uncovered disagrees), and from reading the
//! clear where a sample was written or a sample where nothing was (only-sample-0 and
//! nothing-covered are the pair that catch that). `every-sample-alike` and `nothing-covered` are
//! **indistinguishable by construction** — there is one value to pick in the first and only the
//! clear in the second — and they are in the grid because they are what the other five are read
//! against.
//!
//! # The store action's half of the cross
//!
//! The stencil-only case states both store actions Metal defines that resolve, as the depth grid
//! does, and asks the pair's own question: **is the resolve itself the same under both?** Seven
//! cells say so under each, so a side whose resolve depends on the store action — one that
//! resolved from the stored copy under one and from the tile under the other — fails. A per-value
//! sweep of either store action cannot make that claim.
//!
//! `resolve-stencil-under-depth-filters` states `MultisampleResolve` on both aspects and **not**
//! `StoreAndMultisampleResolve`, deliberately: a separate depth texture stored beside a separate
//! stencil one renders through a combined scratch image whose multisampled aspects are copied back
//! after the pass, and on a device whose fragment depth does not store a value unchanged that copy
//! is an announced near-equivalent (`metal2vulkan_runtime`'s `depth_stencil_scratch.rs`,
//! `Degradation::DrawnDepthCopy`). A resolving store that keeps no samples needs no copy back, so
//! the cell asks about the resolve and nothing else. The store-action pair is already answered by
//! the case above it and by `resolve-depth-filters-4`.
//!
//! # What is not asked, and why
//!
//! - **`MTLMultisampleStencilResolveFilterDepthResolvedSample`**, the other of the two values.
//!   Metal takes it only where the depth and the stencil are **one** attachment and the resolve
//!   texture is of the attachment's own format, so neither case here can state it: both attach the
//!   stencil as a texture of its own. It is
//!   [`crate::cases::stencil_resolved_sample`]'s, which crosses both stencil filters with every
//!   depth filter on one combined `Depth32Float_Stencil8` attachment and reads its resolve
//!   texture's two aspects through [`RenderWork::ReadTexture`]'s plane.
//! - **A non-resolving store action beside a `resolveTexture`.** Measured to resolve nothing and
//!   leave the texture untouched, and a shape **Metal's API validation stops**; record mode runs a
//!   validated leg, so a case for it is one keystroke from an assertion that ends the process.
//!   It stays a runtime-lane fact.
//! - **Two samples.** Vulkan leaves 2 samples optional and lavapipe does not render them, so the
//!   depth grid gave the count its own case. Here the count is held at four, the one every venue
//!   we run answers, and the sample count is not this family's axis: what the patterns say is
//!   which sample's stencil a filter takes, which four samples state more sharply than two.
//!
//! # Why this family is safe
//!
//! One command buffer a case: two or three render passes over an 8 by 8 target and one blit each
//! (two for the pair case, which reads both resolve textures). At most twenty-three draws of three
//! vertices a pass, no loop in either stage, every scissor rectangle inside the attachments, every
//! vertex inside the clip volume, every depth inside `[0, 1]`, every stencil reference a byte, and
//! no sample count past what the venue answers. A venue that does not render the case's sample
//! count refuses the target and the suite counts the case
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving it uncompared.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, PassStencil, Plane, ReadAt, RenderPass, RenderPipelineSpec,
    RenderStep, RenderWork, ScissorBox, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation};
use metal2vulkan_runtime::{
    Degradation, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
};

/// The owned fixture both cases draw with: the fullscreen triangle at buffer 0's depth, and the
/// fragment whose sample mask is buffer 0's.
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

/// The samples every case of this family renders at. Four, for the reason the module states.
const SAMPLES: u32 = 4;

/// The stencil attachment's and its resolve texture's format: one byte a texel, which a blit reads
/// back as itself.
const STENCIL_FORMAT: PixelFormat = PixelFormat::Stencil8;

/// Bytes a stencil resolve texture's whole level 0 takes: one a texel.
const STENCIL_PLANE: u32 = SIDE * SIDE;

/// The separate depth attachment's and its resolve texture's format, for the pair case: one
/// `float` a texel, which a blit reads back as itself and which holds every depth the patterns
/// state exactly.
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// Bytes one texel of [`DEPTH_FORMAT`] takes.
const DEPTH_TEXEL: u32 = 4;

/// Bytes a depth resolve texture's whole level 0 takes.
const DEPTH_PLANE: u32 = DEPTH_TEXEL * SIDE * SIDE;

/// The colour attachment's format. The fragment function returns a `float4` at colour 0, so a pass
/// must attach one; nothing reads it and it is stored `DontCare`.
const COLOR_FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// `clearStencil`: a byte no pattern's draw writes, so a sample no draw covered is read rather
/// than inferred.
const CLEAR_STENCIL: u32 = 0x40;

/// `clearDepth` in the pair case: the far plane, which no pattern's drawn depth equals.
const CLEAR_DEPTH: f64 = 1.0;

/// The depth the stencil-only case's draws carry. The pass attaches no depth, so any finite depth
/// inside the clip volume draws the same.
const FLAT_DEPTH: f32 = 0.5;

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

/// Every `MTLMultisampleDepthResolveFilter`, which the pair case crosses the stencil filter with.
const DEPTH_FILTERS: [MultisampleDepthResolveFilter; 3] = [
    MultisampleDepthResolveFilter::Sample0,
    MultisampleDepthResolveFilter::Min,
    MultisampleDepthResolveFilter::Max,
];

/// The one `MTLMultisampleStencilResolveFilter` this family states, for the reason the module
/// gives.
const STENCIL_FILTER: MultisampleStencilResolveFilter = MultisampleStencilResolveFilter::Sample0;

/// One cell of the grid: the stencil each sample is left holding, and, where the case states a
/// depth attachment, the depth beside it. `None` is a sample no draw's mask named, which keeps the
/// pass's clears.
///
/// Four entries, one a sample at `SAMPLES`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pattern {
    name: &'static str,
    /// The stencil byte and the depth each covered sample's draw writes.
    samples: [Option<(u8, f32)>; 4],
    /// Whether every depth filter must leave this pattern's **depth** alone, which two of the
    /// seven do by construction.
    depth_agrees: bool,
}

/// The depth a draw writes, as an exact eighth: exact in a `float` and in a `Depth32Float` texel
/// alike, so the round trip through the attachment changes nothing.
const fn eighth(of: u32) -> f32 {
    of as f32 / 8.0
}

/// The seven patterns, in the order the module's table states them.
const PATTERNS: [Pattern; 7] = [
    Pattern {
        name: "ascending",
        samples: [
            Some((0x11, eighth(1))),
            Some((0x22, eighth(2))),
            Some((0x33, eighth(3))),
            Some((0x44, eighth(4))),
        ],
        depth_agrees: false,
    },
    Pattern {
        name: "descending",
        samples: [
            Some((0x44, eighth(4))),
            Some((0x33, eighth(3))),
            Some((0x22, eighth(2))),
            Some((0x11, eighth(1))),
        ],
        depth_agrees: false,
    },
    Pattern {
        name: "sample-0-between",
        samples: [
            Some((0x33, eighth(3))),
            Some((0x11, eighth(1))),
            Some((0x55, eighth(5))),
            Some((0x22, eighth(2))),
        ],
        depth_agrees: false,
    },
    Pattern {
        name: "sample-0-uncovered",
        samples: [
            None,
            Some((0x11, eighth(1))),
            Some((0x55, eighth(5))),
            Some((0x22, eighth(2))),
        ],
        depth_agrees: false,
    },
    Pattern {
        name: "only-sample-0",
        samples: [Some((0x66, eighth(6))), None, None, None],
        depth_agrees: false,
    },
    Pattern {
        name: "every-sample-alike",
        samples: [
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
        ],
        depth_agrees: true,
    },
    Pattern {
        name: "nothing-covered",
        samples: [None, None, None, None],
        depth_agrees: true,
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

/// The multisampled stencil attachment.
///
/// `MTLTextureUsageShaderRead` beside the render-target usage, and not a detail: a device that
/// resolves a combined format's aspects one at a time, or has no mode for the depth filter beside
/// it, reads the attachment's samples through a sampled view
/// (`metal2vulkan_runtime`'s `depth_stencil_resolve.rs` and `depth_stencil_scratch.rs`), and an
/// attachment Metal made without the usage is a texture those paths cannot take. The usage is
/// therefore part of what the case means on every venue, not something the side may choose.
fn stencil_target() -> TargetSpec {
    TargetSpec::read(STENCIL_FORMAT, [SIDE; 2]).multisampled(SAMPLES)
}

/// The single-sampled texture a pass resolves its stencil into, which the blit reads back.
fn stencil_resolve_target() -> TargetSpec {
    TargetSpec::plain(STENCIL_FORMAT, [SIDE; 2])
}

/// The multisampled depth attachment the pair case states beside the stencil one, as a **separate
/// texture**: the one shape in which a resolved stencil can be read back at all, since Metal takes
/// a resolve texture of the attachment's own format and the suite's one texture read has no aspect.
fn depth_target() -> TargetSpec {
    TargetSpec::read(DEPTH_FORMAT, [SIDE; 2]).multisampled(SAMPLES)
}

/// The single-sampled texture the pair case resolves its depth into.
fn depth_resolve_target() -> TargetSpec {
    TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2])
}

/// The multisampled colour attachment, which nothing reads.
fn color_target() -> TargetSpec {
    TargetSpec::plain(COLOR_FORMAT, [SIDE; 2]).multisampled(SAMPLES)
}

/// The layout a stencil resolve texture's whole level 0 is read under: one byte an element, so a
/// diff names the texel whose resolved stencil moved.
fn stencil_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Stencil,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: SIDE,
        row_pitch: SIDE,
        rows: SIDE,
        image_pitch: SIDE * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The layout a depth resolve texture's whole level 0 is read under: one `float` an element.
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

/// The one colour state a pipeline holds.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(COLOR_FORMAT))]
}

/// The pipeline, naming the attachment formats the pass attaches: the stencil always, the depth
/// only in the pair case.
fn pipeline_spec<'a>(depth: bool, states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
        .multisampled(SAMPLES)
        .depth_stencil(depth.then_some(DEPTH_FORMAT), Some(STENCIL_FORMAT))
}

/// The depth/stencil state every draw takes: both faces compare `Always` and replace the stencil
/// with the draw's own reference under every mask bit, and the depth — where the case attaches one
/// — is written whatever is there.
///
/// Both faces are stated alike on purpose: which face Metal's fullscreen triangle takes is the
/// winding's question and `depthstencil-facing-faces`' subject, not this family's, so no cell's
/// answer may turn on it.
fn state(depth: bool) -> DepthStencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = CompareFunction::Always;
    face.stencil_failure_operation = StencilOperation::Keep;
    face.depth_stencil_pass_operation = StencilOperation::Replace;
    face.read_mask = 0xff;
    face.write_mask = 0xff;
    let mut descriptor = DepthStencilDescriptor::default();
    descriptor.front_face_stencil = face;
    descriptor.back_face_stencil = face;
    descriptor.depth_compare_function = CompareFunction::Always;
    descriptor.depth_write_enabled = depth;
    descriptor
}

/// The steps one pass encodes: a draw a covered sample of every pattern, each writing its own
/// stencil byte and its own depth, masked to its own sample and confined to its pattern's cell.
///
/// Every setting is stated at every draw, so no draw's answer rests on what the draw before it
/// left behind.
fn steps(depth: bool, pipeline: Handle, held: Handle) -> Vec<RenderStep> {
    let mut steps = Vec::new();
    for (at, pattern) in PATTERNS.iter().enumerate() {
        for (sample, carried) in pattern.samples.iter().enumerate() {
            let Some((stencil, z)) = carried else {
                continue;
            };
            let carried_depth = match depth {
                true => *z,
                false => FLAT_DEPTH,
            };
            steps.push(RenderStep::Pipeline(pipeline));
            steps.push(RenderStep::DepthStencilState(held));
            steps.push(RenderStep::StencilReference(u32::from(*stencil)));
            steps.push(RenderStep::VertexBytes {
                bytes: carried_depth.to_le_bytes().to_vec(),
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
    stencil: Handle,
    stencil_resolve: Handle,
    /// The separate depth attachment and its resolve texture, which the pair case states.
    depth: Option<(Handle, Handle)>,
    pipeline: Handle,
    state: Handle,
}

impl Handles {
    /// The set a fingerprint is taken over: every handle zeroed, the depth pair present exactly
    /// where the case states one.
    const fn fingerprint(depth: bool) -> Self {
        let zero = Handle(0);
        Self {
            color: zero,
            stencil: zero,
            stencil_resolve: zero,
            depth: match depth {
                true => Some((zero, zero)),
                false => None,
            },
            pipeline: zero,
            state: zero,
        }
    }
}

/// One cell's pass: the colour attachment stored `DontCare`, the stencil attachment cleared and
/// resolved through `STENCIL_FILTER` under `store`, the separate depth attachment resolved
/// through `filter` under the same store where the case states one, and the patterns' draws.
fn pass(filter: MultisampleDepthResolveFilter, store: StoreAction, handles: Handles) -> RenderPass {
    let mut stencil = PassStencil::cleared(handles.stencil, CLEAR_STENCIL);
    stencil.resolve = Some(handles.stencil_resolve);
    stencil.filter = STENCIL_FILTER;
    stencil.store = store;
    let built = RenderPass::clearing(vec![
        PassColor::cleared(handles.color, [0.0; 4]).stored(StoreAction::DontCare)
    ])
    .with_stencil(stencil);
    let built = match handles.depth {
        Some((texture, resolve)) => {
            let mut depth = PassDepth::cleared(texture, CLEAR_DEPTH);
            depth.resolve = Some(resolve);
            depth.filter = filter;
            depth.store = store;
            built.with_depth(depth)
        }
        None => built,
    };
    built.drawing(steps(
        handles.depth.is_some(),
        handles.pipeline,
        handles.state,
    ))
}

/// The filter's enumerator name, which is also what the recording's site names hold.
fn depth_filter_name(filter: MultisampleDepthResolveFilter) -> &'static str {
    match filter {
        MultisampleDepthResolveFilter::Sample0 => "MTLMultisampleDepthResolveFilterSample0",
        MultisampleDepthResolveFilter::Min => "MTLMultisampleDepthResolveFilterMin",
        MultisampleDepthResolveFilter::Max => "MTLMultisampleDepthResolveFilterMax",
        _ => "MTLMultisampleDepthResolveFilterUnknown",
    }
}

/// The stencil filter's enumerator name.
fn stencil_filter_name(filter: MultisampleStencilResolveFilter) -> &'static str {
    match filter {
        MultisampleStencilResolveFilter::Sample0 => "MTLMultisampleStencilResolveFilterSample0",
        MultisampleStencilResolveFilter::DepthResolvedSample => {
            "MTLMultisampleStencilResolveFilterDepthResolvedSample"
        }
        _ => "MTLMultisampleStencilResolveFilterUnknown",
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

/// Which of the family's two crosses a case states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cross {
    /// The stencil filter by the two resolving store actions, on a stencil-only pass.
    Stores,
    /// The stencil filter by every depth resolve filter, both aspects resolving.
    DepthFilters,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(StencilResolves {
            cross: Cross::Stores,
        }),
        Box::new(StencilResolves {
            cross: Cross::DepthFilters,
        }),
    ]
}

/// One of the family's two crosses, over the seven per-sample stencil patterns.
pub struct StencilResolves {
    cross: Cross,
}

impl StencilResolves {
    /// The name the case's cross gives it.
    fn case_name(&self) -> &'static str {
        match self.cross {
            Cross::Stores => "resolve-stencil-sample0",
            Cross::DepthFilters => "resolve-stencil-under-depth-filters",
        }
    }

    /// Whether the case states a separate depth attachment beside the stencil one.
    fn has_depth(&self) -> bool {
        self.cross == Cross::DepthFilters
    }

    /// The `(depth filter, store action)` cells, in the order the case runs them.
    ///
    /// The stores cross holds the depth filter at Metal's own default, which no cell reads because
    /// the pass attaches no depth; the depth-filter cross holds the store at
    /// `MultisampleResolve` for the reason the module states.
    fn cells(&self) -> Vec<(MultisampleDepthResolveFilter, StoreAction)> {
        match self.cross {
            Cross::Stores => RESOLVING
                .iter()
                .map(|store| (MultisampleDepthResolveFilter::Sample0, *store))
                .collect(),
            Cross::DepthFilters => DEPTH_FILTERS
                .iter()
                .map(|filter| (*filter, StoreAction::MultisampleResolve))
                .collect(),
        }
    }

    /// The name a cell's sites take: the stencil filter every cell states, and then whichever of
    /// the two the case sweeps.
    ///
    /// The stencil filter is in **every** site name although it is one value, so a reader of a
    /// recording sees which `MTLMultisampleStencilResolveFilter` the bytes beside it were resolved
    /// under rather than having to find the module.
    fn cell_name(&self, filter: MultisampleDepthResolveFilter, store: StoreAction) -> String {
        let stencil = stencil_filter_name(STENCIL_FILTER);
        match self.cross {
            Cross::Stores => format!("{stencil}/{}", store_name(store)),
            Cross::DepthFilters => format!("{stencil}/{}", depth_filter_name(filter)),
        }
    }
}

impl Case for StencilResolves {
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
        // **No value is loosened**: every resolved byte and every resolved depth Metal leaves is
        // one we must leave, and the whole declared set is notices. Each is a `Performance`
        // degradation — a structurally different path to the same bits — and each is named with
        // the shape of this case that provokes it, so a reader can tell a notice this family
        // earns from one copied in.
        let mut named = vec![
            // The colour attachment nothing reads is stored `DontCare`, and a device whose
            // `DontCare` store is not Metal's announces that it took the near-equivalent.
            // Measured not to fire on MoltenVK, where the store is the device's own.
            crate::replay::notice_allowance(Degradation::DontCareStore.code()),
        ];
        if self.has_depth() {
            named.extend([
                // The pair case's depth and stencil are **two textures**, which renders through a
                // scratch image of the combined format: announced on every venue, and measured
                // firing on MoltenVK.
                crate::replay::notice_allowance(Degradation::SeparateDepthStencil.code()),
                // And their two aspects resolve into two textures, which resolves one aspect at a
                // time: announced on every venue, and measured firing on MoltenVK.
                crate::replay::notice_allowance(Degradation::OneAspectResolve.code()),
                // A device with no `MIN` or `MAX` depth resolve mode reaches Metal's depth
                // through a draw after the pass rather than through the store. On MoltenVK the
                // modes are the device's and this does not fire; on a device that lacks them the
                // notice is what says the exact path ran.
                crate::replay::notice_allowance(Degradation::ShaderResolve.code()),
            ]);
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let states = colors();
        let depth = self.has_depth();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &pipeline_spec(depth, &states).words()),
            Input::bytes("target:color", &color_target().words()),
            Input::bytes("target:stencil", &stencil_target().words()),
            Input::bytes("target:stencil-resolve", &stencil_resolve_target().words()),
            Input::bytes("clear:stencil", &CLEAR_STENCIL.to_le_bytes()),
            Input::bytes("filter:stencil", &STENCIL_FILTER.raw().to_le_bytes()),
        ];
        if depth {
            inputs.push(Input::bytes("target:depth", &depth_target().words()));
            inputs.push(Input::bytes(
                "target:depth-resolve",
                &depth_resolve_target().words(),
            ));
            inputs.push(Input::bytes("clear:depth", &CLEAR_DEPTH.to_le_bytes()));
        }
        for (filter, store) in self.cells() {
            inputs.push(Input::bytes(
                format!("pass:{}", self.cell_name(filter, store)),
                &pass(filter, store, Handles::fingerprint(depth)).words(),
            ));
        }
        // The patterns by their own numbers, so a pattern that moved reads as a change to the
        // case's inputs rather than a quiet pass against another grid's answer. A sample no draw
        // covered writes the clear's byte and a NaN depth, neither of which any draw states.
        let mut rows = Vec::new();
        for pattern in PATTERNS {
            for carried in pattern.samples {
                let (stencil, z) = match carried {
                    Some(held) => (u32::from(held.0), held.1),
                    None => (CLEAR_STENCIL, f32::NAN),
                };
                rows.extend(stencil.to_le_bytes());
                rows.extend(z.to_le_bytes());
            }
        }
        inputs.push(Input::bytes("patterns", &rows));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let depth = self.has_depth();
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let states = colors();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &pipeline_spec(depth, &states)),
        )?;
        let color = made(sites, "target:color", side.target(&color_target()))?;
        let stencil = made(sites, "target:stencil", side.target(&stencil_target()))?;
        let depth_texture = match depth {
            true => Some(made(sites, "target:depth", side.target(&depth_target()))?),
            false => None,
        };
        let held_state = made(sites, "state", side.depth_stencil_state(&state(depth)))?;

        let cells = self.cells();
        // One resolve texture and one read-back buffer a cell and an aspect, so no cell's answer
        // can be another cell's leftovers.
        let mut stencil_resolves = Vec::new();
        let mut stencil_reads = Vec::new();
        let mut depth_resolves = Vec::new();
        let mut depth_reads = Vec::new();
        for (filter, store) in &cells {
            let name = self.cell_name(*filter, *store);
            stencil_resolves.push(made(
                sites,
                &format!("target:stencil-resolve:{name}"),
                side.target(&stencil_resolve_target()),
            )?);
            stencil_reads.push(made(
                sites,
                &format!("buffer:stencil-read:{name}"),
                side.zeroed_buffer(STENCIL_PLANE as usize),
            )?);
            if depth {
                depth_resolves.push(made(
                    sites,
                    &format!("target:depth-resolve:{name}"),
                    side.target(&depth_resolve_target()),
                )?);
                depth_reads.push(made(
                    sites,
                    &format!("buffer:depth-read:{name}"),
                    side.zeroed_buffer(DEPTH_PLANE as usize),
                )?);
            }
        }

        // One command buffer: a pass a cell, each followed by the blit that reads its resolve
        // texture or textures, so the whole grid is one submission on both sides.
        let mut work = Vec::new();
        let mut names = Vec::new();
        for (at, (filter, store)) in cells.iter().enumerate() {
            let name = self.cell_name(*filter, *store);
            work.push(RenderWork::Pass(pass(
                *filter,
                *store,
                Handles {
                    color,
                    stencil,
                    stencil_resolve: stencil_resolves[at],
                    depth: depth_texture.map(|texture| (texture, depth_resolves[at])),
                    pipeline,
                    state: held_state,
                },
            )));
            names.push(format!("pass:{name}"));
            work.push(RenderWork::ReadTexture {
                texture: stencil_resolves[at],
                into: stencil_reads[at],
                offset: 0,
                row_pitch: SIDE,
                size: [SIDE; 2],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            names.push(format!("read:stencil:{name}"));
            if depth {
                work.push(RenderWork::ReadTexture {
                    texture: depth_resolves[at],
                    into: depth_reads[at],
                    offset: 0,
                    row_pitch: DEPTH_TEXEL * SIDE,
                    size: [SIDE; 2],
                    plane: Plane::Whole,
                    at: ReadAt::BASE,
                });
                names.push(format!("read:depth:{name}"));
            }
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

        let mut resolved = Node::branch("resolved-stencil", Vec::new());
        for (at, (filter, store)) in cells.iter().enumerate() {
            let name = self.cell_name(*filter, *store);
            let bytes = call(
                sites,
                &format!("read-back/stencil/{name}"),
                side.read_buffer(stencil_reads[at], 0, STENCIL_PLANE as usize),
                |_| Value::Absent,
            )?;
            resolved = resolved.with(Node::leaf(name, Observed::bytes(bytes, stencil_layout())));
        }
        sites.push(resolved);
        if depth {
            let mut planes = Node::branch("resolved-depth", Vec::new());
            for (at, (filter, store)) in cells.iter().enumerate() {
                let name = self.cell_name(*filter, *store);
                let bytes = call(
                    sites,
                    &format!("read-back/depth/{name}"),
                    side.read_buffer(depth_reads[at], 0, DEPTH_PLANE as usize),
                    |_| Value::Absent,
                )?;
                planes = planes.with(Node::leaf(name, Observed::bytes(bytes, depth_layout())));
            }
            sites.push(planes);
        }
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

    /// What `STENCIL_FILTER` resolves the pattern's stencils to: sample 0's own byte, and the
    /// clear where no draw covered sample 0.
    ///
    /// Metal's rule applied here rather than read back from a recording, which is what lets the
    /// recording be held to a reason.
    fn resolved_stencil(pattern: &Pattern) -> u8 {
        match pattern.samples[0] {
            Some((stencil, _)) => stencil,
            None => CLEAR_STENCIL as u8,
        }
    }

    /// The depths the pattern's samples come to: a sample the pattern leaves uncovered is the
    /// clear.
    fn depths(pattern: &Pattern) -> Vec<f32> {
        pattern
            .samples
            .iter()
            .map(|held| match held {
                Some((_, z)) => *z,
                None => CLEAR_DEPTH as f32,
            })
            .collect()
    }

    /// What `filter` resolves the pattern's depths to, Metal's rule applied here.
    fn resolved_depth(filter: MultisampleDepthResolveFilter, pattern: &Pattern) -> f32 {
        let held = depths(pattern);
        match filter {
            MultisampleDepthResolveFilter::Min => {
                held.iter().copied().fold(f32::INFINITY, f32::min)
            }
            MultisampleDepthResolveFilter::Max => {
                held.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            }
            // `Sample0` and anything Apple adds: sample 0's own depth, which is what the name says
            // and the only reading a case may give a value it does not know.
            _ => held[0],
        }
    }

    /// The stencil a side that let the **depth** filter pick the stencil's sample would answer:
    /// the stencil of the sample whose depth that filter took, of equal depths the first.
    ///
    /// This is the wrong answer the pair exists to catch, written out so the test can hold that it
    /// really differs from Metal's rather than trusting that it would.
    fn stencil_if_the_depth_filter_picked(
        filter: MultisampleDepthResolveFilter,
        pattern: &Pattern,
    ) -> u8 {
        let picked = resolved_depth(filter, pattern);
        let at = depths(pattern)
            .iter()
            .position(|z| z.to_bits() == picked.to_bits())
            .expect("the resolved depth is one of the samples'");
        match pattern.samples[at] {
            Some((stencil, _)) => stencil,
            None => CLEAR_STENCIL as u8,
        }
    }

    /// **Every filter this family states is a value Metal defines**, and the two store actions are
    /// the two that resolve.
    ///
    /// The denominator has to come from the vocabulary rather than from a list written here, or a
    /// value Apple adds lowers the family's ratio with nothing failing.
    #[test]
    fn the_stated_filters_are_values_metal_defines() {
        assert_eq!(STENCIL_FILTER.raw(), 0);
        assert_eq!(
            stencil_filter_name(STENCIL_FILTER),
            "MTLMultisampleStencilResolveFilterSample0"
        );
        // The other value exists and is named, so the module's claim that it is unreachable is
        // about a value the vocabulary really has rather than one nobody spelled.
        assert_eq!(
            stencil_filter_name(MultisampleStencilResolveFilter::DepthResolvedSample),
            "MTLMultisampleStencilResolveFilterDepthResolvedSample"
        );
        let raws: Vec<u64> = DEPTH_FILTERS.iter().map(|filter| filter.raw()).collect();
        assert_eq!(
            raws,
            vec![0, 1, 2],
            "the depth filters are not Metal's three"
        );
        let names: BTreeSet<&str> = DEPTH_FILTERS
            .iter()
            .map(|f| depth_filter_name(*f))
            .collect();
        assert_eq!(names.len(), DEPTH_FILTERS.len(), "two filters share a name");
        assert!(!names.contains("MTLMultisampleDepthResolveFilterUnknown"));
        let stores: Vec<u64> = RESOLVING.iter().map(|store| store.raw()).collect();
        assert_eq!(stores.len(), 2);
        assert!(stores.iter().all(|raw| *raw != 0));
        let store_names: BTreeSet<&str> = RESOLVING.iter().map(|s| store_name(*s)).collect();
        assert_eq!(store_names.len(), RESOLVING.len());
        assert!(!store_names.contains("MTLStoreActionUnknown"));
    }

    /// **The seven patterns answer six distinct stencils**, and the set separates `Sample0` from
    /// every nearby rule a side could have implemented instead: the last covered sample, the
    /// smallest or largest byte, any covered sample at all, and the clear.
    ///
    /// A sweep that could not fire is a sweep that compares nothing many times, so this is the
    /// property the grid rests on and it is held with no device.
    #[test]
    fn the_patterns_separate_sample_zero_from_its_neighbours() {
        let answers: Vec<u8> = PATTERNS.iter().map(resolved_stencil).collect();
        let distinct: BTreeSet<u8> = answers.iter().copied().collect();
        assert_eq!(distinct.len(), 6, "the patterns answer {distinct:?}");
        // The last covered sample, which `descending` and `only-sample-0` tell from sample 0's.
        let last = |p: &Pattern| -> u8 {
            p.samples
                .iter()
                .flatten()
                .last()
                .map(|held| held.0)
                .unwrap_or(CLEAR_STENCIL as u8)
        };
        assert!(
            PATTERNS.iter().any(|p| last(p) != resolved_stencil(p)),
            "no pattern tells sample 0 from the last covered sample"
        );
        // The smallest and the largest byte, which `sample-0-between` tells from sample 0's both
        // ways.
        let between = PATTERNS
            .iter()
            .find(|p| p.name == "sample-0-between")
            .expect("the between pattern");
        let bytes: Vec<u8> = between.samples.iter().flatten().map(|h| h.0).collect();
        let smallest = *bytes.iter().min().expect("a covered sample");
        let largest = *bytes.iter().max().expect("a covered sample");
        assert_ne!(resolved_stencil(between), smallest);
        assert_ne!(resolved_stencil(between), largest);
        // Any covered sample at all, which `sample-0-uncovered` tells from sample 0's: its answer
        // is the clear although three of its samples were written.
        let uncovered = PATTERNS
            .iter()
            .find(|p| p.name == "sample-0-uncovered")
            .expect("the uncovered pattern");
        assert_eq!(resolved_stencil(uncovered), CLEAR_STENCIL as u8);
        assert_eq!(uncovered.samples.iter().flatten().count(), 3);
        // And the pair that catches reading the clear where a sample was written, or a sample
        // where nothing was.
        let only = PATTERNS
            .iter()
            .find(|p| p.name == "only-sample-0")
            .expect("the single-sample pattern");
        assert_ne!(resolved_stencil(only), CLEAR_STENCIL as u8);
        let nothing = PATTERNS
            .iter()
            .find(|p| p.name == "nothing-covered")
            .expect("the empty pattern");
        assert_eq!(resolved_stencil(nothing), CLEAR_STENCIL as u8);
        let names: BTreeSet<&str> = PATTERNS.iter().map(|p| p.name).collect();
        assert_eq!(names.len(), PATTERNS.len(), "two patterns share a name");
        assert!(PATTERNS.len() <= CELLS, "the patterns do not fit the grid");
    }

    /// **The cross is not vacuous in either direction**, which is the whole reason the pair case
    /// exists.
    ///
    /// The depth column must move with the depth filter — three distinct seven-tuples — while the
    /// stencil column must not move at all; and a side that let the depth filter pick the stencil's
    /// sample must answer a **different byte** in at least one cell of at least two of the three
    /// filters, or the cross could not see the mistake it is built for.
    #[test]
    fn the_depth_filter_moves_the_depth_and_must_not_move_the_stencil() {
        let tuple = |filter| -> Vec<u32> {
            PATTERNS
                .iter()
                .map(|p| resolved_depth(filter, p).to_bits())
                .collect()
        };
        let tuples: Vec<Vec<u32>> = DEPTH_FILTERS.iter().map(|f| tuple(*f)).collect();
        let distinct: BTreeSet<&Vec<u32>> = tuples.iter().collect();
        assert_eq!(
            distinct.len(),
            DEPTH_FILTERS.len(),
            "two depth filters answer alike, so the cross's depth column is flat"
        );
        for pattern in PATTERNS {
            let answers: BTreeSet<u32> = DEPTH_FILTERS
                .iter()
                .map(|f| resolved_depth(*f, &pattern).to_bits())
                .collect();
            match pattern.depth_agrees {
                true => assert_eq!(
                    answers.len(),
                    1,
                    "{} is named as agreeing on the depth and does not",
                    pattern.name
                ),
                false => assert!(
                    answers.len() > 1,
                    "{} separates no depth filter",
                    pattern.name
                ),
            }
        }
        // The stencil is sample 0's whatever the depth filter: the claim, stated as a property of
        // the rule rather than of a recording.
        for filter in DEPTH_FILTERS {
            for pattern in PATTERNS {
                let _ = filter;
                assert_eq!(resolved_stencil(&pattern), resolved_stencil(&pattern));
            }
        }
        // And the mistake the cross exists to catch really differs: under `Min` and under `Max`
        // at least one pattern's depth-picked stencil is not sample 0's, while under `Sample0`
        // every one of them is.
        for filter in [
            MultisampleDepthResolveFilter::Min,
            MultisampleDepthResolveFilter::Max,
        ] {
            let moved: Vec<&str> = PATTERNS
                .iter()
                .filter(|p| stencil_if_the_depth_filter_picked(filter, p) != resolved_stencil(p))
                .map(|p| p.name)
                .collect();
            assert!(
                !moved.is_empty(),
                "{}: a stencil that followed the depth filter would answer alike everywhere",
                depth_filter_name(filter)
            );
        }
        for pattern in PATTERNS {
            assert_eq!(
                stencil_if_the_depth_filter_picked(
                    MultisampleDepthResolveFilter::Sample0,
                    &pattern
                ),
                resolved_stencil(&pattern),
                "{}: `Sample0` is the cell the mistake passes, and it does not",
                pattern.name
            );
        }
    }

    /// **Every stated depth is an exact binary fraction inside the clip volume and none is the
    /// clear**, and every stated stencil is a byte that is not the clear.
    #[test]
    fn every_stated_value_is_exact_and_apart_from_the_clear() {
        for pattern in PATTERNS {
            for (stencil, z) in pattern.samples.iter().flatten() {
                assert!(
                    (0.0..1.0).contains(z),
                    "{} states the depth {z}, which is not inside [0, 1)",
                    pattern.name
                );
                assert_eq!(*z * 8.0, (*z * 8.0).round());
                assert_ne!(f64::from(*z), CLEAR_DEPTH);
                assert_ne!(u32::from(*stencil), CLEAR_STENCIL);
            }
        }
        assert!((0.0..1.0).contains(&FLAT_DEPTH));
        const { assert!(CLEAR_STENCIL <= 0xff) };
    }

    /// **Every pass of every case stays inside its attachments and states each draw's settings**,
    /// and the resolve shape is the one Metal resolves at all.
    #[test]
    fn every_pass_is_inside_the_target_and_states_each_draws_settings() {
        // The check sees a rectangle that reaches past the target, so a green run means the cells
        // are inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, u64::from(SIDE) + 1, 1).past([SIDE; 2]));
        for case in all() {
            let depth = case.name() == "resolve-stencil-under-depth-filters";
            let drawn: usize = PATTERNS
                .iter()
                .map(|p| p.samples.iter().flatten().count())
                .sum();
            let encoded = steps(depth, Handle(0), Handle(0));
            // Seven calls a draw: the pipeline, the state, the stencil reference, the two
            // bindings, the scissor and the draw itself.
            assert_eq!(encoded.len(), 7 * drawn, "{}", case.name());
            for step in &encoded {
                if let RenderStep::Scissor(rect) = step {
                    assert!(
                        !rect.past([SIDE; 2]),
                        "{}: {rect:?} reaches past the target",
                        case.name()
                    );
                }
            }
            // Every draw states its own stencil reference, so no draw's stencil rests on what the
            // draw before it left set.
            let references = encoded
                .iter()
                .filter(|step| matches!(step, RenderStep::StencilReference(_)))
                .count();
            assert_eq!(references, drawn, "{}", case.name());
            // The resolve textures are single-sampled and the attachments are not, which is the
            // one shape Metal resolves at all.
            assert_eq!(stencil_resolve_target().samples, 1);
            assert_eq!(depth_resolve_target().samples, 1);
            assert_eq!(stencil_target().samples, SAMPLES);
            assert_eq!(depth_target().samples, SAMPLES);
            assert_eq!(color_target().samples, SAMPLES);
            // An attachment a resolve draw or an aspect copy may have to read is of shader-read
            // usage.
            assert!(stencil_target().shader_read);
            assert!(depth_target().shader_read);
            assert_eq!(stencil_layout().extent(), STENCIL_PLANE as usize);
            assert_eq!(depth_layout().extent(), DEPTH_PLANE as usize);
        }
    }

    /// **The family names its two cases apart**, each over the cells its cross states, and the
    /// pair case attaches a depth where the stores case does not.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 2);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // The declared allowances loosen no value: each names a notice and nothing else, and
            // the pair case declares the three its separate textures and split resolve provoke
            // beside the one both cases share.
            let expected = match case.name() {
                "resolve-stencil-under-depth-filters" => 4,
                _ => 1,
            };
            assert_eq!(case.allowances().len(), expected, "{}", case.name());
        }
        let stores = StencilResolves {
            cross: Cross::Stores,
        };
        let filters = StencilResolves {
            cross: Cross::DepthFilters,
        };
        assert_eq!(stores.cells().len(), RESOLVING.len());
        assert_eq!(filters.cells().len(), DEPTH_FILTERS.len());
        assert!(!stores.has_depth());
        assert!(filters.has_depth());
        for case in [&stores, &filters] {
            let seen: BTreeSet<String> = case
                .cells()
                .iter()
                .map(|(filter, store)| case.cell_name(*filter, *store))
                .collect();
            assert_eq!(
                seen.len(),
                case.cells().len(),
                "{}: two cells share a name",
                case.case_name()
            );
        }
        // The stores case attaches no depth, so no cell of it can read one; the pair case does,
        // and its pipeline names the format.
        let states = colors();
        assert_eq!(pipeline_spec(false, &states).depth_format, None);
        assert_eq!(
            pipeline_spec(true, &states).depth_format,
            Some(DEPTH_FORMAT)
        );
        assert_eq!(
            pipeline_spec(false, &states).stencil_format,
            Some(STENCIL_FORMAT)
        );
        assert!(!state(false).depth_write_enabled);
        assert!(state(true).depth_write_enabled);
        assert_eq!(
            pass(
                MultisampleDepthResolveFilter::Sample0,
                StoreAction::MultisampleResolve,
                Handles::fingerprint(false)
            )
            .depth,
            None
        );
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

    /// The `length` bytes of `case`'s `<branch>/<what>` site.
    fn recorded(case: &str, branch: &str, what: &str, length: usize) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child(branch)
            .and_then(|node| node.child(what))?;
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: {branch}/{what} holds no bytes");
        };
        assert_eq!(bytes.len(), length);
        Some(bytes.clone())
    }

    /// The byte the texel at `(x, y)` of a read-back stencil plane holds.
    fn stencil_texel(bytes: &[u8], x: u32, y: u32) -> u8 {
        bytes[(y * SIDE + x) as usize]
    }

    /// The `float` the texel at `(x, y)` of a read-back depth plane holds.
    fn depth_texel(bytes: &[u8], x: u32, y: u32) -> f32 {
        let at = (y * SIDE + x) as usize * DEPTH_TEXEL as usize;
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[at..at + 4]);
        f32::from_le_bytes(word)
    }

    /// The one value every texel of cell `at` holds, which must be one value: the cell's pass
    /// resolved the same samples at each of its pixels.
    fn one_value<T: PartialEq + std::fmt::Debug>(
        bytes: &[u8],
        at: usize,
        read: impl Fn(&[u8], u32, u32) -> T,
    ) -> T {
        let rect = cell(at);
        let first = read(bytes, rect.x as u32, rect.y as u32);
        for y in rect.y as u32..(rect.y + rect.height) as u32 {
            for x in rect.x as u32..(rect.x + rect.width) as u32 {
                let held = read(bytes, x, y);
                assert_eq!(
                    held, first,
                    "cell {at} is not one value: ({x}, {y}) holds {held:?}"
                );
            }
        }
        first
    }

    /// **Metal's recorded stencil resolve is sample 0's byte at every cell of the grid, the two
    /// resolving store actions resolve alike, and the stencil does not follow the depth filter.**
    ///
    /// The third claim is the pair's, and no sweep of either filter on its own can make it: it is
    /// read as the three depth-filter cells holding one stencil plane while their depth planes
    /// differ.
    #[test]
    fn metal_resolved_sample_zeros_stencil_under_every_cell() {
        for case in all() {
            let depth = case.name() == "resolve-stencil-under-depth-filters";
            let built = StencilResolves {
                cross: match depth {
                    true => Cross::DepthFilters,
                    false => Cross::Stores,
                },
            };
            let mut planes = Vec::new();
            for (filter, store) in built.cells() {
                let name = built.cell_name(filter, store);
                let Some(bytes) = recorded(
                    case.name(),
                    "resolved-stencil",
                    &name,
                    STENCIL_PLANE as usize,
                ) else {
                    println!("{}: no recording of {name} in the tree yet", case.name());
                    return;
                };
                for (at, pattern) in PATTERNS.iter().enumerate() {
                    assert_eq!(
                        one_value(&bytes, at, stencil_texel),
                        resolved_stencil(pattern),
                        "{}: {name} did not resolve {}'s stencil to sample 0's",
                        case.name(),
                        pattern.name
                    );
                }
                // A cell no pattern took is the clear, resolved: what says no draw escaped its
                // rectangle.
                for at in PATTERNS.len()..CELLS {
                    assert_eq!(
                        u32::from(one_value(&bytes, at, stencil_texel)),
                        CLEAR_STENCIL,
                        "{}: {name} cell {at} holds no pattern and is not the clear",
                        case.name()
                    );
                }
                let depths = match depth {
                    true => recorded(case.name(), "resolved-depth", &name, DEPTH_PLANE as usize),
                    false => None,
                };
                if let Some(held) = &depths {
                    for (at, pattern) in PATTERNS.iter().enumerate() {
                        assert_eq!(
                            one_value(held, at, depth_texel).to_bits(),
                            resolved_depth(filter, pattern).to_bits(),
                            "{}: {name} did not resolve {}'s depth as its filter says",
                            case.name(),
                            pattern.name
                        );
                    }
                }
                planes.push((name, bytes, depths));
            }
            // The stencil plane is the same under every cell of either cross, byte for byte: the
            // stores case's claim that the resolve does not depend on the store action, and the
            // pair case's that it does not depend on the depth filter.
            let (first, held, _) = &planes[0];
            for (name, other, _) in &planes[1..] {
                assert_eq!(
                    held,
                    other,
                    "{}: {first} and {name} resolve the stencil differently",
                    case.name()
                );
            }
            // And in the pair case the depth planes are **not** all alike, or the cross would be
            // comparing one answer three times.
            if depth {
                let depths: BTreeSet<&Vec<u8>> =
                    planes.iter().filter_map(|(_, _, d)| d.as_ref()).collect();
                assert_eq!(
                    depths.len(),
                    DEPTH_FILTERS.len(),
                    "{}: the three depth filters did not answer three planes",
                    case.name()
                );
            }
        }
    }
}
