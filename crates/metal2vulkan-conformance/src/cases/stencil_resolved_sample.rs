//! `MTLMultisampleStencilResolveFilter` crossed with `MTLMultisampleDepthResolveFilter` on **one
//! combined `Depth32Float_Stencil8` attachment**: which sample's stencil the resolve texture holds.
//!
//! `resolve-stencil-sample0` and `resolve-stencil-under-depth-filters`
//! ([`crate::cases::stencil_resolve_pairs`]) closed `Sample0` on a stencil-only pass and on a pass
//! whose depth was a **separate** texture. Neither can state the stencil filter's other value:
//! Metal takes `DepthResolvedSample` only where the depth and the stencil are one attachment, and
//! it takes a resolve texture of the attachment's own format, so the resolve is a
//! `Depth32Float_Stencil8` texture whose **stencil** has to be read back — which needs
//! `MTLBlitOptionStencilFromDepthStencil` ([`Plane::Stencil`]).
//!
//! | what | this module's one case |
//! |---|---|
//! | the axes | both `MTLMultisampleStencilResolveFilter` values by all three `MTLMultisampleDepthResolveFilter` values |
//! | the attachment | one multisampled `Depth32Float_Stencil8` texture, attached as the pass's depth **and** its stencil |
//! | the resolve | one single-sampled `Depth32Float_Stencil8` texture a cell, named by both attachments |
//! | the read | both aspects of that texture, by [`Plane::Stencil`] and [`Plane::Depth`] |
//!
//! # What only the cross decides
//!
//! Under `Sample0` the resolved stencil is sample 0's whatever the depth filter; under
//! `DepthResolvedSample` it is the stencil of the sample the **depth** filter took. So the answer
//! is a function of the pair and of neither value alone:
//!
//! - a side that read the stencil filter and ignored the depth filter is right on all six cells of
//!   `Sample0` and on the `DepthResolvedSample`/`Sample0` cell, and wrong on the other two;
//! - a side that took the depth-picked sample's stencil whatever the stencil filter is right on
//!   every `DepthResolvedSample` cell and wrong on two of the three `Sample0` cells;
//! - a per-value sweep of either axis can see neither, which is what makes this a crossing rather
//!   than two sweeps.
//!
//! Our runtime answers `DepthResolvedSample` after `Min` or `Max` through a whole private path —
//! `metal2vulkan_runtime`'s `depth_stencil_resolve.rs`, eight bit-plane draws after the pass that
//! read every sample's depth and stencil through sampled views — and until this case no run ever
//! put a byte of it beside Metal's.
//!
//! # The patterns, and what each tells apart
//!
//! One pass puts a different `(stencil, depth)` in each sample of each cell, by the fixture and the
//! method the resolve families share: `rt6_resolve_masked` returns its fragment buffer's `uint` as
//! `[[sample_mask]]` and `rt6_resolve_full` draws the fullscreen triangle at its vertex buffer's
//! depth, so one masked draw a sample, under a state whose faces compare `Always` and replace the
//! stencil with the draw's own reference, writes that sample's stencil and depth alone.
//!
//! `-` is a sample no draw's mask named, which keeps the pass's clears (`CLEAR_STENCIL`,
//! `CLEAR_DEPTH`). Each cell below is `stencil@depth`.
//!
//! | pattern | sample 0 | 1 | 2 | 3 | `Sample0` | `DepthResolvedSample` after `Min` | after `Max` |
//! |---|---|---|---|---|---|---|---|
//! | ascending | `0x11@1/8` | `0x22@2/8` | `0x33@3/8` | `0x44@4/8` | `0x11` | `0x11` | `0x44` |
//! | descending | `0x44@4/8` | `0x33@3/8` | `0x22@2/8` | `0x11@1/8` | `0x44` | `0x11` | `0x44` |
//! | sample-0-between | `0x33@3/8` | `0x11@1/8` | `0x55@5/8` | `0x22@2/8` | `0x33` | `0x11` | `0x55` |
//! | sample-0-uncovered | `-` | `0x11@1/8` | `0x55@5/8` | `0x22@2/8` | `0x40` | `0x11` | `0x40` |
//! | only-sample-0 | `0x66@6/8` | `-` | `-` | `-` | `0x66` | `0x66` | `0x40` |
//! | every-sample-alike | `0x77@2/8` | `0x77@2/8` | `0x77@2/8` | `0x77@2/8` | `0x77` | `0x77` | `0x77` |
//! | nothing-covered | `-` | `-` | `-` | `-` | `0x40` | `0x40` | `0x40` |
//!
//! Five of the seven answer three different bytes across the three filters under
//! `DepthResolvedSample`, and `sample-0-between` is the one that separates all three at once while
//! sample 0's own stencil is neither the smallest nor the largest byte: a side that took the
//! smallest or the largest **stencil** rather than the stencil of the depth-picked sample answers
//! `0x11` or `0x55` where Metal answers `0x33`. `descending` and `only-sample-0` separate the
//! depth-picked sample from the first and the last sample. `sample-0-uncovered` and
//! `only-sample-0` are the pair that catch a side that skipped the samples no draw covered: the
//! clear depth is the far plane, so an uncovered sample is the **`Max` pick**, and its stencil is
//! the clear.
//!
//! `every-sample-alike` and `nothing-covered` are indistinguishable by construction — one value to
//! pick, and only the clear — and they are here because they are what the other five are read
//! against. The depths are exact eighths, so each is exact in a `float` and in a
//! `Depth32Float_Stencil8` texel alike and no cell's answer rests on a rounding.
//!
//! **No cell's answer rests on Metal's tie-break among equal depths** (our runtime's own probe:
//! the last sample after `Min` and the first after `Max`). Two samples of a pattern hold one depth
//! only where both are uncovered and so both hold the clear depth, and such samples hold the
//! **clear stencil** too, so whichever of them Metal's tie-break takes the resolved stencil is the
//! same byte. Every *covered* sample of a pattern holds a depth of its own. The tie-break is a
//! question of its own and not this cross's, and
//! `tests::a_tie_among_equal_depths_cannot_move_a_cell` is what holds the grid to it.
//!
//! # Why the depth plane is read too
//!
//! A case that asked for the stencil aspect and was answered the **depth** aspect would prove
//! nothing if the bytes happened to agree, so the two aspects are made to hold different values
//! and both are read:
//!
//! - every stencil byte of the grid is one of `0x11`, `0x22`, `0x33`, `0x44`, `0x55`, `0x66`,
//!   `0x77` and the clear `0x40`, and **none of them is zero**;
//! - every depth is an exact eighth or the far plane, whose `f32` bits are
//!   `0x3e000000`..`0x3f800000` — each with a **zero** low byte.
//!
//! So a side whose stencil read returned the depth plane answers `0x00` at every texel of every
//! cell, and a side whose depth read returned the stencil plane answers the stencil bytes where
//! Metal answers the depths. The two aspects' copy units differ as well — one byte a texel
//! against four — so a read that lost its aspect is refused for the buffer's length before it can
//! answer agreeing bytes at all.
//!
//! The depth column is also what says the depth filter reached the device: it moves with the depth
//! filter in five of the seven patterns, under **both** stencil filters, which no stencil answer
//! alone could show.
//!
//! Each read-back site is named `<MTLBlitOption>/<cell>`, so the recording says which aspect its
//! bytes came from, and the suite's option-space axis reads the option from the recording rather
//! than only from the suite's shared vocabulary ([`Plane::option_name`]).
//!
//! # What is not asked, and why
//!
//! - **`StoreAndMultisampleResolve`.** `resolve-stencil-sample0` and `resolve-depth-filters-4`
//!   already say the resolve is the same under both resolving store actions. A combined attachment
//!   that keeps its samples as well is a second question (the stored copy beside the resolve), not
//!   this cross's.
//! - **A non-resolving store action beside a `resolveTexture`.** A shape Metal's API validation
//!   stops, and the suite's record mode runs a validated leg: it stays a runtime-lane fact, as the
//!   sibling modules state.
//!   The **separate** depth and stencil shape, which `resolve-stencil-under-depth-filters` owns.
//! - **A sample count other than four.** Vulkan leaves 2 samples optional and lavapipe does not
//!   render them; four is the count every venue we run answers, and what the patterns say is which
//!   sample's stencil a filter takes, which four samples state more sharply than two.
//!
//! # Why this case is safe
//!
//! One command buffer: six render passes over an 8 by 8 target and two blits each. Twenty-three
//! draws of three vertices a pass, no loop in either stage, every scissor rectangle inside the
//! attachments, every vertex inside the clip volume, every depth inside `[0, 1]`, every stencil
//! reference a byte, and no sample count past what the venue answers. A venue that does not render
//! the case's sample count or make its format refuses the target and the suite counts the case
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving it uncompared.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, PassStencil, Plane, RenderPass, RenderPipelineSpec,
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

/// The case's name: its recording's file name and its row in the report.
const NAME: &str = "resolve-stencil-depth-resolved-sample";

/// The owned fixture the resolve families share: the fullscreen triangle at buffer 0's depth, and
/// the fragment whose sample mask is buffer 0's.
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

/// The samples the case renders at. Four, for the reason the module states.
const SAMPLES: u32 = 4;

/// The one attachment's format and its resolve texture's: the combined depth/stencil format, which
/// is the only shape in which Metal takes
/// [`MultisampleStencilResolveFilter::DepthResolvedSample`].
const COMBINED: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// Bytes one texel of the combined format's **depth** aspect takes, which is also the copy unit a
/// [`Plane::Depth`] read moves.
const DEPTH_TEXEL: u32 = 4;

/// Bytes a [`Plane::Stencil`] read of the resolve texture's whole level 0 takes: one a texel.
const STENCIL_PLANE: u32 = SIDE * SIDE;

/// Bytes a [`Plane::Depth`] read of the resolve texture's whole level 0 takes.
const DEPTH_PLANE: u32 = DEPTH_TEXEL * SIDE * SIDE;

/// The colour attachment's format. The fragment function returns a `float4` at colour 0, so the
/// pass must attach one; nothing reads it and it is stored `DontCare`.
const COLOR_FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// `clearStencil`: a byte no pattern's draw writes, so a sample no draw covered is read rather
/// than inferred. Not zero, so a stencil read answered the depth plane diverges at every texel.
const CLEAR_STENCIL: u32 = 0x40;

/// `clearDepth`: the far plane, which no pattern's drawn depth equals, so an uncovered sample is
/// the `Max` pick and its stencil is the clear.
const CLEAR_DEPTH: f64 = 1.0;

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How many runs record mode takes the case over: one command buffer, one queue and nothing a
/// device may reorder, so the runs are expected to fold into one outcome and a case whose answer
/// moved between runs is the finding rather than one run being read as the answer.
const RUNS: u32 = 8;

/// Both `MTLMultisampleStencilResolveFilter` values, named value by value so the suite's
/// option-space axis reads them, and held against the runtime's own list by
/// `tests::the_swept_stencil_filters_are_every_value_metal_defines`.
const STENCIL_FILTERS: [MultisampleStencilResolveFilter; 2] = [
    MultisampleStencilResolveFilter::Sample0,
    MultisampleStencilResolveFilter::DepthResolvedSample,
];

/// Every `MTLMultisampleDepthResolveFilter`, the other axis of the cross.
const DEPTH_FILTERS: [MultisampleDepthResolveFilter; 3] = [
    MultisampleDepthResolveFilter::Sample0,
    MultisampleDepthResolveFilter::Min,
    MultisampleDepthResolveFilter::Max,
];

/// One cell of the grid: the stencil byte and the depth each sample is left holding. `None` is a
/// sample no draw's mask named, which keeps the pass's clears.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pattern {
    name: &'static str,
    samples: [Option<(u8, f32)>; 4],
}

/// The depth a draw writes, as an exact eighth: exact in a `float` and in the combined format's
/// depth texel alike, so the round trip through the attachment changes nothing. Its low byte is
/// zero, which is what makes a stencil read answered the depth plane diverge.
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
    },
    Pattern {
        name: "descending",
        samples: [
            Some((0x44, eighth(4))),
            Some((0x33, eighth(3))),
            Some((0x22, eighth(2))),
            Some((0x11, eighth(1))),
        ],
    },
    Pattern {
        name: "sample-0-between",
        samples: [
            Some((0x33, eighth(3))),
            Some((0x11, eighth(1))),
            Some((0x55, eighth(5))),
            Some((0x22, eighth(2))),
        ],
    },
    Pattern {
        name: "sample-0-uncovered",
        samples: [
            None,
            Some((0x11, eighth(1))),
            Some((0x55, eighth(5))),
            Some((0x22, eighth(2))),
        ],
    },
    Pattern {
        name: "only-sample-0",
        samples: [Some((0x66, eighth(6))), None, None, None],
    },
    Pattern {
        name: "every-sample-alike",
        samples: [
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
            Some((0x77, eighth(2))),
        ],
    },
    Pattern {
        name: "nothing-covered",
        samples: [None, None, None, None],
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

/// The one multisampled attachment, which the pass names as both its depth and its stencil.
///
/// `MTLTextureUsageShaderRead` beside the render-target usage, and not a detail: the stencil of the
/// depth-picked sample is drawn after the pass from the attachment's samples read through sampled
/// views (`metal2vulkan_runtime`'s `depth_stencil_resolve.rs`), and an attachment Metal made
/// without the usage is a texture that path cannot take. The usage is therefore part of what the
/// case means on every venue, not something the side may choose.
fn combined_target() -> TargetSpec {
    TargetSpec::read(COMBINED, [SIDE; 2]).multisampled(SAMPLES)
}

/// The single-sampled texture a cell resolves **both** aspects into, of the attachment's own
/// format, which Metal requires of a resolve texture.
fn combined_resolve_target() -> TargetSpec {
    TargetSpec::plain(COMBINED, [SIDE; 2])
}

/// The multisampled colour attachment, which nothing reads.
fn color_target() -> TargetSpec {
    TargetSpec::plain(COLOR_FORMAT, [SIDE; 2]).multisampled(SAMPLES)
}

/// The layout a [`Plane::Stencil`] read of the resolve texture is compared under: one byte an
/// element, so a diff names the texel whose resolved stencil moved.
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

/// The layout a [`Plane::Depth`] read of the resolve texture is compared under: one `float` an
/// element.
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

/// The pipeline, naming the combined format as **both** attachment formats, which is what Metal
/// takes for one combined attachment.
fn pipeline_spec<'a>(states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
        .multisampled(SAMPLES)
        .depth_stencil(Some(COMBINED), Some(COMBINED))
}

/// The depth/stencil state every draw takes: both faces compare `Always` and replace the stencil
/// with the draw's own reference under every mask bit, and the depth is written whatever is there.
///
/// Both faces are stated alike on purpose: which face Metal's fullscreen triangle takes is the
/// winding's question and `depthstencil-facing-faces`' subject, not this case's, so no cell's
/// answer may turn on it.
fn state() -> DepthStencilDescriptor {
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
    descriptor.depth_write_enabled = true;
    descriptor
}

/// The steps one pass encodes: a draw a covered sample of every pattern, each writing its own
/// stencil byte and its own depth, masked to its own sample and confined to its pattern's cell.
///
/// Every setting is stated at every draw, so no draw's answer rests on what the draw before it
/// left behind.
fn steps(pipeline: Handle, held: Handle) -> Vec<RenderStep> {
    let mut steps = Vec::new();
    for (at, pattern) in PATTERNS.iter().enumerate() {
        for (sample, carried) in pattern.samples.iter().enumerate() {
            let Some((stencil, z)) = carried else {
                continue;
            };
            steps.push(RenderStep::Pipeline(pipeline));
            steps.push(RenderStep::DepthStencilState(held));
            steps.push(RenderStep::StencilReference(u32::from(*stencil)));
            steps.push(RenderStep::VertexBytes {
                bytes: z.to_le_bytes().to_vec(),
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
    /// The one combined attachment, named as both the depth and the stencil.
    combined: Handle,
    /// The one combined resolve texture, named by both attachments.
    resolve: Handle,
    pipeline: Handle,
    state: Handle,
}

impl Handles {
    /// The set a fingerprint is taken over: every handle zeroed.
    const fn fingerprint() -> Self {
        let zero = Handle(0);
        Self {
            color: zero,
            combined: zero,
            resolve: zero,
            pipeline: zero,
            state: zero,
        }
    }
}

/// One cell's pass: the colour attachment stored `DontCare`, and the one combined attachment
/// cleared and attached twice — as the depth, resolved through `depth`, and as the stencil,
/// resolved through `stencil` — both into the one resolve texture under `MultisampleResolve`.
fn pass(
    stencil_filter: MultisampleStencilResolveFilter,
    depth_filter: MultisampleDepthResolveFilter,
    handles: Handles,
) -> RenderPass {
    let mut depth = PassDepth::cleared(handles.combined, CLEAR_DEPTH);
    depth.resolve = Some(handles.resolve);
    depth.filter = depth_filter;
    depth.store = StoreAction::MultisampleResolve;
    let mut stencil = PassStencil::cleared(handles.combined, CLEAR_STENCIL);
    stencil.resolve = Some(handles.resolve);
    stencil.filter = stencil_filter;
    stencil.store = StoreAction::MultisampleResolve;
    RenderPass::clearing(vec![
        PassColor::cleared(handles.color, [0.0; 4]).stored(StoreAction::DontCare)
    ])
    .with_depth(depth)
    .with_stencil(stencil)
    .drawing(steps(handles.pipeline, handles.state))
}

/// The depth filter's enumerator name, which is also what the recording's site names hold.
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

/// The name a cell's sites take: both filters, the stencil one first, since it is the axis the
/// case is named for.
fn cell_name(
    stencil: MultisampleStencilResolveFilter,
    depth: MultisampleDepthResolveFilter,
) -> String {
    format!(
        "{}/{}",
        stencil_filter_name(stencil),
        depth_filter_name(depth)
    )
}

/// The six cells, in the order the case runs them.
fn cells() -> Vec<(
    MultisampleStencilResolveFilter,
    MultisampleDepthResolveFilter,
)> {
    let mut cells = Vec::new();
    for stencil in STENCIL_FILTERS {
        for depth in DEPTH_FILTERS {
            cells.push((stencil, depth));
        }
    }
    cells
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

/// The module's one case.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(ResolvedSample)]
}

/// Both stencil resolve filters by every depth resolve filter, on one combined attachment.
pub struct ResolvedSample;

impl Case for ResolvedSample {
    fn name(&self) -> &'static str {
        NAME
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
        // the shape of this case that provokes it, so a reader can tell a notice this case earns
        // from one copied in.
        vec![
            // The colour attachment nothing reads is stored `DontCare`, and a device whose
            // `DontCare` store is not Metal's announces that it took the near-equivalent.
            // Measured not to fire on MoltenVK, where the store is the device's own.
            crate::replay::notice_allowance(Degradation::DontCareStore.code()),
            // The stencil of the depth-picked sample, which no Vulkan resolve mode states: the
            // `DepthResolvedSample` cells after `Min` and `Max` reach it on **every** venue,
            // through the draws after the pass, and this notice firing is what says the exact
            // path ran rather than only that the run passed. Measured firing on MoltenVK.
            //
            // `one-aspect-resolve` is deliberately **not** named: it is what a pass whose two
            // aspects resolve into two textures earns, and this case's one attachment resolves
            // both into one, so a run that announced it would be a finding rather than a notice
            // this case justifies.
            crate::replay::notice_allowance(Degradation::ShaderResolve.code()),
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
            Input::bytes("render-pipeline", &pipeline_spec(&states).words()),
            Input::bytes("target:color", &color_target().words()),
            Input::bytes("target:combined", &combined_target().words()),
            Input::bytes("target:resolve", &combined_resolve_target().words()),
            Input::bytes("clear:stencil", &CLEAR_STENCIL.to_le_bytes()),
            Input::bytes("clear:depth", &CLEAR_DEPTH.to_le_bytes()),
        ];
        for (stencil, depth) in cells() {
            inputs.push(Input::bytes(
                format!("pass:{}", cell_name(stencil, depth)),
                &pass(stencil, depth, Handles::fingerprint()).words(),
            ));
        }
        // The two reads' own blit options, which is where this case's vocabulary differs from
        // every case recorded before it: `Plane::Whole` is Metal's own `MTLBlitOptionNone`, so a
        // read at the default writes nothing here and every recording taken before the plane
        // could be stated still reads as fresh. These two are not the default, so they are
        // fingerprinted, and a read that lost its aspect reads as a change to the case's inputs
        // rather than as a quiet pass against another aspect's bytes.
        let mut planes = Vec::new();
        for (what, plane) in [("stencil", Plane::Stencil), ("depth", Plane::Depth)] {
            planes.extend(what.as_bytes());
            planes.extend(plane.words());
        }
        inputs.push(Input::bytes("read:planes", &planes));
        // The patterns by their own numbers, so a pattern that moved reads as a change to the
        // case's inputs rather than a quiet pass against another grid's answer. A sample no draw
        // covered writes the clear's byte and the clear depth, neither of which any draw states.
        let mut rows = Vec::new();
        for pattern in PATTERNS {
            for carried in pattern.samples {
                let (stencil, z) = match carried {
                    Some(held) => (u32::from(held.0), held.1),
                    None => (CLEAR_STENCIL, CLEAR_DEPTH as f32),
                };
                rows.extend(stencil.to_le_bytes());
                rows.extend(z.to_le_bytes());
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
            side.render_pipeline(library, &pipeline_spec(&states)),
        )?;
        let color = made(sites, "target:color", side.target(&color_target()))?;
        let combined = made(sites, "target:combined", side.target(&combined_target()))?;
        let held_state = made(sites, "state", side.depth_stencil_state(&state()))?;

        let cells = cells();
        // One resolve texture and one read-back buffer an aspect a cell, so no cell's answer can
        // be another cell's leftovers.
        let mut resolves = Vec::new();
        let mut stencil_reads = Vec::new();
        let mut depth_reads = Vec::new();
        for (stencil, depth) in &cells {
            let name = cell_name(*stencil, *depth);
            resolves.push(made(
                sites,
                &format!("target:resolve:{name}"),
                side.target(&combined_resolve_target()),
            )?);
            stencil_reads.push(made(
                sites,
                &format!("buffer:stencil-read:{name}"),
                side.zeroed_buffer(STENCIL_PLANE as usize),
            )?);
            depth_reads.push(made(
                sites,
                &format!("buffer:depth-read:{name}"),
                side.zeroed_buffer(DEPTH_PLANE as usize),
            )?);
        }

        // One command buffer: a pass a cell, each followed by the two blits that read its resolve
        // texture's two aspects, so the whole grid is one submission on both sides.
        let mut work = Vec::new();
        let mut names = Vec::new();
        for (at, (stencil, depth)) in cells.iter().enumerate() {
            let name = cell_name(*stencil, *depth);
            work.push(RenderWork::Pass(pass(
                *stencil,
                *depth,
                Handles {
                    color,
                    combined,
                    resolve: resolves[at],
                    pipeline,
                    state: held_state,
                },
            )));
            names.push(format!("pass:{name}"));
            work.push(RenderWork::ReadTexture {
                texture: resolves[at],
                into: stencil_reads[at],
                offset: 0,
                row_pitch: SIDE,
                size: [SIDE; 2],
                plane: Plane::Stencil,
            });
            names.push(format!("read:stencil:{name}"));
            work.push(RenderWork::ReadTexture {
                texture: resolves[at],
                into: depth_reads[at],
                offset: 0,
                row_pitch: DEPTH_TEXEL * SIDE,
                size: [SIDE; 2],
                plane: Plane::Depth,
            });
            names.push(format!("read:depth:{name}"));
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
        for (at, (stencil, depth)) in cells.iter().enumerate() {
            let name = cell_name(*stencil, *depth);
            let bytes = call(
                sites,
                &format!("read-back/stencil/{name}"),
                side.read_buffer(stencil_reads[at], 0, STENCIL_PLANE as usize),
                |_| Value::Absent,
            )?;
            resolved = resolved.with(Node::leaf(
                format!("{}/{name}", Plane::Stencil.option_name()),
                Observed::bytes(bytes, stencil_layout()),
            ));
        }
        sites.push(resolved);

        let mut planes = Node::branch("resolved-depth", Vec::new());
        for (at, (stencil, depth)) in cells.iter().enumerate() {
            let name = cell_name(*stencil, *depth);
            let bytes = call(
                sites,
                &format!("read-back/depth/{name}"),
                side.read_buffer(depth_reads[at], 0, DEPTH_PLANE as usize),
                |_| Value::Absent,
            )?;
            planes = planes.with(Node::leaf(
                format!("{}/{name}", Plane::Depth.option_name()),
                Observed::bytes(bytes, depth_layout()),
            ));
        }
        sites.push(planes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case states **every** `MTLMultisampleStencilResolveFilter` by name, which is what
    /// licenses the crossing row in `scripts/conformance-coverage/crossings.tsv`: a module that
    /// cannot vary an axis cannot cross it.
    #[test]
    fn the_swept_stencil_filters_are_every_value_metal_defines() {
        let mut swept: Vec<u64> = STENCIL_FILTERS.iter().map(|f| f.raw()).collect();
        swept.sort_unstable();
        assert_eq!(swept, vec![0, 1], "{STENCIL_FILTERS:?}");
    }

    /// And every `MTLMultisampleDepthResolveFilter`, the cross's other axis.
    #[test]
    fn the_swept_depth_filters_are_every_value_metal_defines() {
        let mut swept: Vec<u64> = DEPTH_FILTERS.iter().map(|f| f.raw()).collect();
        swept.sort_unstable();
        assert_eq!(swept, vec![0, 1, 2], "{DEPTH_FILTERS:?}");
    }

    /// The two aspects hold **different** values, which is what makes the read-back honest: a
    /// side whose stencil read returned the depth plane would answer a zero at every texel, and
    /// no stencil the grid states is zero.
    #[test]
    fn no_stencil_byte_is_a_depth_byte() {
        let mut depths = vec![(CLEAR_DEPTH as f32).to_le_bytes()];
        let mut stencils = vec![CLEAR_STENCIL as u8];
        for pattern in PATTERNS {
            for carried in pattern.samples.iter().flatten() {
                stencils.push(carried.0);
                depths.push(carried.1.to_le_bytes());
            }
        }
        for bytes in &depths {
            assert_eq!(bytes[0], 0, "a depth whose low byte is not zero: {bytes:?}");
        }
        for stencil in &stencils {
            assert_ne!(*stencil, 0, "a stencil byte of zero");
        }
    }

    /// Where two samples of a pattern hold one depth they hold one **stencil** too, so whichever
    /// of them Metal's tie-break takes the resolved stencil is the same byte and no cell's answer
    /// rests on the tie-break.
    ///
    /// This is the grid's own guard rather than a remark: three of the seven patterns do hold ties
    /// (the uncovered samples all sit at the clear depth), and the case is only honest about
    /// `DepthResolvedSample` because every one of those ties is between samples holding the clear
    /// stencil.
    #[test]
    fn a_tie_among_equal_depths_cannot_move_a_cell() {
        for pattern in PATTERNS {
            let held: Vec<(u32, u8)> = pattern
                .samples
                .iter()
                .map(|carried| match carried {
                    Some((stencil, z)) => (z.to_bits(), *stencil),
                    None => ((CLEAR_DEPTH as f32).to_bits(), CLEAR_STENCIL as u8),
                })
                .collect();
            for (depth, stencil) in &held {
                for (other_depth, other_stencil) in &held {
                    if depth == other_depth {
                        assert_eq!(
                            stencil, other_stencil,
                            "{}: two samples at one depth hold two stencils",
                            pattern.name
                        );
                    }
                }
            }
        }
    }

    /// Every **covered** sample of a pattern holds a depth of its own, so the `Min` and the `Max`
    /// pick of a pattern whose samples are all covered is one sample and not a set.
    #[test]
    fn no_pattern_covers_two_samples_at_one_depth() {
        for pattern in PATTERNS {
            let mut drawn: Vec<u32> = pattern
                .samples
                .iter()
                .flatten()
                .map(|(_, z)| z.to_bits())
                .collect();
            let held = drawn.len();
            drawn.sort_unstable();
            drawn.dedup();
            // `every-sample-alike` is one depth by construction, and the module states it as one
            // of the two the other five are read against.
            let alike = pattern.name == "every-sample-alike";
            assert_eq!(drawn.len() == held, !alike, "{}", pattern.name);
        }
    }

    /// Every cell of the grid lies inside the attachments.
    #[test]
    fn every_cell_lies_in_the_attachment() {
        for at in 0..PATTERNS.len() {
            let held = cell(at);
            assert!(held.x + held.width <= u64::from(SIDE), "cell {at}");
            assert!(held.y + held.height <= u64::from(SIDE), "cell {at}");
        }
    }

    /// The six cells are the whole cross and name no pair twice.
    #[test]
    fn the_cells_are_every_pair_of_the_two_axes() {
        let held = cells();
        assert_eq!(held.len(), STENCIL_FILTERS.len() * DEPTH_FILTERS.len());
        let mut names: Vec<String> = held
            .iter()
            .map(|(stencil, depth)| cell_name(*stencil, *depth))
            .collect();
        names.sort();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two cells share a name");
    }
}
