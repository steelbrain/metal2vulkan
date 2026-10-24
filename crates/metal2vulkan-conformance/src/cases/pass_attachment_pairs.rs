//! **Per-attachment pass state across two colour attachments**: a load action, a store action, a
//! format and a resolve texture stated *differently* at colour 0 and colour 1 in one pass, and read
//! back out of both.
//!
//! The suite had 36 (load, store, options) triples and both resolve filters on the **descriptor**
//! ([`crate::pass_descriptors`]'s `pass-attachment-actions` and `pass-attachment-filters`) and not
//! one case that rendered through two colour attachments whose state disagreed. So nothing
//! anywhere held a side to the thing a two-attachment pass can get wrong: **mixing the two
//! attachments up.** A side that applied colour 0's load action to colour 1, or wrote colour 0's
//! fragment output into attachment 1, or resolved colour 1 into colour 0's resolve texture, passes
//! every per-value sweep of a single attachment there has ever been.
//!
//! | case | what it crosses |
//! |---|---|
//! | `pass-two-attachments-loads` | three per-attachment format pairs by four per-attachment load-action pairs, plus the two cells where the pipeline writes attachment 1 alone |
//! | `pass-two-attachments-resolves` | which of the two attachments resolves — colour 0 alone, colour 1 alone, both, and both at two different formats |
//!
//! # The blocker was a fixture, and this is it
//!
//! `RenderPipelineSpec::colors` has always been a slice of `Option<ColorState>`, so the vocabulary
//! could state two attachments apart. What was missing was a fixture that writes two `float4`
//! attachments from **per-pixel buffer data** without framebuffer fetch:
//! `engine/color_groups.metal`'s `cg_five` writes five attachments from an interpolated varying,
//! whose last bits are a rasterizer's answer rather than Metal's, so a cross over it would have
//! needed a tolerance — which defeats a cross whose whole point is an exact texel; and
//! `engine/fbl_fetch_blend16_safe.metal` writes a constant to colour 1 and *fetches* colour 0.
//! The owned fixture [`FIXTURE`] is the shape `engine/xrb_dual.metal` already had for dual-source
//! blending, moved across two attachment **indices**: `rt6_pair_two` returns colour 0 from
//! fragment buffer 0 and colour 1 from fragment buffer 2, each at `[[position]]`'s own texel, and
//! `rt6_pair_second` returns colour 1 alone.
//!
//! Every value either buffer holds is an exact binary fraction and both attachment formats are
//! float formats, so no cell's answer turns on a conversion and **nothing here is loosened**.
//!
//! # What each axis of the loads case is for
//!
//! The pass draws into a rectangle that covers the **top half** of the target
//! ([`DRAWN`]), so every cell's two planes hold two regions a reader can tell apart: what the
//! fragment wrote, and what the attachment's own load action left where no fragment ran.
//!
//! - **The load actions.** `Load` and `Clear` at each attachment, all four pairs. An earlier pass
//!   of the same command buffer clears both attachments to **their own** prior values and stores,
//!   so a `Load` shows a value no clear of this pass states and the two attachments' prior values
//!   differ from each other. The two off-diagonal pairs are the cells only the pair can see: a side
//!   that applied attachment 0's load action to attachment 1 answers the diagonal correctly and
//!   both off-diagonal cells wrong. `DontCare` is **not** swept: what it leaves is undefined on
//!   both sides, so a cell over it would compare two devices' choices rather than Metal's rule.
//! - **The formats.** `RGBA32Float` at both, then `RGBA32Float` beside `R32Float` and the swap. A
//!   one-channel attachment beside a four-channel one is the sharpest format pair available without
//!   a conversion: the two planes are different sizes *and* attachment 1 keeps only the red
//!   channel, so a side that took one attachment's format for the other cannot answer the cell at
//!   all. Crossed with the load actions rather than swept beside them, because the question is
//!   whether the **pair** of formats is honoured per index.
//! - **The pipeline's own attachment set.** Two cells state a pipeline that writes colour 1 alone
//!   (`rt6_pair_second`, with `None` at colour 0 of the pipeline's attachments) while the pass
//!   attaches both. Metal's rule is that attachment 0 still takes its own load and store action,
//!   so the plane holds its load everywhere — the one shape that tells a load action reaching an
//!   attachment from a fragment writing it.
//!
//! # What only the resolves case can see
//!
//! `MultisampleResolve` at one attachment beside `DontCare` at the other is the cross the depth
//! grid could not make, because a depth-only pass has one attachment to state. Three cells put the
//! resolving store at colour 0 alone, at colour 1 alone and at both, and a fourth puts it at both
//! with the two attachments at **different formats**. A side that took attachment 0's resolve
//! texture for attachment 1 writes colour 0's value into colour 1's texture in the two cells where
//! attachment 1 resolves, and is right in the cell where colour 0 resolves alone.
//!
//! Every sample of every covered pixel carries the same value — the draw is one triangle over the
//! whole rectangle, so no pixel is partly covered — so each resolve is that value exactly and no
//! cell's answer is an average a device may round. An attachment stored `DontCare` carries **no**
//! `resolveTexture` and is **not read**: `DontCare` beside a resolve texture is a shape Metal's API
//! validation stops and record mode runs a validated leg, so this family does not find that out by
//! trying.
//!
//! # Why this family is safe
//!
//! One command buffer a case: one priming pass and one pass a cell over an 8 by 8 target, with one
//! blit per attachment read. One draw of three vertices a pass, no loop in either stage, every
//! scissor rectangle inside the attachments, every vertex inside the clip volume, and no sample
//! count past what the venue answers. A venue that does not render the resolves case's sample count
//! refuses the target and the suite counts the case
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving it uncompared.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, ScissorBox,
    StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{Degradation, LoadAction, StoreAction};

/// The owned fixture: the fullscreen triangle, the fragment that writes both colour attachments
/// from two buffers, and the one that writes colour 1 alone.
const FIXTURE: &str = "rt6_pair";

/// `rt6_pair_vertex`: `float4(4 * a - 1, 0.5, 1)` at each of three vertex ids.
const VERTEX: &str = "rt6_pair_vertex";

/// `rt6_pair_two`: colour 0 from fragment buffer 0 and colour 1 from fragment buffer 2, each at
/// the fragment's own texel.
const BOTH: &str = "rt6_pair_two";

/// `rt6_pair_second`: colour 1 alone, from fragment buffer 2.
const SECOND: &str = "rt6_pair_second";

/// The side of every attachment, in texels.
const SIDE: u32 = 8;

/// The rectangle the draw is confined to: the target's **top half**, so every plane holds what the
/// fragment wrote beside what the attachment's load action left.
const DRAWN: ScissorBox = ScissorBox::new(0, 0, SIDE as u64, (SIDE / 2) as u64);

/// The samples the resolves case renders at. Four, the count every venue we run answers.
const SAMPLES: u32 = 4;

/// A four-channel 32-bit float attachment: every value either buffer holds reaches its texel
/// unchanged, so no cell's answer turns on a conversion.
const WIDE: PixelFormat = PixelFormat::RGBA32Float;

/// A one-channel 32-bit float attachment: the same exactness, and **one** channel, so an
/// attachment of this format beside one of [`WIDE`] cannot answer the other's cell.
const NARROW: PixelFormat = PixelFormat::R32Float;

/// Bytes one texel of a format takes.
const fn texel_bytes(format: PixelFormat) -> u32 {
    match format {
        NARROW => 4,
        _ => 16,
    }
}

/// `float`s one texel of a format holds.
const fn texel_floats(format: PixelFormat) -> u32 {
    texel_bytes(format) / 4
}

/// Bytes a whole level 0 of `format` takes.
const fn plane_bytes(format: PixelFormat) -> u32 {
    texel_bytes(format) * SIDE * SIDE
}

/// How many runs record mode takes a case of this family over: one command buffer, one queue and
/// nothing a device may reorder, so the runs are expected to fold into one outcome and a case whose
/// answer moved between runs is the finding rather than one run being read as the answer.
const RUNS: u32 = 8;

/// The two colour attachments, by index, wherever a pair is stated.
const INDICES: [usize; 2] = [0, 1];

/// The clear the priming pass gives attachment 0 and attachment 1: **different** values, each an
/// exact binary fraction, so an attachment that loaded the other's prior content is told apart from
/// one that loaded its own.
///
/// Every value of this family is a multiple of `2^-12`, exact in a `float` and in a 32-bit float
/// texel alike, and the clears and the fragment's values sit in **disjoint bands**: a clear is
/// below `1/128` and a drawn value is in `[1/4, 3/8]`. That is not decoration — a plane is read as
/// four regions and a cell could not name which of them put a texel there if two shared a value.
/// The module's own test holds the bands apart, and it caught the first table written here, where
/// one prior value and one drawn value collided at `0.0625`.
const PRIOR: [[f64; 4]; 2] = [
    [1.0 / 4096.0, 2.0 / 4096.0, 3.0 / 4096.0, 4.0 / 4096.0],
    [5.0 / 4096.0, 6.0 / 4096.0, 7.0 / 4096.0, 8.0 / 4096.0],
];

/// The clear a test pass's `Clear` load gives attachment 0 and attachment 1: again different, and
/// equal to neither attachment's prior value nor any value a fragment writes, so every region of
/// every plane names which of the four things put it there.
const CLEARS: [[f64; 4]; 2] = [
    [9.0 / 4096.0, 10.0 / 4096.0, 11.0 / 4096.0, 12.0 / 4096.0],
    [13.0 / 4096.0, 14.0 / 4096.0, 15.0 / 4096.0, 16.0 / 4096.0],
];

/// The `float4` a fragment writes into attachment `which` at texel `at`, as four exact binary
/// fractions no clear of this family states.
fn drawn(which: usize, at: u32) -> [f32; 4] {
    // Eight codes a texel, four an attachment, from 1024: every channel of every texel of either
    // attachment is its own multiple of `2^-12`, the whole band is `[1/4, 3/8]` at 64 texels, and
    // nothing in it is a clear of this family's.
    let code = 1024 + at * 8 + which as u32 * 4;
    [0, 1, 2, 3].map(|channel| (code + channel) as f32 / 4096.0)
}

/// The bytes of the buffer a fragment reads attachment `which`'s values out of: one `float4` a
/// texel of the whole target, in row-major order, which is how the fixture indexes it
/// (`y * width + x`).
fn drawn_bytes(which: usize) -> Vec<u8> {
    (0..SIDE * SIDE)
        .flat_map(|at| drawn(which, at).into_iter().flat_map(f32::to_le_bytes))
        .collect()
}

/// The bytes of the `uint` the fixture reads its row width from, at fragment buffer index 1.
fn width_bytes() -> Vec<u8> {
    SIDE.to_le_bytes().to_vec()
}

/// The layout a plane of `format` is read under: one `float` an element, so a diff names the
/// channel of the texel that moved.
fn layout(format: PixelFormat) -> ByteLayout {
    let row_pitch = texel_bytes(format) * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: texel_floats(format) * SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// A target of `format` at `samples` samples: of plain render-target usage single-sampled, and a
/// shader may read a multisampled one, which is what a resolve drawn after the pass needs.
fn target(format: PixelFormat, samples: u32) -> TargetSpec {
    match samples {
        1 => TargetSpec::plain(format, [SIDE; 2]),
        _ => TargetSpec::read(format, [SIDE; 2]).multisampled(samples),
    }
}

/// Which fragment function a cell's pipeline takes, and so which attachments its fragment writes.
///
/// **Both shapes state both attachments' formats on the pipeline.** A pipeline that left colour 0
/// at `MTLPixelFormatInvalid` while the pass attached a texture there was tried and is **not**
/// authored: Metal's API validation stops it ("For color attachment 0, the render pipeline's
/// pixelFormat (MTLPixelFormatInvalid) does not match the framebuffer's pixelFormat
/// (MTLPixelFormatRGBA32Float)"), and record mode runs a validated leg. Plain Metal does run it,
/// and what it leaves in attachment 0 is **attachment 1's own fragment value** — measured on an
/// M3 Max, macOS 27.0.1 — but a shape Metal's validation calls a programming error is not a rule
/// this suite holds anybody to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writes {
    /// `rt6_pair_two`: the fragment returns both attachments.
    Both,
    /// `rt6_pair_second`: the fragment returns colour 1 alone. The pipeline still names colour 0's
    /// format, as Metal requires of a pipeline whose pass attaches one.
    SecondOnly,
}

impl Writes {
    /// The fragment function's entry name.
    fn entry(self) -> &'static str {
        match self {
            Writes::Both => BOTH,
            Writes::SecondOnly => SECOND,
        }
    }
}

/// The pipeline's colour attachments for `formats`: both stated, whichever fragment function the
/// cell takes.
///
/// Which attachments the **fragment** writes is the function's own business and is not said here:
/// Metal requires a pipeline to name the format of every attachment its pass attaches, whether a
/// fragment writes it or not ([`Writes`]).
fn color_states(formats: [PixelFormat; 2]) -> [Option<ColorState>; 2] {
    [
        Some(ColorState::new(formats[0])),
        Some(ColorState::new(formats[1])),
    ]
}

/// The pipeline over `states` at `samples` samples, of `writes`' fragment function.
fn pipeline_spec<'a>(
    states: &'a [Option<ColorState>],
    writes: Writes,
    samples: u32,
) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::of(
        StageFunction::named(VERTEX),
        Some(StageFunction::named(writes.entry())),
        states,
    )
    .multisampled(samples)
}

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// The steps a test pass encodes: the pipeline, the two value buffers and the row width, the
/// rectangle the draw is confined to, and one triangle.
///
/// A pipeline that writes colour 1 alone binds attachment 0's buffer all the same, so the one
/// thing that differs between the two shapes is which attachments the fragment returns.
fn steps(pipeline: Handle, zeros: Handle, ones: Handle, width: Handle) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::FragmentBuffer {
            buffer: zeros,
            offset: 0,
            index: 0,
        },
        RenderStep::FragmentBuffer {
            buffer: width,
            offset: 0,
            index: 1,
        },
        RenderStep::FragmentBuffer {
            buffer: ones,
            offset: 0,
            index: 2,
        },
        RenderStep::Scissor(DRAWN),
        TRIANGLE,
    ]
}

/// One cell of the loads case: the two attachments' formats, their two load actions, and which
/// attachments the pipeline writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoadCell {
    formats: [PixelFormat; 2],
    loads: [LoadAction; 2],
    writes: Writes,
}

/// One cell of the resolves case: the two attachments' formats and their two store actions.
///
/// An attachment whose store does not resolve carries **no** resolve texture and is not read, for
/// the reason the module states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolveCell {
    name: &'static str,
    formats: [PixelFormat; 2],
    stores: [StoreAction; 2],
}

/// The load actions a cell may state: the two whose answer is defined on both sides.
const LOADS: [LoadAction; 2] = [LoadAction::Load, LoadAction::Clear];

/// The format pairs the loads case crosses the load actions with.
const FORMAT_PAIRS: [[PixelFormat; 2]; 3] = [[WIDE, WIDE], [WIDE, NARROW], [NARROW, WIDE]];

/// The resolves case's four cells.
const RESOLVE_CELLS: [ResolveCell; 4] = [
    ResolveCell {
        name: "resolve-at-zero-only",
        formats: [WIDE, WIDE],
        stores: [StoreAction::MultisampleResolve, StoreAction::DontCare],
    },
    ResolveCell {
        name: "resolve-at-one-only",
        formats: [WIDE, WIDE],
        stores: [StoreAction::DontCare, StoreAction::MultisampleResolve],
    },
    ResolveCell {
        name: "resolve-at-both",
        formats: [WIDE, WIDE],
        stores: [
            StoreAction::MultisampleResolve,
            StoreAction::MultisampleResolve,
        ],
    },
    ResolveCell {
        name: "resolve-at-both-mixed-formats",
        formats: [WIDE, NARROW],
        stores: [
            StoreAction::MultisampleResolve,
            StoreAction::MultisampleResolve,
        ],
    },
];

/// The load action's enumerator name, which is also what a recording's site names hold.
fn load_name(load: LoadAction) -> &'static str {
    match load {
        LoadAction::DontCare => "MTLLoadActionDontCare",
        LoadAction::Load => "MTLLoadActionLoad",
        LoadAction::Clear => "MTLLoadActionClear",
        _ => "MTLLoadActionUnknown",
    }
}

/// The store action's enumerator name.
fn store_name(store: StoreAction) -> &'static str {
    match store {
        StoreAction::DontCare => "MTLStoreActionDontCare",
        StoreAction::Store => "MTLStoreActionStore",
        StoreAction::MultisampleResolve => "MTLStoreActionMultisampleResolve",
        StoreAction::StoreAndMultisampleResolve => "MTLStoreActionStoreAndMultisampleResolve",
        _ => "MTLStoreActionUnknown",
    }
}

/// The format's own short name, for a site name a reader can follow.
fn format_name(format: PixelFormat) -> &'static str {
    match format {
        NARROW => "MTLPixelFormatR32Float",
        _ => "MTLPixelFormatRGBA32Float",
    }
}

impl ResolveCell {
    /// The name the cell's sites take: the shape in words, then the two store actions by their
    /// own enumerator names, so a reader of a recording sees which `MTLStoreAction` each
    /// attachment was stored under rather than having to find the module.
    fn site(&self) -> String {
        format!(
            "{}/{}+{}",
            self.name,
            store_name(self.stores[0]),
            store_name(self.stores[1])
        )
    }
}

impl LoadCell {
    /// The name the cell's sites take: the two formats, the two load actions, and which
    /// attachments the pipeline writes where that is not both.
    fn site(&self) -> String {
        let formats = format!(
            "{}+{}",
            format_name(self.formats[0]),
            format_name(self.formats[1])
        );
        let loads = format!("{}+{}", load_name(self.loads[0]), load_name(self.loads[1]));
        match self.writes {
            Writes::Both => format!("{formats}/{loads}"),
            Writes::SecondOnly => format!("{formats}/{loads}/second-only"),
        }
    }
}

/// The `Store` the loads case gives both attachments, so every cell reads both planes.
const KEPT: StoreAction = StoreAction::Store;

/// One attachment of a test pass: `which`'s texture under `load`, cleared to its own colour, and
/// stored under `store`, with `resolve` where the store resolves.
fn attachment(
    which: usize,
    texture: Handle,
    load: LoadAction,
    store: StoreAction,
    resolve: Option<Handle>,
) -> PassColor {
    let mut color = PassColor::cleared(texture, CLEARS[which]);
    color.load = load;
    color.store = store;
    color.resolve = resolve;
    color
}

/// The priming pass: both attachments cleared to their own prior values and stored, so a `Load` in
/// a test pass loads a value that pass states nowhere.
fn priming(textures: [Handle; 2]) -> RenderPass {
    RenderPass::clearing(
        INDICES
            .map(|which| PassColor::cleared(textures[which], PRIOR[which]).stored(KEPT))
            .to_vec(),
    )
}

/// The shader reference both texts of the fixture give, for each of the two fragment functions a
/// case may use.
fn shader_refs(entries: &[&str]) -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(FIXTURE);
    let mut refs = Vec::new();
    for extension in [RUNTIME_TEXT, METAL_TEXT] {
        let digest = crate::sha256_hex(held.text(extension)?.as_bytes());
        for entry in entries {
            refs.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: format!("{VERTEX}+{entry}"),
                source_sha256: digest.clone(),
            });
        }
    }
    Ok(refs)
}

/// Which of the family's two crosses a case states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cross {
    /// Per-attachment formats by per-attachment load actions, single-sampled.
    Loads,
    /// Which of the two attachments resolves, at four samples.
    Resolves,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(AttachmentPairs {
            cross: Cross::Loads,
        }),
        Box::new(AttachmentPairs {
            cross: Cross::Resolves,
        }),
    ]
}

/// One of the family's two crosses over two colour attachments.
pub struct AttachmentPairs {
    cross: Cross,
}

impl AttachmentPairs {
    /// The name the case's cross gives it.
    fn case_name(&self) -> &'static str {
        match self.cross {
            Cross::Loads => "pass-two-attachments-loads",
            Cross::Resolves => "pass-two-attachments-resolves",
        }
    }

    /// The samples the case renders at.
    fn samples(&self) -> u32 {
        match self.cross {
            Cross::Loads => 1,
            Cross::Resolves => SAMPLES,
        }
    }

    /// The loads case's cells: every format pair by every load-action pair, then the two cells
    /// whose pipeline writes colour 1 alone.
    fn load_cells() -> Vec<LoadCell> {
        let mut cells = Vec::new();
        for formats in FORMAT_PAIRS {
            for zero in LOADS {
                for one in LOADS {
                    cells.push(LoadCell {
                        formats,
                        loads: [zero, one],
                        writes: Writes::Both,
                    });
                }
            }
        }
        // The pipeline that writes attachment 1 alone, under both load actions at attachment 0:
        // what colour 0's own load action leaves where no fragment wrote it.
        for zero in LOADS {
            cells.push(LoadCell {
                formats: [WIDE, WIDE],
                loads: [zero, LoadAction::Load],
                writes: Writes::SecondOnly,
            });
        }
        cells
    }

    /// Every cell of the case's cross, in the order it runs them: the one list `inputs`, `run`
    /// and the tests all read, so no two of them can state a different grid.
    fn cells(&self) -> Vec<Cell> {
        match self.cross {
            Cross::Loads => Self::load_cells()
                .iter()
                .map(|cell| Cell {
                    site: cell.site(),
                    formats: cell.formats,
                    loads: cell.loads,
                    stores: [KEPT; 2],
                    writes: cell.writes,
                    // Every attachment is stored, so every cell reads both planes.
                    read: INDICES.to_vec(),
                })
                .collect(),
            Cross::Resolves => RESOLVE_CELLS
                .iter()
                .map(|cell| Cell {
                    site: cell.site(),
                    formats: cell.formats,
                    loads: [LoadAction::Clear; 2],
                    stores: cell.stores,
                    writes: Writes::Both,
                    // Only an attachment whose store resolves has a texture to read.
                    read: INDICES
                        .into_iter()
                        .filter(|which| resolves(cell.stores[*which]))
                        .collect(),
                })
                .collect(),
        }
    }
}

/// One cell's whole statement: its site name, the two attachments' formats, load actions and store
/// actions, which attachments the pipeline writes, and which planes the cell reads back.
struct Cell {
    site: String,
    formats: [PixelFormat; 2],
    loads: [LoadAction; 2],
    stores: [StoreAction; 2],
    writes: Writes,
    read: Vec<usize>,
}

/// Whether a store action resolves, which is what decides whether an attachment carries a resolve
/// texture at all.
fn resolves(store: StoreAction) -> bool {
    matches!(
        store,
        StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
    )
}

/// The handles a test pass names.
#[derive(Clone, Copy)]
struct Handles {
    textures: [Handle; 2],
    /// Each attachment's resolve texture, where its store resolves.
    resolves: [Option<Handle>; 2],
    pipeline: Handle,
    zeros: Handle,
    ones: Handle,
    width: Handle,
}

impl Handles {
    /// The set a fingerprint is taken over: every handle zeroed, a resolve texture present exactly
    /// where the cell states one. A side's handle numbers are its own and none reaches a
    /// fingerprint.
    const fn fingerprint(resolves: [bool; 2]) -> Self {
        let zero = Handle(0);
        Self {
            textures: [zero; 2],
            resolves: [
                match resolves[0] {
                    true => Some(zero),
                    false => None,
                },
                match resolves[1] {
                    true => Some(zero),
                    false => None,
                },
            ],
            pipeline: zero,
            zeros: zero,
            ones: zero,
            width: zero,
        }
    }
}

/// One cell's test pass: the two attachments under their own load and store actions, and one draw
/// confined to [`DRAWN`].
fn pass(loads: [LoadAction; 2], stores: [StoreAction; 2], handles: Handles) -> RenderPass {
    RenderPass::clearing(
        INDICES
            .map(|which| {
                attachment(
                    which,
                    handles.textures[which],
                    loads[which],
                    stores[which],
                    handles.resolves[which],
                )
            })
            .to_vec(),
    )
    .drawing(steps(
        handles.pipeline,
        handles.zeros,
        handles.ones,
        handles.width,
    ))
}

impl Case for AttachmentPairs {
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
        // **No value is loosened**: every texel Metal leaves is one we must leave, and every
        // declared code is a notice. The resolves case stores an attachment `DontCare` in two of
        // its four cells, and a device whose `DontCare` store is not Metal's announces that it took
        // the near-equivalent; the loads case stores every attachment and declares nothing.
        match self.cross {
            Cross::Loads => Vec::new(),
            Cross::Resolves => vec![
                crate::replay::notice_allowance(Degradation::DontCareStore.code()),
                // A device whose own `AVERAGE` resolve is not measured to be Metal's draws the
                // colour resolve after the pass by Metal's rules instead. Every sample of every
                // covered pixel carries one value here, so either path reaches the same texel.
                crate::replay::notice_allowance(Degradation::ShaderResolve.code()),
            ],
        }
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        match self.cross {
            Cross::Loads => shader_refs(&[BOTH, SECOND]),
            Cross::Resolves => shader_refs(&[BOTH]),
        }
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("drawn:color0", &drawn_bytes(0)),
            Input::bytes("drawn:color1", &drawn_bytes(1)),
            Input::bytes("width", &width_bytes()),
            Input::bytes("drawn-rect", &DRAWN.words()),
        ];
        // Both clear tables, by their own bits, so a clear that moves by one `f64` step is a change
        // to the case's inputs rather than a quiet pass against another grid's answer.
        for (what, table) in [("clear:prior", PRIOR), ("clear:pass", CLEARS)] {
            let mut bytes = Vec::new();
            for colour in table {
                for channel in colour {
                    bytes.extend(channel.to_bits().to_le_bytes());
                }
            }
            inputs.push(Input::bytes(what, &bytes));
        }
        let samples = self.samples();
        for cell in self.cells() {
            let states = color_states(cell.formats);
            inputs.push(Input::bytes(
                format!("pipeline:{}", cell.site),
                &pipeline_spec(&states, cell.writes, samples).words(),
            ));
            for which in INDICES {
                inputs.push(Input::bytes(
                    format!("target:{}:{which}", cell.site),
                    &target(cell.formats[which], samples).words(),
                ));
                if resolves(cell.stores[which]) {
                    inputs.push(Input::bytes(
                        format!("target:{}:{which}:resolve", cell.site),
                        &target(cell.formats[which], 1).words(),
                    ));
                }
            }
            let held = INDICES.map(|which| resolves(cell.stores[which]));
            inputs.push(Input::bytes(
                format!("pass:{}", cell.site),
                &pass(cell.loads, cell.stores, Handles::fingerprint(held)).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let samples = self.samples();
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let zeros = made(sites, "buffer:color0", side.buffer(&drawn_bytes(0)))?;
        let ones = made(sites, "buffer:color1", side.buffer(&drawn_bytes(1)))?;
        let width = made(sites, "buffer:width", side.buffer(&width_bytes()))?;

        let cells = self.cells();
        let mut work = Vec::new();
        let mut names = Vec::new();
        // What each cell reads back: the attachment index, its format, and the buffer the blit
        // puts it in.
        let mut reads: Vec<Vec<(usize, PixelFormat, Handle)>> = Vec::new();
        for cell in &cells {
            let states = color_states(cell.formats);
            let pipeline = made(
                sites,
                &format!("pipeline:{}", cell.site),
                side.render_pipeline(library, &pipeline_spec(&states, cell.writes, samples)),
            )?;
            let mut textures = [Handle(0); 2];
            let mut resolve_textures = [None; 2];
            let mut cell_reads = Vec::new();
            for which in INDICES {
                let format = cell.formats[which];
                textures[which] = made(
                    sites,
                    &format!("target:{}:{which}", cell.site),
                    side.target(&target(format, samples)),
                )?;
                // A resolve texture only where the store resolves: `DontCare` beside one is a
                // shape Metal's API validation stops.
                if resolves(cell.stores[which]) {
                    resolve_textures[which] = Some(made(
                        sites,
                        &format!("target:{}:{which}:resolve", cell.site),
                        side.target(&target(format, 1)),
                    )?);
                }
                if cell.read.contains(&which) {
                    cell_reads.push((
                        which,
                        format,
                        made(
                            sites,
                            &format!("buffer:read:{}:{which}", cell.site),
                            side.zeroed_buffer(plane_bytes(format) as usize),
                        )?,
                    ));
                }
            }
            // The loads case primes its attachments in the same command buffer: a pass that clears
            // both to their own prior values and stores, so a `Load` loads a value no test pass
            // states.
            if self.cross == Cross::Loads {
                work.push(RenderWork::Pass(priming(textures)));
                names.push(format!("prime:{}", cell.site));
            }
            work.push(RenderWork::Pass(pass(
                cell.loads,
                cell.stores,
                Handles {
                    textures,
                    resolves: resolve_textures,
                    pipeline,
                    zeros,
                    ones,
                    width,
                },
            )));
            names.push(format!("pass:{}", cell.site));
            for (which, format, into) in &cell_reads {
                // A multisampled attachment is read through its resolve texture; a single-sampled
                // one is read straight.
                let from = match resolve_textures[*which] {
                    Some(resolve) => resolve,
                    None => textures[*which],
                };
                work.push(RenderWork::ReadTexture {
                    texture: from,
                    into: *into,
                    offset: 0,
                    row_pitch: texel_bytes(*format) * SIDE,
                    size: [SIDE; 2],
                });
                names.push(format!("read:{}:{which}", cell.site));
            }
            reads.push(cell_reads);
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

        let mut planes = Node::branch("planes", Vec::new());
        for (cell, cell_reads) in cells.iter().zip(&reads) {
            let mut branch = Node::branch(cell.site.clone(), Vec::new());
            for (which, format, from) in cell_reads {
                let bytes = call(
                    sites,
                    &format!("read-back/{}/{which}", cell.site),
                    side.read_buffer(*from, 0, plane_bytes(*format) as usize),
                    |_| Value::Absent,
                )?;
                branch = branch.with(Node::leaf(
                    format!("color{which}"),
                    Observed::bytes(bytes, layout(*format)),
                ));
            }
            planes = planes.with(branch);
        }
        sites.push(planes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// What attachment `which` of `cell` must hold at texel `(x, y)`, channel by channel: the
    /// fragment's value where the draw reached it and the pipeline wrote that attachment, else
    /// what the attachment's own load action left.
    ///
    /// Metal's rule applied here rather than read back from a recording, which is what lets the
    /// recording be held to a reason. Only the channels the format holds are compared, so a
    /// one-channel attachment is read as its red alone.
    fn expected(cell: &LoadCell, which: usize, x: u32, y: u32) -> Vec<f32> {
        let channels = texel_floats(cell.formats[which]) as usize;
        // Attachment 0 is written only where the cell's fragment function returns it; attachment 1
        // is written by both of them.
        let written = cell.writes == Writes::Both || which == 1;
        let inside = x < DRAWN.width as u32 && y < DRAWN.height as u32;
        let held: Vec<f32> = match (inside && written, cell.loads[which]) {
            (true, _) => drawn(which, y * SIDE + x).to_vec(),
            (false, LoadAction::Load) => PRIOR[which].iter().map(|c| *c as f32).collect(),
            (false, _) => CLEARS[which].iter().map(|c| *c as f32).collect(),
        };
        held[..channels].to_vec()
    }

    /// **Every stated value is an exact binary fraction inside `[0, 1)`, and no two of the four
    /// roles a texel's value can come from ever collide.**
    ///
    /// That is the property the whole family rests on: a plane is read as four regions — what the
    /// draw wrote into attachment 0, what it wrote into attachment 1, what each attachment's prior
    /// value was, and what each attachment's clear was — and a cell could not name which of them
    /// put a texel there if two of them shared a value.
    #[test]
    fn every_stated_value_is_exact_and_no_two_roles_collide() {
        let exact = |value: f64| {
            assert!((0.0..1.0).contains(&value), "{value} is not inside [0, 1)");
            // A multiple of 2^-12 is exact in a `float` and in a 32-bit float texel alike, so the
            // round trip through the attachment changes nothing.
            assert_eq!(
                value * 4096.0,
                (value * 4096.0).round(),
                "{value} is not exact"
            );
        };
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for table in [PRIOR, CLEARS] {
            for colour in table {
                for channel in colour {
                    exact(channel);
                    // The clears' band: below `1/128`, which no drawn value reaches.
                    assert!(
                        channel < 1.0 / 128.0,
                        "{channel} is not in the clears' band"
                    );
                    assert!(
                        seen.insert((channel as f32).to_bits()),
                        "{channel} is stated by two different roles"
                    );
                }
            }
        }
        for which in INDICES {
            for at in 0..SIDE * SIDE {
                for channel in drawn(which, at) {
                    exact(f64::from(channel));
                    // The drawn band: `[1/4, 3/8]`, which no clear reaches.
                    assert!(
                        (0.25..=0.375).contains(&channel),
                        "the fragment writes {channel}, which is not in the drawn band"
                    );
                    assert!(
                        seen.insert(channel.to_bits()),
                        "the fragment writes {channel} into attachment {which} at texel {at}, \
                         which another role also states"
                    );
                }
            }
        }
        // Every texel of every attachment is a distinct `float4`, so a plane that holds one
        // attachment's value at another's texel is caught texel by texel rather than only in the
        // aggregate.
        assert_eq!(
            seen.len(),
            2 * 4 * 2 + (SIDE * SIDE * 4 * 2) as usize,
            "two stated values share a bit pattern"
        );
        assert_eq!(drawn_bytes(0).len(), plane_bytes(WIDE) as usize);
    }

    /// **Each axis of the loads case moves at least one texel, and the two off-diagonal
    /// load-action pairs are cells no single-attachment sweep can see.**
    ///
    /// Read as the rule rather than as a recording: the four load pairs answer four distinct
    /// (plane 0, plane 1) pairs, and swapping the two load actions moves **both** planes, so a
    /// side that applied attachment 0's load action to attachment 1 is right on the diagonal and
    /// wrong off it.
    #[test]
    fn each_axis_moves_a_texel_and_the_pair_is_not_vacuous() {
        let cells = AttachmentPairs::load_cells();
        assert_eq!(
            cells.len(),
            FORMAT_PAIRS.len() * LOADS.len() * LOADS.len() + LOADS.len()
        );
        let plane = |cell: &LoadCell, which: usize| -> Vec<u32> {
            (0..SIDE * SIDE)
                .flat_map(|at| expected(cell, which, at % SIDE, at / SIDE))
                .map(f32::to_bits)
                .collect()
        };
        // The load-action axis: four pairs, four distinct answers, at every format pair.
        for formats in FORMAT_PAIRS {
            let answers: BTreeSet<(Vec<u32>, Vec<u32>)> = LOADS
                .iter()
                .flat_map(|zero| LOADS.iter().map(move |one| [*zero, *one]))
                .map(|loads| {
                    let cell = LoadCell {
                        formats,
                        loads,
                        writes: Writes::Both,
                    };
                    (plane(&cell, 0), plane(&cell, 1))
                })
                .collect();
            assert_eq!(
                answers.len(),
                LOADS.len() * LOADS.len(),
                "{formats:?}: the four load pairs do not answer four planes"
            );
        }
        // And the pair's own claim: swapping the two load actions moves both planes, so neither
        // off-diagonal cell is reachable by getting one attachment right.
        let swapped = |formats: [PixelFormat; 2], loads: [LoadAction; 2]| {
            let straight = LoadCell {
                formats,
                loads,
                writes: Writes::Both,
            };
            let crossed = LoadCell {
                formats,
                loads: [loads[1], loads[0]],
                writes: Writes::Both,
            };
            (straight, crossed)
        };
        let (straight, crossed) = swapped([WIDE, WIDE], [LoadAction::Load, LoadAction::Clear]);
        for which in INDICES {
            assert_ne!(
                plane(&straight, which),
                plane(&crossed, which),
                "attachment {which}'s plane does not move when the two load actions are swapped"
            );
        }
        // The format axis: a one-channel attachment's plane is a quarter of a four-channel one's,
        // so a side that took the other attachment's format cannot answer the cell.
        assert_eq!(plane_bytes(NARROW) * 4, plane_bytes(WIDE));
        assert_eq!(texel_floats(NARROW), 1);
        assert_eq!(texel_floats(WIDE), 4);
        // The pipeline axis: with colour 0 unwritten its whole plane is its load action's, which
        // differs from the plane the same load action gives when the fragment writes it.
        let second = LoadCell {
            formats: [WIDE, WIDE],
            loads: [LoadAction::Load, LoadAction::Load],
            writes: Writes::SecondOnly,
        };
        let both = LoadCell {
            writes: Writes::Both,
            ..second
        };
        assert_ne!(plane(&second, 0), plane(&both, 0));
        // and attachment 1's plane is the same under either pipeline, since both write it: the
        // cell says what attachment 0's own actions do and nothing about attachment 1.
        assert_eq!(plane(&second, 1), plane(&both, 1));
    }

    /// **The resolves case states a resolve texture exactly where its store resolves**, reads
    /// exactly those attachments, and its three store pairs are three different shapes.
    ///
    /// `DontCare` beside a `resolveTexture` is the shape Metal's API validation stops, so this is
    /// the guard that keeps the family off it, held from the cells' own numbers with no device.
    #[test]
    fn a_resolve_texture_is_stated_only_where_the_store_resolves() {
        let case = AttachmentPairs {
            cross: Cross::Resolves,
        };
        assert_eq!(RESOLVE_CELLS.len(), 4);
        let names: BTreeSet<&str> = RESOLVE_CELLS.iter().map(|cell| cell.name).collect();
        assert_eq!(names.len(), RESOLVE_CELLS.len(), "two cells share a name");
        for cell in RESOLVE_CELLS {
            let held = INDICES.map(|which| resolves(cell.stores[which]));
            let built = pass(
                [LoadAction::Clear; 2],
                cell.stores,
                Handles::fingerprint(held),
            );
            for which in INDICES {
                assert_eq!(
                    built.colors[which].resolve.is_some(),
                    resolves(cell.stores[which]),
                    "{}: attachment {which} carries a resolve texture where its store {} resolve",
                    cell.name,
                    match resolves(cell.stores[which]) {
                        true => "does",
                        false => "does not",
                    }
                );
            }
            // At least one attachment of every cell resolves, or the cell reads nothing.
            assert!(held.iter().any(|resolves| *resolves), "{}", cell.name);
        }
        // The three store pairs the case states are three shapes: one attachment, the other, and
        // both. A pair that resolved at neither would read nothing and one that resolved alike at
        // both twice would be one shape stated twice.
        let shapes: BTreeSet<[bool; 2]> = RESOLVE_CELLS
            .iter()
            .map(|cell| INDICES.map(|which| resolves(cell.stores[which])))
            .collect();
        assert_eq!(shapes.len(), 3, "the cells are not three resolve shapes");
        let read: Vec<usize> = case.cells().iter().map(|cell| cell.read.len()).collect();
        assert_eq!(read, vec![1, 1, 2, 2]);
        // No cell's store action is `Store`, so no multisampled attachment of this case is ever
        // asked to be stored for a host to read, which Metal does not do.
        for cell in RESOLVE_CELLS {
            for store in cell.stores {
                assert!(matches!(
                    store,
                    StoreAction::MultisampleResolve | StoreAction::DontCare
                ));
            }
        }
    }

    /// **Every pass of every case stays inside its attachments and names what it holds**, and the
    /// draw's rectangle is a strict part of the target so every plane holds both regions.
    #[test]
    fn every_pass_is_inside_the_target_and_draws_a_strict_part_of_it() {
        // The check sees a rectangle that reaches past the target, so a green run means the draw's
        // is inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, u64::from(SIDE) + 1, 1).past([SIDE; 2]));
        assert!(!DRAWN.past([SIDE; 2]));
        assert!(
            DRAWN.height < u64::from(SIDE),
            "the draw covers the whole target, so no plane holds what a load action left"
        );
        assert!(DRAWN.width > 0 && DRAWN.height > 0);
        for case in all() {
            let built = AttachmentPairs {
                cross: match case.name() {
                    "pass-two-attachments-resolves" => Cross::Resolves,
                    _ => Cross::Loads,
                },
            };
            let samples = built.samples();
            let sites: BTreeSet<String> =
                built.cells().iter().map(|cell| cell.site.clone()).collect();
            assert_eq!(
                sites.len(),
                built.cells().len(),
                "{}: two cells share a site name",
                case.name()
            );
            for Cell { formats, read, .. } in built.cells() {
                assert!(!read.is_empty());
                for which in INDICES {
                    let spec = target(formats[which], samples);
                    assert_eq!(spec.samples, samples);
                    assert_eq!(spec.size, [SIDE; 2]);
                    // A multisampled attachment a resolve drawn after the pass must read is of
                    // shader-read usage; a single-sampled one needs none.
                    assert_eq!(spec.shader_read, samples > 1);
                    // A resolve texture is always single-sampled, which is the one shape Metal
                    // resolves at all.
                    assert_eq!(target(formats[which], 1).samples, 1);
                    assert_eq!(
                        layout(formats[which]).extent(),
                        plane_bytes(formats[which]) as usize
                    );
                }
            }
            // One draw a pass, and the fragment reads its two value buffers and its width at the
            // three indices the fixture declares.
            let encoded = steps(Handle(0), Handle(0), Handle(0), Handle(0));
            assert_eq!(
                encoded
                    .iter()
                    .filter(|step| matches!(step, RenderStep::Draw { .. }))
                    .count(),
                1
            );
            let bound: Vec<u32> = encoded
                .iter()
                .filter_map(|step| match step {
                    RenderStep::FragmentBuffer { index, .. } => Some(*index),
                    _ => None,
                })
                .collect();
            assert_eq!(bound, vec![0, 1, 2]);
            for step in &encoded {
                if let RenderStep::Scissor(rect) = step {
                    assert!(!rect.past([SIDE; 2]), "{}", case.name());
                }
            }
        }
    }

    /// **The family names its two cases apart**, and the loads case loosens nothing at all.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 2);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
        }
        // The loads case stores every attachment, so it has nothing to declare: a case that
        // declared a notice it cannot provoke would be a case nobody can read.
        assert!(AttachmentPairs {
            cross: Cross::Loads
        }
        .allowances()
        .is_empty());
        assert_eq!(
            AttachmentPairs {
                cross: Cross::Resolves
            }
            .allowances()
            .len(),
            2
        );
        // Both fragment functions the family names are the fixture's, and the loads case names
        // both while the resolves case names one.
        assert_eq!(shader_refs(&[BOTH, SECOND]).expect("the fixture").len(), 4);
        assert_eq!(shader_refs(&[BOTH]).expect("the fixture").len(), 2);
        // **Both pipelines name both attachments' formats.** A pipeline that left colour 0 at
        // `MTLPixelFormatInvalid` beside a pass that attached one is the shape Metal's API
        // validation stops, so no cell of this family may state it and this is the guard.
        for writes in [Writes::Both, Writes::SecondOnly] {
            let _ = writes;
            for state in color_states([WIDE, WIDE]) {
                assert!(
                    state.is_some(),
                    "a pipeline of this family leaves an attachment at MTLPixelFormatInvalid, \
                     which Metal's validation stops beside a pass that attaches one"
                );
            }
        }
        let names: BTreeSet<&str> = [
            load_name(LoadAction::Load),
            load_name(LoadAction::Clear),
            load_name(LoadAction::DontCare),
            store_name(StoreAction::Store),
            store_name(StoreAction::DontCare),
            store_name(StoreAction::MultisampleResolve),
            store_name(StoreAction::StoreAndMultisampleResolve),
            format_name(WIDE),
            format_name(NARROW),
        ]
        .into_iter()
        .collect();
        assert_eq!(names.len(), 9, "two enumerators share a name");
        assert!(!names.iter().any(|name| name.ends_with("Unknown")));
    }

    /// The one outcome of `case`'s recording, where the tree holds one.
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

    /// The bytes of `case`'s `planes/<site>/color<which>` site.
    fn recorded(case: &str, site: &str, which: usize) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("planes")
            .and_then(|node| node.child(site))
            .and_then(|node| node.child(&format!("color{which}")))?;
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: planes/{site}/color{which} holds no bytes");
        };
        Some(bytes.clone())
    }

    /// The `float`s the texel at `(x, y)` of a read-back plane of `format` holds.
    fn texel(bytes: &[u8], format: PixelFormat, x: u32, y: u32) -> Vec<f32> {
        let channels = texel_floats(format);
        let at = ((y * SIDE + x) * texel_bytes(format)) as usize;
        (0..channels as usize)
            .map(|channel| {
                let start = at + channel * 4;
                let mut word = [0u8; 4];
                word.copy_from_slice(&bytes[start..start + 4]);
                f32::from_le_bytes(word)
            })
            .collect()
    }

    /// **Metal's recorded loads case holds, at every texel of every cell, what the attachment's
    /// own load action and the pipeline's own attachment set say — not the other attachment's.**
    #[test]
    fn metal_held_each_attachments_own_state() {
        let case = "pass-two-attachments-loads";
        for cell in AttachmentPairs::load_cells() {
            let site = cell.site();
            for which in INDICES {
                let Some(bytes) = recorded(case, &site, which) else {
                    println!("{case}: no recording of {site} in the tree yet");
                    return;
                };
                assert_eq!(bytes.len(), plane_bytes(cell.formats[which]) as usize);
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let held = texel(&bytes, cell.formats[which], x, y);
                        let want = expected(&cell, which, x, y);
                        assert_eq!(
                            held.iter().map(|c| c.to_bits()).collect::<Vec<u32>>(),
                            want.iter().map(|c| c.to_bits()).collect::<Vec<u32>>(),
                            "{site}: attachment {which} texel ({x}, {y}) holds {held:?}, not \
                             {want:?}"
                        );
                    }
                }
            }
        }
    }

    /// **Metal's recorded resolves case holds each attachment's own drawn value in its own resolve
    /// texture**, in every cell that resolves it.
    ///
    /// Every sample of every covered pixel carries one value, so the resolve is that value
    /// exactly; outside the draw's rectangle each plane is its clear, resolved.
    #[test]
    fn metal_resolved_each_attachment_into_its_own_texture() {
        let case = "pass-two-attachments-resolves";
        for cell in RESOLVE_CELLS {
            for which in INDICES.into_iter().filter(|w| resolves(cell.stores[*w])) {
                let Some(bytes) = recorded(case, cell.name, which) else {
                    println!("{case}: no recording of {} in the tree yet", cell.name);
                    return;
                };
                let format = cell.formats[which];
                let channels = texel_floats(format) as usize;
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let inside = x < DRAWN.width as u32 && y < DRAWN.height as u32;
                        let want: Vec<f32> = match inside {
                            true => drawn(which, y * SIDE + x)[..channels].to_vec(),
                            false => CLEARS[which][..channels]
                                .iter()
                                .map(|c| *c as f32)
                                .collect(),
                        };
                        let held = texel(&bytes, format, x, y);
                        assert_eq!(
                            held.iter().map(|c| c.to_bits()).collect::<Vec<u32>>(),
                            want.iter().map(|c| c.to_bits()).collect::<Vec<u32>>(),
                            "{}: attachment {which} texel ({x}, {y}) holds {held:?}, not {want:?}",
                            cell.name
                        );
                    }
                }
            }
        }
    }
}
