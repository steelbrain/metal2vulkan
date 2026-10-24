//! **One attachment, two passes in a row**: what pass B *loads* given what pass A *stored*, over
//! the same texture.
//!
//! The suite had all 36 (load, store, options) triples on the **descriptor**
//! ([`crate::pass_descriptors`]'s `pass-attachment-actions`) — set the field, read it back — and
//! [`crate::cases::pass_attachment_pairs`] crossed two attachments' actions *inside one pass*.
//! `MTLStoreAction` was crossed by [`crate::cases::depth_resolve_pairs`]'s resolve filters and
//! `MTLLoadAction` by nothing at all. So no case anywhere asked the one question a load action
//! exists to answer: **does the next pass see what this one stored?** That is where the two enums
//! become attachment layouts, tile loads and a store's timing, and a side can be exactly right
//! about each action's own descriptor field and wrong about the sequence.
//!
//! | case | what it crosses |
//! |---|---|
//! | `pass-sequence-color-actions` | pass A's store action by pass B's load action over one colour attachment, by whether the attachment's prior value came from the **host** or from pass A's own clear, by pass B's **render area** |
//! | `pass-sequence-aspect-actions` | the **depth** aspect's actions against the **stencil** aspect's, stated differently in each pass: pass A's two store actions by pass B's two load actions |
//!
//! # The three regions every plane holds
//!
//! Each cell runs one command buffer over one 40 by 40 attachment whose every texel the **host**
//! wrote before anything ran, each texel its own value. Pass A draws into the 32 by 16 at the
//! origin; pass B draws into the 16 by 32 at the origin. So every plane read back has three
//! regions a reader can name:
//!
//! - **B's**, `x < 16 && y < 32`: what pass B's fragment wrote. It is there in every cell, and a
//!   plane that does not hold it says the sequence did not run.
//! - **A's alone**, `16 <= x < 32 && y < 16`: what pass A wrote and pass B did not. This is the
//!   region that separates "pass A's store reached memory and pass B's load read it back" from
//!   every other answer.
//! - **neither's**, the rest: a texel no fragment of either pass ever covered. It holds the
//!   attachment's prior value where both passes kept it — and **which** prior value it holds says
//!   whether the host's write or pass A's clear is the thing that survived.
//!
//! That third region is why the host writes the texture at all. A case whose prior value came only
//! from an earlier pass cannot tell "the pass preserved it" from "the texture happened to hold
//! it": both answers are the same bits. Here they are not. Pass A's load action is the axis: under
//! `MTLLoadActionLoad` the neither's region still holds the **host's** per-texel value, and under
//! `MTLLoadActionClear` it holds pass A's clear, a value the host never wrote.
//!
//! # What each cell of the colour grid shows
//!
//! Twelve cells. Eight over the whole attachment — pass A's load by pass A's store by pass B's
//! load — and four more with pass B's **render area** stated at 32 by 32 of a 40 by 40 attachment.
//!
//! - **`MTLStoreActionStore` then `MTLLoadActionLoad`** is the only pair that must preserve bytes,
//!   and it is the cell that reads all three regions at their own values at once.
//! - **`MTLStoreActionDontCare` then `MTLLoadActionLoad`**: Metal's answer is **the texels the
//!   attachment held before pass A** — the host's write, in A's region as much as anywhere, so
//!   pass A's clear *and* its draw are both gone. Recorded, not assumed: a reader who guessed
//!   "the clear survives" or "the draw survives" would have written a different table. It is also
//!   the cell where pass A's **load** action turns out to decide nothing, since the pass's whole
//!   result is discarded.
//! - **`MTLStoreActionStore` then `MTLLoadActionClear`**: pass B's clear covers the attachment, so
//!   pass A's store action moves **no texel**. Four cells hold one plane. That is an independence
//!   claim and not a gap — and the bound cells are where it stops holding.
//! - **The render area.** `pass_bound_pairs` measured inside one pass that a `Clear` load clears
//!   the render area rounded up to whole tiles of `TILE` pixels a side and clipped to the
//!   attachment, and that a `Load` load keeps every texel past it. Both carry across the pass
//!   boundary here, and the second is the sharper statement: under pass B's `MTLLoadActionLoad` the
//!   32 by 32 bound moves **not one texel** against the whole-area cell, so a side that clipped a
//!   `Load` to the render area, or that re-cleared past it, fails a cell whose twin it passes.
//!   Under pass B's `MTLLoadActionClear` the bound does the opposite: 32 rounds to 32 and the
//!   texels past it keep **what pass A left there**, so pass A's store action is visible through a
//!   `Clear` after all. A side that read the clear's reach as the whole attachment answers both
//!   bound `Clear` cells alike and both of them wrongly.
//!
//! # The aspect case, and the asymmetry it is for
//!
//! A pass's depth and its stencil carry **their own** load and store actions, and a side that
//! reads one aspect's action for both passes every symmetric cell. So pass A stores the depth and
//! discards the stencil, or the reverse, and pass B loads the depth and clears the stencil, or the
//! reverse: nine cells of (pass A's two stores) by (pass B's two loads), of which the six with an
//! asymmetry are cells no symmetric grid reaches.
//!
//! The two aspects are **separate textures**, a `Depth32Float` beside a `Stencil8`, because
//! reading a `Depth32Float_Stencil8` texture back needs `MTLBlitOptionDepthFromDepthStencil` or
//! its stencil twin and [`RenderWork::ReadTexture`] has no field for it — the same wall
//! `depth_resolve_pairs` and `stencil_resolve_pairs` both record. Separate textures are not the
//! weaker shape they look like: a device with no single-aspect stencil format holds a `Stencil8`
//! texture in a combined one and has to apply the stencil's own operations to one aspect of it
//! while leaving the other alone, which is exactly the place an aspect's action goes to the wrong
//! half.
//!
//! # What is not asked, and why
//!
//! - **`MTLLoadActionDontCare` as pass B's load.** Metal keeps the texels of the tiles no
//!   primitive of the pass touched; which tiles those are is the GPU's placement of the vertices,
//!   and our runtime clears the render area instead and announces the near-equivalent
//!   (`Degradation::DontCareLoadTiles`). A cell over it would compare two rasterizers rather than
//!   Metal's rule, which is the reading `pass_bound_pairs` and `pass_attachment_pairs` both
//!   reached. `MTLStoreActionDontCare` as pass A's **store** is a different matter and is swept:
//!   Metal's answer there is the earlier texels, measured over 926 passes on an M3 Max
//!   (`metal2vulkan_runtime`'s `kept_bytes.rs`), and the eight runs of each cell are what say
//!   whether it held.
//! - **A resolving store followed by a load.** `MTLStoreActionMultisampleResolve` writes the
//!   resolve texture and keeps nothing of the multisampled one, and
//!   `MTLStoreActionStoreAndMultisampleResolve` keeps both — so a pass B that loads the
//!   multisampled texture is the cell that parts them. It wants a second multisampled pass whose
//!   own resolve is what a blit can read, and it is a family of its own rather than a cell of this
//!   one. Recorded as the open crossing it is.
//! - **A non-resolving store beside a resolve texture, and a memoryless attachment.** Metal's own
//!   API validation ends the process on both, and record mode runs a validated leg beside the
//!   plain one. Neither is authored.
//!
//! # Why this family is safe
//!
//! One command buffer a cell: a host write before it, then two render passes over one 40 by 40
//! single-sampled attachment (three over two attachments in the aspect case, counting the pass
//! that states their prior values) and one blit an attachment. One draw of three vertices a pass,
//! no loop in either stage, **every viewport and every scissor inside its own pass's render
//! area**, every vertex inside the clip volume, every depth inside `[0, 1]`, and no multisampling.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, PassStencil, Plane, ReadAt, RenderPass, RenderPipelineSpec,
    RenderStep, RenderWork, ScissorBox, StageFunction, TargetSpec, ViewportRect,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation};
use metal2vulkan_runtime::{Degradation, LoadAction, StoreAction};

/// The colour fixture: the fullscreen triangle, and the fragment that returns its own texel's
/// `float4` of a buffer times its own texel's entry of a second one.
///
/// Taken rather than written: with the second buffer every `float4(1)`, the fragment's answer is
/// the first buffer's value at that texel **exactly** (one times a finite float is that float), so
/// every texel of every pass is its own stated binary fraction and no cell's answer turns on a
/// conversion or on a rasterizer's interpolation.
const COLOR_FIXTURE: &str = "xrb_computed";

/// `xrb_computed_vertex`: `float4(4 * a - 1, 0, 1)` at each of three vertex ids.
const VERTEX: &str = "xrb_computed_vertex";

/// `xrb_computed_float`: `colors[i] * ones[i]` at the fragment's own texel `i`.
const COLOR_FRAGMENT: &str = "xrb_computed_float";

/// The depth-and-stencil fixture: the fragment that writes **both** a per-pixel `[[depth(any)]]`
/// and a per-pixel `[[stencil]]`, which is what lets one draw state the two aspects apart.
const ASPECT_FIXTURE: &str = "stx_stencil";

/// `stx_stencil_with_depth`: stencil from buffer 0, row width from buffer 1, depth from buffer 2.
const ASPECT_FRAGMENT: &str = "stx_stencil_with_depth";

/// The side of every attachment, in texels.
///
/// Forty, as `pass_bound_pairs`'s is and for its reason: a tile of [`TILE`] pixels does not divide
/// it, so a render area of 32 rounds up to 32 and leaves 8 texels of the attachment past the
/// reach, where a 32- or 64-wide attachment would leave none.
const SIDE: u32 = 40;

/// The side, in pixels, of the tiles Metal's `Clear` load clears in past the render area, on both
/// axes: what `metal2vulkan_runtime`'s `clear_reach.rs` measured on an M3 Max and an M4.
///
/// Stated as the number this family **expects Metal to answer**; the recording is what says
/// whether it does.
const TILE: u32 = 32;

/// The render area pass B states in the bound cells: 32 by 32 of the 40 by 40 attachment, which is
/// a whole tile on each axis, so a `Clear` load's reach is 32 and the 8 texels past it are pass
/// A's to decide.
const BOUND: u32 = 32;

/// The rectangle pass A's draw is confined to: the 32 by 16 at the origin.
const A_RECT: ScissorBox = ScissorBox::new(0, 0, 32, 16);

/// The rectangle pass B's draw is confined to: the 16 by 32 at the origin.
///
/// Both rectangles are inside 32 by 32, so one scissor and one viewport a pass serve the
/// whole-area cells and the bound cells alike and no draw can reach past its own render area.
const B_RECT: ScissorBox = ScissorBox::new(0, 0, 16, 32);

/// The colour attachment's format: one `float` a texel, which a blit reads back as itself and
/// which holds every value this family states exactly.
const COLOR_FORMAT: PixelFormat = PixelFormat::R32Float;

/// Bytes one colour texel takes.
const COLOR_TEXEL: u32 = 4;

/// The depth attachment's format.
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// The stencil attachment's format, its own texture rather than an aspect of a combined one, for
/// the reason the module states.
const STENCIL_FORMAT: PixelFormat = PixelFormat::Stencil8;

/// How many runs record mode takes a case of this family over.
///
/// Eight, and load-bearing here rather than a habit: a `DontCare` store is the one action whose
/// answer Metal's own documentation leaves to the GPU, and eight runs folding into one outcome is
/// what says this GPU's answer is the measured one. A case whose answer moved between runs is the
/// finding.
const RUNS: u32 = 8;

/// The host's value at texel `at`: its own multiple of `2^-16`, in the band `[1/16, 3/32)`.
///
/// Per texel rather than one constant, so a plane that holds the attachment's prior value at the
/// wrong texel is caught texel by texel and not only in the aggregate.
fn host(at: u32) -> f32 {
    (4096 + at) as f32 / 65536.0
}

/// What pass A's fragment writes at texel `at`: its own multiple of `2^-16`, in `[1/4, 9/32)`.
fn a_drawn(at: u32) -> f32 {
    (16384 + at) as f32 / 65536.0
}

/// What pass B's fragment writes at texel `at`: its own multiple of `2^-16`, in `[1/2, 17/32)`.
fn b_drawn(at: u32) -> f32 {
    (32768 + at) as f32 / 65536.0
}

/// The colour pass A's `Clear` load writes: 3/4, which neither the host nor either fragment ever
/// states.
const A_CLEAR: f64 = 49152.0 / 65536.0;

/// The colour pass B's `Clear` load writes: 13/16, likewise its own value.
const B_CLEAR: f64 = 53248.0 / 65536.0;

/// Texels the attachment holds.
const TEXELS: u32 = SIDE * SIDE;

/// Bytes the colour attachment's whole level 0 takes.
const COLOR_PLANE: u32 = COLOR_TEXEL * TEXELS;

/// The bytes of the buffer the host writes into the colour attachment before anything runs: one
/// `float` a texel, in the row-major order a blit reads back.
fn host_bytes() -> Vec<u8> {
    (0..TEXELS).flat_map(|at| host(at).to_le_bytes()).collect()
}

/// The bytes of a fragment's value buffer: one `float4` a texel, in the order the fixture indexes
/// it (`y * width + x`), every channel the texel's own value.
fn drawn_bytes(value: fn(u32) -> f32) -> Vec<u8> {
    (0..TEXELS)
        .flat_map(|at| {
            let held = value(at);
            [held; 4].into_iter().flat_map(f32::to_le_bytes)
        })
        .collect()
}

/// The bytes of the buffer the fixture multiplies by: every `float4` one, so a fragment's answer
/// is its value buffer's own texel exactly.
fn ones_bytes() -> Vec<u8> {
    (0..TEXELS)
        .flat_map(|_| [1.0f32; 4].into_iter().flat_map(f32::to_le_bytes))
        .collect()
}

/// The bytes of the `uint` the fixtures read their row width from.
fn width_bytes() -> Vec<u8> {
    SIDE.to_le_bytes().to_vec()
}

/// The layout the colour attachment's whole level 0 is read under: one `float` an element, so a
/// diff names the texel that moved.
fn color_layout() -> ByteLayout {
    plane_layout(Aspect::Color, Element::F32, COLOR_TEXEL)
}

/// The layout a plane of `texel` bytes an element is read under.
fn plane_layout(aspect: Aspect, element: Element, texel: u32) -> ByteLayout {
    let row_pitch = texel * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element,
        row_offset: 0,
        row_elements: SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The colour attachment: a single-sampled render target of **shared** storage, so the host may
/// write its every texel with `replaceRegion:` before the command buffer runs.
///
/// Shared rather than private because that is the only route to an arbitrary prior value: neither
/// a clear nor a kernel write can put a per-texel code in, which is the same reason the blend
/// families make theirs shared.
fn color_target() -> TargetSpec {
    TargetSpec::plain(COLOR_FORMAT, [SIDE; 2]).shared()
}

/// The one colour state the colour pipeline holds.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(COLOR_FORMAT))]
}

/// The colour pipeline: the fullscreen triangle and the per-texel fragment over the one
/// attachment.
fn color_pipeline(states: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(VERTEX, COLOR_FRAGMENT, states)
}

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// The viewport a draw confined to `rect` states: the rectangle itself, so the fullscreen triangle
/// lands exactly on it and the fragment's `[[position]]` is the attachment's own texel.
fn viewport(rect: ScissorBox) -> ViewportRect {
    ViewportRect::new(
        [rect.x as f64, rect.y as f64],
        [rect.width as f64, rect.height as f64],
    )
}

/// The extent from the origin a `Clear` load of this family's attachment clears in a pass over
/// `area` that has a draw call: Metal's rule as `clear_reach.rs` measured it — the render area
/// rounded up to whole [`TILE`]s and clipped to the attachment.
///
/// Every pass of this family has a draw call, so the one-layer no-draw-call rule
/// (`pass_bound_pairs`'s own cells) is not reached here.
fn reach(area: [u32; 2]) -> [u32; 2] {
    [0, 1].map(|axis| area[axis].div_ceil(TILE).saturating_mul(TILE).min(SIDE))
}

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

/// One cell of the colour case: pass A's load action, pass A's store action, pass B's load action,
/// and the render area pass B states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColorCell {
    a_load: LoadAction,
    a_store: StoreAction,
    b_load: LoadAction,
    /// Pass B's `renderTargetWidth` and `renderTargetHeight`, stated only in the bound cells; the
    /// whole-area cells state none and take the attachment's extent, as Metal does.
    b_area: Option<[u32; 2]>,
}

impl ColorCell {
    /// The name the cell's sites take: the three actions by their own enumerator names, then the
    /// render area where the cell states one.
    fn site(&self) -> String {
        let actions = format!(
            "{}+{}/{}",
            load_name(self.a_load),
            store_name(self.a_store),
            load_name(self.b_load)
        );
        match self.b_area {
            None => format!("{actions}/whole-area"),
            Some([width, height]) => format!("{actions}/area-{width}x{height}"),
        }
    }

    /// The render area pass B's `Clear` load clears over: the cell's own bound, or the
    /// attachment's extent where it states none.
    fn b_reach(&self) -> [u32; 2] {
        reach(self.b_area.unwrap_or([SIDE; 2]))
    }
}

/// The two load actions either pass states: the two whose answer is defined on both sides.
const LOADS: [LoadAction; 2] = [LoadAction::Load, LoadAction::Clear];

/// The two store actions pass A states.
const STORES: [StoreAction; 2] = [StoreAction::Store, StoreAction::DontCare];

/// Every cell of the colour case, in the order it runs them.
fn color_cells() -> Vec<ColorCell> {
    let mut cells = Vec::new();
    for a_load in LOADS {
        for a_store in STORES {
            for b_load in LOADS {
                cells.push(ColorCell {
                    a_load,
                    a_store,
                    b_load,
                    b_area: None,
                });
            }
        }
    }
    // The bound cells, at pass A's `Clear` load alone: that is what puts a value pass A states
    // into the texels past pass B's reach, so pass A's store action is what decides them. Under
    // pass A's `Load` both store actions would leave the host's value there and the bound could
    // say nothing.
    for a_store in STORES {
        for b_load in LOADS {
            cells.push(ColorCell {
                a_load: LoadAction::Clear,
                a_store,
                b_load,
                b_area: Some([BOUND; 2]),
            });
        }
    }
    cells
}

/// The handles a colour cell's passes name.
#[derive(Clone, Copy)]
struct ColorHandles {
    target: Handle,
    pipeline: Handle,
    a_values: Handle,
    b_values: Handle,
    ones: Handle,
    width: Handle,
}

impl ColorHandles {
    /// The zeroed set a fingerprint is taken over: a side's handle numbers are its own and none
    /// reaches a fingerprint.
    const fn fingerprint() -> Self {
        let zero = Handle(0);
        Self {
            target: zero,
            pipeline: zero,
            a_values: zero,
            b_values: zero,
            ones: zero,
            width: zero,
        }
    }
}

/// The steps a colour pass encodes: the pipeline, the value buffer, the row width, the ones, the
/// viewport and scissor that confine the draw to `rect`, and one triangle.
fn color_steps(
    rect: ScissorBox,
    pipeline: Handle,
    values: Handle,
    ones: Handle,
    width: Handle,
) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::FragmentBuffer {
            buffer: values,
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
        RenderStep::Viewport(viewport(rect)),
        RenderStep::Scissor(rect),
        TRIANGLE,
    ]
}

/// Pass A of a colour cell: the attachment under the cell's load and store actions, cleared to
/// [`A_CLEAR`], and one draw confined to [`A_RECT`].
fn color_pass_a(cell: &ColorCell, handles: ColorHandles) -> RenderPass {
    let mut color = PassColor::cleared(handles.target, [A_CLEAR; 4]);
    color.load = cell.a_load;
    color.store = cell.a_store;
    RenderPass::clearing(vec![color]).drawing(color_steps(
        A_RECT,
        handles.pipeline,
        handles.a_values,
        handles.ones,
        handles.width,
    ))
}

/// Pass B of a colour cell: the attachment under the cell's load action and **always** stored, so
/// every cell reads a plane, with the cell's render area where it states one.
fn color_pass_b(cell: &ColorCell, handles: ColorHandles) -> RenderPass {
    let mut color = PassColor::cleared(handles.target, [B_CLEAR; 4]);
    color.load = cell.b_load;
    color.store = StoreAction::Store;
    let mut pass = RenderPass::clearing(vec![color]).drawing(color_steps(
        B_RECT,
        handles.pipeline,
        handles.b_values,
        handles.ones,
        handles.width,
    ));
    pass.size = cell.b_area;
    pass
}

/// One cell of the aspect case: pass A's two store actions and pass B's two load actions, each
/// stated per aspect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AspectCell {
    /// Pass A's store action at the depth attachment and at the stencil attachment.
    a_stores: [StoreAction; 2],
    /// Pass B's load action at the depth attachment and at the stencil attachment.
    b_loads: [LoadAction; 2],
}

impl AspectCell {
    /// The name the cell's sites take: each pass's two actions, aspect by aspect.
    fn site(&self) -> String {
        format!(
            "depth-{}+stencil-{}/depth-{}+stencil-{}",
            store_name(self.a_stores[0]),
            store_name(self.a_stores[1]),
            load_name(self.b_loads[0]),
            load_name(self.b_loads[1])
        )
    }
}

/// The prior depth every texel holds, written by the pass that primes the two attachments: a value
/// no draw of either pass writes.
const PRIOR_DEPTH: f64 = 0.125;

/// The prior stencil every texel holds.
const PRIOR_STENCIL: u32 = 0x11;

/// The depth pass B's `Clear` load writes.
const B_CLEAR_DEPTH: f64 = 0.875;

/// The stencil pass B's `Clear` load writes: past both fragments' bands, so no texel's byte can
/// be read as two different things having put it there.
const B_CLEAR_STENCIL: u32 = 0xf5;

/// The depth pass A's fragment writes at texel `at`, and pass B's: each its own multiple of
/// `2^-16` in its own band, so a plane names which pass put a texel there.
fn a_depth(at: u32) -> f32 {
    (16384 + at) as f32 / 65536.0
}

/// The depth pass B's fragment writes at texel `at`.
fn b_depth(at: u32) -> f32 {
    (32768 + at) as f32 / 65536.0
}

/// The stencil pass A's fragment writes at texel `at`: `0x40` plus the texel's low six bits.
///
/// A stencil byte holds 256 values and the attachment has 1600 texels, so a per-texel value cannot
/// be unique here as the colour case's is. What it can be, and is, is its **pass's own band**:
/// pass A writes `0x40..0x80`, pass B `0x80..0xc0`, and neither clear is in either, so a byte says
/// which pass put it there and where in its 64-texel run it came from.
fn a_stencil(at: u32) -> u32 {
    0x40 + (at & 0x3f)
}

/// The stencil pass B's fragment writes at texel `at`.
fn b_stencil(at: u32) -> u32 {
    0x80 + (at & 0x3f)
}

/// The three store pairs the aspect case states at pass A, and the three load pairs at pass B:
/// each aspect's own action equal to the other's, then each asymmetry.
const ASPECT_STORES: [[StoreAction; 2]; 3] = [
    [StoreAction::Store, StoreAction::Store],
    [StoreAction::Store, StoreAction::DontCare],
    [StoreAction::DontCare, StoreAction::Store],
];

/// Pass B's three load pairs.
const ASPECT_LOADS: [[LoadAction; 2]; 3] = [
    [LoadAction::Load, LoadAction::Load],
    [LoadAction::Load, LoadAction::Clear],
    [LoadAction::Clear, LoadAction::Load],
];

/// Every cell of the aspect case: the three store pairs by the three load pairs.
fn aspect_cells() -> Vec<AspectCell> {
    let mut cells = Vec::new();
    for a_stores in ASPECT_STORES {
        for b_loads in ASPECT_LOADS {
            cells.push(AspectCell { a_stores, b_loads });
        }
    }
    cells
}

/// The bytes of a stencil value buffer: one `uint` a texel, as the fixture indexes it.
fn stencil_bytes(value: fn(u32) -> u32) -> Vec<u8> {
    (0..TEXELS).flat_map(|at| value(at).to_le_bytes()).collect()
}

/// The bytes of a depth value buffer: one `float` a texel.
fn depth_bytes(value: fn(u32) -> f32) -> Vec<u8> {
    (0..TEXELS).flat_map(|at| value(at).to_le_bytes()).collect()
}

/// The depth attachment: a plain single-sampled target, whose prior value an earlier pass of the
/// same command buffer states.
fn depth_target() -> TargetSpec {
    TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2])
}

/// The stencil attachment.
fn stencil_target() -> TargetSpec {
    TargetSpec::plain(STENCIL_FORMAT, [SIDE; 2])
}

/// The depth/stencil state every draw of the aspect case takes: write the depth the fragment
/// carries whatever is there, and replace the stencil with the reference the fragment's
/// `[[stencil]]` output is.
fn aspect_state() -> DepthStencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = CompareFunction::Always;
    face.depth_stencil_pass_operation = StencilOperation::Replace;
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state.front_face_stencil = face;
    state.back_face_stencil = face;
    state
}

/// A depth/stencil descriptor's fields as the bytes a case's input fingerprint covers: the depth
/// compare and write, then each face's compare, three operations and two masks.
fn state_words(state: &DepthStencilDescriptor) -> Vec<u8> {
    let mut bytes = state.depth_compare_function.as_raw().to_le_bytes().to_vec();
    bytes.push(u8::from(state.depth_write_enabled));
    for face in [&state.front_face_stencil, &state.back_face_stencil] {
        bytes.extend(face.stencil_compare_function.as_raw().to_le_bytes());
        bytes.extend(face.stencil_failure_operation.as_raw().to_le_bytes());
        bytes.extend(face.depth_failure_operation.as_raw().to_le_bytes());
        bytes.extend(face.depth_stencil_pass_operation.as_raw().to_le_bytes());
        bytes.extend(face.read_mask.to_le_bytes());
        bytes.extend(face.write_mask.to_le_bytes());
    }
    bytes
}

/// The aspect case's pipeline: the colour fixture's fullscreen triangle, the depth-and-stencil
/// fragment of its own library, and **no** colour attachment.
fn aspect_pipeline(library: Handle) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(
        StageFunction::named(VERTEX),
        Some(StageFunction::of(library, ASPECT_FRAGMENT)),
        &[],
    )
    .depth_stencil(Some(DEPTH_FORMAT), Some(STENCIL_FORMAT))
}

/// The handles an aspect cell's passes name.
#[derive(Clone, Copy)]
struct AspectHandles {
    depth: Handle,
    stencil: Handle,
    pipeline: Handle,
    state: Handle,
    stencils: Handle,
    depths: Handle,
    width: Handle,
}

impl AspectHandles {
    /// The zeroed set a fingerprint is taken over.
    const fn fingerprint() -> Self {
        let zero = Handle(0);
        Self {
            depth: zero,
            stencil: zero,
            pipeline: zero,
            state: zero,
            stencils: zero,
            depths: zero,
            width: zero,
        }
    }
}

/// The steps an aspect pass encodes: the pipeline, the state, the two value buffers and the row
/// width, the viewport and scissor that confine the draw to `rect`, and one triangle.
fn aspect_steps(rect: ScissorBox, handles: AspectHandles) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(handles.pipeline),
        RenderStep::DepthStencilState(handles.state),
        RenderStep::FragmentBuffer {
            buffer: handles.stencils,
            offset: 0,
            index: 0,
        },
        RenderStep::FragmentBuffer {
            buffer: handles.width,
            offset: 0,
            index: 1,
        },
        RenderStep::FragmentBuffer {
            buffer: handles.depths,
            offset: 0,
            index: 2,
        },
        RenderStep::Viewport(viewport(rect)),
        RenderStep::Scissor(rect),
        TRIANGLE,
    ]
}

/// The pass that states the two attachments' prior values: both cleared and stored, with no draw
/// call, so every texel of both holds a known value before pass A opens.
fn aspect_priming(handles: AspectHandles) -> RenderPass {
    RenderPass::clearing(Vec::new())
        .with_depth(PassDepth::cleared(handles.depth, PRIOR_DEPTH))
        .with_stencil(PassStencil::cleared(handles.stencil, PRIOR_STENCIL))
}

/// Pass A of an aspect cell: both attachments loaded, each stored under its **own** store action,
/// and one draw confined to [`A_RECT`].
fn aspect_pass_a(cell: &AspectCell, handles: AspectHandles) -> RenderPass {
    let mut depth = PassDepth::cleared(handles.depth, PRIOR_DEPTH);
    depth.load = LoadAction::Load;
    depth.store = cell.a_stores[0];
    let mut stencil = PassStencil::cleared(handles.stencil, PRIOR_STENCIL);
    stencil.load = LoadAction::Load;
    stencil.store = cell.a_stores[1];
    RenderPass::clearing(Vec::new())
        .with_depth(depth)
        .with_stencil(stencil)
        .drawing(aspect_steps(A_RECT, handles))
}

/// Pass B of an aspect cell: each attachment under its **own** load action and both stored, and
/// one draw confined to [`B_RECT`].
fn aspect_pass_b(cell: &AspectCell, handles: AspectHandles) -> RenderPass {
    let mut depth = PassDepth::cleared(handles.depth, B_CLEAR_DEPTH);
    depth.load = cell.b_loads[0];
    let mut stencil = PassStencil::cleared(handles.stencil, B_CLEAR_STENCIL);
    stencil.load = cell.b_loads[1];
    RenderPass::clearing(Vec::new())
        .with_depth(depth)
        .with_stencil(stencil)
        .drawing(aspect_steps(B_RECT, handles))
}

/// Which of the family's two crosses a case states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cross {
    /// Pass A's store action by pass B's load action over one colour attachment, by the prior
    /// value's origin and by pass B's render area.
    Color,
    /// The depth aspect's actions against the stencil aspect's, in both passes.
    Aspects,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(SequenceActions {
            cross: Cross::Color,
        }),
        Box::new(SequenceActions {
            cross: Cross::Aspects,
        }),
    ]
}

/// One of the family's two crosses over two passes in a row.
pub struct SequenceActions {
    cross: Cross,
}

impl SequenceActions {
    /// The name the case's cross gives it.
    fn case_name(&self) -> &'static str {
        match self.cross {
            Cross::Color => "pass-sequence-color-actions",
            Cross::Aspects => "pass-sequence-aspect-actions",
        }
    }
}

/// The shader references a fixture's two texts give for `entries`.
fn shader_refs(name: &'static str, entries: &[&str]) -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(name);
    let mut refs = Vec::new();
    for extension in [RUNTIME_TEXT, METAL_TEXT] {
        let digest = crate::sha256_hex(held.text(extension)?.as_bytes());
        for entry in entries {
            refs.push(ShaderRef {
                fixture: format!("{name}{extension}"),
                entry: (*entry).to_owned(),
                source_sha256: digest.clone(),
            });
        }
    }
    Ok(refs)
}

impl Case for SequenceActions {
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
        // **No value is loosened**: every texel Metal leaves is one we must leave, in every cell
        // of both cases. Two notices are declared and neither is a difference.
        //
        // `DontCareStore`: every cell of both cases stores an attachment `DontCare` somewhere, and
        // a device whose `DontCare` store is not Metal's own announces that it took the
        // near-equivalent. It fires on no venue whose driver stores `DONT_CARE` as Metal does
        // (MoltenVK on this M3 Max among them), which is why it is declared rather than relied on.
        //
        // `SharedInSystemMemory`: both cases read their attachments back through a `Shared`
        // buffer, and a device with no unified memory keeps a `Shared` allocation in system memory
        // because that is the only memory both sides can reach. It belongs to the venue's memory
        // model rather than to a load or a store action.
        let mut allowances = vec![
            crate::replay::notice_allowance(Degradation::DontCareStore.code()),
            crate::replay::notice_allowance(Degradation::SharedInSystemMemory.code()),
        ];
        // `SeparateDepthStencil`: the aspect case's two attachments are a `Depth32Float` beside a
        // `Stencil8`, which the runtime renders through a scratch image of a combined format,
        // copying each loaded aspect in and each stored one back. That route is the whole point of
        // the case -- it is where an aspect's own action can reach the wrong half -- and it is a
        // performance notice, not a value.
        // `ShaderSideStencil`: the same scratch route makes the stencil reference and the stencil
        // test the shader's work on a device whose own separate-aspect path is not Metal's, which
        // is a statement about how the aspect was reached and not about the aspect's value.
        if self.cross == Cross::Aspects {
            allowances.extend([
                crate::replay::notice_allowance(Degradation::SeparateDepthStencil.code()),
                crate::replay::notice_allowance(Degradation::ShaderSideStencil.code()),
            ]);
        }
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut refs = shader_refs(COLOR_FIXTURE, &[VERTEX])?;
        match self.cross {
            Cross::Color => refs.extend(shader_refs(COLOR_FIXTURE, &[COLOR_FRAGMENT])?),
            Cross::Aspects => refs.extend(shader_refs(ASPECT_FIXTURE, &[ASPECT_FRAGMENT])?),
        }
        Ok(refs)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(Fixture::new(COLOR_FIXTURE).text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(Fixture::new(COLOR_FIXTURE).text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("width", &width_bytes()),
            Input::bytes("a-rect", &A_RECT.words()),
            Input::bytes("b-rect", &B_RECT.words()),
        ];
        match self.cross {
            Cross::Color => {
                inputs.push(Input::bytes("host", &host_bytes()));
                inputs.push(Input::bytes("drawn:a", &drawn_bytes(a_drawn)));
                inputs.push(Input::bytes("drawn:b", &drawn_bytes(b_drawn)));
                inputs.push(Input::bytes("ones", &ones_bytes()));
                let mut clears = Vec::new();
                for value in [A_CLEAR, B_CLEAR] {
                    clears.extend(value.to_bits().to_le_bytes());
                }
                inputs.push(Input::bytes("clears", &clears));
                inputs.push(Input::bytes("target", &color_target().words()));
                let states = colors();
                inputs.push(Input::bytes("pipeline", &color_pipeline(&states).words()));
                for cell in color_cells() {
                    let handles = ColorHandles::fingerprint();
                    // The reach the cell expects pass B's `Clear` load to clear over, as
                    // `pass_bound_pairs` fingerprints its own: Metal's tile rule is part of what
                    // the cell means, so a recording decays loudly if the rule ever moves.
                    let mut bound = Vec::new();
                    for axis in cell.b_reach() {
                        bound.extend(axis.to_le_bytes());
                    }
                    inputs.push(Input::bytes(format!("reach:{}", cell.site()), &bound));
                    inputs.push(Input::bytes(
                        format!("pass:{}:a", cell.site()),
                        &color_pass_a(&cell, handles).words(),
                    ));
                    inputs.push(Input::bytes(
                        format!("pass:{}:b", cell.site()),
                        &color_pass_b(&cell, handles).words(),
                    ));
                }
            }
            Cross::Aspects => {
                inputs.push(Input {
                    name: "air:aspects".to_owned(),
                    digest: Digest::sha256(
                        Fixture::new(ASPECT_FIXTURE).text(RUNTIME_TEXT)?.as_bytes(),
                    ),
                });
                inputs.push(Input {
                    name: "air-typed:aspects".to_owned(),
                    digest: Digest::sha256(
                        Fixture::new(ASPECT_FIXTURE).text(METAL_TEXT)?.as_bytes(),
                    ),
                });
                inputs.push(Input::bytes("stencils:a", &stencil_bytes(a_stencil)));
                inputs.push(Input::bytes("stencils:b", &stencil_bytes(b_stencil)));
                inputs.push(Input::bytes("depths:a", &depth_bytes(a_depth)));
                inputs.push(Input::bytes("depths:b", &depth_bytes(b_depth)));
                let mut clears = Vec::new();
                for value in [PRIOR_DEPTH, B_CLEAR_DEPTH] {
                    clears.extend(value.to_bits().to_le_bytes());
                }
                for value in [PRIOR_STENCIL, B_CLEAR_STENCIL] {
                    clears.extend(value.to_le_bytes());
                }
                inputs.push(Input::bytes("clears", &clears));
                inputs.push(Input::bytes("target:depth", &depth_target().words()));
                inputs.push(Input::bytes("target:stencil", &stencil_target().words()));
                inputs.push(Input::bytes(
                    "pipeline",
                    &aspect_pipeline(Handle(0)).words(),
                ));
                inputs.push(Input::bytes("state", &state_words(&aspect_state())));
                for cell in aspect_cells() {
                    let handles = AspectHandles::fingerprint();
                    inputs.push(Input::bytes(
                        format!("pass:{}:prime", cell.site()),
                        &aspect_priming(handles).words(),
                    ));
                    inputs.push(Input::bytes(
                        format!("pass:{}:a", cell.site()),
                        &aspect_pass_a(&cell, handles).words(),
                    ));
                    inputs.push(Input::bytes(
                        format!("pass:{}:b", cell.site()),
                        &aspect_pass_b(&cell, handles).words(),
                    ));
                }
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match self.cross {
            Cross::Color => self.run_color(side, sites),
            Cross::Aspects => self.run_aspects(side, sites),
        }
    }
}

/// The answer every piece of a `render` call is read out of, and the first refusal among them.
///
/// One helper for both cases: each piece's own site holds its refusal or its `Ok`, and a refused
/// piece aborts the case at the piece's name, so a reader sees which pass or which blit refused.
fn pieces(
    sites: &mut Sites,
    names: &[String],
    outcome: &crate::case::PassOutcome,
) -> Result<(), Abort> {
    let mut branch = Node::branch("pieces", Vec::new());
    for (at, what) in names.iter().enumerate() {
        branch = branch.with(Node::leaf(
            what.clone(),
            Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    sites.push(branch);
    match names
        .iter()
        .enumerate()
        .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (what, f)))
    {
        Some((what, failure)) => Err(Abort {
            at: format!("pieces/{what}"),
            failure: failure.clone(),
        }),
        None => Ok(()),
    }
}

impl SequenceActions {
    /// The colour case: one command buffer a cell, each of two passes over one host-written
    /// attachment, and one blit a cell.
    fn run_color(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(COLOR_FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let states = colors();
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &color_pipeline(&states)),
        )?;
        let a_values = made(sites, "buffer:a", side.buffer(&drawn_bytes(a_drawn)))?;
        let b_values = made(sites, "buffer:b", side.buffer(&drawn_bytes(b_drawn)))?;
        let ones = made(sites, "buffer:ones", side.buffer(&ones_bytes()))?;
        let width = made(sites, "buffer:width", side.buffer(&width_bytes()))?;

        let cells = color_cells();
        let mut work = Vec::new();
        let mut names = Vec::new();
        let mut reads = Vec::new();
        for cell in &cells {
            let site = cell.site();
            let target = made(
                sites,
                &format!("target:{site}"),
                side.target(&color_target()),
            )?;
            // The prior value goes in before the command buffer, as a host write of every texel:
            // the one route to a per-texel code no clear and no fragment of this family states.
            call(
                sites,
                &format!("replace-region:{site}"),
                side.replace_region(target, [SIDE; 2], COLOR_TEXEL * SIDE, &host_bytes()),
                |_| Value::Absent,
            )?;
            let into = made(
                sites,
                &format!("buffer:read:{site}"),
                side.zeroed_buffer(COLOR_PLANE as usize),
            )?;
            let handles = ColorHandles {
                target,
                pipeline,
                a_values,
                b_values,
                ones,
                width,
            };
            work.push(RenderWork::Pass(color_pass_a(cell, handles)));
            names.push(format!("pass-a:{site}"));
            work.push(RenderWork::Pass(color_pass_b(cell, handles)));
            names.push(format!("pass-b:{site}"));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into,
                offset: 0,
                row_pitch: COLOR_TEXEL * SIDE,
                size: [SIDE; 2],
                at: ReadAt::BASE,
                plane: Plane::Whole,
            });
            names.push(format!("read:{site}"));
            reads.push((site, into));
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        pieces(sites, &names, &outcome)?;

        let mut planes = Node::branch("planes", Vec::new());
        for (site, from) in &reads {
            let bytes = call(
                sites,
                &format!("read-back/{site}"),
                side.read_buffer(*from, 0, COLOR_PLANE as usize),
                |_| Value::Absent,
            )?;
            planes = planes.with(Node::leaf(
                site.clone(),
                Observed::bytes(bytes, color_layout()),
            ));
        }
        sites.push(planes);
        Ok(())
    }

    /// The aspect case: one command buffer a cell, a priming pass and two passes over a
    /// `Depth32Float` beside a `Stencil8`, and one blit an attachment.
    fn run_aspects(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let triangle = Fixture::new(COLOR_FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let fragment = Fixture::new(ASPECT_FIXTURE).loaded().map_err(|why| Abort {
            at: "library:aspects".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let vertex_library = made(sites, "library", side.library(&triangle.source()))?;
        let fragment_library = made(sites, "library:aspects", side.library(&fragment.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(vertex_library, &aspect_pipeline(fragment_library)),
        )?;
        let state = made(sites, "state", side.depth_stencil_state(&aspect_state()))?;
        let width = made(sites, "buffer:width", side.buffer(&width_bytes()))?;

        let cells = aspect_cells();
        let mut work = Vec::new();
        let mut names = Vec::new();
        let mut reads = Vec::new();
        for cell in &cells {
            let site = cell.site();
            let depth = made(
                sites,
                &format!("target:depth:{site}"),
                side.target(&depth_target()),
            )?;
            let stencil = made(
                sites,
                &format!("target:stencil:{site}"),
                side.target(&stencil_target()),
            )?;
            let depth_into = made(
                sites,
                &format!("buffer:read:depth:{site}"),
                side.zeroed_buffer((4 * TEXELS) as usize),
            )?;
            let stencil_into = made(
                sites,
                &format!("buffer:read:stencil:{site}"),
                side.zeroed_buffer(TEXELS as usize),
            )?;
            // Each pass's own value buffers: pass A's and pass B's differ at every texel, so a
            // plane says which pass wrote it.
            let a_stencils = made(
                sites,
                &format!("buffer:stencils:a:{site}"),
                side.buffer(&stencil_bytes(a_stencil)),
            )?;
            let b_stencils = made(
                sites,
                &format!("buffer:stencils:b:{site}"),
                side.buffer(&stencil_bytes(b_stencil)),
            )?;
            let a_depths = made(
                sites,
                &format!("buffer:depths:a:{site}"),
                side.buffer(&depth_bytes(a_depth)),
            )?;
            let b_depths = made(
                sites,
                &format!("buffer:depths:b:{site}"),
                side.buffer(&depth_bytes(b_depth)),
            )?;
            let base = AspectHandles {
                depth,
                stencil,
                pipeline,
                state,
                stencils: a_stencils,
                depths: a_depths,
                width,
            };
            work.push(RenderWork::Pass(aspect_priming(base)));
            names.push(format!("prime:{site}"));
            work.push(RenderWork::Pass(aspect_pass_a(cell, base)));
            names.push(format!("pass-a:{site}"));
            work.push(RenderWork::Pass(aspect_pass_b(
                cell,
                AspectHandles {
                    stencils: b_stencils,
                    depths: b_depths,
                    ..base
                },
            )));
            names.push(format!("pass-b:{site}"));
            work.push(RenderWork::ReadTexture {
                texture: depth,
                into: depth_into,
                offset: 0,
                row_pitch: 4 * SIDE,
                size: [SIDE; 2],
                at: ReadAt::BASE,
                plane: Plane::Whole,
            });
            names.push(format!("read:depth:{site}"));
            work.push(RenderWork::ReadTexture {
                texture: stencil,
                into: stencil_into,
                offset: 0,
                row_pitch: SIDE,
                size: [SIDE; 2],
                at: ReadAt::BASE,
                plane: Plane::Whole,
            });
            names.push(format!("read:stencil:{site}"));
            reads.push((site, depth_into, stencil_into));
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        pieces(sites, &names, &outcome)?;

        let mut planes = Node::branch("planes", Vec::new());
        for (site, depth_into, stencil_into) in &reads {
            let depth_bytes_read = call(
                sites,
                &format!("read-back/depth/{site}"),
                side.read_buffer(*depth_into, 0, (4 * TEXELS) as usize),
                |_| Value::Absent,
            )?;
            let stencil_bytes_read = call(
                sites,
                &format!("read-back/stencil/{site}"),
                side.read_buffer(*stencil_into, 0, TEXELS as usize),
                |_| Value::Absent,
            )?;
            planes = planes.with(
                Node::branch(site.clone(), Vec::new())
                    .with(Node::leaf(
                        "depth",
                        Observed::bytes(
                            depth_bytes_read,
                            plane_layout(Aspect::Depth, Element::F32, 4),
                        ),
                    ))
                    .with(Node::leaf(
                        "stencil",
                        Observed::bytes(
                            stencil_bytes_read,
                            plane_layout(Aspect::Stencil, Element::U8, 1),
                        ),
                    )),
            );
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

    /// Whether `rect` holds the texel `(x, y)`.
    fn inside(rect: ScissorBox, x: u32, y: u32) -> bool {
        u64::from(x) >= rect.x
            && u64::from(x) < rect.x + rect.width
            && u64::from(y) >= rect.y
            && u64::from(y) < rect.y + rect.height
    }

    /// Whether the texel `(x, y)` is inside the extent `reach` from the origin.
    fn reached(reach: [u32; 2], x: u32, y: u32) -> bool {
        x < reach[0] && y < reach[1]
    }

    impl ColorCell {
        /// What the colour attachment's texel `(x, y)` holds when the command buffer is done, as
        /// Metal's rules give it rather than as a recording holds it.
        ///
        /// Read in the order the hardware takes: what pass A left, what its store action kept of that,
        /// and what pass B's load and draw then did.
        fn expected(&self, x: u32, y: u32) -> f32 {
            let at = y * SIDE + x;
            // Pass A: its fragment where its draw reached, else its load action's answer. Its render
            // area is the whole attachment, so a `Clear` reaches every texel.
            let after_a = match (inside(A_RECT, x, y), self.a_load) {
                (true, _) => a_drawn(at),
                (false, LoadAction::Load) => host(at),
                (false, _) => A_CLEAR as f32,
            };
            // Pass A's store action: `Store` keeps what the pass made of the attachment, and
            // `DontCare` keeps the texels it held **before** the pass -- so the clear and the draw
            // are both gone, which is the one answer this cell is for.
            let stored = match self.a_store {
                StoreAction::Store => after_a,
                _ => host(at),
            };
            // Pass B: its fragment where its draw reached, else its load action's answer -- and a
            // `Clear` reaches only the render area's own tiles, so a texel past them keeps what pass
            // A left there.
            match (inside(B_RECT, x, y), self.b_load) {
                (true, _) => b_drawn(at),
                (false, LoadAction::Load) => stored,
                (false, _) => match reached(self.b_reach(), x, y) {
                    true => B_CLEAR as f32,
                    false => stored,
                },
            }
        }
    }

    impl AspectCell {
        /// What the depth attachment's texel `(x, y)` holds when the command buffer is done.
        fn expected_depth(&self, x: u32, y: u32) -> f32 {
            let at = y * SIDE + x;
            let after_a = match inside(A_RECT, x, y) {
                true => a_depth(at),
                false => PRIOR_DEPTH as f32,
            };
            let stored = match self.a_stores[0] {
                StoreAction::Store => after_a,
                _ => PRIOR_DEPTH as f32,
            };
            match (inside(B_RECT, x, y), self.b_loads[0]) {
                (true, _) => b_depth(at),
                (false, LoadAction::Load) => stored,
                (false, _) => B_CLEAR_DEPTH as f32,
            }
        }

        /// What the stencil attachment's texel `(x, y)` holds when the command buffer is done.
        fn expected_stencil(&self, x: u32, y: u32) -> u8 {
            let at = y * SIDE + x;
            let after_a = match inside(A_RECT, x, y) {
                true => a_stencil(at),
                false => PRIOR_STENCIL,
            };
            let stored = match self.a_stores[1] {
                StoreAction::Store => after_a,
                _ => PRIOR_STENCIL,
            };
            let held = match (inside(B_RECT, x, y), self.b_loads[1]) {
                (true, _) => b_stencil(at),
                (false, LoadAction::Load) => stored,
                (false, _) => B_CLEAR_STENCIL,
            };
            held as u8
        }
    }

    /// The one outcome of `case`'s recording, where the tree holds one.
    ///
    /// **Eight runs folding into one outcome is part of the claim here**, not a detail: a
    /// `DontCare` store is the one action Metal's own documentation leaves to the GPU, so a case
    /// whose answer moved between runs would be the finding and not an answer.
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

    /// The bytes of a `planes/<site>` leaf, or of its `<aspect>` child where it has one.
    fn recorded(case: &str, site: &str, aspect: Option<&str>) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript.root.child("planes").and_then(|node| {
            let held = node.child(site)?;
            match aspect {
                None => Some(held.clone()),
                Some(name) => held.child(name).cloned(),
            }
        })?;
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: planes/{site} holds no bytes");
        };
        Some(bytes.clone())
    }

    /// The `float` the texel at `at` of a read-back plane holds.
    fn word(bytes: &[u8], at: u32) -> f32 {
        let start = (at * 4) as usize;
        let mut held = [0u8; 4];
        held.copy_from_slice(&bytes[start..start + 4]);
        f32::from_le_bytes(held)
    }

    /// **Metal's recorded colour case holds, at every texel of every cell, what pass A's store
    /// action and pass B's load action say together** — the rule the module states, applied here
    /// rather than read back out of the recording, which is what holds the recording to a reason.
    #[test]
    fn metal_held_what_the_sequence_says() {
        let case = "pass-sequence-color-actions";
        for cell in color_cells() {
            let site = cell.site();
            let Some(bytes) = recorded(case, &site, None) else {
                println!("{case}: no recording of {site} in the tree yet");
                return;
            };
            assert_eq!(bytes.len(), COLOR_PLANE as usize);
            for at in 0..TEXELS {
                let (x, y) = (at % SIDE, at / SIDE);
                let held = word(&bytes, at);
                let want = cell.expected(x, y);
                assert_eq!(
                    held.to_bits(),
                    want.to_bits(),
                    "{site}: texel ({x}, {y}) holds {held}, not {want}"
                );
            }
        }
    }

    /// **Metal's recorded aspect case holds each aspect's own actions in its own plane**, which is
    /// the whole content of the asymmetric cells.
    #[test]
    fn metal_held_each_aspects_own_actions() {
        let case = "pass-sequence-aspect-actions";
        for cell in aspect_cells() {
            let site = cell.site();
            let Some(depth) = recorded(case, &site, Some("depth")) else {
                println!("{case}: no recording of {site} in the tree yet");
                return;
            };
            let stencil = recorded(case, &site, Some("stencil")).expect("the stencil plane");
            assert_eq!(depth.len(), (4 * TEXELS) as usize);
            assert_eq!(stencil.len(), TEXELS as usize);
            for at in 0..TEXELS {
                let (x, y) = (at % SIDE, at / SIDE);
                let held = word(&depth, at);
                let want = cell.expected_depth(x, y);
                assert_eq!(
                    held.to_bits(),
                    want.to_bits(),
                    "{site}: the depth at ({x}, {y}) holds {held}, not {want}"
                );
                let held = stencil[at as usize];
                let want = cell.expected_stencil(x, y);
                assert_eq!(
                    held, want,
                    "{site}: the stencil at ({x}, {y}) holds {held:#04x}, not {want:#04x}"
                );
            }
        }
    }

    /// The colour cell's whole plane, as the rule gives it.
    fn color_plane(cell: &ColorCell) -> Vec<u32> {
        (0..TEXELS)
            .map(|at| cell.expected(at % SIDE, at / SIDE).to_bits())
            .collect()
    }

    /// **Every colour value this family states is an exact binary fraction in `[0, 1)`, and no two
    /// of the five roles a texel's value can come from ever collide.**
    ///
    /// The property the whole colour case rests on: a plane is read as the host's write, pass A's
    /// clear, pass A's fragment, pass B's clear and pass B's fragment, and no cell could name
    /// which of them put a texel there if two shared a value.
    #[test]
    fn every_colour_value_is_exact_and_no_two_roles_collide() {
        let exact = |value: f64| {
            assert!((0.0..1.0).contains(&value), "{value} is not inside [0, 1)");
            assert_eq!(
                value * 65536.0,
                (value * 65536.0).round(),
                "{value} is not a multiple of 2^-16"
            );
        };
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for clear in [A_CLEAR, B_CLEAR] {
            exact(clear);
            assert!(
                seen.insert((clear as f32).to_bits()),
                "{clear} is stated by two roles"
            );
        }
        for value in [host as fn(u32) -> f32, a_drawn, b_drawn] {
            for at in 0..TEXELS {
                let held = value(at);
                exact(f64::from(held));
                assert!(
                    seen.insert(held.to_bits()),
                    "{held} at texel {at} is stated by two roles"
                );
            }
        }
        assert_eq!(seen.len(), 2 + 3 * TEXELS as usize);
        assert_eq!(host_bytes().len(), COLOR_PLANE as usize);
        assert_eq!(drawn_bytes(a_drawn).len(), 4 * COLOR_PLANE as usize);
        assert_eq!(ones_bytes().len(), 4 * COLOR_PLANE as usize);
    }

    /// **Every rectangle is inside every render area the cell that draws it states**, which is
    /// what keeps the family off `setScissorRect:` and `setViewport:` past the render target —
    /// the one shape Metal's validation calls a programming error.
    #[test]
    fn every_rectangle_is_inside_its_own_render_area() {
        for rect in [A_RECT, B_RECT] {
            assert!(!rect.past([SIDE; 2]), "{rect:?} is past the attachment");
            assert!(!rect.past([BOUND; 2]), "{rect:?} is past the bound");
        }
        // Pass A's render area is always the attachment's whole extent, so only pass B's bound
        // can clip a rectangle, and only pass B's rectangle is drawn under it.
        for cell in color_cells() {
            if let Some(area) = cell.b_area {
                assert!(
                    !B_RECT.past(area),
                    "{:?}: B_RECT is past {area:?}",
                    cell.site()
                );
            }
        }
    }

    /// **The three regions of a plane are each non-empty and name themselves.**
    #[test]
    fn the_three_regions_are_each_non_empty() {
        let mut counts = [0u32; 3];
        for at in 0..TEXELS {
            let (x, y) = (at % SIDE, at / SIDE);
            counts[match (inside(B_RECT, x, y), inside(A_RECT, x, y)) {
                (true, _) => 0,
                (false, true) => 1,
                (false, false) => 2,
            }] += 1;
        }
        assert_eq!(counts[0], 16 * 32, "B's region");
        assert_eq!(counts[1], 16 * 16, "A's region alone");
        assert_eq!(counts[2], TEXELS - 16 * 32 - 16 * 16, "neither's region");
    }

    /// **Pass A's store action and pass B's load action each move a texel, and the cross is not a
    /// sweep of either** — plus the three equalities this family claims, said as claims.
    #[test]
    fn the_colour_cross_is_not_vacuous_and_its_equalities_hold() {
        let cells = color_cells();
        assert_eq!(cells.len(), 12);
        let names: BTreeSet<String> = cells.iter().map(ColorCell::site).collect();
        assert_eq!(names.len(), cells.len(), "two cells share a site");
        let at = |a_load, a_store, b_load, b_area| {
            let cell = cells
                .iter()
                .find(|cell| {
                    cell.a_load == a_load
                        && cell.a_store == a_store
                        && cell.b_load == b_load
                        && cell.b_area == b_area
                })
                .expect("the cell is in the grid");
            color_plane(cell)
        };
        // Pass A's store action moves the plane under pass B's `Load`, at either of pass A's own
        // load actions: that is the pair's own cell.
        for a_load in LOADS {
            assert_ne!(
                at(a_load, StoreAction::Store, LoadAction::Load, None),
                at(a_load, StoreAction::DontCare, LoadAction::Load, None),
                "pass A's store action moves no texel under pass B's Load"
            );
        }
        // Pass B's load action moves the plane under either of pass A's store actions.
        for a_store in STORES {
            assert_ne!(
                at(LoadAction::Clear, a_store, LoadAction::Load, None),
                at(LoadAction::Clear, a_store, LoadAction::Clear, None),
                "pass B's load action moves no texel"
            );
        }
        // **Claim one.** Over the whole attachment pass B's `Clear` hides pass A entirely: all
        // four cells hold one plane.
        let cleared: BTreeSet<Vec<u32>> = LOADS
            .iter()
            .flat_map(|a_load| STORES.iter().map(move |a_store| (*a_load, *a_store)))
            .map(|(a_load, a_store)| at(a_load, a_store, LoadAction::Clear, None))
            .collect();
        assert_eq!(cleared.len(), 1, "pass B's Clear does not hide pass A");
        // **Claim two.** Under pass A's `DontCare` store its own load action decides nothing: the
        // pass's whole result is discarded.
        assert_eq!(
            at(
                LoadAction::Load,
                StoreAction::DontCare,
                LoadAction::Load,
                None
            ),
            at(
                LoadAction::Clear,
                StoreAction::DontCare,
                LoadAction::Load,
                None
            ),
            "pass A's load action moves a texel through its DontCare store"
        );
        // **Claim three.** Pass B's bound moves not one texel of a `Load`, across the pass
        // boundary as inside one pass.
        for a_store in STORES {
            assert_eq!(
                at(LoadAction::Clear, a_store, LoadAction::Load, None),
                at(
                    LoadAction::Clear,
                    a_store,
                    LoadAction::Load,
                    Some([BOUND; 2])
                ),
                "pass B's bound moves a texel a Load keeps"
            );
        }
        // And the bound's own cell: under pass B's `Clear` the texels past the reach keep what
        // pass A left, so pass A's store action is visible through a `Clear` after all -- which is
        // where claim one stops holding.
        assert_ne!(
            at(
                LoadAction::Clear,
                StoreAction::Store,
                LoadAction::Clear,
                Some([BOUND; 2])
            ),
            at(
                LoadAction::Clear,
                StoreAction::DontCare,
                LoadAction::Clear,
                Some([BOUND; 2])
            ),
            "pass A's store action is invisible past pass B's bound"
        );
        assert_ne!(
            at(
                LoadAction::Clear,
                StoreAction::Store,
                LoadAction::Clear,
                None
            ),
            at(
                LoadAction::Clear,
                StoreAction::Store,
                LoadAction::Clear,
                Some([BOUND; 2])
            ),
            "pass B's bound moves no texel of a Clear"
        );
        // The reach is the bound rounded up to whole tiles and clipped: 32 of a 40-texel
        // attachment leaves 8 texels past it on each axis.
        assert_eq!(reach([BOUND; 2]), [BOUND; 2]);
        assert_eq!(reach([SIDE; 2]), [SIDE; 2]);
        assert_eq!(reach([TILE + 1; 2]), [SIDE; 2]);
    }

    /// **The aspect grid's two aspects move apart**: each aspect's own action moves its own plane
    /// and leaves the other's where it was, which is the whole content of the asymmetry.
    #[test]
    fn each_aspect_moves_its_own_plane_alone() {
        let cells = aspect_cells();
        assert_eq!(cells.len(), ASPECT_STORES.len() * ASPECT_LOADS.len());
        let names: BTreeSet<String> = cells.iter().map(AspectCell::site).collect();
        assert_eq!(names.len(), cells.len(), "two cells share a site");
        let depth = |cell: &AspectCell| -> Vec<u32> {
            (0..TEXELS)
                .map(|at| cell.expected_depth(at % SIDE, at / SIDE).to_bits())
                .collect()
        };
        let stencil = |cell: &AspectCell| -> Vec<u8> {
            (0..TEXELS)
                .map(|at| cell.expected_stencil(at % SIDE, at / SIDE))
                .collect()
        };
        let symmetric = AspectCell {
            a_stores: [StoreAction::Store; 2],
            b_loads: [LoadAction::Load; 2],
        };
        // Pass A's stencil store alone: the stencil plane moves, the depth plane does not.
        let stencil_discarded = AspectCell {
            a_stores: [StoreAction::Store, StoreAction::DontCare],
            ..symmetric
        };
        assert_eq!(depth(&symmetric), depth(&stencil_discarded));
        assert_ne!(stencil(&symmetric), stencil(&stencil_discarded));
        // Pass A's depth store alone: the other way round.
        let depth_discarded = AspectCell {
            a_stores: [StoreAction::DontCare, StoreAction::Store],
            ..symmetric
        };
        assert_ne!(depth(&symmetric), depth(&depth_discarded));
        assert_eq!(stencil(&symmetric), stencil(&depth_discarded));
        // Pass B's stencil load alone, and its depth load alone.
        let stencil_cleared = AspectCell {
            b_loads: [LoadAction::Load, LoadAction::Clear],
            ..symmetric
        };
        assert_eq!(depth(&symmetric), depth(&stencil_cleared));
        assert_ne!(stencil(&symmetric), stencil(&stencil_cleared));
        let depth_cleared = AspectCell {
            b_loads: [LoadAction::Clear, LoadAction::Load],
            ..symmetric
        };
        assert_ne!(depth(&symmetric), depth(&depth_cleared));
        assert_eq!(stencil(&symmetric), stencil(&depth_cleared));
        // Every stencil value is its own role: the prior, pass B's clear, and the two fragments'
        // bands never meet.
        let mut seen: BTreeSet<u8> = BTreeSet::new();
        assert!(seen.insert(PRIOR_STENCIL as u8));
        assert!(seen.insert(B_CLEAR_STENCIL as u8));
        for at in 0..TEXELS {
            for value in [a_stencil(at), b_stencil(at)] {
                assert!(value < 0x100, "{value} is past a stencil byte");
            }
        }
        for band in [0x40u32, 0x80] {
            assert!(
                !(band..band + 0x40).contains(&PRIOR_STENCIL)
                    && !(band..band + 0x40).contains(&B_CLEAR_STENCIL),
                "a clear lies in a fragment's band"
            );
            for at in 0..TEXELS {
                assert!(
                    (0x40..0xc0).contains(&a_stencil(at)) && (0x40..0xc0).contains(&b_stencil(at)),
                    "a fragment writes a stencil byte outside the two bands"
                );
            }
        }
        // And the four depths are four distinct exact values, as the colour case's five are.
        let mut depths: BTreeSet<u32> = BTreeSet::new();
        assert!(depths.insert((PRIOR_DEPTH as f32).to_bits()));
        assert!(depths.insert((B_CLEAR_DEPTH as f32).to_bits()));
        for at in 0..TEXELS {
            for value in [a_depth(at), b_depth(at)] {
                assert!(
                    (0.0..1.0).contains(&value),
                    "{value} is outside the clip volume"
                );
                assert!(
                    depths.insert(value.to_bits()),
                    "{value} is stated by two roles"
                );
            }
        }
        assert_eq!(depths.len(), 2 + 2 * TEXELS as usize);
    }
}
