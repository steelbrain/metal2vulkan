//! A draw that reads its own pass's color attachment's texture after a memory barrier at the
//! pass's start: what it reads of the pass's clear.
//!
//! | case | what it asks |
//! |---|---|
//! | `pass-start-barrier-{private,over-buffer}-barrier-read` | a pass that clears a 32x32 target and begins with a barrier over the render targets, then a draw that reads the target at its own pixels through a scissor |
//! | `pass-start-barrier-{private,over-buffer}-barrier-sample` | the same draw sampling the target in pixel coordinates |
//! | `pass-start-barrier-{private,over-buffer}-read`, `-sample` | the same two with no barrier |
//! | `pass-start-barrier-{private,over-buffer}-draw-barrier-read` | a solid draw over the target, then the barrier, then the read |
//! | `pass-start-barrier-display-masked-read` | a display-sized compositor target holding an earlier `DontCare` pass's leftovers, cleared, the barrier, and a read at its own pixels through a 453x100 scissor beside an `A8Unorm` mask over a buffer at texture 3 |
//! | `pass-start-barrier-new-texture-chain` | two new 896x576 textures, each cleared, the barrier, and a read at its own pixels through an 841x521 scissor (the second beside the first at texture 3), then a target over a buffer, loaded, the barrier, and a read of the second at texture 3 over it |
//! | `pass-start-barrier-new-texture-tile` | a new 448x576 texture cleared, the barrier, and a read at its own pixels over all of it |
//! | `pass-start-barrier-private-composited` | a compositor's pass twice in a command buffer over a 64x40 target: cleared, the barrier, a sample of the target in pixel coordinates over a quad of a quarter of it, then three blended quads that sample three other textures |
//!
//! Every target is `BGRA8Unorm` with render-target and shader-read usage; a private one is a new
//! texture of the case's, which holds before its first pass what Metal gives a new texture. The
//! functions are the owned fixture `psb_reads`: each read returns the texel's channels reordered
//! as `(b, r, g)` and opaque, or with a mask's alpha, so a draw that read is told from one that
//! did not run. The barrier is `memoryBarrierWithScope:MTLBarrierScopeRenderTargets
//! afterStages:MTLRenderStageFragment beforeStages:MTLRenderStageFragment`, the pass's first
//! command.
//!
//! **What Metal answers, and what this suite holds us to.** The Metal Feature Set Tables give the
//! render-target scope to Mac2 GPUs alone, and the M3 Max this suite records on orders nothing with
//! it: every draw here reads, at its pixels, the texels from before its pass, the barrier or not,
//! a draw before the barrier or not. That is recorded as Metal's answer, and every row compares
//! against it. The rows with a barrier accept one other answer for their texels, the one a Mac2 GPU
//! gives and the runtime orders (`fence.rs`), which the case states from what each draw writes:
//! the clear where the barrier is the pass's first command, and the solid draw's color where that
//! draw came before it. Its justification is `docs/API.md`'s accepted difference
//! [`MAC2_BARRIER`]. The rows with no barrier accept nothing else: a draw reads, at its own pixel,
//! the texels from before the pass on the M3 Max and on the runtime alike
//! (`texture/attached_reads.rs`).
//!
//! A texture's texels are recorded as their runs of equal rows, each row's texels as their runs
//! of equal texels (`fresh_contents::texel_runs`): a lossless encoding whose size follows how often
//! the answer changes down the target and along a row, so a target whose answer changes at a
//! scissor's edges is a few rows of a few runs each, where its runs of texels alone would be two a
//! row (1,042 for each of the chain's targets, 1.46 MB of recording for the three).
//!
//! **What the runtime announces, and the case declares.** Every row with a barrier announces
//! `render-pass-split`: the draw that reads the target records into a Vulkan render pass instance
//! after the one that runs the clear (`texture/attached_reads.rs`). The display row announces two
//! more, from the parts of its shape the compositor gave it: `dont-care-load-tiles` for the earlier
//! pass's `DontCare` load, whose draws cover the whole target before the read's pass clears it, and
//! `shadowed-texture` for the mask over a buffer's bytes. Measured on the Apple M3 Max (MoltenVK)
//! and on lavapipe (LLVM 23.1.1), under the exact policy and the runtime's defaults alike, with
//! every texel and every other field of the transcripts compared. On the RTX 5080 Laptop, which
//! renders a target over a buffer's bytes as a linear image (`VK_NV_linear_color_attachment`),
//! the small rows over a buffer announce `linear-attachment-clear-copies` as well, for a pass that
//! clears it, and every row announces `shared-in-system-memory`; the display row's read takes no
//! split there.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, PlaneTexture, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, OverBuffer, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec,
    RenderStep, RenderWork, ScissorBox, TargetSpec,
};
use crate::replay::notice_allowance;
use crate::transcript::{Digest, Node, Observed, Sites, Value};
use metal2vulkan::metal::{BarrierScope, BlendFactor, PixelFormat, PrimitiveType, RenderStages};
use metal2vulkan_runtime::Degradation;

/// The owned fixture of the rectangle and the reads.
const FIXTURE: &str = "psb_reads";

/// `psb_rect`: a rectangle from a buffer's two clip-space corners.
const RECT: &str = "psb_rect";

/// `psb_solid`, `psb_read`, `psb_sample`, `psb_masked` and `psb_under` (the fixture's own
/// comments).
const SOLID: &str = "psb_solid";
const READ: &str = "psb_read";
const SAMPLE: &str = "psb_sample";
const MASKED: &str = "psb_masked";
const UNDER: &str = "psb_under";

/// The format of every target.
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: usize = 4;

/// The one colour attachment every pipeline carries.
const COLOR_STATE: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The composited row's blended attachment: the source over the destination by the source's
/// alpha. Every source it draws is opaque, so a blend of any precision writes the source's codes.
const BLENDED_STATE: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT).blended(
    Blend::added(BlendFactor::SourceAlpha, BlendFactor::OneMinusSourceAlpha),
))];

/// The slug of `docs/API.md`'s accepted difference the rows with a barrier name.
pub const MAC2_BARRIER: &str =
    "a-barrier-over-the-render-targets-orders-what-it-names-as-a-mac2-gpu-s-does";

/// The runs record mode takes each row over.
const RUNS: u32 = 3;

/// The side of the small rows' targets.
const SMALL: u32 = 32;

/// The small rows' reading scissor: inside the target, off its edges.
const AREA: ScissorBox = ScissorBox::new(4, 4, 20, 12);

/// What the small rows' targets hold before the read's pass: [`BEFORE`] everywhere, and [`BAND`]
/// over their left half, from an earlier pass.
const BEFORE: [f64; 4] = [0.2, 0.4, 0.6, 1.0];
const BAND: [f64; 4] = [0.8, 0.6, 0.2, 1.0];

/// The clear of the small rows' read pass, and the color their solid draw writes. Each channel is
/// a whole code over 255, so a store keeps it exactly.
const CLEAR: [f64; 4] = [200.0 / 255.0, 40.0 / 255.0, 120.0 / 255.0, 160.0 / 255.0];
const DRAWN: [f64; 4] = [30.0 / 255.0, 230.0 / 255.0, 90.0 / 255.0, 1.0];

/// The large rows' clear: transparent black.
const ZERO: [f64; 4] = [0.0; 4];

/// The display row's target, its reading scissor and its earlier pass's leftovers: a solid
/// [`SCRATCH`] over all of it, [`SCRATCH_WIDE`] in its 61x55 corner and [`SCRATCH_SMALL`] in its
/// 45x45 one.
const DISPLAY: [u32; 2] = [1728, 1216];
const DISPLAY_AREA: ScissorBox = ScissorBox::new(0, 0, 453, 100);
const SCRATCH: [f64; 4] = [0.4, 0.4, 0.4, 1.0];
const SCRATCH_WIDE: ScissorBox = ScissorBox::new(0, 0, 61, 55);
const SCRATCH_WIDE_COLOR: [f64; 4] = [1.0, 0.2, 0.0, 1.0];
const SCRATCH_SMALL: ScissorBox = ScissorBox::new(0, 0, 45, 45);
const SCRATCH_SMALL_COLOR: [f64; 4] = [0.0, 0.6, 1.0, 1.0];

/// The display row's mask: `A8Unorm`, [`MASK`] texels, its rows [`MASK_ROW`] bytes apart over a
/// buffer, row `y` all [`mask_code`]`(y)`.
const MASK: [u32; 2] = [64, 16];
const MASK_ROW: u64 = 256;

/// The alpha code of the display row's mask in its row `y`: never zero, so a texel the read
/// wrote is never the clear.
fn mask_code(y: u32) -> u8 {
    (0x40 + 8 * y) as u8
}

/// The chain's and the tile's targets, and the chain's scissor.
const CHAIN: [u32; 2] = [896, 576];
const CHAIN_AREA: ScissorBox = ScissorBox::new(0, 0, 841, 521);
const TILE: [u32; 2] = [448, 576];

/// What the chain's panel, a target over a buffer, holds before its pass, as `BGRA8Unorm` bytes.
const PANEL: [u8; TEXEL] = [0x50, 0x60, 0x70, 0xc0];

/// The composited row's target, the quad of a quarter of it its sample of itself covers, and the
/// compositor's passes over it, each cleared to [`CLEAR`]. Before them an earlier pass leaves it
/// as the small rows' targets are left: [`BEFORE`], and [`BAND`] over its left half.
const COMPOSITED: [u32; 2] = [64, 40];
const QUAD: ScissorBox = ScissorBox::new(16, 10, 32, 20);
const COMPOSITES: usize = 2;

/// The composited row's three other textures, each a target of [`COMPOSITED`]'s size cleared to
/// its color by a pass of its own, and the areas of the blended quads that sample them, drawn in
/// this order after the sample of the target: each overlaps [`QUAD`], and the third the second.
const LAYER_COLORS: [[f64; 4]; 3] = [
    [10.0 / 255.0, 70.0 / 255.0, 250.0 / 255.0, 1.0],
    [250.0 / 255.0, 120.0 / 255.0, 20.0 / 255.0, 1.0],
    [60.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0, 1.0],
];
const LAYERS: [ScissorBox; 3] = [
    ScissorBox::new(0, 0, 24, 16),
    ScissorBox::new(40, 24, 24, 16),
    ScissorBox::new(8, 28, 48, 8),
];

/// The code a store of `channel` keeps.
fn code(channel: f64) -> u8 {
    (channel.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The `BGRA8Unorm` bytes of the color `rgba`.
fn texel(rgba: [f64; 4]) -> [u8; TEXEL] {
    [code(rgba[2]), code(rgba[1]), code(rgba[0]), code(rgba[3])]
}

/// What `psb_read` and `psb_sample` write for the texel `seen` (bytes): its channels as
/// `(b, r, g)`, opaque.
fn reordered([b, g, r, _]: [u8; TEXEL]) -> [u8; TEXEL] {
    [g, r, b, 255]
}

/// What `psb_masked` writes for the texel `seen` beside a mask texel of alpha `alpha`.
fn masked(seen: [u8; TEXEL], alpha: u8) -> [u8; TEXEL] {
    let [b, g, r, _] = reordered(seen);
    [b, g, r, alpha]
}

/// What `psb_under` writes for the mask texel `mask` over the texel `seen`.
fn under(mask: [u8; TEXEL], [.., alpha]: [u8; TEXEL]) -> [u8; TEXEL] {
    let [b, g, r, _] = reordered(mask);
    [b, g, r, alpha]
}

/// The bytes of a `size` target, tight rows, whose texel `(x, y)` is `at(x, y)`.
fn image(size: [u32; 2], at: impl Fn(u32, u32) -> [u8; TEXEL]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size[0] as usize * size[1] as usize * TEXEL);
    for y in 0..size[1] {
        for x in 0..size[0] {
            bytes.extend(at(x, y));
        }
    }
    bytes
}

/// Whether `area` holds the pixel `(x, y)`.
fn inside(area: ScissorBox, x: u32, y: u32) -> bool {
    let (x, y) = (u64::from(x), u64::from(y));
    (area.x..area.x + area.width).contains(&x) && (area.y..area.y + area.height).contains(&y)
}

/// The corners, in clip space, of the whole target: what `psb_rect` takes.
fn whole() -> Vec<u8> {
    [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|edge| edge.to_le_bytes())
        .collect()
}

/// The corners, in clip space, of a `size` target's pixels inside `area`.
fn corners(size: [u32; 2], area: ScissorBox) -> Vec<u8> {
    let x = |p: u64| p as f32 / size[0] as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / size[1] as f32 * 2.0;
    [
        x(area.x),
        y(area.y + area.height),
        x(area.x + area.width),
        y(area.y),
    ]
    .iter()
    .flat_map(|edge| edge.to_le_bytes())
    .collect()
}

/// A color's bytes, as `psb_solid`'s `constant float4 &` takes them.
fn color_bytes(rgba: [f64; 4]) -> Vec<u8> {
    rgba.iter()
        .flat_map(|channel| (*channel as f32).to_le_bytes())
        .collect()
}

/// The draw of a rectangle: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The barrier every row with one begins its read's pass with.
const BARRIER: RenderStep = RenderStep::BarrierScope {
    scope: BarrierScope::RENDER_TARGETS,
    after: RenderStages::FRAGMENT,
    before: RenderStages::FRAGMENT,
};

/// The pipeline of the fragment function `fragment`.
fn pipeline(fragment: &str) -> RenderPipelineSpec<'_> {
    RenderPipelineSpec::new(RECT, fragment, &COLOR_STATE)
}

/// The fragment functions of the fixture, in the order [`Handles::pipelines`] holds them. A row
/// makes the pipelines of those it draws with alone ([`Row::functions`]).
const FUNCTIONS: [&str; 5] = [SOLID, READ, SAMPLE, MASKED, UNDER];

/// The site and input of the composited row's blended [`SAMPLE`] pipeline, which no other row
/// makes.
const BLENDED: &str = "psb_sample-blended";

/// What a small row's read pass encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Small {
    /// The barrier, then a read at the draw's own pixels.
    BarrierRead,
    /// The barrier, then a sample in pixel coordinates.
    BarrierSample,
    /// The read with no barrier.
    Read,
    /// The sample with no barrier.
    Sample,
    /// A solid draw over the target, the barrier, then the read.
    DrawBarrierRead,
}

/// A row of the family.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// A 32x32 target, private or over a buffer, and its read pass.
    Small {
        over_buffer: bool,
        read: Small,
    },
    Display,
    Chain,
    Tile,
    Composited,
}

/// A row: its case's name and its shape.
struct Row {
    case: &'static str,
    shape: Shape,
}

/// Every row.
const ROWS: [Row; 14] = [
    small_row(
        "pass-start-barrier-private-barrier-read",
        false,
        Small::BarrierRead,
    ),
    small_row(
        "pass-start-barrier-private-barrier-sample",
        false,
        Small::BarrierSample,
    ),
    small_row("pass-start-barrier-private-read", false, Small::Read),
    small_row("pass-start-barrier-private-sample", false, Small::Sample),
    small_row(
        "pass-start-barrier-private-draw-barrier-read",
        false,
        Small::DrawBarrierRead,
    ),
    small_row(
        "pass-start-barrier-over-buffer-barrier-read",
        true,
        Small::BarrierRead,
    ),
    small_row(
        "pass-start-barrier-over-buffer-barrier-sample",
        true,
        Small::BarrierSample,
    ),
    small_row("pass-start-barrier-over-buffer-read", true, Small::Read),
    small_row("pass-start-barrier-over-buffer-sample", true, Small::Sample),
    small_row(
        "pass-start-barrier-over-buffer-draw-barrier-read",
        true,
        Small::DrawBarrierRead,
    ),
    Row {
        case: "pass-start-barrier-display-masked-read",
        shape: Shape::Display,
    },
    Row {
        case: "pass-start-barrier-new-texture-chain",
        shape: Shape::Chain,
    },
    Row {
        case: "pass-start-barrier-new-texture-tile",
        shape: Shape::Tile,
    },
    Row {
        case: "pass-start-barrier-private-composited",
        shape: Shape::Composited,
    },
];

const fn small_row(case: &'static str, over_buffer: bool, read: Small) -> Row {
    Row {
        case,
        shape: Shape::Small { over_buffer, read },
    }
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    ROWS.iter()
        .map(|row| Box::new(PassStartRead { row }) as _)
        .collect()
}

/// A case: one row.
pub struct PassStartRead {
    row: &'static Row,
}

/// The handles a row's work names: its pipelines in [`FUNCTIONS`]' order (`Handle(0)` for one the
/// row does not make), and the textures and
/// buffers its shape makes.
struct Handles {
    pipelines: [Handle; 5],
    /// The composited row's [`BLENDED`] pipeline.
    blended: Handle,
    /// The targets the row reads back, in order, and the buffer each is read into.
    targets: Vec<(Handle, Handle)>,
    /// The display row's mask, or the chain's panel's buffer.
    extra: Handle,
}

impl Handles {
    /// Placeholders: the work's shape with no side's handles, for the input fingerprint.
    fn placeholders(targets: usize) -> Self {
        Self {
            pipelines: [Handle(0); 5],
            blended: Handle(0),
            targets: vec![(Handle(0), Handle(0)); targets],
            extra: Handle(0),
        }
    }

    fn pipeline(&self, function: &str) -> Handle {
        let at = FUNCTIONS
            .iter()
            .position(|name| *name == function)
            .expect("a function of the fixture");
        self.pipelines[at]
    }
}

/// A target's spec and size, as a row makes it.
#[derive(Clone)]
struct Made {
    size: [u32; 2],
    /// The bytes of the buffer the target lies over, where it does.
    over: Option<Vec<u8>>,
}

impl Made {
    fn spec(&self, buffer: Handle) -> TargetSpec {
        let spec = TargetSpec::read(FORMAT, self.size);
        match self.over {
            Some(_) => {
                let row = u64::from(self.size[0]) * TEXEL as u64;
                spec.over_buffer(OverBuffer::new(buffer, 0, row, row))
            }
            None => spec,
        }
    }
}

impl Row {
    /// The targets the row makes, in the order it reads them back.
    fn targets(&self) -> Vec<Made> {
        match self.shape {
            Shape::Small { over_buffer, .. } => vec![Made {
                size: [SMALL; 2],
                over: over_buffer.then(|| vec![0u8; (SMALL * SMALL) as usize * TEXEL]),
            }],
            Shape::Display => vec![Made {
                size: DISPLAY,
                over: None,
            }],
            Shape::Chain => vec![
                Made {
                    size: CHAIN,
                    over: None,
                },
                Made {
                    size: CHAIN,
                    over: None,
                },
                Made {
                    size: CHAIN,
                    over: Some(PANEL.repeat((CHAIN[0] * CHAIN[1]) as usize)),
                },
            ],
            Shape::Tile => vec![Made {
                size: TILE,
                over: None,
            }],
            Shape::Composited => vec![
                Made {
                    size: COMPOSITED,
                    over: None,
                };
                1 + LAYERS.len()
            ],
        }
    }

    /// How many of [`Row::targets`], from the first, a read after a barrier renders: those
    /// [`Row::ordered`] states an answer for. The composited row's other textures are read back
    /// too, and hold their clears on every GPU.
    fn barriered_targets(&self) -> usize {
        match self.shape {
            Shape::Composited => 1,
            _ => self.targets().len(),
        }
    }

    /// The fragment functions of [`FUNCTIONS`] the row draws with, in that order: the pipelines it
    /// makes. A pipeline made and never drawn with would cost its creation's prediction for nothing
    /// (`cost::the_creation_prediction`), and no row asks a question of one.
    fn functions(&self) -> Vec<&'static str> {
        let drawn: &[&str] = match self.shape {
            Shape::Small {
                read: Small::BarrierSample | Small::Sample,
                ..
            } => &[SOLID, SAMPLE],
            Shape::Small { .. } => &[SOLID, READ],
            Shape::Display => &[SOLID, MASKED],
            Shape::Chain => &[READ, MASKED, UNDER],
            Shape::Tile => &[READ],
            Shape::Composited => &[SOLID, SAMPLE],
        };
        FUNCTIONS
            .into_iter()
            .filter(|function| drawn.contains(function))
            .collect()
    }

    /// The mask's bytes: row `y` all [`mask_code`]`(y)`, padded to [`MASK_ROW`].
    fn mask_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity((MASK_ROW * u64::from(MASK[1])) as usize);
        for y in 0..MASK[1] {
            let mut row = vec![mask_code(y); MASK[0] as usize];
            row.resize(MASK_ROW as usize, 0);
            bytes.extend(row);
        }
        bytes
    }

    /// The row's work: its passes, then a read of each target into its buffer.
    fn work(&self, h: &Handles) -> Vec<RenderWork> {
        let made = self.targets();
        let target = |at: usize| h.targets[at].0;
        let read = |function: &str, texture: Handle, area: ScissorBox| {
            vec![
                RenderStep::Pipeline(h.pipeline(function)),
                RenderStep::VertexBytes {
                    bytes: whole(),
                    index: 0,
                },
                RenderStep::FragmentTexture { texture, index: 0 },
                RenderStep::Scissor(area),
                RECT_DRAW,
            ]
        };
        let solid = |color: [f64; 4], size: [u32; 2], area: ScissorBox| {
            vec![
                RenderStep::Pipeline(h.pipeline(SOLID)),
                RenderStep::VertexBytes {
                    bytes: corners(size, area),
                    index: 0,
                },
                RenderStep::FragmentBytes {
                    bytes: color_bytes(color),
                    index: 0,
                },
                RenderStep::Scissor(area),
                RECT_DRAW,
            ]
        };
        let mut work = Vec::new();
        match self.shape {
            Shape::Small { read: shape, .. } => {
                let full = ScissorBox::new(0, 0, u64::from(SMALL), u64::from(SMALL));
                let half = ScissorBox::new(0, 0, u64::from(SMALL / 2), u64::from(SMALL));
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target(0), BEFORE)])
                        .drawing(solid(BAND, [SMALL; 2], half)),
                ));
                let mut steps = Vec::new();
                if shape == Small::DrawBarrierRead {
                    steps.extend(solid(DRAWN, [SMALL; 2], full));
                }
                if matches!(
                    shape,
                    Small::BarrierRead | Small::BarrierSample | Small::DrawBarrierRead
                ) {
                    steps.push(BARRIER);
                }
                let function = match shape {
                    Small::BarrierSample | Small::Sample => SAMPLE,
                    _ => READ,
                };
                steps.extend(read(function, target(0), AREA));
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target(0), CLEAR)]).drawing(steps),
                ));
            }
            Shape::Display => {
                let full = ScissorBox::new(0, 0, u64::from(DISPLAY[0]), u64::from(DISPLAY[1]));
                let mut scratch = solid(SCRATCH, DISPLAY, full);
                scratch.extend(solid(SCRATCH_WIDE_COLOR, DISPLAY, SCRATCH_WIDE));
                scratch.extend(solid(SCRATCH_SMALL_COLOR, DISPLAY, SCRATCH_SMALL));
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::discarded(target(0))]).drawing(scratch),
                ));
                let mut steps = vec![BARRIER];
                steps.extend(read(MASKED, target(0), DISPLAY_AREA));
                steps.insert(
                    4,
                    RenderStep::FragmentTexture {
                        texture: h.extra,
                        index: 3,
                    },
                );
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target(0), ZERO)]).drawing(steps),
                ));
            }
            Shape::Chain => {
                let pass =
                    |color: PassColor, function: &str, own: Handle, other: Option<Handle>| {
                        let mut steps = vec![BARRIER];
                        steps.extend(read(function, own, CHAIN_AREA));
                        if let Some(texture) = other {
                            steps.insert(4, RenderStep::FragmentTexture { texture, index: 3 });
                        }
                        RenderWork::Pass(RenderPass::clearing(vec![color]).drawing(steps))
                    };
                work.push(pass(
                    PassColor::cleared(target(0), ZERO),
                    READ,
                    target(0),
                    None,
                ));
                work.push(pass(
                    PassColor::cleared(target(1), ZERO),
                    MASKED,
                    target(1),
                    Some(target(0)),
                ));
                work.push(pass(
                    PassColor::loaded(target(2)),
                    UNDER,
                    target(2),
                    Some(target(1)),
                ));
            }
            Shape::Tile => {
                let full = ScissorBox::new(0, 0, u64::from(TILE[0]), u64::from(TILE[1]));
                let mut steps = vec![BARRIER];
                steps.extend(read(READ, target(0), full));
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target(0), ZERO)]).drawing(steps),
                ));
            }
            Shape::Composited => {
                let quad = |pipeline: Handle, texture: Handle, area: ScissorBox| {
                    vec![
                        RenderStep::Pipeline(pipeline),
                        RenderStep::VertexBytes {
                            bytes: corners(COMPOSITED, area),
                            index: 0,
                        },
                        RenderStep::FragmentTexture { texture, index: 0 },
                        RenderStep::Scissor(area),
                        RECT_DRAW,
                    ]
                };
                for (at, color) in LAYER_COLORS.iter().enumerate() {
                    work.push(RenderWork::Pass(RenderPass::clearing(vec![
                        PassColor::cleared(target(1 + at), *color),
                    ])));
                }
                let half =
                    ScissorBox::new(0, 0, u64::from(COMPOSITED[0] / 2), u64::from(COMPOSITED[1]));
                work.push(RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target(0), BEFORE)])
                        .drawing(solid(BAND, COMPOSITED, half)),
                ));
                for _ in 0..COMPOSITES {
                    let mut steps = vec![BARRIER];
                    steps.extend(quad(h.pipeline(SAMPLE), target(0), QUAD));
                    for (at, area) in LAYERS.iter().enumerate() {
                        steps.extend(quad(h.blended, target(1 + at), *area));
                    }
                    work.push(RenderWork::Pass(
                        RenderPass::clearing(vec![PassColor::cleared(target(0), CLEAR)])
                            .drawing(steps),
                    ));
                }
            }
        }
        for (at, made) in made.iter().enumerate() {
            work.push(RenderWork::ReadTexture {
                texture: h.targets[at].0,
                into: h.targets[at].1,
                offset: 0,
                row_pitch: made.size[0] * TEXEL as u32,
                size: made.size,
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
        }
        work
    }

    /// Whether the row's reads come after a barrier.
    fn barriered(&self) -> bool {
        !matches!(
            self.shape,
            Shape::Small {
                read: Small::Read | Small::Sample,
                ..
            }
        )
    }

    /// What a Mac2 GPU's barrier answers for each of the row's [`Row::barriered_targets`], in
    /// order (the module docs): the clear, or the solid draw's color, where each read draw reads.
    /// `None` for a row with no barrier.
    fn ordered(&self) -> Option<Vec<Vec<u8>>> {
        if !self.barriered() {
            return None;
        }
        let zero = texel(ZERO);
        Some(match self.shape {
            Shape::Small { read, .. } => {
                let seen = match read {
                    Small::DrawBarrierRead => texel(DRAWN),
                    _ => texel(CLEAR),
                };
                vec![image([SMALL; 2], |x, y| match inside(AREA, x, y) {
                    true => reordered(seen),
                    false => seen,
                })]
            }
            Shape::Display => vec![image(DISPLAY, |x, y| match inside(DISPLAY_AREA, x, y) {
                true => masked(zero, mask_code(y % MASK[1])),
                false => zero,
            })],
            Shape::Chain => {
                let first = |x, y| match inside(CHAIN_AREA, x, y) {
                    true => reordered(zero),
                    false => zero,
                };
                let second = |x, y| match inside(CHAIN_AREA, x, y) {
                    true => masked(zero, first(x, y)[3]),
                    false => zero,
                };
                let panel = |x, y| match inside(CHAIN_AREA, x, y) {
                    true => under(second(x, y), PANEL),
                    false => PANEL,
                };
                vec![
                    image(CHAIN, first),
                    image(CHAIN, second),
                    image(CHAIN, panel),
                ]
            }
            Shape::Tile => vec![image(TILE, |_, _| reordered(zero))],
            Shape::Composited => vec![composited(|_, _| reordered(texel(CLEAR)))],
        })
    }
}

/// The composited row's target after its passes, where the sample of itself wrote `sampled(x,
/// y)` at the pixels of [`QUAD`] no later quad covers: [`CLEAR`] outside every quad, and where a
/// blended quad is the last drawn over the pixel, its texture's color as [`SAMPLE`] reorders it.
fn composited(sampled: impl Fn(u32, u32) -> [u8; TEXEL]) -> Vec<u8> {
    image(COMPOSITED, |x, y| {
        match LAYERS.iter().rposition(|area| inside(*area, x, y)) {
            Some(at) => reordered(texel(LAYER_COLORS[at])),
            None if inside(QUAD, x, y) => sampled(x, y),
            None => texel(CLEAR),
        }
    })
}

/// The site a target's texels are recorded at.
fn texels_site(at: usize) -> String {
    format!("texels-{at}")
}

/// A target's texels, `width` to a row, as its runs of equal rows (the module docs): each
/// `{at, rows, texels}`, `at` the run's first row and `texels` the row's runs of equal texels.
fn texel_rows(bytes: &[u8], width: u32) -> Value {
    let row = (width as usize * TEXEL).max(1);
    let mut runs = Vec::new();
    let mut held: Option<(usize, &[u8], u64)> = None;
    for (at, texels) in bytes.chunks(row).enumerate() {
        match held {
            Some((_, last, ref mut rows)) if last == texels => *rows += 1,
            _ => {
                runs.extend(held.map(rows_value));
                held = Some((at, texels, 1));
            }
        }
    }
    runs.extend(held.map(rows_value));
    Value::List(runs)
}

/// One run of [`texel_rows`].
fn rows_value((at, texels, rows): (usize, &[u8], u64)) -> Value {
    Value::fields([
        ("at", Value::Uint(at as u64)),
        ("rows", Value::Uint(rows)),
        ("texels", super::fresh_contents::texel_runs(texels, TEXEL)),
    ])
}

impl Case for PassStartRead {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Each target's texels, where the read follows a barrier, may be the Mac2 answer as well as
        // the M3 Max's, and nothing else: the answer whole, stated by the case (the module docs).
        let targets = self.row.targets();
        let mut allowances: Vec<Allowance> = self
            .row
            .ordered()
            .into_iter()
            .flatten()
            .zip(&targets)
            .enumerate()
            .map(|(at, (bytes, made))| {
                Allowance::new(
                    format!("case/{}", texels_site(at)),
                    Loosening::AnyOf(vec![texel_rows(&bytes, made.size[0])]),
                    Justification::accepted(MAC2_BARRIER),
                )
            })
            .collect();
        // The notices the runtime announces for the row's shape (the module docs). Each loosens
        // the notices site alone, so every texel still compares.
        if self.row.barriered() && !matches!(self.row.shape, Shape::Composited) {
            allowances.push(notice_allowance(Degradation::RenderPassSplit.code()));
        }
        if matches!(self.row.shape, Shape::Display) {
            allowances.push(notice_allowance(Degradation::DontCareLoadTiles.code()));
            allowances.push(notice_allowance(Degradation::ShadowedTexture.code()));
        }
        if matches!(
            self.row.shape,
            Shape::Small {
                over_buffer: true,
                ..
            }
        ) {
            allowances.push(notice_allowance(
                Degradation::LinearAttachmentClearCopies.code(),
            ));
        }
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason): the RTX 5080 Laptop's
        // replay announced it on every row.
        allowances.push(crate::replay::shared_memory_notice());
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(FIXTURE);
        let entry = std::iter::once(RECT)
            .chain(FUNCTIONS)
            .collect::<Vec<_>>()
            .join("+");
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: entry.clone(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: format!("air:{FIXTURE}"),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: format!("air-typed:{FIXTURE}"),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for function in self.row.functions() {
            inputs.push(Input::bytes(
                format!("pipeline:{function}"),
                &pipeline(function).words(),
            ));
        }
        if matches!(self.row.shape, Shape::Composited) {
            inputs.push(Input::bytes(
                format!("pipeline:{BLENDED}"),
                &RenderPipelineSpec::new(RECT, SAMPLE, &BLENDED_STATE).words(),
            ));
        }
        let targets = self.row.targets();
        for (at, made) in targets.iter().enumerate() {
            inputs.push(Input::bytes(
                format!("target:{at}"),
                &made.spec(Handle(0)).words(),
            ));
            if let Some(bytes) = &made.over {
                inputs.push(Input::bytes(format!("buffer:{at}"), bytes));
            }
        }
        if matches!(self.row.shape, Shape::Display) {
            inputs.push(Input::bytes("mask", &Row::mask_bytes()));
        }
        for (at, work) in self
            .row
            .work(&Handles::placeholders(targets.len()))
            .iter()
            .enumerate()
        {
            if let RenderWork::Pass(pass) = work {
                inputs.push(Input::bytes(format!("pass:{at}"), &pass.words()));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: format!("library:{FIXTURE}"),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut pipelines = [Handle(0); 5];
        for (at, function) in FUNCTIONS.iter().enumerate() {
            if !self.row.functions().contains(function) {
                continue;
            }
            pipelines[at] = made(
                sites,
                &format!("pipeline:{function}"),
                side.render_pipeline(library, &pipeline(function)),
            )?;
        }
        let blended = match self.row.shape {
            Shape::Composited => made(
                sites,
                &format!("pipeline:{BLENDED}"),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::new(RECT, SAMPLE, &BLENDED_STATE),
                ),
            )?,
            _ => Handle(0),
        };
        let made_targets = self.row.targets();
        let mut targets = Vec::new();
        let mut panel = None;
        for (at, made_target) in made_targets.iter().enumerate() {
            let buffer = match &made_target.over {
                Some(bytes) => {
                    let buffer = made(sites, &format!("buffer:{at}"), side.buffer(bytes))?;
                    panel = Some(buffer);
                    buffer
                }
                None => Handle(0),
            };
            let target = made(
                sites,
                &format!("target:{at}"),
                side.target(&made_target.spec(buffer)),
            )?;
            let [width, height] = made_target.size;
            let into = made(
                sites,
                &format!("read-back:{at}"),
                side.zeroed_buffer(width as usize * height as usize * TEXEL),
            )?;
            targets.push((target, into));
        }
        let extra = match self.row.shape {
            Shape::Display => {
                let buffer = made(sites, "buffer:mask", side.buffer(&Row::mask_bytes()))?;
                made(
                    sites,
                    "mask",
                    side.plane_texture(
                        buffer,
                        &PlaneTexture {
                            ostype: None,
                            format: PixelFormat::A8Unorm.as_raw(),
                            size: MASK,
                            plane: 0,
                            planes: [(0, MASK_ROW), (0, 0)],
                        },
                    ),
                )?
            }
            _ => panel.unwrap_or(Handle(0)),
        };
        let handles = Handles {
            pipelines,
            blended,
            targets,
            extra,
        };
        let work = self.row.work(&handles);
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        // Each piece's own answer, so a refusal is named at its piece rather than read as a
        // difference in the texels.
        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, piece) in work.iter().enumerate() {
            pieces = pieces.with(Node::leaf(
                format!("{at}-{}", piece.call()),
                Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        sites.push(pieces);
        if let Some((at, failure)) =
            (0..work.len()).find_map(|at| outcome.refused.get(at)?.as_ref().map(|f| (at, f)))
        {
            return Err(Abort {
                at: format!("pieces/{at}"),
                failure: failure.clone(),
            });
        }
        for (at, (made_target, (_, into))) in made_targets.iter().zip(&handles.targets).enumerate()
        {
            let [width, height] = made_target.size;
            let bytes = call(
                sites,
                &format!("read:{at}"),
                side.read_buffer(*into, 0, width as usize * height as usize * TEXEL),
                |_| Value::Absent,
            )?;
            sites.push(Node::leaf(
                texels_site(at),
                Observed::value(texel_rows(&bytes, width)),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_row_has_a_name_of_its_own_and_a_barrier_row_an_answer_of_its_own() {
        let names: BTreeSet<&str> = ROWS.iter().map(|row| row.case).collect();
        assert_eq!(names.len(), ROWS.len());
        assert!(crate::allowance::accepted_differences()
            .iter()
            .any(|slug| slug == MAC2_BARRIER));
        for row in &ROWS {
            let targets = row.targets();
            match row.ordered() {
                Some(answers) => {
                    assert_eq!(answers.len(), row.barriered_targets(), "{}", row.case);
                    for (answer, made) in answers.iter().zip(&targets) {
                        let [width, height] = made.size;
                        assert_eq!(answer.len(), width as usize * height as usize * TEXEL);
                    }
                }
                None => assert!(!row.barriered(), "{}", row.case),
            }
        }
    }

    /// Each row makes the pipelines its work draws with and no other: a pipeline made and never
    /// drawn with costs its creation's prediction for nothing (`Row::functions`).
    #[test]
    fn every_row_makes_the_pipelines_it_draws_with_alone() {
        for row in &ROWS {
            let targets = row.targets().len();
            let handles = Handles {
                pipelines: [1, 2, 3, 4, 5].map(Handle),
                blended: Handle(6),
                targets: (0..targets as u32)
                    .map(|at| (Handle(100 + 2 * at), Handle(101 + 2 * at)))
                    .collect(),
                extra: Handle(99),
            };
            let mut drawn = BTreeSet::new();
            for work in row.work(&handles) {
                let RenderWork::Pass(pass) = work else {
                    continue;
                };
                for step in &pass.steps {
                    if let RenderStep::Pipeline(Handle(at)) = step {
                        drawn.insert(*at);
                    }
                }
            }
            let mut made: BTreeSet<u32> = (FUNCTIONS.iter().zip(1..))
                .filter(|(function, _)| row.functions().contains(function))
                .map(|(_, at)| at)
                .collect();
            if matches!(row.shape, Shape::Composited) {
                made.insert(6);
            }
            assert_eq!(made, drawn, "{}", row.case);
        }
    }

    /// The colors are apart: a read of the clear, of the texels from before the pass and of the
    /// solid draw write three different texels, and a texel a read wrote is never the clear.
    #[test]
    fn a_read_of_each_answer_writes_a_texel_of_its_own() {
        let seen = [
            texel(CLEAR),
            texel(BEFORE),
            texel(BAND),
            texel(DRAWN),
            texel(ZERO),
        ];
        let written: BTreeSet<[u8; TEXEL]> = seen.iter().map(|t| reordered(*t)).collect();
        assert_eq!(written.len(), seen.len());
        for texel in &seen {
            assert_ne!(reordered(*texel), *texel);
        }
        for y in 0..MASK[1] {
            assert_ne!(mask_code(y), 0);
        }
        for color in [CLEAR, DRAWN] {
            for channel in color {
                assert_eq!(f64::from(code(channel)), channel * 255.0);
            }
        }
    }

    /// What the M3 Max's answer is for the composited row (the module docs): each pass's sample
    /// reads the texels from before its pass, so after the second the quad holds the earlier pass's
    /// texels reordered twice.
    fn composited_on_the_m3() -> Vec<u8> {
        composited(|x, _| {
            let before = match x < COMPOSITED[0] / 2 {
                true => texel(BAND),
                false => texel(BEFORE),
            };
            (0..COMPOSITES).fold(before, |seen, _| reordered(seen))
        })
    }

    /// The composited row's answers are apart: every color its quads write differs from the
    /// clear and from each other, both answers have pixels of the sample of the target's own, and
    /// the two answers differ only there.
    #[test]
    fn the_composited_rows_answers_differ_only_where_it_samples_its_target() {
        let written: BTreeSet<[u8; TEXEL]> = LAYER_COLORS
            .iter()
            .map(|color| reordered(texel(*color)))
            .chain([
                texel(CLEAR),
                reordered(texel(CLEAR)),
                reordered(reordered(texel(BAND))),
                reordered(reordered(texel(BEFORE))),
            ])
            .collect();
        assert_eq!(written.len(), LAYER_COLORS.len() + 4);
        let row = ROWS
            .iter()
            .find(|row| matches!(row.shape, Shape::Composited))
            .expect("the composited row");
        let [mac2] = row
            .ordered()
            .expect("a barrier row")
            .try_into()
            .expect("one answer");
        let m3 = composited_on_the_m3();
        let mut apart = 0;
        for (at, (ours, theirs)) in mac2.chunks(TEXEL).zip(m3.chunks(TEXEL)).enumerate() {
            let (x, y) = (at as u32 % COMPOSITED[0], at as u32 / COMPOSITED[0]);
            let sampled = inside(QUAD, x, y) && !LAYERS.iter().any(|area| inside(*area, x, y));
            assert_eq!(ours != theirs, sampled, "({x}, {y})");
            apart += usize::from(sampled);
        }
        assert!(apart > 0);
    }

    /// Metal's recorded answer on the composited row: [`composited_on_the_m3`] for the target, and
    /// its clear for each other texture.
    #[test]
    fn metal_reads_the_texels_from_before_each_pass_on_the_composited_row() {
        let row = ROWS
            .iter()
            .find(|row| matches!(row.shape, Shape::Composited))
            .expect("the composited row");
        let Ok(recording) = crate::recording::read(row.case) else {
            println!("{}: no recording in the tree yet", row.case);
            return;
        };
        let mut want = vec![texel_rows(&composited_on_the_m3(), COMPOSITED[0])];
        for color in LAYER_COLORS {
            want.push(texel_rows(
                &image(COMPOSITED, |_, _| texel(color)),
                COMPOSITED[0],
            ));
        }
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for (at, want) in want.iter().enumerate() {
                    let node = outcome
                        .transcript
                        .root
                        .child(&texels_site(at))
                        .expect("no texels site");
                    assert_eq!(node.observed.value, *want, "{} texels {at}", row.case);
                }
            }
        }
    }

    /// The bytes [`texel_rows`] encodes, `width` texels to a row.
    fn rebuild(value: &Value, width: u32) -> Vec<u8> {
        let Value::List(runs) = value else {
            panic!("the rows are a list");
        };
        let mut bytes = Vec::new();
        let mut rows = 0u64;
        for run in runs {
            let Value::Fields(fields) = run else {
                panic!("a run of rows is fields");
            };
            let word = |name: &str| match fields[name] {
                Value::Uint(word) => word,
                _ => panic!("{name} is a word"),
            };
            assert_eq!(word("at"), rows);
            let Value::List(texel_runs) = &fields["texels"] else {
                panic!("a row's texels are a list");
            };
            let mut row = Vec::new();
            for texel_run in texel_runs {
                let Value::Fields(texel_run) = texel_run else {
                    panic!("a run of texels is fields");
                };
                let (Value::Uint(at), Value::Uint(count), Value::Bytes(texel)) =
                    (&texel_run["at"], &texel_run["texels"], &texel_run["texel"])
                else {
                    panic!("a run of texels is its first, its count and its texel");
                };
                assert_eq!(*at as usize, row.len() / TEXEL);
                for _ in 0..*count {
                    row.extend_from_slice(texel);
                }
            }
            assert!(row.len() <= width as usize * TEXEL);
            for _ in 0..word("rows") {
                bytes.extend_from_slice(&row);
            }
            rows += word("rows");
        }
        bytes
    }

    /// The runs of rows are lossless: every answer a row states, and an image whose every row and
    /// texel differs from its neighbours, rebuild from them; and a run of rows holds rows that
    /// differ from the next run's.
    #[test]
    fn the_rows_rebuild_the_bytes_they_came_from() {
        let mut images: Vec<(Vec<u8>, u32)> = Vec::new();
        for row in &ROWS {
            for (answer, made) in row.ordered().into_iter().flatten().zip(row.targets()) {
                images.push((answer, made.size[0]));
            }
        }
        images.push((image([7, 5], |x, y| [x as u8, y as u8, 3, 4]), 7));
        images.push((Vec::new(), 7));
        for (bytes, width) in &images {
            let rows = texel_rows(bytes, *width);
            assert_eq!(rebuild(&rows, *width), *bytes);
            let Value::List(runs) = rows else {
                panic!("the rows are a list");
            };
            for pair in runs.windows(2) {
                let texels = |value: &Value| match value {
                    Value::Fields(fields) => fields["texels"].clone(),
                    _ => panic!("a run of rows is fields"),
                };
                assert_ne!(texels(&pair[0]), texels(&pair[1]));
            }
        }
        // The chain's answer is a few runs of rows, not two runs of texels a row.
        let chain = ROWS
            .iter()
            .find(|row| matches!(row.shape, Shape::Chain))
            .and_then(Row::ordered)
            .expect("the chain row");
        for answer in &chain {
            let Value::List(runs) = texel_rows(answer, CHAIN[0]) else {
                panic!("the rows are a list");
            };
            assert_eq!(runs.len(), 2);
        }
    }

    /// Metal's recorded answer on the small rows: every read reads, at its pixels, the texels
    /// from before its pass, the barrier or not (the module docs), and every texel outside the
    /// read's scissor is the pass's clear, or its solid draw's color.
    #[test]
    fn metal_reads_the_texels_from_before_the_pass_on_the_small_rows() {
        for row in &ROWS {
            let Shape::Small { read, .. } = row.shape else {
                continue;
            };
            let Ok(recording) = crate::recording::read(row.case) else {
                println!("{}: no recording in the tree yet", row.case);
                continue;
            };
            let outside = match read {
                Small::DrawBarrierRead => texel(DRAWN),
                _ => texel(CLEAR),
            };
            let before = image([SMALL; 2], |x, y| match inside(AREA, x, y) {
                true if x < SMALL / 2 => reordered(texel(BAND)),
                true => reordered(texel(BEFORE)),
                false => outside,
            });
            let want = texel_rows(&before, SMALL);
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    let node = outcome
                        .transcript
                        .root
                        .child(&texels_site(0))
                        .expect("no texels site");
                    assert_eq!(node.observed.value, want, "{}", row.case);
                }
            }
        }
    }
}
