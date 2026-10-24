//! **Layered and sliced rendering**: which level, slice or depth plane of a render target a pass
//! renders into, how many layers it renders at once, and where a resolving store puts the result.
//!
//! Until this family there was no case anywhere that rendered into anything but level 0, slice 0
//! of a plain 2D texture, because the vocabulary could not say otherwise: a target had no
//! `arrayLength`, `mipmapLevelCount` or `depth`, an attachment had no `level`, `slice` or
//! `depthPlane`, a pass had no `renderTargetArrayLength`, and the read that brings a private
//! target's texels to the host read level 0, slice 0 and nothing else. So a side that ignored
//! every one of those fields passed the whole suite.
//!
//! | case | what it crosses |
//! |---|---|
//! | `layered-attachment-places` | the attachment's `slice` over an array target's three slices, and its `depthPlane` over a 3D target's three planes |
//! | `layered-attachment-levels` | the attachment's `level` over a mip chain's three levels, where the **level's own extent** is the frame the scissor rectangle cuts |
//! | `layered-render-target-array-length` | `renderTargetArrayLength` 0, 1 and 2 by a layered and a non-layered draw, at the attachment's slice 0 and slice 1 |
//! | `layered-resolve-places` | `resolveSlice` and `resolveLevel` past the first, the shape no resolve case could state |
//! | `layered-depth-other-slice` | the depth attachment's slice crossed with the colour attachment's, which is where a side that takes one slice for the whole pass is wrong |
//!
//! # The targets the cases make
//!
//! Four of Metal's texture types, which is what gives the planes a case renders into: an
//! `MTLTextureType2DArray` of three slices, an `MTLTextureType3D` of three planes, an
//! `MTLTextureType2D` of a three-level mip chain, and an `MTLTextureType2DMultisampleArray` of
//! three slices for the resolve the array case needs. The type comes **with** the shape
//! ([`TargetShape`]) rather than beside it, so no case can state a type and a count that
//! disagree, and which field of the attachment then selects a plane -- `slice` for an array,
//! `depthPlane` for a volume -- is what the places case crosses.
//!
//! # Metal's rules, as measured on this M3 Max (macOS 27.0.1, plain Metal)
//!
//! - An attachment's `slice`, `level` and `depthPlane` each select one plane of the target, and the
//!   pass touches **no other** plane: the two planes a cell does not render keep the values an
//!   earlier pass left in them.
//! - A pass at a level past the first has that **level's** extent as its frame: a scissor
//!   rectangle of 2 by 2 at level 1 of an 8 by 8 target covers a quarter of the 4 by 4 level, not a
//!   sixteenth of the level 0 extent.
//! - `renderTargetArrayLength` 0 and 1 both render the attachment's own slice alone, and a vertex
//!   function's `[[render_target_array_index]]` is then **ignored**: two instances naming layers 0
//!   and 1 both land in the attachment's slice and the later one wins.
//! - At 2 or more, the shader's layer `i` reaches slice `slice + i` — **relative to the
//!   attachment's own slice**, not absolute — and the load and store actions cover every one of
//!   those layers, so a layer a draw never reaches still carries the pass's clear.
//! - A layer past the target's last slice is **dropped**, silently: no refusal, no command-buffer
//!   error. It is a ledger row rather than a case, because what the slices it did not reach hold
//!   is whatever the target was made with.
//! - `resolveSlice` and `resolveLevel` put the resolve exactly where they name, leaving the resolve
//!   texture's other slices and levels as they were.
//! - The depth attachment's `slice` is its own: a pass whose colour attachment is at slice 2 and
//!   whose depth attachment is at slice 0 writes both, each at its own slice.
//!
//! # Why a cell's answer cannot be read the wrong way
//!
//! **Every plane the cell does not render is primed, and read back.** A case that renders slice 1
//! and reads slice 1 proves nothing if the reader still reads slice 0 and the two happen to agree,
//! so each plane of each target is first cleared to a **prior value of its own** by a pass of this
//! same command buffer, and every plane is read, not only the rendered one. A side that read the
//! wrong plane answers another plane's prior value, which no other role of this family states.
//!
//! The draw is confined to the **top-left quadrant of the level's own extent**, so every plane
//! holds two regions a reader tells apart: what the fragment wrote, and what the attachment's load
//! action left where no fragment ran. The three roles a texel's value can come from sit in disjoint
//! bands — a prior below `1/256`, the pass's clear at `1/8`, and a fragment's value in
//! `[1/4, 17/64]` — and the module's own test holds them apart.
//!
//! Every value is a multiple of `2^-12`, exact in a `float` and in an `R32Float` texel alike, so no
//! cell's answer turns on a conversion and **nothing here is loosened**. The resolve case's four
//! samples of a pixel all carry one value, so the average every resolve filter computes is that
//! value exactly.
//!
//! # Why this family is safe
//!
//! One command buffer a case: a priming pass per plane, one pass a cell, and one blit per plane
//! read. One or two draws of three vertices a pass, no loop in either stage, every scissor
//! rectangle inside the level it applies to, every vertex inside the clip volume, and no sample
//! count past what the venue answers. A venue that makes no multisampled array texture refuses the
//! target and the suite counts the resolve case [`LacksFeature`](crate::verdict::Verdict::LacksFeature)
//! rather than leaving it uncompared.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    AttachmentPlace, ColorState, PassColor, PassDepth, Plane, ReadAt, RenderPass,
    RenderPipelineSpec, RenderStep, RenderWork, ScissorBox, StageFunction, TargetShape, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan_runtime::StoreAction;

/// The fixture every non-layered draw of this family comes from: the oracle's own fullscreen
/// triangle and its per-sample fills.
const FILL: &str = "rt6_samples";

/// `rt6_fill_vertex`: the fullscreen triangle at depth 0, with no layer output.
const FILL_VERTEX: &str = "rt6_fill_vertex";

/// `rt6_fill_r32float`: `values[(y * width + x) * samples + sample]` as a `float`.
const FILL_FLOAT: &str = "rt6_fill_r32float";

/// `rt6_fill_r32uint`: the same as a `uint`, for the targets the layered draw's own format needs.
const FILL_WORD: &str = "rt6_fill_r32uint";

/// The fixture the layered draws come from.
const LAYERED: &str = "rt6_layered";

/// `rt6_layered_vertex`: the fullscreen triangle once per instance, instance `i` naming layer
/// `layers[i]` through `[[render_target_array_index]]` and tagging its fragments `i + 1`.
const LAYERED_VERTEX: &str = "rt6_layered_vertex";

/// `rt6_layered_fragment`: `tag << 8 | layer`, so a texel names both the instance that wrote it
/// and the layer that instance asked for.
const LAYERED_FRAGMENT: &str = "rt6_layered_fragment";

/// The side of every target's level 0, in texels.
const SIDE: u32 = 8;

/// The slices an array target of this family holds.
const SLICES: u32 = 3;

/// The levels a mip chain of this family holds: 8 by 8, 4 by 4 and 2 by 2.
const LEVELS: u32 = 3;

/// The planes a 3D target of this family holds.
const PLANES: u32 = 3;

/// The slices the layer-count case's target holds: one more than the two layers it renders, so a
/// layer past the pass's own count is told from one inside it.
const RTAL_SLICES: u32 = 4;

/// The samples the resolve case renders at. Four, the count every venue we run answers.
const SAMPLES: u32 = 4;

/// The float target every case but the layer-count one renders into: every value this family
/// states reaches its texel unchanged.
const FLOAT: PixelFormat = PixelFormat::R32Float;

/// The unsigned target the layer-count case renders into, because the layered fixture's fragment
/// returns a `uint`.
const WORDS: PixelFormat = PixelFormat::R32Uint;

/// The depth target of the depth case: the one depth format with no stencil beside it, so its
/// texels are read as plain `float`s.
const DEPTH: PixelFormat = PixelFormat::Depth32Float;

/// How many runs record mode takes a case of this family over: one command buffer, one queue and
/// nothing a device may reorder, so the runs are expected to fold into one outcome and a case whose
/// answer moved between runs is the finding rather than one run being read as the answer.
const RUNS: u32 = 8;

/// The value the fill fragment writes into texel `at` of a plane: `(1024 + at) / 4096`, an exact
/// binary fraction in the drawn band `[1/4, 17/64]`.
fn drawn(at: u32) -> f32 {
    (1024 + at) as f32 / 4096.0
}

/// The value a plane's priming pass clears plane `which` to: `(1 + which) / 4096`, in the priors'
/// band below `1/256`, so a side that read another plane answers a value no other role states.
fn prior(which: u32) -> f64 {
    f64::from(1 + which) / 4096.0
}

/// The value a test pass's own `Clear` load gives the plane it renders: `1/8`, which is no prior
/// and no fragment's value, so a texel the draw did not reach names the pass's clear rather than
/// the plane's prior.
const CLEAR: f64 = 512.0 / 4096.0;

/// The depth a fragment of this family leaves where it ran: the fullscreen triangle's own `z`,
/// which `rt6_fill_vertex` places at 0.
const DRAWN_DEPTH: f64 = 0.0;

/// The depth a test pass's `Clear` load gives the plane it renders: `3/4`, which no prior and no
/// fragment's depth states.
const CLEAR_DEPTH: f64 = 3072.0 / 4096.0;

/// The unsigned value the plain fill fragment writes into texel `at`: `4096 + at`, which no
/// layered fragment's `tag << 8 | layer` and no prior of the layer-count case reaches.
fn drawn_word(at: u32) -> u32 {
    4096 + at
}

/// The unsigned value a plane's priming pass clears slice `which` to in the layer-count case.
fn prior_word(which: u32) -> f64 {
    f64::from(400 + which)
}

/// The unsigned value a test pass's `Clear` load gives every layer it covers, so a layer the pass
/// covered but no draw reached is told from one the pass never covered.
const CLEAR_WORD: f64 = 7.0;

/// The layers the layer-count case's two instances name, which is what its vertex buffer holds.
const LAYERS: [u32; 2] = [0, 1];

/// The extent of level `level` of a target of `size`: each axis halved once a level, as Metal
/// sizes a mip chain.
///
/// It is the **target's own** size and not the family's, because the resolve-level cell's
/// multisampled attachment is the size of the level it resolves into: Metal's validation stops a
/// resolve whose texture and resolve texture differ in size, measured on this M3 Max, so the one
/// shape that renders into a level past the first is an attachment of that level's extent.
fn extent_of(size: [u32; 2], level: u32) -> [u32; 2] {
    TargetShape::level_size(size, level)
}

/// Bytes a whole plane of `extent` takes, one four-byte texel each.
fn plane_bytes(extent: [u32; 2]) -> u32 {
    4 * extent[0] * extent[1]
}

/// The rectangle a cell's draw is confined to: the **top-left quadrant of `extent`**, which is the
/// level's own extent and not the target's, so a plane holds the fragment's values beside what the
/// load action left.
fn quadrant(extent: [u32; 2]) -> ScissorBox {
    ScissorBox::new(
        0,
        0,
        u64::from(extent[0] / 2).max(1),
        u64::from(extent[1] / 2).max(1),
    )
}

/// The bytes of the value buffer the fill fragment reads: one value a (texel, sample) pair in the
/// sample-minor order the fixture indexes, every sample of a texel carrying **one** value so the
/// average a resolve filter computes is that value exactly.
fn drawn_bytes(extent: [u32; 2], samples: u32) -> Vec<u8> {
    (0..extent[0] * extent[1])
        .flat_map(|at| {
            (0..samples)
                .flat_map(move |_| drawn(at).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The same for the unsigned fill fragment.
fn drawn_word_bytes(extent: [u32; 2], samples: u32) -> Vec<u8> {
    (0..extent[0] * extent[1])
        .flat_map(|at| {
            (0..samples)
                .flat_map(move |_| drawn_word(at).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The bytes of the fixture's `Rt6Shape` at fragment buffer 1: the plane's width and its samples,
/// which is how the fill fragment finds its own texel's value.
fn shape_bytes(extent: [u32; 2], samples: u32) -> Vec<u8> {
    let mut bytes = extent[0].to_le_bytes().to_vec();
    bytes.extend(samples.to_le_bytes());
    bytes
}

/// The bytes of the layer buffer the layered vertex function reads at vertex buffer 0.
fn layer_bytes() -> Vec<u8> {
    LAYERS.iter().flat_map(|l| l.to_le_bytes()).collect()
}

/// The layout a plane of `extent` at `at` of a target is read under: one element a texel, so a diff
/// names the texel that moved and the place it was read from.
fn layout(extent: [u32; 2], at: ReadAt, element: Element, aspect: Aspect) -> ByteLayout {
    let row_pitch = 4 * extent[0];
    ByteLayout {
        region: Region::Texture {
            level: at.level,
            slice: at.slice,
            aspect,
            sample: 0,
            origin: [0, 0, at.depth_plane],
            size: [extent[0], extent[1], 1],
        },
        element,
        row_offset: 0,
        row_elements: extent[0],
        row_pitch,
        rows: extent[1],
        image_pitch: row_pitch * extent[1],
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// The draw the layered pass takes: the same triangle once per layer the case names, each instance
/// naming its own layer.
const INSTANCED: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: LAYERS.len() as u32,
};

/// The depth and stencil state the depth case's draw sets: write the depth, test nothing, so the
/// depth a fragment leaves is the fragment's own and no cell's draw depends on another's.
fn writing_state() -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state
}

/// The words a `MTLDepthStencilDescriptor` fingerprints as, as the suite's other depth cases state
/// theirs: the type is `#[non_exhaustive]` and of another crate, so a case cannot destructure it.
fn state_words(state: &DepthStencilDescriptor) -> Vec<u8> {
    let mut bytes = state.depth_compare_function.as_raw().to_le_bytes().to_vec();
    bytes.push(u8::from(state.depth_write_enabled));
    bytes
}

/// Which draw a cell's pass encodes, and so which fixture, which functions and which target format
/// the cell takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Draw {
    /// `rt6_fill_vertex` and `rt6_fill_r32float`: one triangle, no layer output, a `float` a texel.
    Float,
    /// `rt6_fill_vertex` and `rt6_fill_r32uint`: the same draw into an unsigned target, which is
    /// the format the layered fragment's own return type needs.
    Word,
    /// `rt6_layered_vertex` and `rt6_layered_fragment`: the triangle once per layer the case
    /// names, each instance naming its layer through `[[render_target_array_index]]`.
    Layered,
}

impl Draw {
    /// The fixture the draw's two functions come from.
    fn fixture(self) -> &'static str {
        match self {
            Draw::Float | Draw::Word => FILL,
            Draw::Layered => LAYERED,
        }
    }

    /// The vertex function's entry name.
    fn vertex(self) -> &'static str {
        match self {
            Draw::Float | Draw::Word => FILL_VERTEX,
            Draw::Layered => LAYERED_VERTEX,
        }
    }

    /// The fragment function's entry name.
    fn fragment(self) -> &'static str {
        match self {
            Draw::Float => FILL_FLOAT,
            Draw::Word => FILL_WORD,
            Draw::Layered => LAYERED_FRAGMENT,
        }
    }

    /// The target format the draw's fragment returns into.
    fn format(self) -> PixelFormat {
        match self {
            Draw::Float => FLOAT,
            Draw::Word | Draw::Layered => WORDS,
        }
    }

    /// How a plane the draw wrote is read back.
    fn element(self) -> Element {
        match self {
            Draw::Float => Element::F32,
            Draw::Word | Draw::Layered => Element::U32,
        }
    }

    /// `inputPrimitiveTopology`, which Metal **requires** of a pipeline whose vertex function
    /// writes `[[render_target_array_index]]`: without it `newRenderPipelineStateWithDescriptor:`
    /// answers nil with "Vertex shader writes render_target_array_index but inputPrimitiveTopology
    /// is not specified", measured on this M3 Max. Every other draw of this family leaves it at
    /// Metal's `Unspecified`.
    fn topology(self) -> PrimitiveTopologyClass {
        match self {
            Draw::Layered => PrimitiveTopologyClass::Triangle,
            _ => PrimitiveTopologyClass::Unspecified,
        }
    }
}

/// Which of the family's five crosses a case states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cross {
    /// The attachment's `slice` over an array target, and its `depthPlane` over a 3D one.
    Places,
    /// The attachment's `level` over a mip chain, where the level's own extent is the frame.
    Levels,
    /// `renderTargetArrayLength` by the draw's own layer output and by the attachment's slice.
    ArrayLength,
    /// `resolveSlice` and `resolveLevel` past the first.
    Resolves,
    /// The depth attachment's slice crossed with the colour attachment's.
    DepthSlice,
}

impl Cross {
    /// The case's name: its recording's file name and its row in the report.
    fn case_name(self) -> &'static str {
        match self {
            Cross::Places => "layered-attachment-places",
            Cross::Levels => "layered-attachment-levels",
            Cross::ArrayLength => "layered-render-target-array-length",
            Cross::Resolves => "layered-resolve-places",
            Cross::DepthSlice => "layered-depth-other-slice",
        }
    }
}

/// One cell's whole statement: what it makes, where it renders, and which planes it reads back.
#[derive(Clone, Debug)]
struct Cell {
    /// The name the cell's sites take.
    site: String,
    /// The colour target the pass attaches.
    target: TargetSpec,
    /// Where the colour attachment renders in it.
    place: AttachmentPlace,
    /// `renderTargetArrayLength`.
    layers: u32,
    /// Which draw the pass encodes.
    draw: Draw,
    /// The resolve target a resolving store writes, where the cell has one.
    resolve: Option<TargetSpec>,
    /// The depth target and where the depth attachment renders in it, where the cell has one.
    depth: Option<(TargetSpec, AttachmentPlace)>,
}

impl Cell {
    /// The target whose planes the cell primes and reads: the resolve target where the cell
    /// resolves, since a multisampled attachment's own texels are read by neither side, else the
    /// colour target itself.
    fn read_target(&self) -> &TargetSpec {
        self.resolve.as_ref().unwrap_or(&self.target)
    }

    /// Every plane of a target, derived from the **shape it was made with** rather than stated a
    /// second time: a mip chain's levels, an array's slices, a volume's planes, or the one plane a
    /// plain target has.
    fn planes(spec: &TargetSpec) -> Vec<ReadAt> {
        let Some(shape) = &spec.shape else {
            return vec![ReadAt::BASE];
        };
        let mut places = Vec::new();
        for level in 0..shape.levels {
            for slice in 0..shape.array_length {
                for depth_plane in 0..shape.depth {
                    places.push(ReadAt {
                        level,
                        slice,
                        depth_plane,
                    });
                }
            }
        }
        places
    }

    /// The place an attachment takes to render the plane `at` names.
    fn attached(at: ReadAt) -> AttachmentPlace {
        AttachmentPlace {
            level: at.level,
            slice: at.slice,
            depth_plane: at.depth_plane,
            ..AttachmentPlace::BASE
        }
    }

    /// Every plane the cell primes and reads, in order: the read target's, then the depth
    /// target's.
    fn read_planes(&self) -> Vec<(ReadAt, bool)> {
        let mut planes: Vec<(ReadAt, bool)> = Self::planes(self.read_target())
            .into_iter()
            .map(|at| (at, false))
            .collect();
        if let Some((spec, _)) = &self.depth {
            planes.extend(Self::planes(spec).into_iter().map(|at| (at, true)));
        }
        planes
    }

    /// The extent of the plane `at` of whichever target it belongs to: the depth target's or the
    /// read target's own size at that plane's level.
    fn read_extent(&self, at: ReadAt, is_depth: bool) -> [u32; 2] {
        let size = match is_depth {
            true => self.depth.as_ref().map(|(spec, _)| spec.size),
            false => Some(self.read_target().size),
        };
        extent_of(size.unwrap_or([SIDE; 2]), at.level)
    }

    /// The extent the cell's draw runs over: the colour attachment's own level's.
    fn pass_extent(&self) -> [u32; 2] {
        extent_of(self.target.size, self.place.level)
    }

    /// What the cell's priming pass clears plane number `which` to, and how the plane is read: the
    /// unsigned band for an unsigned target, the float band otherwise.
    fn prior_of(&self, which: usize, depth: bool) -> f64 {
        match (depth, self.draw) {
            (false, Draw::Word | Draw::Layered) => prior_word(which as u32),
            _ => prior(which as u32),
        }
    }
}

/// A float target of `shape`, of plain render-target usage.
fn float_target(shape: TargetShape) -> TargetSpec {
    TargetSpec::plain(FLOAT, [SIDE; 2]).shaped(shape)
}

/// Every cell of `cross`, in the order the case runs them: the one list `inputs`, `run` and the
/// tests all read, so no two of them can state a different grid.
fn cells(cross: Cross) -> Vec<Cell> {
    let base = |site: String, target: TargetSpec, place: AttachmentPlace, draw: Draw| Cell {
        site,
        target,
        place,
        layers: 0,
        draw,
        resolve: None,
        depth: None,
    };
    match cross {
        Cross::Places => {
            let mut cells = Vec::new();
            for slice in 0..SLICES {
                cells.push(base(
                    format!("array-slice-{slice}"),
                    float_target(TargetShape::array(SLICES)),
                    AttachmentPlace::slice(slice),
                    Draw::Float,
                ));
            }
            for plane in 0..PLANES {
                cells.push(base(
                    format!("volume-plane-{plane}"),
                    float_target(TargetShape::volume(PLANES)),
                    AttachmentPlace::plane(plane),
                    Draw::Float,
                ));
            }
            cells
        }
        Cross::Levels => (0..LEVELS)
            .map(|level| {
                base(
                    format!("level-{level}"),
                    float_target(TargetShape::chain(LEVELS)),
                    AttachmentPlace::level(level),
                    Draw::Float,
                )
            })
            .collect(),
        Cross::ArrayLength => {
            let mut cells = Vec::new();
            // Every (layer count, draw, attachment slice) the cross states.
            //
            // **A layer count of 0 under the layered draw is not here**, and that is a
            // measurement: Metal's API validation stops it ("the renderPipelineState has a vertex
            // shader that writes to render_target_array_index but renderTargetArrayLength is set
            // to zero"), so it is a shape Metal calls a programming error rather than a rule this
            // suite holds a side to. Plain Metal does run it, and what it leaves is what a count
            // of **1** leaves -- the layer output ignored, both instances in the attachment's own
            // slice, the later one winning -- measured on this M3 Max, one shape per forked
            // process. The count of 1 is validation-clean and is the cell that states that answer.
            for (layers, draw, slice) in [
                (1, Draw::Layered, 0),
                (2, Draw::Layered, 0),
                (2, Draw::Layered, 1),
                (0, Draw::Word, 0),
                (2, Draw::Word, 0),
                (2, Draw::Word, 1),
            ] {
                let which = match draw {
                    Draw::Layered => "layered",
                    _ => "plain",
                };
                cells.push(Cell {
                    layers,
                    ..base(
                        format!("rtal-{layers}/{which}/slice-{slice}"),
                        TargetSpec::plain(WORDS, [SIDE; 2]).shaped(TargetShape::array(RTAL_SLICES)),
                        AttachmentPlace::slice(slice),
                        draw,
                    )
                });
            }
            cells
        }
        Cross::Resolves => vec![
            Cell {
                resolve: Some(float_target(TargetShape::array(SLICES))),
                ..base(
                    "resolve-slice-1".to_owned(),
                    float_target(TargetShape::multisample_array(SLICES)).multisampled(SAMPLES),
                    AttachmentPlace::slice(1).resolving_into(0, 1),
                    Draw::Float,
                )
            },
            Cell {
                resolve: Some(float_target(TargetShape::chain(LEVELS))),
                ..base(
                    "resolve-level-1".to_owned(),
                    // The attachment is the size of **level 1**, not of the resolve target's level
                    // 0: Metal's API validation stops a pass whose texture and resolveTexture
                    // differ in size ("MTLRenderPassDescriptor texture and resolveTexture size
                    // mismatch", measured on this M3 Max), so the one authorable shape that
                    // resolves into a level past the first is an attachment of that level's own
                    // extent.
                    TargetSpec::plain(FLOAT, TargetShape::level_size([SIDE; 2], 1))
                        .multisampled(SAMPLES),
                    AttachmentPlace::BASE.resolving_into(1, 0),
                    Draw::Float,
                )
            },
        ],
        Cross::DepthSlice => [(2, 0), (0, 2), (1, 1)]
            .into_iter()
            .map(|(colour, depth)| Cell {
                depth: Some((
                    TargetSpec::plain(DEPTH, [SIDE; 2]).shaped(TargetShape::array(SLICES)),
                    AttachmentPlace::slice(depth),
                )),
                ..base(
                    format!("color-slice-{colour}+depth-slice-{depth}"),
                    float_target(TargetShape::array(SLICES)),
                    AttachmentPlace::slice(colour),
                    Draw::Float,
                )
            })
            .collect(),
    }
}

/// The handles one cell's program names. A side's handle numbers are its own and none reaches a
/// fingerprint, so [`Handles::fingerprint`] zeroes every one of them.
#[derive(Clone, Debug)]
struct Handles {
    /// The colour target the pass attaches.
    target: Handle,
    /// The resolve target, where the cell resolves.
    resolve: Option<Handle>,
    /// The depth target, where the cell has one.
    depth: Option<Handle>,
    /// The target the cell primes and reads: the resolve target where there is one.
    read: Handle,
    pipeline: Handle,
    /// The depth and stencil state the draw sets, where the cell has a depth attachment.
    state: Option<Handle>,
    /// The per-texel values the fill fragment reads, and the plane's width and samples.
    values: Handle,
    shape: Handle,
    /// The layers the layered vertex function reads.
    layers: Handle,
    /// The buffer each read plane's texels are blitted into, one per plane of
    /// [`Cell::read_planes`].
    reads: Vec<Handle>,
}

impl Handles {
    /// The set a fingerprint is taken over: every handle zeroed, a resolve target, a depth target
    /// and a state present exactly where the cell states one.
    fn fingerprint(cell: &Cell) -> Self {
        let zero = Handle(0);
        Self {
            target: zero,
            resolve: cell.resolve.as_ref().map(|_| zero),
            depth: cell.depth.as_ref().map(|_| zero),
            read: zero,
            pipeline: zero,
            state: cell.depth.as_ref().map(|_| zero),
            values: zero,
            shape: zero,
            layers: zero,
            reads: cell.read_planes().iter().map(|_| zero).collect(),
        }
    }
}

/// The steps a cell's test pass encodes.
fn steps(cell: &Cell, handles: &Handles, extent: [u32; 2]) -> Vec<RenderStep> {
    let mut steps = vec![RenderStep::Pipeline(handles.pipeline)];
    if let Some(state) = handles.state {
        steps.push(RenderStep::DepthStencilState(state));
    }
    match cell.draw {
        Draw::Layered => steps.push(RenderStep::VertexBuffer {
            buffer: handles.layers,
            offset: 0,
            index: 0,
        }),
        _ => {
            steps.push(RenderStep::FragmentBuffer {
                buffer: handles.values,
                offset: 0,
                index: 0,
            });
            steps.push(RenderStep::FragmentBuffer {
                buffer: handles.shape,
                offset: 0,
                index: 1,
            });
        }
    }
    steps.push(RenderStep::Scissor(quadrant(extent)));
    steps.push(match cell.draw {
        Draw::Layered => INSTANCED,
        _ => TRIANGLE,
    });
    steps
}

/// The clear a cell's test pass gives every layer it covers.
fn clear_of(cell: &Cell) -> f64 {
    match cell.draw {
        Draw::Word | Draw::Layered => CLEAR_WORD,
        Draw::Float => CLEAR,
    }
}

/// One cell's whole program: a priming pass per plane it reads, the test pass, and a read of every
/// plane, with the name each piece's answer is emitted under.
fn program(cell: &Cell, handles: &Handles) -> (Vec<RenderWork>, Vec<String>) {
    let mut work = Vec::new();
    let mut names = Vec::new();
    for (which, (at, is_depth)) in cell.read_planes().iter().enumerate() {
        let value = cell.prior_of(which, *is_depth);
        let place = Cell::attached(*at);
        let pass = match is_depth {
            true => RenderPass::clearing(Vec::new()).with_depth(
                PassDepth::cleared(handles.depth.unwrap_or(Handle(0)), value)
                    .stored(StoreAction::Store)
                    .placed(place),
            ),
            false => RenderPass::clearing(vec![PassColor::cleared(
                handles.read,
                [value, 0.0, 0.0, 1.0],
            )
            .stored(StoreAction::Store)
            .placed(place)]),
        };
        work.push(RenderWork::Pass(pass));
        names.push(format!(
            "prime:{}:{}",
            cell.site,
            plane_name(*at, *is_depth)
        ));
    }
    let extent = extent_of(cell.target.size, cell.place.level);
    let mut color = PassColor::cleared(handles.target, [clear_of(cell), 0.0, 0.0, 1.0]);
    color = match handles.resolve {
        Some(resolve) => color.resolving(resolve),
        None => color.stored(StoreAction::Store),
    };
    let mut pass = RenderPass::clearing(vec![color.placed(cell.place)])
        .drawing(steps(cell, handles, extent))
        .layered(cell.layers);
    if let Some((_, place)) = &cell.depth {
        pass = pass.with_depth(
            PassDepth::cleared(handles.depth.unwrap_or(Handle(0)), CLEAR_DEPTH)
                .stored(StoreAction::Store)
                .placed(*place),
        );
    }
    work.push(RenderWork::Pass(pass));
    names.push(format!("pass:{}", cell.site));
    for ((at, is_depth), into) in cell.read_planes().iter().zip(&handles.reads) {
        let extent = cell.read_extent(*at, *is_depth);
        work.push(RenderWork::ReadTexture {
            texture: match is_depth {
                true => handles.depth.unwrap_or(Handle(0)),
                false => handles.read,
            },
            into: *into,
            offset: 0,
            row_pitch: 4 * extent[0],
            size: extent,
            at: *at,
            plane: Plane::Whole,
        });
        names.push(format!("read:{}:{}", cell.site, plane_name(*at, *is_depth)));
    }
    (work, names)
}

/// The name a plane is emitted under: its level, slice and plane, and which attachment it belongs
/// to, so a recording names the place a texel was read from.
fn plane_name(at: ReadAt, is_depth: bool) -> String {
    let what = match is_depth {
        true => "depth",
        false => "color",
    };
    format!(
        "{what}/level-{}/slice-{}/plane-{}",
        at.level, at.slice, at.depth_plane
    )
}

/// The pipeline a cell's draw takes: its two functions, the target's format at colour 0, the depth
/// format where the cell has a depth attachment, the samples, and the topology class the layered
/// vertex function needs.
fn pipeline_spec<'a>(cell: &Cell, states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    let mut spec = RenderPipelineSpec::of(
        StageFunction::named(cell.draw.vertex()),
        Some(StageFunction::named(cell.draw.fragment())),
        states,
    )
    .multisampled(cell.target.samples)
    .topology_class(cell.draw.topology());
    if cell.depth.is_some() {
        spec = spec.depth_stencil(Some(DEPTH), None);
    }
    spec
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    [
        Cross::Places,
        Cross::Levels,
        Cross::ArrayLength,
        Cross::Resolves,
        Cross::DepthSlice,
    ]
    .into_iter()
    .map(|cross| Box::new(LayeredTargets { cross }) as Box<dyn Case>)
    .collect()
}

/// One of the family's five crosses over where a pass renders.
pub struct LayeredTargets {
    cross: Cross,
}

impl LayeredTargets {
    /// The fixtures the case's draws come from, each named once.
    fn fixtures(&self) -> Vec<&'static str> {
        let mut named: Vec<&'static str> = Vec::new();
        for cell in cells(self.cross) {
            let fixture = cell.draw.fixture();
            if !named.contains(&fixture) {
                named.push(fixture);
            }
        }
        named
    }

    /// Every (fixture, vertex, fragment) the case's cells name, each once.
    fn functions(&self) -> Vec<(&'static str, &'static str, &'static str)> {
        let mut named = Vec::new();
        for cell in cells(self.cross) {
            let entry = (
                cell.draw.fixture(),
                cell.draw.vertex(),
                cell.draw.fragment(),
            );
            if !named.contains(&entry) {
                named.push(entry);
            }
        }
        named
    }
}

impl Case for LayeredTargets {
    fn name(&self) -> &'static str {
        self.cross.case_name()
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // **Nothing here is loosened, the resolve case included.** Every texel Metal leaves is one
        // we must leave, and this M3 Max's own run announces no notice at any cell: a
        // `metal2vulkan-degrade-shader-resolve` allowance was declared first and **removed**,
        // because it fired on no venue and an allowance no run exercises justifies nothing. A
        // device that does announce one is a divergence to read, not one to excuse in advance.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut refs = Vec::new();
        for (fixture, vertex, fragment) in self.functions() {
            let held = Fixture::new(fixture);
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                let digest = crate::sha256_hex(held.text(extension)?.as_bytes());
                refs.push(ShaderRef {
                    fixture: format!("{fixture}{extension}"),
                    entry: format!("{vertex}+{fragment}"),
                    source_sha256: digest,
                });
            }
        }
        Ok(refs)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for fixture in self.fixtures() {
            let held = Fixture::new(fixture);
            for (what, extension) in [("air", RUNTIME_TEXT), ("air-typed", METAL_TEXT)] {
                inputs.push(Input {
                    name: format!("{what}:{fixture}"),
                    digest: Digest::sha256(held.text(extension)?.as_bytes()),
                });
            }
        }
        if self.cross == Cross::ArrayLength {
            inputs.push(Input::bytes("layers", &layer_bytes()));
        }
        if self.cross == Cross::DepthSlice {
            inputs.push(Input::bytes("depth-state", &state_words(&writing_state())));
        }
        // Both tables of values a texel can take, by their own bits, so a value that moves by one
        // step is a change to the case's inputs rather than a quiet pass against another grid's
        // answer.
        let mut band = Vec::new();
        for which in 0..RTAL_SLICES + SLICES {
            band.extend(prior(which).to_bits().to_le_bytes());
            band.extend(prior_word(which).to_bits().to_le_bytes());
        }
        for value in [CLEAR, CLEAR_WORD, CLEAR_DEPTH, DRAWN_DEPTH] {
            band.extend(value.to_bits().to_le_bytes());
        }
        inputs.push(Input::bytes("bands", &band));
        for cell in cells(self.cross) {
            let extent = cell.pass_extent();
            let samples = cell.target.samples;
            inputs.push(Input::bytes(
                format!("target:{}", cell.site),
                &cell.target.words(),
            ));
            if let Some(resolve) = &cell.resolve {
                inputs.push(Input::bytes(
                    format!("resolve:{}", cell.site),
                    &resolve.words(),
                ));
            }
            if let Some((spec, place)) = &cell.depth {
                inputs.push(Input::bytes(format!("depth:{}", cell.site), &spec.words()));
                inputs.push(Input::bytes(
                    format!("depth-place:{}", cell.site),
                    &place.words(),
                ));
            }
            inputs.push(Input::bytes(
                format!("place:{}", cell.site),
                &cell.place.words(),
            ));
            let states = [Some(ColorState::new(cell.draw.format()))];
            inputs.push(Input::bytes(
                format!("pipeline:{}", cell.site),
                &pipeline_spec(&cell, &states).words(),
            ));
            inputs.push(Input::bytes(
                format!("values:{}", cell.site),
                &values_bytes(&cell, extent, samples),
            ));
            inputs.push(Input::bytes(
                format!("shape:{}", cell.site),
                &shape_bytes(extent, samples),
            ));
            inputs.push(Input::bytes(
                format!("drawn-rect:{}", cell.site),
                &quadrant(extent).words(),
            ));
            let (work, names) = program(&cell, &Handles::fingerprint(&cell));
            for (piece, what) in work.iter().zip(&names) {
                if let RenderWork::Pass(pass) = piece {
                    inputs.push(Input::bytes(what.clone(), &pass.words()));
                }
                if let RenderWork::ReadTexture { at, size, .. } = piece {
                    let mut bytes = at.words();
                    bytes.extend(size[0].to_le_bytes());
                    bytes.extend(size[1].to_le_bytes());
                    inputs.push(Input::bytes(what.clone(), &bytes));
                }
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // One library per fixture the case's draws come from.
        let mut libraries: Vec<(&'static str, Handle)> = Vec::new();
        for fixture in self.fixtures() {
            let loaded = Fixture::new(fixture).loaded().map_err(|why| Abort {
                at: format!("library:{fixture}"),
                failure: Failure::named("FixtureNotRead", why),
            })?;
            let handle = made(
                sites,
                &format!("library:{fixture}"),
                side.library(&loaded.source()),
            )?;
            libraries.push((fixture, handle));
        }
        let library_of = |fixture: &str| -> Handle {
            libraries
                .iter()
                .find(|(named, _)| *named == fixture)
                .map(|(_, handle)| *handle)
                .unwrap_or(Handle(0))
        };

        let grid = cells(self.cross);
        let mut work = Vec::new();
        let mut names = Vec::new();
        // What each cell reads back: the plane, whether it is the depth attachment's, and the
        // buffer the blit puts it in.
        let mut reads: Vec<Vec<(ReadAt, bool, Handle)>> = Vec::new();
        for cell in &grid {
            let extent = cell.pass_extent();
            let samples = cell.target.samples;
            let states = [Some(ColorState::new(cell.draw.format()))];
            let pipeline = made(
                sites,
                &format!("pipeline:{}", cell.site),
                side.render_pipeline(
                    library_of(cell.draw.fixture()),
                    &pipeline_spec(cell, &states),
                ),
            )?;
            let target = made(
                sites,
                &format!("target:{}", cell.site),
                side.target(&cell.target),
            )?;
            let resolve = match &cell.resolve {
                Some(spec) => Some(made(
                    sites,
                    &format!("resolve:{}", cell.site),
                    side.target(spec),
                )?),
                None => None,
            };
            let depth = match &cell.depth {
                Some((spec, _)) => Some(made(
                    sites,
                    &format!("depth:{}", cell.site),
                    side.target(spec),
                )?),
                None => None,
            };
            let state = match cell.depth.is_some() {
                true => Some(made(
                    sites,
                    &format!("depth-state:{}", cell.site),
                    side.depth_stencil_state(&writing_state()),
                )?),
                false => None,
            };
            let values = made(
                sites,
                &format!("buffer:values:{}", cell.site),
                side.buffer(&values_bytes(cell, extent, samples)),
            )?;
            let shape = made(
                sites,
                &format!("buffer:shape:{}", cell.site),
                side.buffer(&shape_bytes(extent, samples)),
            )?;
            let layers = made(
                sites,
                &format!("buffer:layers:{}", cell.site),
                side.buffer(&layer_bytes()),
            )?;
            let mut cell_reads = Vec::new();
            let mut read_handles = Vec::new();
            for (at, is_depth) in cell.read_planes() {
                let extent = cell.read_extent(at, is_depth);
                let into = made(
                    sites,
                    &format!("buffer:read:{}:{}", cell.site, plane_name(at, is_depth)),
                    side.zeroed_buffer(plane_bytes(extent) as usize),
                )?;
                cell_reads.push((at, is_depth, into));
                read_handles.push(into);
            }
            let handles = Handles {
                target,
                resolve,
                depth,
                read: resolve.unwrap_or(target),
                pipeline,
                state,
                values,
                shape,
                layers,
                reads: read_handles,
            };
            let (pieces, piece_names) = program(cell, &handles);
            work.extend(pieces);
            names.extend(piece_names);
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
        for (cell, cell_reads) in grid.iter().zip(&reads) {
            let mut branch = Node::branch(cell.site.clone(), Vec::new());
            for (at, is_depth, from) in cell_reads {
                let extent = cell.read_extent(*at, *is_depth);
                let bytes = call(
                    sites,
                    &format!("read-back/{}/{}", cell.site, plane_name(*at, *is_depth)),
                    side.read_buffer(*from, 0, plane_bytes(extent) as usize),
                    |_| Value::Absent,
                )?;
                let element = match is_depth {
                    true => Element::F32,
                    false => cell.draw.element(),
                };
                let aspect = match is_depth {
                    true => Aspect::Depth,
                    false => Aspect::Color,
                };
                branch = branch.with(Node::leaf(
                    plane_name(*at, *is_depth),
                    Observed::bytes(bytes, layout(extent, *at, element, aspect)),
                ));
            }
            planes = planes.with(branch);
        }
        sites.push(planes);
        Ok(())
    }
}

/// The bytes of the value buffer a cell's fill fragment reads: the float band for a float target,
/// the unsigned band for an unsigned one.
fn values_bytes(cell: &Cell, extent: [u32; 2], samples: u32) -> Vec<u8> {
    match cell.draw {
        Draw::Word => drawn_word_bytes(extent, samples),
        // The layered fragment reads no values buffer, but the buffer is still made and bound
        // nowhere, so its bytes are the same table the plain unsigned draw states.
        Draw::Layered => drawn_word_bytes(extent, samples),
        Draw::Float => drawn_bytes(extent, samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The value `rt6_layered_fragment` leaves for instance `instance` naming layer `layer`:
    /// `(instance + 1) << 8 | layer`.
    fn layered_word(instance: u32, layer: u32) -> u32 {
        ((instance + 1) << 8) | layer
    }
    /// Whether texel `(x, y)` of a plane of `extent` is inside the drawn quadrant.
    fn inside(extent: [u32; 2], x: u32, y: u32) -> bool {
        let rect = quadrant(extent);
        u64::from(x) < rect.width && u64::from(y) < rect.height
    }

    /// Every cross of the family, so a test states the whole grid rather than one case's.
    const EVERY: [Cross; 5] = [
        Cross::Places,
        Cross::Levels,
        Cross::ArrayLength,
        Cross::Resolves,
        Cross::DepthSlice,
    ];

    /// Which slices of the colour target a cell's pass covers -- what its load and store actions
    /// reach -- which is the attachment's own slice at a layer count of 0 or 1 and the `layers`
    /// slices from it at 2 or more, as measured on this M3 Max.
    fn covered(cell: &Cell) -> Vec<u32> {
        match cell.layers >= 2 {
            true => (0..cell.layers).map(|i| cell.place.slice + i).collect(),
            false => vec![cell.place.slice],
        }
    }

    /// The layer the shader's instance `instance` asks for, and the slice it reaches: relative to
    /// the attachment's own slice at a layer count of 2 or more, and the attachment's slice itself
    /// at 0 or 1, where the layer output is ignored.
    fn reached(cell: &Cell, instance: usize) -> u32 {
        match cell.layers >= 2 {
            true => cell.place.slice + LAYERS[instance],
            false => cell.place.slice,
        }
    }

    /// What plane `at` of the cell's colour or depth target must hold at texel `(x, y)`, by
    /// Metal's rules as this family measured them -- applied here rather than read back from a
    /// recording, which is what lets the recording be held to a reason.
    fn expected(cell: &Cell, at: ReadAt, is_depth: bool, x: u32, y: u32) -> f64 {
        let planes = cell.read_planes();
        let which = planes
            .iter()
            .position(|(place, depth)| *place == at && *depth == is_depth)
            .expect("the plane is one the cell reads");
        let prior = cell.prior_of(which, is_depth);
        if is_depth {
            let (_, place) = cell
                .depth
                .as_ref()
                .expect("a depth plane has a depth target");
            if at.slice != place.slice || at.level != place.level {
                return prior;
            }
            let extent = cell.pass_extent();
            return match inside(extent, x, y) {
                true => DRAWN_DEPTH,
                false => CLEAR_DEPTH,
            };
        }
        // A cell that resolves reads the **resolve** target, whose one written plane is the one
        // `resolveLevel` and `resolveSlice` name; every other plane keeps its prior.
        if cell.resolve.is_some() {
            if at.level != cell.place.resolve_level || at.slice != cell.place.resolve_slice {
                return prior;
            }
            let extent = cell.pass_extent();
            return match inside(extent, x, y) {
                true => f64::from(drawn(y * extent[0] + x)),
                false => clear_of(cell),
            };
        }
        if at.level != cell.place.level || at.depth_plane != cell.place.depth_plane {
            return prior;
        }
        if !covered(cell).contains(&at.slice) {
            return prior;
        }
        let extent = cell.pass_extent();
        if !inside(extent, x, y) {
            return clear_of(cell);
        }
        match cell.draw {
            // The later instance wins where two reach one slice, which is what a layer count of 0
            // or 1 makes them do.
            Draw::Layered => {
                let last = (0..LAYERS.len()).rfind(|instance| reached(cell, *instance) == at.slice);
                match last {
                    Some(instance) => f64::from(layered_word(instance as u32, LAYERS[instance])),
                    // A slice the pass covers that no instance reached keeps the pass's clear.
                    None => clear_of(cell),
                }
            }
            Draw::Word => match at.slice == cell.place.slice {
                true => f64::from(drawn_word(y * extent[0] + x)),
                // A non-layered draw lands in layer 0 alone; the layers past it carry the clear
                // the pass's load action gave them.
                false => clear_of(cell),
            },
            Draw::Float => match at.slice == cell.place.slice {
                true => f64::from(drawn(y * extent[0] + x)),
                false => clear_of(cell),
            },
        }
    }

    /// **Every value this family states is exact, and no two of the roles a texel's value can come
    /// from ever collide.**
    ///
    /// That is the property the whole family rests on: a plane is read as a prior, a clear or a
    /// fragment's value, and a cell could not name which of them put a texel there -- nor which
    /// plane it was read from -- if two of them shared a value.
    #[test]
    fn every_stated_value_is_exact_and_no_two_roles_collide() {
        let exact = |value: f64| {
            // A multiple of 2^-12 is exact in a `float` and in a 32-bit float texel alike, so the
            // round trip through the attachment changes nothing.
            assert_eq!(
                value * 4096.0,
                (value * 4096.0).round(),
                "{value} is not exact"
            );
        };
        let mut floats: BTreeSet<u32> = BTreeSet::new();
        for which in 0..RTAL_SLICES + SLICES {
            let value = prior(which);
            exact(value);
            assert!(
                value < 1.0 / 256.0,
                "the prior {value} is not in the priors' band"
            );
            assert!(
                floats.insert((value as f32).to_bits()),
                "{value} is stated twice"
            );
        }
        exact(CLEAR);
        assert_eq!(CLEAR, 0.125, "the clear left the clears' band");
        assert!(floats.insert((CLEAR as f32).to_bits()));
        for at in 0..SIDE * SIDE {
            let value = drawn(at);
            exact(f64::from(value));
            assert!(
                (0.25..=17.0 / 64.0).contains(&value),
                "the fragment writes {value}, which is not in the drawn band"
            );
            assert!(floats.insert(value.to_bits()), "{value} is stated twice");
        }
        // The depths: the fragment's own, the pass's clear, and every prior, all apart.
        exact(CLEAR_DEPTH);
        exact(DRAWN_DEPTH);
        assert!(floats.insert((CLEAR_DEPTH as f32).to_bits()));
        assert!(floats.insert((DRAWN_DEPTH as f32).to_bits()));
        // The unsigned band: the priors, the clear, the plain draw's values and every value the
        // layered fragment can leave, all distinct.
        let mut words: BTreeSet<u32> = BTreeSet::new();
        for which in 0..RTAL_SLICES + SLICES {
            assert!(
                words.insert(prior_word(which) as u32),
                "a prior word is stated twice"
            );
        }
        assert!(words.insert(CLEAR_WORD as u32), "the clear word is a prior");
        for at in 0..SIDE * SIDE {
            assert!(words.insert(drawn_word(at)), "a drawn word is stated twice");
        }
        for instance in 0..LAYERS.len() as u32 {
            for layer in 0..RTAL_SLICES {
                assert!(
                    words.insert(layered_word(instance, layer)),
                    "the layered fragment's {instance}/{layer} collides with another role"
                );
            }
        }
    }

    /// **Every plane of every target a cell renders into is primed and read**, which is what makes
    /// a cell's answer an honest one: a case that rendered slice 1 and read slice 1 alone would
    /// pass with a reader that still read slice 0.
    #[test]
    fn every_cell_primes_and_reads_every_plane_of_every_target_it_reads() {
        for cross in EVERY {
            for cell in cells(cross) {
                let planes = cell.read_planes();
                let colour: Vec<ReadAt> = Cell::planes(cell.read_target());
                for at in &colour {
                    assert!(
                        planes.contains(&(*at, false)),
                        "{} does not read {at:?} of its colour target",
                        cell.site
                    );
                }
                if let Some((spec, _)) = &cell.depth {
                    for at in Cell::planes(spec) {
                        assert!(
                            planes.contains(&(at, true)),
                            "{} does not read {at:?} of its depth target",
                            cell.site
                        );
                    }
                }
                // One priming pass and one read a plane, and one test pass.
                let (work, _) = program(&cell, &Handles::fingerprint(&cell));
                assert_eq!(
                    work.len(),
                    2 * planes.len() + 1,
                    "{}'s program is not a prime and a read per plane",
                    cell.site
                );
                // The plane the cell renders is one of the planes it reads, so no cell renders
                // somewhere nothing looks.
                let rendered = match cell.resolve.is_some() {
                    true => ReadAt {
                        level: cell.place.resolve_level,
                        slice: cell.place.resolve_slice,
                        depth_plane: cell.place.resolve_depth_plane,
                    },
                    false => ReadAt {
                        level: cell.place.level,
                        slice: cell.place.slice,
                        depth_plane: cell.place.depth_plane,
                    },
                };
                assert!(
                    colour.contains(&rendered),
                    "{} renders {rendered:?}, which it does not read",
                    cell.site
                );
            }
        }
    }

    /// **No two planes of a cell ever hold the same contents**, so a side that read the wrong
    /// plane diverges whichever plane it read.
    ///
    /// This is the read-back's own honesty, stated as the grid's property rather than as a hope:
    /// the prior each plane is primed to is its own, so even two planes a pass never touched hold
    /// different values.
    #[test]
    fn no_two_planes_of_a_cell_hold_the_same_contents() {
        for cross in EVERY {
            for cell in cells(cross) {
                let mut seen: Vec<(Vec<u64>, String)> = Vec::new();
                for (at, is_depth) in cell.read_planes() {
                    let extent = cell.read_extent(at, is_depth);
                    let mut contents: Vec<u64> = Vec::new();
                    for y in 0..extent[1] {
                        for x in 0..extent[0] {
                            contents.push(expected(&cell, at, is_depth, x, y).to_bits());
                        }
                    }
                    if let Some((_, other)) = seen.iter().find(|(held, _)| *held == contents) {
                        panic!(
                            "{}: {} and {other} hold the same contents, so a side that read one \
                             for the other would pass",
                            cell.site,
                            plane_name(at, is_depth)
                        );
                    }
                    seen.push((contents, plane_name(at, is_depth)));
                }
            }
        }
    }

    /// The layer-count cross states **both** of Metal's "the attachment's slice alone" counts and
    /// its layered one, under both draws, and at two attachment slices: the generator's rule that
    /// a crossing needs two values of each axis, held here where a reader can see it.
    #[test]
    fn the_layer_count_cross_states_two_values_of_every_axis() {
        let grid = cells(Cross::ArrayLength);
        let counts: BTreeSet<u32> = grid.iter().map(|cell| cell.layers).collect();
        assert_eq!(counts, BTreeSet::from([0, 1, 2]));
        let draws: BTreeSet<&'static str> = grid
            .iter()
            .map(|cell| match cell.draw {
                Draw::Layered => "layered",
                _ => "plain",
            })
            .collect();
        assert_eq!(draws.len(), 2, "the cross states one draw only");
        let slices: BTreeSet<u32> = grid.iter().map(|cell| cell.place.slice).collect();
        assert_eq!(slices, BTreeSet::from([0, 1]));
        // Every layer either instance reaches is a slice the target holds, so no cell of the case
        // depends on Metal's dropping of a layer past the last slice.
        for cell in &grid {
            for instance in 0..LAYERS.len() {
                assert!(
                    reached(cell, instance) < RTAL_SLICES,
                    "{} asks for a layer past the target's last slice",
                    cell.site
                );
            }
            for slice in covered(cell) {
                assert!(
                    slice < RTAL_SLICES,
                    "{} covers a slice past the target's last",
                    cell.site
                );
            }
        }
    }

    /// Every cell's scissor rectangle lies inside the **level's own** extent, and covers a part of
    /// it rather than all of it, so every plane holds the fragment's values beside what the load
    /// action left.
    #[test]
    fn every_scissor_is_a_part_of_its_own_level() {
        for cross in EVERY {
            for cell in cells(cross) {
                let extent = cell.pass_extent();
                let rect = quadrant(extent);
                assert!(rect.x + rect.width <= u64::from(extent[0]), "{}", cell.site);
                assert!(
                    rect.y + rect.height <= u64::from(extent[1]),
                    "{}",
                    cell.site
                );
                assert!(rect.width >= 1 && rect.height >= 1, "{}", cell.site);
                let whole =
                    rect.width == u64::from(extent[0]) && rect.height == u64::from(extent[1]);
                // The 2 by 2 level's quadrant is a single texel, which is a part of it; nothing
                // bigger may cover a whole level.
                assert!(
                    !whole || extent == [1, 1],
                    "{} draws over the whole level, so its plane holds no loaded region",
                    cell.site
                );
            }
        }
    }

    /// A target that states no shape fingerprints to exactly the bytes it did before a shape could
    /// be stated, and one that states a shape appends its shape's words after them.
    #[test]
    fn a_target_of_no_shape_fingerprints_as_it_always_did() {
        let plain = TargetSpec::plain(FLOAT, [SIDE; 2]);
        assert_eq!(plain.words().len(), 4 + 2 * 4 + 4 + 3);
        let shaped = plain.shaped(TargetShape::array(SLICES));
        assert!(shaped.words().starts_with(&plain.words()));
        assert_eq!(
            shaped.words().len(),
            plain.words().len() + TargetShape::array(SLICES).words().len()
        );
        // An attachment and a pass at Metal's own defaults are the same bytes they always were.
        let colour = PassColor::cleared(Handle(0), [0.0; 4]);
        assert_eq!(colour.words(), colour.placed(AttachmentPlace::BASE).words());
        assert_ne!(
            colour.words(),
            colour.placed(AttachmentPlace::slice(1)).words()
        );
        let pass = RenderPass::clearing(vec![colour]);
        assert_eq!(pass.words(), pass.clone().layered(0).words());
        assert_ne!(pass.words(), pass.clone().layered(2).words());
        // And a read of the first level of the first slice fingerprints as it always did.
        assert!(ReadAt::BASE.words().is_empty());
        assert!(!ReadAt::new(1, 0).words().is_empty());
        assert!(!ReadAt::new(0, 1).words().is_empty());
        assert!(!ReadAt::depth_plane(1).words().is_empty());
    }

    /// Every level of a mip chain of this family has an extent of its own, which is what makes the
    /// level cross's answer a statement about the level's extent and not about a rectangle.
    #[test]
    fn every_level_has_an_extent_of_its_own() {
        let sizes: Vec<[u32; 2]> = (0..LEVELS).map(|l| extent_of([SIDE; 2], l)).collect();
        assert_eq!(sizes, vec![[8, 8], [4, 4], [2, 2]]);
        let apart: BTreeSet<[u32; 2]> = sizes.iter().copied().collect();
        assert_eq!(apart.len(), LEVELS as usize);
    }
}
