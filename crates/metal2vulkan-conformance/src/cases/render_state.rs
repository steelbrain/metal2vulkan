//! What a render encoder's **state setters** do to the draw that follows them: the boxes that
//! confine a draw, the facing rules that discard its triangles, and the binding calls that move
//! what it reads.
//!
//! `MTLRenderCommandEncoder` is the encoder the suite leans on hardest and the one it touched
//! least: before this family it reached 23 of its 119 items, and most of the 96 it did not are
//! state. A state setter is cheap to call and easy to get wrong in a way no existing case could
//! see, because every render case the suite had set its state once, before its only draw, so none
//! of them could tell a setter that reached the draw from one that reached nothing.
//!
//! Each case here is **one pass of many draws**, every draw confined to a region of its own and
//! every region a value no other region holds, so the target is a record of what each draw saw. A
//! region nobody writes keeps the pass's clear, and that is as much of the answer as a region
//! somebody wrote: a scissor rectangle, a viewport and a cull mode are all claims about where a
//! draw did **not** land, and a case that read only what was written could not read them.
//!
//! | case | what it sets, and what the target then says |
//! |---|---|
//! | `render-state-boxes` | `setScissorRect:`, `setScissorRects:count:` and `setViewports:count:` confine six draws while the bindings move under them: a vertex and a fragment buffer offset, a range of buffers, inline fragment bytes, and a range of fragment textures |
//! | `render-state-facing` | `setCullMode:`, `setFrontFacingWinding:` and `setTriangleFillMode:` over one triangle and its reverse, ten regions |
//! | `render-state-unread-slots` | the vertex texture, sampler and lod-clamped sampler setters at slots the pass's own functions read nothing from, and the draw that follows them |
//!
//! **The expectation is arithmetic rather than a measurement adopted from the run.** The target is
//! `RGBA32Float`, so a texel is the `float4` the fragment function returned: `tob_solid` returns
//! the `float4` its binding holds unchanged and `tob_read` returns the texel of its texture under
//! the pixel. Each case's module test states its region map as a function and holds Metal's
//! recorded bytes to it, so a disagreement is a finding about which state a draw saw rather than a
//! number to copy.
//!
//! `render-state-facing`'s map is stated **without assuming Metal's facing convention**, because
//! the convention is not what the case asks about. It reads one region to learn whether the first
//! triangle is the front-facing one under the winding the first draws set, and holds the other
//! nine to the consequences: flipping the winding flips which of the pair draws, flipping the cull
//! mode flips it back, a cull mode of none draws both, and a fill mode of lines leaves a region no
//! edge of the triangle passes through entirely clear. Each of those is falsifiable whichever way
//! the convention goes.
//!
//! Nothing here can fault a GPU: one pass a case, at most twelve draws of three or six vertices
//! over a 16 by 16 or 4 by 4 target, no loop in the fixture, and every scissor rectangle and
//! viewport inside its target, which `every_box_lies_inside_its_target` holds with no device
//! asked.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side, TextureRole,
    TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, LodClamp, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, ScissorBox, TargetSpec, ViewportRect,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{
    CullMode, PixelFormat, PrimitiveType, SamplerAddressMode, TriangleFillMode, Winding,
};
use metal2vulkan_runtime::Degradation;

/// The format every target here carries: a `float` stored as it is, so the whole expectation is
/// arithmetic rather than a rounding rule.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The side, in texels, of the targets the region maps are drawn over.
const SIDE: u32 = 16;

/// The side of the one-draw target `render-state-unread-slots` uses.
const SMALL: u32 = 4;

/// The clear every pass loads its target with. A region no draw wrote holds it, which is half of
/// what a scissor rectangle, a viewport and a cull mode each say.
const CLEAR: f32 = -7.5;

/// The owned fixture the rectangle, the triangle, the solid colour and the texel read come from.
const FIXTURE: &str = "tob_renders";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners, six vertices.
const RECT: &str = "tob_rect";

/// `tob_triangle`: the triangle of the three points a buffer holds.
const TRIANGLE: &str = "tob_triangle";

/// `tob_solid`: the `float4` a fragment buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// `tob_read`: the texel of fragment texture 0 under the pixel.
const READ: &str = "tob_read";

/// The entry points a case's shader reference names.
const ENTRIES: &str = "tob_rect+tob_triangle+tob_solid+tob_read";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The draw one triangle takes: three vertices, one instance.
const TRIANGLE_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// The rectangle covering the whole target, as `tob_rect` takes one.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// The rectangle covering the target's right half — pixel columns 8 to 16 of a 16-column target.
/// `render-state-boxes` reaches it through `setVertexBufferOffset:atIndex:` alone, so a side that
/// did not move the offset draws the whole target and the clear it left behind says so.
const RIGHT: [f32; 4] = [0.0, -1.0, 1.0, 1.0];

/// How far apart the rectangles lie in the rects buffer.
const RECT_STRIDE: u64 = 16;

/// The two rectangles the rects buffer holds.
fn rects_buffer() -> Vec<u8> {
    [WHOLE, RIGHT]
        .iter()
        .flat_map(|rect| {
            rect.iter()
                .flat_map(|edge| edge.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The clip-space points of the first triangle: the whole clip square and more, so what confines a
/// draw of it is the scissor rectangle the case set and nothing of the geometry.
const FIRST_TRIANGLE: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

/// The same three points in the opposite order, which is the same triangle of the opposite
/// winding.
const SECOND_TRIANGLE: [[f32; 2]; 3] = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]];

/// How far apart the two triangles lie in the points buffer: three `float2`s take 24 bytes and the
/// stride is the next multiple of 16, which is the offset granularity the suite's render cases
/// bind at.
const POINT_STRIDE: u64 = 32;

/// The two triangles' points as one buffer.
fn points_buffer() -> Vec<u8> {
    let mut bytes = Vec::new();
    for triangle in [FIRST_TRIANGLE, SECOND_TRIANGLE] {
        let start = bytes.len();
        for point in triangle {
            bytes.extend(point.iter().flat_map(|axis| axis.to_le_bytes()));
        }
        bytes.resize(start + POINT_STRIDE as usize, 0);
    }
    bytes
}

/// The colours the draws read, each channel an exact binary fraction and every colour its own, so
/// a draw that saw another draw's binding writes a value a module test can name.
const COLORS: [[f32; 4]; 10] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.125, 0.375, 0.625, 0.875],
    [2.5, -1.25, 0.0, 1.0],
    [-0.5, 3.75, 0.5, 0.25],
    [1.5, -2.25, 4.5, -0.75],
    [3.25, 0.125, -3.5, 0.5],
    [-1.75, 2.125, 1.25, 0.75],
    [0.625, -4.5, 2.75, -1.5],
    [4.25, 1.875, -0.25, 3.5],
    [-2.5, 0.75, 3.125, -0.125],
];

/// How far apart the colours lie in the colours buffer.
const COLOR_STRIDE: u64 = 16;

/// The colours as one buffer.
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

/// One colour as the fragment function's `constant float4 &` binding takes it, for the inline
/// form (`setFragmentBytes:length:atIndex:`).
fn color_bytes(color: [f32; 4]) -> Vec<u8> {
    color
        .iter()
        .flat_map(|channel| channel.to_le_bytes())
        .collect()
}

/// What the read texture holds at texel `(x, y)`: values no draw of this module can write, and
/// each texel its own, so a read of the wrong texel is named rather than guessed.
fn read_texel(x: u32, y: u32) -> [f32; 4] {
    [32.0 + x as f32, 64.0 + y as f32, 0.5, 2.0]
}

/// The read texture's bytes at `side` texels a side.
fn read_texels(side: u32) -> Vec<u8> {
    (0..side)
        .flat_map(|y| (0..side).map(move |x| read_texel(x, y)))
        .flat_map(|texel| {
            texel
                .iter()
                .flat_map(|channel| channel.to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// The read texture's spec: shader-read alone, at the target's extent, so the texel read under a
/// pixel is that pixel's own.
fn read_spec(side: u32) -> TextureSpec {
    TextureSpec::new(FORMAT, [side; 2], TEXEL * side, TextureRole::Read)
}

/// The target a case draws into: a private render target of [`FORMAT`] and `side` texels, of
/// `MTLTextureUsageRenderTarget` alone.
fn target_spec(side: u32) -> TargetSpec {
    TargetSpec::plain(FORMAT, [side; 2])
}

/// Bytes the whole of a `side` by `side` level 0 takes.
fn plane(side: u32) -> u32 {
    TEXEL * side * side
}

/// The layout a target's whole level 0 is read under: one row of the layout a row of texels, one
/// element a channel, and the bytes between the rows compared.
fn layout(side: u32) -> ByteLayout {
    let row_pitch = TEXEL * side;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [side, side, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * side,
        row_pitch,
        rows: side,
        image_pitch: row_pitch * side,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The box of the 4 by 4 grid of 4 by 4 cells at column `c`, row `r`: how the facing map lays its
/// draws out over a 16 by 16 target.
const fn cell(c: u32, r: u32) -> ScissorBox {
    ScissorBox::new((4 * c) as u64, (4 * r) as u64, 4, 4)
}

/// The whole target as a scissor rectangle, which is Metal's own default.
const fn whole_box(side: u32) -> ScissorBox {
    ScissorBox::new(0, 0, side as u64, side as u64)
}

/// The whole target as a viewport, which is Metal's own default.
fn whole_viewport(side: u32) -> ViewportRect {
    ViewportRect::new([0.0, 0.0], [f64::from(side), f64::from(side)])
}

/// The shader reference both texts of the one fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs the fixture's own texts are.
fn fixture_inputs() -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of the module's one fixture.
fn library(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// Run `work` as one command buffer, emit what each of its two pieces answered, and read the
/// target's texels back out of `read_back`.
///
/// One helper, so the sites it emits are named once for the whole module and no sibling of them
/// shares a name.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    work: &[RenderWork],
    read_back: Handle,
    extent: u32,
) -> Result<(), Abort> {
    let outcome = call(sites, "render", side.render(work), |outcome| {
        outcome.status.clone()
    })?;
    let names = ["pass", "read"];
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
        side.read_buffer(read_back, 0, plane(extent) as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout(extent)))),
    );
    Ok(())
}

/// The piece that copies a target's level 0 into a buffer the host reads.
fn read_target(texture: Handle, into: Handle, side: u32) -> RenderWork {
    RenderWork::ReadTexture {
        texture,
        into,
        offset: 0,
        row_pitch: TEXEL * side,
        size: [side; 2],
        plane: Plane::Whole,
        at: ReadAt::BASE,
    }
}

/// How many runs record mode takes a case of this module over.
///
/// Eight rather than the suite's three. Nothing here is ordered by anything a device may reorder —
/// one pass, one queue, no visibility counter, nothing touching sample or tile order — so the runs
/// are expected to fold into one outcome, and the point of asking for more is that the recording
/// then says so on evidence rather than on the expectation. A state setter whose effect moved
/// between runs would be the finding, and three runs are too few to call it.
const RUNS: u32 = 8;

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(Boxes), Box::new(Facing), Box::new(UnreadSlots)]
}

// ------------------------------------------------------------------------------------------------
// render-state-boxes

/// The six scissor rectangles and viewports the boxes case confines its draws with, in draw order.
///
/// Each is inside the target, which `every_box_lies_inside_its_target` holds: a rectangle reaching
/// past the attachments is the one shape of `setScissorRect:` Metal's own validation calls a
/// programming error, so no case here may state one.
const A1: ScissorBox = ScissorBox::new(0, 0, 4, 8);
const A3: ScissorBox = ScissorBox::new(4, 0, 8, 8);
const A4: ScissorBox = ScissorBox::new(12, 0, 4, 8);
const B1: ScissorBox = ScissorBox::new(4, 8, 4, 4);
/// The second rectangle of the pair `setScissorRects:count:` sets. No draw may reach it: a draw
/// reads the rectangle its `[[viewport_array_index]]` names and the case's vertex function writes
/// none, so index 0 applies and this one is the control.
const B1_UNREAD: ScissorBox = ScissorBox::new(0, 8, 4, 4);
const B4: ScissorBox = ScissorBox::new(12, 8, 4, 8);

/// The viewport the fifth draw aims at: two pixel columns and two rows at (8, 12).
const B3: ViewportRect = ViewportRect {
    origin: [8.0, 12.0],
    size: [2.0, 2.0],
    depth: [0.0, 1.0],
};

/// The second viewport of the pair `setViewports:count:` sets, which no draw may reach for the
/// reason [`B1_UNREAD`] is unreachable.
const B3_UNREAD: ViewportRect = ViewportRect {
    origin: [0.0, 12.0],
    size: [4.0, 4.0],
    depth: [0.0, 1.0],
};

/// Where the boxes case's draws read their colours from.
const BOX_COLORS: [usize; 5] = [0, 1, 2, 3, 4];

/// `setScissorRect:`, `setScissorRects:count:` and `setViewports:count:` confining six draws of
/// one pass, while the bindings move under them.
pub struct Boxes;

impl Boxes {
    /// The two pipelines the pass sets: the solid colour, and the texel read.
    fn pipelines(colors: &[Option<ColorState>]) -> [RenderPipelineSpec<'_>; 2] {
        [
            RenderPipelineSpec::new(RECT, SOLID, colors),
            RenderPipelineSpec::new(RECT, READ, colors),
        ]
    }

    /// The pass's steps, in the order the case encodes them.
    fn steps(
        solid: Handle,
        reading: Handle,
        rects: Handle,
        colors: Handle,
        read: Handle,
    ) -> Vec<RenderStep> {
        vec![
            // Draw 1: the bindings arrive as ranges, and one scissor rectangle confines the
            // whole-target rectangle to four columns of the top half.
            RenderStep::Pipeline(solid),
            RenderStep::VertexBuffers {
                buffers: vec![(rects, 0)],
                first: 0,
            },
            RenderStep::FragmentBuffers {
                buffers: vec![(colors, 0)],
                first: 0,
            },
            RenderStep::Scissor(A1),
            RECT_DRAW,
            // Draw 2: both offsets move and nothing else. The vertex offset reaches the right-half
            // rectangle, so the four columns of this draw's scissor rectangle that its geometry
            // does not cover keep the clear -- which is what says the offset reached the draw.
            RenderStep::VertexBufferOffset {
                offset: RECT_STRIDE,
                index: 0,
            },
            RenderStep::FragmentBufferOffset {
                offset: COLOR_STRIDE,
                index: 0,
            },
            RenderStep::Scissor(A3),
            RECT_DRAW,
            // Draw 3: the rectangle goes back through the range form, and the colour arrives as
            // inline bytes in place of the buffer binding.
            RenderStep::VertexBuffers {
                buffers: vec![(rects, 0)],
                first: 0,
            },
            RenderStep::FragmentBytes {
                bytes: color_bytes(COLORS[BOX_COLORS[2]]),
                index: 0,
            },
            RenderStep::Scissor(A4),
            RECT_DRAW,
            // Draw 4: two scissor rectangles, of which only the first applies, and the colour
            // binding comes back as a range at an offset of its own.
            RenderStep::Scissors(vec![B1, B1_UNREAD]),
            RenderStep::FragmentBuffers {
                buffers: vec![(colors, 3 * COLOR_STRIDE)],
                first: 0,
            },
            RECT_DRAW,
            // Draw 5: two viewports, of which only the first applies, with the scissor rectangle
            // opened back to the whole target so the viewport is the only thing confining it.
            RenderStep::Viewports(vec![B3, B3_UNREAD]),
            RenderStep::Scissor(whole_box(SIDE)),
            RenderStep::FragmentBufferOffset {
                offset: 4 * COLOR_STRIDE,
                index: 0,
            },
            RECT_DRAW,
            // Draw 6: the viewport goes back to the whole target, the pipeline changes to the one
            // that reads a texture, and the texture arrives as a range.
            RenderStep::Viewports(vec![whole_viewport(SIDE)]),
            RenderStep::Pipeline(reading),
            RenderStep::FragmentTextures {
                textures: vec![read],
                first: 0,
            },
            RenderStep::Scissor(B4),
            RECT_DRAW,
        ]
    }
}

impl Case for Boxes {
    fn name(&self) -> &'static str {
        "render-state-boxes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened. A `RGBA32Float` texel is the `float4` a fragment function
        // returned, so what each draw saw is an exact answer or it is a bug; there is no
        // normalized conversion here and no arithmetic of the shader's -- `tob_solid` returns its
        // binding and `tob_read` reads a texel at integer coordinates, neither of which can take a
        // device's float bounds.
        //
        // One notice is declared, and it is the venue's and not this case's: the buffer the pass
        // reads its target back through is `Shared`, which a device with no unified memory keeps
        // in system memory because that is the only memory both sides can reach.
        vec![crate::replay::notice_allowance(
            Degradation::SharedInSystemMemory.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors = [Some(ColorState::new(FORMAT))];
        for (at, pipeline) in Self::pipelines(&colors).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes("target", &target_spec(SIDE).words()));
        inputs.push(Input::bytes("texture:read", &read_texels(SIDE)));
        inputs.push(Input::bytes("buffer:rects", &rects_buffer()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        // The steps' own parameters reach the fingerprint through the pass's words, so the boxes
        // are covered there; this is the map the module test reads, stated as an input so a region
        // that moves is a change to the case's inputs.
        let mut boxes = Vec::new();
        for rect in [A1, A3, A4, B1, B1_UNREAD, B4, whole_box(SIDE)] {
            boxes.extend(rect.words());
        }
        for viewport in [B3, B3_UNREAD, whole_viewport(SIDE)] {
            boxes.extend(viewport.words());
        }
        inputs.push(Input::bytes("boxes", &boxes));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&colors_state).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let target = made(sites, "target", side.target(&target_spec(SIDE)))?;
        let read = made(
            sites,
            "texture:read",
            side.texture_2d_in_role(&read_spec(SIDE), &read_texels(SIDE)),
        )?;
        let rects = made(sites, "buffer:rects", side.buffer(&rects_buffer()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SIDE) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(states[0], states[1], rects, colors, read)),
            ),
            read_target(target, read_back, SIDE),
        ];
        drawn(side, sites, &work, read_back, SIDE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-facing

/// One draw of the facing case: the cull mode and winding it sets, if it changes them, the fill
/// mode if it changes that, which of the two triangles it draws, and the cell it is confined to.
struct FacingDraw {
    cull: Option<CullMode>,
    winding: Option<Winding>,
    fill: Option<TriangleFillMode>,
    /// `false` for [`FIRST_TRIANGLE`], `true` for [`SECOND_TRIANGLE`].
    reversed: bool,
    cell: ScissorBox,
}

/// The ten draws, in order. Each reads its own colour, at its own index in [`COLORS`].
///
/// The first eight are the four cull-and-winding settings over both triangles; the ninth sets a
/// fill mode of lines and the tenth sets it back. The cells they are **not** in are the control:
/// no draw may reach them, whatever the facing convention turns out to be.
fn facing_draws() -> Vec<FacingDraw> {
    let plain = |cull, winding, reversed, cell| FacingDraw {
        cull,
        winding,
        fill: None,
        reversed,
        cell,
    };
    vec![
        // Cull the back face under a counter-clockwise front: one of the pair draws.
        plain(
            Some(CullMode::Back),
            Some(Winding::CounterClockwise),
            false,
            cell(0, 0),
        ),
        plain(None, None, true, cell(1, 0)),
        // The same cull under a clockwise front: the other one draws.
        plain(None, Some(Winding::Clockwise), false, cell(2, 0)),
        plain(None, None, true, cell(3, 0)),
        // Cull the front face under a counter-clockwise front: the first pair's answer, flipped.
        plain(
            Some(CullMode::Front),
            Some(Winding::CounterClockwise),
            false,
            cell(0, 1),
        ),
        plain(None, None, true, cell(1, 1)),
        // Cull nothing: both draw.
        plain(Some(CullMode::None), None, false, cell(2, 1)),
        plain(None, None, true, cell(3, 1)),
        // A fill mode of lines over a cell no edge of the triangle passes through, and the fill
        // mode set back over the cell beside it.
        FacingDraw {
            cull: None,
            winding: None,
            fill: Some(TriangleFillMode::Lines),
            reversed: false,
            cell: cell(1, 2),
        },
        FacingDraw {
            cull: None,
            winding: None,
            fill: Some(TriangleFillMode::Fill),
            reversed: false,
            cell: cell(2, 2),
        },
    ]
}

/// `setCullMode:`, `setFrontFacingWinding:` and `setTriangleFillMode:` over one triangle and its
/// reverse, ten cells of one pass.
pub struct Facing;

impl Facing {
    /// The one pipeline the pass sets: the triangle and the solid colour.
    fn pipeline(colors: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
        RenderPipelineSpec::new(TRIANGLE, SOLID, colors)
    }

    /// The pass's steps.
    fn steps(state: Handle, points: Handle, colors: Handle) -> Vec<RenderStep> {
        let mut steps = vec![RenderStep::Pipeline(state)];
        for (at, draw) in facing_draws().iter().enumerate() {
            if let Some(cull) = draw.cull {
                steps.push(RenderStep::Cull(cull));
            }
            if let Some(winding) = draw.winding {
                steps.push(RenderStep::FrontFacing(winding));
            }
            if let Some(fill) = draw.fill {
                steps.push(RenderStep::FillMode(fill));
            }
            steps.push(RenderStep::VertexBuffer {
                buffer: points,
                offset: u64::from(draw.reversed) * POINT_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::FragmentBuffer {
                buffer: colors,
                offset: at as u64 * COLOR_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::Scissor(draw.cell));
            steps.push(TRIANGLE_DRAW);
        }
        steps
    }
}

impl Case for Facing {
    fn name(&self) -> &'static str {
        "render-state-facing"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened, for the reason `render-state-boxes` states: every texel this case
        // reads compared exactly. What is declared is one notice, and this case's own shape is
        // what provokes it.
        //
        // **Metal's triangle fill mode is encoder state and Vulkan's polygon mode is pipeline
        // state**, so the draw after `setTriangleFillMode:MTLTriangleFillModeLines` needs a
        // pipeline the pass had not made, and a draw whose pipeline is being made cannot record
        // into an instance that recorded inline: it records into a new render pass instance after
        // it (`metal2vulkan_runtime`'s `render/inline_first.rs`, counted as that crate's
        // `work_counts::Split::PendingPipeline`). A mid-pass fill-mode change therefore splits the
        // Vulkan render pass, which is work Metal does not do, and the runtime announces it.
        //
        // Measured on an Apple M3 Max (macOS 27.0.1, build 26A434), under the exact policy and
        // under the runtime's own defaults alike: our leg announces
        // `metal2vulkan-degrade-render-pass-split` once, its notice naming a pass of one color
        // attachment in two instances, while every texel this case reads and every other field of
        // its transcript compared exactly. The declaration loosens no value -- it says the
        // announcement is this case's -- so the case still fails on any byte that moves.
        //
        // The notice the venue's own memory model provokes for the `Shared` read-back buffer is
        // declared beside it.
        [
            Degradation::RenderPassSplit,
            Degradation::SharedInSystemMemory,
        ]
        .iter()
        .map(|kind| crate::replay::notice_allowance(kind.code()))
        .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&colors_state).words(),
        ));
        inputs.push(Input::bytes("target", &target_spec(SIDE).words()));
        inputs.push(Input::bytes("buffer:points", &points_buffer()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        let cells: Vec<u8> = facing_draws()
            .iter()
            .flat_map(|draw| draw.cell.words())
            .collect();
        inputs.push(Input::bytes("cells", &cells));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&colors_state)),
        )?;
        let target = made(sites, "target", side.target(&target_spec(SIDE)))?;
        let points = made(sites, "buffer:points", side.buffer(&points_buffer()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SIDE) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(state, points, colors)),
            ),
            read_target(target, read_back, SIDE),
        ];
        drawn(side, sites, &work, read_back, SIDE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-unread-slots

/// The two lod clamps the case binds, each a pair of exact binary fractions.
const LOW_CLAMP: LodClamp = LodClamp::new(0.0, 4.0);
const HIGH_CLAMP: LodClamp = LodClamp::new(1.0, 3.0);

/// The vertex and fragment texture, sampler and lod-clamped sampler setters, at slots the pass's
/// own functions read nothing from.
///
/// What it claims is narrow and worth having: **Metal answers each of these calls** at a slot no
/// function of the bound pipeline declares, and the draw that follows them stores exactly the
/// colour it would have stored without them. It claims nothing about what a stage reads through
/// such a slot, because no function here reads one.
///
/// It is a case of its own rather than a few more steps of [`Boxes`] for a reason the round has
/// already paid for once: a refused step takes the whole pass with it, so a setter our runtime
/// refuses where Metal does not would cost every region of a dense case its answer. Here it costs
/// one small target, and the refusal is the finding.
pub struct UnreadSlots;

impl UnreadSlots {
    /// The one pipeline: the rectangle and the solid colour, neither of which declares a texture
    /// or a sampler.
    fn pipeline(colors: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
        RenderPipelineSpec::new(RECT, SOLID, colors)
    }

    /// The pass's steps: the two bindings the functions do read, then every binding they do not,
    /// then one draw of the whole target.
    fn steps(
        state: Handle,
        rects: Handle,
        colors: Handle,
        textures: [Handle; 2],
        samplers: [Handle; 2],
    ) -> Vec<RenderStep> {
        vec![
            RenderStep::Pipeline(state),
            RenderStep::VertexBuffer {
                buffer: rects,
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentBuffer {
                buffer: colors,
                offset: 0,
                index: 0,
            },
            RenderStep::VertexTexture {
                texture: textures[0],
                index: 0,
            },
            RenderStep::VertexTextures {
                textures: textures.to_vec(),
                first: 0,
            },
            RenderStep::VertexSampler {
                sampler: samplers[0],
                index: 0,
            },
            RenderStep::VertexSamplers {
                samplers: samplers.to_vec(),
                first: 0,
            },
            RenderStep::VertexSamplerClamped {
                sampler: samplers[0],
                clamp: LOW_CLAMP,
                index: 0,
            },
            RenderStep::VertexSamplersClamped {
                samplers: vec![(samplers[0], LOW_CLAMP), (samplers[1], HIGH_CLAMP)],
                first: 0,
            },
            RenderStep::FragmentSamplers {
                samplers: samplers.to_vec(),
                first: 0,
            },
            RenderStep::FragmentSamplerClamped {
                sampler: samplers[1],
                clamp: HIGH_CLAMP,
                index: 0,
            },
            RenderStep::FragmentSamplersClamped {
                samplers: vec![(samplers[0], LOW_CLAMP), (samplers[1], HIGH_CLAMP)],
                first: 0,
            },
            RECT_DRAW,
        ]
    }

    /// The two sampler states the case makes: Metal's own defaults, and the same addressed to
    /// repeat, so the two are two states rather than one twice.
    fn sampler_specs() -> [SamplerSpec; 2] {
        [
            SamplerSpec::new(),
            SamplerSpec::new().addressed(SamplerAddressMode::Repeat),
        ]
    }
}

impl Case for UnreadSlots {
    fn name(&self) -> &'static str {
        "render-state-unread-slots"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened, for the reason `render-state-boxes` states, and the one notice the
        // venue's own memory model provokes for the `Shared` read-back buffer.
        vec![crate::replay::notice_allowance(
            Degradation::SharedInSystemMemory.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&colors_state).words(),
        ));
        inputs.push(Input::bytes("target", &target_spec(SMALL).words()));
        inputs.push(Input::bytes("buffer:rects", &rects_buffer()));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer()));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        for (at, spec) in Self::sampler_specs().iter().enumerate() {
            inputs.push(Input::bytes(format!("sampler:{at}"), &spec.words()));
        }
        for (at, texels) in [read_texels(SMALL), read_texels(SMALL)].iter().enumerate() {
            inputs.push(Input::bytes(format!("texture:{at}"), texels));
        }
        let mut clamps = LOW_CLAMP.words();
        clamps.extend(HIGH_CLAMP.words());
        inputs.push(Input::bytes("clamps", &clamps));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&colors_state)),
        )?;
        let target = made(sites, "target", side.target(&target_spec(SMALL)))?;
        let rects = made(sites, "buffer:rects", side.buffer(&rects_buffer()))?;
        let colors = made(sites, "buffer:colors", side.buffer(&colors_buffer()))?;
        let mut textures = Vec::new();
        for at in 0..2 {
            textures.push(made(
                sites,
                &format!("texture:{at}"),
                side.texture_2d_in_role(&read_spec(SMALL), &read_texels(SMALL)),
            )?);
        }
        let mut samplers = Vec::new();
        for (at, spec) in Self::sampler_specs().iter().enumerate() {
            samplers.push(made(sites, &format!("sampler:{at}"), side.sampler(spec))?);
        }
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SMALL) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(
                        state,
                        rects,
                        colors,
                        [textures[0], textures[1]],
                        [samplers[0], samplers[1]],
                    )),
            ),
            read_target(target, read_back, SMALL),
        ];
        drawn(side, sites, &work, read_back, SMALL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allowance::Justification;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing has recorded it.
    ///
    /// Every case of this module is one pass of one queue with nothing a device may reorder, so
    /// the recording's runs are expected to fold into one outcome. The assertion is the module's
    /// own check on that: a case whose runs came out differently would say so here rather than
    /// have one of its runs read as the answer.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so a state setter's effect moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The target's recorded texels of `case`.
    fn recorded(case: &str, side: u32) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), plane(side) as usize);
        Some(bytes.clone())
    }

    /// What the boxes case says texel `(x, y)` of its target holds, from the state each draw was given
    /// and nothing Metal was asked.
    fn boxes_expected(x: u32, y: u32) -> [f32; 4] {
        let inside = |rect: ScissorBox| {
            u64::from(x) >= rect.x
                && u64::from(x) < rect.x + rect.width
                && u64::from(y) >= rect.y
                && u64::from(y) < rect.y + rect.height
        };
        // Draw 6 reads its texture, last and over the rightmost columns of the bottom half.
        if inside(B4) {
            return read_texel(x, y);
        }
        // Draw 5, confined by the first of its two viewports to two columns and two rows.
        if (8..10).contains(&x) && (12..14).contains(&y) {
            return COLORS[BOX_COLORS[4]];
        }
        // Draw 4, confined by the first of its two scissor rectangles.
        if inside(B1) {
            return COLORS[BOX_COLORS[3]];
        }
        // Draw 3, whose colour came in as inline bytes.
        if inside(A4) {
            return COLORS[BOX_COLORS[2]];
        }
        // Draw 2, whose vertex buffer offset reached the right-half rectangle, so only the columns of
        // its scissor rectangle that the rectangle covers were written.
        if inside(A3) && x >= 8 {
            return COLORS[BOX_COLORS[1]];
        }
        // Draw 1.
        if inside(A1) {
            return COLORS[BOX_COLORS[0]];
        }
        [CLEAR; 4]
    }

    /// Which of the ten cells each facing draw's colour is expected in, given `first_is_front`:
    /// whether [`FIRST_TRIANGLE`] is the front-facing one under `MTLWindingCounterClockwise`.
    ///
    /// The convention is read from the recording rather than assumed, because what the case asks is
    /// what the cull mode and the winding **do**, not which way round Metal calls a triangle. Every
    /// entry below is a consequence that holds whichever way it goes.
    fn facing_drawn(first_is_front: bool) -> [bool; 10] {
        let front = first_is_front;
        [
            // Cull the back face, counter-clockwise front: the front-facing one of the pair survives.
            front, !front, // The winding flips, so the other one survives.
            !front, front, // The cull flips instead, so the first pair's answer flips.
            !front, front, // Nothing is culled.
            true, true,
            // A fill mode of lines over a cell no edge of the triangle passes through.
            false, // The fill mode set back.
            true,
        ]
    }

    /// The `float4` at texel `(x, y)` of a target's bytes.
    fn value(bytes: &[u8], x: u32, y: u32, side: u32) -> [f32; 4] {
        let at = ((y * side + x) * TEXEL) as usize;
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

    /// **Every scissor rectangle and viewport the module states lies inside its target.**
    ///
    /// A rectangle reaching past the attachments is the one shape of `setScissorRect:` Metal's own
    /// validation calls a programming error, and what plain Metal's driver programs for one is
    /// unmeasured; it reaches the rasterizer's own clip. So the suite does not find out by trying
    /// it, and the guard is here rather than in a side, where it holds before a device is asked
    /// and holds for both legs at once.
    #[test]
    fn every_box_lies_inside_its_target() {
        // The check sees a rectangle that reaches past one, so a green run means the module's
        // boxes are inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, SIDE as u64 + 1, 1).past([SIDE; 2]));
        assert!(ScissorBox::new(SIDE as u64, 0, 1, 1).past([SIDE; 2]));
        assert!(!whole_box(SIDE).past([SIDE; 2]));
        let boxes = [A1, A3, A4, B1, B1_UNREAD, B4, whole_box(SIDE)];
        for rect in boxes {
            assert!(!rect.past([SIDE; 2]), "{rect:?} reaches past the target");
        }
        for draw in facing_draws() {
            assert!(
                !draw.cell.past([SIDE; 2]),
                "{:?} reaches past the target",
                draw.cell
            );
        }
        // A viewport's extent is a `double` and Metal takes one past the target, so this is the
        // weaker claim the type allows: every viewport the module states is inside it too, which
        // is what makes its region map readable.
        for viewport in [B3, B3_UNREAD, whole_viewport(SIDE)] {
            for axis in 0..2 {
                assert!(viewport.origin[axis] >= 0.0);
                assert!(viewport.origin[axis] + viewport.size[axis] <= f64::from(SIDE));
            }
        }
    }

    /// **The regions of the boxes map are distinct answers**, so a draw that saw another draw's
    /// state writes a value this case can name rather than one that happens to pass.
    ///
    /// Written over the rule itself rather than over a recording, so it holds with no device and
    /// fails the moment a colour, the clear or a box is changed into another region's value.
    #[test]
    fn the_boxes_map_names_every_region_apart() {
        let regions: Vec<[u32; 4]> = [
            boxes_expected(0, 0),
            boxes_expected(4, 0),
            boxes_expected(8, 0),
            boxes_expected(12, 0),
            boxes_expected(4, 8),
            boxes_expected(8, 12),
        ]
        .iter()
        .map(|color| color.map(f32::to_bits))
        .collect();
        let distinct: BTreeSet<[u32; 4]> = regions.iter().copied().collect();
        assert_eq!(distinct.len(), regions.len(), "two regions answer alike");
        // The four columns of draw 2's scissor rectangle its right-half rectangle does not cover
        // keep the clear, which is the whole reading of `setVertexBufferOffset:atIndex:` here.
        assert_eq!(boxes_expected(4, 0), [CLEAR; 4]);
        assert_eq!(boxes_expected(7, 7), [CLEAR; 4]);
        assert_eq!(boxes_expected(8, 0), COLORS[BOX_COLORS[1]]);
        // The second scissor rectangle and the second viewport are unreachable, so their own
        // boxes hold the clear.
        assert_eq!(boxes_expected(0, 8), [CLEAR; 4]);
        assert_eq!(boxes_expected(0, 12), [CLEAR; 4]);
        // Draw 5's viewport confined it: the two columns past it keep the clear.
        assert_eq!(boxes_expected(8, 12), COLORS[BOX_COLORS[4]]);
        assert_eq!(boxes_expected(10, 12), [CLEAR; 4]);
        // The read texture holds a value no colour of the module is, in every texel, and each
        // texel its own, so a read at the wrong texel is caught.
        let texels: BTreeSet<[u32; 4]> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| read_texel(x, y).map(f32::to_bits)))
            .collect();
        assert_eq!(texels.len(), (SIDE * SIDE) as usize);
        for color in COLORS.iter().chain(std::iter::once(&[CLEAR; 4])) {
            assert!(!texels.contains(&color.map(f32::to_bits)));
        }
        assert_eq!(layout(SIDE).extent(), plane(SIDE) as usize);
        // The two pipelines are two pipelines.
        let colors = [Some(ColorState::new(FORMAT))];
        let [solid, reading] = Boxes::pipelines(&colors);
        assert_ne!(solid.words(), reading.words());
    }

    /// **Metal's recorded target holds, in every texel, what the boxes map says it should.**
    #[test]
    fn metal_confined_each_box_draw_as_its_boxes_say() {
        let Some(bytes) = recorded(Boxes.name(), SIDE) else {
            println!("render-state-boxes: no recording in the tree yet");
            return;
        };
        let mut differ = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (got, want) = (value(&bytes, x, y, SIDE), boxes_expected(x, y));
                if got != want {
                    differ.push(format!("({x}, {y}): Metal {got:?}, the map says {want:?}"));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("BOXES-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} texels differ from what the boxes each draw was given say",
            differ.len(),
            SIDE * SIDE
        );
    }

    /// **The ten facing cells are what the cull mode, the winding and the fill mode imply**, with
    /// the facing convention read from the first cell rather than assumed.
    #[test]
    fn metal_culled_each_facing_draw_as_its_state_says() {
        let Some(bytes) = recorded(Facing.name(), SIDE) else {
            println!("render-state-facing: no recording in the tree yet");
            return;
        };
        let draws = facing_draws();
        // What each cell holds, read as one of three answers: its own colour, the clear, or
        // something neither -- which would be a draw that leaked out of its scissor rectangle.
        let mut held = Vec::new();
        for (at, draw) in draws.iter().enumerate() {
            let mut colored = 0usize;
            let mut cleared = 0usize;
            let mut other = Vec::new();
            for y in draw.cell.y..draw.cell.y + draw.cell.height {
                for x in draw.cell.x..draw.cell.x + draw.cell.width {
                    let got = value(&bytes, x as u32, y as u32, SIDE);
                    if got == COLORS[at] {
                        colored += 1;
                    } else if got == [CLEAR; 4] {
                        cleared += 1;
                    } else {
                        other.push(format!("({x}, {y}): {got:?}"));
                    }
                }
            }
            assert!(
                other.is_empty(),
                "cell {at} holds {} texel(s) that are neither its colour nor the clear: {}",
                other.len(),
                other.join(", ")
            );
            let whole = (draw.cell.width * draw.cell.height) as usize;
            assert!(
                colored == whole || cleared == whole,
                "cell {at} is {colored} of {whole} its colour, which is neither drawn nor clear"
            );
            held.push(colored == whole);
        }
        let expected = facing_drawn(held[0]);
        assert_eq!(
            held.as_slice(),
            expected.as_slice(),
            "the cells Metal drew are not what the cull mode, the winding and the fill mode imply \
             (the first triangle is {} under a counter-clockwise front)",
            match held[0] {
                true => "front-facing",
                false => "back-facing",
            }
        );
        // Every cell no draw was confined to holds the clear, which is what says no draw escaped
        // its scissor rectangle.
        let used: BTreeSet<(u64, u64)> = draws
            .iter()
            .flat_map(|draw| {
                let cell = draw.cell;
                (cell.y..cell.y + cell.height)
                    .flat_map(move |y| (cell.x..cell.x + cell.width).map(move |x| (x, y)))
            })
            .collect();
        for y in 0..u64::from(SIDE) {
            for x in 0..u64::from(SIDE) {
                if used.contains(&(x, y)) {
                    continue;
                }
                assert_eq!(
                    value(&bytes, x as u32, y as u32, SIDE),
                    [CLEAR; 4],
                    "({x}, {y}) is outside every cell and is not the clear"
                );
            }
        }
    }

    /// **The facing map's own rule**: flipping the winding or the cull mode flips which of the
    /// pair draws, a cull mode of none draws both, and the two fill modes disagree.
    ///
    /// Stated over the rule rather than a recording, so it holds with no device, and stated for
    /// both conventions, because the case may not assume either.
    #[test]
    fn the_facing_rule_holds_for_either_convention() {
        for front in [false, true] {
            let drawn = facing_drawn(front);
            // The four cull-and-winding pairs: exactly one of each pair draws.
            for pair in 0..3 {
                assert_ne!(
                    drawn[2 * pair],
                    drawn[2 * pair + 1],
                    "pair {pair} draws both or neither under front={front}"
                );
            }
            // Flipping the winding flips which of the pair draws.
            assert_ne!(drawn[0], drawn[2]);
            // Flipping the cull mode flips it back.
            assert_ne!(drawn[0], drawn[4]);
            assert_eq!(drawn[2], drawn[4]);
            // A cull mode of none draws both.
            assert!(drawn[6] && drawn[7]);
            // The fill modes disagree over the same triangle and the same cull mode.
            assert!(!drawn[8]);
            assert!(drawn[9]);
        }
        // The ten draws take ten colours, and the cells are ten distinct cells.
        let draws = facing_draws();
        assert_eq!(draws.len(), COLORS.len());
        let cells: BTreeSet<[u64; 4]> = draws
            .iter()
            .map(|draw| [draw.cell.x, draw.cell.y, draw.cell.width, draw.cell.height])
            .collect();
        assert_eq!(cells.len(), draws.len(), "two draws share a cell");
        // The lines cell is where no edge of the triangle can reach it. The triangle covers the
        // whole clip square and more, so the edges that reach the target are its clip against the
        // view volume: the target's own border. The cell is inside that border by a pixel at
        // least on every side, which is what makes "entirely clear" the reading of a fill mode of
        // lines rather than a guess about line width.
        let lines = draws[8].cell;
        assert!(lines.x >= 1 && lines.y >= 1);
        assert!(lines.x + lines.width < u64::from(SIDE));
        assert!(lines.y + lines.height < u64::from(SIDE));
    }

    /// **The unread-slot case's target is the colour its bindings hold**, in every texel: the
    /// vertex and fragment slots no function of the pipeline reads changed nothing.
    #[test]
    fn metal_drew_the_unread_slot_case_as_its_bindings_say() {
        let Some(bytes) = recorded(UnreadSlots.name(), SMALL) else {
            println!("render-state-unread-slots: no recording in the tree yet");
            return;
        };
        for y in 0..SMALL {
            for x in 0..SMALL {
                assert_eq!(
                    value(&bytes, x, y, SMALL),
                    COLORS[0],
                    "({x}, {y}) is not the colour the fragment binding holds"
                );
            }
        }
        // The two sampler states are two states and the clamps are two clamps, so the range forms
        // bind something that differs rather than one thing twice.
        let [first, second] = UnreadSlots::sampler_specs();
        assert_ne!(first.words(), second.words());
        assert_ne!(LOW_CLAMP.words(), HIGH_CLAMP.words());
    }

    /// **Every case of the family names itself once**, and the names are the ones the recordings
    /// are keyed by.
    #[test]
    fn the_family_names_three_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 3);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // Every case declares the notice the venue's own memory model provokes for the
            // `Shared` read-back buffer, and exactly one -- the facing case -- declares the pass
            // split besides. No value of any case is loosened, so every declaration sits at the
            // notices site.
            let declared = case.allowances();
            for allowance in &declared {
                assert!(allowance.path.starts_with("case/notices"), "{allowance:?}");
            }
            let codes: Vec<&Justification> =
                declared.iter().map(|held| &held.justified_by).collect();
            let shared =
                Justification::Degradation(Degradation::SharedInSystemMemory.code().to_owned());
            let split = Justification::Degradation(Degradation::RenderPassSplit.code().to_owned());
            match case.name() {
                "render-state-facing" => assert_eq!(codes, vec![&split, &shared]),
                _ => assert_eq!(codes, vec![&shared], "{}", case.name()),
            }
        }
    }
}
