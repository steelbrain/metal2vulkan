//! Patches drawn into a colour target: what a patch draw renders, rather than the domain points the
//! tessellator generates (`tessellation_sweeps`, `tessellation_crossings`).
//!
//! The owned fixture `tess_rendered_patches` is a triangle patch of three control points that the
//! vertex descriptor fetches (a clip-space `float2` each, per patch control point) and a colour
//! fetched per patch. Its post-tessellation vertex function places each domain point at its
//! barycentric blend of the control points, so the tessellated triangles cover the patch's own
//! triangle whatever the factors, and passes the colour on flat for its fragment function to write.
//! A texel therefore names the patch that covers it, or holds the clear value where none does.
//!
//! # The patches and the passes
//!
//! Four patches, under the state a recorded stream's patch draws use: `fractionalOdd` up to 64,
//! factors per patch, counter-clockwise output. Their factors are 8 bytes into the factor buffer,
//! past 8 bytes of zero, so a draw that ignored the offset would read patch 0's factors as zeros.
//!
//! | patch | covers | colour | factors (edges, inside) |
//! |---|---|---|---|
//! | 0 | the top half of clip space | green | 3, 5, 7; 4 |
//! | 1 | the whole of it | blue | 0, 5, 7; 4: the zero edge culls it |
//! | 2 | a triangle at the bottom-left corner | red | 2, 2, 2; 2 |
//! | 3 | patch 2's triangle, its last two control points swapped | yellow | 3, 3, 3; 3 |
//!
//! - `drawn`: no culling; one draw of patches 0 and 1, then one of patch 3 alone, from patch start
//!   3. Patch 1 reads its own factor record, whose zero edge culls it, and patch 3's draw reads its
//!   factors and colour at the patch start's record, so the target holds green above and yellow at
//!   the corner (Metal's answer).
//! - `back-culled`: back faces culled, counter-clockwise front faces; one draw of all four
//!   patches. Swapping two control points mirrors patch 2 into patch 3, so the tessellator's output
//!   winding makes exactly one of the two front-facing, and patch 0 faces as patch 2 does: Metal
//!   keeps patches 0 and 2 and culls patch 3, which drawn after patch 2 would have covered its red.
//!
//! # Why the answer is exact
//!
//! Every colour is 0 or 1 in each channel and passed flat, so an `RGBA8Unorm` texel holds 0 or 255
//! exactly. No patch's edge runs through a texel centre of the 8 by 8 target: the axis-aligned edges
//! lie on texel boundaries, and the bottom-left patch's third edge, `y = -0.75 - x / 2`, meets no
//! centre (`tests::no_edge_crosses_a_texel_centre`). Where the tessellator places its inner points
//! (the fractional spacing, the triangles' order) moves no texel, because the patch's triangles
//! cover its own triangle and carry one colour.
//!
//! # Safety
//!
//! Two passes of at most four patches each, at factors no higher than 7, into an 8 by 8 target.
//! Every fetch lies inside its buffer: four patches' control points and colours, and four factor
//! records past the 8-byte offset (`tests::every_fetch_lies_inside_its_buffer`).

use super::indirect_commands::{fixture_inputs, shaders_of};
use super::tessellation_sweeps::{tessellation_notices, Domain};
use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::degradation::Degradation;
use crate::fixture::Fixture;
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec, TessellationState, VertexAttribute, VertexBufferLayout,
    VertexLayout, VertexStride,
};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    CullMode, PixelFormat, TessellationFactorStepFunction, TessellationPartitionMode, VertexFormat,
    VertexStepFunction, Winding,
};

/// The owned fixture, whose post-tessellation vertex function is its own name.
const FIXTURE: &str = "tess_rendered_patches";

/// The fixture's fragment function.
const FRAGMENT: &str = "tess_rendered_patches_fragment";

/// The target's side in texels.
const SIDE: u32 = 8;

/// The target's format: a channel of 0 or 1 is a byte of 0 or 255.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// Bytes one row of the target takes.
const ROW_PITCH: u32 = SIDE * 4;

/// Bytes one pass's target takes, its slice of the read-back buffer.
const TARGET_BYTES: u32 = SIDE * ROW_PITCH;

/// The bytes the factor buffer holds before patch 0's record, which the draws skip by their offset.
const FACTOR_OFFSET: u64 = 8;

/// Each patch's control points, in clip space (y up): the module's table.
const POINTS: [[[f32; 2]; 3]; 4] = [
    [[-1.0, 0.0], [3.0, 0.0], [-1.0, 4.0]],
    [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]],
    [[-1.0, -1.0], [0.5, -1.0], [-1.0, -0.25]],
    [[-1.0, -1.0], [-1.0, -0.25], [0.5, -1.0]],
];

/// Each patch's colour.
const COLOURS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// Each patch's edge factors and inside factor.
const FACTORS: [([f32; 3], f32); 4] = [
    ([3.0, 5.0, 7.0], 4.0),
    ([0.0, 5.0, 7.0], 4.0),
    ([2.0, 2.0, 2.0], 2.0),
    ([3.0, 3.0, 3.0], 3.0),
];

/// The control points' `float2` at attribute 0 of buffer 0, and the colour at attribute 1 of
/// buffer 1.
const ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute::new(0, VertexFormat::Float2, 0, 0),
    VertexAttribute::new(1, VertexFormat::Float4, 0, 1),
];

/// Buffer 0 steps per patch control point, buffer 1 per patch.
const LAYOUTS: [VertexBufferLayout; 2] = [
    VertexBufferLayout {
        index: 0,
        stride: VertexStride::Bytes(8),
        step: VertexStepFunction::PerPatchControlPoint,
        rate: 1,
    },
    VertexBufferLayout {
        index: 1,
        stride: VertexStride::Bytes(16),
        step: VertexStepFunction::PerPatch,
        rate: 1,
    },
];

/// The one colour attachment.
const COLORS: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// The tessellation state: `fractionalOdd` up to 64, per patch, counter-clockwise.
fn state() -> TessellationState {
    let mut state = TessellationState::new().partitioned(TessellationPartitionMode::FractionalOdd);
    state.max_factor = 64;
    state.factor_step = TessellationFactorStepFunction::PerPatch;
    state.winding = Winding::CounterClockwise;
    state
}

/// The pipeline: the fixture's two functions, its vertex descriptor and [`state`].
fn pipeline_spec() -> RenderPipelineSpec<'static> {
    let mut spec = RenderPipelineSpec::of(
        StageFunction::named(FIXTURE),
        Some(StageFunction::named(FRAGMENT)),
        &COLORS,
    );
    spec.vertex_layout = Some(VertexLayout::new(&ATTRIBUTES, &LAYOUTS));
    spec.tessellation = Some(state());
    spec
}

/// The target both passes draw into: private, since a blit reads it back.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [SIDE, SIDE])
}

/// The bytes of `values`, each little-endian.
fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// The control points' buffer: four patches of three `float2`s.
fn points_bytes() -> Vec<u8> {
    floats(POINTS.iter().flatten().flatten().copied())
}

/// The colours' buffer: a `float4` a patch.
fn colours_bytes() -> Vec<u8> {
    floats(COLOURS.iter().flatten().copied())
}

/// The factor buffer: [`FACTOR_OFFSET`] bytes of zero, then a triangle factor record a patch.
fn factor_bytes() -> Vec<u8> {
    let mut bytes = vec![0; FACTOR_OFFSET as usize];
    for (edges, inside) in FACTORS {
        bytes.extend(Domain::Triangle.factors([edges[0], edges[1], edges[2], 0.0], [inside, 0.0]));
    }
    bytes
}

/// The buffers a pass binds: control points, colours and factors.
#[derive(Clone, Copy)]
struct Bound {
    points: Handle,
    colours: Handle,
    factors: Handle,
}

/// A draw of `count` patches from `first`, one instance.
fn patches(first: u32, count: u32) -> RenderStep {
    RenderStep::DrawPatches {
        control_points: 3,
        first,
        count,
        patch_indices: None,
        patch_index_offset: 0,
        instances: 1,
        base_instance: 0,
    }
}

/// The two passes' names and steps (the module's list).
fn passes(pipeline: Handle, bound: Bound) -> [(&'static str, Vec<RenderStep>); 2] {
    let binds = vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::VertexBuffer {
            buffer: bound.points,
            offset: 0,
            index: 0,
        },
        RenderStep::VertexBuffer {
            buffer: bound.colours,
            offset: 0,
            index: 1,
        },
        RenderStep::TessellationFactorBuffer {
            buffer: Some(bound.factors),
            offset: FACTOR_OFFSET,
            instance_stride: 0,
        },
    ];
    let mut drawn = binds.clone();
    drawn.extend([patches(0, 2), patches(3, 1)]);
    let mut culled = binds;
    culled.extend([
        RenderStep::Cull(CullMode::Back),
        RenderStep::FrontFacing(Winding::CounterClockwise),
        patches(0, 4),
    ]);
    [("drawn", drawn), ("back-culled", culled)]
}

/// The layout a pass's target is read under: a row of the layout per row of texels, a byte per
/// channel, so a diff names the texel and the channel.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: SIDE * 4,
        row_pitch: ROW_PITCH,
        rows: SIDE,
        image_pitch: TARGET_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The case.
pub struct RenderedPatches;

/// The case the suite holds.
pub fn all() -> Vec<RenderedPatches> {
    vec![RenderedPatches]
}

impl Case for RenderedPatches {
    fn name(&self) -> &'static str {
        "tessellation-rendered-patches"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The notices every tessellation pipeline of ours can announce, and the fractional
        // spacing's, which a pipeline of a fractional partition drawn through the device's
        // tessellator announces. Declarations of the notices site alone: every texel compares
        // exactly.
        let mut allowances = tessellation_notices();
        allowances.push(crate::replay::notice_allowance(
            Degradation::TessellationFractionalSpacing.code(),
        ));
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = shaders_of(FIXTURE, FIXTURE)?;
        shaders.extend(shaders_of(FIXTURE, FRAGMENT)?);
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let bound = Bound {
            points: Handle(1),
            colours: Handle(2),
            factors: Handle(3),
        };
        let mut inputs = fixture_inputs(FIXTURE)?;
        inputs.push(Input::bytes("render-pipeline", &pipeline_spec().words()));
        inputs.push(Input::bytes("target", &target_spec().words()));
        inputs.push(Input::bytes("buffer:points", &points_bytes()));
        inputs.push(Input::bytes("buffer:colours", &colours_bytes()));
        inputs.push(Input::bytes("buffer:factors", &factor_bytes()));
        for (name, steps) in passes(Handle(0), bound) {
            let pass =
                RenderPass::clearing(vec![PassColor::cleared(Handle(4), [0.0, 0.0, 0.0, 1.0])])
                    .drawing(steps);
            inputs.push(Input::bytes(format!("pass:{name}"), &pass.words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &pipeline_spec()),
        )?;
        let bound = Bound {
            points: made(sites, "buffer:points", side.buffer(&points_bytes()))?,
            colours: made(sites, "buffer:colours", side.buffer(&colours_bytes()))?,
            factors: made(sites, "buffer:factors", side.buffer(&factor_bytes()))?,
        };
        let target = made(sites, "target", side.target(&target_spec()))?;
        let passes = passes(pipeline, bound);
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(passes.len() * TARGET_BYTES as usize),
        )?;
        // Both passes in one command buffer, each followed by the blit that reads its target into
        // its own slice of the read-back buffer.
        let mut work = Vec::with_capacity(2 * passes.len());
        for (at, (_, steps)) in passes.iter().enumerate() {
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [0.0, 0.0, 0.0, 1.0])])
                    .drawing(steps.clone()),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into: read_back,
                offset: at as u64 * u64::from(TARGET_BYTES),
                row_pitch: ROW_PITCH,
                size: [SIDE, SIDE],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, passes.len() * TARGET_BYTES as usize),
            |_| Value::Absent,
        )?;
        for (at, (name, _)) in passes.iter().enumerate() {
            let first = at * TARGET_BYTES as usize;
            let texels = bytes[first..first + TARGET_BYTES as usize].to_vec();
            let mut node = Node::branch(*name, Vec::new())
                .with(Node::leaf("texels", Observed::bytes(texels, layout())));
            // What each of the pass's two pieces of work answered, which is `None` on Metal's
            // side: a piece of ours that refuses reads as a refusal rather than a divergence.
            for (piece, what) in [(2 * at, "pass"), (2 * at + 1, "read")] {
                if let Some(Some(failure)) = outcome.refused.get(piece) {
                    node = node.with(Node::leaf(what, Observed::value(failure.recorded())));
                }
            }
            sites.push(node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No patch edge that is not a texel boundary passes through a texel centre: the centres lie
    /// at odd multiples of 1/8 in clip space, and the one slanted edge inside the target, from
    /// `(0.5, -1)` to `(-1, -0.25)`, is `y = -0.75 - x / 2`, which at a centre's `x` is an odd
    /// multiple of 1/16 and so never a centre's `y`.
    #[test]
    fn no_edge_crosses_a_texel_centre() {
        let centre = |k: u32| -1.0 + (2.0 * k as f64 + 1.0) / f64::from(SIDE);
        for k in 0..SIDE {
            let y = -0.75 - centre(k) / 2.0;
            for j in 0..SIDE {
                assert_ne!(y, centre(j), "texel ({k}, {j})");
            }
        }
        // The axis-aligned edges inside the target lie on texel boundaries.
        for boundary in [0.0f64, -1.0] {
            assert_eq!((boundary + 1.0) * f64::from(SIDE) / 2.0 % 1.0, 0.0);
        }
    }

    /// Every fetch lies inside its buffer: the control points and colours of the four patches the
    /// passes draw, and each factor record past the offset.
    #[test]
    fn every_fetch_lies_inside_its_buffer() {
        assert_eq!(points_bytes().len(), 4 * 3 * 8);
        assert_eq!(colours_bytes().len(), 4 * 16);
        assert_eq!(factor_bytes().len() as u64, FACTOR_OFFSET + 4 * 8);
        for (_, steps) in passes(
            Handle(0),
            Bound {
                points: Handle(1),
                colours: Handle(2),
                factors: Handle(3),
            },
        ) {
            for step in steps {
                if let RenderStep::DrawPatches { first, count, .. } = step {
                    assert!(first + count <= 4, "patches {first}..{}", first + count);
                }
            }
        }
    }

    /// Metal's recorded answer is the module's: `drawn` holds green over the top half and yellow
    /// at the corner, `back-culled` green and red, and every other texel the clear value.
    #[test]
    fn metal_answered_as_the_module_says() {
        let case = RenderedPatches;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let texel = |pass: &str, x: u32, y: u32| -> [u8; 4] {
            // The corner patch covers row 5 to 1 texel, row 6 to 3 and row 7 to 5.
            let corner = y >= 5 && x <= 2 * (y - 5);
            let colour = match (y < 4, corner, pass) {
                (true, _, _) => COLOURS[0],
                (false, true, "drawn") => COLOURS[3],
                (false, true, _) => COLOURS[2],
                (false, false, _) => [0.0, 0.0, 0.0, 1.0],
            };
            colour.map(|c| (c * 255.0) as u8)
        };
        let mut checked = 0;
        for recorded in &recording.answers {
            for outcome in &recorded.outcomes {
                for pass in ["drawn", "back-culled"] {
                    let Some(Value::Bytes(bytes)) = (outcome.transcript.root.child(pass))
                        .and_then(|node| node.child("texels"))
                        .map(|node| &node.observed.value)
                    else {
                        panic!("{pass}: no texels recorded");
                    };
                    let want: Vec<u8> = (0..SIDE)
                        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
                        .flat_map(|(x, y)| texel(pass, x, y))
                        .collect();
                    assert_eq!(bytes, &want, "{pass}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "the recording holds no pass");
    }

    /// The patch that swaps two control points is the other's mirror: their clip-space triangles
    /// are one triangle wound the other way.
    #[test]
    fn patch_3_mirrors_patch_2() {
        let area = |p: [[f32; 2]; 3]| {
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
        };
        assert_eq!(area(POINTS[2]), -area(POINTS[3]));
        assert!(area(POINTS[0]) > 0.0 && area(POINTS[2]) > 0.0);
    }
}
