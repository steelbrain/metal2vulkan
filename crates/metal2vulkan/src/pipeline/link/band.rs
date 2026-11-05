//! The synthesized stages of a render pipeline variant that draws its lines, or its triangles'
//! edges, as bands (`VariantKey::line_band`): Apple's `setLineWidth:` at a width other than 1 on a
//! device whose own lines do not draw it (the runtime's `render/line_width.rs` chooses the route).
//!
//! Metal draws a line of width `w` (its quantized width, a multiple of 1/16 in `[0, 16]`) as the
//! parallelogram Vulkan's non-strict wide lines may be (the Vulkan specification's line segments,
//! `nonStrictWideLinesUseParallelogram`): the segment's major axis is the larger of its
//! framebuffer extents, x on a tie; the parallelogram's two minor-axis edges pass through the
//! endpoints, each `w` long and centred on its endpoint, and its other two edges are parallel to
//! the segment; a pixel is drawn where its centre lies inside, by the polygon rules' ties, the
//! endpoints first snapped to 1/256 pixel (Metal probes on the Apple M3 Max, macOS 27.0.1). Metal
//! draws width 0 exactly as 1/16. It draws a triangle in fill mode `Lines` as its three edges,
//! each that parallelogram, with no joins: where two overlap a pixel takes two fragments. The
//! triangle's fragments read its facing, its cull mode culls them by it (a zero-area triangle is
//! back-facing), and they read its flat inputs from its provoking vertex and its primitive ID.
//!
//! The variant's pipeline draws each primitive as one patch of its vertices in order (a
//! `PATCH_LIST`, [`BandPrimitive::patch_control_points`]) through two stages link adds between the
//! vertex and fragment stages:
//!
//! - the **control stage** ([`LineBand::control_module`]), one invocation per control point,
//!   copies every per-vertex value its vertex writes (`Position`, clip distances, `PointSize` where
//!   the device's tessellation stages take it, and every varying), so its inputs are the vertex
//!   stage's outputs entire, as an implementation that lays a stage interface out as one record
//!   needs (MoltenVK lays a vertex stage's outputs out so when tessellation follows); and writes
//!   the tessellation levels in the quad domain: every level 1 for a line, so the tessellator makes
//!   the patch's four corners and two triangles; for a triangle the levels of [`EDGE_LEVELS`];
//! - the **evaluation stage** ([`LineBand::evaluation_module`]) places each vertex the tessellator
//!   makes at a corner of a parallelogram, offset from an endpoint across its segment by `h`
//!   framebuffer pixels along the segment's minor axis, `h` half the width: `n = (0, sign dx)` for
//!   an x-major segment and `(-sign dy, 0)` otherwise, `(dx, dy)` the segment's framebuffer extent.
//!   The offset is taken to clip space through the draw's Vulkan viewport 0, whose half extent
//!   `(sx, sy)` the draw pushes beside `h` (`PushField::LineBand`, the stage's own push range). The
//!   rasterizer snaps each corner to the device's subpixel grid, where Metal snaps the endpoints to
//!   1/256 pixel and then offsets them: on a grid of 1/256 the two agree, `h` being a multiple of
//!   1/32 pixel, but for an endpoint half a step off the grid, which Metal rounds up and the
//!   device's rasterizer as it rounds (`DriverFact::SubpixelSnapIsMetals`; the runtime announces
//!   the band's draws on a driver without it).
//!
//! A line's corner `(u, v)` lies at endpoint `u` (0 the first, 1 the second), offset by `(2v - 1)
//! h`, so the two triangles cover the parallelogram exactly and wind alike in every direction
//! ([`LineBand::FRONT_FACE`]).
//!
//! A triangle's patch is drawn by the quad domain's fans alone, so its coverage follows from the
//! Vulkan specification's quad tessellation and not from any choice it leaves to the device. Its
//! inner levels `(3, 2)` make two inner vertices, `I1 = (1/3, 1/2)` and `I2 = (2/3, 1/2)`, an
//! inner rectangle degenerate to the segment between them; the area near the `u = 0` edge is then
//! the fan of triangles joining `I1` to each vertex of that edge, and near the `u = 1` edge the fan
//! from `I2` ("If either edge of the innermost rectangle is degenerate, the area near the
//! corresponding outer edges is filled by connecting each vertex on the outer edge with the single
//! vertex making up the inner edge"); the `v = 0` and `v = 1` edges, of one segment each, fill the
//! rest with two triangles each. The stage rotates the triangle `(a, b, c)` so that `a` is the
//! vertex of the largest clip-space `w`, and places `I1` at `a` and `I2` at `b`, the middles of
//! the ends of the parallelograms of edges `ab` and `ac` at `a`, and of `bc` at `b`; every vertex
//! of the `u = 0` edge at `a`, a corner of `ab`'s or `ac`'s parallelogram, every vertex of the
//! `u = 1` edge at `b` or a corner of `bc`'s, and every corner of the domain at `a` or `b`
//! ([`LEFT`], [`RIGHT`]). The fans from `a` over each parallelogram's corners, and from `b` over
//! `bc`'s, cover the three parallelograms exactly, each by three triangles, and every other
//! triangle has two vertices at one point, so it covers nothing (the `v = 0` and `v = 1` edges'
//! included, which lie at `a` and `b`). The `u = 1` fan runs over its corners the other way, so
//! every triangle winds as a line's band does; where the triangle faces the other way than a line
//! would under the draw's front face, the stage takes `-n` for `n`, which turns every one, so
//! they face as the triangle does, and the draw's cull mode culls them by its facing. The facing
//! is the sign of the determinant of the vertices' clip-space `(x, y, w)` scaled by `sx sy`,
//! which is the sign of their framebuffer area where they lie in front of the eye and that of the
//! part in front where they do not; a zero-area triangle faces back under either front face,
//! which the stage reads from the band's fourth value.
//!
//! An endpoint at `w < 2^-20` (at or behind the eye) is first moved along the segment toward the
//! other to `w = 2^-20`, its clip distances and float varyings with it, so the band's direction is
//! the visible segment's; the corners' triangles are then clipped as any triangle is. A segment
//! that crosses the near or far plane is cut where those triangles are, which can differ from
//! Metal's segment clipped first by part of a pixel at the cut. A triangle's `a`, of the largest
//! `w`, never moves: its edges at `a` meet there. A triangle that crosses the near plane is drawn
//! as its edges' visible parts, with no edge along the plane: what Metal draws there is not
//! measured.
//!
//! A corner's float varyings and clip distances are its endpoint's, so they interpolate along the
//! segment and are constant across it, as its depth is (a triangle's edge too, as Metal's: a
//! horizontal edge of a sloped triangle is drawn at the edge's depth); a varying the fragment
//! stage reads flat (or an integer one), and the point size, is the first control point's at
//! every corner, the primitive's provoking vertex. What Metal's line bands' fragments read is not
//! measured. A triangle's band is drawn with the draw's depth bias, which the device computes
//! from each band triangle's slope and depths, where Metal biases a triangle's edges by the
//! triangle's slope and its largest depth's exponent (the runtime announces the difference).
//!
//! Refused (`Unsupported{WideLines}`): a device without `tessellationShader` (needing it), and as
//! not implemented a post-tessellation vertex function, a vertex function that writes
//! `[[render_target_array_index]]` or `[[viewport_array_index]]`, a pipeline with synthesized
//! geometry stages (`link::geometry`), a fragment function that reads `[[barycentric_coord]]`, a
//! pipeline drawn as multiview, and stages past `maxTessellationEvaluationOutputComponents`.

use crate::budget::Meter;
use crate::device::requirements::{Requirements, VkFeature, VkLimit};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::pipeline::artifact::{Reconvergence, SpecializationLayout, StageModule, StrategyCensus};
use crate::pipeline::graphics::BandPrimitive;
use crate::pipeline::interface::StageInterface;
use crate::pipeline::link::geometry::{array, kf, ku, scalar};
use crate::pipeline::stage_io::{Interpolation, IoNumeric, IoRole, IoType, StageBuiltin};
use crate::pipeline::sym::{PushField, Stage, VaryingKey};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConvOp, ExecMode, ExecModel, GlslOp, IoDecor, IoDir, MemAccess, Type, TypedModule, UnOp,
};
use crate::spirv::serialize::{SymbolTable, ENTRY_NAME};
use crate::spv::BuiltIn as B;
use crate::vk;

#[cfg(test)]
mod tests;

/// The input arrays' length: `gl_MaxPatchVertices`, the smallest `maxTessellationPatchSize`
/// Vulkan allows.
const INPUT_ARRAY: u32 = 32;

/// The least clip-space `w` an endpoint is drawn at (the module docs).
const LEAST_W: f32 = 1.0 / 1_048_576.0;

/// A point of a triangle's band (the module docs): an end of one of its edges, `ab`, `ac` or `bc`
/// (0, 1, 2), `a` its vertex of the largest `w` and `b` and `c` the next in its order, offset
/// across the edge by `side` half widths along the edge's minor axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Point {
    edge: u32,
    /// The edge's second end (`b` of `ab`, `c` of `ac` and `bc`), else its first.
    end: bool,
    side: i32,
}

impl Point {
    /// Its code in the evaluation stage: `6 edge + 3 end + side + 1`.
    const fn code(self) -> u32 {
        self.edge * 6 + self.end as u32 * 3 + (self.side + 1) as u32
    }
}

/// The point at `side` half widths from the `end` of `edge`.
const fn at(edge: u32, end: bool, side: i32) -> Point {
    Point { edge, end, side }
}

/// `a`, the first end of `ab` and `ac`: the `u = 0` fan's apex.
const A: Point = at(0, false, 0);

/// `b` as `bc` takes it, its first end: the `u = 1` fan's apex.
const B: Point = at(2, false, 0);

/// The points of the `u = 0` edge's vertices, the `k`th at `v = k / 11` (the fan from `a`): from
/// the corner at `a`, the corners of `ab`'s parallelogram, then of `ac`'s, each from `a`'s side
/// around to it, between `a`s, to the corner at `b`.
const LEFT: [Point; 12] = [
    A,
    at(0, false, 1),
    at(0, true, 1),
    at(0, true, -1),
    at(0, false, -1),
    A,
    at(1, false, 1),
    at(1, true, 1),
    at(1, true, -1),
    at(1, false, -1),
    A,
    B,
];

/// The points of the `u = 1` edge's vertices, the `k`th at `v = k / 6` (the fan from `b`): from
/// the corner at `a` to `b`, the corners of `bc`'s parallelogram the other way round from `b`'s
/// side, and the corner at `b`.
const RIGHT: [Point; 7] = [
    A,
    B,
    at(2, false, -1),
    at(2, true, -1),
    at(2, true, 1),
    at(2, false, 1),
    B,
];

/// A triangle's tessellation levels in the quad domain (the module docs): the outer levels of the
/// `u = 0`, `v = 0`, `u = 1` and `v = 1` edges, then the inner levels in `u` and `v`.
const EDGE_LEVELS: ([u32; 4], [u32; 2]) = (
    [LEFT.len() as u32 - 1, 1, RIGHT.len() as u32 - 1, 1],
    [3, 2],
);

/// One varying the stages carry: the vertex output's key and type, and whether every corner takes
/// the first control point's (a flat or integer fragment input).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandVarying {
    pub key: VaryingKey,
    pub ty: IoType,
    pub flat: bool,
}

/// What a band pipeline's synthesized stages carry and where its push range is (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineBand {
    /// What each patch is.
    pub primitive: BandPrimitive,
    /// The vertex stage's outputs, in output order.
    pub varyings: Vec<BandVarying>,
    /// The elements of the vertex function's `ClipDistance`, 0 for none.
    pub clip_distances: u32,
    /// The stages carry the vertex stage's `PointSize`: it writes one and the device enables
    /// `shaderTessellationAndGeometryPointSize`.
    pub point_size: bool,
    /// The byte offset of the evaluation stage's push range ([`LineBand::PUSH_BYTES`]), a
    /// multiple of 16, the `vec4`'s base alignment, which link places.
    pub push_offset: u32,
}

impl LineBand {
    /// The evaluation stage's push range: one `vec4`, `PushField::LineBand`.
    pub const PUSH_BYTES: u32 = 16;

    /// The facing a line's band's triangles take under Vulkan's default upper-left tessellation
    /// domain origin, whatever the segment's direction: a draw of lines sets it as its front face,
    /// so the band is front-facing, as Vulkan's own lines are. A triangle's band faces this way
    /// where the triangle faces front under it, else the other way (the module docs).
    pub const FRONT_FACE: vk::FrontFace = vk::FrontFace::CLOCKWISE;
}

/// The band stages of the render pipeline of stages `ifaces` (vertex, then fragment when it has
/// one), drawing each `primitive` as a patch, whose synthesized geometry stages are `geometry`
/// (`link::geometry`), drawn with vertex amplification up to `amplification`; its push offset left
/// 0 for link to place. Refused as the module docs state.
pub fn plan(
    ifaces: &[&StageInterface],
    (primitive, geometry): (BandPrimitive, bool),
    amplification: u32,
    p: &DeviceProfile,
) -> Result<LineBand, Error> {
    let f = p.fields();
    if !f.enabled.features.contains(&VkFeature::TessellationShader) {
        return Err(Unsupported::needs(
            Feature::WideLines,
            Capability::Feature(VkFeature::TessellationShader),
        )
        .into());
    }
    let not_implemented = || Error::from(Unsupported::not_implemented(Feature::WideLines));
    let vertex = ifaces
        .iter()
        .find(|i| i.stage == Stage::Vertex)
        .filter(|v| v.tessellation.is_none())
        .ok_or_else(not_implemented)?;
    let fragment = ifaces.iter().find(|i| i.stage == Stage::Fragment);
    let selects = vertex
        .io
        .builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::Layer | StageBuiltin::ViewportIndex));
    let weights = fragment.is_some_and(|f| {
        f.io.builtins
            .iter()
            .any(|b| matches!(b, StageBuiltin::BaryCoord { .. }))
            || f.io
                .inputs
                .iter()
                .any(|i| matches!(i.role, IoRole::Barycentric { .. }))
    });
    let multiview = amplification.min(f.limits.max_multiview_view_count.max(1)) > 1;
    if selects || weights || geometry || multiview {
        return Err(not_implemented());
    }
    let clip_distances = vertex
        .io
        .builtins
        .iter()
        .find_map(|b| match b {
            StageBuiltin::ClipDistance { count } => Some(*count),
            _ => None,
        })
        .unwrap_or(0);
    let point_size = f
        .enabled
        .features
        .contains(&VkFeature::ShaderTessellationAndGeometryPointSize)
        && vertex
            .io
            .builtins
            .iter()
            .any(|b| matches!(b, StageBuiltin::PointSize { .. }));
    let varyings: Vec<BandVarying> = vertex
        .io
        .outputs
        .iter()
        .map(|o| {
            let input = fragment.and_then(|f| f.io.inputs.iter().find(|i| i.key == o.key));
            let flat = o.ty.numeric != IoNumeric::Float
                || input.is_some_and(|input| {
                    !matches!(
                        input.role,
                        IoRole::Varying {
                            interpolation: Some(Interpolation::Smooth { .. }),
                            ..
                        } | IoRole::WindowDepth
                    )
                });
            BandVarying {
                key: o.key.clone(),
                ty: o.ty,
                flat,
            }
        })
        .collect();
    // Every location the evaluation stage writes, counted in whole locations as link counts the
    // vertex and fragment limits, with the position's.
    let components = 4 * (varyings.len() as u64 + 1) + u64::from(clip_distances);
    let available = u64::from(f.limits.max_tessellation_evaluation_output_components);
    if components > available {
        return Err(Unsupported::needs(
            Feature::WideLines,
            Capability::Limit {
                limit: VkLimit::MaxTessellationEvaluationOutputComponents,
                required: components,
                available,
            },
        )
        .into());
    }
    Ok(LineBand {
        primitive,
        varyings,
        clip_distances,
        point_size,
        push_offset: 0,
    })
}

/// A stage's per-vertex values: `Position`, `ClipDistance` and `PointSize` where the stages carry
/// them, then each varying, each an input and an output global.
struct Carried {
    position: (GlobalId, GlobalId),
    clip: Option<(GlobalId, GlobalId)>,
    point_size: Option<(GlobalId, GlobalId)>,
    varyings: Vec<(GlobalId, GlobalId)>,
}

/// Where an evaluation invocation places its vertex: its clip-space position, the control point
/// whose values it takes (`end`) moved by `t` toward control point `other` (the module docs).
struct Corner {
    position: Value,
    end: Value,
    other: Value,
    t: Value,
}

impl LineBand {
    /// The band's control and evaluation stages as final modules under link table `table`, with
    /// their requirements: verified by the owned verifier, serialized and relocated, as a
    /// compiled function's module is.
    pub fn stages(
        &self,
        table: &impl SymbolTable,
        p: &DeviceProfile,
        m: &mut Meter,
    ) -> Result<Vec<(StageModule, Requirements)>, Error> {
        let mut out = Vec::with_capacity(2);
        for (stage, module) in [
            (
                vk::ShaderStage::TESSELLATION_CONTROL,
                self.control_module()?,
            ),
            (
                vk::ShaderStage::TESSELLATION_EVALUATION,
                self.evaluation_module()?,
            ),
        ] {
            let stats = crate::spirv::verify::verify(&module, p, m)?;
            let module = crate::spirv::serialize::serialize_for(module, p)?;
            let words = crate::spirv::serialize::relocate(&module, table)?;
            out.push((
                StageModule {
                    stage,
                    entry_point: ENTRY_NAME.to_owned(),
                    spirv: words.into(),
                    specialization: SpecializationLayout::default(),
                    required_subgroup_size: None,
                    stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
                    reconvergence: Reconvergence::Guaranteed,
                    precision_notes: Vec::new(),
                    strategies: StrategyCensus::default(),
                    constant_blob: None,
                    rounded_colors: Vec::new(),
                    specialized_notes: Vec::new(),
                    own_pixel_textures: Vec::new(),
                    snapshot_textures: Vec::new(),
                    private_bytes: 0,
                    stats,
                },
                module.requirements,
            ));
        }
        Ok(out)
    }

    /// Declare the stage's per-vertex values, its inputs arrays of [`INPUT_ARRAY`] and its outputs
    /// arrays of `outputs` where `Some` (the control stage), else single values.
    fn declare(&self, mb: &mut ModuleBuilder, outputs: Option<u32>) -> BResult<Carried> {
        let f32_ = scalar(mb, IoNumeric::Float)?;
        let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
        let output_ty = |mb: &mut ModuleBuilder, ty: TypeId| match outputs {
            Some(n) => array(mb, ty, n),
            None => Ok(ty),
        };
        let input = array(mb, vec4, INPUT_ARRAY)?;
        let output = output_ty(mb, vec4)?;
        let position = (
            mb.builtin(B::Position, IoDir::Input, input)?,
            mb.builtin(B::Position, IoDir::Output, output)?,
        );
        let clip = match self.clip_distances {
            0 => None,
            n => {
                let clip = array(mb, f32_, n)?;
                let input = array(mb, clip, INPUT_ARRAY)?;
                let output = output_ty(mb, clip)?;
                Some((
                    mb.builtin(B::ClipDistance, IoDir::Input, input)?,
                    mb.builtin(B::ClipDistance, IoDir::Output, output)?,
                ))
            }
        };
        let point_size = match self.point_size {
            false => None,
            true => {
                let input = array(mb, f32_, INPUT_ARRAY)?;
                let output = output_ty(mb, f32_)?;
                Some((
                    mb.builtin(B::PointSize, IoDir::Input, input)?,
                    mb.builtin(B::PointSize, IoDir::Output, output)?,
                ))
            }
        };
        let packed = IoDecor {
            packed: true,
            ..IoDecor::default()
        };
        let mut varyings = Vec::with_capacity(self.varyings.len());
        for v in &self.varyings {
            let ty = varying_type(mb, v.ty)?;
            let input = array(mb, ty, INPUT_ARRAY)?;
            let output = output_ty(mb, ty)?;
            varyings.push((
                mb.io(v.key.clone(), IoDir::Input, input, packed)?,
                mb.io(v.key.clone(), IoDir::Output, output, packed)?,
            ));
        }
        Ok(Carried {
            position,
            clip,
            point_size,
            varyings,
        })
    }

    /// The modes both stages declare: the quad domain, equal spacing (every level is a whole
    /// number), and the vertex order [`FRONT_FACE`](Self::FRONT_FACE) follows from.
    fn domain_modes() -> Vec<ExecMode> {
        vec![
            ExecMode::Quads,
            ExecMode::SpacingEqual,
            ExecMode::VertexOrderCw,
        ]
    }

    /// The control stage (the module docs).
    pub fn control_module(&self) -> BResult<TypedModule> {
        let points = self.primitive.patch_control_points();
        let mut mb = ModuleBuilder::new();
        let carried = self.declare(&mut mb, Some(points))?;
        let u32_ = scalar(&mut mb, IoNumeric::Uint)?;
        let f32_ = scalar(&mut mb, IoNumeric::Float)?;
        let void = mb.ty(Type::Void)?;
        let invocation = mb.builtin(B::InvocationId, IoDir::Input, u32_)?;
        let outer_ty = array(&mut mb, f32_, 4)?;
        let inner_ty = array(&mut mb, f32_, 2)?;
        let outer = mb.builtin(B::TessLevelOuter, IoDir::Output, outer_ty)?;
        let inner = mb.builtin(B::TessLevelInner, IoDir::Output, inner_ty)?;
        let mut fb = mb.function(void)?;
        let from = fb.module().global_value(invocation)?;
        let at = fb.load(from, MemAccess::default())?;
        let pairs = std::iter::once(carried.position)
            .chain(carried.clip)
            .chain(carried.point_size)
            .chain(carried.varyings);
        for (input, output) in pairs {
            let base = fb.module().global_value(input)?;
            let ptr = fb.access_chain(base, &[at])?;
            let value = fb.load(ptr, MemAccess::default())?;
            let base = fb.module().global_value(output)?;
            let ptr = fb.access_chain(base, &[at])?;
            fb.store(ptr, value, MemAccess::default())?;
        }
        let (outer_levels, inner_levels) = match self.primitive {
            BandPrimitive::Lines => ([1; 4], [1; 2]),
            BandPrimitive::Triangles => EDGE_LEVELS,
        };
        for (levels, values) in [(outer, &outer_levels[..]), (inner, &inner_levels[..])] {
            let base = fb.module().global_value(levels)?;
            for (k, &level) in (0..).zip(values) {
                let value = kf(&mut fb, level as f32)?;
                let index = ku(&mut fb, k)?;
                let ptr = fb.access_chain(base, &[index])?;
                fb.store(ptr, value, MemAccess::default())?;
            }
        }
        fb.ret(None)?;
        let f = fb.finish()?;
        let mut modes = Self::domain_modes();
        modes.push(ExecMode::OutputVertices { n: points });
        mb.entry(ExecModel::TessellationControl, f, modes)?;
        mb.finalize()
    }

    /// The evaluation stage (the module docs).
    pub fn evaluation_module(&self) -> BResult<TypedModule> {
        let mut mb = ModuleBuilder::new();
        let carried = self.declare(&mut mb, None)?;
        let f32_ = scalar(&mut mb, IoNumeric::Float)?;
        let vec3 = mb.ty(Type::Vector { elem: f32_, len: 3 })?;
        let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
        let void = mb.ty(Type::Void)?;
        let coord = mb.builtin(B::TessCoord, IoDir::Input, vec3)?;
        let member = mb
            .push_member(PushField::LineBand, vec4, self.push_offset)?
            .index;
        let push = mb.push_global()?;
        let mut fb = mb.function(void)?;
        let from = fb.module().global_value(coord)?;
        let coord = fb.load(from, MemAccess::default())?;
        let u = fb.composite_extract(coord, &[0])?;
        let v = fb.composite_extract(coord, &[1])?;
        let base = fb.module().global_value(push)?;
        let index = ku(&mut fb, member)?;
        let ptr = fb.access_chain(base, &[index])?;
        let band = fb.load(ptr, MemAccess::default())?;
        let (positions, _) = carried.position;
        let corner = match self.primitive {
            BandPrimitive::Lines => line_corner(&mut fb, positions, (u, v), band)?,
            BandPrimitive::Triangles => edge_corner(&mut fb, positions, (u, v), band)?,
        };
        let (_, output) = carried.position;
        let to = fb.module().global_value(output)?;
        fb.store(to, corner.position, MemAccess::default())?;
        // The clip distances and float varyings moved with the endpoint; the flat ones and the
        // point size the first control point's.
        let mut values = Vec::new();
        if let Some((input, output)) = carried.clip {
            values.push((input, output, false, Some(self.clip_distances)));
        }
        if let Some((input, output)) = carried.point_size {
            values.push((input, output, true, None));
        }
        for (v, &(input, output)) in self.varyings.iter().zip(&carried.varyings) {
            values.push((input, output, v.flat, None));
        }
        let zero = ku(&mut fb, 0)?;
        for (input, output, flat, elements) in values {
            let value = match flat {
                true => load_at(&mut fb, input, zero)?,
                false => {
                    let at_end = load_at(&mut fb, input, corner.end)?;
                    let at_other = load_at(&mut fb, input, corner.other)?;
                    mix(&mut fb, (at_end, at_other, corner.t), elements)?
                }
            };
            let to = fb.module().global_value(output)?;
            fb.store(to, value, MemAccess::default())?;
        }
        fb.ret(None)?;
        let f = fb.finish()?;
        mb.entry(ExecModel::TessellationEvaluation, f, Self::domain_modes())?;
        mb.finalize()
    }
}

/// Element `at` of input array `global`.
fn load_at(fb: &mut FunctionBuilder<'_>, global: GlobalId, at: Value) -> BResult<Value> {
    let base = fb.module().global_value(global)?;
    let ptr = fb.access_chain(base, &[at])?;
    fb.load(ptr, MemAccess::default())
}

/// The minor axis `[nx, ny]` of the segment from clip-space `a` to `b` in framebuffer pixels,
/// through a viewport of half extent `(sx, sy)` (the module docs).
fn minor_axis(
    fb: &mut FunctionBuilder<'_>,
    (a, b): (Value, Value),
    (sx, sy): (Value, Value),
) -> BResult<[Value; 2]> {
    let ndc = |fb: &mut FunctionBuilder<'_>, p: Value| -> BResult<[Value; 2]> {
        let w = fb.composite_extract(p, &[3])?;
        let x = fb.composite_extract(p, &[0])?;
        let y = fb.composite_extract(p, &[1])?;
        Ok([fb.binary(BinOp::FDiv, x, w)?, fb.binary(BinOp::FDiv, y, w)?])
    };
    let [ax, ay] = ndc(fb, a)?;
    let [bx, by] = ndc(fb, b)?;
    let dx = fb.binary(BinOp::FSub, bx, ax)?;
    let dx = fb.binary(BinOp::FMul, dx, sx)?;
    let dy = fb.binary(BinOp::FSub, by, ay)?;
    let dy = fb.binary(BinOp::FMul, dy, sy)?;
    let ex = fb.glsl(GlslOp::FAbs, &[dx])?;
    let ey = fb.glsl(GlslOp::FAbs, &[dy])?;
    let x_major = fb.binary(BinOp::FOrdGreaterThanEqual, ex, ey)?;
    let sign_x = fb.glsl(GlslOp::FSign, &[dx])?;
    let sign_y = fb.glsl(GlslOp::FSign, &[dy])?;
    let minus_y = fb.unary(UnOp::FNegate, sign_y)?;
    let none = kf(fb, 0.0)?;
    let nx = fb.select(x_major, none, minus_y)?;
    let ny = fb.select(x_major, sign_x, none)?;
    Ok([nx, ny])
}

/// Clip-space `corner` offset by `across` framebuffer pixels along `[nx, ny]`, through a viewport
/// of half extent `(sx, sy)`, at the corner's `w`.
fn offset(
    fb: &mut FunctionBuilder<'_>,
    corner: Value,
    ([nx, ny], across): ([Value; 2], Value),
    (sx, sy): (Value, Value),
) -> BResult<Value> {
    let vec4 = fb.value_ty(corner)?;
    let w = fb.composite_extract(corner, &[3])?;
    let mut placed = Vec::with_capacity(4);
    for (lane, (n, s)) in [(nx, sx), (ny, sy)].into_iter().enumerate() {
        let offset = fb.binary(BinOp::FMul, n, across)?;
        let offset = fb.binary(BinOp::FDiv, offset, s)?;
        let offset = fb.binary(BinOp::FMul, offset, w)?;
        let at = fb.composite_extract(corner, &[lane as u32])?;
        placed.push(fb.binary(BinOp::FAdd, at, offset)?);
    }
    placed.push(fb.composite_extract(corner, &[2])?);
    placed.push(w);
    fb.composite_construct(vec4, &placed)
}

/// A line's corner `(u, v)` (the module docs): at endpoint `u`, `(2v - 1) h` pixels across the
/// segment, `band` the pushed `[sx, sy, h, 0]`.
fn line_corner(
    fb: &mut FunctionBuilder<'_>,
    positions: GlobalId,
    (u, v): (Value, Value),
    band: Value,
) -> BResult<Corner> {
    let sx = fb.composite_extract(band, &[0])?;
    let sy = fb.composite_extract(band, &[1])?;
    let h = fb.composite_extract(band, &[2])?;
    // The corner's endpoint (`u`), and the other.
    let half = kf(fb, 0.5)?;
    let second = fb.binary(BinOp::FOrdGreaterThan, u, half)?;
    let (zero, one) = (ku(fb, 0)?, ku(fb, 1)?);
    let end = fb.select(second, one, zero)?;
    let other = fb.select(second, zero, one)?;
    let first = load_at(fb, positions, zero)?;
    let last = load_at(fb, positions, one)?;
    let ends = [load_at(fb, positions, end)?, load_at(fb, positions, other)?];
    // Each endpoint moved to the least `w` (the module docs), the segment's NDC endpoints.
    let (first, _) = moved(fb, first, last)?;
    let (last, _) = moved(fb, last, first)?;
    let (corner, t) = moved(fb, ends[0], ends[1])?;
    let n = minor_axis(fb, (first, last), (sx, sy))?;
    // `2v - 1` half widths, in pixels.
    let two = kf(fb, 2.0)?;
    let unit = kf(fb, 1.0)?;
    let side = fb.binary(BinOp::FMul, v, two)?;
    let side = fb.binary(BinOp::FSub, side, unit)?;
    let across = fb.binary(BinOp::FMul, h, side)?;
    let position = offset(fb, corner, (n, across), (sx, sy))?;
    Ok(Corner {
        position,
        end,
        other,
        t,
    })
}

/// One end of a triangle's edge as the evaluation stage places it: its clip-space position, its
/// control point, the edge's other end's, and how far it moved toward it.
#[derive(Clone, Copy)]
struct End {
    position: Value,
    point: Value,
    other: Value,
    t: Value,
}

/// A triangle's corner at `(u, v)` (the module docs), `band` the pushed `[sx, sy, h, f]`.
fn edge_corner(
    fb: &mut FunctionBuilder<'_>,
    positions: GlobalId,
    (u, v): (Value, Value),
    band: Value,
) -> BResult<Corner> {
    let sx = fb.composite_extract(band, &[0])?;
    let sy = fb.composite_extract(band, &[1])?;
    let h = fb.composite_extract(band, &[2])?;
    let front = fb.composite_extract(band, &[3])?;
    let u32_ = scalar(fb.module(), IoNumeric::Uint)?;
    let f32_ = scalar(fb.module(), IoNumeric::Float)?;
    let (zero, one, two, three) = (ku(fb, 0)?, ku(fb, 1)?, ku(fb, 2)?, ku(fb, 3)?);
    let corners = [
        load_at(fb, positions, zero)?,
        load_at(fb, positions, one)?,
        load_at(fb, positions, two)?,
    ];
    // `a`, the vertex of the largest `w` (the first of equals), then `b` and `c` in order.
    let [w0, w1, w2] = corners.map(|p| fb.composite_extract(p, &[3]));
    let (w0, w1, w2) = (w0?, w1?, w2?);
    let ge01 = fb.binary(BinOp::FOrdGreaterThanEqual, w0, w1)?;
    let ge02 = fb.binary(BinOp::FOrdGreaterThanEqual, w0, w2)?;
    let first = fb.binary(BinOp::LogicalAnd, ge01, ge02)?;
    let ge12 = fb.binary(BinOp::FOrdGreaterThanEqual, w1, w2)?;
    let later = fb.select(ge12, one, two)?;
    let ia = fb.select(first, zero, later)?;
    let ib = fb.binary(BinOp::IAdd, ia, one)?;
    let ib = fb.binary(BinOp::UMod, ib, three)?;
    let ic = fb.binary(BinOp::IAdd, ia, two)?;
    let ic = fb.binary(BinOp::UMod, ic, three)?;
    let a = load_at(fb, positions, ia)?;
    let b = load_at(fb, positions, ib)?;
    let c = load_at(fb, positions, ic)?;
    // The edges' ends, each moved to the least `w` toward the other end (`a` never moves), and
    // their minor axes from the ends moved as a line's are.
    let none = kf(fb, 0.0)?;
    let a_end = |other| End {
        position: a,
        point: ia,
        other,
        t: none,
    };
    let (b_ab, t_b_ab) = moved(fb, b, a)?;
    let (c_ac, t_c_ac) = moved(fb, c, a)?;
    let (b_bc, t_b_bc) = moved(fb, b, c)?;
    let (c_bc, t_c_bc) = moved(fb, c, b)?;
    let (c_bc_line, _) = moved(fb, c, b_bc)?;
    let edges = [
        (
            [
                a_end(ib),
                End {
                    position: b_ab,
                    point: ib,
                    other: ia,
                    t: t_b_ab,
                },
            ],
            (a, b_ab),
        ),
        (
            [
                a_end(ic),
                End {
                    position: c_ac,
                    point: ic,
                    other: ia,
                    t: t_c_ac,
                },
            ],
            (a, c_ac),
        ),
        (
            [
                End {
                    position: b_bc,
                    point: ib,
                    other: ic,
                    t: t_b_bc,
                },
                End {
                    position: c_bc,
                    point: ic,
                    other: ib,
                    t: t_c_bc,
                },
            ],
            (b_bc, c_bc_line),
        ),
    ];
    // The triangle's facing: the sign of its vertices' `(x, y, w)` determinant scaled by
    // `sx sy`, the sign of its framebuffer area where it lies in front of the eye. Negative, or
    // zero under a clockwise front face (a zero-area triangle faces back), turns the band.
    let [p0, p1, p2] = corners.map(|p| -> BResult<[Value; 3]> {
        Ok([
            fb.composite_extract(p, &[0])?,
            fb.composite_extract(p, &[1])?,
            fb.composite_extract(p, &[3])?,
        ])
    });
    let ([x0, y0, q0], [x1, y1, q1], [x2, y2, q2]) = (p0?, p1?, p2?);
    let minor = |fb: &mut FunctionBuilder<'_>, (a, b, c, d): (Value, Value, Value, Value)| {
        let ad = fb.binary(BinOp::FMul, a, d)?;
        let bc = fb.binary(BinOp::FMul, b, c)?;
        fb.binary(BinOp::FSub, ad, bc)
    };
    let m0 = minor(fb, (y1, q1, y2, q2))?;
    let m1 = minor(fb, (x1, q1, x2, q2))?;
    let m2 = minor(fb, (x1, y1, x2, y2))?;
    let t0 = fb.binary(BinOp::FMul, x0, m0)?;
    let t1 = fb.binary(BinOp::FMul, y0, m1)?;
    let t2 = fb.binary(BinOp::FMul, q0, m2)?;
    let det = fb.binary(BinOp::FSub, t0, t1)?;
    let det = fb.binary(BinOp::FAdd, det, t2)?;
    let scale = fb.binary(BinOp::FMul, sx, sy)?;
    let area = fb.binary(BinOp::FMul, det, scale)?;
    let negative = fb.binary(BinOp::FOrdLessThan, area, none)?;
    let flat = fb.binary(BinOp::FOrdEqual, area, none)?;
    let clockwise = fb.binary(BinOp::FOrdGreaterThan, front, none)?;
    let flat_back = fb.binary(BinOp::LogicalAnd, flat, clockwise)?;
    let turns = fb.binary(BinOp::LogicalOr, negative, flat_back)?;
    let unit = kf(fb, 1.0)?;
    let minus = kf(fb, -1.0)?;
    let facing = fb.select(turns, minus, unit)?;
    let mut normals = Vec::with_capacity(3);
    for (_, line) in &edges {
        let [nx, ny] = minor_axis(fb, *line, (sx, sy))?;
        normals.push([
            fb.binary(BinOp::FMul, nx, facing)?,
            fb.binary(BinOp::FMul, ny, facing)?,
        ]);
    }
    // The vertex's point (`LEFT`, `RIGHT`): the `u = 0` edge's `k`th, the `u = 1` edge's, or an
    // inner vertex, `a` at `u = 1/3` and `b` at `2/3`.
    let quarter = kf(fb, 0.25)?;
    let three_quarters = kf(fb, 0.75)?;
    let half = kf(fb, 0.5)?;
    let left = fb.binary(BinOp::FOrdLessThan, u, quarter)?;
    let right = fb.binary(BinOp::FOrdGreaterThan, u, three_quarters)?;
    let step = |fb: &mut FunctionBuilder<'_>, segments: usize| -> BResult<Value> {
        let n = kf(fb, segments as f32)?;
        let k = fb.binary(BinOp::FMul, v, n)?;
        let k = fb.glsl(GlslOp::Round, &[k])?;
        fb.convert(ConvOp::ConvertFToU, k, u32_)
    };
    let k_left = step(fb, LEFT.len() - 1)?;
    let k_right = step(fb, RIGHT.len() - 1)?;
    let right_first = ku(fb, LEFT.len() as u32)?;
    let k_right = fb.binary(BinOp::IAdd, k_right, right_first)?;
    let b_at = ku(fb, (LEFT.len() - 1) as u32)?;
    let inner_first = fb.binary(BinOp::FOrdLessThan, u, half)?;
    let inner = fb.select(inner_first, zero, b_at)?;
    let at = fb.select(right, k_right, inner)?;
    let at = fb.select(left, k_left, at)?;
    debug_assert_eq!((LEFT[0], LEFT[LEFT.len() - 1]), (A, B));
    let mut code = ku(fb, LEFT[0].code())?;
    for (k, point) in (0..).zip(LEFT.iter().chain(&RIGHT)).skip(1) {
        let k = ku(fb, k)?;
        let is = fb.binary(BinOp::IEqual, at, k)?;
        let point = ku(fb, point.code())?;
        code = fb.select(is, point, code)?;
    }
    let six = ku(fb, 6)?;
    let edge = fb.binary(BinOp::UDiv, code, six)?;
    let rest = fb.binary(BinOp::UMod, code, six)?;
    let second = fb.binary(BinOp::UGreaterThanEqual, rest, three)?;
    let side = fb.binary(BinOp::UMod, rest, three)?;
    let side = fb.convert(ConvOp::ConvertUToF, side, f32_)?;
    let side = fb.binary(BinOp::FSub, side, unit)?;
    // The point's edge's end, and its minor axis, chosen lane by lane.
    let is_ac = fb.binary(BinOp::IEqual, edge, one)?;
    let is_bc = fb.binary(BinOp::IEqual, edge, two)?;
    let pick = |fb: &mut FunctionBuilder<'_>, [ab, ac, bc]: [Value; 3]| -> BResult<Value> {
        let x = fb.select(is_ac, ac, ab)?;
        fb.select(is_bc, bc, x)
    };
    let ends = |e: usize| {
        let [start, end] = edges[e].0;
        (start, end)
    };
    let mut chosen = Vec::with_capacity(4);
    for lane in 0..4u32 {
        let mut by_edge = [none; 3];
        for (e, value) in by_edge.iter_mut().enumerate() {
            let (start, end) = ends(e);
            let start = fb.composite_extract(start.position, &[lane])?;
            let end = fb.composite_extract(end.position, &[lane])?;
            *value = fb.select(second, end, start)?;
        }
        chosen.push(pick(fb, by_edge)?);
    }
    let vec4 = fb.value_ty(a)?;
    let point = fb.composite_construct(vec4, &chosen)?;
    let mut fields = [none; 3];
    for (field, get) in fields.iter_mut().zip([
        (|e: End| e.point) as fn(End) -> Value,
        |e: End| e.other,
        |e: End| e.t,
    ]) {
        let mut by_edge = [none; 3];
        for (e, value) in by_edge.iter_mut().enumerate() {
            let (start, end) = ends(e);
            *value = fb.select(second, get(end), get(start))?;
        }
        *field = pick(fb, by_edge)?;
    }
    let [end, other, t] = fields;
    let nx = pick(fb, [normals[0][0], normals[1][0], normals[2][0]])?;
    let ny = pick(fb, [normals[0][1], normals[1][1], normals[2][1]])?;
    let across = fb.binary(BinOp::FMul, h, side)?;
    let position = offset(fb, point, ([nx, ny], across), (sx, sy))?;
    Ok(Corner {
        position,
        end,
        other,
        t,
    })
}

/// The type a stage declares a varying of `ty` with.
fn varying_type(mb: &mut ModuleBuilder, ty: IoType) -> BResult<TypeId> {
    let elem = scalar(mb, ty.numeric)?;
    match ty.components {
        1 => Ok(elem),
        c => mb.ty(Type::Vector { elem, len: c }),
    }
}

/// Clip-space endpoint `e` of the segment to `o`, moved along it to `w =` [`LEAST_W`] where its
/// `w` is less and `o`'s more, and the fraction `t` of the segment it moved (0 where it did not;
/// 1 at most, where both lie below).
fn moved(fb: &mut FunctionBuilder<'_>, e: Value, o: Value) -> BResult<(Value, Value)> {
    let we = fb.composite_extract(e, &[3])?;
    let wo = fb.composite_extract(o, &[3])?;
    let least = kf(fb, LEAST_W)?;
    let zero = kf(fb, 0.0)?;
    let one = kf(fb, 1.0)?;
    let behind = fb.binary(BinOp::FSub, least, we)?;
    let toward = fb.binary(BinOp::FSub, wo, we)?;
    let is_behind = fb.binary(BinOp::FOrdGreaterThan, behind, zero)?;
    let rises = fb.binary(BinOp::FOrdGreaterThan, toward, zero)?;
    let moves = fb.binary(BinOp::LogicalAnd, is_behind, rises)?;
    // A divisor that is not positive divides nothing kept.
    let divisor = fb.select(rises, toward, one)?;
    let t = fb.binary(BinOp::FDiv, behind, divisor)?;
    let t = fb.glsl(GlslOp::FMin, &[t, one])?;
    let t = fb.select(moves, t, zero)?;
    let moved = mix(fb, (e, o, t), Some(4))?;
    Ok((moved, t))
}

/// `a + t (b - a)` of float values `a` and `b` of `elements` lanes (an array's or a vector's;
/// `None` for a scalar or a vector whose type the values give), `t` a scalar, each lane through
/// `FMix`, which is `a` at `t = 0`.
fn mix(
    fb: &mut FunctionBuilder<'_>,
    (a, b, t): (Value, Value, Value),
    elements: Option<u32>,
) -> BResult<Value> {
    let ty = fb.value_ty(a)?;
    let lanes = match elements {
        Some(n) => n,
        None => match fb.module().types().get(ty)? {
            Type::Vector { len, .. } => u32::from(*len),
            _ => return fb.glsl(GlslOp::FMix, &[a, b, t]),
        },
    };
    let mut out = Vec::with_capacity(lanes as usize);
    for lane in 0..lanes {
        let x = fb.composite_extract(a, &[lane])?;
        let y = fb.composite_extract(b, &[lane])?;
        out.push(fb.glsl(GlslOp::FMix, &[x, y, t])?);
    }
    fb.composite_construct(ty, &out)
}
