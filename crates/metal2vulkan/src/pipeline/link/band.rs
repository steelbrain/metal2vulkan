//! The synthesized stages of a render pipeline variant that draws its lines as bands
//! (`VariantKey::line_band`): Apple's `setLineWidth:` at a width other than 1 on a device whose own
//! lines do not draw it (the runtime's `render/line_width.rs` chooses the route).
//!
//! Metal draws a line of width `w` (its quantized width, a multiple of 1/16 in `[0, 16]`) as the
//! parallelogram Vulkan's non-strict wide lines may be (the Vulkan specification's line segments,
//! `nonStrictWideLinesUseParallelogram`): the segment's major axis is the larger of its
//! framebuffer extents, x on a tie; the parallelogram's two minor-axis edges pass through the
//! endpoints, each `w` long and centred on its endpoint, and its other two edges are parallel to
//! the segment; a pixel is drawn where its centre lies inside, by the polygon rules' ties (Metal
//! probes on the Apple M3 Max, macOS 27.0.1). At width 0 Metal draws exactly the pixel centres on
//! the segment, by the same end rule.
//!
//! The variant's pipeline draws each line as one patch of two control points (a `PATCH_LIST`,
//! the line's vertices in order) through two stages link adds between the vertex and fragment
//! stages:
//!
//! - the **control stage** ([`LineBand::control_module`]), one invocation per control point
//!   (`OutputVertices` 2), copies every per-vertex value its vertex writes (`Position`, clip
//!   distances, `PointSize` where the device's tessellation stages take it, and every varying),
//!   so its inputs are the vertex stage's outputs entire, as an implementation that lays a stage
//!   interface out as one record needs (MoltenVK lays a vertex stage's outputs out so when
//!   tessellation follows); and it writes every tessellation level 1 in the quad domain, so the
//!   tessellator makes the patch's four corners and two triangles;
//! - the **evaluation stage** ([`LineBand::evaluation_module`]) places corner `(u, v)` at endpoint
//!   `u` (0 the first, 1 the second), offset across the segment by `(2v - 1) h` framebuffer pixels
//!   along its minor axis, `h` half the width: `n = (0, sign dx)` for an x-major segment and
//!   `(-sign dy, 0)` otherwise, `(dx, dy)` the segment's framebuffer extent, so the two triangles
//!   cover the parallelogram exactly and wind alike in every direction ([`LineBand::FRONT_FACE`]).
//!   The offset is taken to clip space through the draw's Vulkan viewport 0, whose half extent
//!   `(sx, sy)` the draw pushes beside `h` (`PushField::LineBand`, the stage's own push range).
//!   Width 0 takes `h = 2^-subPixelPrecisionBits`, one unit of the device's subpixel grid: a pixel
//!   centre on the segment lies inside, one off it by at least that much outside.
//!
//! An endpoint at `w < 2^-20` (at or behind the eye) is first moved along the segment to `w =
//! 2^-20`, its clip distances and float varyings with it, so the band's direction is the visible
//! segment's; the corners' triangles are then clipped as any triangle is. A segment that crosses
//! the near or far plane is cut where those triangles are, which can differ from Metal's segment
//! clipped first by part of a pixel at the cut.
//!
//! A corner's float varyings and clip distances are its endpoint's, so they interpolate along the
//! segment and are constant across it; a varying the fragment stage reads flat (or an integer
//! one), and the point size, is the first control point's at every corner, the line's provoking
//! vertex. What Metal's band fragments read is not measured.
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
use crate::pipeline::interface::StageInterface;
use crate::pipeline::link::geometry::{array, kf, ku, scalar};
use crate::pipeline::stage_io::{Interpolation, IoNumeric, IoRole, IoType, StageBuiltin};
use crate::pipeline::sym::{PushField, Stage, VaryingKey};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ExecMode, ExecModel, GlslOp, IoDecor, IoDir, MemAccess, Type, TypedModule, UnOp,
};
use crate::spirv::serialize::{SymbolTable, ENTRY_NAME};
use crate::spv::BuiltIn as B;
use crate::vk;

/// The input arrays' length: `gl_MaxPatchVertices`, the smallest `maxTessellationPatchSize`
/// Vulkan allows.
const INPUT_ARRAY: u32 = 32;

/// The least clip-space `w` an endpoint is drawn at (the module docs).
const LEAST_W: f32 = 1.0 / 1_048_576.0;

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

    /// A patch's control points: a line's two vertices.
    pub const PATCH_CONTROL_POINTS: u32 = 2;

    /// The facing the band's triangles take under Vulkan's default upper-left tessellation domain
    /// origin, whatever the segment's direction: a draw sets it as its front face, so the band is
    /// front-facing, as Vulkan's own lines are.
    pub const FRONT_FACE: vk::FrontFace = vk::FrontFace::CLOCKWISE;
}

/// The band stages of the render pipeline of stages `ifaces` (vertex, then fragment when it has
/// one), whose synthesized geometry stages are `geometry` (`link::geometry`), drawn with vertex
/// amplification up to `amplification`; its push offset left 0 for link to place. Refused as the
/// module docs state.
pub fn plan(
    ifaces: &[&StageInterface],
    geometry: bool,
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

    /// The modes both stages declare: the quad domain, equal spacing (every level is 1), and the
    /// vertex order [`FRONT_FACE`](Self::FRONT_FACE) follows from.
    fn domain_modes() -> Vec<ExecMode> {
        vec![
            ExecMode::Quads,
            ExecMode::SpacingEqual,
            ExecMode::VertexOrderCw,
        ]
    }

    /// The control stage (the module docs).
    pub fn control_module(&self) -> BResult<TypedModule> {
        let mut mb = ModuleBuilder::new();
        let carried = self.declare(&mut mb, Some(Self::PATCH_CONTROL_POINTS))?;
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
        let one = kf(&mut fb, 1.0)?;
        for (levels, n) in [(outer, 4), (inner, 2)] {
            let base = fb.module().global_value(levels)?;
            for k in 0..n {
                let index = ku(&mut fb, k)?;
                let ptr = fb.access_chain(base, &[index])?;
                fb.store(ptr, one, MemAccess::default())?;
            }
        }
        fb.ret(None)?;
        let f = fb.finish()?;
        let mut modes = Self::domain_modes();
        modes.push(ExecMode::OutputVertices {
            n: Self::PATCH_CONTROL_POINTS,
        });
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
        let sx = fb.composite_extract(band, &[0])?;
        let sy = fb.composite_extract(band, &[1])?;
        let h = fb.composite_extract(band, &[2])?;
        // The corner's endpoint (`u`), and the other.
        let half = kf(&mut fb, 0.5)?;
        let second = fb.binary(BinOp::FOrdGreaterThan, u, half)?;
        let (zero, one) = (ku(&mut fb, 0)?, ku(&mut fb, 1)?);
        let end = fb.select(second, one, zero)?;
        let other = fb.select(second, zero, one)?;
        let load = |fb: &mut FunctionBuilder<'_>, global: GlobalId, at: Value| {
            let base = fb.module().global_value(global)?;
            let ptr = fb.access_chain(base, &[at])?;
            fb.load(ptr, MemAccess::default())
        };
        let (positions, _) = carried.position;
        let first = load(&mut fb, positions, zero)?;
        let last = load(&mut fb, positions, one)?;
        let ends = [
            load(&mut fb, positions, end)?,
            load(&mut fb, positions, other)?,
        ];
        // Each endpoint moved to the least `w` (the module docs), the segment's NDC endpoints.
        let (first, _) = moved(&mut fb, first, last)?;
        let (last, _) = moved(&mut fb, last, first)?;
        let (corner, t) = moved(&mut fb, ends[0], ends[1])?;
        let ndc = |fb: &mut FunctionBuilder<'_>, p: Value| -> BResult<[Value; 2]> {
            let w = fb.composite_extract(p, &[3])?;
            let x = fb.composite_extract(p, &[0])?;
            let y = fb.composite_extract(p, &[1])?;
            Ok([fb.binary(BinOp::FDiv, x, w)?, fb.binary(BinOp::FDiv, y, w)?])
        };
        let [ax, ay] = ndc(&mut fb, first)?;
        let [bx, by] = ndc(&mut fb, last)?;
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
        let none = kf(&mut fb, 0.0)?;
        let nx = fb.select(x_major, none, minus_y)?;
        let ny = fb.select(x_major, sign_x, none)?;
        // `2v - 1` half widths, in pixels, then in clip space at the corner's `w`.
        let two = kf(&mut fb, 2.0)?;
        let unit = kf(&mut fb, 1.0)?;
        let side = fb.binary(BinOp::FMul, v, two)?;
        let side = fb.binary(BinOp::FSub, side, unit)?;
        let across = fb.binary(BinOp::FMul, h, side)?;
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
        let position = fb.composite_construct(vec4, &placed)?;
        let (_, output) = carried.position;
        let to = fb.module().global_value(output)?;
        fb.store(to, position, MemAccess::default())?;
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
        for (input, output, flat, elements) in values {
            let value = match flat {
                true => load(&mut fb, input, zero)?,
                false => {
                    let at_end = load(&mut fb, input, end)?;
                    let at_other = load(&mut fb, input, other)?;
                    mix(&mut fb, (at_end, at_other, t), elements)?
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
