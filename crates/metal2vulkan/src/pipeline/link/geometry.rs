//! The synthesized geometry stages of a render pipeline whose fragment function reads
//! `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric`, or whose vertex
//! function selects its primitive's render target slice or viewport on a device that cannot store
//! `Layer` or `ViewportIndex` from a vertex function.
//!
//! Such a fragment function reads the weights from a `float2` center varying of each form it reads
//! (`IoRole::Barycentric`, perspective-correct or not), and a `float3` read computes the third as
//! `1 - (x + y)`, as Metal does (`lower::stage_io`). Metal's first two weights are, bit for bit,
//! the interpolation of a varying that holds `(1, 0)`, `(0, 1)` and `(0, 0)` at a triangle's
//! vertices in its vertex order, which is Vulkan's first-vertex order: a list's in index order, a
//! strip's even triangle `(i, i + 1, i + 2)` and odd one `(i, i + 2, i + 1)`, the parity restarting
//! after a primitive restart; a line's `(1 - t, t)` from its first vertex; a point's `(1, 0)`
//! (Metal probes on the Apple M3 Max, macOS 27.0.1).
//!
//! Such a vertex function stores the `Layer` or `ViewportIndex` it would store to a flat `uint`
//! varying instead (`IoRole::SelectedLayer`, `IoRole::SelectedViewport`): on a device without
//! `shaderOutputLayer`, or with `multiViewport` but without `shaderOutputViewportIndex`. Metal
//! renders a primitive into the slice and through the viewport of its first vertex, in the same
//! order, whatever its other vertices write, and holds that vertex's index below the pass's length
//! or the draw's viewport count (Metal probes on the Apple M3 Max, macOS 27.0.1: triangle lists in
//! each order of three different indices, strips, lines and points, indices in and past range).
//!
//! A geometry stage receives a primitive's vertices in that order (measured on RADV, NVIDIA, ANV
//! and lavapipe, every topology), so link adds one between the vertex and fragment stages
//! ([`module`](GeometryStages::module)) that emits the primitive's vertices in the order it
//! receives them:
//!
//! - one invocation per input primitive, of a topology class: points (`InputPoints`,
//!   `OutputPoints`), lines (`InputLines`, `OutputLineStrip`) or triangles (`Triangles`,
//!   `OutputTriangleStrip`), emitting one primitive of the same vertices, so its winding, and
//!   with it its facing, is the input's, and its provoking vertex, the one a flat input reads,
//!   stays the first;
//! - each vertex's `Position` and clip distances, and in the point class its point size, as the
//!   vertex function wrote them;
//! - each vertex output the fragment stage reads, at its location and component;
//! - `PrimitiveId` from its input `PrimitiveId` where the fragment stage reads it: the count
//!   Vulkan gives the fragment stage without the stage;
//! - the weights, at the locations link gives them after the vertex outputs' (`link::varyings`);
//! - `Layer` and `ViewportIndex`, each the selected index of the primitive's first vertex at every
//!   vertex it emits: Metal's index, and the one value per primitive Vulkan requires of a stage
//!   that writes either ("it must write the same value to Layer for all output vertices of a given
//!   primitive", and to `ViewportIndex` alike: the Vulkan specification's built-in variables).
//!   `ViewportIndex` there needs `MultiViewport`, which a device that selects through the stage
//!   has.
//!
//! The point class writes a point size exactly where the profile enables
//! `shaderTessellationAndGeometryPointSize`, under which a geometry stage that writes none draws
//! points of an undefined size: the vertex function's, or 1, its default (Metal sizes a point of
//! the point class 1 where the vertex function writes none). Without the feature it writes none,
//! and Vulkan draws points of size 1.
//!
//! A pipeline draws a class other than its input primitive topology class's where Metal does (the
//! unspecified class draws every class, and a draw of another class draws as plain Metal does), so
//! it carries the three stages, the point class's, the line class's and the triangle class's, in
//! that order before its fragment stage (or last, without one): a `VkPipeline` runs the one of its
//! draw's class (`RenderPipeline::runs_stage`). A vertex function's variant for a pass of one layer
//! (`LoweringInputs::dropped_vertex_layer`) selects no slice, so a pipeline of it carries no stage
//! for its slice.
//!
//! Refused, as `Unsupported{Barycentrics}` where the fragment stage reads the weights and as
//! `Unsupported{LayeredRendering}` otherwise:
//!
//! - a post-tessellation vertex function, and a mesh render pipeline that reads the weights:
//!   not implemented; and a vertex function that writes `[[render_target_array_index]]` or
//!   `[[viewport_array_index]]` to its builtin beside a fragment function that reads the weights,
//!   whose `Layer` and `ViewportIndex` a geometry stage cannot read: not implemented;
//! - a pipeline drawn as multiview (vertex amplification above 1 within `maxMultiviewViewCount`,
//!   where no stage declares `Layer`: such a pipeline is never drawn as views) on a device without
//!   `multiviewGeometryShader`;
//! - a pipeline of the point or the unspecified class whose vertex function writes
//!   `[[point_size]]`, on a device without `shaderTessellationAndGeometryPointSize`;
//! - a stage past `maxGeometryInputComponents`, `maxGeometryOutputComponents` or
//!   `maxGeometryTotalOutputComponents`, counted in whole locations as link counts the vertex and
//!   fragment limits.
//!
//! Each needs `fragmentShaderBarycentric` instead, or `shaderOutputLayer` and
//! `shaderOutputViewportIndex` for the indices the stage selects. A device without
//! `geometryShader` either is refused at the function (`lower::stage_io`).

use crate::budget::Meter;
use crate::device::requirements::{Requirements, VkFeature, VkLimit};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::metal::PrimitiveTopologyClass;
use crate::pipeline::artifact::{Reconvergence, SpecializationLayout, StageModule, StrategyCensus};
use crate::pipeline::interface::StageInterface;
use crate::pipeline::link::varyings::Locations;
use crate::pipeline::stage_io::{IoNumeric, IoRole, IoType, StageBuiltin};
use crate::pipeline::sym::{Stage, VaryingKey};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    ConstBits, Constant, ExecMode, ExecModel, IoDecor, IoDir, MemAccess, Type, TypedModule,
};
use crate::spirv::serialize::{SymbolTable, ENTRY_NAME};
use crate::spv::BuiltIn as B;
use crate::vk;

/// The topology classes a pipeline's geometry stages are made for, in their order among its
/// stages.
pub const CLASSES: [PrimitiveTopologyClass; 3] = [
    PrimitiveTopologyClass::Point,
    PrimitiveTopologyClass::Line,
    PrimitiveTopologyClass::Triangle,
];

/// What a pipeline's geometry stages pass through and write (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryStages {
    /// The vertex outputs the fragment stage reads, each with its type, in output order.
    pub varyings: Vec<(VaryingKey, IoType)>,
    /// The forms of the weights the fragment stage reads: `true` perspective-correct.
    pub forms: Vec<bool>,
    /// The stages store `Layer` from the vertex function's selected slice
    /// (`IoRole::SelectedLayer`).
    pub layer: bool,
    /// The stages store `ViewportIndex` from the vertex function's selected viewport
    /// (`IoRole::SelectedViewport`).
    pub viewport: bool,
    /// The elements of the vertex function's `ClipDistance`, 0 for none.
    pub clip_distances: u32,
    /// The vertex function declares `PointSize`.
    pub point_size: bool,
    /// The point class's stage writes `PointSize` (the profile enables
    /// `shaderTessellationAndGeometryPointSize`).
    pub writes_point_size: bool,
    /// The fragment stage reads `PrimitiveId`.
    pub primitive_id: bool,
    /// The pipeline is drawn as multiview.
    pub multiview: bool,
}

/// Why the stages are made: the fragment stage reads the weights, or else the vertex function
/// selects `Layer` or `ViewportIndex` through them.
#[derive(Clone, Copy)]
struct Reason {
    weights: bool,
    layer: bool,
    viewport: bool,
}

impl Reason {
    /// Refused as not implemented on every device. The refusal names `Barycentrics` where the
    /// fragment stage reads the weights, else `LayeredRendering`.
    fn not_implemented(self) -> Error {
        if self.weights {
            Unsupported::not_implemented(Feature::Barycentrics).into()
        } else {
            Unsupported::not_implemented(Feature::LayeredRendering).into()
        }
    }

    /// The refusal as needing the features that make the stages needless, or `capability`.
    fn refused(self, capability: Capability) -> Error {
        let instead = if self.weights {
            vec![Capability::Feature(VkFeature::FragmentShaderBarycentric)]
        } else {
            let mut instead = Vec::new();
            if self.layer {
                instead.push(Capability::Feature(VkFeature::ShaderOutputLayer));
            }
            if self.viewport {
                instead.push(Capability::Feature(VkFeature::ShaderOutputViewportIndex));
            }
            instead
        };
        let needs = vec![instead, vec![capability]];
        if self.weights {
            Unsupported::needs_any_of(Feature::Barycentrics, needs).into()
        } else {
            Unsupported::needs_any_of(Feature::LayeredRendering, needs).into()
        }
    }
}

/// The geometry stages of the render pipeline of stages `ifaces` (vertex, then fragment when it
/// has one), drawn as `class` with vertex amplification up to `amplification`, where its fragment
/// stage reads the weights as varyings or its vertex function selects an index through them;
/// `None` where neither. Refused as the module docs state. A mesh pipeline (`mesh`) whose fragment
/// stage reads the weights is not implemented.
pub fn plan(
    ifaces: &[&StageInterface],
    class: PrimitiveTopologyClass,
    amplification: u32,
    mesh: bool,
    p: &DeviceProfile,
) -> Result<Option<GeometryStages>, Error> {
    let fragment = ifaces.iter().find(|i| i.stage == Stage::Fragment);
    let vertex = ifaces.iter().find(|i| i.stage == Stage::Vertex);
    let mut forms: Vec<bool> = fragment
        .iter()
        .flat_map(|f| &f.io.inputs)
        .filter_map(|v| match v.role {
            IoRole::Barycentric { perspective } => Some(perspective),
            _ => None,
        })
        .collect();
    forms.sort_unstable();
    forms.dedup();
    let selects =
        |role: IoRole| !mesh && vertex.is_some_and(|v| v.io.outputs.iter().any(|o| o.role == role));
    let why = Reason {
        weights: !forms.is_empty(),
        layer: selects(IoRole::SelectedLayer),
        viewport: selects(IoRole::SelectedViewport),
    };
    if !why.weights && !why.layer && !why.viewport {
        return Ok(None);
    }
    let vertex = match vertex {
        Some(v) if !mesh && v.tessellation.is_none() => v,
        _ => return Err(why.not_implemented()),
    };
    let builtins = &vertex.io.builtins;
    // A stage after the vertex function stores the indices Vulkan uses: a vertex function's own
    // `Layer` or `ViewportIndex` would not reach the rasterizer.
    if builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::Layer | StageBuiltin::ViewportIndex))
    {
        return Err(why.not_implemented());
    }
    let f = p.fields();
    let enabled = &f.enabled.features;
    // Drawn as multiview where its views fit the device's, as `compile::api` requires them, unless
    // a stage declares `Layer`: the binder draws such a pipeline's amplifications one at a time.
    let declares_layer = why.layer
        || fragment.is_some_and(|f| {
            f.io.builtins.contains(&StageBuiltin::Layer)
                || f.io.zero_indices.contains(&StageBuiltin::Layer)
        });
    let multiview =
        !declares_layer && amplification.min(f.limits.max_multiview_view_count.max(1)) > 1;
    if multiview && !enabled.contains(&VkFeature::MultiviewGeometryShader) {
        return Err(why.refused(Capability::Feature(VkFeature::MultiviewGeometryShader)));
    }
    let point_size = builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::PointSize { .. }));
    let written_size = builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::PointSize { default: false }));
    let writes_point_size = enabled.contains(&VkFeature::ShaderTessellationAndGeometryPointSize);
    let draws_points = matches!(
        class,
        PrimitiveTopologyClass::Point | PrimitiveTopologyClass::Unspecified
    );
    if draws_points && written_size && !writes_point_size {
        return Err(why.refused(Capability::Feature(
            VkFeature::ShaderTessellationAndGeometryPointSize,
        )));
    }
    let clip_distances = builtins
        .iter()
        .find_map(|b| match b {
            StageBuiltin::ClipDistance { count } => Some(*count),
            _ => None,
        })
        .unwrap_or(0);
    let varyings = vertex
        .io
        .outputs
        .iter()
        .filter(|o| fragment.is_some_and(|f| f.io.inputs.iter().any(|i| i.key == o.key)))
        .map(|o| (o.key.clone(), o.ty))
        .collect();
    Ok(Some(GeometryStages {
        varyings,
        forms,
        layer: why.layer,
        viewport: why.viewport,
        clip_distances,
        point_size,
        writes_point_size,
        primitive_id: fragment.is_some_and(|f| f.io.builtins.contains(&StageBuiltin::PrimitiveId)),
        multiview,
    }))
}

impl GeometryStages {
    /// Why the stages are made ([`Reason`]).
    fn reason(&self) -> Reason {
        Reason {
            weights: !self.forms.is_empty(),
            layer: self.layer,
            viewport: self.viewport,
        }
    }

    /// The selected indices the stages read, each with its key and the builtin it stores.
    fn selected(&self) -> impl Iterator<Item = (VaryingKey, crate::spv::BuiltIn)> {
        let layer = self.layer.then(|| (VaryingKey::selected_layer(), B::Layer));
        let viewport = self
            .viewport
            .then(|| (VaryingKey::selected_viewport(), B::ViewportIndex));
        layer.into_iter().chain(viewport)
    }

    /// Refuses stages whose inputs or outputs at `locations` pass the device's geometry limits
    /// (the module docs).
    pub fn check_limits(&self, locations: &Locations, p: &DeviceProfile) -> Result<(), Error> {
        let l = &p.fields().limits;
        let end = |keys: &mut dyn Iterator<Item = &VaryingKey>| -> Result<u64, Error> {
            let mut end = 0u64;
            for k in keys {
                let (location, _) = locations.get(k).copied().ok_or_else(|| {
                    Error::Internal(crate::error::InternalError::new(
                        crate::error::InternalKind::Invariant,
                        "a geometry stage's varying without its location",
                    ))
                })?;
                end = end.max(u64::from(location) + 1);
            }
            Ok(end)
        };
        let weights: Vec<VaryingKey> = self
            .forms
            .iter()
            .map(|&p| VaryingKey::barycentric(p))
            .collect();
        let selected: Vec<VaryingKey> = self.selected().map(|(k, _)| k).collect();
        let inputs = end(&mut self.varyings.iter().map(|(k, _)| k).chain(&selected))?;
        let outputs = end(&mut self.varyings.iter().map(|(k, _)| k).chain(&weights))?;
        let vertices = 3u64;
        for (limit, required, available) in [
            (
                VkLimit::MaxGeometryInputComponents,
                inputs * 4,
                l.max_geometry_input_components,
            ),
            (
                VkLimit::MaxGeometryOutputComponents,
                outputs * 4,
                l.max_geometry_output_components,
            ),
            (
                VkLimit::MaxGeometryTotalOutputComponents,
                outputs * 4 * vertices,
                l.max_geometry_total_output_components,
            ),
        ] {
            if required > u64::from(available) {
                return Err(self.reason().refused(Capability::Limit {
                    limit,
                    required,
                    available: u64::from(available),
                }));
            }
        }
        Ok(())
    }

    /// The pipeline's geometry stages, one per class of [`CLASSES`] in its order, as final modules
    /// under link table `table`, with their requirements: verified by the owned verifier,
    /// serialized and relocated, as a compiled function's module is.
    pub fn stages(
        &self,
        table: &impl SymbolTable,
        p: &DeviceProfile,
        m: &mut Meter,
    ) -> Result<Vec<(StageModule, Requirements)>, Error> {
        let mut out = Vec::with_capacity(CLASSES.len());
        for class in CLASSES {
            let module = self.module(class)?;
            let stats = crate::spirv::verify::verify(&module, p, m)?;
            let module = crate::spirv::serialize::serialize_for(module, p)?;
            let words = crate::spirv::serialize::relocate(&module, table)?;
            let mut requirements = module.requirements;
            if self.multiview {
                requirements
                    .features
                    .insert(VkFeature::MultiviewGeometryShader);
            }
            out.push((
                StageModule {
                    stage: vk::ShaderStage::GEOMETRY,
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
                    snapshot_refusals: Vec::new(),
                    level_reads: Vec::new(),
                    private_bytes: 0,
                    stats,
                },
                requirements,
            ));
        }
        Ok(out)
    }

    /// The geometry stage of `class` (the module docs).
    pub fn module(&self, class: PrimitiveTopologyClass) -> BResult<TypedModule> {
        let (n, input, output) = match class {
            PrimitiveTopologyClass::Point => (1, ExecMode::InputPoints, ExecMode::OutputPoints),
            PrimitiveTopologyClass::Line => (2, ExecMode::InputLines, ExecMode::OutputLineStrip),
            _ => (3, ExecMode::Triangles, ExecMode::OutputTriangleStrip),
        };
        let mut mb = ModuleBuilder::new();
        let u32_ = scalar(&mut mb, IoNumeric::Uint)?;
        let f32_ = scalar(&mut mb, IoNumeric::Float)?;
        let vec2 = mb.ty(Type::Vector { elem: f32_, len: 2 })?;
        let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
        let void = mb.ty(Type::Void)?;
        let per_vertex = |mb: &mut ModuleBuilder, elem: TypeId| array(mb, elem, n);
        // Each value a vertex carries: its input, read per vertex, and its output.
        let mut carried: Vec<(GlobalId, GlobalId)> = Vec::new();
        let ty = per_vertex(&mut mb, vec4)?;
        carried.push((
            mb.builtin(B::Position, IoDir::Input, ty)?,
            mb.builtin(B::Position, IoDir::Output, vec4)?,
        ));
        if self.clip_distances > 0 {
            let clip = array(&mut mb, f32_, self.clip_distances)?;
            let ty = per_vertex(&mut mb, clip)?;
            carried.push((
                mb.builtin(B::ClipDistance, IoDir::Input, ty)?,
                mb.builtin(B::ClipDistance, IoDir::Output, clip)?,
            ));
        }
        let points = class == PrimitiveTopologyClass::Point && self.writes_point_size;
        // The vertex function's point size, or 1 where it declares none.
        let mut size = None;
        if points {
            let output = mb.builtin(B::PointSize, IoDir::Output, f32_)?;
            let input = match self.point_size {
                true => {
                    let ty = per_vertex(&mut mb, f32_)?;
                    Some(mb.builtin(B::PointSize, IoDir::Input, ty)?)
                }
                false => None,
            };
            size = Some((input, output));
        }
        let packed = IoDecor {
            packed: true,
            ..IoDecor::default()
        };
        for (key, io) in &self.varyings {
            let elem = scalar(&mut mb, io.numeric)?;
            let ty = match io.components {
                1 => elem,
                c => mb.ty(Type::Vector { elem, len: c })?,
            };
            let input_ty = per_vertex(&mut mb, ty)?;
            carried.push((
                mb.io(key.clone(), IoDir::Input, input_ty, packed)?,
                mb.io(key.clone(), IoDir::Output, ty, packed)?,
            ));
        }
        let mut weights = Vec::with_capacity(self.forms.len());
        for &perspective in &self.forms {
            let key = VaryingKey::barycentric(perspective);
            weights.push(mb.io(key, IoDir::Output, vec2, IoDecor::default())?);
        }
        // Each selected index: its flat per-vertex input, and the builtin it stores.
        let mut selected = Vec::new();
        for (key, builtin) in self.selected() {
            let ty = per_vertex(&mut mb, u32_)?;
            selected.push((
                mb.io(key, IoDir::Input, ty, packed)?,
                mb.builtin(builtin, IoDir::Output, u32_)?,
            ));
        }
        let primitive = match self.primitive_id {
            true => Some((
                mb.builtin(B::PrimitiveId, IoDir::Input, u32_)?,
                mb.builtin(B::PrimitiveId, IoDir::Output, u32_)?,
            )),
            false => None,
        };
        let mut fb = mb.function(void)?;
        // The indices of the primitive's first vertex, which select its slice and viewport.
        let mut first = Vec::with_capacity(selected.len());
        for &(input, output) in &selected {
            let at = ku(&mut fb, 0)?;
            let base = fb.module().global_value(input)?;
            let from = fb.access_chain(base, &[at])?;
            first.push((fb.load(from, MemAccess::default())?, output));
        }
        for k in 0..n {
            let at = ku(&mut fb, k)?;
            for &(input, output) in &carried {
                let base = fb.module().global_value(input)?;
                let from = fb.access_chain(base, &[at])?;
                let value = fb.load(from, MemAccess::default())?;
                let to = fb.module().global_value(output)?;
                fb.store(to, value, MemAccess::default())?;
            }
            if let Some((input, output)) = size {
                let value = match input {
                    Some(input) => {
                        let base = fb.module().global_value(input)?;
                        let from = fb.access_chain(base, &[at])?;
                        fb.load(from, MemAccess::default())?
                    }
                    None => kf(&mut fb, 1.0)?,
                };
                let to = fb.module().global_value(output)?;
                fb.store(to, value, MemAccess::default())?;
            }
            // `(1, 0)`, `(0, 1)`, `(0, 0)` at the first, second and third vertex.
            let x = kf(&mut fb, if k == 0 { 1.0 } else { 0.0 })?;
            let y = kf(&mut fb, if k == 1 { 1.0 } else { 0.0 })?;
            let value = fb.composite_construct(vec2, &[x, y])?;
            for &g in &weights {
                let to = fb.module().global_value(g)?;
                fb.store(to, value, MemAccess::default())?;
            }
            if let Some((input, output)) = primitive {
                let from = fb.module().global_value(input)?;
                let value = fb.load(from, MemAccess::default())?;
                let to = fb.module().global_value(output)?;
                fb.store(to, value, MemAccess::default())?;
            }
            for &(value, output) in &first {
                let to = fb.module().global_value(output)?;
                fb.store(to, value, MemAccess::default())?;
            }
            fb.emit_vertex()?;
        }
        fb.end_primitive()?;
        fb.ret(None)?;
        let f = fb.finish()?;
        let modes = vec![
            input,
            output,
            ExecMode::OutputVertices { n },
            ExecMode::Invocations { n: 1 },
        ];
        mb.entry(ExecModel::Geometry, f, modes)?;
        mb.finalize()
    }
}

/// An array of `n` elements of `elem`.
pub(super) fn array(mb: &mut ModuleBuilder, elem: TypeId, n: u32) -> BResult<TypeId> {
    let u32_ = scalar(mb, IoNumeric::Uint)?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(n)),
    })?;
    mb.ty(Type::Array {
        elem,
        len,
        stride: None,
    })
}

/// The `uint` constant `v`.
pub(super) fn ku(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// The `float` constant `v`.
pub(super) fn kf(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<Value> {
    let ty = fb.module().ty(Type::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}

/// The scalar type of `numeric`.
pub(super) fn scalar(mb: &mut ModuleBuilder, numeric: IoNumeric) -> BResult<TypeId> {
    match numeric {
        IoNumeric::Float => mb.ty(Type::Float { width: 32 }),
        IoNumeric::Sint => mb.ty(Type::Int {
            width: 32,
            signed: true,
        }),
        IoNumeric::Uint => mb.ty(Type::Int {
            width: 32,
            signed: false,
        }),
    }
}
