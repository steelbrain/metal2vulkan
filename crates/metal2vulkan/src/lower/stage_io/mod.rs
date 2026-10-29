//! `lower::stage_io`: vertex fetch, varyings, stage builtins, fragment inputs and outputs.
//!
//! A kernel has no stage interface beyond its dispatch builtins (`lower::dispatch`), so
//! [`plan_entry`] accepts it without outputs. Its `[[stage_in]]` attributes are fetched at entry
//! like pulled vertex attributes, under its compute pipeline's stage-input descriptor, and its
//! `[[grid_origin]]` and `[[grid_size]]` read the stage-in region (its `stage_in` module).
//!
//! For vertex and fragment functions this family plans one Vulkan interface object per stage
//! input or output ([`IoObject`], planned in its `plan` module) and links every used parameter and every output to
//! one ([`Link`]):
//!
//! - User interface variables (vertex attributes, varyings, color outputs) carry a relocatable
//!   `Location` keyed by a [`VaryingKey`] whose role the artifact states
//!   ([`crate::pipeline::stage_io::StageIo`]); link numbers them. Every one is 32-bit: integers are
//!   declared `uint` (held zero-extended), narrow floats `float` (held widened), but a `half` color
//!   output on a profile that enables `storageInputOutput16` and `shaderFloat16`, declared with
//!   16-bit `float` lanes (`StageIo::half_outputs`) and stored by the exact conversion of the
//!   `half` it holds, so the device receives the `half` the function returns, as Metal's attachment
//!   does, and blends and converts it as its own `half` output. Only that output needs
//!   `storageInputOutput16`. A 32-bit integer color output to a signed-integer attachment
//!   (`LoweringInputs::signed_color_attachments`) is declared `int` and stored by `OpBitcast`; a
//!   narrower one to such an attachment is `NotImplemented`.
//! - Builtins map one to one (`vertex_id` → `VertexIndex`, `[[position]]` → `Position` /
//!   `FragCoord`, ...), each checked against the profile feature its capability needs. A fragment
//!   `PrimitiveId`, `Layer`, or `ViewportIndex` input is declared through the first of its
//!   enabling capabilities the profile enables (`requirements::builtin_enablers`);
//!   `[[barycentric_coord]]` reads `BaryCoordKHR` or `BaryCoordNoPerspKHR`, a `float` or `float2`
//!   its leading lanes, a `float3` its first two lanes and `1 - (x + y)` of them as the third,
//!   Metal's third weight. On a device without `fragmentShaderBarycentric` it reads the first two
//!   lanes from a `float2` center varying of its perspective (`IoRole::Barycentric`) that the
//!   pipeline's synthesized geometry stage writes (`link::geometry`), read at the pixel center
//!   as a center varying is, the same way.
//! - On a device that enables neither of `PrimitiveId`'s capabilities (`geometryShader`,
//!   `tessellationShader`), `[[primitive_id]]` reads a flat `uint` varying (`IoRole::PrimitiveId`)
//!   that the pipeline's vertex function, compiled under `LoweringInputs::primitive_id_varying`,
//!   writes at each return as `(VertexIndex - first) / n`, from the draw's base vertex and
//!   `PushField::VerticesPerPrimitive`. Vulkan's flat inputs take the provoking vertex's
//!   value ("The output values assigned are those of the provoking vertex of the primitive"), and
//!   without a geometry, tessellation or mesh stage "the provoking vertex is determined by the
//!   primitive topology": "the provoking vertex for p_i is v_3i" in a triangle list, "v_2i" in a
//!   line list, and "v_i" in a strip and of points (the Vulkan specification's flat shading and
//!   primitive topologies, in a pipeline of the first provoking vertex, Vulkan's default; under
//!   `VK_EXT_provoking_vertex`'s last vertex a list's primitive `i` reads `(n i + n - 1) / n`, `i`
//!   too, and a strip's `i` plus its primitives' vertices less one). So each
//!   fragment reads `i`, its primitive's index among the draw's primitives of its instance, which
//!   is Metal's `[[primitive_id]]`: restarted in each instance, moved by no vertex start, base
//!   vertex or base instance, a flat varying taking the first vertex (Metal probes on the M3 Max,
//!   macOS 27.0.1: lists, strips, lines, points, indirect draws). The varying's `VertexIndex` is a
//!   draw's own only without indices: an indexed draw's primitives share vertices, so it runs in
//!   the de-indexed form (`LoweringInputs::primitive_id_indices`), `index count` vertices from 0
//!   that read their indices at the draw's index address ([`CapturedIndex`]), whose first vertex
//!   is 0. A vertex function that writes memory is
//!   refused there (it would run once per index), and so are the draws the binder cannot run
//!   so (`NotImplemented(PrimitiveId)`). A draw of one of Metal's private primitive types in the
//!   private list form (`LoweringInputs::private_list`) runs its list's positions from its first
//!   vertex without indices, each once, as Metal runs them: the same `(VertexIndex - first) / n`
//!   is its list primitive's index.
//!   A post-tessellation vertex function emulated on a device without `tessellationShader` writes
//!   the same varying under that field, from another source: its patch's position in the draw
//!   (`lower::tessellation::emulated`), which a fragment function compiled under the field reads
//!   on any device, since Vulkan's `PrimitiveId` would count the tessellator's triangles there.
//! - A `bool` varying (a vertex output or a fragment input) is a flat `uint` variable: written as
//!   1 or 0, read as `!= 0`.
//! - Vertex amplification: `[[amplification_id]]` is `ViewIndex` (the binder's view mask makes view
//!   `a` amplification `a`; without multiview both are 0), `[[amplification_count]]` the
//!   `SpecKey::AmplificationCount` constant (default 1), and a `[[shared]]` output (`air.shared`)
//!   is stored like any other: each view computes it again. A vertex function that writes
//!   `[[render_target_array_index]]` is never drawn amplified (the binder refuses a pipeline
//!   declaring `Layer` at a count above 1), so its `[[amplification_id]]` is 0, not `ViewIndex`:
//!   SPIRV-Cross gives a vertex function that reads `ViewIndex` a second
//!   `[[render_target_array_index]]` output, which Metal refuses.
//! - A vertex function's `[[render_target_array_index]]` selects a slice only in a render pass of
//!   at least two layers. The function depends on the pass's length
//!   (`RuntimeFact::RenderTargetArrayLength`): under `LoweringInputs::dropped_vertex_layer`, the
//!   variant a pass of at most one layer runs on, no `Layer` is declared (every index renders slice
//!   0); otherwise the function reads the length from `PushField::RenderTargetArrayLength` and
//!   stores `Layer` as `index < length ? index : 0`, as Metal renders the pass's first layer for an
//!   index at or past it (Metal probes on an M4), where Vulkan leaves the layer undefined. Metal's
//!   fragment function reads the index as written in both, so in both the vertex function writes it
//!   to a flat `uint` varying (`IoRole::RenderTargetArrayIndex`), which a fragment function
//!   compiled under `LoweringInputs::render_target_array_index_varying` reads as its
//!   `[[render_target_array_index]]`; any other fragment function reads `Layer` (0 when no stage
//!   writes it).
//! - A vertex function's `[[viewport_array_index]]` selects a viewport only below the draw's
//!   viewport count (`setViewports:count:`): Metal draws an index at or past it through viewport 0
//!   (Metal probes on the M3 Max and an M4), where Vulkan leaves the viewport undefined. The
//!   function depends on the count (`RuntimeFact::ViewportCount`), which it reads from
//!   `PushField::ViewportCount`, and stores `ViewportIndex` as `index < count ? index : 0`. Metal's
//!   fragment function reads the index as written, so the vertex function also writes it to a flat
//!   `uint` varying (`IoRole::ViewportIndex`), which a fragment function compiled under
//!   `LoweringInputs::viewport_index_varying` reads as its `[[viewport_array_index]]`; any other
//!   fragment function reads `ViewportIndex` (0 when no stage writes it).
//! - Where the device cannot store `Layer` or `ViewportIndex` from a vertex function
//!   (`shaderOutputLayer`, `shaderOutputViewportIndex`), the function stores the value it would
//!   store there to a flat `uint` varying instead (`IoRole::SelectedLayer`,
//!   `IoRole::SelectedViewport`), and the pipeline's synthesized geometry stages store the builtin
//!   from the primitive's first vertex (`link::geometry`), Metal's rule; that needs
//!   `geometryShader`, else the function is refused needing either feature. On a device without
//!   `multiViewport` a draw has one viewport, so every index selects viewport 0 and no
//!   `ViewportIndex` is stored at all: the raw varying alone. A fragment function that reads an
//!   index no stage can have written (no varying input names it, and the device declares neither
//!   builtin) reads 0, Metal's value with no vertex write (`StageIo::zero_indices`).
//!
//! Reading an input ([`param_value`]) loads the variable and brings it to the parameter's lowered
//! representation: a narrow integer is masked to its width (a signed-integer attribute format
//! sign-extends), a `half` is rounded to the format (an interpolated or fetched `float` is not a
//! `half` value). Writing outputs ([`emit_return`]) takes the returned value (the output itself,
//! or a struct with one member per output) apart and stores each part. A vertex function without
//! a point-size output writes `PointSize = 1.0`; one with it stores a size past Metal's largest
//! point, 511 pixels, as 511 ([`metal_point_size`]), where a Vulkan device clamps only to its own
//! `pointSizeRange`. Fragment functions declare `OriginUpperLeft` (Vulkan's only origin; the binder
//! flips y with a negative viewport) and their depth and stencil modes.
//!
//! A vertex function's position is decorated `Invariant`, `[[invariant]]` in Metal or not: every
//! pipeline that draws a vertex function computes its position the same way, so a draw run
//! through more than one variant of its pipeline (each color group's) rasterizes the same
//! positions in each.
//!
//! A fragment function compiled for a render pipeline that rasterizes more than one sample
//! (`LoweringInputs::multisampled`) reads what Metal reads at the pixel center in every invocation
//! there ([`Source::PixelCenter`]): `[[position]]`'s x and y as `floor(FragCoord.xy) + 0.5`, and,
//! where the profile has interpolation functions (`sampleRateShading`, and on a portability-subset
//! device `shaderSampleRateInterpolationFunctions`) whose offset reads from the pixel center
//! (`VkProperty::InterpolateAtOffsetCentered`), each center-sampled float varying through
//! `InterpolateAtOffset` at offset 0; where the device reads such a varying at the pixel center
//! itself, as Metal does (`VkProperty::MetalCenterVaryings`), as it is, and the module requires
//! that property. Its interface records both, and whether it may read `[[position]]`'s z or w,
//! which have no such form (`StageIo::{multisampled, varyings_at_pixel, position_depth}`): those
//! read where Vulkan evaluates them, a sample's under sample shading, and the compile layer notes
//! `PrecisionNote::MultisamplePosition`; link refuses a center varying a sample-rate invocation
//! would read elsewhere.
//!
//! A pulled vertex attribute (`VertexFetch::Pulled`) has no interface variable: its value is
//! fetched from its source buffer once, in the function's first block before any instruction of the
//! body ([`emit_entry`]), through the slot the access family chose for that buffer
//! (`access::pulled`), and every use reads that value. Metal's attribute is an input fetched before
//! the body runs, so no write of the body, through any pointer to the source buffer's memory,
//! changes it.
//!
//! A color output in a raster order group is stored like any other: the fragment family runs the
//! whole body inside the interlock.
//!
//! Refused as `NotImplemented`: array varyings, Metal's interpolation functions (`interpolant`),
//! and a fragment position or barycentric input sampled other than at the pixel center.
//! A 64-bit interface value is invalid input. A post-tessellation vertex function's inputs are
//! `lower::tessellation`'s; its outputs are planned here.
//!
//! Its hooks into `select` and `lower::emit`: the entry ([`plan_entry`], [`finish`]), stage-IO
//! parameters ([`select_param`], [`param_value`]), the work before the body ([`emit_entry`]), the
//! entry's returned value ([`select_return`], [`emit_return`]; `fragment::emit_return` then ends
//! the function), and the interface variables and execution modes ([`declare`]).

pub(crate) mod color_write;
pub(crate) mod extended_color;
pub mod fetch;
mod plan;
mod stage_in;
pub(crate) mod word_color;

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{Builtin, ParamBinding, ParamRole};
use crate::air::ir::{ArgIdx, InstId, Value, ValueId};
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::{narrow, ops, FloatFmt, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::{Plan, SelectCx};
use crate::pipeline::stage_io::{IoNumeric, StageIo};
use crate::pipeline::sym::{PushField, SpecKey, Stage, VaryingKey};
use crate::spirv::builder::constructs::KillForm;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, GlslOp, IoDecor, IoDir, MemAccess, StorageClass,
    Type as SType, UnOp,
};

pub(crate) use plan::{
    amplification_count_form, finish_merged, note_device_point_size, output_types,
    view_index_enabled,
};
pub use plan::{DrawBase, Link, Place, Shape};

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::StageIo)).into()
}

/// A Vulkan interface object of the stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IoObject {
    /// A user variable: `Location` relocated by `key`.
    User {
        key: VaryingKey,
        dir: IoDir,
        /// `components` lanes of `float`, `uint`, or (a color output to a signed-integer
        /// attachment) `int`.
        numeric: IoNumeric,
        components: u8,
        deco: IoDecor,
        /// Declared with 16-bit `float` lanes: a `half` color output on a device that enables
        /// `storageInputOutput16` (`plan::half_variable`), stored by the conversion of the `half` it
        /// holds widened, exact. Every other user variable is 32-bit.
        half: bool,
    },
    /// A builtin variable.
    Builtin {
        builtin: crate::spv::BuiltIn,
        dir: IoDir,
        shape: Shape,
        /// The enabling capability chosen among the builtin's alternatives
        /// (`requirements::builtin_enablers`); `None` for a builtin without any.
        enabler: Option<crate::spv::Capability>,
    },
}

/// Where a used stage-IO parameter's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// An interface variable (a builtin, a fixed-function attribute, a varying).
    Io(Link),
    /// An interface variable read at the pixel center whatever samples the invocation shades, as
    /// Metal reads it in every invocation of a multisampled pipeline
    /// (`LoweringInputs::multisampled`): `FragCoord` with x and y `floor(..) + 0.5`, or a
    /// center-sampled float varying through `InterpolateAtOffset` at offset 0, which
    /// `GLSL.std.450` defines as the center of the pixel.
    PixelCenter(Link),
    /// A pulled vertex attribute, fetched from its source buffer once at entry.
    Pulled(fetch::PulledRead),
    /// `[[sample_id]]` of a function looping over its covered samples, into an integer of this
    /// form: the sample of the loop's pass, the fragment family's `loop_sample`.
    LoopSample(Form),
    /// A sample-qualified float varying of a function looping over its covered samples: the
    /// variable, declared without `Sample`, interpolated at the loop's pass's sample through
    /// `InterpolateAtSample`.
    AtLoopSample(Link),
    /// `[[amplification_count]]` into an integer of this form: the `SpecKey::AmplificationCount`
    /// specialization constant.
    AmplificationCount(Form),
    /// `[[amplification_id]]` of a vertex function that writes its render target slice, which is
    /// never drawn amplified: 0, an integer of this form.
    Unamplified(Form),
    /// A fragment function's stage input its pipeline's linking drops and no vertex output writes
    /// (`LoweringInputs::absent_varyings`): zero of this lowered form, no interface variable.
    Absent(Form),
    /// A `u32` push field read into an integer of this form: under
    /// `LoweringInputs::amplification_draws`, `[[amplification_id]]` from
    /// `PushField::AmplificationId`, and a fragment function's `[[render_target_array_index]]`
    /// or `[[viewport_array_index]]` whose vertex function writes none from the amplification's
    /// offset.
    Pushed { field: PushField, form: Form },
    /// A kernel's `[[grid_origin]]` (`size` false) or `[[grid_size]]` (`size` true): the
    /// stage-in region's, from `PushField::StageInRegion`, projected like a dispatch builtin.
    StageInRegion {
        size: bool,
        projection: crate::lower::dispatch::Projection,
    },
    /// A vertex function's `[[base_vertex]]` or `[[base_instance]]` on a profile without
    /// `shaderDrawParameters`: the lane of `PushField::BaseVertexInstance` that holds the draw's
    /// `base`, into an integer of this form.
    DrawBase { base: DrawBase, form: Form },
    /// A vertex function's `[[vertex_id]]` in the de-indexed or private list form: the vertex its
    /// index names ([`CapturedIndex`]), into an integer of this form.
    CapturedVertex(Form),
    /// A vertex function's `[[base_vertex]]` in the de-indexed or private list form: the draw's
    /// base vertex, lane 3 of `PushField::CaptureLayout` (the word before the private list),
    /// into an integer of this form.
    CapturedBaseVertex(Form),
    /// A merged draw's vertex function's `[[vertex_id]]`, `[[instance_id]]`, `[[base_vertex]]` or
    /// `[[base_instance]]` (`LoweringInputs::merged_draws`): its corner's command's, which the
    /// entry finds ([`MergedRead`]), into an integer of this form.
    Merged { value: MergedValue, form: Form },
    /// A `float` or `half` varying the fragment function reads flat because its vertex function
    /// writes it the same at every vertex of a primitive (`LoweringInputs::uniform_varyings`):
    /// the variable, declared `Flat`, read as Metal's interpolation returns such a value
    /// ([`metal_interpolated_constant`]).
    Uniform(Link),
    /// A fragment function's `[[primitive_id]]` under `LoweringInputs::quad_primitive_ids`:
    /// Vulkan's `PrimitiveId` (the variable) shifted right by one, the quad of the list's
    /// triangle, as Metal counts one primitive a quad of its private quads and quad strip.
    QuadPrimitive(Link),
}

/// One used stage-IO parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParam {
    /// The LLVM parameter index.
    pub param: u32,
    pub source: Source,
}

/// The shape of an entry's returned value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Returned {
    /// `void`: the entry has no outputs.
    #[default]
    Void,
    /// The entry returns its single output itself.
    Value,
    /// A struct with one field per decoded output (a single output may also come as a
    /// one-field struct).
    Struct,
}

/// One output the entry stores: the returned value's field that holds it and its interface
/// object. A gated-off output's field has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlannedOutput {
    /// The field of a returned struct ([`Returned::Struct`]); 0 for a returned value itself.
    pub field: u32,
    pub link: Link,
    /// The count push field the part is stored below, and 0 where it is at or past it: a vertex
    /// function's `ViewportIndex` below `PushField::ViewportCount`, its `Layer` below
    /// `PushField::RenderTargetArrayLength`.
    pub below: Option<PushField>,
    /// The color attachment a fragment function's output of dual-source index 0 writes, whose
    /// lanes [`StageIoPlan::rounded`] may round.
    pub color: Option<u32>,
    /// A fragment function's color output, of either dual-source index, to an attachment held
    /// with its channels rotated (`LoweringInputs::rotated_colors`): its four lanes are stored
    /// rotated (`metal::format::to_rotated_image`), after their conversion.
    pub rotated: bool,
    /// A [`rotated`](Self::rotated) output to an attachment whose image holds Metal's alpha in
    /// red (`LoweringInputs::alpha_in_red_colors`): the image's alpha lane holds Metal's alpha
    /// too, not Metal's blue, so alpha to coverage reads Metal's alpha (the image keeps only its
    /// red).
    pub alpha_red: bool,
    /// A fragment function's first output to a rotated attachment whose blend takes
    /// `RotatedBlendForm::Premultiplied` (`LoweringInputs::rotated_blends`): after the rotation,
    /// the image's alpha lane (Metal's blue) holds Metal's blue times Metal's alpha, each clamped
    /// to `[0, 1]` with a NaN taken as 0, multiplied without contraction.
    pub premultiplied: bool,
    /// The `u32` push field added to the part before `below` holds it: under
    /// `LoweringInputs::amplification_draws`, the amplification's offset added to a vertex
    /// function's viewport and render target slice indices, in their builtins and raw varyings
    /// alike.
    pub offset: Option<PushField>,
}

/// An output a vertex function compiled under `LoweringInputs::amplification_draws` stores
/// though it writes no such output of its own: `Layer` or `ViewportIndex` as the amplification's
/// offset (`offset`), 0 where that is at or past the count push field `below`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AmplifiedOutput {
    pub link: Link,
    pub offset: PushField,
    pub below: PushField,
}

/// The private alpha test a fragment function compiled under `LoweringInputs::alpha_test` runs
/// at each return ([`alpha_test`]): the returned field of its color 0, four `float` lanes (a
/// `half` output held widened), and the comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AlphaTest {
    /// The field of a returned struct ([`Returned::Struct`]); 0 for a returned value itself.
    pub field: u32,
    pub function: crate::metal::CompareFunction,
}

/// The stage-IO part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageIoPlan {
    /// Every interface object, in declaration order.
    pub objects: Vec<IoObject>,
    /// Every used stage-IO parameter.
    pub params: Vec<PlannedParam>,
    /// One per output of the specialized entry, in output order (a gated-off output is not
    /// one); a vertex function's `[[viewport_array_index]]` is two, its `ViewportIndex` and then
    /// the raw varying, and its `[[render_target_array_index]]` two, its `Layer` and then the raw
    /// varying, or under `dropped_vertex_layer` the raw varying alone.
    pub outputs: Vec<PlannedOutput>,
    /// The outputs a vertex function compiled under `LoweringInputs::amplification_draws` stores
    /// beyond its own: `Layer` and `ViewportIndex` from the amplification's offsets where it
    /// writes no such index.
    pub amplified: Vec<AmplifiedOutput>,
    /// The color outputs a fragment function compiled under `LoweringInputs::zeroed_colors`
    /// stores zero to at every return, one per such color.
    pub zeroed: Vec<Link>,
    /// The shape of the returned value.
    pub returned: Returned,
    /// A vertex function stores every slice and viewport index it stores to its selected index
    /// (`IoRole::SelectedLayer`, `IoRole::SelectedViewport`), as it stores one of them there.
    pub indices_selected: bool,
    /// The `PointSize` object a vertex function without a point-size output writes 1.0 to.
    pub point_size_default: Option<usize>,
    /// The `Position` object of a vertex function, decorated `Invariant`.
    pub invariant: Option<usize>,
    /// A fragment function of one of its pipeline's color groups, where the pipeline takes more
    /// than one (`LoweringInputs::color_group`): its `FragDepth`, `SampleMask` and
    /// `FragStencilRefEXT` outputs are decorated `Invariant`, so every group's variant computes
    /// them alike (the module docs of `pipeline::compile::color_group`).
    pub invariant_fragment_outputs: bool,
    /// The entry's execution modes (fragment origin, early tests, depth and stencil modes).
    pub modes: Vec<ExecMode>,
    /// The artifact's stage-IO interface.
    pub interface: StageIo,
    /// The Metal buffer indices pulled attributes read, a kernel's stage-in attributes and index
    /// buffer included. Their slots are the access family's (`access::pulled`).
    pub buffers: BTreeSet<u32>,
    /// Whether a kernel's stage-in attribute steps by thread position in the grid, which the
    /// dispatch family then derives (`dispatch::plan_kernel`).
    pub reads_grid_position: bool,
    /// Whether a kernel reads the stage-in region (`RuntimeFact::StageInRegion`).
    pub region: bool,
    /// An upper estimate of the words the interface adds: declarations, one load per use of an
    /// interface parameter, one fetch per pulled attribute, the output stores, and the roundings of
    /// color outputs.
    pub words: u64,
    /// The lanes of each color attachment's output a fragment function rounds itself
    /// (`color_write`), through an output variable or in the storage form: each by its rule, a
    /// half-float channel's under `TextureWriteRoundingMode::NearestEven`.
    pub rounded: BTreeMap<u32, [Option<crate::pipeline::lower_inputs::ChannelWrite>; 4]>,
    /// The measured driver deviations, as their properties, a fragment function's color writes
    /// rely on: channels that record no device conversion because the device converts them as
    /// Metal does (`color_write`). The module requires them.
    pub color_properties: BTreeSet<crate::device::requirements::VkProperty>,
    /// The extended-range color attachments whose `float` or `half` output each return stores
    /// as the raw texel Metal stores (`extended_color::encode`), with the form each is held in
    /// (and, a bfloat one's rounding).
    pub extended: BTreeMap<u32, crate::pipeline::lower_inputs::ExtendedRangeColor>,
    /// The word attachments whose `float` or `half` output each return stores as the word Metal
    /// stores (`word_color::encode`), with each word's layout.
    pub words_written: BTreeMap<u32, crate::metal::format::WordAttachment>,
    /// A vertex function's viewport compensation (`LoweringInputs::viewport_compensation`).
    pub compensation: Option<Compensation>,
    /// A fragment function's depth output, which each return gives Metal's answer
    /// (`clamp_depth`).
    pub depth_clamp: Option<DepthClamp>,
    /// The render pipeline's private masks a fragment function applies itself
    /// (`LoweringInputs::sample_mask`), which each return ANDs into the function's own
    /// `[[sample_mask]]` output or stores as it.
    pub private_mask: Option<PrivateMask>,
    /// The render pipeline's private alpha test the fragment function runs at each return
    /// (`LoweringInputs::alpha_test`).
    pub alpha_test: Option<AlphaTest>,
    /// A fragment function's per-primitive inputs (`StageIo::primitive_inputs`) are flat alone,
    /// not decorated `PerPrimitiveEXT`: its device does not enable `VK_EXT_mesh_shader`, so the
    /// pipeline's mesh function is emulated, and its synthesized vertex stage writes a
    /// primitive's values at every corner of it.
    pub flat_primitive_inputs: bool,
    /// The second outputs a fragment function writes for attachments held with their channels
    /// rotated (`LoweringInputs::rotated_factors`), each beside its first output there.
    pub rotated_seconds: Vec<RotatedSecond>,
    /// A fragment function compiled under `LoweringInputs::multisampled` reads its center
    /// varyings as they are, as their pixel form, because its device reads them at the pixel
    /// center as Metal does (`VkProperty::MetalCenterVaryings`): the module requires that
    /// property.
    pub center_varyings_by_device: bool,
    /// A fragment function reads a varying the device interpolates and records no
    /// `PrecisionNote::DeviceInterpolation`, because its device interpolates as Metal does
    /// (`VkProperty::MetalInterpolation`): the module requires that property.
    pub interpolation_by_metal: bool,
    /// A vertex function stores an index that differs among a primitive's vertices to its
    /// builtin, because its device takes the first vertex's as Metal does
    /// (`VkProperty::MetalFirstVertexIndices`): the module requires that property.
    pub first_vertex_indices_by_device: bool,
    /// A vertex function compiled under `LoweringInputs::window_depth_varying` writes its
    /// position's window depth to a varying ([`WindowDepth`]).
    pub window_depth: Option<WindowDepth>,
    /// A vertex function compiled in the de-indexed or private list form reads its vertex from
    /// the draw's indices ([`CapturedIndex`]): planned where the function reads it
    /// (`[[vertex_id]]`, a per-vertex attribute, `[[base_vertex]]` or the primitive ID).
    pub captured_index: Option<CapturedIndex>,
    /// A vertex function compiled under `LoweringInputs::merged_draws` finds its corner's command
    /// at entry ([`MergedRead`]).
    pub merged: Option<MergedRead>,
    /// A vertex function compiled under `LoweringInputs::primitive_id_varying` writes the
    /// primitive ID of each vertex's primitive to its varying ([`PrimitiveIdStore`]).
    pub primitive_id: Option<PrimitiveIdStore>,
}

/// How a vertex function compiled under `LoweringInputs::merged_draws` finds the command its
/// corner falls in: at entry, before the body, from its `VertexIndex` and `InstanceIndex` (the
/// input builtins' objects `vertex` and `instance`) and the header at
/// `PushField::MergedDrawAddress` (`bind::replay`'s `merged.rs`, `bind::merged_vertex`). Its
/// `[[vertex_id]]`, `[[instance_id]]`, `[[base_vertex]]` and `[[base_instance]]` read the
/// command's ([`Source::Merged`]), and its `RootTableAddress` is the command's vertex slot table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MergedRead {
    pub vertex: usize,
    pub instance: usize,
}

/// Which of a merged draw's command's values a vertex function reads ([`Source::Merged`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergedValue {
    Vertex,
    Instance,
    BaseVertex,
    BaseInstance,
}

/// The words a merged draw's vertex function adds at entry: the two input builtins' variables,
/// decorations and interface entries and their loads, the header's loads, the corner's 64-bit
/// arithmetic, the search's loop (its variables, test, halving, load and selection), the
/// command's draw's loads, the division and remainder, and the slot table's address.
pub(super) const MERGED_WORDS: u64 = 600;

/// How a vertex function compiled under `LoweringInputs::primitive_id_varying` writes the
/// primitive ID a fragment function reads from its varying on a device without a fragment
/// `PrimitiveId`: at each return, `(VertexIndex - first) / n` (`vertex` is the input builtin's
/// object), in wrapping `u32` arithmetic, `first` the draw's first vertex ([`FirstVertex`]) and
/// `n` `PushField::VerticesPerPrimitive`, to its `IoRole::PrimitiveId` varying (`object`). A
/// primitive's flat inputs read its provoking vertex, its first in a pipeline of Vulkan's default
/// provoking vertex, so each of its fragments reads the index of the primitive in its instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrimitiveIdStore {
    pub vertex: usize,
    pub first: FirstVertex,
    pub object: usize,
}

/// Where a vertex function's primitive ID reads its draw's first vertex, Vulkan's base vertex of
/// a draw without indices (its `firstVertex`, in an indirect draw's arguments too).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FirstVertex {
    /// The `BaseVertex` builtin, stage-IO object `index` (`shaderDrawParameters`).
    Builtin { index: usize },
    /// Lane 0 of `PushField::BaseVertexInstance`, which the binder writes for each draw (an
    /// indirect one's from its arguments read on the CPU).
    Push,
    /// Vertex 0: the de-indexed form draws an indexed draw's indices as `index count` vertices
    /// from 0 (`LoweringInputs::primitive_id_indices`).
    Zero,
}

/// The words a vertex function's primitive ID adds: its output variable, decorations and
/// interface entry, the vertex index's and base vertex's inputs, their loads, the push loads and
/// an extract, the subtraction, the division and the store.
const PRIMITIVE_ID_WORDS: u64 = 56;

/// How a vertex function compiled under `LoweringInputs::window_depth_varying` writes its
/// window depth: at each return, the `z / w` of the value it stores to its `Position` object
/// (`position`, after any compensation or clamp) to its `IoRole::WindowDepth` varying
/// (`object`), for a fragment function that computes each sample's depth from it
/// (`lower::fragment`'s `depth_test`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowDepth {
    pub position: usize,
    pub object: usize,
}

/// The words a vertex function's window depth adds: its variable, decorations and interface
/// entry, two extracts, the division, and the store.
const WINDOW_DEPTH_WORDS: u64 = 40;

/// How a vertex function compiled in the primitive ID's de-indexed form
/// (`LoweringInputs::primitive_id_indices`) reads its vertex: its stream position
/// `p = VertexIndex` (`vertex` is the input builtin's object; the draw starts at vertex 0) names
/// the raw index of type `ty` at `PushField::CaptureIndexAddress + p · size`, read once at entry
/// (`u16`: from the aligned word that holds it, shifted by its address's bit 1), and the vertex is
/// the raw index plus the layout's base vertex (lane 3 of `PushField::CaptureLayout`), wrapping
/// as `u32`s. The draw is a list or points: a strip, which Metal restarts at the all-ones index,
/// does not run so (`bind::graphics`).
///
/// In the private list form (`LoweringInputs::private_list`, `first` is the draw's first vertex)
/// the draw runs its list from `first` without indices: the raw index, a `u32`, lies at
/// `PushField::CaptureIndexAddress + 4 · VertexIndex` all the same, Metal's base vertex at
/// `PushField::CaptureIndexAddress + 4 · (first - 1)`, the word before the list (read once at
/// entry), which the vertex adds in place of the layout's; nothing restarts, and no layout is
/// read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapturedIndex {
    pub vertex: usize,
    pub ty: crate::metal::IndexType,
    pub first: Option<FirstVertex>,
}

/// The words a vertex function's index read adds: its input builtin's variable, decorations and
/// interface entry, its load, the layout's extract, the 64-bit address arithmetic, the device
/// place (its block and pointer types, conversion and access chain), the load, a `u16`'s
/// alignment and shift, and the base vertex's extract and add (at most 224, a `u16` read's).
pub(super) const CAPTURED_INDEX_WORDS: u64 = 400;

/// Plan an emulated post-tessellation vertex function's primitive ID output under
/// `LoweringInputs::primitive_id_varying`: the index of its object, the flat `uint` varying of
/// `IoRole::PrimitiveId` to which its entry wrapper writes its patch's position in the draw
/// (`lower::tessellation`).
pub fn primitive_id_varying(sp: &mut StageIoPlan) -> Result<usize, Error> {
    plan::primitive_id(sp, IoDir::Output)
}

/// Plan the fragment's window-depth input for a fragment function that computes each sample's
/// depth for its in-shader depth test: the index of its object, a no-perspective `float` varying of
/// `IoRole::WindowDepth`.
pub fn window_depth_input(cx: &mut SelectCx<'_>) -> Result<usize, Error> {
    let sp = &mut cx.families.stage_io;
    sp.words = sp.words.saturating_add(WINDOW_DEPTH_WORDS);
    plan::window_depth(sp, IoDir::Input)
}

/// The second output (`[[color(n), index(1)]]`) a fragment function writes for an attachment held
/// with its channels rotated (`LoweringInputs::rotated_factors`): beside its first output there,
/// the returned value's field `field` (0 for a returned value itself), whose channels or
/// constants `lanes` copies into the image's channels, in the object of `link` and its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RotatedSecond {
    pub field: u32,
    pub link: Link,
    pub lanes: [crate::pipeline::lower_inputs::FactorLane; 4],
}

/// How a fragment function's returns store its depth output as Metal does (`clamp_depth`): its
/// `FragDepth` object, and whether they clamp it to `[0, 1]`, where Vulkan no longer does
/// (`VK_EXT_depth_range_unrestricted`). Every return kills a positive NaN where it can. Under
/// `range` (`LoweringInputs::depth_clamp_range`) they clamp it to `[0, 1]` on every profile and
/// then to the draw's range, a negative NaN storing 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthClamp {
    pub object: usize,
    pub clamps: bool,
    pub range: bool,
}

/// How a vertex function compiled under `LoweringInputs::viewport_compensation` stores its
/// position: its `Position` object, whose value it compensates by
/// `PushField::ViewportCompensation` before storing it, and, where it compensates the depth, its
/// `ClipDistance` object and the first of the two elements it writes there after its own. Under
/// Metal's guard band for points, `guard` names the object and the first of the four elements it
/// writes after those. Under `mirrors` it negates `x` where the viewport it selects is mirrored
/// (`PushField::ViewportMirrors`). Under `vertex_clamp`, which only a compensated depth takes, it
/// clamps that depth by `PushField::VertexDepthClamp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Compensation {
    pub position: usize,
    pub clip: Option<(usize, u32)>,
    pub guard: Option<(usize, u32)>,
    pub mirrors: bool,
    pub vertex_clamp: bool,
}

impl StageIoPlan {
    /// The index of an object equal to `o`, adding it when there is none.
    fn object(&mut self, o: IoObject) -> usize {
        match self.objects.iter().position(|x| *x == o) {
            Some(i) => i,
            None => {
                self.objects.push(o);
                self.objects.len() - 1
            }
        }
    }
}

/// The stage interface's module-scope objects: one global per planned object.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    pub globals: Vec<GlobalId>,
    /// By object, the `Private` array a sample-qualified varying of a function looping over its
    /// covered samples is read into at every sample before the loop, where the driver's
    /// interpolation needs the whole quad ([`read_at_every_sample`]).
    pub at_samples: BTreeMap<usize, GlobalId>,
}

/// Plan the entry's stage interface: its outputs and execution modes. A kernel has none (and a
/// kernel with outputs is refused); a post-tessellation vertex function's outputs are a vertex
/// function's (its inputs are `lower::tessellation`'s).
pub fn plan_entry(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    match cx.stage {
        // An object or mesh function returns nothing; a mesh function's outputs are its mesh
        // handle's.
        Stage::Kernel | Stage::Object | Stage::Mesh if cx.entry.ep.outputs.is_empty() => Ok(()),
        Stage::Kernel | Stage::Object | Stage::Mesh => Err(not_lowered()),
        Stage::Vertex | Stage::Fragment => plan::entry(cx),
    }
}

/// Plan a used stage-IO parameter (`stage_in`, vertex and fragment inputs, patch inputs) or a
/// builtin of a vertex or fragment entry.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    match (&b.role, cx.stage) {
        (ParamRole::StageIn { location }, Stage::Kernel) => stage_in::param(cx, b, location),
        (ParamRole::Builtin(Builtin::StageInGridOrigin), Stage::Kernel) => {
            stage_in::region_param(cx, b, false)
        }
        (ParamRole::Builtin(Builtin::StageInGridSize), Stage::Kernel) => {
            stage_in::region_param(cx, b, true)
        }
        (ParamRole::StageIn { .. }, _) => {
            Err(cx.invariant("a stage-in parameter outside a kernel"))
        }
        // A mesh function's amplification is a render stage's.
        (
            ParamRole::Builtin(Builtin::AmplificationId | Builtin::AmplificationCount),
            Stage::Mesh,
        ) => plan::param(cx, b),
        (_, Stage::Kernel | Stage::Object | Stage::Mesh) => Err(not_lowered()),
        _ => plan::param(cx, b),
    }
}

/// Check the entry's return at instruction `at`, of value `v` when the entry returns one (its
/// stage outputs): its type is the one [`plan_entry`] planned the outputs from. A value whose
/// every output is gated off is stored nowhere.
pub fn select_return(cx: &mut SelectCx<'_>, _at: InstId, v: Option<Value>) -> Result<(), Error> {
    let void = cx.families.stage_io.returned == Returned::Void;
    match v {
        None if void => Ok(()),
        None => Err(cx.invariant("an entry with outputs returns no value")),
        Some(_) if void => Err(cx.invariant("an entry without outputs returns a value")),
        Some(_)
            if cx.families.stage_io.outputs.is_empty()
                && !crate::lower::fragment::writes_outside_interface(cx) =>
        {
            Ok(())
        }
        Some(v) => {
            let ty = cx
                .value_ty(v)
                .ok_or_else(|| cx.invariant("a returned value without a type"))?;
            // The representation the planned outputs take apart must be the value's.
            cx.lowered_repr(v, ty).map(|_| ())
        }
    }
}

/// The varyings vertex function `e`, compiled under `inputs`, writes the same value at every
/// vertex of a primitive (`StageIo::uniform_outputs`), sorted: its user varyings, and the raw
/// varyings of the slice and viewport indices it writes (`VaryingKey::render_target_array_index`,
/// `VaryingKey::viewport_index`), which decide where it stores those indices. At each return, in
/// a block every vertex of a primitive reaches alike, the varying's field of the returned value is
/// uniform
/// among those vertices (`simd::uniform::Among::Primitive`), computed only from constants,
/// `constant` memory read through uniform pointers, the draw's instance, base vertex, base
/// instance, amplification and patch, and the vertex attributes `inputs` fetches per instance or
/// at a constant step. Decided from the structure of the specialized entry alone, never from a
/// name. Empty for every other stage.
pub(crate) fn uniform_varyings(
    e: &crate::air::Entry,
    inputs: &crate::lower::select::LowerInputs,
    m: &mut crate::budget::Meter,
) -> Result<BTreeSet<VaryingKey>, Error> {
    use crate::air::abi::{AbiInt, OutputRole, Stage as AirStage};
    use crate::air::ir::{InstKind, Type};
    use crate::lower::simd::uniform::Uniformity;
    use crate::metal::VertexStepFunction as Step;
    use crate::pipeline::lower_inputs::VertexFetch;

    let mut out = BTreeSet::new();
    if !matches!(e.ep.stage, AirStage::Vertex { .. }) {
        return Ok(out);
    }
    let fields: Vec<(u32, VaryingKey)> =
        e.ep.outputs
            .iter()
            .filter_map(|o| match &o.role {
                OutputRole::VertexOutput(key) => Some((o.field, VaryingKey::user(&key.0))),
                OutputRole::RenderTargetArrayIndex => {
                    Some((o.field, VaryingKey::render_target_array_index()))
                }
                OutputRole::ViewportArrayIndex => Some((o.field, VaryingKey::viewport_index())),
                _ => None,
            })
            .collect();
    let func = e.module.func(e.func);
    let Some(body) = func.and_then(|f| f.body.as_ref()) else {
        return Ok(out);
    };
    if fields.is_empty() {
        return Ok(out);
    }
    // The attributes the vertices of one instance read alike: pulled at a per-instance or
    // constant step, or fetched in fixed function from a layout stepped per instance.
    let stepped: Vec<u32> =
        e.ep.params
            .iter()
            .filter(|b| match &b.role {
                ParamRole::VertexInput { location } => match location.index {
                    AbiInt::Const(index) => match inputs
                        .vertex_fetch
                        .as_ref()
                        .and_then(|p| p.attributes.get(&index))
                    {
                        Some(VertexFetch::Pulled { step_function, .. }) => {
                            matches!(step_function, Step::PerInstance | Step::Constant)
                        }
                        _ => inputs.instanced_attributes.contains(&index),
                    },
                    AbiInt::Global(_) => false,
                },
                _ => false,
            })
            .map(|b| b.param)
            .collect();
    let returns_struct = func
        .and_then(|f| match e.module.types().get(f.ty) {
            Some(Type::Function { ret, .. }) => e.module.types().get(*ret),
            _ => None,
        })
        .is_some_and(|t| matches!(t, Type::Struct { .. }));
    let mut returns = Vec::new();
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            if let Some(InstKind::Ret(Some(v))) = body.insts.get(id).map(|i| &i.kind) {
                returns.push((b, *v));
            }
        }
    }
    if returns.is_empty() {
        return Ok(out);
    }
    let u = Uniformity::among_primitive(e, body, &stepped, m)?;
    let consts = e.module.consts();
    // Whether member `field` of the value `v` returns is uniform: the insertions that build it,
    // or the whole value.
    let member = |mut v: crate::air::ir::Value, field: u32| -> bool {
        use crate::air::ir::{Const, Value};
        if !returns_struct {
            return u.is_uniform_value(e, v);
        }
        for _ in 0..=body.insts.len() {
            match v {
                Value::Inst(i) => match body.insts.get(i).map(|x| &x.kind) {
                    Some(InstKind::InsertValue { agg, val, indices }) => {
                        match indices.split_first() {
                            Some((&f, rest)) if f == field => {
                                if !u.is_uniform_value(e, *val) {
                                    return false;
                                }
                                if rest.is_empty() {
                                    return true;
                                }
                                v = *agg;
                            }
                            Some(_) => v = *agg,
                            None => return false,
                        }
                    }
                    _ => return u.is_uniform_value(e, v),
                },
                Value::Const(c) => {
                    return match consts.get(c).map(|k| &k.value) {
                        Some(Const::Aggregate(elems)) => elems
                            .get(field as usize)
                            .is_some_and(|&k| u.is_uniform_value(e, Value::Const(k))),
                        Some(Const::Null) => true,
                        _ => false,
                    };
                }
                Value::Arg(_) => return u.is_uniform_value(e, v),
            }
        }
        false
    };
    for (field, key) in fields {
        if returns
            .iter()
            .all(|&(b, v)| u.is_uniform(b) && member(v, field))
        {
            out.insert(key);
        }
    }
    m.release(u.ledger_bytes());
    Ok(out)
}

/// Complete the stage-IO plan once every instruction and parameter is selected: the per-stage
/// interface limits and the size estimate.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    plan::finish(cx)
}

/// The words the stage interface adds to the size estimate.
pub fn estimate_words(plan: &StageIoPlan) -> u64 {
    plan.words
}

/// Count `words` of the fragment family's part of the stage interface (input attachments and
/// the discard predicate's kill at each return) in the interface's size estimate. Called from
/// `fragment::finish`, after [`finish`] computed this family's own words.
pub fn add_interface_words(cx: &mut SelectCx<'_>, words: u64) {
    let sp = &mut cx.families.stage_io;
    sp.words = sp.words.saturating_add(words);
}

/// Plan the fragment's `FragCoord` input for a family that reads the invocation's pixel (the
/// fragment imageblock): the index of its object, shared with a `[[position]]` input.
pub fn frag_coord(cx: &mut SelectCx<'_>) -> usize {
    plan::builtin(
        &mut cx.families.stage_io,
        crate::spv::BuiltIn::FragCoord,
        IoDir::Input,
        Shape::FVec(4),
        crate::pipeline::stage_io::StageBuiltin::FragCoord,
    )
}

/// Plan the fragment's `SampleId` input for a family that reads the invocation's sample
/// (framebuffer fetch in a multisampled pass): the index of its object, shared with a
/// `[[sample_id]]` input. Its stage interface states `SampleId`, so the function runs once per
/// covered sample (link's sample invocations); the caller checked `sampleRateShading`.
pub fn sample_id(cx: &mut SelectCx<'_>) -> usize {
    plan::builtin(
        &mut cx.families.stage_io,
        crate::spv::BuiltIn::SampleId,
        IoDir::Input,
        Shape::U32,
        crate::pipeline::stage_io::StageBuiltin::SampleId,
    )
}

/// Plan the fragment's `FrontFacing` input for a family that reads the fragment's face (the
/// shader-side stencil): the index of its object, shared with a `[[front_facing]]` input.
pub fn front_facing(cx: &mut SelectCx<'_>) -> usize {
    plan::builtin(
        &mut cx.families.stage_io,
        crate::spv::BuiltIn::FrontFacing,
        IoDir::Input,
        Shape::Bool,
        crate::pipeline::stage_io::StageBuiltin::FrontFacing,
    )
}

/// Plan the fragment's `SampleMask` input and output for the shader-side stencil's multisampled
/// form: the indexes of their objects, the input shared with a `[[sample_mask]]` input. The form
/// reads the invocation's own coverage, which is not Metal's `[[sample_mask]]` (the pixel's
/// coverage), so the stage interface states only the output, which the module writes.
pub fn sample_masks(cx: &mut SelectCx<'_>) -> (usize, usize) {
    let sp = &mut cx.families.stage_io;
    let covered = plan::builtin_object(
        sp,
        crate::spv::BuiltIn::SampleMask,
        IoDir::Input,
        Shape::U32Array(1),
    );
    let kept = plan::builtin(
        sp,
        crate::spv::BuiltIn::SampleMask,
        IoDir::Output,
        Shape::U32Array(1),
        crate::pipeline::stage_io::StageBuiltin::SampleMaskOut,
    );
    (covered, kept)
}

/// The form of an output of AIR type `ty` that must be a 32-bit integer scalar, as a stencil or
/// sample-mask output must, for a family that writes it outside the stage interface (the
/// shader-side stencil).
pub fn int_scalar_output(cx: &SelectCx<'_>, ty: crate::air::ir::TypeId) -> Result<Form, Error> {
    plan::int_scalar_form(cx, ty)
}

/// The global [`declare`] declared for planned object `i`.
pub fn object_global(cx: &mut EmitCx<'_>, i: usize) -> BResult<GlobalId> {
    global(cx, i)
}

/// The global of object `i`.
fn global(cx: &mut EmitCx<'_>, i: usize) -> BResult<GlobalId> {
    match cx.decl.families.stage_io.globals.get(i).copied() {
        Some(g) => Ok(g),
        None => Err(cx.fail(format!("stage-IO object {i} is not declared"))),
    }
}

/// A pointer to the variable of `link` (element 0 of an array object for `Place::Element0`).
fn pointer(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, link: &Link) -> BResult<SValue> {
    let g = global(cx, link.object)?;
    let base = fb.module().global_value(g)?;
    match link.place {
        Place::Whole | Place::Weights { .. } => Ok(base),
        Place::Element0 | Place::Spread(_) => {
            let zero = fb.u32_const(0)?;
            let zero = fb.module().const_value(zero)?;
            fb.access_chain(base, &[zero])
        }
    }
}

/// Whether `link` is a user variable holding a `bool` value: the interface carries it as a
/// `uint` 0 or 1 (a `bool` has no interface type).
fn bool_as_uint(cx: &EmitCx<'_>, link: &Link) -> bool {
    link.form.held == Held::Bool
        && matches!(
            cx.plan.families.stage_io.objects.get(link.object),
            Some(IoObject::User { .. })
        )
}

/// The `uint` form with the lanes of `form`.
fn uint_lanes(form: Form) -> Form {
    form.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

/// `x`, a value loaded through `link`, as the lowered representation of its form: a narrow
/// integer masked to its width, a narrow float rounded to its format, a `bool` carried as `uint`
/// compared with 0, the weights of a [`Place::Weights`] read.
fn to_lowered(
    cx: &EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    link: &Link,
    x: SValue,
) -> BResult<SValue> {
    let form = link.form;
    let x = placed(fb, link, x)?;
    if bool_as_uint(cx, link) {
        let zero = ops::splat(fb, uint_lanes(form), 0)?;
        return fb.binary(BinOp::INotEqual, x, zero);
    }
    let x = ops::mask(fb, form, x)?;
    narrow::round(fb, form, x)
}

/// The value `x` loaded through `link` holds at its place: the leading barycentric weights, or
/// the whole.
fn placed(fb: &mut FunctionBuilder<'_>, link: &Link, x: SValue) -> BResult<SValue> {
    let Place::Weights { lanes, rotated } = link.place else {
        return Ok(x);
    };
    let x = match rotated {
        true => rotated_weights(fb, x)?,
        false => x,
    };
    Ok(match lanes {
        1 => fb.composite_extract(x, &[0])?,
        2 => fb.vector_shuffle(x, x, &[0, 1])?,
        _ => weights(fb, x)?,
    })
}

/// `1 - (x + y)` of `x` and `y`, each operation rounded on its own (`NoContraction`), so no
/// device fuses or reorders it.
fn one_less_sum(fb: &mut FunctionBuilder<'_>, x: SValue, y: SValue) -> BResult<SValue> {
    use crate::lower::arith::FloatFmt;
    let sum = fb.binary_no_contraction(BinOp::FAdd, x, y)?;
    let f32_ = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let one = ops::splat(fb, f32_, u64::from(1.0f32.to_bits()))?;
    fb.binary_no_contraction(BinOp::FSub, one, sum)
}

/// The three barycentric weights of `x`, a `float3` (`BaryCoordKHR`) or `float2`
/// (`IoRole::Barycentric`, or [`rotated_weights`]') whose first two lanes are the first two
/// weights: those, and `1 - (x + y)` as the third, Metal's (`plan::input_builtin`).
fn weights(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let f32_ty = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f32_ty,
        len: 3,
    })?;
    let first = fb.composite_extract(x, &[0])?;
    let second = fb.composite_extract(x, &[1])?;
    let third = one_less_sum(fb, first, second)?;
    fb.composite_construct(ty, &[first, second, third])
}

/// Metal's first two barycentric weights of a triangle drawn with Metal's last vertex first,
/// `(c, a, b)` for Metal's `(a, b, c)` (`Place::Weights`'s `rotated`), of `x`, whose first two
/// lanes `(x, y)` are the drawn triangle's first two weights: `(y, 1 - (x + y))`. The first is
/// Metal's bit for bit, since Metal interpolates a rotated triangle alike, and the second its own
/// rounding of Metal's, as the third weight [`weights`] computes from them is (Metal probes on the
/// Apple M3 Max, macOS 27.0.1, 200 triangles: 67,821 of 67,821 pixels' first weight, 9,150 the
/// first two, the rest within `2^-15` of Metal's; 61,548, 9,877 and `2^-17` without perspective).
fn rotated_weights(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let f32_ty = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f32_ty,
        len: 2,
    })?;
    let first = fb.composite_extract(x, &[0])?;
    let second = fb.composite_extract(x, &[1])?;
    let third = one_less_sum(fb, first, second)?;
    fb.composite_construct(ty, &[second, third])
}

/// `x`, the `f32` lanes (of a `float` or `half` form `form`) of a varying the fragment function
/// reads flat because its vertex function writes it the same at every vertex of a primitive
/// ([`Source::Uniform`]), as Metal's interpolation returns such a value, measured on the M3 Max
/// in every qualifier (`center`, `centroid` and `sample`, with and without perspective): a zero or
/// subnormal as `+0`, an infinity as the largest finite value of its sign,
/// a NaN as `+FLT_MAX`, and every other value as it is. (Metal's word for a NaN under
/// perspective varies by pixel; `+FLT_MAX` is its word without perspective.) Computed on the
/// bits, so no device's float controls touch it; a `half` form's rounding then gives Metal's
/// `half` (an infinity from `FLT_MAX`).
fn metal_interpolated_constant(
    fb: &mut FunctionBuilder<'_>,
    form: Form,
    x: SValue,
) -> BResult<SValue> {
    let u = uint_lanes(form);
    let ut = ops::ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exponent = ops::splat(fb, u, 0x7F80_0000)?;
    let e = fb.binary(BinOp::BitwiseAnd, b, exponent)?;
    let zero = ops::splat(fb, u, 0)?;
    let tiny = fb.binary(BinOp::IEqual, e, zero)?;
    let special = fb.binary(BinOp::IEqual, e, exponent)?;
    let magnitude = ops::splat(fb, u, 0x7FFF_FFFF)?;
    let m = fb.binary(BinOp::BitwiseAnd, b, magnitude)?;
    let nan = fb.binary(BinOp::UGreaterThan, m, exponent)?;
    let sign = ops::splat(fb, u, 0x8000_0000)?;
    let s = fb.binary(BinOp::BitwiseAnd, b, sign)?;
    let max = ops::splat(fb, u, 0x7F7F_FFFF)?;
    let signed_max = fb.binary(BinOp::BitwiseOr, s, max)?;
    let big = fb.select(nan, max, signed_max)?;
    let r = fb.select(special, big, b)?;
    let r = fb.select(tiny, zero, r)?;
    let ft = ops::ty(
        fb,
        form.with(Held::Float(crate::lower::arith::FloatFmt::F32)),
    )?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// Fetch every pulled attribute in the function's first block, before the walk emits any
/// instruction of the body, and hold each value for all its uses. The first block dominates every
/// block, and the fetch is straight-line code, so each value is defined in a block every use
/// follows; the body's writes, through the source buffer's slot or any other pointer to its memory,
/// come after it in program order and never reach it, as Metal's attribute is fetched before its
/// function runs.
pub fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    // The de-indexed or private list form's raw index, read once in the first block, where it
    // dominates every use, and the private list form's base vertex.
    if let Some(c) = cx.plan.families.stage_io.captured_index {
        cx.captured_index(fb)?;
        if c.first.is_some() {
            cx.listed_base(fb)?;
        }
    }
    // A merged draw's vertex finds its corner's command before the body, which every read of the
    // command's values and slot table follows.
    if let Some(m) = cx.plan.families.stage_io.merged {
        let header = cx.push_field(fb, PushField::MergedDrawAddress)?;
        let mut index = [None; 2];
        for (x, object) in index.iter_mut().zip([m.vertex, m.instance]) {
            let g = global(cx, object)?;
            let p = fb.module().global_value(g)?;
            *x = Some(fb.load(p, MemAccess::default())?);
        }
        let [Some(vertex), Some(instance)] = index else {
            return Err(cx.fail("a merged draw's vertex without its index builtins"));
        };
        cx.merged_draw = Some(crate::bind::merged_vertex(fb, header, (vertex, instance))?);
    }
    let pulled: Vec<(u32, fetch::PulledRead)> = cx
        .plan
        .families
        .stage_io
        .params
        .iter()
        .filter_map(|p| match p.source {
            Source::Pulled(read) => Some((p.param, read)),
            Source::Io(_)
            | Source::PixelCenter(_)
            | Source::LoopSample(_)
            | Source::AtLoopSample(_)
            | Source::AmplificationCount(_)
            | Source::Unamplified(_)
            | Source::Absent(_)
            | Source::Pushed { .. }
            | Source::DrawBase { .. }
            | Source::StageInRegion { .. }
            | Source::CapturedVertex(_)
            | Source::CapturedBaseVertex(_)
            | Source::Merged { .. }
            | Source::Uniform(_)
            | Source::QuadPrimitive(_) => None,
        })
        .collect();
    for (param, read) in pulled {
        let Some(v) = cx.plan.values.id(Value::Arg(ArgIdx::new(param))) else {
            return Err(cx.fail(format!("pulled parameter {param} is out of range")));
        };
        let x = fetch::emit(cx, fb, &read)?;
        cx.hold(v, x);
    }
    Ok(())
}

/// The value of stage-IO parameter `b` in its lowered representation, loaded at each use (the
/// use may sit in a construct whose values do not dominate the rest of the block). A pulled
/// attribute's value is the one [`emit_entry`] holds, so it never reaches here.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    let planned = cx
        .plan
        .families
        .stage_io
        .params
        .iter()
        .find(|p| p.param == b.param)
        .map(|p| p.source);
    let link = match planned {
        Some(Source::Io(link)) => link,
        Some(Source::PixelCenter(link)) => return pixel_center(cx, fb, &link),
        Some(Source::LoopSample(form)) => {
            let s = loop_sample(cx, fb)?;
            return ops::mask(fb, form, s);
        }
        Some(Source::AtLoopSample(link)) => return at_loop_sample(cx, fb, &link),
        Some(Source::AmplificationCount(form)) => return amplification_count(fb, form),
        Some(Source::Unamplified(_)) => return unamplified_id(fb),
        Some(Source::Absent(form)) => {
            return match crate::lower::arith::legal::zero(&form.demote()) {
                Some(k) => match crate::lower::arith::legal::materialize(fb, &k)? {
                    Some(x) => Ok(x),
                    None => Err(cx.fail("an absent input's zero does not materialize")),
                },
                None => Err(cx.fail("an absent input of a type without a zero")),
            };
        }
        Some(Source::Pushed { field, form }) => {
            let x = cx.push_field(fb, field)?;
            return ops::mask(fb, form, x);
        }
        Some(Source::DrawBase { base, form }) => {
            let v = cx.push_field(fb, PushField::BaseVertexInstance)?;
            let x = fb.composite_extract(v, &[base.lane()])?;
            return ops::mask(fb, form, x);
        }
        Some(Source::CapturedVertex(form)) => {
            let x = captured_vertex(cx, fb)?;
            return ops::mask(fb, form, x);
        }
        Some(Source::CapturedBaseVertex(form)) => {
            let x = captured_base_vertex(cx, fb)?;
            return ops::mask(fb, form, x);
        }
        Some(Source::Merged { value, form }) => {
            let Some(m) = cx.merged_draw else {
                return Err(cx.fail("a merged draw's value read before its entry found it"));
            };
            let x = match value {
                MergedValue::Vertex => m.vertex,
                MergedValue::Instance => m.instance,
                MergedValue::BaseVertex => m.base_vertex,
                MergedValue::BaseInstance => m.base_instance,
            };
            return ops::mask(fb, form, x);
        }
        Some(Source::StageInRegion { size, projection }) => {
            return stage_in::region_value(cx, fb, size, projection)
        }
        Some(Source::Pulled(_)) => {
            return Err(cx.fail(format!(
                "pulled parameter {} was not fetched at entry",
                b.param
            )))
        }
        Some(Source::Uniform(link)) => {
            let p = pointer(cx, fb, &link)?;
            let x = fb.load(p, MemAccess::default())?;
            let x = placed(fb, &link, x)?;
            let x = metal_interpolated_constant(fb, link.form, x)?;
            return narrow::round(fb, link.form, x);
        }
        Some(Source::QuadPrimitive(link)) => {
            let p = pointer(cx, fb, &link)?;
            let x = fb.load(p, MemAccess::default())?;
            let one = crate::memory::leaf::k32(fb, 1)?;
            let x = fb.binary(BinOp::ShiftRightLogical, x, one)?;
            return to_lowered(cx, fb, &link, x);
        }
        None => return Err(cx.fail(format!("parameter {} has no stage-IO value", b.param))),
    };
    let x = match link.place {
        Place::Element0 if is_coverage_input(cx, link.object) => coverage_in(cx, fb, link.object)?,
        _ => {
            let p = pointer(cx, fb, &link)?;
            fb.load(p, MemAccess::default())?
        }
    };
    to_lowered(cx, fb, &link, x)
}

/// Whether planned object `i` is the `SampleMask` input.
fn is_coverage_input(cx: &EmitCx<'_>, i: usize) -> bool {
    matches!(
        cx.plan.families.stage_io.objects.get(i),
        Some(IoObject::Builtin {
            builtin: crate::spv::BuiltIn::SampleMask,
            dir: IoDir::Input,
            ..
        })
    )
}

/// The invocation's coverage word, `SampleMask[0]` of the input object `i`, loaded: from the copy
/// the entry stores before it calls the body, where the entry wraps the body, which SPIRV-Cross
/// (MoltenVK) cannot pass the input array to; else from the variable.
pub fn coverage_in(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, i: usize) -> BResult<SValue> {
    if let Some(copy) = cx.decl.families.fragment.coverage_copy {
        let p = fb.module().global_value(copy)?;
        return fb.load(p, MemAccess::default());
    }
    let g = global(cx, i)?;
    let base = fb.module().global_value(g)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let word = fb.access_chain(base, &[zero])?;
    fb.load(word, MemAccess::default())
}

/// The value of `link`'s variable at the pixel center ([`Source::PixelCenter`]), in its lowered
/// representation. `FragCoord`'s x and y are the shaded location's, a sample's under sample
/// shading, which lies in the pixel `[x, x + 1) × [y, y + 1)` (sample locations lie in the unit
/// square); `floor(..) + 0.5` is that pixel's center, exact in `float` for every framebuffer
/// coordinate. A varying is interpolated at offset `(0, 0)`, which `GLSL.std.450`'s
/// `InterpolateAtOffset` defines as the center of the pixel.
fn pixel_center(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, link: &Link) -> BResult<SValue> {
    use crate::lower::arith::FloatFmt;
    use crate::spirv::ir::GlslOp;
    let f32_lanes = |lanes| Form {
        held: Held::Float(FloatFmt::F32),
        lanes,
    };
    let p = pointer(cx, fb, link)?;
    let x = match cx.plan.families.stage_io.objects.get(link.object) {
        Some(IoObject::Builtin {
            builtin: crate::spv::BuiltIn::FragCoord,
            ..
        }) => {
            let v = fb.load(p, MemAccess::default())?;
            let floor = fb.glsl(GlslOp::Floor, &[v])?;
            let half = ops::splat(fb, f32_lanes(4), u64::from(0.5f32.to_bits()))?;
            let center = fb.binary(BinOp::FAdd, floor, half)?;
            // The center's x and y, the shaded location's z and w.
            fb.vector_shuffle(center, v, &[0, 1, 6, 7])?
        }
        Some(IoObject::User {
            numeric: IoNumeric::Float,
            ..
        }) => {
            let zero = ops::splat(fb, f32_lanes(2), 0)?;
            fb.glsl(GlslOp::InterpolateAtOffset, &[p, zero])?
        }
        _ => {
            return Err(cx.fail(format!(
                "stage-IO object {} is read at the pixel center",
                link.object
            )))
        }
    };
    to_lowered(cx, fb, link, x)
}

/// The sample of the pass of the loop over the covered samples: the fragment family's
/// `loop_sample`, a `u32`.
fn loop_sample(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let Some(g) = cx.decl.families.fragment.loop_sample else {
        return Err(cx.fail("a loop sample read without the loop"));
    };
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default())
}

/// The value of `link`'s variable, a sample-qualified float varying declared without `Sample`,
/// at the sample of the loop's pass, in its lowered representation: `InterpolateAtSample` at
/// [`loop_sample`], or, where the entry read the varying at every sample before the loop
/// ([`read_at_every_sample`]), that sample's element of its array.
fn at_loop_sample(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    link: &Link,
) -> BResult<SValue> {
    use crate::spirv::ir::GlslOp;
    let s = loop_sample(cx, fb)?;
    let x = match cx
        .decl
        .families
        .stage_io
        .at_samples
        .get(&link.object)
        .copied()
    {
        Some(array) => {
            let base = fb.module().global_value(array)?;
            let element = fb.access_chain(base, &[s])?;
            fb.load(element, MemAccess::default())?
        }
        None => {
            let p = pointer(cx, fb, link)?;
            fb.glsl(GlslOp::InterpolateAtSample, &[p, s])?
        }
    };
    to_lowered(cx, fb, link, x)
}

/// Read each sample-qualified varying of a function looping over its covered samples into its
/// array ([`Declared::at_samples`]) at every sample before the loop, in the open block of the
/// entry `fb` builds, where the driver's interpolation needs the whole quad
/// (`DriverWorkaround::InterpolationNeedsTheWholeQuad`): the helper invocations of a partly covered
/// quad leave the loop at once, and there that driver's interpolation at a sample reads garbage, so
/// the reads come first, where every invocation of the quad runs. Sample `k` is read by the
/// constant `k`, for `k` from 0 to the most samples a pipeline of the device may have less one,
/// each `k` but 0 under `SampleCount > k`, the pipeline's count, so no read names a sample the
/// pipeline lacks; that condition is the same in every invocation.
pub fn read_at_every_sample(
    fb: &mut FunctionBuilder<'_>,
    plan: &Plan,
    decls: &crate::lower::emit::FamilyDecls,
) -> Result<(), Error> {
    use crate::spirv::builder::constructs::ArmEnd;
    use crate::spirv::ir::GlslOp;
    let Some(n) = plan
        .families
        .fragment
        .coverage
        .and_then(|c| c.samples_before_loop)
    else {
        return Ok(());
    };
    let declared = &decls.stage_io;
    if declared.at_samples.is_empty() {
        return Ok(());
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let count = fb.module().spec(SpecKey::SampleCount, u32_, ConstBits(1))?;
    let count = fb.module().const_value(count)?;
    for (&object, &array) in &declared.at_samples {
        let Some(var) = declared.globals.get(object).copied() else {
            return Err(Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "a varying read at every sample without its variable",
            )));
        };
        let var = fb.module().global_value(var)?;
        let array = fb.module().global_value(array)?;
        for k in 0..n {
            let at = fb.u32_const(k)?;
            let at = fb.module().const_value(at)?;
            let read = move |fb: &mut FunctionBuilder<'_>| -> BResult<()> {
                let x = fb.glsl(GlslOp::InterpolateAtSample, &[var, at])?;
                let element = fb.access_chain(array, &[at])?;
                fb.store(element, x, MemAccess::default())
            };
            if k == 0 {
                read(fb)?;
                continue;
            }
            let has = fb.binary(BinOp::UGreaterThan, count, at)?;
            fb.if_(
                &mut (),
                has,
                Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                    read(fb)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
        }
    }
    Ok(())
}

/// `[[amplification_count]]` into an integer of `form`: the `SpecKey::AmplificationCount`
/// specialization constant, default 1 (Metal's count without amplification), which the binder sets
/// to the encoder's count. The count is at most 8 (Metal's largest), so a `ushort` holds it; the
/// mask keeps the narrow representation whatever the value.
pub(crate) fn amplification_count(fb: &mut FunctionBuilder<'_>, form: Form) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let c = fb
        .module()
        .spec(SpecKey::AmplificationCount, u32_, ConstBits(1))?;
    let x = fb.module().const_value(c)?;
    ops::mask(fb, form, x)
}

/// `[[amplification_id]]` of a vertex function that is never drawn amplified: 0, in the 32-bit
/// container of every amplification id form (no mask needed).
fn unamplified_id(fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let c = fb.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(0),
    })?;
    fb.module().const_value(c)
}

/// The `u32` constant `x`.
fn u32_constant(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let c = fb.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(x)),
    })?;
    fb.module().const_value(c)
}

/// The render pipeline's private masks a fragment function applies itself
/// (`LoweringInputs::sample_mask`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateMask {
    /// The `SampleMask` output's object.
    pub object: usize,
    /// The masks by the window pixel's place in its 2x2 quad, index `(x & 1) | (y & 1) << 1`.
    pub masks: [u32; 4],
    /// The `FragCoord` input's object, whose pixel picks the mask, where the masks differ
    /// between the places; `None` where they are alike.
    pub coord: Option<usize>,
}

/// The private mask `m` of the fragment's pixel: the one mask where the masks are alike, else
/// picked by the window pixel's coordinates modulo 2 from the masks packed a byte each (a
/// mask that differs between places is Metal's sample coverage, at 4 samples or fewer).
fn private_mask_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    m: PrivateMask,
) -> BResult<SValue> {
    let Some(coord) = m.coord else {
        return u32_constant(fb, m.masks[0]);
    };
    if m.masks.iter().any(|&mask| mask > 0xff) {
        return Err(cx.fail("a per-pixel private mask past 8 samples"));
    }
    let packed = m
        .masks
        .iter()
        .enumerate()
        .fold(0u32, |packed, (i, &mask)| packed | mask << (8 * i));
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let g = global(cx, coord)?;
    let p = fb.module().global_value(g)?;
    let frag = fb.load(p, MemAccess::default())?;
    // `FragCoord` is the pixel's center (or a sample's position inside the pixel): its integer
    // part is the window pixel.
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    let one = u32_constant(fb, 1)?;
    let x = fb.binary(BinOp::BitwiseAnd, x, one)?;
    let y = fb.binary(BinOp::BitwiseAnd, y, one)?;
    let y = fb.binary(BinOp::ShiftLeftLogical, y, one)?;
    let place = fb.binary(BinOp::BitwiseOr, x, y)?;
    let three = u32_constant(fb, 3)?;
    let shift = fb.binary(BinOp::ShiftLeftLogical, place, three)?;
    let packed = u32_constant(fb, packed)?;
    let mask = fb.binary(BinOp::ShiftRightLogical, packed, shift)?;
    let byte = u32_constant(fb, 0xff)?;
    fb.binary(BinOp::BitwiseAnd, mask, byte)
}

/// Store the entry's returned value `v` into its stage outputs, before the return terminator
/// (`fragment::emit_return`), and write the default point size. A gated-off output's field is
/// not stored. A fragment function compiled under `LoweringInputs::sample_mask` ANDs its pixel's
/// mask into its `[[sample_mask]]` output, or stores it as that output.
pub fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<ValueId>,
) -> BResult<()> {
    let plan = &cx.plan.families.stage_io;
    let (outputs, returned, point) = (plan.outputs.clone(), plan.returned, plan.point_size_default);
    let seconds = plan.rotated_seconds.clone();
    let compensation = plan.compensation;
    let depth_clamp = plan.depth_clamp;
    let window_depth = plan.window_depth;
    let primitive_id = plan.primitive_id;
    // The private masks: the pixel's ANDed into the function's own `[[sample_mask]]` output
    // where this return stores one, else stored as that output.
    let private_mask = plan.private_mask;
    let mut masked = false;
    // The private alpha test, before any output is stored: a failing fragment is killed there.
    if let (Some(v), Some(test)) = (v, plan.alpha_test) {
        let x = cx.operand(fb, v)?;
        alpha_test(cx, fb, test, x, returned)?;
    }
    match (v, returned) {
        (None, Returned::Void) => {}
        (None, _) => return Err(cx.fail("an entry with outputs returns no value")),
        (Some(v), Returned::Void) => {
            return Err(cx.fail(format!("the entry returns {v:?} without outputs")))
        }
        // Every output is gated off: the value is stored nowhere.
        (Some(_), _) if outputs.is_empty() => {}
        (Some(v), _) => {
            let x = cx.operand(fb, v)?;
            // The viewport the function selects, which a compensation of mirrors reads.
            let selected = match compensation.filter(|c| c.mirrors) {
                Some(_) => Some(selected_viewport(cx, fb, x, returned, &outputs)?),
                None => None,
            };
            for o in &outputs {
                let part = output_part(cx, fb, x, returned, o)?;
                let part =
                    match point_size_object(&cx.plan.families.stage_io.objects, o.link.object) {
                        true => metal_point_size(fb, part)?,
                        false => part,
                    };
                let part = match compensation.filter(|c| c.position == o.link.object) {
                    Some(c) => compensate(cx, fb, c, part, selected)?,
                    None => part,
                };
                let part = match depth_clamp.filter(|d| d.object == o.link.object) {
                    Some(d) => {
                        let kill = nan_drop(cx)?;
                        let range = match d.range {
                            true => Some(cx.push_field(fb, PushField::DepthClampRange)?),
                            false => None,
                        };
                        clamp_depth(fb, part, kill, d.clamps || d.range, range)?
                    }
                    None => part,
                };
                if let Some(w) = window_depth.filter(|w| w.position == o.link.object) {
                    store_window_depth(cx, fb, w, part)?;
                }
                let part = match private_mask.filter(|m| m.object == o.link.object) {
                    Some(m) => {
                        masked = true;
                        let mask = private_mask_value(cx, fb, m)?;
                        fb.binary(BinOp::BitwiseAnd, part, mask)?
                    }
                    None => part,
                };
                store(cx, fb, &o.link, part)?;
            }
            for s in &seconds {
                let first = match returned {
                    Returned::Struct => fb.composite_extract(x, &[s.field])?,
                    Returned::Value | Returned::Void => x,
                };
                let second = rotated_second(fb, s, first)?;
                store(cx, fb, &s.link, second)?;
            }
        }
    }
    if let Some(m) = private_mask.filter(|_| !masked) {
        let link = Link {
            object: m.object,
            place: Place::Element0,
            form: Form {
                held: Held::Int {
                    bits: 32,
                    container: 32,
                },
                lanes: 1,
            },
        };
        let mask = private_mask_value(cx, fb, m)?;
        store(cx, fb, &link, mask)?;
    }
    // The primitive ID of the primitive the vertex provokes, wherever the function returns its
    // outputs.
    if let (Some(_), Some(p)) = (v, primitive_id) {
        store_primitive_id(cx, fb, p)?;
    }
    // The indices a draw of one amplification stores though the function writes none.
    let amplified = cx.plan.families.stage_io.amplified.clone();
    for o in &amplified {
        let part = cx.push_field(fb, o.offset)?;
        let part = below_count(cx, fb, part, o.below)?;
        store(cx, fb, &o.link, part)?;
    }
    // The pass's colors the pipeline lacks, which plain Metal zeroes where the draw covers them.
    let zeroed = cx.plan.families.stage_io.zeroed.clone();
    for link in &zeroed {
        let zero = ops::splat(fb, link.form, 0)?;
        store(cx, fb, link, zero)?;
    }
    if let Some(i) = point {
        let g = global(cx, i)?;
        let p = fb.module().global_value(g)?;
        let f32_ = fb.module().ty(SType::Float { width: 32 })?;
        let one = fb.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(1.0f32.to_bits())),
        })?;
        let one = fb.module().const_value(one)?;
        fb.store(p, one, MemAccess::default())?;
    }
    Ok(())
}

/// Metal's largest point, in pixels: a `[[point_size]]` past it draws as this size (an Apple M4;
/// bracketed in `(510.996, 511.004]`), where a Vulkan device clamps only to its own
/// `pointSizeRange`, up to 2047.94 on the RTX 5080 and 8191.875 on RADV.
pub(crate) const METAL_POINT_SIZE_LIMIT: f32 = 511.0;

/// Whether interface object `object` of `objects` is the `PointSize` output, whose stores take
/// Metal's largest point ([`metal_point_size`]).
fn point_size_object(objects: &[IoObject], object: usize) -> bool {
    matches!(
        objects.get(object),
        Some(IoObject::Builtin {
            builtin: crate::spv::BuiltIn::PointSize,
            dir: IoDir::Output,
            ..
        })
    )
}

/// The point size `size` (an `f32`) a vertex or mesh function stores to `PointSize`, as Metal
/// draws it: [`METAL_POINT_SIZE_LIMIT`] where `size` is past it (an infinity too), `size` itself
/// otherwise. A NaN fails the ordered comparison and is stored as written: Metal's answer for a
/// NaN, a negative or a denormal size is not measured.
pub(crate) fn metal_point_size(fb: &mut FunctionBuilder<'_>, size: SValue) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let limit = fb.module().constant(Constant::Scalar {
        ty: f32_,
        bits: ConstBits(u64::from(METAL_POINT_SIZE_LIMIT.to_bits())),
    })?;
    let limit = fb.module().const_value(limit)?;
    let past = fb.binary(BinOp::FOrdGreaterThan, size, limit)?;
    fb.select(past, limit, size)
}

/// The words [`metal_point_size`] adds to a store: the comparison, the select and the limit's
/// constant.
pub(crate) const POINT_SIZE_WORDS: u64 = 3;

/// Store the window depth `z / w` of position `position`, as the vertex function stores it, to
/// its varying ([`WindowDepth`]): the division of the measured form, unfused.
fn store_window_depth(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    w: WindowDepth,
    position: SValue,
) -> BResult<()> {
    let z = fb.composite_extract(position, &[2])?;
    let clip_w = fb.composite_extract(position, &[3])?;
    let depth = fb.binary(BinOp::FDiv, z, clip_w)?;
    let link = Link {
        object: w.object,
        place: Place::Whole,
        form: Form {
            held: Held::Float(crate::lower::arith::FloatFmt::F32),
            lanes: 1,
        },
    };
    store(cx, fb, &link, depth)
}

/// Store the primitive ID of the primitive the vertex provokes to its varying
/// ([`PrimitiveIdStore`]): `(VertexIndex - first) / n`, `n` from
/// `PushField::VerticesPerPrimitive`.
fn store_primitive_id(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: PrimitiveIdStore,
) -> BResult<()> {
    let first = match p.first {
        FirstVertex::Builtin { index } => Some(fetch::load_builtin(cx, fb, index)?),
        FirstVertex::Push => {
            let bases = cx.push_field(fb, PushField::BaseVertexInstance)?;
            Some(fb.composite_extract(bases, &[0])?)
        }
        FirstVertex::Zero => None,
    };
    let per_primitive = cx.push_field(fb, PushField::VerticesPerPrimitive)?;
    let vertex = fetch::load_builtin(cx, fb, p.vertex)?;
    let position = match first {
        Some(first) => fb.binary(BinOp::ISub, vertex, first)?,
        None => vertex,
    };
    let id = fb.binary(BinOp::UDiv, position, per_primitive)?;
    let link = Link {
        object: p.object,
        place: Place::Whole,
        form: Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        },
    };
    store(cx, fb, &link, id)
}

/// The raw index a de-indexed or private list form's vertex reads ([`CapturedIndex`]), loaded in
/// the open block: `PushField::CaptureIndexAddress + p · size` with `p = VertexIndex` (the
/// de-indexed draw starts at vertex 0), in 64 bits. A `u32` index is read where it lies (the address is a
/// multiple of 4); a `u16` one from the aligned word that holds it, shifted right by 16 where its
/// address's bit 1 is set, and masked to 16 bits. [`EmitCx::captured_index`] reads it once where
/// it dominates.
pub(crate) fn read_captured_index(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
) -> BResult<SValue> {
    let Some(c) = cx.plan.families.stage_io.captured_index else {
        return Err(cx.fail("an index read without its plan"));
    };
    let u64_ = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let k64 = |fb: &mut FunctionBuilder<'_>, x: u64| -> BResult<SValue> {
        let k = fb.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(x),
        })?;
        fb.module().const_value(k)
    };
    let k32 = |fb: &mut FunctionBuilder<'_>, x: u32| crate::memory::leaf::k32(fb, x);
    let g = global(cx, c.vertex)?;
    let p = fb.module().global_value(g)?;
    let vertex = fb.load(p, MemAccess::default())?;
    let position = fb.convert(ConvOp::UConvert, vertex, u64_)?;
    let size = match c.ty {
        crate::metal::IndexType::UInt16 => 2,
        crate::metal::IndexType::UInt32 => 4,
    };
    let size = k64(fb, size)?;
    let offset = fb.binary(BinOp::IMul, position, size)?;
    let base = cx.push_field(fb, PushField::CaptureIndexAddress)?;
    let at = fb.binary(BinOp::IAdd, base, offset)?;
    let aligned = MemAccess {
        aligned: Some(4),
        ..MemAccess::default()
    };
    match c.ty {
        crate::metal::IndexType::UInt32 => {
            let place = fb.device_place(at, u32_)?;
            fb.load(place, aligned)
        }
        crate::metal::IndexType::UInt16 => {
            let mask = k64(fb, !3)?;
            let word_at = fb.binary(BinOp::BitwiseAnd, at, mask)?;
            let place = fb.device_place(word_at, u32_)?;
            let word = fb.load(place, aligned)?;
            // Bit 1 of the address selects the word's high half: a shift of 0 or 16.
            let low = fb.convert(ConvOp::UConvert, at, u32_)?;
            let two = k32(fb, 2)?;
            let half = fb.binary(BinOp::BitwiseAnd, low, two)?;
            let three = k32(fb, 3)?;
            let shift = fb.binary(BinOp::ShiftLeftLogical, half, three)?;
            let shifted = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
            let bits = k32(fb, 0xffff)?;
            fb.binary(BinOp::BitwiseAnd, shifted, bits)
        }
    }
}

/// The vertex a de-indexed or private list form's function runs ([`CapturedIndex`]): its raw
/// index plus the base vertex ([`captured_base_vertex`]), wrapping as `u32`s.
pub(super) fn captured_vertex(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
) -> BResult<SValue> {
    let raw = cx.captured_index(fb)?;
    let base = captured_base_vertex(cx, fb)?;
    fb.binary(BinOp::IAdd, raw, base)
}

/// The base vertex of a form that reads its draw's indices ([`CapturedIndex`]): the layout's
/// (lane 3 of `PushField::CaptureLayout`), or the private list form's word before its list.
fn captured_base_vertex(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    match cx
        .plan
        .families
        .stage_io
        .captured_index
        .and_then(|c| c.first)
    {
        Some(_) => cx.listed_base(fb),
        None => {
            let layout = cx.push_field(fb, PushField::CaptureLayout)?;
            fb.composite_extract(layout, &[0, 3])
        }
    }
}

/// The private list form's base vertex ([`CapturedIndex`]), loaded in the open block: the `u32`
/// at `PushField::CaptureIndexAddress + 4 · (first - 1)`, the word before the list's first
/// index, `first` the draw's first vertex, in 64 bits. [`EmitCx::listed_base`] reads it once
/// where it dominates.
pub(crate) fn read_listed_base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
) -> BResult<SValue> {
    let Some(first) = cx
        .plan
        .families
        .stage_io
        .captured_index
        .and_then(|c| c.first)
    else {
        return Err(cx.fail("a list's base vertex read without its plan"));
    };
    let first = load_first_vertex(cx, fb, first)?;
    let u64_ = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let k64 = |fb: &mut FunctionBuilder<'_>, x: u64| -> BResult<SValue> {
        let k = fb.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(x),
        })?;
        fb.module().const_value(k)
    };
    let first = fb.convert(ConvOp::UConvert, first, u64_)?;
    let four = k64(fb, 4)?;
    let offset = fb.binary(BinOp::IMul, first, four)?;
    let offset = fb.binary(BinOp::ISub, offset, four)?;
    let base = cx.push_field(fb, PushField::CaptureIndexAddress)?;
    let at = fb.binary(BinOp::IAdd, base, offset)?;
    let place = fb.device_place(at, u32_)?;
    fb.load(
        place,
        MemAccess {
            aligned: Some(4),
            ..MemAccess::default()
        },
    )
}

/// The draw's first vertex, read where `first` says ([`FirstVertex`]): `None` for
/// [`FirstVertex::Zero`].
fn load_first_vertex(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    first: FirstVertex,
) -> BResult<SValue> {
    match first {
        FirstVertex::Builtin { index } => fetch::load_builtin(cx, fb, index),
        FirstVertex::Push => {
            let bases = cx.push_field(fb, PushField::BaseVertexInstance)?;
            fb.composite_extract(bases, &[0])
        }
        FirstVertex::Zero => crate::memory::leaf::k32(fb, 0),
    }
}

/// Output `o`'s part of the returned value `x`, as it is stored: its field, the amplification's
/// offset added, held below its count, and a color's rounding.
fn output_part(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    returned: Returned,
    o: &PlannedOutput,
) -> BResult<SValue> {
    let part = match returned {
        // A `bool` vector field is held as `u32` lanes of 0 and 1.
        Returned::Struct if o.link.form.is_bool() && o.link.form.lanes > 1 => {
            let x = fb.composite_extract(x, &[o.field])?;
            ops::member_to_bools(fb, x, o.link.form.lanes)?
        }
        Returned::Struct => fb.composite_extract(x, &[o.field])?,
        Returned::Value | Returned::Void => x,
    };
    let part = match o.offset {
        Some(offset) => {
            let offset = cx.push_field(fb, offset)?;
            fb.binary(BinOp::IAdd, part, offset)?
        }
        None => part,
    };
    let part = match o.below {
        Some(count) => below_count(cx, fb, part, count)?,
        None => part,
    };
    let part = match o.color {
        // An extended-range attachment's blend the function computes over its input attachment.
        Some(color) if cx.plan.families.fragment.input_blends.contains_key(&color) => {
            input_blended(cx, fb, (x, returned), color, (o.link.form, part))?
        }
        Some(color) => match (
            cx.plan.families.stage_io.words_written.get(&color).copied(),
            cx.plan.families.stage_io.extended.get(&color).copied(),
        ) {
            (Some(w), _) => {
                let word = word_color::encode(fb, w, (o.link.form, part))?;
                // A write of some of the word's channels over the attachment's input attachment.
                match cx.plan.families.fragment.word_merges.get(&color).copied() {
                    Some(kept) => {
                        crate::lower::fragment::word_input_merge(cx, fb, color, kept, word)?
                    }
                    None => word,
                }
            }
            (None, Some(e)) => extended_color::encode(fb, e, o.link.form, part)?,
            (None, None) => color_write::round(cx, fb, color, o.link.form, part)?,
        },
        None => part,
    };
    // Metal's red, green, blue and alpha, held as the image's green, blue, alpha and red; an
    // image holding Metal's alpha in red takes that alpha in its alpha lane as well.
    let part = match (o.rotated, o.alpha_red) {
        (true, false) => fb.vector_shuffle(part, part, &[3, 0, 1, 2])?,
        (true, true) => fb.vector_shuffle(part, part, &[3, 0, 1, 3])?,
        (false, _) => part,
    };
    match o.premultiplied {
        true => premultiplied_blue(fb, o.link.form, part),
        false => Ok(part),
    }
}

/// The raw components a write of first output `part` (of `form`) to extended-range attachment
/// `color` stores, whose blend the function computes over the attachment's input attachment
/// (`lower::fragment`'s `input_blend`), the second source read from the returned value `x` (of
/// shape `returned`) where the blend reads it.
fn input_blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (x, returned): (SValue, Returned),
    color: u32,
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    let plan = cx.plan;
    let Some(ib) = plan.families.fragment.input_blends.get(&color) else {
        return Err(cx.fail("an input blend of an attachment the plan does not name"));
    };
    let second = match (ib.second, returned) {
        (Some((field, form)), Returned::Struct) => Some((form, fb.composite_extract(x, &[field])?)),
        (Some(_), _) => return Err(cx.fail("a second output outside a returned structure")),
        (None, _) => None,
    };
    let texel =
        crate::lower::fragment::input_blend_texel(cx, fb, (color, ib), (form, part), second)?;
    extended_color::raw_components(fb, ib.form, texel)
}

/// The rotated first output `part`, of float form `form`, with its alpha lane (Metal's blue)
/// premultiplied by its red lane (Metal's alpha) (`PlannedOutput::premultiplied`): each clamped
/// to `[0, 1]` with a NaN taken as 0, as Metal's blend clamps its source, and their product not
/// contracted, which Apple's blend then adds as its own `s * a`
/// (`RotatedBlendForm::Premultiplied`).
fn premultiplied_blue(fb: &mut FunctionBuilder<'_>, form: Form, part: SValue) -> BResult<SValue> {
    let one = Form { lanes: 1, ..form };
    let zero = ops::fsplat(fb, one, 0.0)?;
    let unit = ops::fsplat(fb, one, 1.0)?;
    let saturated = |fb: &mut FunctionBuilder<'_>, lane: u32| -> BResult<SValue> {
        let v = fb.composite_extract(part, &[lane])?;
        let is_nan = fb.unary(UnOp::IsNan, v)?;
        let v = fb.select(is_nan, zero, v)?;
        fb.glsl(GlslOp::NClamp, &[v, zero, unit])
    };
    let blue = saturated(fb, 3)?;
    let alpha = saturated(fb, 0)?;
    let product = fb.binary_no_contraction(BinOp::FMul, blue, alpha)?;
    fb.composite_insert(product, part, &[3])
}

/// The second output `s` holds beside the first output `first`, as the function returns it
/// (`LoweringInputs::rotated_factors`): each channel of the image a copy of one of `first`'s
/// channels, or `0` or `1`, so the blend's `SRC1` factors read the values Metal's factors read.
fn rotated_second(
    fb: &mut FunctionBuilder<'_>,
    s: &RotatedSecond,
    first: SValue,
) -> BResult<SValue> {
    use crate::lower::arith::FloatFmt;
    use crate::pipeline::lower_inputs::FactorLane;
    let one = match s.link.form.float_fmt() {
        Some(FloatFmt::F64) => 1.0f64.to_bits(),
        _ => u64::from(1.0f32.to_bits()),
    };
    // `(0, 1, 0, 0)`, whose first two lanes the shuffle below reads as 4 and 5.
    let zero = ops::splat(fb, s.link.form, 0)?;
    let ones = ops::splat(fb, s.link.form, one)?;
    let constants = fb.vector_shuffle(zero, ones, &[0, 4, 0, 0])?;
    let lanes = s.lanes.map(|l| match l {
        FactorLane::Zero => 4,
        FactorLane::One => 5,
        FactorLane::Output(m) => u32::from(m),
    });
    fb.vector_shuffle(first, constants, &lanes)
}

/// The viewport a vertex function returning `x` selects, as Vulkan reads it: the `ViewportIndex`
/// it stores (`output_part`) or the `IoRole::SelectedViewport` varying the synthesized geometry
/// stages store it from, else the one an amplification stores (`StageIoPlan::amplified`), else
/// the first.
fn selected_viewport(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    returned: Returned,
    outputs: &[PlannedOutput],
) -> BResult<SValue> {
    let sp = &cx.plan.families.stage_io;
    // The `ViewportIndex` the function stores, or the varying the synthesized geometry stages
    // store it from (`IoRole::SelectedViewport`): either holds the index below the count.
    let selected_key = crate::pipeline::sym::VaryingKey::selected_viewport();
    let is_index = |o: &IoObject| match o {
        IoObject::Builtin {
            builtin: crate::spv::BuiltIn::ViewportIndex,
            dir: IoDir::Output,
            ..
        } => true,
        IoObject::User {
            key,
            dir: IoDir::Output,
            ..
        } => *key == selected_key,
        _ => false,
    };
    let object = sp.objects.iter().position(is_index);
    let amplified = sp
        .amplified
        .iter()
        .find(|a| Some(a.link.object) == object)
        .copied();
    if let Some(o) = outputs.iter().find(|o| Some(o.link.object) == object) {
        return output_part(cx, fb, x, returned, o);
    }
    match amplified {
        Some(a) => {
            let offset = cx.push_field(fb, a.offset)?;
            below_count(cx, fb, offset, a.below)
        }
        None => {
            let zero = fb.u32_const(0)?;
            fb.module().const_value(zero)
        }
    }
}

/// `position`, a vertex function's `float4` position, compensated for the draw's viewport by
/// `PushField::ViewportCompensation`'s `[sx, a, b, clip]`
/// (`LoweringInputs::viewport_compensation`): `(sx x, y, a w + b z, w)`, or `(sx x, y, z, w)` where
/// only `x` is compensated. Where the depth is, the two clip distances `c.clip` names are stored
/// too: `z` and `w - z` where `clip` is 1, which clip at Metal's near and far planes as the Vulkan
/// viewport's no longer do, and 1 where it is 0 (depth clamp mode, which clips at neither). Under
/// the point guard band (`c.guard`) `x` and `y` are scaled by `PushField::PointGuard`'s `[gx, gy]`
/// too, and the four clip distances `c.guard` names are stored: `1.5 w - x`, `1.5 w + x`, `1.5 w -
/// y` and `1.5 w + y` of the function's own `x` and `y`, which cull a point where Metal does. Under
/// `c.mirrors` `x` is negated too where the `selected` viewport's bit of
/// `PushField::ViewportMirrors` is set. Under `c.vertex_clamp` the compensated depth is clamped to
/// `[lo w, hi w]` of `PushField::VertexDepthClamp`'s `[lo, hi]` where `w` is positive, as Metal
/// clamps each vertex's depth before the interpolation; a NaN depth takes the bound.
fn compensate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: Compensation,
    position: SValue,
    selected: Option<SValue>,
) -> BResult<SValue> {
    let field = cx.push_field(fb, PushField::ViewportCompensation)?;
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let lane = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
        let bits = fb.composite_extract(field, &[0, i])?;
        fb.convert(ConvOp::Bitcast, bits, f32_)
    };
    let x = fb.composite_extract(position, &[0])?;
    let y = fb.composite_extract(position, &[1])?;
    let z = fb.composite_extract(position, &[2])?;
    let w = fb.composite_extract(position, &[3])?;
    let sx = lane(fb, 0)?;
    let sxx = fb.binary(BinOp::FMul, sx, x)?;
    // Mirrored and unmirrored viewports in one draw: `x` negated where the selected viewport's
    // bit is set.
    let sxx = match selected.filter(|_| c.mirrors) {
        None => sxx,
        Some(index) => {
            let mirrors = cx.push_field(fb, PushField::ViewportMirrors)?;
            let bits = fb.binary(BinOp::ShiftRightLogical, mirrors, index)?;
            let one = fb.u32_const(1)?;
            let one = fb.module().const_value(one)?;
            let bit = fb.binary(BinOp::BitwiseAnd, bits, one)?;
            let zero = fb.u32_const(0)?;
            let zero = fb.module().const_value(zero)?;
            let mirrored = fb.binary(BinOp::INotEqual, bit, zero)?;
            let negated = fb.unary(UnOp::FNegate, sxx)?;
            fb.select(mirrored, negated, sxx)?
        }
    };
    let f32_const = |fb: &mut FunctionBuilder<'_>, v: f32| -> BResult<SValue> {
        let k = fb.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(v.to_bits())),
        })?;
        fb.module().const_value(k)
    };
    // Metal's guard band for points: the position scaled into the widened Vulkan viewports,
    // and the four planes at `1.5 w` of the function's own `x` and `y`.
    let (sxx, y_out) = match c.guard {
        None => (sxx, y),
        Some((object, first)) => {
            let guard = cx.push_field(fb, PushField::PointGuard)?;
            let scale = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
                let bits = fb.composite_extract(guard, &[i])?;
                fb.convert(ConvOp::Bitcast, bits, f32_)
            };
            let (gx, gy) = (scale(fb, 0)?, scale(fb, 1)?);
            let band = f32_const(fb, 1.5)?;
            let band = fb.binary(BinOp::FMul, band, w)?;
            let g = global(cx, object)?;
            let base = fb.module().global_value(g)?;
            let mut k = first;
            for v in [x, y] {
                for op in [BinOp::FSub, BinOp::FAdd] {
                    let d = fb.binary(op, band, v)?;
                    let kc = fb.u32_const(k)?;
                    let kc = fb.module().const_value(kc)?;
                    let p = fb.access_chain(base, &[kc])?;
                    fb.store(p, d, MemAccess::default())?;
                    k += 1;
                }
            }
            (
                fb.binary(BinOp::FMul, gx, sxx)?,
                fb.binary(BinOp::FMul, gy, y)?,
            )
        }
    };
    let depth = match c.clip {
        None => z,
        Some((object, first)) => {
            let (a, b, clip) = (lane(fb, 1)?, lane(fb, 2)?, lane(fb, 3)?);
            let aw = fb.binary(BinOp::FMul, a, w)?;
            let bz = fb.binary(BinOp::FMul, b, z)?;
            let depth = fb.binary(BinOp::FAdd, aw, bz)?;
            let (half, one) = (f32_const(fb, 0.5)?, f32_const(fb, 1.0)?);
            let clips = fb.binary(BinOp::FOrdGreaterThan, clip, half)?;
            let far = fb.binary(BinOp::FSub, w, z)?;
            let g = global(cx, object)?;
            let base = fb.module().global_value(g)?;
            for (k, distance) in [(first, z), (first + 1, far)] {
                let d = fb.select(clips, distance, one)?;
                let kc = fb.u32_const(k)?;
                let kc = fb.module().const_value(kc)?;
                let p = fb.access_chain(base, &[kc])?;
                fb.store(p, d, MemAccess::default())?;
            }
            // Metal clamps each vertex's depth before the interpolation: `z / w` to the draw's
            // `[lo, hi]` where `w` is positive.
            match c.vertex_clamp {
                false => depth,
                true => {
                    let range = cx.push_field(fb, PushField::VertexDepthClamp)?;
                    let bound = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
                        let bits = fb.composite_extract(range, &[i])?;
                        let v = fb.convert(ConvOp::Bitcast, bits, f32_)?;
                        fb.binary(BinOp::FMul, v, w)
                    };
                    let (low, high) = (bound(fb, 0)?, bound(fb, 1)?);
                    let raised = fb.glsl(GlslOp::NMax, &[depth, low])?;
                    let clamped = fb.glsl(GlslOp::NMin, &[raised, high])?;
                    let zero = f32_const(fb, 0.0)?;
                    let positive = fb.binary(BinOp::FOrdGreaterThan, w, zero)?;
                    fb.select(positive, clamped, depth)?
                }
            }
        }
    };
    let vec4 = fb.module().ty(SType::Vector { elem: f32_, len: 4 })?;
    fb.composite_construct(vec4, &[sxx, y_out, depth, w])
}

/// How a return drops a positive NaN depth, which fails Metal's depth test (`clamp_depth`).
#[derive(Clone, Copy, Debug)]
enum NanDrop {
    /// Kill the invocation there (`FragmentPlan::return_kill`).
    Kill(KillForm),
    /// Inside an interlocked body, record it in the `discarded` predicate: the storage form's
    /// color stores and the other writes at the return skip it, and the entry kills it after the
    /// critical section, so a color stored in the shader is dropped as Vulkan's tests would drop an
    /// output.
    Record(GlobalId),
}

/// Kill the fragment whose returned value `x` fails the pipeline's private alpha test `test`
/// (`LoweringInputs::alpha_test`), as Metal tests it after the function runs (probes on an
/// Apple M3 Max): color 0's alpha and the encoder's reference are each clamped to `[0, 1]`, a NaN
/// taken as 0, and converted to an 8-bit unorm code from the exact product with 255 rounded to
/// nearest, as Metal stores an 8-bit unorm channel, whatever the attachment's format; the codes
/// are compared, and a fragment whose comparison fails writes nothing and counts no visibility.
/// The reference's code is `PushField::AlphaTestReference`, which the binder converts
/// (`bind::alpha_test_code`); `Never` kills every fragment and reads no reference. Inside an
/// interlocked body the failure is recorded in the `discarded` predicate, as a NaN depth's is
/// ([`nan_drop`]).
fn alpha_test(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    test: AlphaTest,
    x: SValue,
    returned: Returned,
) -> BResult<()> {
    use crate::metal::CompareFunction as F;
    let Some(drop) = nan_drop(cx)? else {
        return Err(cx.fail("an alpha test in a function that loops over its covered samples"));
    };
    let fails = match test.function {
        F::Never => {
            let bool_ = fb.module().ty(SType::Bool)?;
            let k = fb.module().constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(1),
            })?;
            fb.module().const_value(k)?
        }
        function => {
            let color = match returned {
                Returned::Struct => fb.composite_extract(x, &[test.field])?,
                Returned::Value | Returned::Void => x,
            };
            let alpha = fb.composite_extract(color, &[3])?;
            let scalar = Form {
                held: Held::Float(FloatFmt::F32),
                lanes: 1,
            };
            let code = crate::lower::arith::color_write::unorm8_code(fb, scalar, alpha)?;
            let u32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let code = fb.convert(ConvOp::ConvertFToU, code, u32_)?;
            let reference = cx.push_field(fb, PushField::AlphaTestReference)?;
            let op = match function {
                F::Less => BinOp::UGreaterThanEqual,
                F::Equal => BinOp::INotEqual,
                F::LessEqual => BinOp::UGreaterThan,
                F::Greater => BinOp::ULessThanEqual,
                F::NotEqual => BinOp::IEqual,
                F::GreaterEqual => BinOp::ULessThan,
                _ => return Err(cx.fail("an alpha test of a comparison that tests nothing")),
            };
            fb.binary(op, code, reference)?
        }
    };
    match drop {
        NanDrop::Kill(form) => crate::lower::fragment::kill_if(fb, fails, form),
        NanDrop::Record(discarded) => {
            let p = fb.module().global_value(discarded)?;
            let was = fb.load(p, MemAccess::default())?;
            let now = fb.binary(BinOp::LogicalOr, was, fails)?;
            fb.store(p, now, MemAccess::default())
        }
    }
}

/// How the entry's returns drop a positive NaN depth, or `None` in a function looping over its
/// covered samples, where a kill would drop every sample's outputs.
fn nan_drop(cx: &mut EmitCx<'_>) -> BResult<Option<NanDrop>> {
    let fragment = &cx.plan.families.fragment;
    if let Some(form) = fragment.return_kill() {
        return Ok(Some(NanDrop::Kill(form)));
    }
    if fragment.interlock.is_none() {
        return Ok(None);
    }
    match cx.decl.families.fragment.discarded {
        Some(discarded) => Ok(Some(NanDrop::Record(discarded))),
        None => Err(cx.fail("an interlocked depth output without its discard predicate")),
    }
}

/// `depth`, a fragment function's depth output, as Metal stores it through any viewport (Metal
/// probes, an M3 Max and an M4 alike): a value with the sign bit set (a negative value, -0, a
/// negative denormal, -inf, a negative NaN) as +0, one past 1 up to +inf as 1, and any other (`[0,
/// 1]`'s values, a positive denormal kept) unchanged. Compared by its bits, so a device that
/// flushes denormals keeps a positive one and makes -0 and a negative one +0 alike. A positive NaN
/// fails Metal's depth test, `Always` included: neither depth nor color is written and no sample
/// counts; the invocation is dropped by `kill` ([`nan_drop`]), which drops the same, though Metal
/// also runs the stencil's depth-fail operation. The drop is capability on every profile: no device
/// has a defined form, since Vulkan leaves a NaN depth undefined after its clamp and range step,
/// with or without depth clamp ("Depth Clamping and Range Adjustment": "If zf is not in the range
/// [zmin, zmax], then zf is undefined following this step", and a NaN lies in no range; only
/// `depthClampZeroOne`, which no profile enables, keeps a float depth unchanged), and a device
/// that clamps the depth itself stores a NaN as anything (ANV: 0, its color written). In a
/// function looping over its covered samples the value passes unchanged. The selects run only where the plan `clamps`,
/// the device storing the value unclamped; elsewhere its own clamp gives Metal's answers.
///
/// Under `range`, `PushField::DepthClampRange`'s `[low, high]`, the value is then clamped to the
/// draw's depth clamp range, as Metal's clamp mode stores it, but a negative NaN stores 0 (a Metal
/// probe, an M3 Max and an M4 alike: 0 through a range of `[0.25, 0.5]`), which the draw's Vulkan
/// clamp, holding 0, keeps.
fn clamp_depth(
    fb: &mut FunctionBuilder<'_>,
    depth: SValue,
    kill: Option<NanDrop>,
    clamps: bool,
    range: Option<SValue>,
) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let u = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let k = fb.u32_const(v)?;
        fb.module().const_value(k)
    };
    let f = |fb: &mut FunctionBuilder<'_>, v: f32| -> BResult<SValue> {
        let k = fb.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(v.to_bits())),
        })?;
        fb.module().const_value(k)
    };
    let bits = fb.convert(ConvOp::Bitcast, depth, u32_)?;
    let sign = u(fb, 0x8000_0000)?;
    let one_bits = u(fb, 1.0f32.to_bits())?;
    let inf_bits = u(fb, f32::INFINITY.to_bits())?;
    if let Some(drop) = kill {
        let past_inf = fb.binary(BinOp::UGreaterThan, bits, inf_bits)?;
        let positive = fb.binary(BinOp::ULessThan, bits, sign)?;
        let nan = fb.binary(BinOp::LogicalAnd, past_inf, positive)?;
        match drop {
            NanDrop::Kill(form) => crate::lower::fragment::kill_if(fb, nan, form)?,
            NanDrop::Record(discarded) => {
                let p = fb.module().global_value(discarded)?;
                let was = fb.load(p, MemAccess::default())?;
                let now = fb.binary(BinOp::LogicalOr, was, nan)?;
                fb.store(p, now, MemAccess::default())?;
            }
        }
    }
    if !clamps {
        return Ok(depth);
    }
    let negative = fb.binary(BinOp::UGreaterThanEqual, bits, sign)?;
    let past_one = fb.binary(BinOp::UGreaterThan, bits, one_bits)?;
    let at_most_inf = fb.binary(BinOp::ULessThanEqual, bits, inf_bits)?;
    let over = fb.binary(BinOp::LogicalAnd, past_one, at_most_inf)?;
    let zero = f(fb, 0.0)?;
    let one = f(fb, 1.0)?;
    let high = fb.select(over, one, depth)?;
    let clamped = fb.select(negative, zero, high)?;
    let Some(range) = range else {
        return Ok(clamped);
    };
    let bound = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<SValue> {
        let bits = fb.composite_extract(range, &[i])?;
        fb.convert(ConvOp::Bitcast, bits, f32_)
    };
    let (low, high) = (bound(fb, 0)?, bound(fb, 1)?);
    let below = fb.binary(BinOp::FOrdLessThan, clamped, low)?;
    let raised = fb.select(below, low, clamped)?;
    let above = fb.binary(BinOp::FOrdGreaterThan, raised, high)?;
    let ranged = fb.select(above, high, raised)?;
    // A negative NaN's bits lie above -inf's.
    let negative_inf = u(fb, f32::NEG_INFINITY.to_bits())?;
    let negative_nan = fb.binary(BinOp::UGreaterThan, bits, negative_inf)?;
    fb.select(negative_nan, zero, ranged)
}

/// A vertex function's index `index` (an integer held zero-extended in its 32-bit container)
/// where it is below the `u32` count push field `count`, and 0 where it is not:
///
/// - the viewport index below the draw's viewport count (`PushField::ViewportCount`), as Metal
///   draws an index at or past it through viewport 0. A count of 0, which the binder writes for an
///   encoder whose viewports were never set, keeps every index 0, as Metal's one default viewport
///   does;
/// - the render target slice index below the pass's `renderTargetArrayLength`
///   (`PushField::RenderTargetArrayLength`), as Metal renders the pass's first layer for an index
///   at or past it. Only a pass of at least two layers runs the variant that reads it.
fn below_count(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: SValue,
    count: PushField,
) -> BResult<SValue> {
    let count = cx.push_field(fb, count)?;
    let below = fb.binary(BinOp::ULessThan, index, count)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    fb.select(below, index, zero)
}

/// The element type of a user variable of class `numeric`: 32-bit, but a 16-bit `float` where
/// `half` (`IoObject::User::half`).
fn io_elem(mb: &mut ModuleBuilder, numeric: IoNumeric, half: bool) -> BResult<TypeId> {
    match numeric {
        IoNumeric::Float if half => mb.ty(SType::Float { width: 16 }),
        IoNumeric::Float => mb.ty(SType::Float { width: 32 }),
        IoNumeric::Uint => mb.ty(SType::Int {
            width: 32,
            signed: false,
        }),
        IoNumeric::Sint => mb.ty(SType::Int {
            width: 32,
            signed: true,
        }),
    }
}

/// Store output part `x` (in its lowered representation, which is the interface type, except a
/// `bool`, carried as `uint` 1 or 0, a 32-bit integer declared `int`, whose bits are stored by
/// `OpBitcast`, and a `half` declared 16-bit, stored by `OpFConvert` of the `f32` holding it,
/// whose value is a `half`, so the conversion is exact) through `link`.
fn store(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, link: &Link, x: SValue) -> BResult<()> {
    let x = if let Some(IoObject::User {
        numeric: IoNumeric::Float,
        components,
        half: true,
        ..
    }) = cx.plan.families.stage_io.objects.get(link.object)
    {
        let components = *components;
        let elem = io_elem(fb.module(), IoNumeric::Float, true)?;
        let ty = if components == 1 {
            elem
        } else {
            fb.module().ty(SType::Vector {
                elem,
                len: components,
            })?
        };
        fb.convert(ConvOp::FConvert, x, ty)?
    } else if bool_as_uint(cx, link) {
        let one = ops::splat(fb, uint_lanes(link.form), 1)?;
        let zero = ops::splat(fb, uint_lanes(link.form), 0)?;
        fb.select(x, one, zero)?
    } else if let Some(IoObject::User {
        numeric: IoNumeric::Sint,
        components,
        ..
    }) = cx.plan.families.stage_io.objects.get(link.object)
    {
        let components = *components;
        let elem = io_elem(fb.module(), IoNumeric::Sint, false)?;
        let ty = if components == 1 {
            elem
        } else {
            fb.module().ty(SType::Vector {
                elem,
                len: components,
            })?
        };
        fb.convert(ConvOp::Bitcast, x, ty)?
    } else {
        x
    };
    match link.place {
        Place::Weights { .. } => Err(cx.fail("an output stored through the barycentric weights")),
        Place::Whole | Place::Element0 => {
            let p = pointer(cx, fb, link)?;
            fb.store(p, x, MemAccess::default())
        }
        Place::Spread(n) => {
            let g = global(cx, link.object)?;
            let base = fb.module().global_value(g)?;
            for k in 0..n {
                let e = fb.composite_extract(x, &[k])?;
                let kc = fb.u32_const(k)?;
                let kc = fb.module().const_value(kc)?;
                let p = fb.access_chain(base, &[kc])?;
                fb.store(p, e, MemAccess::default())?;
            }
            Ok(())
        }
    }
}

/// Declare the stage interface's variables and add its entry execution modes to `modes`.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let sp = &plan.families.stage_io;
    let mut d = Declared::default();
    // The sample-qualified varyings a looping entry reads at every sample before the loop, and
    // how many samples.
    let before_loop = plan
        .families
        .fragment
        .coverage
        .and_then(|c| c.samples_before_loop);
    let read_before_loop: BTreeSet<usize> = sp
        .params
        .iter()
        .filter_map(|p| match p.source {
            Source::AtLoopSample(link) if before_loop.is_some() => Some(link.object),
            _ => None,
        })
        .collect();
    for (i, o) in sp.objects.iter().enumerate() {
        let g = match o {
            IoObject::User {
                key,
                dir,
                numeric,
                components,
                deco,
                half,
            } => {
                let elem = io_elem(mb, *numeric, *half)?;
                let ty = if *components == 1 {
                    elem
                } else {
                    mb.ty(SType::Vector {
                        elem,
                        len: *components,
                    })?
                };
                if let (Some(n), true) = (before_loop, read_before_loop.contains(&i)) {
                    let u32_ = mb.ty(SType::Int {
                        width: 32,
                        signed: false,
                    })?;
                    let len = mb.constant(Constant::Scalar {
                        ty: u32_,
                        bits: ConstBits(u64::from(n)),
                    })?;
                    let array = mb.ty(SType::Array {
                        elem: ty,
                        len,
                        stride: None,
                    })?;
                    d.at_samples
                        .insert(i, mb.plain_global(StorageClass::Private, array, None)?);
                }
                let g = mb.io(key.clone(), *dir, ty, *deco)?;
                // An input of a key the mesh function writes per primitive, flat alone behind an
                // emulated mesh function.
                if *dir == IoDir::Input
                    && !sp.flat_primitive_inputs
                    && sp.interface.primitive_inputs.binary_search(key).is_ok()
                {
                    mb.per_primitive(g)?;
                }
                g
            }
            IoObject::Builtin {
                builtin,
                dir,
                shape,
                enabler,
            } => {
                let ty = shape.spirv_type(mb)?;
                let g = mb.builtin(*builtin, *dir, ty)?;
                if let Some(cap) = enabler {
                    mb.enable_builtin(g, *cap)?;
                }
                g
            }
        };
        d.globals.push(g);
    }
    if let Some(g) = sp.invariant.and_then(|i| d.globals.get(i)) {
        mb.invariant(*g)?;
    }
    if sp.invariant_fragment_outputs {
        use crate::spv::BuiltIn as B;
        for (o, g) in sp.objects.iter().zip(&d.globals) {
            if let IoObject::Builtin {
                builtin: B::FragDepth | B::SampleMask | B::FragStencilRefEXT,
                dir: IoDir::Output,
                ..
            } = o
            {
                mb.invariant(*g)?;
            }
        }
    }
    modes.extend(sp.modes.iter().copied());
    Ok(d)
}
