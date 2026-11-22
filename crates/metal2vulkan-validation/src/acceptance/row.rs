//! The pipeline inputs of an L4 row, derived from the entry's AIR.
//!
//! A corpus row is a function with no authored case, so nothing states the vertex descriptor or
//! the attachments Metal would draw it with. The sweep builds the ones the function's own
//! interface implies, from the AIR input layer ([`AirInterface`], the facts the checker compares
//! a manifest against), never from a name:
//!
//! - a **vertex** function: one per-vertex buffer layout for each `[[attribute(n)]]` it declares,
//!   at offset 0 of its own Metal buffer, each at the slot the executors' rule gives a
//!   vertex-fetched input ([`input_slots`]: the highest slot no AIR argument occupies, in
//!   location order), of the Metal vertex format that reads exactly the attribute's IR type
//!   (32-bit floats, halves, and 32/16/8-bit integers as unsigned, the class the product declares
//!   every integer input as, of 1 to 4 lanes), with a stride of that size rounded up to 4 bytes;
//!   rasterization is disabled, so no attachment is needed. A **post-tessellation** vertex
//!   function's pipeline is a tessellation pipeline in Metal's default tessellation state: each
//!   patch attribute (`air.patch_input`) it declares is laid out so from a per-patch layout, and
//!   each control-point field it reads from a per-control-point layout, a field's IR type being its
//!   control-point function's return member (a field Metal marks unused, or whose function-constant
//!   predicate is off, is not laid out);
//! - a **kernel** function reading `[[stage_in]]` attributes: the stage-input descriptor its
//!   interface implies, each attribute its own buffer at the executors' slot for it
//!   (`executor_contract::stage_inputs::implied`), derived only when the kernel asks for one;
//! - a **fragment** function: drawn with the generated passthrough vertex companion the executors
//!   draw fragment cases with ([`crate::companion::passthrough_vertex`]), into one color attachment
//!   per render target it writes or reads, of its value's class (`Float` or `Uint`) with as many
//!   32-bit channels as the fewest components an output to it writes (`R32`, `RG32`, or `RGBA32`;
//!   `RG32` for three, since no 32-bit color format has three channels; `RGBA32` for a target it
//!   only reads), every one color-renderable on every Vulkan device: Metal refuses a pipeline whose
//!   color output has fewer components than its attachment's format, and so does the product, so a
//!   wider format would make the row's own descriptor invalid. Then a `Depth32Float` attachment
//!   when a live output writes depth and a `Stencil8` one when one writes stencil
//!   (`Depth32Float_Stencil8` when it writes both;
//!   [`crate::executor_contract::depth_stencil_test`]), as a triangle list, under Metal's default
//!   depth-stencil state.
//!
//! What the tool cannot build is an [`Unmodelled`] of a typed [`UnmodelledKind`] with a detail
//! naming it (an attribute or target type no format reads, a varying no companion writes, a
//! location that is not a constant, a control-point function that returns no struct of its
//! fields): the row is `UNMODELLED`, never compiled against a guessed input, and a sweep counts
//! it by kind.

use crate::candidate::engine::graphics::RenderTemplate;
use crate::check::interface::{bound_constants, constant_int, AirInterface, NumShape};
use crate::check::loop_free::DecodedModule;
use crate::companion::{Companion, PASSTHROUGH_VERTEX_ENTRY};
use crate::executor_contract::buffer_slots::{input_slots, NativeBufferSlots};
use crate::executor_contract::depth_stencil_test;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, VertexFormat, VertexStepFunction};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_air::abi::{OutputRole, ParamRole, PatchInput, Slot};
use metal2vulkan_air::ir::{FuncId, Module, Type, TypeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The color attachments Metal's descriptor has.
const COLOR_ATTACHMENTS: u32 = 8;

/// What kind of pipeline input the tool does not build for an L4 row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnmodelledKind {
    /// The function's AIR-layer interface under the row's function-constant values: its AIR does
    /// not decode as the checker reads it, or a slot, an attachment, or metadata needs a value of a
    /// constant the row leaves undefined.
    Interface,
    /// A vertex attribute: a location that is not a constant or is declared twice, or an IR type
    /// no Metal vertex format reads exactly.
    VertexAttribute,
    /// The buffer slots of the vertex-fetched attributes.
    BufferSlots,
    /// A color attachment: an index that is not a constant below 8, an IR type no color format
    /// holds, or one read and written as different classes.
    Attachment,
    /// The generated passthrough vertex companion (a varying it does not write).
    Companion,
    /// A kernel stage input: a location that is not a constant, or an IR type no format reads
    /// exactly (`executor_contract::stage_inputs::implied`).
    StageInput,
    /// An object or mesh function: the derivation builds no mesh render pipeline around it yet.
    MeshPipeline,
}

impl UnmodelledKind {
    /// The kind's stable key, as a sweep reports and counts it.
    pub fn key(self) -> &'static str {
        match self {
            UnmodelledKind::Interface => "interface",
            UnmodelledKind::VertexAttribute => "vertex_attribute",
            UnmodelledKind::BufferSlots => "buffer_slots",
            UnmodelledKind::Attachment => "attachment",
            UnmodelledKind::Companion => "companion",
            UnmodelledKind::StageInput => "stage_input",
            UnmodelledKind::MeshPipeline => "mesh_pipeline",
        }
    }
}

/// An L4 row's pipeline input the tool does not build: its kind, and a detail naming it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unmodelled {
    pub kind: UnmodelledKind,
    pub detail: String,
}

impl Unmodelled {
    fn of(kind: UnmodelledKind) -> impl Fn(String) -> Unmodelled {
        move |detail| Unmodelled { kind, detail }
    }
}

/// The unmodelled pipeline of the object or mesh function `entry`.
pub(crate) fn mesh_pipeline(entry: &str) -> Unmodelled {
    Unmodelled {
        kind: UnmodelledKind::MeshPipeline,
        detail: format!("{entry} is an object or mesh function; no mesh render pipeline is built"),
    }
}

impl std::fmt::Display for Unmodelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind.key(), self.detail)
    }
}

/// A vertex row's inputs: its template and the Metal buffer indices its descriptor lays out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexRow {
    pub template: RenderTemplate,
    pub buffers: BTreeSet<u32>,
}

/// A fragment row's inputs: its template and its generated passthrough vertex companion, whose
/// entry is `vertex_entry`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentRow {
    pub template: RenderTemplate,
    pub companion: Companion,
    pub vertex_entry: &'static str,
}

/// The decoded module of `bitcode` and the interface of `entry` under the function-constant
/// values `constants` the row compiles with (none: Metal's "undefined", the plain census's).
fn interface(
    bitcode: &[u8],
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<(DecodedModule, AirInterface), Unmodelled> {
    let unmodelled = Unmodelled::of(UnmodelledKind::Interface);
    let decoded = DecodedModule::from_bitcode(entry, bitcode)
        .map_err(|evidence| unmodelled(format!("the AIR does not decode: {evidence}")))?;
    let abi = decoded
        .abi()
        .map_err(|error| unmodelled(format!("the AIR metadata ABI does not decode: {error}")))?;
    let interface = AirInterface::new(decoded.module(), abi, entry, &bound_constants(constants))
        .map_err(unmodelled)?;
    Ok((decoded, interface))
}

/// The inputs of vertex function `entry` of `bitcode` under the function-constant values
/// `constants` the row compiles with.
pub fn vertex_row(
    bitcode: &[u8],
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<VertexRow, Unmodelled> {
    let (decoded, interface) = interface(bitcode, entry, constants)?;
    vertex_row_of(decoded.module(), &interface)
}

/// One attribute a vertex row lays out: its location, its IR shape, and its layout's step.
struct RowAttribute {
    location: u32,
    shape: Option<NumShape>,
    step: VertexStepFunction,
}

/// The inputs of the vertex function whose AIR-layer interface is `interface` in `module` (the
/// module docs): its `[[attribute(n)]]`s per vertex, or a post-tessellation function's patch
/// attributes per patch and read control-point fields per control point.
pub fn vertex_row_of(module: &Module, interface: &AirInterface) -> Result<VertexRow, Unmodelled> {
    let attribute = Unmodelled::of(UnmodelledKind::VertexAttribute);
    let constant = |slot: &Slot, param: u32| {
        constant_int(slot.index).ok_or_else(|| {
            attribute(format!(
                "vertex attribute at parameter {param} has no constant location"
            ))
        })
    };
    let mut read = Vec::new();
    for param in &interface.params {
        let at = param.binding.param;
        match &param.binding.role {
            ParamRole::VertexInput { location } => read.push(RowAttribute {
                location: constant(location, at)?,
                shape: param.value,
                step: VertexStepFunction::PerVertex,
            }),
            ParamRole::PatchInput(PatchInput::Patch { location }) => read.push(RowAttribute {
                location: constant(location, at)?,
                shape: param.value,
                step: VertexStepFunction::PerPatch,
            }),
            ParamRole::PatchInput(PatchInput::ControlPoints { function, fields }) => {
                let members = control_point_members(module, *function).ok_or_else(|| {
                    attribute(format!(
                        "the control-point function of parameter {at} returns no struct of its \
                         fields"
                    ))
                })?;
                if members.len() != fields.len() {
                    return Err(attribute(format!(
                        "the control-point function of parameter {at} returns {} members for {} \
                         fields",
                        members.len(),
                        fields.len()
                    )));
                }
                for (field, member) in fields.iter().zip(members) {
                    if field.unused || field.gated_off {
                        continue;
                    }
                    read.push(RowAttribute {
                        location: constant(&field.location, at)?,
                        shape: NumShape::of(module, member),
                        step: VertexStepFunction::PerPatchControlPoint,
                    });
                }
            }
            _ => {}
        }
    }
    let mut attributes = BTreeMap::new();
    for a in read {
        let location = a.location;
        let format = a.shape.and_then(vertex_format).ok_or_else(|| {
            attribute(format!(
                "vertex attribute {location} has IR type {:?}, which no Metal vertex format reads \
                 exactly",
                a.shape
            ))
        })?;
        if attributes.insert(location, (format, a.step)).is_some() {
            return Err(attribute(format!(
                "vertex attribute {location} is declared twice"
            )));
        }
    }
    let mut template = RenderTemplate::new(PrimitiveTopologyClass::Unspecified);
    let mut buffers = BTreeSet::new();
    if !attributes.is_empty() {
        let slots = NativeBufferSlots::from_interface(interface)
            .and_then(|slots| input_slots(slots, attributes.keys().copied()))
            .map_err(Unmodelled::of(UnmodelledKind::BufferSlots))?;
        let mut descriptor = VertexDescriptor::new();
        for (&location, &((format, bytes), step)) in &attributes {
            let buffer = slots[&location];
            let mut layout = VertexBufferLayoutDescriptor::new(Some(bytes.next_multiple_of(4)));
            layout.step_function = step;
            descriptor.layouts.insert(buffer, layout);
            descriptor
                .attributes
                .insert(location, VertexAttributeDescriptor::new(format, 0, buffer));
            buffers.insert(buffer);
        }
        template.vertex_descriptor = Some(descriptor);
    }
    Ok(VertexRow { template, buffers })
}

/// The IR types of the members of the struct control-point function `function` of `module`
/// returns: its fields', in field order.
fn control_point_members(module: &Module, function: FuncId) -> Option<Vec<TypeId>> {
    let types = module.types();
    let Type::Function { ret, .. } = types.get(module.func(function)?.ty)? else {
        return None;
    };
    match types.get(*ret)? {
        Type::Struct { fields, .. } => Some(fields.to_vec()),
        _ => None,
    }
}

/// The Metal vertex format that reads exactly a value of IR shape `shape`, and its size in bytes.
fn vertex_format(shape: NumShape) -> Option<(VertexFormat, u32)> {
    use VertexFormat as V;
    let row: [V; 4] = match (shape.float, shape.bits) {
        (true, 32) => [V::Float, V::Float2, V::Float3, V::Float4],
        (true, 16) => [V::Half, V::Half2, V::Half3, V::Half4],
        (false, 32) => [V::UInt, V::UInt2, V::UInt3, V::UInt4],
        (false, 16) => [V::UShort, V::UShort2, V::UShort3, V::UShort4],
        (false, 8) => [V::UChar, V::UChar2, V::UChar3, V::UChar4],
        _ => return None,
    };
    let lane = usize::try_from(shape.lanes).ok()?.checked_sub(1)?;
    Some((*row.get(lane)?, shape.bits / 8 * shape.lanes))
}

/// The inputs of fragment function `entry` of `bitcode` under the function-constant values
/// `constants` the row compiles with: its attachments are those of the outputs the values leave.
/// The companion writes every varying the function declares, gated or not. The caller assembles
/// the companion (`Companion::opaque_bitcode`, in the worker, under its deadline); a failure there
/// is the host's tooling, not an unmodelled input.
pub fn fragment_row(
    bitcode: &[u8],
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<FragmentRow, Unmodelled> {
    let (decoded, interface) = interface(bitcode, entry, constants)?;
    let abi = decoded.abi().map_err(|error| Unmodelled {
        kind: UnmodelledKind::Interface,
        detail: format!("the AIR metadata ABI does not decode: {error}"),
    })?;
    let passthrough = crate::companion::passthrough_vertex(decoded.module(), abi, entry)
        .map_err(Unmodelled::of(UnmodelledKind::Companion))?;
    let template = fragment_template(&interface)?;
    Ok(FragmentRow {
        template,
        companion: Companion::PassthroughVertex(passthrough),
        vertex_entry: PASSTHROUGH_VERTEX_ENTRY,
    })
}

/// One color attachment a fragment row draws into: whether its value is a float, and the fewest
/// components an output to it writes (`None` for a target the function only reads).
#[derive(Clone, Copy)]
struct ColorTarget {
    float: bool,
    written: Option<u32>,
}

impl ColorTarget {
    /// The attachment's format: of its class, with as many 32-bit channels as the fewest
    /// components written to it, which no output may have fewer of. Every one is color-renderable
    /// on every Vulkan device; none has three channels, so three take two.
    fn format(self) -> PixelFormat {
        use PixelFormat as P;
        match (self.float, self.written) {
            (true, Some(1)) => P::R32Float,
            (true, Some(2 | 3)) => P::RG32Float,
            (true, _) => P::RGBA32Float,
            (false, Some(1)) => P::R32Uint,
            (false, Some(2 | 3)) => P::RG32Uint,
            (false, _) => P::RGBA32Uint,
        }
    }
}

/// The template of the fragment function whose AIR-layer interface is `interface`.
pub fn fragment_template(interface: &AirInterface) -> Result<RenderTemplate, Unmodelled> {
    let unmodelled = Unmodelled::of(UnmodelledKind::Attachment);
    let mut colors: BTreeMap<u32, ColorTarget> = BTreeMap::new();
    // `written`: whether `shape` is a value written to the target (an output) or read from it.
    let mut color = |what: &str, index: Option<u32>, shape: Option<NumShape>, written: bool| {
        let index = index
            .filter(|&index| index < COLOR_ATTACHMENTS)
            .ok_or_else(|| format!("a {what} whose attachment is not a constant below 8"))?;
        let Some(shape) = shape else {
            return Err(format!(
                "{what} {index} has an IR type no color format holds"
            ));
        };
        let lanes = written.then_some(shape.lanes);
        let target = colors.entry(index).or_insert(ColorTarget {
            float: shape.float,
            written: lanes,
        });
        if target.float != shape.float {
            return Err(format!(
                "attachment {index} is read and written as different classes"
            ));
        }
        target.written = match (target.written, lanes) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Ok(())
    };
    for output in &interface.outputs {
        if let OutputRole::RenderTarget { attachment, .. } = &output.binding.role {
            color(
                "render target",
                constant_int(*attachment),
                output.value,
                true,
            )
            .map_err(&unmodelled)?
        }
    }
    for param in &interface.params {
        if let ParamRole::ColorInput { index, .. } = &param.binding.role {
            color(
                "framebuffer-fetch input",
                constant_int(*index),
                param.value,
                false,
            )
            .map_err(&unmodelled)?;
        }
    }
    let mut template = RenderTemplate::new(PrimitiveTopologyClass::Triangle);
    template.color_attachments = colors
        .into_iter()
        .map(|(index, target)| (index, ColorAttachmentDescriptor::new(target.format())))
        .collect();
    // An aspect for each test the function's live outputs set: the depth-stencil state's own
    // derivation, so an output its (unsupplied) constants gate off gets none.
    let test = depth_stencil_test(interface);
    (template.depth_format, template.stencil_format) = match (test.depth.is_some(), test.stencil) {
        (true, true) => (
            Some(PixelFormat::Depth32FloatStencil8),
            Some(PixelFormat::Depth32FloatStencil8),
        ),
        (true, false) => (Some(PixelFormat::Depth32Float), None),
        (false, true) => (None, Some(PixelFormat::Stencil8)),
        (false, false) => (None, None),
    };
    Ok(template)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::interface::tests::of_text;

    /// The IR shapes the formats read exactly, and the ones none does.
    #[test]
    fn a_vertex_format_reads_exactly_the_attributes_ir_type() {
        let shape = |float, bits, lanes| NumShape { float, bits, lanes };
        assert_eq!(
            vertex_format(shape(true, 32, 3)),
            Some((VertexFormat::Float3, 12))
        );
        assert_eq!(
            vertex_format(shape(true, 16, 3)),
            Some((VertexFormat::Half3, 6))
        );
        assert_eq!(
            vertex_format(shape(false, 8, 1)),
            Some((VertexFormat::UChar, 1))
        );
        assert_eq!(
            vertex_format(shape(false, 16, 4)),
            Some((VertexFormat::UShort4, 8))
        );
        for unread in [
            shape(false, 1, 1),
            shape(false, 64, 1),
            shape(true, 64, 1),
            shape(true, 32, 5),
        ] {
            assert_eq!(vertex_format(unread), None, "{unread:?}");
        }
    }

    /// The interface of `entry` in public fixture `file`.
    fn fixture(file: &str, entry: &str) -> Option<AirInterface> {
        let path = metal2vulkan_devtools::fixtures_dir().join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        match of_text(&text, entry, &BTreeMap::new()) {
            Ok(interface) => Some(interface),
            Err(error) if error.contains("llvm-as is unavailable") => {
                eprintln!("SKIPPED: {error}");
                None
            }
            Err(error) => panic!("{file}: {error}"),
        }
    }

    /// The row of `entry` in public fixture `file`.
    fn row(file: &str, entry: &str) -> Option<Result<VertexRow, Unmodelled>> {
        let path = metal2vulkan_devtools::fixtures_dir().join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        match crate::check::loop_free::assemble(&text) {
            Ok(bitcode) => Some(vertex_row(
                &bitcode,
                entry,
                &FunctionConstantValues::defaults(),
            )),
            Err(error) if error.contains("llvm-as is unavailable") => {
                eprintln!("SKIPPED: {error}");
                None
            }
            Err(error) => panic!("{file}: {error}"),
        }
    }

    /// A post-tessellation vertex function's row lays out its patch attribute per patch and the
    /// control-point fields its function reads per control point, typed by the control-point
    /// function's return members; the field Metal marks unused is not laid out.
    #[test]
    fn a_post_tessellation_vertex_row_lays_out_its_patch_inputs() {
        let Some(triangle) = row("engine/tessellation.ll", "tess_triangle_points") else {
            return;
        };
        let triangle = triangle.unwrap();
        let descriptor = triangle.template.vertex_descriptor.as_ref().unwrap();
        let attribute = |location: u32| {
            let a = &descriptor.attributes[&location];
            let layout = &descriptor.layouts[&a.buffer_index];
            (a.format, layout.step_function, layout.stride)
        };
        assert_eq!(
            descriptor.attributes.keys().copied().collect::<Vec<_>>(),
            [0, 1, 5]
        );
        assert_eq!(
            attribute(0),
            (
                VertexFormat::Float4,
                VertexStepFunction::PerPatchControlPoint,
                Some(16)
            )
        );
        assert_eq!(
            attribute(1),
            (
                VertexFormat::Half2,
                VertexStepFunction::PerPatchControlPoint,
                Some(4)
            )
        );
        assert_eq!(
            attribute(5),
            (VertexFormat::Float4, VertexStepFunction::PerPatch, Some(16))
        );
        assert_eq!(triangle.buffers.len(), 3);
        // A function that reads its control points from a buffer lays out nothing.
        let quad = row("engine/tessellation.ll", "tess_quad_buffer").unwrap();
        assert_eq!(quad.unwrap().template.vertex_descriptor, None);
    }

    /// Each attribute gets its own buffer at the highest free slot in location order (slot 0 is
    /// the function's own buffer), the format of its IR type, and a stride of its size rounded up
    /// to 4 bytes.
    #[test]
    fn a_vertex_rows_descriptor_lays_out_every_declared_attribute() {
        let Some(narrow) = row("vertex_narrow_attributes.ll", "vertex_narrow_attributes") else {
            return;
        };
        let row = narrow.unwrap();
        assert_eq!(row.buffers, BTreeSet::from([29, 30]));
        let descriptor = row.template.vertex_descriptor.as_ref().unwrap();
        let attribute = |location: u32| {
            let a = &descriptor.attributes[&location];
            let layout = &descriptor.layouts[&a.buffer_index];
            (a.format, a.offset, a.buffer_index, layout.stride)
        };
        assert_eq!(attribute(0), (VertexFormat::UChar, 0, 30, Some(4)));
        assert_eq!(attribute(1), (VertexFormat::UShort2, 0, 29, Some(4)));
        assert!(row.template.color_attachments.is_empty());
    }

    /// A color attachment for each target written, of its value's class; a depth and stencil
    /// output share one combined attachment format.
    #[test]
    fn a_fragment_rows_attachments_follow_its_outputs() {
        let Some(interface) = fixture("fragment_varying_color.ll", "fragment_varying_color") else {
            return;
        };
        let template = fragment_template(&interface).unwrap();
        let formats: Vec<(u32, PixelFormat)> = template
            .color_attachments
            .iter()
            .map(|(&index, attachment)| (index, attachment.pixel_format))
            .collect();
        assert_eq!(formats, [(0, PixelFormat::RGBA32Float)]);
        assert_eq!(
            (template.depth_format, template.stencil_format),
            (None, None)
        );
        assert_eq!(
            template.input_primitive_topology,
            PrimitiveTopologyClass::Triangle
        );

        let Some(interface) = fixture("fragment_depth_stencil.ll", "fragment_depth_stencil") else {
            return;
        };
        let template = fragment_template(&interface).unwrap();
        assert!(template.color_attachments.is_empty());
        assert_eq!(
            (template.depth_format, template.stencil_format),
            (
                Some(PixelFormat::Depth32FloatStencil8),
                Some(PixelFormat::Depth32FloatStencil8)
            )
        );
        // A row compiles with no constants unless it is given values, so a depth output a
        // constant gates is dropped and gets no aspect; the stencil output beside it keeps its
        // own (`acceptance`'s `a_row_with_constants_derives_its_inputs_and_compiles_under_them`
        // gives the constant).
        let gated = crate::executor_contract::depth_stencil::tests::depth_stencil_module(
            Some("less"),
            true,
            true,
        );
        let interface = of_text(&gated, "frag", &BTreeMap::new()).unwrap();
        let template = fragment_template(&interface).unwrap();
        assert_eq!(
            (template.depth_format, template.stencil_format),
            (None, Some(PixelFormat::Stencil8))
        );
    }

    /// The formats of `entry`'s color attachments in `ll` (a fixture's text), by index.
    fn color_formats(ll: &str, entry: &str) -> Option<Vec<(u32, PixelFormat)>> {
        let interface = match of_text(ll, entry, &BTreeMap::new()) {
            Ok(interface) => interface,
            Err(error) if error.contains("llvm-as is unavailable") => {
                eprintln!("SKIPPED: {error}");
                return None;
            }
            Err(error) => panic!("{entry}: {error}"),
        };
        let template = fragment_template(&interface).unwrap();
        Some(
            template
                .color_attachments
                .iter()
                .map(|(&index, attachment)| (index, attachment.pixel_format))
                .collect(),
        )
    }

    /// An attachment has as many channels as the fewest components an output to it writes, never
    /// more (Metal refuses a narrower output): one for a scalar, two for two or three components
    /// (no 32-bit color format has three), four for four or a target the function only reads, each
    /// of its value's class.
    #[test]
    fn a_fragment_rows_attachments_have_no_more_channels_than_its_outputs_write() {
        const NARROW: &str = r#"
define <{ <3 x half>, float, <2 x float>, <3 x i32>, <4 x half> }> @narrow_targets(<4 x float> %position) {
  ret <{ <3 x half>, float, <2 x float>, <3 x i32>, <4 x half> }> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @narrow_targets, !1, !7}
!1 = !{!2, !3, !4, !5, !6}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half3", !"air.arg_name", !"rgb"}
!3 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!4 = !{!"air.render_target", i32 2, i32 0, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!5 = !{!"air.render_target", i32 3, i32 0, !"air.arg_type_name", !"uint3", !"air.arg_name", !"ids"}
!6 = !{!"air.render_target", i32 4, i32 0, !"air.arg_type_name", !"half4", !"air.arg_name", !"tint"}
!7 = !{!8}
!8 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
"#;
        let Some(formats) = color_formats(NARROW, "narrow_targets") else {
            return;
        };
        assert_eq!(
            formats,
            [
                (0, PixelFormat::RG32Float),
                (1, PixelFormat::R32Float),
                (2, PixelFormat::RG32Float),
                (3, PixelFormat::RG32Uint),
                (4, PixelFormat::RGBA32Float),
            ]
        );
        // A target read and written takes the written components; one only read keeps four.
        let fetch = |file: &str, entry| {
            let path = metal2vulkan_devtools::engine_dir().join(file);
            color_formats(&std::fs::read_to_string(path).unwrap(), entry)
        };
        let Some(scalar) = fetch("fragment_fetch.ll", "fetch_scalar") else {
            return;
        };
        assert_eq!(scalar, [(0, PixelFormat::R32Float)]);
        assert_eq!(
            fetch("fragment_fetch.ll", "fetch_blend").unwrap(),
            [
                (0, PixelFormat::RGBA32Float),
                (1, PixelFormat::RGBA32Uint),
                (2, PixelFormat::RGBA32Float),
            ]
        );
        assert_eq!(
            fetch("exf_fetch_three.ll", "exf_fetch_three").unwrap(),
            [
                (0, PixelFormat::RGBA32Float),
                (1, PixelFormat::R32Uint),
                (2, PixelFormat::RGBA32Float),
            ]
        );
    }
}
