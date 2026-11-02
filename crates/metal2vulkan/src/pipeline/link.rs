//! Link: binding numbers, locations, spec ids, the pipeline layout, and pipeline-wide limits.
//!
//! Link assigns only numbers inside fixed storage classes; nothing it produces changes which
//! instructions a function contains. It reads each stage's link contract (`StageInterface`), never
//! the words:
//!
//! - **Bindings.** Every descriptor binding symbol of every stage is numbered densely in set 0 in
//!   [`ResourceSlot`] order (stage, kind, index), so the layout has no hole and no two Metal
//!   argument tables collide. Each binding lists exactly the Vulkan stage of the stage that
//!   declares it (`StageInterface::vk_stage`: a post-tessellation vertex function's is the
//!   evaluation stage), as does each push constant range. A combined image sampler's, a heap
//!   sampler's, or a separate sampler's constexpr sampler is its binding's immutable sampler, made
//!   for its use.
//! - **The heap**. Set 1 exists exactly when some stage declares a heap array
//!   (`ResourceKind::Heap`): one runtime array per heap class, numbered densely in class order and
//!   shared by every stage that declares the class (its stage flags are theirs), `PARTIALLY_BOUND
//!   | UPDATE_AFTER_BIND` in an `UPDATE_AFTER_BIND_POOL` set, and `UPDATE_UNUSED_WHILE_PENDING`
//!   where the profile has `descriptorBindingUpdateUnusedWhilePending`
//!   ([`heap_unused_while_pending`]), so one heap set can serve every command buffer. Every array has the pipeline's heap
//!   capacity: the most elements the update-after-bind limits allow once set 0 is counted
//!   (`maxDescriptorSetUpdateAfterBind*`, `maxPerStageDescriptorUpdateAfterBind*`,
//!   `maxPerStageUpdateAfterBindResources`, each split evenly over the classes of its kind, with
//!   a fragment stage's eight color attachments reserved), at most the handle table's reach
//!   (`memory::handles::TABLE_CAPACITY`). The binder writes element 0 of each array with its
//!   class's neutral and every resident texture at its own element. The function tables' array
//!   (`HandleClass::FunctionTable`) is a storage-buffer array, which needs
//!   `descriptorBindingStorageBufferUpdateAfterBind` and counts in the storage-buffer limits.
//! - **Specialization ids** are numbered densely in [`SpecKey`] order over the pipeline; a key two
//!   stages declare gets one id.
//! - **Locations** ([`varyings`]) from each stage's stage-IO contract: attributes densely by index,
//!   colors at their attachment, varyings by user key from the vertex side, packed into components
//!   when the dense numbering exceeds the interface limits.
//! - **Pipeline-wide limits**: the `maxDescriptorSet*` sums over stages and
//!   `maxBoundDescriptorSets`. A breach is `Unsupported{DescriptorCount}` naming the limit, never
//!   a code change. Per-stage limits are `select`'s.
//! - **`push_descriptor_eligible`** is set 0 fitting `maxPushDescriptors` on a profile with push
//!   descriptors; otherwise the same layout is used as a pooled set.
//! - **Render state** ([`vertex`], [`graphics`]): `VertexInputState` and `GraphicsState` from the
//!   descriptor, checked against the vertex stage's compiled fetch projection, with the fragment
//!   stage's imageblock ([`imageblock`]) and interlocked framebuffer fetch ([`fetch`]), and the
//!   precision notes of the formats fixed-function vertex input converts for the vertex stage
//!   (`vertex::fixed_function_notes`), which the module, shared by every such format, cannot
//!   record, with the measured properties a conversion relies on in place of a note. A mesh
//!   render pipeline has an object stage when it has an object function, its mesh stage, and a
//!   fragment stage exactly when it rasterizes; its `GraphicsState` is a render pipeline's with the
//!   mesh stage in the vertex stage's place, and it has no vertex input. An emulated mesh
//!   function's pipeline ([`mesh`]) adds a synthesized vertex stage, which reads the mesh
//!   function's records binding too (its stage flags are both stages').

pub mod band;
pub mod contract;
pub mod fetch;
pub mod geometry;
pub mod graphics;
pub mod imageblock;
pub mod mesh;
pub mod shader_tests;
pub mod stencil;
pub mod tessellation;
pub mod varyings;
pub mod vertex;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use crate::cache::VariantKey;
use crate::device::requirements::VkFeature;
use crate::device::requirements::VkLimit;
use crate::device::requirements::VkProperty;
use crate::device::DeviceProfile;
use crate::error::{
    Capability, DescriptorError, Error, Feature, InternalError, InternalKind, Unsupported,
};
use crate::library::FunctionArtifact;
use crate::metal;
use crate::pipeline::artifact::{PrecisionNote, SpecConstant, SpecializationLayout};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor,
    TessellationDescriptor, VertexDescriptor,
};
use crate::pipeline::graphics::GraphicsState;
use crate::pipeline::interface::{PipelineInterface, StageDescriptor, StageInterface};
use crate::pipeline::layout::{
    BindingSource, DescriptorBinding, DescriptorSetLayout, PipelineLayout, PushConstantRange,
};
use crate::pipeline::lower_inputs::ColorGroup;
use crate::pipeline::sym::{
    CombinedSampler, HandleClass, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::sym::{Symbol, VaryingKey};
use crate::pipeline::vertex::VertexInputState;
use crate::spirv::serialize::SymbolTable;
use crate::vk;

/// What link assigned.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkTable {
    /// `(set, binding)` of each descriptor binding symbol.
    pub bindings: BTreeMap<ResourceSlot, (u32, u32)>,
    /// `(location, component)` of each varying.
    pub locations: BTreeMap<VaryingKey, (u32, u32)>,
    /// The `SpecId` of each specialization key.
    pub spec: BTreeMap<SpecKey, u32>,
}

impl SymbolTable for LinkTable {
    fn resolve(&self, sym: &Symbol) -> Option<u32> {
        match sym {
            Symbol::Binding(r) => self.bindings.get(r).map(|b| b.1),
            Symbol::DescriptorSet(r) => self.bindings.get(r).map(|b| b.0),
            Symbol::Location(k) => self.locations.get(k).map(|l| l.0),
            Symbol::Component(k) => self.locations.get(k).map(|l| l.1),
            Symbol::SpecId(k) => self.spec.get(k).copied(),
        }
    }
}

/// The pipeline-level inputs of link.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LinkInputs {
    /// One function linked alone (per-function tests and validation): any stage; varyings are
    /// numbered by the stage-only scheme; no vertex input or graphics state.
    #[default]
    Standalone,
    /// A compute pipeline: exactly one kernel.
    Compute,
    /// A render pipeline: a vertex stage, then a fragment stage when rasterization is enabled.
    Render(Box<RenderLinkInputs>),
    /// A mesh render pipeline: an object stage when it has an object function, its mesh stage,
    /// then a fragment stage exactly when rasterization is enabled. Its fixed-function state is
    /// a render pipeline's without a vertex descriptor, tessellation or amplification.
    Mesh(Box<RenderLinkInputs>),
}

impl LinkInputs {
    /// One function linked alone.
    pub fn standalone() -> LinkInputs {
        LinkInputs::Standalone
    }

    /// A compute pipeline.
    pub fn compute() -> LinkInputs {
        LinkInputs::Compute
    }

    /// A render pipeline of `desc`'s fixed-function state under `variant`'s pulled vertex
    /// buffers, the variant of color group `group` where it draws one, whose descriptor `desc`
    /// holds the group's attachments alone (`pipeline::compile::color_group::descriptor`).
    pub fn render_variant(
        desc: &RenderPipelineDescriptor,
        variant: &VariantKey,
        group: Option<ColorGroup>,
    ) -> LinkInputs {
        let mut r = RenderLinkInputs::from(desc);
        r.pulled_vertex_buffers = variant.pulled_vertex_buffers.clone();
        r.color_group = group;
        r.line_band = variant.line_band;
        LinkInputs::Render(Box::new(r))
    }

    /// A mesh render pipeline of `desc`'s fixed-function state.
    pub fn mesh(desc: &MeshPipelineDescriptor) -> LinkInputs {
        LinkInputs::Mesh(Box::new(RenderLinkInputs::from(desc)))
    }
}

/// The fixed-function part of a render pipeline descriptor that link maps (the stage sources
/// are compiled before link).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderLinkInputs {
    pub vertex_descriptor: Option<VertexDescriptor>,
    pub color_attachments: [Option<ColorAttachmentDescriptor>; 8],
    pub depth_format: Option<metal::PixelFormat>,
    pub stencil_format: Option<metal::PixelFormat>,
    pub raster_sample_count: u32,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub rasterization_enabled: bool,
    pub input_primitive_topology: metal::PrimitiveTopologyClass,
    /// The layouts a pulled-vertex-buffer variant pulls (`VariantKey::pulled_vertex_buffers`);
    /// empty on the common path.
    pub pulled_vertex_buffers: BTreeSet<u32>,
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u32,
    /// The descriptor's tessellation state: read when the vertex stage is a post-tessellation
    /// vertex function (`None`: Metal's defaults), ignored otherwise.
    pub tessellation: Option<TessellationDescriptor>,
    /// `textureWriteRoundingMode`: the color writes the fragment stage rounds by.
    pub texture_write_rounding_mode: metal::TextureWriteRoundingMode,
    /// The group of the pipeline's color attachments the variant draws, whose attachments
    /// alone [`color_attachments`](Self::color_attachments) holds, at their locations
    /// (`GraphicsState::color_group`); `None` where each attachment is at its own index.
    pub color_group: Option<ColorGroup>,
    /// The variant draws its lines, or its triangles' edges, as bands through synthesized
    /// tessellation stages (`VariantKey::line_band`, [`band`]), each patch one primitive of this
    /// kind; `None` for every other variant.
    pub line_band: Option<crate::pipeline::BandPrimitive>,
}

impl RenderLinkInputs {
    /// Metal's defaults: one sample, rasterization on, no attachments, no vertex descriptor.
    pub fn new() -> RenderLinkInputs {
        RenderLinkInputs {
            vertex_descriptor: None,
            color_attachments: Default::default(),
            depth_format: None,
            stencil_format: None,
            raster_sample_count: 1,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterization_enabled: true,
            input_primitive_topology: metal::PrimitiveTopologyClass::Unspecified,
            pulled_vertex_buffers: BTreeSet::new(),
            max_vertex_amplification_count: 1,
            tessellation: None,
            texture_write_rounding_mode: metal::TextureWriteRoundingMode::TowardZero,
            color_group: None,
            line_band: None,
        }
    }
}

impl Default for RenderLinkInputs {
    fn default() -> RenderLinkInputs {
        RenderLinkInputs::new()
    }
}

impl From<&RenderPipelineDescriptor> for RenderLinkInputs {
    fn from(d: &RenderPipelineDescriptor) -> RenderLinkInputs {
        RenderLinkInputs {
            vertex_descriptor: d.vertex_descriptor.clone(),
            color_attachments: d.color_attachments,
            depth_format: d.depth_format,
            stencil_format: d.stencil_format,
            raster_sample_count: d.raster_sample_count,
            alpha_to_coverage: d.alpha_to_coverage,
            alpha_to_one: d.alpha_to_one,
            rasterization_enabled: d.rasterization_enabled,
            input_primitive_topology: d.input_primitive_topology,
            pulled_vertex_buffers: BTreeSet::new(),
            max_vertex_amplification_count: d.max_vertex_amplification_count,
            tessellation: d.tessellation,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            color_group: None,
            line_band: None,
        }
    }
}

impl From<&MeshPipelineDescriptor> for RenderLinkInputs {
    fn from(d: &MeshPipelineDescriptor) -> RenderLinkInputs {
        RenderLinkInputs {
            color_attachments: d.color_attachments,
            depth_format: d.depth_format,
            stencil_format: d.stencil_format,
            raster_sample_count: d.raster_sample_count,
            alpha_to_coverage: d.alpha_to_coverage,
            alpha_to_one: d.alpha_to_one,
            rasterization_enabled: d.rasterization_enabled,
            texture_write_rounding_mode: d.texture_write_rounding_mode,
            ..RenderLinkInputs::new()
        }
    }
}

/// What link produces for one pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Linked {
    pub table: LinkTable,
    pub layout: PipelineLayout,
    /// Empty except for a render pipeline.
    pub vertex_input: VertexInputState,
    /// The precision notes of the vertex stage's fixed-function attributes under this pipeline's
    /// vertex descriptor (`vertex::fixed_function_notes`), sorted: the compile API adds them to the
    /// vertex stage's notes. Empty but for a render pipeline whose vertex input converts a
    /// format otherwise than Metal's fetch may.
    pub vertex_notes: Vec<PrecisionNote>,
    /// The measured properties the vertex stage's fixed-function attributes rely on in place of a
    /// note under this pipeline's vertex descriptor (`vertex::fixed_function_notes`): the compile
    /// API adds them to the pipeline's requirements. Empty but for a render pipeline whose vertex
    /// input converts a normalized format on a profile measured to convert it as Metal does
    /// (`VkProperty::MetalNormalizedVertexFetch`).
    pub vertex_properties: BTreeSet<VkProperty>,
    /// The precision notes of the pipeline's fixed-function fragment operations under its graphics
    /// state (`graphics::coverage_notes`), sorted: the compile API adds them to the fragment
    /// stage's notes. Empty but for a render pipeline that covers by alpha on a profile not
    /// measured to cover as Metal does (`PrecisionNote::DeviceAlphaCoverage`).
    pub fragment_notes: Vec<PrecisionNote>,
    /// The measured properties the pipeline's fixed-function fragment operations rely on in place
    /// of a note (`graphics::coverage_notes`): the compile API adds them to the pipeline's
    /// requirements. Empty but for a render pipeline that covers by alpha on a profile measured to
    /// cover as Metal does (`VkProperty::MetalAlphaCoverage`).
    pub fragment_properties: BTreeSet<VkProperty>,
    pub interface: PipelineInterface,
    /// `Some` exactly for a render pipeline.
    pub graphics: Option<GraphicsState>,
    /// The specialization constants of each stage, with their link-assigned ids, in stage order.
    pub specialization: Vec<(Stage, SpecializationLayout)>,
    /// The synthesized control stage of a tessellation pipeline, with its push fields placed: the
    /// compile API builds its vertex and control stages from it.
    pub control: Option<tessellation::control::Control>,
    /// The synthesized geometry stages of a render pipeline whose fragment function reads
    /// `[[barycentric_coord]]` as varyings, or whose vertex function selects its slice or
    /// viewport through them ([`geometry`]): the compile API builds them.
    pub geometry: Option<geometry::GeometryStages>,
    /// The synthesized control and evaluation stages of a render pipeline variant that draws its
    /// lines as bands ([`band`]), with their push range placed: the compile API builds them.
    pub band: Option<band::LineBand>,
}

impl SymbolTable for Linked {
    fn resolve(&self, sym: &Symbol) -> Option<u32> {
        self.table.resolve(sym)
    }
}

/// Link the stages of one pipeline (see the module docs). `stages` are in pipeline order
/// (vertex, then fragment).
pub fn link(
    stages: &[&FunctionArtifact],
    inputs: &LinkInputs,
    p: &DeviceProfile,
) -> Result<Linked, Error> {
    let ifaces: Vec<&StageInterface> = stages.iter().map(|a| &a.interface).collect();
    check_stages(&ifaces, inputs)?;
    // A render pipeline whose vertex function is post-tessellation adds the synthesized vertex
    // and control stages: the control stage's factor and patch index buffers and push range.
    // On a device without `tessellationShader` the vertex function draws the tessellator's
    // triangles itself: no stage is added, and the pipeline states the emulation.
    let emulated = match (inputs, ifaces.first().and_then(|i| i.tessellation)) {
        (LinkInputs::Render(r), Some(stage)) if stage.emulated.is_some() => Some(
            tessellation::emulated_state(r.tessellation.as_ref(), stage)?,
        ),
        _ => None,
    };
    let plan = match (inputs, ifaces.first().and_then(|i| i.tessellation)) {
        (LinkInputs::Render(r), Some(stage)) if stage.emulated.is_none() => {
            tessellation::check_fragment(ifaces.get(1).copied())?;
            Some(tessellation::control_plan(
                r.tessellation.as_ref(),
                stage,
                r.max_vertex_amplification_count,
                p,
            )?)
        }
        _ => None,
    };
    // A fragment function that reads `[[barycentric_coord]]` as varyings, or a vertex function
    // that selects its slice or viewport through them, draws through synthesized geometry stages,
    // which write them.
    let geometry = match inputs {
        LinkInputs::Render(r) => geometry::plan(
            &ifaces,
            r.input_primitive_topology,
            r.max_vertex_amplification_count,
            false,
            p,
        )?,
        LinkInputs::Mesh(r) => geometry::plan(
            &ifaces,
            r.input_primitive_topology,
            r.max_vertex_amplification_count,
            true,
            p,
        )?,
        LinkInputs::Standalone | LinkInputs::Compute => None,
    };
    // A variant that draws its lines, or its triangles' edges, as bands draws them through
    // synthesized tessellation stages.
    let band = match inputs {
        LinkInputs::Render(r) => match r.line_band {
            Some(primitive) => Some(band::plan(
                &ifaces,
                (primitive, geometry.is_some()),
                r.max_vertex_amplification_count,
                p,
            )?),
            None => None,
        },
        _ => None,
    };
    // A mesh function reads its payload within its object function's. An emulated mesh
    // function's pipeline draws through a synthesized vertex stage, which reads the records the
    // mesh stage writes.
    let records = match inputs {
        LinkInputs::Mesh(_) => {
            mesh::payload(&ifaces)?;
            mesh::records(&ifaces)?
        }
        _ => None,
    };
    let mut table = LinkTable::default();
    let mut synthesized: Vec<(ResourceSlot, vk::ShaderStageFlags)> = plan
        .as_ref()
        .map(tessellation::ControlPlan::slots)
        .unwrap_or_default()
        .into_iter()
        .map(|slot| (slot, vk::ShaderStageFlags::TESSELLATION_CONTROL))
        .collect();
    if records.is_some() {
        synthesized.push((mesh::RECORDS_SLOT, vk::ShaderStageFlags::VERTEX));
        synthesized.push((mesh::WORK_SLOT, vk::ShaderStageFlags::VERTEX));
    }
    let mut layout = layout(&ifaces, &synthesized, &mut table, p)?;
    count_variables(&mut layout, stages, &table);
    let control = match plan {
        Some(plan) => {
            let offset = tessellation::push_offset(
                &layout.push_constants,
                (tessellation::control::PUSH_BYTES, 4),
                Feature::Tessellation,
                p,
            )?;
            layout.push_constants.push(PushConstantRange {
                stages: vk::ShaderStageFlags::TESSELLATION_CONTROL,
                offset,
                size: tessellation::control::PUSH_BYTES,
            });
            Some((plan, offset))
        }
        None => None,
    };
    let band = match band {
        Some(mut b) => {
            b.push_offset = tessellation::push_offset(
                &layout.push_constants,
                (band::LineBand::PUSH_BYTES, band::LineBand::PUSH_BYTES),
                Feature::WideLines,
                p,
            )?;
            layout.push_constants.push(PushConstantRange {
                stages: vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                offset: b.push_offset,
                size: band::LineBand::PUSH_BYTES,
            });
            Some(b)
        }
        None => None,
    };
    let specialization = specialize(&ifaces, &mut table)?;
    table.locations = varyings::assign(stages, p)?;
    if let Some(g) = &geometry {
        g.check_limits(&table.locations, p)?;
    }
    let interface = pipeline_interface(&ifaces);
    let mut vertex_notes = vertex::FixedFunctionNotes::default();
    let (vertex_input, graphics) = match inputs {
        LinkInputs::Render(r) => {
            let vs = ifaces.first().ok_or_else(|| missing_stage(Stage::Vertex))?;
            let vi = match (control, emulated) {
                (Some(_), _) | (_, Some(_)) => {
                    tessellation::vertex_input_state(vs, r.vertex_descriptor.as_ref())?
                }
                (None, None) => {
                    let vi = vertex::vertex_input_state(
                        vs,
                        r.vertex_descriptor.as_ref(),
                        &r.pulled_vertex_buffers,
                        &table.locations,
                        p,
                    )?;
                    vertex_notes =
                        vertex::fixed_function_notes(vs, r.vertex_descriptor.as_ref(), p)?;
                    vi
                }
            };
            let ts = control.map(|(plan, offset)| plan.state(offset));
            let mut gs = render_state(r, vs, stages.get(1).copied(), &vi, ts, p)?;
            // An emulated tessellation pipeline draws a triangle list, which no draw changes.
            if let Some(e) = emulated {
                gs.topology = vk::PrimitiveTopology::TRIANGLE_LIST;
                gs.dynamic_states.retain(|d| {
                    !matches!(
                        *d,
                        vk::DynamicState::PRIMITIVE_TOPOLOGY
                            | vk::DynamicState::PRIMITIVE_RESTART_ENABLE
                    )
                });
                gs.emulated_tessellation = Some(e);
            }
            // A band variant draws each primitive as a patch of its vertices, which no draw
            // changes.
            if let Some(b) = &band {
                gs.topology = vk::PrimitiveTopology::PATCH_LIST;
                gs.dynamic_states.retain(|d| {
                    !matches!(
                        *d,
                        vk::DynamicState::PRIMITIVE_TOPOLOGY
                            | vk::DynamicState::PRIMITIVE_RESTART_ENABLE
                    )
                });
                gs.line_band = Some(crate::pipeline::graphics::LineBandState {
                    push_offset: b.push_offset,
                    primitive: b.primitive,
                });
            }
            (vi, Some(gs))
        }
        // The mesh stage stands in the vertex stage's place; the fragment stage follows it.
        LinkInputs::Mesh(r) => {
            let at = ifaces
                .iter()
                .position(|i| i.stage == Stage::Mesh)
                .ok_or_else(|| missing_stage(Stage::Mesh))?;
            let vi = VertexInputState::default();
            let fragment = stages.get(at + 1).copied();
            let mut gs = render_state(r, ifaces[at], fragment, &vi, None, p)?;
            // An emulated mesh function's primitives are drawn as a list of their kind.
            if let Some(records) = records {
                gs.topology = records.topology();
                gs.mesh_records = Some(records);
            }
            (vi, Some(gs))
        }
        LinkInputs::Standalone | LinkInputs::Compute => (VertexInputState::default(), None),
    };
    let fragment_notes = graphics
        .as_ref()
        .map(|g| graphics::coverage_notes(g, p))
        .unwrap_or_default();
    // Totality: every relocation of every stage resolves. A miss is a link bug.
    for a in stages {
        if let Some((sym, _)) = a
            .module
            .relocs
            .iter()
            .find(|(s, _)| table.resolve(s).is_none())
        {
            return Err(Error::Internal(InternalError::new(
                InternalKind::Invariant,
                format!("link table misses {sym:?}"),
            )));
        }
    }
    Ok(Linked {
        table,
        layout,
        vertex_input,
        vertex_notes: vertex_notes.notes,
        vertex_properties: vertex_notes.properties,
        fragment_notes: fragment_notes.notes,
        fragment_properties: fragment_notes.properties,
        interface,
        graphics,
        specialization,
        control: control.map(|(plan, offset)| plan.control(offset)),
        geometry,
        band,
    })
}

/// The graphics state of a render or mesh render pipeline whose geometry stage (its vertex or
/// mesh stage) is `geometry`, with its fragment stage's.
fn render_state(
    r: &RenderLinkInputs,
    geometry: &StageInterface,
    fragment: Option<&FunctionArtifact>,
    vi: &VertexInputState,
    ts: Option<crate::pipeline::graphics::TessellationState>,
    p: &DeviceProfile,
) -> Result<GraphicsState, Error> {
    // The imageblock's refusals first: a multisampled imageblock is the specific reason where
    // its storage writes would also refuse the sample invocations.
    let ib = imageblock::fragment_imageblock(r, fragment)?;
    // The interlocked fetch's form against this pipeline's state, before the sample invocations
    // its storage writes would also refuse.
    let storage = fetch::storage_attachments(r, fragment, p)?;
    // The shader-side stencil against this pipeline's state, before the sample invocations its
    // shadow writes would also refuse.
    let shadow = stencil::shadow_stencil(r, fragment, p)?;
    // The fragment's own depth test against this pipeline's state.
    let shader_tests = shader_tests::shader_tests(r, fragment, p)?;
    // The extended-range blends the fragment computes over input attachments, against this
    // pipeline's attachments.
    let input_blends = fetch::input_blends(r, fragment, p)?;
    // The word merges the fragment computes over input attachments, against this pipeline's
    // write masks.
    fetch::word_merges(r, fragment, p)?;
    let fs = fragment.map(|f| &f.interface);
    let mut gs = graphics::graphics_state(r, (geometry, fs), vi, ts, &input_blends, p)?;
    // The color writes the fragment rounds against this pipeline's attachments, once the
    // descriptor's checks of its outputs passed.
    crate::pipeline::color_writes::check(
        fragment,
        &r.color_attachments,
        (r.alpha_to_coverage, r.texture_write_rounding_mode),
        p,
    )?;
    gs.fragment_imageblock = ib;
    gs.color_outputs = color_outputs(fs, &storage)?;
    gs.storage_attachments = storage;
    gs.shadow_stencil = shadow;
    gs.shader_tests = shader_tests;
    Ok(gs)
}

/// The color attachments fragment stage `fs` writes (`GraphicsState::color_outputs`), bit `n`
/// for attachment `n`: those a first (index 0) color output of its function names, and those it
/// writes in the interlocked storage form (`storage`). An index past the mask is no output link
/// admits (`maxFragmentOutputAttachments` and Metal's eight colors are below it).
fn color_outputs(
    fs: Option<&StageInterface>,
    storage: &[crate::pipeline::graphics::StorageAttachmentUse],
) -> Result<u32, Error> {
    let outputs = fs
        .into_iter()
        .flat_map(|f| &f.io.outputs)
        .filter_map(|o| match o.role {
            crate::pipeline::stage_io::IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        });
    let stored = storage.iter().filter(|s| s.writes).map(|s| s.color_index);
    outputs.chain(stored).try_fold(0u32, |mask, color| {
        1u32.checked_shl(color)
            .map(|bit| mask | bit)
            .ok_or_else(|| {
                Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    "a color output past the 32 attachments a mask holds",
                ))
            })
    })
}

fn missing_stage(stage: Stage) -> Error {
    Error::InvalidDescriptor(DescriptorError::MissingStage { stage })
}

fn mismatch(expected: Stage, found: Stage) -> Error {
    Error::InvalidDescriptor(DescriptorError::StageMismatch { expected, found })
}

/// The stages match the pipeline kind: one kernel; a vertex stage and a fragment stage exactly
/// when rasterization is enabled; an optional object stage, a mesh stage, and a fragment stage
/// exactly when rasterization is enabled; or any one stage alone.
fn check_stages(ifaces: &[&StageInterface], inputs: &LinkInputs) -> Result<(), Error> {
    let stages: Vec<Stage> = ifaces.iter().map(|i| i.stage).collect();
    match inputs {
        LinkInputs::Standalone => match stages.as_slice() {
            [_] => Ok(()),
            [] => Err(missing_stage(Stage::Kernel)),
            [first, rest @ ..] => Err(mismatch(*first, rest[0])),
        },
        LinkInputs::Compute => match stages.as_slice() {
            [Stage::Kernel] => Ok(()),
            [] => Err(missing_stage(Stage::Kernel)),
            [Stage::Kernel, extra, ..] => Err(mismatch(Stage::Kernel, *extra)),
            [other, ..] => Err(mismatch(Stage::Kernel, *other)),
        },
        // Metal asserts on a mesh pipeline without its mesh function, or without a fragment
        // function where it rasterizes.
        LinkInputs::Mesh(r) => {
            let geometry = match stages.first() {
                Some(Stage::Object) => &stages[1..],
                _ => &stages[..],
            };
            let expected: &[Stage] = if r.rasterization_enabled {
                &[Stage::Mesh, Stage::Fragment]
            } else {
                &[Stage::Mesh]
            };
            for (i, want) in expected.iter().enumerate() {
                match geometry.get(i) {
                    Some(s) if s == want => {}
                    Some(s) => return Err(mismatch(*want, *s)),
                    None => return Err(missing_stage(*want)),
                }
            }
            match geometry.get(expected.len()) {
                Some(extra) => Err(mismatch(*expected.last().unwrap_or(&Stage::Mesh), *extra)),
                None => Ok(()),
            }
        }
        LinkInputs::Render(r) => {
            let expected: &[Stage] = if r.rasterization_enabled {
                &[Stage::Vertex, Stage::Fragment]
            } else {
                &[Stage::Vertex]
            };
            for (i, want) in expected.iter().enumerate() {
                match stages.get(i) {
                    Some(s) if s == want => {}
                    Some(s) => return Err(mismatch(*want, *s)),
                    // A render pipeline with rasterization on may omit the fragment function.
                    None if *want == Stage::Fragment => {}
                    None => return Err(missing_stage(*want)),
                }
            }
            match stages.get(expected.len()) {
                Some(extra) => Err(mismatch(*expected.last().unwrap_or(&Stage::Vertex), *extra)),
                None => Ok(()),
            }
        }
    }
}

/// What feeds the binding of `slot`.
fn binding_source(slot: ResourceSlot) -> BindingSource {
    let metal = |class| SlotRef {
        stage: slot.stage,
        class,
        index: slot.index,
    };
    match slot.kind {
        ResourceKind::Buffer => BindingSource::Buffer(metal(SlotClass::Buffer)),
        ResourceKind::Texture => BindingSource::Texture(metal(SlotClass::Texture)),
        ResourceKind::Sampler => BindingSource::Sampler(metal(SlotClass::Sampler)),
        ResourceKind::Heap(class) => BindingSource::Heap(class),
        ResourceKind::Synthetic(binding) => BindingSource::Synthetic {
            stage: slot.stage,
            binding,
        },
        ResourceKind::InputAttachment(color) => BindingSource::InputAttachment(color),
        ResourceKind::CombinedTextureSampler(sampler) => BindingSource::CombinedTextureSampler {
            texture: metal(SlotClass::Texture),
            sampler,
        },
        ResourceKind::HeapSampler(sampler) => BindingSource::HeapSampler {
            stage: slot.stage,
            sampler: CombinedSampler::from_heap_index(sampler, slot.index),
        },
        ResourceKind::AccelerationStructure(level) => BindingSource::AccelerationStructure {
            slot: metal(SlotClass::Buffer),
            level,
        },
        ResourceKind::TileAttachment(color) => BindingSource::TileAttachment(color),
        ResourceKind::TileMap => BindingSource::TileMap(metal(SlotClass::Texture)),
        ResourceKind::SeparateTexture(sampler) => {
            // The use is above the texture slot in the index (`ResourceSlot::separate_texture`).
            let (texture, sampler) = ResourceSlot::separate_texture_parts(sampler, slot.index);
            BindingSource::SeparateTexture {
                texture: SlotRef {
                    stage: slot.stage,
                    class: SlotClass::Texture,
                    index: texture,
                },
                sampler,
            }
        }
        ResourceKind::SeparateSampler(sampler) => BindingSource::SeparateSampler {
            stage: slot.stage,
            sampler: CombinedSampler::from_heap_index(sampler, slot.index),
        },
        ResourceKind::ExtendedShadow(color) => BindingSource::ExtendedShadow(color),
        ResourceKind::SnapshotTexels(texture) => BindingSource::SnapshotTexels(texture),
        ResourceKind::YCbCrPlanes(_) => BindingSource::YCbCrPlanes(metal(SlotClass::Texture)),
    }
}

/// The immutable sampler of binding `slot`: a combined image sampler's, heap sampler's, or
/// separate sampler's constexpr sampler, made for its use; a planes image's device conversion
/// (`ResourceKind::YCbCrPlanes`).
fn immutable_sampler(
    slot: ResourceSlot,
    p: &DeviceProfile,
) -> Result<Option<vk::SamplerDesc>, Error> {
    let c = match slot.kind {
        ResourceKind::YCbCrPlanes(sampler) => return Ok(Some(vk::SamplerDesc::ycbcr(sampler))),
        ResourceKind::CombinedTextureSampler(c) => c,
        ResourceKind::HeapSampler(sampler) | ResourceKind::SeparateSampler(sampler) => {
            CombinedSampler::from_heap_index(sampler, slot.index)
        }
        _ => return Ok(None),
    };
    match c.sampler {
        crate::pipeline::sym::SamplerRef::Constexpr(words) => Ok(Some(
            crate::lower::sampler::constexpr_desc(words, c.sampler_use(), p)?,
        )),
        _ => Ok(None),
    }
}

/// Set each binding's `variables` (`DescriptorBinding::variables`): the most variables one of
/// `stages`' entry points uses at it. A module's variables are its `Binding` relocations' targets
/// (the decoration's word two before the relocated literal), which `table` resolves to their
/// bindings; those its entry point uses are those its `OpEntryPoint` lists, where a module lists
/// every global variable its entry point's call tree references: from SPIR-V 1.4 on ("the
/// interface's storage classes are all storage classes used in declaring all global variables
/// referenced by the entry point's call tree"). The serializer writes 1.5 (`SPIRV_VERSION`); a
/// module of an earlier version, whose interface lists its inputs and outputs alone, counts each
/// variable it declares.
fn count_variables(layout: &mut PipelineLayout, stages: &[&FunctionArtifact], table: &LinkTable) {
    for a in stages {
        let used = used_variables(&a.module.words, &a.module.relocs, table);
        for set in &mut layout.sets {
            for b in &mut set.bindings {
                if let Some(&n) = used.get(&(set.set, b.binding)) {
                    b.variables = b.variables.max(n);
                }
            }
        }
    }
}

/// The variables the entry point of module `words` uses at each `(set, binding)`, its `Binding`
/// relocations `relocs` resolved through `table` ([`count_variables`]): every variable a module
/// before SPIR-V 1.4 declares there.
fn used_variables(
    words: &[u32],
    relocs: &[(Symbol, u32)],
    table: &LinkTable,
) -> BTreeMap<(u32, u32), u32> {
    let listed = words.get(1).is_some_and(|&version| version >= 0x0001_0400);
    let interface = entry_interface(words);
    let mut used = BTreeMap::new();
    for (sym, at) in relocs {
        let Symbol::Binding(slot) = sym else {
            continue;
        };
        let id = (*at as usize).checked_sub(2).and_then(|i| words.get(i));
        if let (Some(&binding), Some(id)) = (table.bindings.get(slot), id) {
            if !listed || interface.contains(id) {
                *used.entry(binding).or_default() += 1;
            }
        }
    }
    used
}

/// The interface of the first `OpEntryPoint` of module `words`: the ids after its execution model,
/// its function and its name (a string of words, the last holding its NUL byte).
fn entry_interface(words: &[u32]) -> BTreeSet<u32> {
    let entry_point = crate::spv::Op::EntryPoint as u32;
    let mut at = 5;
    while let Some(&word) = words.get(at) {
        let len = (word >> 16) as usize;
        let Some(inst) = words.get(at..at + len).filter(|_| len > 0) else {
            break;
        };
        if word & 0xFFFF == entry_point {
            let name = inst
                .iter()
                .skip(3)
                .position(|w| w.to_le_bytes().contains(&0));
            return name
                .map(|end| inst[3 + end + 1..].iter().copied().collect())
                .unwrap_or_default();
        }
        at += len;
    }
    BTreeSet::new()
}

/// The stages declaring one heap class, and its descriptor type.
struct HeapDecl {
    ty: vk::DescriptorType,
    stages: vk::ShaderStageFlags,
    slots: Vec<ResourceSlot>,
}

/// Number every binding symbol densely in set 0 and every heap array in set 1, build the layout,
/// and check the pipeline-wide limits. `synthesized` are the storage buffers the pipeline's
/// synthesized stages read, with those stages: a tessellation pipeline's control-stage buffers, and
/// an emulated mesh function's records, which its vertex stage reads too.
fn layout(
    ifaces: &[&StageInterface],
    synthesized: &[(ResourceSlot, vk::ShaderStageFlags)],
    table: &mut LinkTable,
    p: &DeviceProfile,
) -> Result<PipelineLayout, Error> {
    let mut decls: BTreeMap<ResourceSlot, (StageDescriptor, vk::ShaderStageFlags)> =
        BTreeMap::new();
    // A tessellation pipeline's factor, patch index, and control-point index buffers are the
    // control stage's; an emulated mesh function's records are its synthesized vertex stage's too.
    for &(slot, stages) in synthesized {
        let d = StageDescriptor::new(slot, vk::DescriptorType::STORAGE_BUFFER, 1);
        decls.insert(d.slot, (d, stages));
    }
    let mut heap: BTreeMap<HandleClass, HeapDecl> = BTreeMap::new();
    for i in ifaces {
        for d in &i.descriptors {
            if d.slot.stage != i.stage {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!("{:?} declares a binding of {:?}", i.stage, d.slot.stage),
                )));
            }
            // A combined image sampler is exactly a `CombinedTextureSampler` symbol, which names
            // its texture slot and its sampler, or a planes image's (`YCbCrPlanes`), which names
            // its texture slot and the conversion its immutable sampler bakes.
            let combined = matches!(
                d.slot.kind,
                ResourceKind::CombinedTextureSampler(_) | ResourceKind::YCbCrPlanes(_)
            );
            if combined != (d.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER) {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!("binding {:?} declared as {:?}", d.slot, d.ty),
                )));
            }
            if let ResourceKind::Heap(class) = d.slot.kind {
                // A heap array's count is link's (the contract states 0).
                if d.count != 0 {
                    return Err(Error::Internal(InternalError::new(
                        InternalKind::Invariant,
                        format!("heap binding {:?} with count {}", d.slot, d.count),
                    )));
                }
                let e = heap.entry(class).or_insert(HeapDecl {
                    ty: d.ty,
                    stages: vk::ShaderStageFlags::empty(),
                    slots: Vec::new(),
                });
                if e.ty != d.ty || e.slots.contains(&d.slot) {
                    return Err(Error::Internal(InternalError::new(
                        InternalKind::Invariant,
                        format!("heap binding {:?} declared twice or as {:?}", d.slot, d.ty),
                    )));
                }
                e.stages |= i.vk_stage().flags();
                e.slots.push(d.slot);
                continue;
            }
            let stage = i.vk_stage().flags();
            let shared = synthesized.iter().any(|(slot, _)| *slot == d.slot);
            match decls.get_mut(&d.slot) {
                // A binding a synthesized stage reads too, of the same descriptor.
                Some((known, stages)) if shared && known == d => *stages |= stage,
                Some(_) => {
                    return Err(Error::Internal(InternalError::new(
                        InternalKind::Invariant,
                        format!("binding {:?} declared twice", d.slot),
                    )))
                }
                None => {
                    decls.insert(d.slot, (*d, stage));
                }
            }
        }
    }
    let mut bindings = Vec::with_capacity(decls.len());
    for (n, (slot, (d, stages))) in decls.into_iter().enumerate() {
        let binding = u32::try_from(n)
            .map_err(|_| Error::from(crate::spirv::error::BuildError::IdSpaceExhausted))?;
        table.bindings.insert(slot, (0, binding));
        bindings.push(DescriptorBinding {
            binding,
            ty: d.ty,
            count: d.count,
            stages,
            source: binding_source(slot),
            immutable_sampler: immutable_sampler(slot, p)?,
            flags: vk::DescriptorBindingFlags::empty(),
            variables: 0,
        });
    }
    check_set_limits(&bindings, p)?;
    let heap_set = heap_set(&heap, &bindings, ifaces, table, p)?;
    let limits = &p.fields().limits;
    // One set, two with the heap: `maxBoundDescriptorSets` is at least 4 on every Vulkan device;
    // checked anyway.
    let sets = 1 + u32::from(heap_set.is_some());
    if sets > limits.max_bound_descriptor_sets {
        return Err(limit_error(
            VkLimit::MaxBoundDescriptorSets,
            u64::from(sets),
            u64::from(limits.max_bound_descriptor_sets),
        ));
    }
    let total: u64 = bindings.iter().map(|b| u64::from(b.count)).sum();
    let push_descriptor_eligible =
        p.fields().descriptors.push_descriptor && total <= u64::from(limits.max_push_descriptors);
    let mut push_constants = Vec::new();
    for i in ifaces {
        if let Some((offset, size)) = i.push.push_range {
            let end = u64::from(offset) + u64::from(size);
            if size == 0 || end > u64::from(limits.max_push_constants_size) {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!(
                        "{:?} push range [{offset}, {end}) outside the device's push-constant range",
                        i.stage
                    ),
                )));
            }
            let stages = i.vk_stage().flags();
            // An emulated mesh pipeline's object and mesh functions are both compute entries:
            // Vulkan admits one range per stage, so theirs is the hull of their disjoint windows.
            match push_constants
                .iter_mut()
                .find(|r: &&mut PushConstantRange| r.stages == stages)
            {
                Some(r) => {
                    let start = r.offset.min(offset);
                    let end = (r.offset + r.size).max(offset + size);
                    r.offset = start;
                    r.size = end - start;
                }
                None => push_constants.push(PushConstantRange {
                    stages,
                    offset,
                    size,
                }),
            }
        }
    }
    let mut sets = vec![DescriptorSetLayout {
        set: 0,
        bindings,
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
    }];
    sets.extend(heap_set);
    Ok(PipelineLayout {
        sets,
        push_constants,
        push_descriptor_eligible,
    })
}

/// A fragment stage's color attachments count in `maxPerStageUpdateAfterBindResources` as in
/// `maxPerStageResources`; link does not see them, so the heap reserves Metal's maximum.
const RESERVED_COLOR_ATTACHMENTS: u64 = 8;

/// Set 1, the heap (the module docs): one array per class of `heap`, numbered in class order,
/// each of the pipeline's heap capacity. `set0` is set 0's bindings, which the update-after-bind
/// limits count too.
fn heap_set(
    heap: &BTreeMap<HandleClass, HeapDecl>,
    set0: &[DescriptorBinding],
    ifaces: &[&StageInterface],
    table: &mut LinkTable,
    p: &DeviceProfile,
) -> Result<Option<DescriptorSetLayout>, Error> {
    if heap.is_empty() {
        return Ok(None);
    }
    let d = &p.fields().descriptors;
    // The layout's own features (the module states its indexing features); `select` refuses a
    // function whose heap the profile cannot hold, so a miss here is a function compiled for
    // another profile.
    let mut missing = Vec::new();
    let mut need = |has: bool, f: VkFeature| {
        if !has && !missing.contains(&Capability::Feature(f)) {
            missing.push(Capability::Feature(f));
        }
    };
    need(
        d.runtime_descriptor_array,
        VkFeature::RuntimeDescriptorArray,
    );
    need(
        d.partially_bound,
        VkFeature::DescriptorBindingPartiallyBound,
    );
    // The feature a missing capability is refused as: the function tables' where only their array
    // needs it, else the textures'.
    let mut feature = Feature::DynamicCall;
    for h in heap.values() {
        match h.ty {
            vk::DescriptorType::SAMPLER | vk::DescriptorType::SAMPLED_IMAGE => {
                feature = Feature::ArgumentBufferTextures;
                need(
                    d.update_after_bind_sampled_image,
                    VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
                )
            }
            vk::DescriptorType::STORAGE_IMAGE => {
                feature = Feature::ArgumentBufferTextures;
                need(
                    d.update_after_bind_storage_image,
                    VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
                )
            }
            vk::DescriptorType::STORAGE_BUFFER => need(
                d.update_after_bind_storage_buffer,
                VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
            ),
            _ => return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into()),
        }
    }
    if !missing.is_empty() {
        return Err(Unsupported::needs_any_of(feature, vec![missing]).into());
    }
    let capacity = heap_capacity(heap, set0, ifaces, p)?;
    let mut flags =
        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
    // A device-wide heap writes an element (a texture made, or gone) while command buffers that
    // read others are pending (`heap_unused_while_pending`).
    if heap_unused_while_pending(p) {
        flags |= vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
    }
    let mut bindings = Vec::with_capacity(heap.len());
    for (n, (class, h)) in heap.iter().enumerate() {
        let binding = u32::try_from(n)
            .map_err(|_| Error::from(crate::spirv::error::BuildError::IdSpaceExhausted))?;
        for slot in &h.slots {
            table.bindings.insert(*slot, (1, binding));
        }
        bindings.push(DescriptorBinding {
            binding,
            ty: h.ty,
            count: capacity,
            stages: h.stages,
            source: BindingSource::Heap(*class),
            immutable_sampler: None,
            flags,
            variables: 0,
        });
    }
    Ok(Some(DescriptorSetLayout {
        set: 1,
        bindings,
        flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
    }))
}

/// Whether set 1's heap arrays are `UPDATE_UNUSED_WHILE_PENDING` on `p`: where the profile has
/// `descriptorBindingUpdateUnusedWhilePending`, so a consumer keeping one heap set for every
/// command buffer can write an element no pending command buffer reads. A function that declares
/// a heap array requires the feature there.
pub fn heap_unused_while_pending(p: &DeviceProfile) -> bool {
    p.fields().descriptors.update_unused_while_pending
}

/// The heap capacity: the most elements every heap array can have (the module docs), or the
/// limit a heap of one element per array already breaches.
fn heap_capacity(
    heap: &BTreeMap<HandleClass, HeapDecl>,
    set0: &[DescriptorBinding],
    ifaces: &[&StageInterface],
    p: &DeviceProfile,
) -> Result<u32, Error> {
    let l = &p.fields().limits;
    let kind = |ty: vk::DescriptorType| match ty {
        vk::DescriptorType::SAMPLER => SAMPLERS,
        vk::DescriptorType::STORAGE_IMAGE => IMAGES,
        vk::DescriptorType::STORAGE_BUFFER => STORAGE,
        _ => SAMPLED,
    };
    let kinds = [
        (
            SAMPLERS,
            VkLimit::MaxDescriptorSetUpdateAfterBindSamplers,
            l.max_descriptor_set_update_after_bind_samplers,
            VkLimit::MaxPerStageDescriptorUpdateAfterBindSamplers,
            l.max_per_stage_descriptor_update_after_bind_samplers,
        ),
        (
            SAMPLED,
            VkLimit::MaxDescriptorSetUpdateAfterBindSampledImages,
            l.max_descriptor_set_update_after_bind_sampled_images,
            VkLimit::MaxPerStageDescriptorUpdateAfterBindSampledImages,
            l.max_per_stage_descriptor_update_after_bind_sampled_images,
        ),
        (
            IMAGES,
            VkLimit::MaxDescriptorSetUpdateAfterBindStorageImages,
            l.max_descriptor_set_update_after_bind_storage_images,
            VkLimit::MaxPerStageDescriptorUpdateAfterBindStorageImages,
            l.max_per_stage_descriptor_update_after_bind_storage_images,
        ),
        (
            STORAGE,
            VkLimit::MaxDescriptorSetUpdateAfterBindStorageBuffers,
            l.max_descriptor_set_update_after_bind_storage_buffers,
            VkLimit::MaxPerStageDescriptorUpdateAfterBindStorageBuffers,
            l.max_per_stage_descriptor_update_after_bind_storage_buffers,
        ),
    ];
    let mut capacity = u64::from(crate::memory::handles::TABLE_CAPACITY);
    // `limit` holds `used + arrays · capacity`.
    let mut bound = |limit: VkLimit, available: u32, used: u64, arrays: u64| -> Result<(), Error> {
        if arrays == 0 {
            return Ok(());
        }
        let room = u64::from(available).saturating_sub(used) / arrays;
        if room == 0 {
            return Err(limit_error(
                limit,
                used.saturating_add(arrays),
                u64::from(available),
            ));
        }
        capacity = capacity.min(room);
        Ok(())
    };
    let all = kind_counts(set0.iter())?;
    for (k, set_limit, set_max, _, _) in kinds {
        let arrays = heap.values().filter(|h| kind(h.ty) == k).count() as u64;
        bound(set_limit, set_max, all[k], arrays)?;
    }
    for i in ifaces {
        let flags = i.vk_stage().flags();
        let own = kind_counts(set0.iter().filter(|b| b.stages.contains(flags)))?;
        let in_stage = |h: &&HeapDecl| h.stages.contains(flags);
        for (k, _, _, stage_limit, stage_max) in kinds {
            let arrays = heap
                .values()
                .filter(in_stage)
                .filter(|h| kind(h.ty) == k)
                .count() as u64;
            bound(stage_limit, stage_max, own[k], arrays)?;
        }
        let resources = own
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != SAMPLERS)
            .map(|(_, n)| *n)
            .sum::<u64>()
            .saturating_add(if i.stage == Stage::Fragment {
                RESERVED_COLOR_ATTACHMENTS
            } else {
                0
            });
        let arrays = heap
            .values()
            .filter(in_stage)
            .filter(|h| kind(h.ty) != SAMPLERS)
            .count() as u64;
        bound(
            VkLimit::MaxPerStageUpdateAfterBindResources,
            l.max_per_stage_update_after_bind_resources,
            resources,
            arrays,
        )?;
    }
    u32::try_from(capacity).map_err(|_| {
        Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "a heap capacity beyond u32",
        ))
    })
}

fn limit_error(limit: VkLimit, required: u64, available: u64) -> Error {
    Unsupported::needs(
        Feature::DescriptorCount,
        Capability::Limit {
            limit,
            required,
            available,
        },
    )
    .into()
}

const SAMPLERS: usize = 0;
const UNIFORM: usize = 1;
const STORAGE: usize = 2;
const SAMPLED: usize = 3;
const IMAGES: usize = 4;
const INPUT: usize = 5;

/// The descriptors of `bindings` per kind (samplers, uniform buffers, storage buffers, sampled
/// images, storage images, input attachments), counted as Vulkan counts them: combined image
/// samplers count as samplers and sampled images, uniform texel buffers as sampled images,
/// storage texel buffers as storage images.
fn kind_counts<'a>(
    bindings: impl Iterator<Item = &'a DescriptorBinding>,
) -> Result<[u64; 6], Error> {
    let mut n = [0u64; 6];
    for b in bindings {
        let c = u64::from(b.count);
        let kinds: &[usize] = match b.ty {
            vk::DescriptorType::SAMPLER => &[SAMPLERS],
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => &[SAMPLERS, SAMPLED],
            vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                &[SAMPLED]
            }
            vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                &[IMAGES]
            }
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                &[UNIFORM]
            }
            vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                &[STORAGE]
            }
            vk::DescriptorType::INPUT_ATTACHMENT => &[INPUT],
            // Counted by their own limits ([`check_set_limits`]), none of these.
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => &[],
            other => {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!("descriptor type {other:?} outside the link contract"),
                )))
            }
        };
        for k in kinds {
            n[*k] = n[*k].saturating_add(c);
        }
    }
    Ok(n)
}

/// The `maxDescriptorSet*` sums of one set ([`kind_counts`], and the acceleration structures).
fn check_set_limits(bindings: &[DescriptorBinding], p: &DeviceProfile) -> Result<(), Error> {
    let l = &p.fields().limits;
    let n = kind_counts(bindings.iter())?;
    let accel: u64 = bindings
        .iter()
        .filter(|b| b.ty == vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .map(|b| u64::from(b.count))
        .sum();
    let checks = [
        (
            VkLimit::MaxDescriptorSetAccelerationStructures,
            accel,
            l.max_descriptor_set_acceleration_structures,
        ),
        (
            VkLimit::MaxDescriptorSetSamplers,
            n[SAMPLERS],
            l.max_descriptor_set_samplers,
        ),
        (
            VkLimit::MaxDescriptorSetUniformBuffers,
            n[UNIFORM],
            l.max_descriptor_set_uniform_buffers,
        ),
        (
            VkLimit::MaxDescriptorSetStorageBuffers,
            n[STORAGE],
            l.max_descriptor_set_storage_buffers,
        ),
        (
            VkLimit::MaxDescriptorSetSampledImages,
            n[SAMPLED],
            l.max_descriptor_set_sampled_images,
        ),
        (
            VkLimit::MaxDescriptorSetStorageImages,
            n[IMAGES],
            l.max_descriptor_set_storage_images,
        ),
        (
            VkLimit::MaxDescriptorSetInputAttachments,
            n[INPUT],
            l.max_descriptor_set_input_attachments,
        ),
    ];
    match checks
        .into_iter()
        .find(|(_, need, max)| *need > u64::from(*max))
    {
        Some((limit, need, max)) => Err(limit_error(limit, need, u64::from(max))),
        None => Ok(()),
    }
}

/// Number every specialization key densely over the pipeline and give each stage its layout.
/// Two stages declaring one key must agree on its size and default; stages that disagree were
/// compiled for another pipeline or profile (`SpecializationConflict`). A key's unread bits are
/// those no stage that declares it reads: one id carries one value to every stage.
fn specialize(
    ifaces: &[&StageInterface],
    table: &mut LinkTable,
) -> Result<Vec<(Stage, SpecializationLayout)>, Error> {
    let mut keys = BTreeMap::new();
    let mut unread: BTreeMap<SpecKey, u64> = BTreeMap::new();
    for i in ifaces {
        for s in &i.spec {
            if let Some(prev) = keys.insert(s.key, (s.size, s.default)) {
                if prev != (s.size, s.default) {
                    return Err(Error::InvalidDescriptor(
                        DescriptorError::SpecializationConflict { key: s.key },
                    ));
                }
            }
            *unread.entry(s.key).or_insert(u64::MAX) &= s.unread_bits;
        }
    }
    for (n, key) in keys.keys().enumerate() {
        let id = u32::try_from(n)
            .map_err(|_| Error::from(crate::spirv::error::BuildError::IdSpaceExhausted))?;
        table.spec.insert(*key, id);
    }
    let mut out = Vec::with_capacity(ifaces.len());
    for i in ifaces {
        let mut constants = Vec::with_capacity(i.spec.len());
        for s in &i.spec {
            let id = table.spec.get(&s.key).copied().unwrap_or_default();
            constants.push(SpecConstant {
                id,
                key: s.key,
                size: s.size,
                default: s.default,
                unread_bits: unread.get(&s.key).copied().unwrap_or(0),
            });
        }
        constants.sort_by_key(|c| c.id);
        out.push((i.stage, SpecializationLayout { constants }));
    }
    Ok(out)
}

/// The union of the stages' interfaces.
fn pipeline_interface(ifaces: &[&StageInterface]) -> PipelineInterface {
    let mut out = PipelineInterface::default();
    let mut seen = BTreeSet::new();
    for i in ifaces {
        for (slot, s) in &i.slots {
            out.slots.insert(*slot, *s);
        }
        for t in &i.threadgroup {
            if seen.insert((t.stage, t.index)) {
                out.threadgroup.push(*t);
            }
        }
        out.needs_pointer_translation |= i.needs_pointer_translation;
        out.needs_handle_translation |= i.needs_handle_translation;
        out.reaches_command_buffers |= i.reaches_command_buffers;
        out.command_writes = out.command_writes.union(i.command_writes);
        if i.writes_textures {
            out.texture_writing_stages.push(i.stage);
        }
        out.numbered_samplers
            .push((i.stage, i.numbered_samplers.clone()));
        out.push.push(i.push.clone());
    }
    out
}
