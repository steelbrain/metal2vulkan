//! The `VkPipeline`s of a render variant: one per graphics key, made when a draw first needs it and
//! memoized on the variant.
//!
//! Everything comes from the artifact's graphics state and vertex input, as the validation
//! executor's builder takes it: the vertex bindings (static strides, zero for a stride the pipeline
//! makes dynamic state, a per-bind stride from the key where the profile keeps it static),
//! divisors, the multisample, blend and dynamic states, and a tessellation pipeline's patch control
//! points. The key adds what the artifact leaves to the consumer: the specialization, the draw's
//! topology within the compiled class with the primitive restart Metal applies to it, its class's
//! list topology where the pipeline makes the topology and restart dynamic and each draw records
//! its own ([`keyed_assembly`]), the compatible render pass's formats and samples, the encoder's
//! fixed-function state the pipeline does not make dynamic, and the pass's sample positions
//! (`sample_positions.rs`). In a pass whose draws fetch through input attachments, every pipeline
//! orders its color attachment accesses by rasterization (`framebuffer_fetch.rs`). A mesh pipeline
//! has no vertex input and no input assembly; its key's topology only enables the depth bias
//! (`render/mesh.rs`). An emulated mesh function's pipeline leaves its compute stage out, which a
//! pipeline of its own runs (`objects.rs`), and draws its synthesized vertex stage's list of the
//! mesh's topology.

use super::creation_log;
use super::kept::Place;
use super::libraries::{Linked, Part};
use super::objects::{RawPipeline, VariantObjects};
use crate::degradation::BoundFormats;
use crate::error::{internal, vulkan, Error};
use crate::render_target::{PassShape, RenderPassObject};
use crate::shader_cache::driver::Making;
use crate::shader_cache::{found_in_cache, REPORTED_STAGES};
use ash::vk;
use metal2vulkan::cache::Hash256;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::metal::state::DepthStencilState;
use metal2vulkan::pipeline::{RenderPipeline, VertexStride};
use std::sync::atomic::AtomicBool;

/// What one `VkPipeline` of a render variant is made from, beside the variant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphicsKey {
    /// The plan's specialization, every stage's.
    pub(crate) specialization: SpecializationKey,
    /// The draw's topology and whether its input assembly restarts primitives.
    pub(crate) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(crate) primitive_restart: bool,
    /// The pass the pipeline draws in.
    pub(crate) pass: PassShape,
    /// The per-bind strides the pipeline keeps static (by Vulkan binding): the draw's
    /// `vertex_strides` on a profile without dynamic strides, empty with them.
    pub(crate) strides: Vec<(u32, u32)>,
    /// The encoder state the pipeline takes statically.
    pub(crate) fixed: FixedState,
    /// The pass's sample positions in sixteenths of a pixel, empty for the device's own
    /// (`sample_positions.rs`).
    pub(crate) sample_positions: Vec<[u8; 2]>,
}

/// A [`GraphicsKey`] of a draw's own values, borrowed: what a draw compares with the key its state
/// last found made, and makes a key of only where it looks one up (`begin_graphics_pipeline`).
/// Each field is the key's of the same name.
#[derive(Clone, Copy)]
pub(crate) struct DrawKey<'a> {
    pub(crate) specialization: &'a SpecializationKey,
    pub(crate) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(crate) primitive_restart: bool,
    pub(crate) pass: &'a PassShape,
    pub(crate) strides: &'a [(u32, u32)],
    pub(crate) fixed: FixedState,
    pub(crate) sample_positions: &'a [[u8; 2]],
}

impl DrawKey<'_> {
    /// Whether `key` is this one.
    pub(crate) fn is(&self, key: &GraphicsKey) -> bool {
        let GraphicsKey {
            specialization,
            topology,
            primitive_restart,
            pass,
            strides,
            fixed,
            sample_positions,
        } = key;
        self.specialization == specialization
            && self.topology == *topology
            && self.primitive_restart == *primitive_restart
            && self.pass == pass
            && self.strides == &strides[..]
            && self.fixed == *fixed
            && self.sample_positions == &sample_positions[..]
    }

    /// The key, owned.
    pub(crate) fn key(&self) -> GraphicsKey {
        GraphicsKey {
            specialization: self.specialization.clone(),
            topology: self.topology,
            primitive_restart: self.primitive_restart,
            pass: self.pass.clone(),
            strides: self.strides.to_vec(),
            fixed: self.fixed,
            sample_positions: self.sample_positions.to_vec(),
        }
    }
}

/// The input assembly a key takes for a draw of `topology` with `primitive_restart` through a
/// pipeline making `dynamic` dynamic: where the pipeline makes the topology dynamic, its class's
/// list topology, and no restart where it makes the restart dynamic, so every draw of the class
/// shares one pipeline and records its own assembly (`Recording`); else the draw's own.
pub(crate) fn keyed_assembly(
    topology: metal2vulkan::vk::PrimitiveTopology,
    primitive_restart: bool,
    dynamic: &[metal2vulkan::vk::DynamicState],
) -> (metal2vulkan::vk::PrimitiveTopology, bool) {
    use metal2vulkan::vk::{DynamicState as D, PrimitiveTopology as T};
    let topology = match topology {
        T::LINE_STRIP if dynamic.contains(&D::PRIMITIVE_TOPOLOGY) => T::LINE_LIST,
        T::TRIANGLE_STRIP if dynamic.contains(&D::PRIMITIVE_TOPOLOGY) => T::TRIANGLE_LIST,
        other => other,
    };
    let primitive_restart = primitive_restart && !dynamic.contains(&D::PRIMITIVE_RESTART_ENABLE);
    (topology, primitive_restart)
}

/// Whether a pipeline of `topology` and `polygon_mode` rasterizes lines: a line topology, or a
/// triangle topology whose fill mode draws each triangle's edges as lines
/// (`MTLTriangleFillModeLines`). A pipeline whose topology is dynamic takes its class's own
/// topology here ([`input_assembly`]), and Vulkan holds a dynamic topology to its class, so a line
/// pipeline stays one.
///
/// [`input_assembly`]: metal2vulkan::metal::state::input_assembly
fn rasterizes_lines(topology: vk::PrimitiveTopology, polygon_mode: vk::PolygonMode) -> bool {
    matches!(
        topology,
        vk::PrimitiveTopology::LINE_LIST
            | vk::PrimitiveTopology::LINE_STRIP
            | vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
            | vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY
    ) || polygon_mode == vk::PolygonMode::LINE
}

/// The topology a graphics making's record is named by (`shader_cache/driver.rs`): a line
/// topology names it as the triangle topology of its shape, so a state's line and triangle
/// pipelines of an otherwise equal key share one record; every other topology names its own.
///
/// The topology class enters no stage's code there on MoltenVK 1.4.1, which keys a converted
/// stage by its SPIR-V (`MVKShaderModule`'s key, its code's size and hash) and its conversion
/// configuration (`MVKPipelineCache::getShaderLibraryImpl`, which takes the first entry whose
/// `SPIRVToMSLConversionConfiguration::matches`); the topology reaches that configuration through
/// `enable_point_size_builtin` alone, set for the point class (`MVKGraphicsPipeline::
/// isRenderingPoints`). So a line pipeline made after its triangle sibling takes the sibling's
/// converted stages from the shared record, where a record of its own made it convert them
/// again from SPIR-V (120 ms and more for a 271k-word fragment module). Points keep their own
/// record: their stages convert otherwise. A record is a cache the driver may use or not: one
/// whose driver compiles the two classes apart holds both pipelines once both are made, which
/// costs a larger record, never a wrong pipeline.
fn record_topology(
    topology: metal2vulkan::vk::PrimitiveTopology,
) -> metal2vulkan::vk::PrimitiveTopology {
    use metal2vulkan::vk::PrimitiveTopology as T;
    match topology {
        T::LINE_LIST => T::TRIANGLE_LIST,
        T::LINE_STRIP => T::TRIANGLE_STRIP,
        T::LINE_LIST_WITH_ADJACENCY => T::TRIANGLE_LIST_WITH_ADJACENCY,
        T::LINE_STRIP_WITH_ADJACENCY => T::TRIANGLE_STRIP_WITH_ADJACENCY,
        other => other,
    }
}

/// The render encoder's state a pipeline takes as static state where it does not make it
/// dynamic; a state it makes dynamic is left at Vulkan's zero value here, so it does not split
/// the memo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FixedState {
    pub(crate) cull_mode: metal2vulkan::vk::CullModeFlags,
    pub(crate) front_face: metal2vulkan::vk::FrontFace,
    pub(crate) depth_stencil: DepthStencilState,
    /// The triangle fill mode (`setTriangleFillMode:`) and the depth clip mode
    /// (`setDepthClipMode:`, clamping when set), never dynamic.
    pub(crate) polygon_mode: metal2vulkan::vk::PolygonMode,
    pub(crate) depth_clamp: bool,
    /// The viewports and scissors the draw sets: more than one only for a pipeline that selects
    /// a viewport per primitive.
    pub(crate) viewports: u32,
    /// A draw through a viewport depth range past `[0, 1]` on a device that takes it natively
    /// (`render_state::native_depth_range`), or whose fragment function clamps the depth it
    /// writes to the range itself there: its depth clip and clamp, never dynamic.
    pub(crate) depth_range: Option<DepthRange>,
    /// A draw every fragment of which fails Metal's depth test (a compare of `Never` in a pass
    /// without a depth attachment): its fragments write no color attachment, so every
    /// attachment's write mask is empty.
    pub(crate) masks_color: bool,
    /// A draw under Apple's last provoking vertex (`setProvokingVertexMode:` 1) on a device
    /// that draws it natively (`render/provoking.rs`): the rasterization state chains
    /// `VkPipelineRasterizationProvokingVertexStateCreateInfoEXT` with
    /// `VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT`, never dynamic.
    pub(crate) provoking_last: bool,
}

/// The depth clip and clamp of a draw through a viewport depth range past `[0, 1]`
/// (`VK_EXT_depth_range_unrestricted`), with `depthClampEnable` set: whether primitives clip at
/// the near and far planes (`VK_EXT_depth_clip_enable`), and the range each depth is clamped to
/// (`VK_EXT_depth_clamp_control`'s user-defined range), as `f32` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DepthRange {
    pub(crate) clip: bool,
    pub(crate) clamp: [u32; 2],
}

/// `VkPipelineViewportDepthClampControlCreateInfoEXT`, which `ash 0.38` predates, laid out as the
/// registry defines it (extension number 583).
#[repr(C)]
struct DepthClampControlCreateInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    /// `VkDepthClampModeEXT`: `VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT` is 1.
    depth_clamp_mode: i32,
    /// `const VkDepthClampRangeEXT *`: `minDepthClamp`, `maxDepthClamp`.
    depth_clamp_range: *const [f32; 2],
}

// SAFETY: the layout is the registry's `VkPipelineViewportDepthClampControlCreateInfoEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for DepthClampControlCreateInfo {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_582_001);
}
// SAFETY: the registry lists `VkPipelineViewportStateCreateInfo` in the structure's
// `structextends`.
unsafe impl vk::ExtendsPipelineViewportStateCreateInfo for DepthClampControlCreateInfo {}

impl FixedState {
    /// `state` with what `dynamic` (the pipeline's dynamic states) covers left out, in one pass
    /// over `dynamic`.
    pub(crate) fn without_dynamic(
        state: FixedState,
        dynamic: &[metal2vulkan::vk::DynamicState],
    ) -> FixedState {
        use metal2vulkan::vk::{CompareOp, CullModeFlags, DynamicState as D, FrontFace, StencilOp};
        let mut fixed = state;
        let ds = &mut fixed.depth_stencil;
        for &s in dynamic {
            match s {
                D::CULL_MODE => fixed.cull_mode = CullModeFlags::NONE,
                D::FRONT_FACE => fixed.front_face = FrontFace::COUNTER_CLOCKWISE,
                D::DEPTH_TEST_ENABLE => ds.depth_test_enable = false,
                D::DEPTH_WRITE_ENABLE => ds.depth_write_enable = false,
                D::DEPTH_COMPARE_OP => ds.depth_compare_op = CompareOp::NEVER,
                D::STENCIL_TEST_ENABLE => ds.stencil_test_enable = false,
                D::STENCIL_OP => {
                    for face in [&mut ds.front, &mut ds.back] {
                        face.fail_op = StencilOp::KEEP;
                        face.pass_op = StencilOp::KEEP;
                        face.depth_fail_op = StencilOp::KEEP;
                        face.compare_op = CompareOp::NEVER;
                    }
                }
                D::STENCIL_COMPARE_MASK => {
                    for face in [&mut ds.front, &mut ds.back] {
                        face.compare_mask = 0;
                    }
                }
                D::STENCIL_WRITE_MASK => {
                    for face in [&mut ds.front, &mut ds.back] {
                        face.write_mask = 0;
                    }
                }
                _ => {}
            }
        }
        fixed
    }
}

/// Where a `VkPipeline`'s specialized notices are announced: always on the thread of the guest's
/// own command, in its order, never on a pool thread.
#[derive(Clone, Copy)]
pub(crate) enum Notices<'a> {
    /// After the making, by the draw or dispatch that makes it on its own thread, naming what it
    /// binds where it knows that (a replay that makes an indirect-bindable pipeline does not).
    Now(Option<BoundFormats<'a>>),
    /// Already, by the draw or dispatch that handed the making to the pool, naming what it binds.
    Announced,
    /// At the first draw that takes it (`RenderVariant::first_use`): made at its state's
    /// creation.
    FirstDraw,
}

/// The debug name of a graphics pipeline made with
/// `VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT` (a pass
/// whose draws fetch through input attachments, `framebuffer_fetch.rs`), where its device's
/// instance reports messages. The flag "forms a framebuffer-local memory dependency for each
/// fragment generated by draw commands for that pipeline", from the color attachment writes of
/// "all previous fragments (as defined by primitive order)" to the fragment's input attachment
/// reads (the Framebuffer chapter, `VkPipelineColorBlendStateCreateFlagBits`). The validation
/// layer's synchronization checks (1.4.357, and its main of 2026-10-18) do not read the flag and
/// report such a read after an earlier draw's write; the name lets a reader tell those reports
/// from others.
pub(crate) const RASTER_ORDER_PIPELINE: &std::ffi::CStr =
    c"metal2vulkan: a pipeline reading its input attachments in rasterization order";

/// What one `VkPipeline` of a render variant is made of beside the variant's modules and layout,
/// owned: each part's state, from which its create infos are built
/// ([`with_infos`](GraphicsContent::with_infos)) and its parts' libraries named
/// ([`names`](GraphicsContent::names)). Holds no device.
pub(crate) struct GraphicsContent {
    /// The pipeline's dynamic states.
    dynamic: Vec<vk::DynamicState>,
    bindings: Vec<vk::VertexInputBindingDescription>,
    divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR>,
    attributes: Vec<vk::VertexInputAttributeDescription>,
    topology: vk::PrimitiveTopology,
    primitive_restart: bool,
    /// The specialization's bytes, every stage's.
    data: Vec<u8>,
    /// The stages the pipeline runs ([`StageState`]): an emulated object or mesh function's
    /// compute stage runs in its own pipeline (`mesh_pipelines`).
    stages: Vec<StageState>,
    viewports: u32,
    /// A depth range past `[0, 1]`: its clamp range, and whether primitives clip.
    clamp_range: Option<[f32; 2]>,
    depth_clip: Option<bool>,
    rasterizer_discard: bool,
    depth_clamp: bool,
    polygon_mode: vk::PolygonMode,
    /// The last vertex provokes ([`FixedState::provoking_last`]).
    provoking_last: bool,
    /// The device rasterizes lines as Metal does on request ([`bresenham_lines`]): a pipeline
    /// that rasterizes lines chains `VkPipelineRasterizationLineStateCreateInfoEXT` with
    /// `VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT`.
    ///
    /// [`bresenham_lines`]: metal2vulkan::device::VkFeature::BresenhamLines
    bresenham_lines: bool,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    depth_bias: bool,
    samples: vk::SampleCountFlags,
    sample_shading: bool,
    alpha_to_coverage: bool,
    alpha_to_one: bool,
    sample_mask: Option<u32>,
    sample_locations: Vec<vk::SampleLocationEXT>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: vk::CompareOp,
    stencil_test: bool,
    stencil: [vk::StencilOpState; 2],
    blends: Vec<vk::PipelineColorBlendAttachmentState>,
    blend_flags: vk::PipelineColorBlendStateCreateFlags,
    /// The pipeline's logic operation (`GraphicsState::logic_op`): Metal's private
    /// `logicOperation` where Vulkan's draws it.
    logic_op: Option<vk::LogicOp>,
    patch_control_points: Option<u32>,
    /// A mesh pipeline: no vertex input and no input assembly.
    meshes: bool,
    /// The pipeline reads textures its pass renders through snapshots
    /// (`GraphicsState::snapshot_reads`, `render/snapshot.rs`): made whole, with
    /// `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`.
    feedback_loop: bool,
}

/// One stage of a [`GraphicsContent`]: its index among the variant's modules, its flags, its
/// required subgroup size, and its specialization's map: the entries of the constants its module
/// declares (`StageModule::specialization`), so a part's library holds its own stages'
/// constants alone ([`stage_entries`]).
struct StageState {
    at: usize,
    flags: vk::PipelineShaderStageCreateFlags,
    subgroup_size: Option<u32>,
    map: Vec<vk::SpecializationMapEntry>,
}

/// The entries of `specialization` whose constants `stage` declares. An entry of a constant the
/// module lacks changes nothing the pipeline does (`VkSpecializationMapEntry`), so a stage given
/// only its own makes the pipeline the whole map makes, and a part's library is named by its own
/// stages' constants (`GraphicsContent::names`): two keys that differ only in another part's
/// constants share it.
fn stage_entries(
    stage: &metal2vulkan::pipeline::StageModule,
    specialization: &SpecializationKey,
) -> Vec<metal2vulkan::vk::SpecializationMapEntry> {
    specialization
        .entries
        .iter()
        .copied()
        .filter(|entry| {
            stage
                .specialization
                .constants
                .iter()
                .any(|constant| constant.id == entry.constant_id)
        })
        .collect()
}

/// The create infos of a [`GraphicsContent`], live for one call ([`GraphicsContent::with_infos`]).
pub(crate) struct Infos<'a> {
    stages: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    pre_rasterization: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    fragment: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    vertex: &'a vk::PipelineVertexInputStateCreateInfo<'a>,
    assembly: &'a vk::PipelineInputAssemblyStateCreateInfo<'a>,
    viewport: &'a vk::PipelineViewportStateCreateInfo<'a>,
    raster: &'a vk::PipelineRasterizationStateCreateInfo<'a>,
    multisample: &'a vk::PipelineMultisampleStateCreateInfo<'a>,
    depth_stencil: &'a vk::PipelineDepthStencilStateCreateInfo<'a>,
    blend: &'a vk::PipelineColorBlendStateCreateInfo<'a>,
    dynamic: &'a vk::PipelineDynamicStateCreateInfo<'a>,
    tessellation: Option<&'a vk::PipelineTessellationStateCreateInfo<'a>>,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    meshes: bool,
    /// The whole pipeline's flags ([`GraphicsContent::feedback_loop`]).
    flags: vk::PipelineCreateFlags,
}

impl<'a> Infos<'a> {
    /// The whole pipeline's create info.
    fn whole(&self) -> vk::GraphicsPipelineCreateInfo<'a> {
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .flags(self.flags)
            .stages(self.stages)
            .viewport_state(self.viewport)
            .rasterization_state(self.raster)
            .multisample_state(self.multisample)
            .depth_stencil_state(self.depth_stencil)
            .color_blend_state(self.blend)
            .dynamic_state(self.dynamic)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(0);
        if let Some(tessellation) = self.tessellation {
            info = info.tessellation_state(tessellation);
        }
        // A mesh pipeline has no vertex input and no input assembly.
        if !self.meshes {
            info = info
                .vertex_input_state(self.vertex)
                .input_assembly_state(self.assembly);
        }
        info
    }

    /// The create info of `part`'s library, its part's state alone, as the whole one holds it,
    /// `library` chained. No library retains what a link-time-optimized link would take: the
    /// runtime links fast only (`libraries.rs`).
    fn part<'b>(
        &'b self,
        part: Part,
        library: &'b mut vk::GraphicsPipelineLibraryCreateInfoEXT<'b>,
    ) -> vk::GraphicsPipelineCreateInfo<'b> {
        let info = vk::GraphicsPipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::LIBRARY_KHR)
            .dynamic_state(self.dynamic)
            .push_next(library);
        match part {
            Part::VertexInput => info
                .vertex_input_state(self.vertex)
                .input_assembly_state(self.assembly),
            Part::PreRasterization => {
                let info = info
                    .stages(self.pre_rasterization)
                    .viewport_state(self.viewport)
                    .rasterization_state(self.raster)
                    .layout(self.layout)
                    .render_pass(self.render_pass)
                    .subpass(0);
                match self.tessellation {
                    Some(tessellation) => info.tessellation_state(tessellation),
                    None => info,
                }
            }
            Part::FragmentShader => info
                .stages(self.fragment)
                .depth_stencil_state(self.depth_stencil)
                .multisample_state(self.multisample)
                .layout(self.layout)
                .render_pass(self.render_pass)
                .subpass(0),
            Part::FragmentOutput => info
                .color_blend_state(self.blend)
                .multisample_state(self.multisample)
                .render_pass(self.render_pass)
                .subpass(0),
        }
    }
}

impl GraphicsContent {
    /// Whether its pipeline links from the four libraries: one with vertex input that rasterizes
    /// (a mesh pipeline, or one that discards every primitive, is made whole), and that reads no
    /// texture through a snapshot, whose creation flag is made whole too ([`feedback_loop`]).
    ///
    /// [`feedback_loop`]: GraphicsContent::feedback_loop
    pub(crate) fn links(&self) -> bool {
        !self.meshes && !self.rasterizer_discard && !self.feedback_loop
    }

    /// The digest of each part's state, in [`Part::ALL`]'s order: the names of its libraries
    /// (`libraries.rs`). Each is over everything that part's create info is made from, as
    /// `pipeline` (the variant's compiled pipeline) and `key` state it: the layout's description
    /// and its stages' words (`VariantObjects::parts_content`), the constants its stages declare
    /// with their values ([`stage_entries`]), the part's state, the dynamic states, and the pass
    /// where the part is made against one (all but the vertex input).
    pub(crate) fn names(
        &self,
        objects: &VariantObjects,
        pipeline: &RenderPipeline,
        key: &GraphicsKey,
    ) -> [Hash256; 4] {
        use metal2vulkan::vk::ShaderStage;
        let graphics = &pipeline.graphics;
        let content = objects.parts_content(pipeline);
        let stages = |fragment: bool| {
            pipeline
                .stages
                .iter()
                .zip(&content.stages)
                .filter(|(stage, _)| {
                    stage.stage != ShaderStage::COMPUTE
                        && (stage.stage == ShaderStage::FRAGMENT) == fragment
                })
                .map(|(stage, digest)| {
                    // The stage's own constants, each with its bytes ([`stage_entries`]).
                    let constants: Vec<(u32, &[u8])> = stage_entries(stage, &key.specialization)
                        .into_iter()
                        .map(|entry| {
                            let start = entry.offset as usize;
                            let bytes = key
                                .specialization
                                .data
                                .get(start..start + entry.size as usize)
                                .unwrap_or_default();
                            (entry.constant_id, bytes)
                        })
                        .collect();
                    (
                        *digest,
                        stage.stage_flags,
                        stage.required_subgroup_size,
                        constants,
                    )
                })
                .collect::<Vec<_>>()
        };
        let multisample = (
            graphics.rasterization_samples,
            graphics.sample_shading,
            graphics.alpha_to_coverage,
            graphics.alpha_to_one,
            graphics.sample_mask,
            &key.sample_positions,
        );
        let named = |part: Part| Making::new(part.kind()).text(&graphics.dynamic_states);
        // The vertex input part takes no pass; the others are made against it.
        let passed = |part: Part| named(part).text(&key.pass);
        [
            named(Part::VertexInput)
                .text(&(
                    &pipeline.vertex_input,
                    &key.strides,
                    key.topology,
                    key.primitive_restart,
                ))
                .finish(),
            passed(Part::PreRasterization)
                .digest(&content.layout)
                .text(&stages(false))
                .text(&(
                    key.fixed.viewports,
                    key.fixed.depth_range,
                    key.fixed.depth_clamp,
                    key.fixed.polygon_mode,
                    key.fixed.provoking_last,
                    key.fixed.cull_mode,
                    key.fixed.front_face,
                    self.depth_bias,
                    graphics.rasterizer_discard,
                    &graphics.tessellation,
                ))
                .finish(),
            passed(Part::FragmentShader)
                .digest(&content.layout)
                .text(&stages(true))
                .text(&key.fixed.depth_stencil)
                .text(&multisample)
                .finish(),
            passed(Part::FragmentOutput)
                .text(&(
                    &graphics.color_attachments,
                    key.fixed.masks_color,
                    key.pass.fetch_inputs,
                    graphics.logic_op,
                ))
                .text(&multisample)
                .finish(),
        ]
    }

    /// Hands `use_infos` the create infos of this content, of `objects`' modules and layout,
    /// against `render_pass`.
    fn with_infos<R>(
        &self,
        objects: &VariantObjects,
        render_pass: vk::RenderPass,
        use_infos: impl FnOnce(&Infos<'_>) -> R,
    ) -> R {
        let specs: Vec<vk::SpecializationInfo<'_>> = self
            .stages
            .iter()
            .map(|stage| {
                vk::SpecializationInfo::default()
                    .map_entries(&stage.map)
                    .data(&self.data)
            })
            .collect();
        let mut subgroups: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>> = self
            .stages
            .iter()
            .map(|stage| {
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                    .required_subgroup_size(stage.subgroup_size.unwrap_or(0))
            })
            .collect();
        let stages: Vec<vk::PipelineShaderStageCreateInfo<'_>> = self
            .stages
            .iter()
            .zip(&specs)
            .zip(subgroups.iter_mut())
            .map(|((stage, spec), subgroup)| {
                let object = &objects.stages[stage.at];
                let mut info = vk::PipelineShaderStageCreateInfo::default()
                    .stage(object.stage)
                    .module(object.module())
                    .name(&object.entry)
                    .flags(stage.flags);
                if !stage.map.is_empty() {
                    info = info.specialization_info(spec);
                }
                if stage.subgroup_size.is_some() {
                    info = info.push_next(subgroup);
                }
                info
            })
            .collect();
        let (fragment, pre_rasterization): (Vec<_>, Vec<_>) = stages
            .iter()
            .copied()
            .partition(|stage| stage.stage == vk::ShaderStageFlags::FRAGMENT);
        let mut divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
            .vertex_binding_divisors(&self.divisors);
        let mut vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attributes);
        if !self.divisors.is_empty() {
            vertex = vertex.push_next(&mut divisor_state);
        }
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology)
            .primitive_restart_enable(self.primitive_restart);
        let mut viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(self.viewports)
            .scissor_count(self.viewports);
        // A depth range past `[0, 1]`: the depth clamped to its range, clipped or not.
        let mut clamp_control =
            self.clamp_range
                .as_ref()
                .map(|range| DepthClampControlCreateInfo {
                    s_type: <DepthClampControlCreateInfo as vk::TaggedStructure>::STRUCTURE_TYPE,
                    p_next: std::ptr::null(),
                    depth_clamp_mode: 1,
                    depth_clamp_range: range,
                });
        if let Some(control) = clamp_control.as_mut() {
            viewport = viewport.push_next(control);
        }
        let mut depth_clip = vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default()
            .depth_clip_enable(self.depth_clip == Some(true));
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .rasterizer_discard_enable(self.rasterizer_discard)
            .depth_clamp_enable(self.depth_clamp)
            .polygon_mode(self.polygon_mode)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(self.depth_bias)
            .line_width(1.0);
        if self.depth_clip.is_some() {
            raster = raster.push_next(&mut depth_clip);
        }
        // "If this structure is not provided when creating the pipeline, the pipeline will use
        // the VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT mode" (the Vulkan specification, Flat
        // Shading): only the last vertex chains it, which `provokingVertexLast` enables
        // (`VUID-VkPipelineRasterizationProvokingVertexStateCreateInfoEXT-provokingVertexMode-04883`).
        let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default()
            .provoking_vertex_mode(vk::ProvokingVertexModeEXT::LAST_VERTEX);
        if self.provoking_last {
            raster = raster.push_next(&mut provoking);
        }
        // Metal covers the pixels whose diamond a line exits. Vulkan's default mode is the
        // device's own: on a device whose `strictLines` is set it is the rectangle of the line's
        // width, which covers both pixels either side of a pixel boundary the line runs along and
        // takes the rectangle's own fill rule at the ends, so a line's first and last pixels and
        // every boundary pixel differ from Metal's (measured on llvmpipe and on an RTX 5080,
        // which both set it; MoltenVK does not, and its lines are Metal's own). Bresenham's mode
        // is Metal's rule, and every device profile this tree holds offers it.
        let mut line = vk::PipelineRasterizationLineStateCreateInfoEXT::default()
            .line_rasterization_mode(vk::LineRasterizationModeEXT::BRESENHAM);
        let multisample_allows_bresenham =
            !self.alpha_to_coverage && !self.alpha_to_one && !self.sample_shading;
        if self.bresenham_lines
            && multisample_allows_bresenham
            && rasterizes_lines(self.topology, self.polygon_mode)
        {
            raster = raster.push_next(&mut line);
        }
        // The pass's sample positions on a grid of one pixel, each already on the sixteenths
        // every device with the extension takes exactly (`sample_positions.rs`).
        let mut sample_locations = vk::PipelineSampleLocationsStateCreateInfoEXT::default()
            .sample_locations_enable(true)
            .sample_locations_info(
                vk::SampleLocationsInfoEXT::default()
                    .sample_locations_per_pixel(self.samples)
                    .sample_location_grid_size(vk::Extent2D {
                        width: 1,
                        height: 1,
                    })
                    .sample_locations(&self.sample_locations),
            );
        let mut multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(self.samples)
            // Metal's once-per-covered-sample invocations of a function that writes memory; the
            // minimum is ignored without sample shading.
            .sample_shading_enable(self.sample_shading)
            .min_sample_shading(if self.sample_shading { 1.0 } else { 0.0 })
            .alpha_to_coverage_enable(self.alpha_to_coverage)
            .alpha_to_one_enable(self.alpha_to_one);
        // The descriptor's private sample mask, one word: no pipeline has more than 32 samples.
        let sample_mask = self.sample_mask.map(|mask| [mask]);
        if let Some(mask) = &sample_mask {
            multisample = multisample.sample_mask(mask);
        }
        if !self.sample_locations.is_empty() {
            multisample = multisample.push_next(&mut sample_locations);
        }
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(self.depth_compare)
            .stencil_test_enable(self.stencil_test)
            .front(self.stencil[0])
            .back(self.stencil[1]);
        let mut blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&self.blends)
            .flags(self.blend_flags);
        if let Some(op) = self.logic_op {
            blend = blend.logic_op_enable(true).logic_op(op);
        }
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic);
        let tessellation = self.patch_control_points.map(|points| {
            vk::PipelineTessellationStateCreateInfo::default().patch_control_points(points)
        });
        use_infos(&Infos {
            stages: &stages,
            pre_rasterization: &pre_rasterization,
            fragment: &fragment,
            vertex: &vertex,
            assembly: &assembly,
            viewport: &viewport,
            raster: &raster,
            multisample: &multisample,
            depth_stencil: &depth_stencil,
            blend: &blend,
            dynamic: &dynamic,
            tessellation: tessellation.as_ref(),
            layout: objects.layout(),
            render_pass,
            meshes: self.meshes,
            // A draw of the pipeline samples a color attachment of its subpass in the feedback
            // loop layout, which "For every image aspect accessed with feedback loops" requires:
            // "The feedback loop attachment is a color attachment, and the bound pipeline was
            // created with VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT" (the Render
            // Pass chapter, feedback loops). The device enabled `attachmentFeedbackLoopLayout`
            // wherever a draw asks for the variant (`render/snapshot.rs`).
            flags: match self.feedback_loop {
                true => vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT,
                false => vk::PipelineCreateFlags::empty(),
            },
        })
    }
}

/// The write mask and blend state of each element of the color blend state of a pipeline of
/// `graphics` made against `pass`, one per location of the draws' subpass
/// (`PassShape::subpass_colors`). An absent attachment's element is the artifact's too: identical
/// to the others on a device without `independentBlend`. A color of the pipeline past the pass's
/// has no element: the pass lacks it, and its output is dropped, as plain Metal drops it (a Metal
/// probe). A location the pipeline has no color at that references a pass color the draw keeps
/// writes nothing and blends nothing: the fragment function has no output there, whose value
/// Vulkan leaves undefined (lavapipe blended and stored one). The check of the state refuses it on
/// a device without `independentBlend` where the others differ (`render.rs`).
fn blend_elements(
    graphics: &metal2vulkan::pipeline::GraphicsState,
    pass: &PassShape,
) -> Vec<(
    metal2vulkan::vk::ColorComponentFlags,
    Option<metal2vulkan::pipeline::BlendState>,
)> {
    let elements = graphics.color_blend_attachments();
    let absent = graphics.color_attachments.iter().flatten().next().map_or(
        (metal2vulkan::vk::ColorComponentFlags::empty(), None),
        |a| (a.write_mask, a.blend),
    );
    let untouched = (metal2vulkan::vk::ColorComponentFlags::empty(), None);
    (0..pass.subpass_colors())
        .map(|location| {
            let own = graphics.color_attachments.get(location).copied().flatten();
            match own.is_none() && pass.references(location as u32) {
                true => untouched,
                false => elements.get(location).copied().unwrap_or(absent),
            }
        })
        .collect()
}

/// The color write mask of each element of the color blend state of a pipeline of `graphics`
/// made against `pass` ([`blend_elements`]), every one empty where the draw `masks_color`
/// (`FixedState::masks_color`): what a pipeline that makes the masks dynamic
/// (`GraphicsState::snapshot_reads`) takes from the draw that writes as the pipeline would
/// (`render/snapshot.rs`), one per location of the draws' subpass.
pub(crate) fn color_write_masks(
    graphics: &metal2vulkan::pipeline::GraphicsState,
    pass: &PassShape,
    masks_color: bool,
) -> Vec<vk::ColorComponentFlags> {
    blend_elements(graphics, pass)
        .into_iter()
        .map(|(write_mask, _)| match masks_color {
            true => vk::ColorComponentFlags::empty(),
            false => write_mask.into(),
        })
        .collect()
}

impl VariantObjects {
    /// The state of `pipeline`'s (the variant's compiled pipeline) `VkPipeline` for `key`
    /// ([`GraphicsContent`]): everything its making takes beside the variant's modules, layout
    /// and pass.
    pub(crate) fn graphics_content(
        &self,
        pipeline: &RenderPipeline,
        key: &GraphicsKey,
    ) -> Result<GraphicsContent, Error> {
        let graphics = &pipeline.graphics;
        if self.stages.len() != pipeline.stages.len() {
            return Err(internal("a render variant's modules are not its stages"));
        }
        let dynamic: Vec<vk::DynamicState> = graphics
            .dynamic_states
            .iter()
            .map(|&state| state.into())
            .collect();
        let dynamic_strides = dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        let input = &pipeline.vertex_input;
        let bindings = input
            .bindings
            .iter()
            .map(|b| {
                let stride = match b.stride {
                    VertexStride::Static(stride) => stride,
                    // The bind's stride is dynamic state: the static one is ignored.
                    VertexStride::PerBind if dynamic_strides => 0,
                    VertexStride::PerBind => key
                        .strides
                        .iter()
                        .find(|(binding, _)| *binding == b.binding)
                        .map(|&(_, stride)| stride)
                        .ok_or_else(|| internal("a per-bind stride the key does not hold"))?,
                };
                Ok(vk::VertexInputBindingDescription::default()
                    .binding(b.binding)
                    .stride(stride)
                    .input_rate(b.input_rate.into()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR> = input
            .bindings
            .iter()
            .filter(|b| b.divisor != 1)
            .map(|b| vk::VertexInputBindingDivisorDescriptionKHR {
                binding: b.binding,
                divisor: b.divisor,
            })
            .collect();
        let attributes: Vec<vk::VertexInputAttributeDescription> = input
            .attributes
            .iter()
            .map(|a| {
                vk::VertexInputAttributeDescription::default()
                    .location(a.location)
                    .binding(a.binding)
                    .format(a.format.into())
                    .offset(a.offset)
            })
            .collect();
        // An emulated object or mesh function's compute stage runs in its own pipeline
        // (`mesh_pipelines`), and of a pipeline's barycentric geometry stages only its draw's
        // class's runs (`RenderPipeline::runs_stage`).
        let stages = pipeline
            .stages
            .iter()
            .zip(&self.stages)
            .enumerate()
            .filter(|(at, (_, object))| {
                object.stage != vk::ShaderStageFlags::COMPUTE
                    && pipeline.runs_stage(*at, key.topology)
            })
            .map(|(at, (stage, _))| StageState {
                at,
                flags: stage.stage_flags.into(),
                subgroup_size: stage.required_subgroup_size,
                map: stage_entries(stage, &key.specialization)
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            })
            .collect();
        // Metal biases triangles only: the bias is enabled by the topology, and its factors are
        // dynamic state (`metal::state::depth_bias`).
        let bias = metal2vulkan::metal::state::depth_bias(
            key.topology,
            0.0,
            0.0,
            0.0,
            None,
            self.device.compile_device().profile(),
        )
        .map_err(|error| internal(&format!("a depth bias of no factors refused: {error:?}")))?;
        let ds = &key.fixed.depth_stencil;
        let face = |ops: metal2vulkan::metal::state::StencilOpState| {
            vk::StencilOpState::default()
                .fail_op(ops.fail_op.into())
                .pass_op(ops.pass_op.into())
                .depth_fail_op(ops.depth_fail_op.into())
                .compare_op(ops.compare_op.into())
                .compare_mask(ops.compare_mask)
                .write_mask(ops.write_mask)
        };
        // One element per location of the draws' subpass ([`blend_elements`]), each with its
        // write mask ([`color_write_masks`]).
        let masks = color_write_masks(graphics, &key.pass, key.fixed.masks_color);
        let blends: Vec<vk::PipelineColorBlendAttachmentState> =
            blend_elements(graphics, &key.pass)
                .into_iter()
                .zip(masks)
                .map(|((_, state), write_mask)| {
                    let blend = vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(write_mask);
                    match state {
                        None => blend,
                        Some(b) => blend
                            .blend_enable(true)
                            .src_color_blend_factor(b.src_color.into())
                            .dst_color_blend_factor(b.dst_color.into())
                            .color_blend_op(b.color_op.into())
                            .src_alpha_blend_factor(b.src_alpha.into())
                            .dst_alpha_blend_factor(b.dst_alpha.into())
                            .alpha_blend_op(b.alpha_op.into()),
                    }
                })
                .collect();
        // In a pass whose draws fetch through input attachments every pipeline orders its color
        // attachment accesses by rasterization (`framebuffer_fetch.rs`), so a fetch reads what an
        // earlier fragment of any pipeline wrote.
        if graphics.rasterization_order_attachment_access && !key.pass.fetch_inputs {
            return Err(internal(
                "a pipeline fetching in rasterization order outside a pass that does",
            ));
        }
        let blend_flags = match key.pass.fetch_inputs {
            true => {
                vk::PipelineColorBlendStateCreateFlags::RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT
            }
            false => vk::PipelineColorBlendStateCreateFlags::empty(),
        };
        Ok(GraphicsContent {
            dynamic,
            bindings,
            divisors,
            attributes,
            topology: key.topology.into(),
            primitive_restart: key.primitive_restart,
            data: key.specialization.data.clone(),
            stages,
            viewports: key.fixed.viewports,
            clamp_range: key
                .fixed
                .depth_range
                .map(|range| range.clamp.map(f32::from_bits)),
            depth_clip: key.fixed.depth_range.map(|range| range.clip),
            rasterizer_discard: graphics.rasterizer_discard,
            depth_clamp: key.fixed.depth_clamp,
            polygon_mode: key.fixed.polygon_mode.into(),
            provoking_last: key.fixed.provoking_last,
            bresenham_lines: self
                .device
                .compile_device()
                .profile()
                .enabled()
                .features
                .contains(&metal2vulkan::device::VkFeature::BresenhamLines),
            cull_mode: key.fixed.cull_mode.into(),
            front_face: key.fixed.front_face.into(),
            depth_bias: bias.enable,
            samples: graphics.rasterization_samples.into(),
            sample_shading: graphics.sample_shading,
            alpha_to_coverage: graphics.alpha_to_coverage,
            alpha_to_one: graphics.alpha_to_one,
            sample_mask: graphics.sample_mask,
            sample_locations: key
                .sample_positions
                .iter()
                .map(|&[x, y]| vk::SampleLocationEXT {
                    x: f32::from(x) / 16.0,
                    y: f32::from(y) / 16.0,
                })
                .collect(),
            depth_test: ds.depth_test_enable,
            depth_write: ds.depth_write_enable,
            depth_compare: ds.depth_compare_op.into(),
            stencil_test: ds.stencil_test_enable,
            stencil: [face(ds.front), face(ds.back)],
            blends,
            blend_flags,
            logic_op: graphics.logic_op.map(Into::into),
            patch_control_points: graphics.tessellation.map(|t| t.patch_control_points),
            meshes: pipeline
                .stages
                .iter()
                .any(|stage| stage.stage == metal2vulkan::vk::ShaderStage::MESH_EXT),
            feedback_loop: graphics.snapshot_reads,
        })
    }

    /// The `VkPipeline` of `pipeline` (the variant's compiled pipeline) for `key`, made against
    /// `render_pass`, a pass of `key.pass`, its specialized notices announced as `notices` says.
    /// The pass is borrowed from its holder for the call: no caller hands the driver a pass that no
    /// holder keeps until the making returns. Where the device links libraries (`libraries.rs`), it
    /// is linked from the four libraries of its parts, `claimed` where its caller claimed them
    /// already (a draw, on its own thread), else claimed here.
    pub(crate) fn graphics_pipeline(
        &self,
        pipeline: &RenderPipeline,
        key: &GraphicsKey,
        (render_pass, notices): (&RenderPassObject, Notices<'_>),
        claimed: Option<Linked>,
    ) -> Result<RawPipeline, Error> {
        let graphics = &pipeline.graphics;
        // The creation log's split, where it times this making (`creation_log.rs`).
        let split = creation_log::splitting();
        let digested = split.then(std::time::Instant::now);
        let content = self.graphics_content(pipeline, key)?;
        if let Some(at) = digested {
            creation_log::note(|| "content".to_owned(), at.elapsed());
        }
        let shared = self.device.shared();
        let libraries = shared.libraries.as_ref().filter(|_| content.links());
        // Whether the making's cache held it and each of its stages, where the making reports
        // them (`shader_cache::found_in_cache`): its stages' feedback where it has at most
        // `REPORTED_STAGES`, none where more (a count of 0 is valid, one equal to `stageCount`
        // else: VUID-VkGraphicsPipelineCreateInfo-pipelineStageCreationFeedbackCount-06594).
        let mut whole = vk::PipelineCreationFeedback::default();
        let mut stages = [vk::PipelineCreationFeedback::default(); REPORTED_STAGES];
        let mut reported_stages = 0;
        let driver_reports = shared.shader_cache.driver_reports() && libraries.is_none();
        // Its record is named by the variant's content and what this making adds to it, a line
        // topology as its triangle sibling's (`record_topology`).
        let sibling = (record_topology(key.topology) != key.topology).then(|| GraphicsKey {
            topology: record_topology(key.topology),
            ..key.clone()
        });
        let making = self
            .making("graphics")
            .text(&pipeline.vertex_input)
            .text(graphics)
            .text(sibling.as_ref().unwrap_or(key));
        // The diagnostics the device's driver is measured to report as it makes this pipeline,
        // withheld inside its creation alone (`driver_facts.rs`).
        let properties = self.device.properties();
        let expected: Vec<_> = (properties.driver_facts.iter())
            .flat_map(|fact| {
                fact.pipeline_creation_diagnostics(&graphics.dynamic_states, key.topology, &[])
            })
            .chain(crate::driver_facts::making_diagnostics(properties))
            .collect();
        // The turn first (none where a piece's thread holds one), once no twin's making is in the
        // air (`flight.rs`): the driver's time alone is timed, not the wait for a turn or a twin.
        // A pipeline linked from libraries shares its parts' makings already.
        let (turn, leader) = match libraries {
            None => {
                let (turn, leader, followed) =
                    shared.flights.board(self.flight(&key.specialization));
                if followed {
                    shared.counts.flight_followed();
                }
                (turn, Some(leader))
            }
            Some(_) => (super::budget::program_turn(), None),
        };
        let started = std::time::Instant::now();
        let (created, withheld) = crate::driver_facts::expecting(expected, || {
            content.with_infos(self, render_pass.raw, |infos| match libraries {
                Some(libraries) => {
                    let names = content.names(self, pipeline, key);
                    let linked = claimed.unwrap_or_else(|| libraries.claim(shared, &names));
                    self.linked(infos, &names, linked)
                        .map(|(raw, linked)| (raw, Some(linked)))
                }
                None => {
                    let info = infos.whole();
                    let count = info.stage_count as usize;
                    reported_stages = if count <= REPORTED_STAGES { count } else { 0 };
                    let making_at = split.then(std::time::Instant::now);
                    let _ = creation_log::take_record();
                    // With what the driver reports of the making: a pipeline found in its
                    // record's cache reads no data back (`shader_cache/driver.rs`).
                    let made = shared.make_pipelines_reported(making, |cache| {
                        let reported = &mut stages[..reported_stages];
                        let mut feedback = vk::PipelineCreationFeedbackCreateInfo::default()
                            .pipeline_creation_feedback(&mut whole)
                            .pipeline_stage_creation_feedbacks(reported);
                        let mut info = info;
                        if driver_reports {
                            info = info.push_next(&mut feedback);
                        }
                        // SAFETY: every pointer in `info` outlives the call; the pipeline
                        // cache is the making's own, destroyed once the call returned.
                        let made =
                            unsafe { shared.vk().create_graphics_pipelines(cache, &[info], None) };
                        let found = driver_reports
                            .then(|| found_in_cache(&whole, &stages[..reported_stages]))
                            .flatten();
                        (made, found)
                    });
                    if let Some(at) = making_at {
                        let record = creation_log::record_text(creation_log::take_record());
                        creation_log::note(|| format!("pipeline (made{record})"), at.elapsed());
                    }
                    made.map_err(|(_, error)| vulkan(error))
                        .and_then(|created| {
                            created
                                .first()
                                .copied()
                                .ok_or_else(|| internal("the driver made no pipeline"))
                        })
                        .map(|raw| (raw, None))
                }
            })
        });
        shared.counts.driver_diagnostics_withheld(withheld);
        let (raw, linked) = created?;
        let created_in = started.elapsed();
        if content.blend_flags.contains(
            vk::PipelineColorBlendStateCreateFlags::RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT,
        ) {
            shared.name_pipeline(raw, RASTER_ORDER_PIPELINE);
        }
        drop(leader);
        drop(turn);
        shared.counts.variant_pipeline_made();
        if driver_reports {
            shared
                .shader_cache
                .driver_reported(&whole, &stages[..reported_stages]);
        }
        let made = RawPipeline {
            device: self.device.clone(),
            raw,
            created_in,
            keep: None,
            unannounced: AtomicBool::new(matches!(notices, Notices::FirstDraw)),
            linked,
        }
        .kept_at(self, Place::Graphics(Box::new(key.clone())));
        if let Notices::Now(formats) = notices {
            self.device.degrade_specialized(
                &self.specialized,
                &key.specialization,
                formats,
                crate::degradation::Occurrence::Made,
            );
        }
        Ok(made)
    }

    /// The pipeline of `infos` linked, without link-time optimization, from `linked`, the
    /// libraries of its parts named `names`: each made here where no caller made it or is
    /// making it, through the device's shader cache under its name, else waited for.
    fn linked(
        &self,
        infos: &Infos<'_>,
        names: &[Hash256; 4],
        linked: Linked,
    ) -> Result<(vk::Pipeline, Linked), Error> {
        let shared = self.device.shared();
        let split = creation_log::splitting();
        let mut raws = [vk::Pipeline::null(); 4];
        for ((part, library), (raw, name)) in Part::ALL
            .into_iter()
            .zip(&linked.0)
            .zip(raws.iter_mut().zip(names))
        {
            // The creation log's split: how long the part took, and whether this making made it.
            let asked = split.then(std::time::Instant::now);
            let mut made_here = None;
            *raw = library.made(|| {
                let mut flag =
                    vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(part.flag());
                let info = infos.part(part, &mut flag);
                let create = |cache| {
                    // SAFETY: every pointer in `info` outlives the call; the pipeline cache is
                    // the making's own, destroyed once the call returned, or none.
                    unsafe { shared.vk().create_graphics_pipelines(cache, &[info], None) }
                };
                // A part with shaders is made through the device's shader cache, under a record
                // of its own; one without holds nothing a record would keep.
                let made = match part.holds_shaders() {
                    true => {
                        let _ = creation_log::take_record();
                        let made =
                            shared.make_pipelines(Making::new(part.kind()).digest(name), create);
                        made_here = Some(creation_log::take_record());
                        made
                    }
                    false => {
                        made_here = Some(None);
                        create(vk::PipelineCache::null())
                    }
                };
                let made = made.map_err(|(_, error)| vulkan(error))?;
                shared.counts.pipeline_library_made(part);
                made.first()
                    .copied()
                    .ok_or_else(|| internal("the driver made no pipeline library"))
            })?;
            if let Some(at) = asked {
                let how = match made_here {
                    Some(record) => format!("made{}", creation_log::record_text(record)),
                    None => "another caller's".to_owned(),
                };
                creation_log::note(|| format!("{} ({how})", part.kind()), at.elapsed());
            }
        }
        let mut libraries = vk::PipelineLibraryCreateInfoKHR::default().libraries(&raws);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .layout(infos.layout)
            .push_next(&mut libraries);
        let linking = split.then(std::time::Instant::now);
        // SAFETY: every pointer in `info` outlives the call, and each library it links is made
        // and held by `linked` for as long as the pipeline lives. A driver's making, which the exit
        // waits for longer on a pool's thread.
        let created = crate::exit_guard::driver_making(|| unsafe {
            shared
                .vk()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        });
        if let Some(at) = linking {
            creation_log::note(|| "link".to_owned(), at.elapsed());
        }
        let created = created.map_err(|(_, error)| vulkan(error))?;
        shared.counts.pipeline_linked();
        let raw = created
            .first()
            .copied()
            .ok_or_else(|| internal("the driver linked no pipeline"))?;
        Ok((raw, linked))
    }
}

#[cfg(test)]
mod tests {
    use super::FixedState;
    use metal2vulkan::metal::state::{depth_stencil_state, DepthStencilDescriptor};
    use metal2vulkan::metal::CompareFunction;
    use metal2vulkan::vk::{CullModeFlags, DynamicState, FrontFace};

    fn state(
        cull_mode: CullModeFlags,
        front_face: FrontFace,
        compare: CompareFunction,
    ) -> FixedState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = compare;
        descriptor.depth_write_enabled = true;
        FixedState {
            cull_mode,
            front_face,
            depth_stencil: depth_stencil_state(&descriptor),
            polygon_mode: metal2vulkan::vk::PolygonMode::FILL,
            depth_clamp: false,
            viewports: 1,
            depth_range: None,
            masks_color: false,
            provoking_last: false,
        }
    }

    /// What a pipeline makes dynamic does not split its memo; what it keeps static does.
    #[test]
    fn dynamic_states_leave_the_key() {
        let a = state(
            CullModeFlags::BACK,
            FrontFace::CLOCKWISE,
            CompareFunction::Less,
        );
        let b = state(
            CullModeFlags::FRONT,
            FrontFace::COUNTER_CLOCKWISE,
            CompareFunction::Greater,
        );
        assert_ne!(
            FixedState::without_dynamic(a, &[]),
            FixedState::without_dynamic(b, &[])
        );
        let all = [
            DynamicState::CULL_MODE,
            DynamicState::FRONT_FACE,
            DynamicState::DEPTH_TEST_ENABLE,
            DynamicState::DEPTH_WRITE_ENABLE,
            DynamicState::DEPTH_COMPARE_OP,
            DynamicState::STENCIL_TEST_ENABLE,
            DynamicState::STENCIL_OP,
            DynamicState::STENCIL_COMPARE_MASK,
            DynamicState::STENCIL_WRITE_MASK,
        ];
        assert_eq!(
            FixedState::without_dynamic(a, &all),
            FixedState::without_dynamic(b, &all)
        );
        // Only the cull mode dynamic: the front faces still differ.
        assert_ne!(
            FixedState::without_dynamic(a, &[DynamicState::CULL_MODE]),
            FixedState::without_dynamic(b, &[DynamicState::CULL_MODE])
        );
        let c = state(
            CullModeFlags::FRONT,
            FrontFace::CLOCKWISE,
            CompareFunction::Less,
        );
        assert_eq!(
            FixedState::without_dynamic(a, &[DynamicState::CULL_MODE]),
            FixedState::without_dynamic(c, &[DynamicState::CULL_MODE])
        );
    }

    /// A line topology names its making's record as the triangle topology of its shape; the
    /// point, triangle and patch topologies name their own (`record_topology`).
    #[test]
    fn a_line_topology_names_its_triangle_sibling_s_record() {
        use metal2vulkan::vk::PrimitiveTopology as T;
        for (topology, named) in [
            (T::POINT_LIST, T::POINT_LIST),
            (T::LINE_LIST, T::TRIANGLE_LIST),
            (T::LINE_STRIP, T::TRIANGLE_STRIP),
            (T::TRIANGLE_LIST, T::TRIANGLE_LIST),
            (T::TRIANGLE_STRIP, T::TRIANGLE_STRIP),
            (T::TRIANGLE_FAN, T::TRIANGLE_FAN),
            (T::LINE_LIST_WITH_ADJACENCY, T::TRIANGLE_LIST_WITH_ADJACENCY),
            (
                T::LINE_STRIP_WITH_ADJACENCY,
                T::TRIANGLE_STRIP_WITH_ADJACENCY,
            ),
            (
                T::TRIANGLE_LIST_WITH_ADJACENCY,
                T::TRIANGLE_LIST_WITH_ADJACENCY,
            ),
            (
                T::TRIANGLE_STRIP_WITH_ADJACENCY,
                T::TRIANGLE_STRIP_WITH_ADJACENCY,
            ),
            (T::PATCH_LIST, T::PATCH_LIST),
        ] {
            assert_eq!(super::record_topology(topology), named, "{topology:?}");
        }
    }

    /// Where a pipeline makes the topology and restart dynamic, every Metal primitive type keys
    /// its class's list topology without restart, so a strip and its list share a key; where it
    /// keeps them static, each type keys its own topology and Metal's restart for it.
    #[test]
    fn a_dynamic_topology_keys_its_class_list() {
        use metal2vulkan::metal::state::input_assembly;
        use metal2vulkan::metal::PrimitiveType;
        use metal2vulkan::vk::PrimitiveTopology as T;
        let dynamic = [
            DynamicState::PRIMITIVE_TOPOLOGY,
            DynamicState::PRIMITIVE_RESTART_ENABLE,
        ];
        for &ty in PrimitiveType::ALL {
            let (topology, restart) = input_assembly(ty);
            let class_list = match ty {
                PrimitiveType::Point => T::POINT_LIST,
                PrimitiveType::Line | PrimitiveType::LineStrip => T::LINE_LIST,
                PrimitiveType::Triangle | PrimitiveType::TriangleStrip => T::TRIANGLE_LIST,
            };
            assert_eq!(
                super::keyed_assembly(topology, restart, &dynamic),
                (class_list, false),
                "{ty:?}"
            );
            assert_eq!(
                super::keyed_assembly(topology, restart, &[DynamicState::CULL_MODE]),
                (topology, restart),
                "{ty:?}"
            );
        }
    }
}
