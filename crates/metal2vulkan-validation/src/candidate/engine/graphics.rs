//! Render cases of the engine candidate.
//!
//! A render case runs through the product's consumer path as one `RenderPipeline`:
//! `Library` → `Function` → `compile_render_pipeline` with the Metal oracle's descriptor
//! ([`RenderTemplate`], compiled in the tools' compile worker as
//! `acceptance::PipelineRequest::VertexDraw`, or as `PipelineRequest::Draw` with both stages'
//! libraries), then `RenderPipeline::plan` and `BindingPlan::instantiate_draw`. This module is the
//! device-free half (case mapping, the bind state, the descriptor, and output selection); the
//! Vulkan half is `platform::render`.
//!
//! What an authored case maps to ([`render_case`]):
//!
//! - a vertex case without a vertex observation is the Metal oracle's rasterization-disabled
//!   draw: the vertex function alone, its stage-in attributes fed by vertex buffers at the
//!   oracle's slots ([`vertex_input_slots`]), the oracle's 1×1 `R8Unorm` color attachment as API
//!   scaffolding, and the case's buffers, textures (texture-array elements at their slots),
//!   samplers, visible-function tables ([`super::tables`]), argument-buffer buffers
//!   ([`super::argument_buffers`]), and argument-buffer textures and command buffers bound to the
//!   vertex stage;
//! - a fragment case draws through its generated passthrough vertex companion, with the case's
//!   resources bound to the fragment stage and its render targets attached;
//! - a vertex case observed through its position or a varying draws into its generated observer
//!   fragment companion, which writes the value to render target 0;
//! - a vertex case of a post-tessellation vertex function draws its patches as the Metal oracle
//!   does (`executor_contract::tessellation`): the pipeline's tessellation state (the case's
//!   partition mode and output winding), its control points and patch attributes fed per patch
//!   control point and per patch at the oracle's slots, its factors in one buffer, the case's cull
//!   mode, and one patch draw through its observer, or with rasterization disabled and no observer
//!   when the case has no vertex observation;
//! - a fragment case's explicit imageblock is the pass's cells as the Metal oracle's initializer
//!   defines them, each authored plane at its offset in the checked reflection's master
//!   ([`super::imageblock`]), and an imageblock output is the oracle resolver's plane of the
//!   selected member.
//!
//! The companion is the checker's ([`crate::companion`], derived from the entry's AIR): the Metal
//! oracle draws with the same generated module, so both sides run the same companion semantics.
//! [`RenderCase`] itself carries both stages, so owned fixtures also execute vertex and fragment
//! functions of two libraries through one pipeline with derived expectations.

use super::argument_buffers::{self, FieldKey};
use super::imageblock::{self, Cells};
use super::tables::{function_tables, FunctionTable};
use super::{
    function_constants, require_case_families, resources, select_range, views_told_apart,
    BufferAddress,
};
use crate::air::reflect::FragmentImageblock;
use crate::case::indirect_commands::CommandBufferResource;
use crate::case::out_of_bounds::OutOfBoundsTarget;
use crate::case::{
    ArgumentBufferSamplerResource, AttributeFormat, AuthoredCase, OutputSelection, Primitive,
    SamplerResource, Stage, TextureFormat,
};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::check::interface::AirInterface;
use crate::companion::Companion;
use crate::coverage::RateMapPassRun;
use crate::executor_contract::buffer_slots::vertex_input_slots;
use crate::executor_contract::multisample::AttachmentTransfer;
use crate::executor_contract::render_target_array::render_target_array_length;
use crate::executor_contract::tessellation as patches;
use crate::executor_contract::DrawDepthStencil;
use crate::literal::{LiteralDepthStencil, LiteralResources, LiteralStageInput, LiteralTexture};
use metal2vulkan::bind::{
    BindShape, BindState, BoundBuffer, DrawCall, IndexBufferRef, RasterizationRateMap,
    RateMapRegions, RegionLimits, StageTable, TessellationFactorBuffer, TessellationFactors,
};
use metal2vulkan::error::{DescriptorError, Error, MetalAnswer};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::metal::{self, RenderStages};
use metal2vulkan::pipeline::Stage as ShaderStage;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipeline, RenderPipelineDescriptor,
    StageModule, StageSource, TessellationDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use std::collections::BTreeMap;

/// The Metal buffer slots are `0..31`.
const BUFFER_SLOTS: u32 = 31;

/// What one stage binds, by Metal index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageResources {
    /// Initial bytes of each bound buffer. An argument buffer's owner covers each of its pointer
    /// fields ([`argument_buffers::embed`]).
    pub buffers: BTreeMap<u32, Vec<u8>>,
    /// Initial bytes of each argument-buffer buffer, by `(owner Metal index, field offset)`: a
    /// used resource whose guest address its owner holds at the field.
    pub argument_buffer_buffers: BTreeMap<FieldKey, Vec<u8>>,
    /// Each argument-buffer texture with its initial contents, by `(owner Metal index, field
    /// offset)`: a used texture whose resource id its owner holds at the field, a heap member.
    pub argument_buffer_textures: BTreeMap<FieldKey, LiteralTexture>,
    /// Each argument-buffer sampler, by `(owner Metal index, field offset)`: an argument sampler
    /// whose id its owner holds at the field.
    pub argument_buffer_samplers: BTreeMap<FieldKey, ArgumentBufferSamplerResource>,
    /// Each pipeline state, by `(owner Metal index, field offset)`: the case's token for it, the
    /// word its owner holds at the field ([`argument_buffers::embed_objects`]).
    pub objects: BTreeMap<FieldKey, u64>,
    /// Each command buffer, by `(owner Metal index, field offset)`: its record storage is the
    /// argument-buffer buffer at the key, at its ID's records address, and its owner holds its ID
    /// ([`records_ids`]); the run checks its final records
    /// ([`argument_buffers::validate_records`]).
    pub command_buffers: BTreeMap<FieldKey, CommandBufferResource>,
    /// Each bound texture (texture buffers and texture-array elements included, each element at
    /// its own slot, [`resources::bound_textures`]) with its initial contents.
    pub textures: BTreeMap<u32, LiteralTexture>,
    pub samplers: BTreeMap<u32, SamplerResource>,
    /// Each visible-function table, by Metal buffer index: a buffer of its table words.
    pub function_tables: BTreeMap<u32, FunctionTable>,
}

/// One stage's function and what it binds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderFunction {
    pub entry: String,
    pub constants: metal2vulkan::library::FunctionConstantValues,
    pub resources: StageResources,
}

/// A vertex attribute and the buffer feeding it: attribute `location` of `format` at offset 0 of
/// Metal vertex buffer `buffer_index`, `stride` bytes per element, stepped by `step` at rate 1
/// (the Metal oracle's vertex descriptor: a vertex input per vertex, a post-tessellation vertex
/// function's control point per patch control point and patch attribute per patch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexBuffer {
    pub location: u32,
    pub buffer_index: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    /// The layout's stride is `MTLBufferLayoutStrideDynamic`: the buffer is bound with the
    /// stride form at `stride`, as the oracle binds it.
    pub dynamic: bool,
    pub step: metal::VertexStepFunction,
    pub bytes: Vec<u8>,
}

/// A color attachment: its format, extent, and initial contents (row-major, tightly packed),
/// loaded before the draw and stored after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorTarget {
    pub format: TextureFormat,
    pub dimensions: [u32; 2],
    /// The samples of each pixel, in order in `bytes`.
    pub samples: u32,
    pub bytes: Vec<u8>,
}

/// The observed bytes of a render run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderOutput {
    None,
    Buffer {
        stage: ShaderStage,
        binding: u32,
        offset: u64,
        length: u64,
    },
    /// A region of a texture of `stage`, as [`resources::select_texture`] reads it (a
    /// texture-array element's is the texture at its slot).
    Texture {
        stage: ShaderStage,
        binding: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    /// A region of the argument-buffer texture at `field_offset` of the buffer bound at Metal
    /// index `owner` of `stage`, as [`resources::select_texture`] reads it.
    ArgumentBufferTexture {
        stage: ShaderStage,
        owner: u32,
        field_offset: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    /// `length` bytes at `offset` of the argument-buffer buffer at `field_offset` of the buffer
    /// bound at Metal index `owner` of `stage`.
    ArgumentBufferBuffer {
        stage: ShaderStage,
        owner: u32,
        field_offset: u32,
        offset: u64,
        length: u64,
    },
    /// A region of color attachment `index`.
    RenderTarget {
        index: u32,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    /// A region of the depth attachment's depth values (little-endian `f32`s).
    Depth {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    /// A region of the stencil attachment's values (one byte each).
    Stencil {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    /// A region of the plane of one member of the fragment imageblock's master, at `member`'s
    /// offset in each pixel's cell.
    Imageblock {
        member: imageblock::Member,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
}

/// What a draw assembles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Assembly {
    /// Primitives of this type, within the topology class the pipeline is compiled for.
    Primitives(Primitive),
    /// The patches of a tessellation pipeline: the pipeline's tessellation state, and the factor
    /// buffer's bytes, bound at `executor_contract::tessellation`'s offset and instance stride with
    /// Metal's default scale.
    Patches {
        tessellation: TessellationDescriptor,
        factors: Vec<u8>,
    },
}

/// One draw through one render pipeline, in the terms the binder takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderCase {
    pub vertex: RenderFunction,
    /// `None`: rasterization is disabled (Metal's `rasterizationEnabled = NO`).
    pub fragment: Option<RenderFunction>,
    pub vertex_buffers: Vec<VertexBuffer>,
    /// Color attachments by Metal index.
    pub color_targets: BTreeMap<u32, ColorTarget>,
    /// The depth/stencil attachment of a fragment case: each authored aspect, loaded before the
    /// draw and stored after it (Metal's separate `Depth32Float` and `Stencil8` attachments).
    pub depth_stencil: Option<LiteralDepthStencil>,
    /// The encoder's depth-stencil state, stencil references, and front-facing winding: the Metal
    /// oracle's, a fragment case's authored state or the one its function's live outputs imply
    /// ([`DrawDepthStencil::of_case`]); a vertex case's is the default (no test).
    pub depth_stencil_state: DrawDepthStencil,
    /// The render pass's `renderTargetArrayLength`: the Metal oracle's, 1 when the case's own
    /// function's stage metadata declares its render target slice and 0 otherwise
    /// ([`render_target_array_length`]), bound as `BindState::render_target_array_length`.
    pub render_target_array_length: u32,
    pub assembly: Assembly,
    pub draw: DrawCall,
    /// An indexed draw's index buffer bytes: `draw` is a `DrawCall::Indexed` of the buffer
    /// [`INDEX_BUFFER_TOKEN`] names; `None` for every other draw.
    pub index_buffer: Option<Vec<u8>>,
    /// The case's out-of-bounds targets (attribute buffers and the index buffer), which the
    /// executor allocates inside guards and reports the containment of.
    pub out_of_bounds: Vec<OutOfBoundsTarget>,
    /// The encoder's cull mode: the Metal oracle's, a tessellation case's authored one
    /// (`executor_contract::tessellation::cull_mode`), and otherwise Metal's default, none. The
    /// front face is [`RenderCase::depth_stencil_state`]'s winding.
    pub cull_mode: metal::CullMode,
    /// A fragment case's explicit imageblock: the pass's cells when the case's draw begins, as
    /// the Metal oracle's initializer leaves them ([`Cells::initial`]). The render executor holds
    /// them in the storage image the plan's `WriteSource::Imageblock` names.
    pub imageblock: Option<Cells>,
    pub output: RenderOutput,
    /// The generated stage that completes the case: `vertex` is its passthrough vertex function, or
    /// `fragment` its observer fragment function. `None` for a rasterization-disabled vertex case
    /// and for owned two-stage fixtures.
    pub companion: Option<Companion>,
    /// The render pass's rasterization-rate map, the product's compile of it (Apple M3's rule),
    /// whose parameter data every map literal of the case binds (`RateMapLiteral::parameter_data`):
    /// the render executor draws the pass region by region into its physical size
    /// ([`rate_map_regions`]), on every device.
    pub rate_map_pass: Option<RasterizationRateMap>,
    /// The render pass's sample count, the case's: `None` for one sample, as every case drew before
    /// the count was authorable; the pipeline's `rasterSampleCount`
    /// ([`RenderTemplate::raster_sample_count`]), and every attachment holds that many samples
    /// of each pixel ([`RenderCase::attachment_transfer`]).
    pub raster_sample_count: Option<u32>,
}

impl RenderCase {
    /// How the executor fills and reads the pass's attachments: by copies for one sample, sample by
    /// sample for more.
    pub fn attachment_transfer(&self) -> AttachmentTransfer {
        AttachmentTransfer::of(self.raster_sample_count.unwrap_or(1))
    }

    /// Each stage the pipeline has, with its function.
    pub fn stages(&self) -> impl Iterator<Item = (ShaderStage, &RenderFunction)> {
        std::iter::once((ShaderStage::Vertex, &self.vertex)).chain(
            self.fragment
                .iter()
                .map(|fragment| (ShaderStage::Fragment, fragment)),
        )
    }

    /// The function of `stage`, when the pipeline has that stage.
    pub fn function(&self, stage: ShaderStage) -> Option<&RenderFunction> {
        self.stages()
            .find(|(of, _)| *of == stage)
            .map(|(_, function)| function)
    }

    /// Each argument-buffer buffer of each stage, by `(stage, owner, field offset)`, in stage
    /// order: its position is its consumer token in the bind state ([`bind_state`]).
    pub fn argument_buffer_buffers(
        &self,
    ) -> impl Iterator<Item = ((ShaderStage, u32, u32), &Vec<u8>)> {
        self.stages().flat_map(|(stage, function)| {
            function
                .resources
                .argument_buffer_buffers
                .iter()
                .map(move |(&(owner, field), bytes)| ((stage, owner, field), bytes))
        })
    }

    /// Each argument-buffer texture of each stage, by `(stage, owner, field offset)`, in stage
    /// order ([`texture_tokens`] gives its consumer token).
    pub fn argument_buffer_textures(
        &self,
    ) -> impl Iterator<Item = ((ShaderStage, u32, u32), &LiteralTexture)> {
        self.stages().flat_map(|(stage, function)| {
            function
                .resources
                .argument_buffer_textures
                .iter()
                .map(move |(&(owner, field), texture)| ((stage, owner, field), texture))
        })
    }

    /// Each argument-buffer sampler of each stage, by `(stage, owner, field offset)`, in stage
    /// order ([`sampler_ids`] gives its id).
    pub fn argument_buffer_samplers(
        &self,
    ) -> impl Iterator<Item = ((ShaderStage, u32, u32), &ArgumentBufferSamplerResource)> {
        self.stages().flat_map(|(stage, function)| {
            function
                .resources
                .argument_buffer_samplers
                .iter()
                .map(move |(&(owner, field), sampler)| ((stage, owner, field), sampler))
        })
    }

    /// Each pipeline state of each stage with its token, by `(stage, owner, field offset)`, in
    /// stage order.
    pub fn objects(&self) -> impl Iterator<Item = ((ShaderStage, u32, u32), u64)> + '_ {
        self.stages().flat_map(|(stage, function)| {
            function
                .resources
                .objects
                .iter()
                .map(move |(&(owner, field), &word)| ((stage, owner, field), word))
        })
    }

    /// Each command buffer of each stage, by `(stage, owner, field offset)`, in stage order: its
    /// record storage is the argument-buffer buffer at the same key.
    pub fn command_buffers(
        &self,
    ) -> impl Iterator<Item = ((ShaderStage, u32, u32), &CommandBufferResource)> {
        self.stages().flat_map(|(stage, function)| {
            function
                .resources
                .command_buffers
                .iter()
                .map(move |(&(owner, field), command)| ((stage, owner, field), command))
        })
    }

    /// The stage of the generated companion's function, which links nothing: the vertex stage of a
    /// passthrough vertex, the fragment stage of an observer fragment, none without a companion
    /// (a render case's is never a mesh render pipeline's stages, which [`render_case`] refuses).
    pub fn companion_stage(&self) -> Option<ShaderStage> {
        self.companion
            .as_ref()
            .and_then(|companion| match companion {
                Companion::PassthroughVertex(_) => Some(ShaderStage::Vertex),
                Companion::ObserverFragment(_) => Some(ShaderStage::Fragment),
                Companion::MeshStages(_) => None,
            })
    }
}

/// The vertex buffers `case` draws with (its literals `literal`, its entry's AIR-layer interface
/// `interface`): each vertex input, and a post-tessellation vertex function's control points and
/// patch attributes, at the oracle's slot ([`vertex_input_slots`]) with its format and stride,
/// stepped as the oracle's vertex descriptor steps it (a vertex input per vertex, a control point
/// per patch control point, a patch attribute per patch). The one derivation behind
/// [`render_case`] and the tests' compile of a vertex case.
pub fn vertex_buffers(
    case: &AuthoredCase,
    literal: &LiteralResources,
    interface: &AirInterface,
) -> Result<Vec<VertexBuffer>, NotSupported> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let slots = vertex_input_slots(case, interface)
        .map_err(|error| gap(&format!("vertex buffer slots: {error}")))?;
    fn stepped(
        inputs: &[LiteralStageInput],
        step: metal::VertexStepFunction,
    ) -> impl Iterator<Item = (&LiteralStageInput, metal::VertexStepFunction)> {
        inputs.iter().map(move |input| (input, step))
    }
    let tessellated = literal.tessellation.iter().flat_map(|tessellation| {
        stepped(
            &tessellation.control_points,
            metal::VertexStepFunction::PerPatchControlPoint,
        )
        .chain(stepped(
            &tessellation.patch_inputs,
            metal::VertexStepFunction::PerPatch,
        ))
    });
    stepped(&literal.vertex_inputs, metal::VertexStepFunction::PerVertex)
        .chain(tessellated)
        .map(|(input, step)| {
            let buffer_index = *slots.get(&input.location).ok_or_else(|| {
                gap(&format!(
                    "no vertex buffer slot for attribute {}",
                    input.location
                ))
            })?;
            Ok(VertexBuffer {
                location: input.location,
                buffer_index,
                format: input.format,
                stride: input.stride,
                dynamic: input.dynamic_stride,
                step,
                bytes: input.bytes.clone(),
            })
        })
        .collect()
}

/// Map `case` onto what the render executor runs, or name what it does not implement yet: the
/// families of the shared vocabulary that stop it ([`require_case_families`], among them device
/// buffer arrays, acceleration structures and intersection function tables, a vertex case's
/// depth/stencil attachment, and a rasterization-disabled vertex case's render targets) are
/// `NotSupported::Requirements`, and a case whose literals do not map is
/// `NotSupported::NotImplemented`. A tessellation case draws its patches as the oracle does
/// ([`patch_draw`]). `interface` is the checked entry's AIR-layer interface, whose buffer slots the
/// vertex buffers avoid as the Metal oracle's do ([`vertex_input_slots`]); `air_ll` is the text of
/// the case's module, whose stage metadata sets the pass's render target array length as the
/// oracle's does ([`render_target_array_length`]); `companion` is the checked entry's generated
/// companion ([`crate::check::CheckedShader::companion`]), which a fragment case and an observed
/// vertex case draw with. A companion that does not complete the case's stage is a failure: the
/// checker derives exactly one for exactly those cases. `fragment_imageblock` is the checked
/// reflection's (`CheckedShader::reflection`), whose master a fragment case's imageblock cells and
/// output follow, as the checker's and the oracle's do ([`imageblock::layout`]).
pub fn render_case(
    case: &AuthoredCase,
    interface: &AirInterface,
    air_ll: &str,
    companion: Option<&Companion>,
    fragment_imageblock: Option<&FragmentImageblock>,
) -> Result<RenderCase, ExecutorError> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let generated = |companion: &Companion| RenderFunction {
        entry: companion.entry().into(),
        constants: FunctionConstantValues::defaults(),
        resources: StageResources::default(),
    };
    // The stage the case's own function and resources are, and the companion's function.
    let (stage, generated) = match (case.stage, case.vertex_observation, companion) {
        (Stage::Kernel, ..) => return Err(gap("a kernel case is not a render case").into()),
        (Stage::Vertex, None, None) => (ShaderStage::Vertex, None),
        (Stage::Vertex, Some(_), Some(observer @ Companion::ObserverFragment(_))) => {
            (ShaderStage::Vertex, Some(generated(observer)))
        }
        (Stage::Fragment, _, Some(passthrough @ Companion::PassthroughVertex(_))) => {
            (ShaderStage::Fragment, Some(generated(passthrough)))
        }
        (stage, observation, companion) => {
            return Err(ExecutorError::Failed(format!(
                "a {stage:?} case with vertex observation {observation:?} has companion {:?}",
                companion.map(Companion::entry)
            )))
        }
    };
    require_case_families(case)?;
    for (present, what) in [
        (
            !case.threadgroup_memory.is_empty(),
            "threadgroup memory in a render case",
        ),
        (case.dispatch.is_some(), "a dispatch in a render case"),
    ] {
        if present {
            return Err(gap(what).into());
        }
    }
    let mut index_buffer = None;
    let (assembly, draw) = match (&case.draw, &case.tessellation) {
        (Some(draw), None) => match &draw.index_buffer {
            None => (
                Assembly::Primitives(draw.primitive),
                DrawCall::Primitives {
                    vertex_start: draw.vertex_start,
                    vertex_count: draw.vertex_count,
                    instance_count: draw.instance_count,
                    base_instance: draw.base_instance,
                },
            ),
            // The oracle's `drawIndexedPrimitives` of `vertex_count` indices from the buffer's
            // offset, with the draw's base vertex and base instance.
            Some(literal) => {
                let bytes = literal
                    .bytes()
                    .ok_or_else(|| gap("index buffer bytes that are not base64"))?;
                let call = DrawCall::Indexed {
                    index_type: index_type(literal.index_type),
                    index_buffer: IndexBufferRef::new(INDEX_BUFFER_TOKEN, bytes.len() as u64),
                    index_count: draw.vertex_count,
                    index_offset: u64::from(literal.offset),
                    instance_count: draw.instance_count,
                    base_vertex: draw.base_vertex,
                    base_instance: draw.base_instance,
                };
                index_buffer = Some(bytes);
                (Assembly::Primitives(draw.primitive), call)
            }
        },
        (None, Some(tessellation)) => {
            patch_draw(tessellation, interface).map_err(|error| gap(&error))?
        }
        _ => return Err(gap("a render case without exactly one draw").into()),
    };
    // The reflected master layout of the function's imageblock, which its authored planes and
    // imageblock output name members of.
    let imageblock_layout = match (stage, fragment_imageblock) {
        (ShaderStage::Fragment, Some(reflected)) => {
            Some(imageblock::layout(reflected).map_err(|error| gap(&error))?)
        }
        _ => None,
    };
    let output = match stage_output(&case.output, stage).map_err(|error| gap(&error))? {
        Some(output) => output,
        None => match case.output {
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } if generated.is_some() => RenderOutput::RenderTarget {
                index,
                origin,
                dimensions,
            },
            OutputSelection::Depth { origin, dimensions } if stage == ShaderStage::Fragment => {
                RenderOutput::Depth { origin, dimensions }
            }
            OutputSelection::Stencil { origin, dimensions } if stage == ShaderStage::Fragment => {
                RenderOutput::Stencil { origin, dimensions }
            }
            OutputSelection::FragmentImageblock {
                ref semantic,
                origin,
                dimensions,
            } if stage == ShaderStage::Fragment => {
                let member = imageblock_layout
                    .as_ref()
                    .and_then(|layout| layout.members.get(semantic))
                    .copied()
                    .ok_or_else(|| {
                        gap(&format!(
                            "imageblock output {semantic}, which no master member of the \
                             function's views names"
                        ))
                    })?;
                RenderOutput::Imageblock {
                    member,
                    origin,
                    dimensions,
                }
            }
            _ => return Err(gap(
                "a render output other than a buffer or argument-buffer buffer range, a region of \
                 a texture, a texture-array element, or an argument-buffer texture, or a region of \
                 a rasterized render target, of a fragment case's depth/stencil attachment, or of \
                 its imageblock",
            )
            .into()),
        },
    };
    let literal =
        LiteralResources::prepare(case).map_err(|error| gap(&format!("resources: {error}")))?;
    let resources = stage_resources(case, &literal)?;
    if stage == ShaderStage::Fragment && !literal.vertex_inputs.is_empty() {
        return Err(gap("vertex inputs of a fragment case").into());
    }
    let vertex_buffers = vertex_buffers(case, &literal, interface)?;
    let color_targets = color_targets(&literal, generated.is_some());
    let imageblock = match (&literal.fragment_imageblock, &imageblock_layout) {
        (None, _) => None,
        (Some(authored), Some(layout)) => {
            Some(Cells::initial(layout, authored).map_err(|error| gap(&error))?)
        }
        (Some(_), None) => {
            return Err(gap("an authored imageblock of a function that reflects none").into())
        }
    };
    let own = RenderFunction {
        entry: case.entry.clone(),
        constants: function_constants(&literal)?,
        resources,
    };
    let (vertex, fragment) = match (stage, generated) {
        (ShaderStage::Fragment, Some(passthrough)) => (passthrough, Some(own)),
        (_, observer) => (own, observer),
    };
    // The oracle attaches depth and stencil only to a fragment case's pass (refused above
    // otherwise), and sets the case's authored state or the one its own fragment function
    // implies.
    let depth_stencil_state = match stage {
        ShaderStage::Fragment => DrawDepthStencil::of_case(
            case.depth_stencil
                .as_ref()
                .and_then(|attachment| attachment.state.as_ref()),
            interface,
        ),
        _ => DrawDepthStencil::default(),
    };
    Ok(RenderCase {
        vertex,
        fragment,
        vertex_buffers,
        color_targets,
        depth_stencil: literal.depth_stencil.clone(),
        depth_stencil_state,
        render_target_array_length: render_target_array_length(case.stage, air_ll),
        assembly,
        draw,
        index_buffer,
        out_of_bounds: case
            .out_of_bounds
            .as_ref()
            .map(|declaration| declaration.targets.clone())
            .unwrap_or_default(),
        cull_mode: case
            .tessellation
            .as_ref()
            .map_or(metal::CullMode::None, patches::cull_mode),
        imageblock,
        output,
        companion: companion.cloned(),
        rate_map_pass: case
            .render_pass_rate_map
            .as_ref()
            .map(|map| map.compiled())
            .transpose()
            .map_err(|error| gap(&error))?,
        raster_sample_count: case.raster_sample_count,
    })
}

/// The color attachments of a pass that rasterizes (`rasterizes`): the case's render targets, by
/// Metal index; or, without rasterization, the oracle's private 1×1 `R8Unorm` sink, since Metal
/// needs an attachment to encode a render pass and nothing rasterizes into it.
pub fn color_targets(literal: &LiteralResources, rasterizes: bool) -> BTreeMap<u32, ColorTarget> {
    if !rasterizes {
        return BTreeMap::from([(
            0,
            ColorTarget {
                format: TextureFormat::R8Unorm,
                dimensions: [1, 1],
                samples: 1,
                bytes: vec![0],
            },
        )]);
    }
    literal
        .render_targets
        .iter()
        .map(|target| {
            (
                target.index,
                ColorTarget {
                    format: target.format,
                    dimensions: target.dimensions,
                    samples: target.samples,
                    bytes: target.bytes.clone(),
                },
            )
        })
        .collect()
}

/// The output `output` selects of the case's own function at `stage`, when it is a resource of that
/// function's (a buffer range, a region of a texture or a texture-array element, an argument
/// buffer's buffer range or texture region) or no output; `None` for an output of the pass (a
/// render target, the depth/stencil attachment, an imageblock). The one mapping a render case's
/// and a mesh case's own stage share.
pub fn stage_output(
    output: &OutputSelection,
    stage: ShaderStage,
) -> Result<Option<RenderOutput>, String> {
    Ok(Some(match *output {
        OutputSelection::None => RenderOutput::None,
        OutputSelection::Buffer {
            binding,
            offset,
            length,
        } => RenderOutput::Buffer {
            stage,
            binding,
            offset,
            length,
        },
        OutputSelection::Texture {
            binding,
            origin,
            dimensions,
        } => RenderOutput::Texture {
            stage,
            binding,
            origin,
            dimensions,
        },
        // The element is the texture bound at its slot (`resources::bound_textures`).
        OutputSelection::TextureArrayElement {
            binding,
            element,
            origin,
            dimensions,
        } => RenderOutput::Texture {
            stage,
            binding: resources::texture_array_slot(binding, element)
                .ok_or("a texture-array element slot overflows")?,
            origin,
            dimensions,
        },
        OutputSelection::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
            offset,
            length,
        } => RenderOutput::ArgumentBufferBuffer {
            stage,
            owner: buffer_binding,
            field_offset,
            offset,
            length,
        },
        OutputSelection::ArgumentBufferTexture {
            buffer_binding,
            field_offset,
            origin,
            dimensions,
        } => RenderOutput::ArgumentBufferTexture {
            stage,
            owner: buffer_binding,
            field_offset,
            origin,
            dimensions,
        },
        _ => return Ok(None),
    }))
}

/// What the case's own function binds, from its literals `literal`: its buffers (an argument
/// buffer's owner covering each pointer field), its argument buffers' members, the pipeline states
/// and command buffers its argument buffers hold, its textures (texture-array elements each at its
/// own slot), samplers and visible-function tables. The one mapping a render case's and a mesh
/// case's own stage share.
pub fn stage_resources(
    case: &AuthoredCase,
    literal: &LiteralResources,
) -> Result<StageResources, ExecutorError> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let mut resources = StageResources::default();
    for buffer in &literal.buffers {
        if resources
            .buffers
            .insert(buffer.binding, buffer.bytes.clone())
            .is_some()
        {
            return Err(gap(&format!("buffer {} bound twice", buffer.binding)).into());
        }
    }
    resources.argument_buffer_buffers =
        argument_buffers::embed(&mut resources.buffers, &literal.argument_buffer_buffers)
            .map_err(|error| gap(&error))?;
    resources.argument_buffer_textures =
        argument_buffers::embed_textures(&mut resources.buffers, &literal.argument_buffer_textures)
            .map_err(|error| gap(&error))?;
    resources.argument_buffer_samplers =
        argument_buffers::embed_samplers(&mut resources.buffers, &case.argument_buffer_samplers)
            .map_err(|error| gap(&error))?;
    let objects = argument_buffers::embed_objects(
        &mut resources.buffers,
        &mut resources.argument_buffer_buffers,
        &case.command_buffers,
        &case.pipeline_states,
    )
    .map_err(|error| gap(&error))?;
    resources.objects = objects.pipeline_tokens;
    resources.command_buffers = objects.command_buffers;
    resources.textures = resources::bound_textures(&literal.textures, &literal.texture_arrays)
        .map_err(|error| gap(&error))?;
    for sampler in &case.samplers {
        if resources
            .samplers
            .insert(sampler.binding, sampler.clone())
            .is_some()
        {
            return Err(gap(&format!("sampler {} bound twice", sampler.binding)).into());
        }
    }
    resources.function_tables = function_tables(&case.visible_function_tables)?;
    if let Some(binding) = resources
        .function_tables
        .keys()
        .find(|binding| resources.buffers.contains_key(binding))
    {
        return Err(gap(&format!(
            "buffer {binding} is also a visible function table"
        ))
        .into());
    }
    Ok(resources)
}

/// The region plan the render executor draws a rate-mapped pass of `map` with: the map's screen as
/// the guest's viewport and no guest scissor (the Metal oracle sets neither), within the device's
/// viewport `limits`, for a pipeline of `graphics`. A refusal of the planner (a viewport the device
/// cannot set, a layered or viewport-selecting pipeline, or a vertex function that writes memory in
/// a pass of several regions) is the binder's on this device (`not_supported{bind}`).
pub fn rate_map_regions(
    map: &RasterizationRateMap,
    limits: &RegionLimits,
    graphics: &metal2vulkan::pipeline::GraphicsState,
) -> Result<RateMapRegions, ExecutorError> {
    let [width, height] = map.screen_size();
    let refused = |error: metal2vulkan::bind::RegionError| {
        ExecutorError::from(NotSupported::Bind(format!(
            "rate-mapped render pass: {error}"
        )))
    };
    let plan = map
        .regions(
            0,
            [0.0, 0.0, f64::from(width), f64::from(height)],
            None,
            limits,
        )
        .map_err(refused)?;
    plan.admits(graphics).map_err(refused)?;
    Ok(plan)
}

/// The census of a rate-mapped pass of `map` drawn by `plan`, with the plan's near-equivalent.
pub fn rate_map_pass_census(map: &RasterizationRateMap, plan: &RateMapRegions) -> RateMapPassRun {
    RateMapPassRun {
        screen_size: map.screen_size(),
        compiled_physical_size: plan.physical_size(),
        regions: plan.regions().len() as u32,
        precision: plan.precision().into_iter().collect(),
    }
}

/// The consumer token of an indexed draw's index buffer (`IndexBufferRef::token`): the one index
/// buffer a case's draw binds.
pub const INDEX_BUFFER_TOKEN: u32 = 0;

/// The Vulkan index type of an authored `MTLIndexType`.
pub fn index_type(index_type: crate::case::IndexType) -> metal2vulkan::vk::IndexType {
    match index_type {
        crate::case::IndexType::Uint16 => metal2vulkan::vk::IndexType::UINT16,
        crate::case::IndexType::Uint32 => metal2vulkan::vk::IndexType::UINT32,
    }
}

/// The index buffer bytes and the draw the executor's own context draws for a case's `draw` of
/// the index buffer `bytes` ([`RenderCase::index_buffer`]), as the runtime draws it
/// (`DrawSource::padded`): where the draw's positions run past where Metal's fetch stops reading
/// the buffer (`IndexBufferRef::fetch_end`), a copy of its whole indices before there, zero at
/// every position past them, drawn from its start, and how many positions the copy pads; else the
/// bytes and the draw unchanged, padding none. Vulkan states no index for a fetch past the bound
/// range, even under `robustBufferAccess2`, so only a draw whose positions lie inside it reads
/// Metal's index 0 there on every device. The executor's buffer is zero from its length on
/// (`padded`), which the copy keeps for a partial last index.
pub fn padded_index_draw(
    draw: DrawCall,
    bytes: &[u8],
) -> Result<(Vec<u8>, DrawCall, Option<u64>), String> {
    let DrawCall::Indexed {
        index_type,
        index_buffer,
        index_count,
        index_offset,
        instance_count,
        base_vertex,
        base_instance,
    } = draw
    else {
        return Ok((bytes.to_vec(), draw, None));
    };
    let size: u64 = if index_type == metal2vulkan::vk::IndexType::UINT16 {
        2
    } else if index_type == metal2vulkan::vk::IndexType::UINT32 {
        4
    } else {
        return Err(format!("an index type of no authored size: {index_type:?}"));
    };
    let inside = index_buffer
        .fetch_end(index_offset)
        .saturating_sub(index_offset)
        / size;
    let count = u64::from(index_count);
    if count <= inside {
        return Ok((bytes.to_vec(), draw, None));
    }
    let positions = usize::try_from(count * size)
        .map_err(|_| format!("{count} index positions of {size} bytes"))?;
    let copied = (inside * size).min(index_buffer.length.saturating_sub(index_offset));
    let from = usize::try_from(index_offset).map_err(|_| "an index offset past usize")?;
    let to = from + usize::try_from(copied).map_err(|_| "an index copy past usize")?;
    let mut padded = vec![0; positions];
    padded[..to - from].copy_from_slice(
        bytes
            .get(from..to)
            .ok_or("an index buffer shorter than its length")?,
    );
    let draw = DrawCall::Indexed {
        index_type,
        index_buffer: IndexBufferRef::new(index_buffer.token, count * size),
        index_count,
        index_offset: 0,
        instance_count,
        base_vertex,
        base_instance,
    };
    Ok((padded, draw, Some(count - inside)))
}

/// The patch draw of a case's tessellation `draw` of the function whose AIR-layer interface is
/// `interface`, as the Metal oracle draws it (`executor_contract::tessellation`): the pipeline's
/// tessellation state of the draw's partition mode and winding, its factor bytes, and `drawPatches`
/// of the function's control points over every authored patch and instance, from patch 0 and base
/// instance 0, with no patch index buffer.
pub fn patch_draw(
    draw: &crate::case::TessellationDraw,
    interface: &AirInterface,
) -> Result<(Assembly, DrawCall), String> {
    let patch_count = u32::try_from(draw.factors.len())
        .map_err(|_| format!("{} patches overflow a draw", draw.factors.len()))?;
    Ok((
        Assembly::Patches {
            tessellation: patches::descriptor(draw),
            factors: patches::factor_bytes(&draw.factors),
        },
        DrawCall::Patches {
            control_points: patches::control_points(interface)?,
            patch_start: patches::PATCH_START,
            patch_count,
            patch_index: None,
            instance_count: draw.instance_count,
            base_instance: patches::BASE_INSTANCE,
            control_point_index: None,
        },
    ))
}

/// The Metal vertex format of an authored attribute format.
pub fn vertex_format(format: AttributeFormat) -> metal::VertexFormat {
    use metal::VertexFormat as V;
    use AttributeFormat as F;
    match format {
        F::Char => V::Char,
        F::Char2 => V::Char2,
        F::Char3 => V::Char3,
        F::Char4 => V::Char4,
        F::Uchar => V::UChar,
        F::Uchar2 => V::UChar2,
        F::Uchar3 => V::UChar3,
        F::Uchar4 => V::UChar4,
        F::Short => V::Short,
        F::Short2 => V::Short2,
        F::Short3 => V::Short3,
        F::Short4 => V::Short4,
        F::Ushort => V::UShort,
        F::Ushort2 => V::UShort2,
        F::Ushort3 => V::UShort3,
        F::Ushort4 => V::UShort4,
        F::Half => V::Half,
        F::Half2 => V::Half2,
        F::Half3 => V::Half3,
        F::Half4 => V::Half4,
        F::Float => V::Float,
        F::Float2 => V::Float2,
        F::Float3 => V::Float3,
        F::Float4 => V::Float4,
        F::Uint => V::UInt,
        F::Uint2 => V::UInt2,
        F::Uint3 => V::UInt3,
        F::Uint4 => V::UInt4,
        F::Int => V::Int,
        F::Int2 => V::Int2,
        F::Int3 => V::Int3,
        F::Int4 => V::Int4,
    }
}

/// The topology class of a draw's primitive (`inputPrimitiveTopology`, as the oracle sets it).
pub fn topology_class(primitive: Primitive) -> metal::PrimitiveTopologyClass {
    primitive.topology_class()
}

/// Whether a replayed draw of `primitive` is one the template compiled for `assembly` defines:
/// Metal compiles a render template for its own draw's topology class, so a draw of another class
/// (a GPU-encoded one; the checker refuses a host-encoded one) fails the run, and a patch template
/// is not replayed.
pub fn replayed_draw_in_class(assembly: &Assembly, primitive: Primitive) -> Result<(), String> {
    match assembly {
        Assembly::Primitives(compiled)
            if topology_class(*compiled) == topology_class(primitive) =>
        {
            Ok(())
        }
        Assembly::Primitives(compiled) => Err(format!(
            "a replayed draw of {primitive:?} is outside the topology class of its template, \
             compiled for {compiled:?}"
        )),
        Assembly::Patches { .. } => Err("a replayed draw's template draws patches".into()),
    }
}

/// The Vulkan input assembly of a draw's primitive, the consumer's pipeline state within the
/// class the pipeline was compiled for: its topology and whether it restarts primitives, as the
/// product maps Metal's primitive type (`metal::state::input_assembly`).
pub fn input_assembly(primitive: Primitive) -> (metal2vulkan::vk::PrimitiveTopology, bool) {
    metal2vulkan::metal::state::input_assembly(
        crate::case::indirect_commands::metal_primitive_type(primitive),
    )
}

/// The Vulkan cull mode of the encoder's Metal one: Vulkan's front face is the encoder's
/// default winding (clockwise) under the executor's negative-height viewport, so each Metal face
/// is the Vulkan face of its name.
pub fn cull_mode_flags(mode: metal::CullMode) -> Result<metal2vulkan::vk::CullModeFlags, String> {
    use metal2vulkan::vk::CullModeFlags as Flags;
    match mode {
        metal::CullMode::None => Ok(Flags::NONE),
        metal::CullMode::Front => Ok(Flags::FRONT),
        metal::CullMode::Back => Ok(Flags::BACK),
    }
}

/// The Vulkan input assembly of what a draw assembles: [`input_assembly`] of its primitive, or a
/// tessellation pipeline's patch list, whose patches never restart.
pub fn assembly_input(assembly: &Assembly) -> (metal2vulkan::vk::PrimitiveTopology, bool) {
    match assembly {
        Assembly::Primitives(primitive) => input_assembly(*primitive),
        Assembly::Patches { .. } => (metal2vulkan::vk::PrimitiveTopology::PATCH_LIST, false),
    }
}

/// The `MTLRenderPipelineDescriptor` the Metal oracle creates, without its functions: the vertex
/// descriptor of the case's vertex buffers (none without any), one color attachment per target
/// (blending off), the draw's topology class (unspecified for a patch draw, which the oracle
/// leaves unset), and a patch draw's tessellation state. It crosses the compile worker's exchange
/// (`acceptance::PipelineRequest::VertexDraw` and `Draw`), where the functions are decoded.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RenderTemplate {
    pub vertex_descriptor: Option<VertexDescriptor>,
    /// Color attachments by Metal index.
    pub color_attachments: BTreeMap<u32, ColorAttachmentDescriptor>,
    /// `depthAttachmentPixelFormat` and `stencilAttachmentPixelFormat`.
    #[serde(default)]
    pub depth_format: Option<metal::PixelFormat>,
    #[serde(default)]
    pub stencil_format: Option<metal::PixelFormat>,
    pub input_primitive_topology: metal::PrimitiveTopologyClass,
    /// The tessellation fields, `None` for Metal's defaults; read only for a post-tessellation
    /// vertex function.
    #[serde(default)]
    pub tessellation: Option<TessellationDescriptor>,
    /// `rasterSampleCount` of a multisampled pass; `None` leaves Metal's default, one sample, as
    /// the oracle does, and is not serialized, so a template of one sample is the one it always
    /// was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_sample_count: Option<u32>,
}

impl RenderTemplate {
    /// A template of topology class `topology` with no vertex descriptor, no attachments, and
    /// Metal's default tessellation state.
    pub fn new(topology: metal::PrimitiveTopologyClass) -> RenderTemplate {
        RenderTemplate {
            vertex_descriptor: None,
            color_attachments: BTreeMap::new(),
            depth_format: None,
            stencil_format: None,
            input_primitive_topology: topology,
            tessellation: None,
            raster_sample_count: None,
        }
    }

    /// The descriptor of `vertex` and `fragment` with this template, rasterization enabled exactly
    /// when there is a fragment function.
    pub fn descriptor(
        &self,
        vertex: StageSource,
        fragment: Option<StageSource>,
    ) -> Result<RenderPipelineDescriptor, Error> {
        let mut descriptor = RenderPipelineDescriptor::new(vertex);
        descriptor.rasterization_enabled = fragment.is_some();
        descriptor.fragment = fragment;
        descriptor.vertex_descriptor = self.vertex_descriptor.clone();
        for (&index, attachment) in &self.color_attachments {
            let slot = descriptor.color_attachments.get_mut(index as usize).ok_or(
                Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                    index,
                    metal: MetalAnswer::EndsProcess,
                }),
            )?;
            *slot = Some(*attachment);
        }
        descriptor.depth_format = self.depth_format;
        descriptor.stencil_format = self.stencil_format;
        descriptor.input_primitive_topology = self.input_primitive_topology;
        descriptor.tessellation = self.tessellation;
        if let Some(samples) = self.raster_sample_count {
            descriptor.raster_sample_count = samples;
        }
        Ok(descriptor)
    }
}

/// The vertex descriptor of `buffers` as the Metal oracle builds it (none without any buffer):
/// each buffer's layout at its index with its stride and step, and its attribute at offset 0 in
/// its format.
pub fn vertex_descriptor(buffers: &[VertexBuffer]) -> Option<VertexDescriptor> {
    if buffers.is_empty() {
        return None;
    }
    let mut layout = VertexDescriptor::new();
    for buffer in buffers {
        let mut buffer_layout =
            VertexBufferLayoutDescriptor::new((!buffer.dynamic).then_some(buffer.stride));
        buffer_layout.step_function = buffer.step;
        layout.layouts.insert(buffer.buffer_index, buffer_layout);
        layout.attributes.insert(
            buffer.location,
            VertexAttributeDescriptor::new(vertex_format(buffer.format), 0, buffer.buffer_index),
        );
    }
    Some(layout)
}

/// The [`RenderTemplate`] of `case`.
pub fn render_template(case: &RenderCase) -> RenderTemplate {
    RenderTemplate {
        vertex_descriptor: vertex_descriptor(&case.vertex_buffers),
        color_attachments: case
            .color_targets
            .iter()
            .map(|(&index, target)| {
                (
                    index,
                    ColorAttachmentDescriptor::new(resources::pixel_format(target.format)),
                )
            })
            .collect(),
        depth_format: case
            .depth_stencil
            .as_ref()
            .and_then(|attachment| attachment.depth.as_ref())
            .map(|_| metal::PixelFormat::Depth32Float),
        stencil_format: case
            .depth_stencil
            .as_ref()
            .and_then(|attachment| attachment.stencil.as_ref())
            .map(|_| metal::PixelFormat::Stencil8),
        input_primitive_topology: match &case.assembly {
            Assembly::Primitives(primitive) => topology_class(*primitive),
            Assembly::Patches { .. } => metal::PrimitiveTopologyClass::Unspecified,
        },
        tessellation: match &case.assembly {
            Assembly::Primitives(_) => None,
            Assembly::Patches { tessellation, .. } => Some(*tessellation),
        },
        raster_sample_count: case.raster_sample_count,
    }
}

/// The `MTLRenderPipelineDescriptor` the Metal oracle creates for `case` with the functions
/// `vertex` and `fragment` ([`render_template`]).
pub fn pipeline_descriptor(
    case: &RenderCase,
    vertex: FunctionDescriptor,
    fragment: Option<FunctionDescriptor>,
) -> Result<RenderPipelineDescriptor, Error> {
    render_template(case).descriptor(
        StageSource::Function(vertex),
        fragment.map(StageSource::Function),
    )
}

/// The guest-visible slot of a stage's buffer: disjoint 4-GiB windows per stage and index, never
/// 0 (see [`BufferAddress::synthetic`]).
pub fn synthetic_address(stage: ShaderStage, index: u32) -> BufferAddress {
    let base = if stage == ShaderStage::Fragment {
        BUFFER_SLOTS
    } else {
        0
    };
    BufferAddress::synthetic(base + index)
}

/// The `BindState` of `case`: each stage's buffers at offset 0 with their lengths and addresses
/// (`addresses` by stage and Metal index), its visible-function tables the same way as buffers of
/// their table words' length (`bind::table`), textures, and samplers; the vertex buffers bound to
/// the vertex stage at their slots (static strides come from the descriptor); each argument-buffer
/// buffer as a used buffer at its address (`embedded` by stage, owner, and field offset; its
/// consumer token is its position in [`RenderCase::argument_buffer_buffers`]) and each
/// argument-buffer texture as a used texture with its resource id (its token follows the buffers',
/// [`texture_tokens`]), each resident for reading and writing in the vertex and fragment stages, as
/// the Metal oracle's `useResource:usage:stages:` makes it; the pass's render target array length;
/// and a patch draw's factor buffer, its whole length at the shared offset and instance stride,
/// with Metal's default scale (`executor_contract::tessellation`); and the draw's depth/stencil
/// descriptor, the one the Metal oracle sets (`RenderCase::depth_stencil_state`), which a
/// pipeline's shader-side stencil reads as push data; telling the executor's view base alignment
/// apart ([`views_told_apart`]).
pub fn bind_state(
    case: &RenderCase,
    addresses: &BTreeMap<(ShaderStage, u32), BufferAddress>,
    embedded: &BTreeMap<(ShaderStage, u32, u32), BufferAddress>,
) -> Result<BindState, String> {
    let bound = |stage: ShaderStage, index: u32, length: u64, attribute_stride: Option<u32>| {
        let address = addresses.get(&(stage, index)).copied().unwrap_or_default();
        BoundBuffer::Buffer {
            length,
            offset: 0,
            gpu_address: address.guest,
            device_address: address.host,
            attribute_stride,
        }
    };
    let mut state = BindState::default();
    state.render_target_array_length = case.render_target_array_length;
    if let Assembly::Patches { factors, .. } = &case.assembly {
        state.tessellation_factors = TessellationFactors::new(
            Some(TessellationFactorBuffer::new(
                factors.len() as u64,
                patches::FACTOR_OFFSET,
                patches::FACTOR_INSTANCE_STRIDE,
            )),
            patches::FACTOR_SCALE,
        );
    }
    for (stage, function) in case.stages() {
        let table = stage_table(stage, function, |index, length| {
            bound(stage, index, length, None)
        })?;
        state.stages.insert(stage, table);
    }
    let table = state.stages.entry(ShaderStage::Vertex).or_default();
    for buffer in &case.vertex_buffers {
        // A dynamic layout's buffer is bound with the stride form.
        let slot = bound(
            ShaderStage::Vertex,
            buffer.buffer_index,
            buffer.bytes.len() as u64,
            buffer.dynamic.then_some(buffer.stride),
        );
        if table.buffers.insert(buffer.buffer_index, slot).is_some() {
            return Err(format!(
                "vertex buffer {} is also a bound buffer",
                buffer.buffer_index
            ));
        }
    }
    for (token, (key, bytes)) in case.argument_buffer_buffers().enumerate() {
        let address = embedded.get(&key).copied().unwrap_or_default();
        state.used_resources.push(argument_buffers::used_buffer(
            token,
            bytes.len(),
            address,
            RenderStages::VERTEX | RenderStages::FRAGMENT,
        )?);
    }
    let tokens = texture_tokens(case)?;
    for (key, texture) in case.argument_buffer_textures() {
        state.used_resources.push(argument_buffers::used_texture(
            tokens[&key],
            texture,
            RenderStages::VERTEX | RenderStages::FRAGMENT,
        )?);
    }
    state.argument_samplers = argument_buffers::argument_samplers(case.argument_buffer_samplers())?;
    state.depth_stencil = case.depth_stencil_state.descriptor;
    Ok(views_told_apart(state))
}

/// The bind table of `function`'s own resources at `stage`: its buffers and visible function
/// tables, each bound as `bound` binds a buffer of its index and length, its textures and its
/// samplers.
pub fn stage_table(
    stage: ShaderStage,
    function: &RenderFunction,
    bound: impl Fn(u32, u64) -> BoundBuffer,
) -> Result<StageTable, String> {
    let mut table = StageTable::default();
    let lengths = function
        .resources
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, bytes.len() as u64))
        .chain(
            function
                .resources
                .function_tables
                .iter()
                .map(|(&index, function_table)| (index, function_table.byte_len())),
        );
    for (index, length) in lengths {
        if table.buffers.insert(index, bound(index, length)).is_some() {
            return Err(format!(
                "{stage:?} buffer {index} is also a visible function table"
            ));
        }
    }
    for (&index, texture) in &function.resources.textures {
        table
            .textures
            .insert(index, resources::bound_texture(texture)?);
    }
    for (&index, sampler) in &function.resources.samplers {
        table
            .samplers
            .insert(index, resources::bound_sampler(sampler));
    }
    Ok(table)
}

/// The id of each argument-buffer sampler of `case`, by `(stage, owner, field offset)`: its
/// [`argument_buffers::sampler_id`] by its position in stage order.
pub fn sampler_ids(case: &RenderCase) -> Result<BTreeMap<(ShaderStage, u32, u32), u64>, String> {
    argument_buffers::sampler_ids(case.argument_buffer_samplers().map(|(key, _)| key))
}

/// The consumer token of each argument-buffer texture of `case`, by `(stage, owner, field
/// offset)`: its position after every argument-buffer buffer
/// ([`argument_buffers::texture_tokens`]).
pub fn texture_tokens(case: &RenderCase) -> Result<BTreeMap<(ShaderStage, u32, u32), u32>, String> {
    argument_buffers::texture_tokens(
        case.argument_buffer_buffers().count(),
        case.argument_buffer_textures().map(|(key, _)| key),
    )
}

/// The synthetic guest address of each argument-buffer buffer of `case`, by its key
/// ([`argument_buffers::synthetic_addresses`]): a command buffer's record storage at its records
/// address ([`records_ids`]), every other buffer at its [`BufferAddress::synthetic_used`].
pub fn synthetic_used_addresses(
    case: &RenderCase,
) -> Result<BTreeMap<(ShaderStage, u32, u32), BufferAddress>, String> {
    argument_buffers::synthetic_addresses(
        case.argument_buffer_buffers().map(|(key, _)| key),
        &records_ids(case)?,
    )
}

/// The `gpuResourceID` of each command buffer of `case`, by `(stage, owner, field offset)`, in
/// stage order ([`argument_buffers::records_ids`]): the word its owner holds, which the product
/// resolves to its record storage. The stages' command buffers share one space, as Metal's do.
pub fn records_ids(case: &RenderCase) -> Result<BTreeMap<(ShaderStage, u32, u32), u64>, String> {
    argument_buffers::records_ids(case.command_buffers().map(|(key, _)| key))
}

/// The `BindShape` the render executor binds `case` with on a device that has device addresses
/// exactly when `addressed` ([`bind_state`] with every buffer's and argument-buffer buffer's
/// synthetic guest address and, when `addressed`, a host address; only whether a buffer has one
/// is part of the shape): what the compile worker compiles the draw for
/// (`variants::compile_render_for_shape`).
pub fn render_shape(case: &RenderCase, addressed: bool) -> Result<BindShape, String> {
    let with_host = |mut address: BufferAddress| {
        address.host = addressed.then_some(address.guest);
        address
    };
    let addresses = case
        .stages()
        .flat_map(|(stage, f)| {
            f.resources
                .buffers
                .keys()
                .chain(f.resources.function_tables.keys())
                .map(move |&index| (stage, index))
        })
        .chain(
            case.vertex_buffers
                .iter()
                .map(|buffer| (ShaderStage::Vertex, buffer.buffer_index)),
        )
        .map(|(stage, index)| ((stage, index), with_host(synthetic_address(stage, index))))
        .collect();
    let embedded = synthetic_used_addresses(case)?
        .into_iter()
        .map(|(key, address)| (key, with_host(address)))
        .collect();
    Ok(bind_state(case, &addresses, &embedded)?.shape())
}

/// The final contents of a render run, by stage and Metal index (color targets by index).
#[derive(Debug, Default)]
pub struct RenderFinals {
    pub buffers: BTreeMap<(ShaderStage, u32), Vec<u8>>,
    /// Each argument-buffer buffer's, by stage, owner, and field offset.
    pub argument_buffer_buffers: BTreeMap<(ShaderStage, u32, u32), Vec<u8>>,
    /// The observed argument-buffer texture's, by stage, owner, and field offset.
    pub argument_buffer_textures: BTreeMap<(ShaderStage, u32, u32), LiteralTexture>,
    pub textures: BTreeMap<(ShaderStage, u32), LiteralTexture>,
    pub targets: BTreeMap<u32, ColorTarget>,
    /// The depth/stencil attachment's final aspects.
    pub depth_stencil: Option<LiteralDepthStencil>,
    /// The fragment imageblock's final cells.
    pub imageblock: Option<Cells>,
}

/// The bytes of `output` from a run's final contents.
pub fn select_render_output(
    output: RenderOutput,
    finals: &RenderFinals,
) -> Result<Vec<u8>, String> {
    match output {
        RenderOutput::None => Ok(Vec::new()),
        RenderOutput::Buffer {
            stage,
            binding,
            offset,
            length,
        } => {
            let bytes = finals
                .buffers
                .get(&(stage, binding))
                .ok_or_else(|| format!("output buffer {binding} of {stage:?} is not bound"))?;
            select_range(bytes, offset, length, &format!("buffer {binding}"))
        }
        RenderOutput::ArgumentBufferBuffer {
            stage,
            owner,
            field_offset,
            offset,
            length,
        } => {
            let what = format!("argument-buffer buffer {owner}+{field_offset}");
            let bytes = finals
                .argument_buffer_buffers
                .get(&(stage, owner, field_offset))
                .ok_or_else(|| format!("output {what} of {stage:?} is not bound"))?;
            select_range(bytes, offset, length, &what)
        }
        RenderOutput::Texture {
            stage,
            binding,
            origin,
            dimensions,
        } => {
            let texture = finals
                .textures
                .get(&(stage, binding))
                .ok_or_else(|| format!("output texture {binding} of {stage:?} is not bound"))?;
            resources::select_texture(texture, origin, dimensions)
        }
        RenderOutput::ArgumentBufferTexture {
            stage,
            owner,
            field_offset,
            origin,
            dimensions,
        } => {
            let texture = finals
                .argument_buffer_textures
                .get(&(stage, owner, field_offset))
                .ok_or_else(|| {
                    format!(
                        "output argument-buffer texture {owner}+{field_offset} of {stage:?} is not \
                         bound"
                    )
                })?;
            resources::select_texture(texture, origin, dimensions)
        }
        RenderOutput::RenderTarget {
            index,
            origin,
            dimensions,
        } => {
            let target = finals
                .targets
                .get(&index)
                .ok_or_else(|| format!("output render target {index} is not attached"))?;
            select_target(target, origin, dimensions)
        }
        RenderOutput::Depth { origin, dimensions }
        | RenderOutput::Stencil { origin, dimensions } => {
            let attachment = finals
                .depth_stencil
                .as_ref()
                .ok_or("the output depth/stencil attachment is not attached")?;
            attachment.select(
                matches!(output, RenderOutput::Depth { .. }),
                origin,
                dimensions,
            )
        }
        RenderOutput::Imageblock {
            member,
            origin,
            dimensions,
        } => finals
            .imageblock
            .as_ref()
            .ok_or("the output imageblock's cells were not read")?
            .select(member, origin, dimensions),
    }
}

/// The `(origin, dimensions)` region of `target`, every sample of each pixel, rows in order,
/// tightly packed.
pub fn select_target(
    target: &ColorTarget,
    origin: [u32; 2],
    dimensions: [u32; 2],
) -> Result<Vec<u8>, String> {
    let texel = target.format.bytes_per_pixel() * target.samples as usize;
    let [width, height] = target.dimensions;
    let fits = |axis: usize, extent: u32| {
        origin[axis]
            .checked_add(dimensions[axis])
            .is_some_and(|end| end <= extent)
    };
    if !(fits(0, width) && fits(1, height)) {
        return Err(format!(
            "output selection {origin:?}+{dimensions:?} exceeds the {width}x{height} target"
        ));
    }
    if target.bytes.len() != width as usize * height as usize * texel {
        return Err(format!(
            "the target holds {} bytes, not {width}x{height} texels of {texel} bytes",
            target.bytes.len()
        ));
    }
    let row = width as usize * texel;
    let mut out = Vec::with_capacity(dimensions[0] as usize * dimensions[1] as usize * texel);
    for y in origin[1]..origin[1] + dimensions[1] {
        let start = y as usize * row + origin[0] as usize * texel;
        out.extend_from_slice(&target.bytes[start..start + dimensions[0] as usize * texel]);
    }
    Ok(out)
}

/// The Metal stage whose function `module` runs in a render pipeline that is `tessellated` or
/// not, `None` for a stage the link synthesized: a tessellation pipeline's Metal stages are its
/// evaluation stage (the vertex function) and its fragment stage, and its vertex and control
/// stages run no Metal function, nor do the geometry stages of barycentric weights read as
/// varyings.
pub fn stage_of(
    module: &StageModule,
    tessellated: bool,
) -> Result<Option<ShaderStage>, ExecutorError> {
    use metal2vulkan::vk::ShaderStage as S;
    match module.stage {
        s if s == S::FRAGMENT => Ok(Some(ShaderStage::Fragment)),
        s if s == S::VERTEX && !tessellated || s == S::TESSELLATION_EVALUATION && tessellated => {
            Ok(Some(ShaderStage::Vertex))
        }
        s if tessellated && (s == S::VERTEX || s == S::TESSELLATION_CONTROL) => Ok(None),
        s if s == S::GEOMETRY => Ok(None),
        other => Err(format!("a render pipeline stage {other:?}").into()),
    }
}

/// The module of `pipeline` that runs its Metal vertex function: its vertex stage, or a
/// tessellation pipeline's evaluation stage.
pub fn vertex_module(pipeline: &RenderPipeline) -> Result<&StageModule, ExecutorError> {
    let tessellated = pipeline.graphics.tessellation.is_some();
    for module in &pipeline.stages {
        if stage_of(module, tessellated)? == Some(ShaderStage::Vertex) {
            return Ok(module);
        }
    }
    Err("a render pipeline without its vertex function's module".into())
}

#[cfg(test)]
pub(crate) mod tests;
