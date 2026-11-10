//! Mesh cases of the engine candidate: an object or mesh case's draw (`case::MeshDraw`) mapped for
//! the runtime's mesh path (`runtime_case::encoded_mesh`), as the Metal oracle draws it
//! (`metal::platform`).
//!
//! The case's own function binds the case's resources, mapped as a render case's own stage's are
//! ([`graphics::stage_resources`]); each partner its draw names is a function of the case's own
//! module, specialized with the case's constant values, that binds none. A case that states an
//! observation or an object launch draws with the checker's generated stages in its partners'
//! places (`companion::MeshStages`: a mesh case's observer fragment and payload object, an object
//! case's launch observer and its observer fragment), as the oracle does. Its output is a resource of its own
//! function ([`graphics::stage_output`]) or, where the draw rasterizes, a region of a render
//! target; its color targets are the case's, or the oracle's one-pixel sink where it does not
//! ([`graphics::color_targets`]).
//!
//! The compile worker compiles its pipeline device-free as the runtime compiles it
//! ([`mesh_request`]): the oracle's descriptor ([`MeshTemplate`]) of its functions, each from the
//! case's library or its companion's, for the bind shape of its own stage's resources
//! ([`mesh_shape`]).

use super::graphics::{self, ColorTarget, RenderFunction, RenderOutput, StageResources};
use super::{
    function_constants, require_case_families, resources, views_told_apart, BufferAddress,
};
use crate::acceptance::{MeshFunction, PipelineRequest, RequestFunction};
use crate::case::{AuthoredCase, MeshGrid, OutputSelection, Stage};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::companion::Companion;
use crate::engine::RowSource;
use crate::literal::LiteralResources;
use metal2vulkan::bind::{BindShape, BindState, BoundBuffer};
use metal2vulkan::error::{DescriptorError, Error, MetalAnswer};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::Stage as ShaderStage;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, MeshPipelineDescriptor, StageSource};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// An object or mesh case, mapped: its pipeline's functions by stage, its pass's color targets,
/// whether it rasterizes, its draw's grid and threadgroup sizes, and its output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshCase {
    /// The case's own stage, object or mesh, whose function binds its resources.
    pub stage: ShaderStage,
    pub object: Option<RenderFunction>,
    pub mesh: RenderFunction,
    /// The fragment partner, or the generated observer fragment ([`companion`](Self::companion)).
    pub fragment: Option<RenderFunction>,
    /// The checker's generated stages of a case that states an observation or an object launch,
    /// whose library holds the functions of the [`generated`](Self::generated) stages.
    pub companion: Option<Companion>,
    /// The stages whose functions are the companion's.
    pub generated: BTreeSet<ShaderStage>,
    /// Color attachments by Metal index.
    pub color_targets: BTreeMap<u32, ColorTarget>,
    /// The pipeline's `rasterizationEnabled`.
    pub rasterization: bool,
    pub grid: MeshGrid,
    pub threads_per_object_threadgroup: [u32; 3],
    pub threads_per_mesh_threadgroup: [u32; 3],
    pub output: RenderOutput,
}

impl MeshCase {
    /// Each stage the pipeline has, with its function, in pipeline order.
    pub fn stages(&self) -> impl Iterator<Item = (ShaderStage, &RenderFunction)> {
        self.object
            .iter()
            .map(|object| (ShaderStage::Object, object))
            .chain([(ShaderStage::Mesh, &self.mesh)])
            .chain(
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
}

/// Map an object or mesh case, with the checker's generated `companion`: refused (`NotSupported`)
/// where it is no case this executor draws.
pub fn mesh_case(
    case: &AuthoredCase,
    companion: Option<&Companion>,
) -> Result<MeshCase, ExecutorError> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let stage = match case.stage {
        Stage::Object => ShaderStage::Object,
        Stage::Mesh => ShaderStage::Mesh,
        Stage::Kernel | Stage::Vertex | Stage::Fragment => {
            return Err(gap("a kernel, vertex or fragment case is not a mesh case").into())
        }
    };
    let draw = case
        .mesh_draw
        .as_ref()
        .ok_or_else(|| gap("an object or mesh case without its mesh draw"))?;
    require_case_families(case)?;
    let literal =
        LiteralResources::prepare(case).map_err(|error| gap(&format!("resources: {error}")))?;
    let constants = function_constants(&literal)?;
    let own = RenderFunction {
        entry: case.entry.clone(),
        constants: constants.clone(),
        resources: graphics::stage_resources(case, &literal)?,
    };
    let partner = |name: &str| RenderFunction {
        entry: name.into(),
        constants: constants.clone(),
        resources: StageResources::default(),
    };
    // The generated stages of a draw that states an observation or an object launch, and of no
    // other.
    match (draw.generates(), companion) {
        (false, None) | (true, Some(Companion::MeshStages(_))) => {}
        (generates, companion) => {
            return Err(ExecutorError::Failed(format!(
                "a mesh draw that takes generated stages ({generates}) has companion {:?}",
                companion.map(Companion::entry)
            )))
        }
    }
    let mut generated = BTreeSet::new();
    // The function at `stage`: the partner the draw names, or the companion's.
    let mut supplied = |stage: Stage, shader: ShaderStage| {
        let named = draw.partner(stage);
        let made = companion.and_then(|companion| companion.stage_entry(stage));
        match (named, made) {
            (named, None) => Ok(named.map(partner)),
            (None, Some(entry)) => {
                generated.insert(shader);
                Ok(Some(RenderFunction {
                    entry: entry.into(),
                    constants: FunctionConstantValues::defaults(),
                    resources: StageResources::default(),
                }))
            }
            (Some(named), Some(entry)) => Err(ExecutorError::Failed(format!(
                "a mesh draw names {named:?} at the {} stage, and its companion generates {entry}",
                stage.entry_list()
            ))),
        }
    };
    let (object, mesh) = match stage {
        ShaderStage::Object => {
            let mesh = supplied(Stage::Mesh, ShaderStage::Mesh)?.ok_or_else(|| {
                gap("an object case's draw that names no mesh function (mesh_companion)")
            })?;
            (Some(own), mesh)
        }
        _ => (supplied(Stage::Object, ShaderStage::Object)?, own),
    };
    let fragment = supplied(Stage::Fragment, ShaderStage::Fragment)?;
    let output =
        match graphics::stage_output(&case.output, stage).map_err(|error| gap(&error))? {
            Some(output) => output,
            None => match case.output {
                OutputSelection::RenderTarget {
                    index,
                    origin,
                    dimensions,
                } if draw.rasterization => RenderOutput::RenderTarget {
                    index,
                    origin,
                    dimensions,
                },
                _ => return Err(gap(
                    "a mesh case's output other than a resource of its own function or a region \
                     of a rasterized render target",
                )
                .into()),
            },
        };
    Ok(MeshCase {
        stage,
        object,
        mesh,
        fragment,
        companion: companion.cloned(),
        generated,
        color_targets: graphics::color_targets(&literal, draw.rasterization),
        rasterization: draw.rasterization,
        grid: draw.grid,
        threads_per_object_threadgroup: draw.threads_per_object_threadgroup,
        threads_per_mesh_threadgroup: draw.threads_per_mesh_threadgroup,
        output,
    })
}

/// The mesh render pipeline descriptor of a mesh case's draw, as the Metal oracle and the runtime
/// make it: its color attachments by Metal index (the case's targets, or the one-pixel sink where
/// it does not rasterize) and whether it rasterizes ([`MeshTemplate::descriptor`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshTemplate {
    pub color_attachments: BTreeMap<u32, ColorAttachmentDescriptor>,
    pub rasterization: bool,
}

impl MeshTemplate {
    /// The template of `mesh`'s draw.
    pub fn of(mesh: &MeshCase) -> MeshTemplate {
        MeshTemplate {
            color_attachments: mesh
                .color_targets
                .iter()
                .map(|(&index, target)| {
                    let format = resources::pixel_format(target.format);
                    (index, ColorAttachmentDescriptor::new(format))
                })
                .collect(),
            rasterization: mesh.rasterization,
        }
    }

    /// The descriptor of `object`, `mesh` and `fragment` with this template.
    pub fn descriptor(
        &self,
        object: Option<StageSource>,
        mesh: StageSource,
        fragment: Option<StageSource>,
    ) -> Result<MeshPipelineDescriptor, Error> {
        let mut descriptor = MeshPipelineDescriptor::new(mesh);
        descriptor.object = object;
        descriptor.fragment = fragment;
        descriptor.rasterization_enabled = self.rasterization;
        for (&index, attachment) in &self.color_attachments {
            let slot = descriptor.color_attachments.get_mut(index as usize).ok_or(
                Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                    index,
                    metal: MetalAnswer::EndsProcess,
                }),
            )?;
            *slot = Some(*attachment);
        }
        Ok(descriptor)
    }
}

/// The `BindShape` the runtime binds `mesh`'s draw with on a device that has device addresses
/// exactly when `addressed`: its own stage's buffers at their synthetic guest addresses (and, when
/// `addressed`, a host address), textures and samplers ([`graphics::stage_table`]); a partner
/// binds none; telling the executor's view base alignment apart ([`views_told_apart`]), so its
/// plan is the runtime's of the same binds. What the compile worker compiles the draw for
/// (`variants::compile_mesh_for_shape`).
/// A case whose own function takes an argument buffer, a command buffer or a pipeline state is not
/// built here (`NotImplemented`).
pub fn mesh_shape(mesh: &MeshCase, addressed: bool) -> Result<BindShape, ExecutorError> {
    let mut state = BindState::default();
    for (stage, function) in mesh.stages() {
        let resources = &function.resources;
        if !(resources.argument_buffer_buffers.is_empty()
            && resources.argument_buffer_textures.is_empty()
            && resources.argument_buffer_samplers.is_empty()
            && resources.command_buffers.is_empty()
            && resources.objects.is_empty())
        {
            return Err(NotSupported::NotImplemented(
                "engine candidate: a device-free compile of a mesh draw whose function takes an \
                 argument buffer"
                    .into(),
            )
            .into());
        }
        let bound = |index: u32, length: u64| {
            let address: BufferAddress = graphics::synthetic_address(stage, index);
            BoundBuffer::Buffer {
                length,
                offset: 0,
                gpu_address: address.guest,
                device_address: addressed.then_some(address.guest),
                attribute_stride: None,
            }
        };
        let table = graphics::stage_table(stage, function, bound).map_err(ExecutorError::Failed)?;
        state.stages.insert(stage, table);
    }
    Ok(views_told_apart(state).shape())
}

/// The compile request for `mesh`'s pipeline for the bind shape `shape`, and its libraries in
/// `PipelineRequest::libraries` order: `library`, the case's, then `companion`, its generated
/// stages' source, where a stage is generated (`MeshCase::generated`). No function links any.
pub fn mesh_request<'a>(
    library: &RowSource<'a>,
    companion: Option<RowSource<'a>>,
    mesh: &MeshCase,
    shape: &BindShape,
) -> Result<(PipelineRequest, Vec<RowSource<'a>>), ExecutorError> {
    let function = |stage: ShaderStage, function: &RenderFunction| MeshFunction {
        library: usize::from(mesh.generated.contains(&stage)),
        function: RequestFunction::new(function.entry.as_str(), function.constants.clone()),
    };
    let libraries = match (companion, mesh.generated.is_empty()) {
        (None, true) => vec![library.clone()],
        (Some(companion), false) => vec![library.clone(), companion],
        _ => {
            return Err(ExecutorError::Failed(
                "a mesh draw's generated stages without their source, or a source without them"
                    .into(),
            ))
        }
    };
    let request = PipelineRequest::MeshDraw {
        object: mesh
            .object
            .as_ref()
            .map(|object| function(ShaderStage::Object, object)),
        mesh: function(ShaderStage::Mesh, &mesh.mesh),
        fragment: mesh
            .fragment
            .as_ref()
            .map(|fragment| function(ShaderStage::Fragment, fragment)),
        template: MeshTemplate::of(mesh),
        shape: Some(shape.clone()),
    };
    Ok((request, libraries))
}
