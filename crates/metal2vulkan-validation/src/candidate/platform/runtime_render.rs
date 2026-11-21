//! The engine render executor through the runtime (`metal2vulkan-runtime`).
//!
//! A render case the engine routes here (`engine::runs_render_through_runtime`) draws as a
//! consumer draws it, on the runtime's public API alone, on the device a kernel's run opens and
//! under the same validation layer and per-attempt ceilings ([`super::runtime`]): one render
//! pipeline state of the oracle's descriptor (`graphics::pipeline_descriptor`), the case's own
//! function from the row's AIR with its constants and linked functions and its generated
//! companion from the companion's AIR; one Shared buffer per bound buffer, vertex buffer and index
//! buffer; each bound texture and sampler as the Metal oracle makes them (`runtime_textures`); each
//! color attachment a Shared 2D texture of its format and extent, written from the case's bytes
//! and loaded and stored by the pass, and the oracle's 1x1 `R8Unorm` attachment where
//! rasterization is disabled, which no draw writes; the depth/stencil attachment's one aspect as
//! the oracle makes it, a Shared `Depth32Float` filled by a draw of the case's depths
//! (`runtime_depth`, as the runtime takes no upload of one) or a Shared `Stencil8` written from its
//! bytes, loaded and stored by the pass; the oracle's depth-stencil state, stencil
//! references, cull mode and winding; one draw, direct or indexed, in one command buffer, waited
//! for under the hang protocol, then read back.
//!
//! The row records what the runtime answers, as a kernel's does: the state's precision notes on
//! the case's own stage, and the float reductions its AIR calls. An out-of-bounds case's facts are
//! read before the state is made (`runtime_bounds::drawn`).

use super::runtime::{self, runtime_error, Identity, Placed};
use super::runtime_bounds;
use super::runtime_depth;
use super::runtime_textures::{self, Bound, Indices, Wanted};
use crate::candidate::engine::graphics::{
    self, Assembly, ColorTarget, RenderCase, RenderFinals, RenderOutput,
};
use crate::candidate::engine::{product_refusal, resources, ComputeRun, RunAttempt, RuntimeDraw};
use crate::case::Stage;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::coverage::ModuleCensus;
use crate::gpu::DeviceType;
use crate::literal::LiteralDepthStencil;
use metal2vulkan::bind::DrawCall;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, Stage as ShaderStage,
};
use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;

/// The executor a row's environment names.
const EXECUTOR: &str = "runtime-render";

/// The format of the 1x1 attachment a rasterization-disabled draw renders into, the Metal
/// oracle's (`metal::platform::attachments::make_rasterization_sink`).
const SINK: rt::PixelFormat = rt::PixelFormat::R8Unorm;

/// Draw `case`, whose own function and companion are `draw`'s, through the runtime on the device
/// of `device_type`, as the module documentation says. Refusals are typed as a kernel's run types
/// them (`runtime::execute_runtime`).
pub fn execute_runtime_render(
    device_type: DeviceType,
    case_id: &str,
    case: &RenderCase,
    draw: RuntimeDraw<'_>,
) -> RunAttempt {
    runtime::under_validation(|| {
        runtime::attempt_on(device_type, EXECUTOR, |device, identity, environment| {
            run(
                device,
                device_type,
                case_id,
                case,
                &draw,
                (identity, environment),
            )
        })
    })
}

/// Make `case`'s pipeline state on `device`, make and bind what it binds, draw it once, wait,
/// and read its output back.
fn run(
    device: &rt::Device,
    device_type: DeviceType,
    case_id: &str,
    case: &RenderCase,
    draw: &RuntimeDraw<'_>,
    (identity, environment): (&Identity, &serde_json::Value),
) -> Result<ComputeRun, ExecutorError> {
    let _grant = runtime::sampler_grant(device)?;
    let descriptor = descriptor(case, draw)?;
    // How the state's vertex function contains each out-of-bounds target, decided before anything
    // is made: a target the device leaves uncontained or checked nowhere refuses the draw. The
    // routed targets are attribute buffers (`engine::runs_render_through_runtime`), which the
    // draw places without guards, so a contained one is not drawn either.
    let class = Some(runtime::execution_class(identity));
    if runtime_bounds::drawn(device, case, &descriptor, class)?.is_some() {
        return Err(NotSupported::NotImplemented(
            "a draw through the runtime whose out-of-bounds targets are contained: it places \
             its vertex buffers without guards"
                .into(),
        )
        .into());
    }
    let state = runtime::made_within(|numbered| {
        device.new_render_pipeline_state(&descriptor, |samplers| numbered(samplers))
    })?;
    let mut indices = Indices::new();
    let stages = Stages::make(device, device_type, (case, draw), &mut indices)?;
    let vertex_buffers = case
        .vertex_buffers
        .iter()
        .map(|buffer| {
            let what = format!("vertex buffer {}", buffer.buffer_index);
            Ok((
                buffer.buffer_index,
                Placed::new(device, &what, &buffer.bytes)?,
            ))
        })
        .collect::<Result<BTreeMap<u32, Placed>, ExecutorError>>()?;
    let index_buffer = case
        .index_buffer
        .as_deref()
        .map(|bytes| Placed::new(device, "the index buffer", bytes))
        .transpose()?;
    let targets = Targets::make(device, device_type, case, &mut indices)?;
    let depth_stencil = device.new_depth_stencil_state(&case.depth_stencil_state.descriptor);
    let label = crate::gpu::wait::SubmissionLabel {
        case_id: case_id.into(),
        module_sha256: crate::hash::sha256_bytes(draw.bitcode),
    };
    crate::gpu::wait::with_submission_label(label, || {
        runtime::submitted(
            device,
            device_type,
            "a draw through the runtime",
            |command_buffer| {
                let fail = |what: &'static str| move |error: rt::Error| runtime_error(what, error);
                let mut encoder = command_buffer
                    .render_command_encoder(&targets.pass())
                    .map_err(fail("the render encoder"))?;
                encoder
                    .set_render_pipeline_state(&state)
                    .map_err(fail("the pipeline state"))?;
                stages.bind(&mut encoder)?;
                for (&index, placed) in &vertex_buffers {
                    encoder
                        .set_vertex_buffer(Some(&placed.buffer), 0, index)
                        .map_err(fail("a vertex buffer"))?;
                }
                encoder
                    .set_depth_stencil_state(Some(&depth_stencil))
                    .map_err(fail("the depth-stencil state"))?;
                encoder.set_stencil_front_back_reference_value(
                    case.depth_stencil_state.front_reference,
                    case.depth_stencil_state.back_reference,
                );
                encoder.set_cull_mode(case.cull_mode);
                encoder.set_front_facing_winding(case.depth_stencil_state.winding);
                drawn(&mut encoder, case, index_buffer.as_ref())?;
                encoder.end_encoding();
                Ok(())
            },
        )
    })?;
    let output = match case.output {
        RenderOutput::Texture {
            stage,
            binding,
            origin,
            dimensions,
        } => {
            let (Some(made), Some(texture)) = (
                stages
                    .of(stage)
                    .and_then(|bound| bound.textures.get(&binding)),
                case.function(stage)
                    .and_then(|function| function.resources.textures.get(&binding)),
            ) else {
                return Err(ExecutorError::Failed(format!(
                    "output texture {binding} of {stage:?} is not bound"
                )));
            };
            runtime_textures::read_back(made, texture, origin, dimensions, runtime_error)?
        }
        output => {
            let finals = RenderFinals {
                buffers: stages.finals(),
                targets: targets.finals(case)?,
                depth_stencil: targets.depth_stencil_finals()?,
                ..RenderFinals::default()
            };
            graphics::select_render_output(output, &finals).map_err(ExecutorError::Failed)?
        }
    };
    Ok(runtime::ran(
        identity,
        environment,
        output,
        ModuleCensus {
            stage: match case.companion_stage() {
                Some(ShaderStage::Vertex) => Stage::Fragment,
                _ => Stage::Vertex,
            },
            subgroup: None,
            strategies: draw
                .reductions
                .iter()
                .map(|&name| name.to_owned())
                .collect(),
            precision: state.precision_notes().into_iter().collect(),
            containment: None,
        },
    ))
}

/// The case's `MTLRenderPipelineDescriptor` as the Metal oracle makes it
/// (`graphics::pipeline_descriptor`): each stage's function from the companion's library where
/// the companion generates that stage, and otherwise the case's own from the row's library,
/// specialized with its constants and linking each function of the modules it links
/// (`acceptance::linked_functions`), as a kernel's run makes its kernel.
fn descriptor(
    case: &RenderCase,
    draw: &RuntimeDraw<'_>,
) -> Result<RenderPipelineDescriptor, ExecutorError> {
    let own = Library::from_air(draw.bitcode).map_err(|error| product_refusal(&error))?;
    let companion = draw
        .companion
        .map(Library::from_air)
        .transpose()
        .map_err(|error| product_refusal(&error))?;
    let function = |stage: ShaderStage| -> Result<Option<FunctionDescriptor>, ExecutorError> {
        let Some(function) = case.function(stage) else {
            return Ok(None);
        };
        if case.companion_stage() == Some(stage) {
            let library = companion.as_ref().ok_or_else(|| {
                ExecutorError::Failed(format!("{stage:?}'s companion has no library"))
            })?;
            let made = library
                .function(&function.entry)
                .map_err(|error| product_refusal(&error))?;
            return Ok(Some(FunctionDescriptor::new(made).with_specialized(true)));
        }
        let made = own
            .function(&function.entry)
            .map_err(|error| product_refusal(&error))?;
        let libraries: Vec<&[u8]> = draw
            .linked
            .iter()
            .map(|module| module.bitcode.as_slice())
            .collect();
        let linked: Vec<crate::acceptance::LinkedLibrary> = draw
            .linked
            .iter()
            .map(|module| module.library.clone())
            .collect();
        let (linked, _) =
            crate::acceptance::linked_functions(&libraries, &linked, &function.constants)
                .map_err(|error| product_refusal(&error))?;
        let mut made = FunctionDescriptor::new(made)
            .with_specialized(true)
            .with_linked(linked);
        made.constants = function.constants.clone();
        Ok(Some(made))
    };
    let vertex = function(ShaderStage::Vertex)?
        .ok_or_else(|| ExecutorError::Failed("the case draws no vertex function".into()))?;
    let fragment = function(ShaderStage::Fragment)?;
    let mut descriptor = graphics::pipeline_descriptor(case, vertex, fragment)
        .map_err(|error| product_refusal(&error))?;
    // The oracle's rasterization-disabled pipeline declares its sink's format at index 0.
    if case.fragment.is_none() {
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(SINK));
    }
    Ok(descriptor)
}

/// What each stage of a draw binds: its buffers, by Metal index, and its textures and samplers.
struct Stages {
    stages: Vec<(ShaderStage, BTreeMap<u32, Placed>, Bound)>,
}

impl Stages {
    /// Each stage's buffers (the case's own stage's as `draw` pads them, a companion's as the case
    /// holds them) and its textures and samplers, made at the run's next indices.
    fn make(
        device: &rt::Device,
        device_type: DeviceType,
        (case, draw): (&RenderCase, &RuntimeDraw<'_>),
        indices: &mut Indices,
    ) -> Result<Stages, ExecutorError> {
        let no_usages = BTreeMap::new();
        let no_member_usages = BTreeMap::new();
        let no_members = BTreeMap::new();
        let no_member_samplers = BTreeMap::new();
        let mut stages = Vec::new();
        for (stage, function) in case.stages() {
            let own = case.companion_stage() != Some(stage);
            let resources = &function.resources;
            let buffers = match own {
                true => draw.buffers,
                false => &resources.buffers,
            };
            let placed = buffers
                .iter()
                .map(|(&index, bytes)| {
                    let what = format!("{stage:?} buffer {index}");
                    Ok((index, Placed::new(device, &what, bytes)?))
                })
                .collect::<Result<BTreeMap<u32, Placed>, ExecutorError>>()?;
            let wanted = Wanted {
                textures: &resources.textures,
                samplers: &resources.samplers,
                members: &no_members,
                member_samplers: &no_member_samplers,
                texture_usages: match own {
                    true => draw.texture_usages,
                    false => &no_usages,
                },
                member_usages: &no_member_usages,
            };
            let bound =
                runtime_textures::make(device, device_type, &wanted, indices, runtime_error)?;
            stages.push((stage, placed, bound));
        }
        Ok(Stages { stages })
    }

    /// What `stage` binds of textures and samplers, where the draw has that stage.
    fn of(&self, stage: ShaderStage) -> Option<&Bound> {
        self.stages
            .iter()
            .find_map(|(at, _, bound)| (*at == stage).then_some(bound))
    }

    /// Bind every stage's buffers, textures and samplers on `encoder`.
    fn bind(&self, encoder: &mut rt::RenderCommandEncoder) -> Result<(), ExecutorError> {
        let fail = |what: &'static str| move |error: rt::Error| runtime_error(what, error);
        for (stage, buffers, bound) in &self.stages {
            let vertex = *stage == ShaderStage::Vertex;
            for (&index, placed) in buffers {
                match vertex {
                    true => encoder.set_vertex_buffer(Some(&placed.buffer), 0, index),
                    false => encoder.set_fragment_buffer(Some(&placed.buffer), 0, index),
                }
                .map_err(fail("a buffer"))?;
            }
            for (&index, texture) in &bound.textures {
                match vertex {
                    true => encoder.set_vertex_texture(Some(texture), index),
                    false => encoder.set_fragment_texture(Some(texture), index),
                }
                .map_err(fail("a texture"))?;
            }
            for (&index, sampler) in &bound.samplers {
                match vertex {
                    true => encoder.set_vertex_sampler_state(Some(sampler), index),
                    false => encoder.set_fragment_sampler_state(Some(sampler), index),
                }
                .map_err(fail("a sampler"))?;
            }
        }
        Ok(())
    }

    /// Every stage's buffers' bytes now, by stage and Metal index.
    fn finals(&self) -> BTreeMap<(ShaderStage, u32), Vec<u8>> {
        self.stages
            .iter()
            .flat_map(|(stage, buffers, _)| {
                buffers
                    .iter()
                    .map(move |(&index, placed)| ((*stage, index), placed.bytes()))
            })
            .collect()
    }
}

/// The pass's color attachments, by Metal index, and its depth/stencil attachment's aspect.
struct Targets {
    textures: BTreeMap<u32, rt::Texture>,
    depth_stencil: Option<DepthStencil>,
    _grant: Option<rt::IndexGrant>,
}

/// A depth/stencil attachment of one sample and one aspect, as the oracle makes it (Metal's
/// separate `Depth32Float` and `Stencil8`): the runtime renders into no separate depth and stencil
/// textures in one pass (`Gap::RenderPass` of `DepthStencilViews`), so a case of both stays on the
/// executor's own context (`engine::runs_render_through_runtime`).
struct DepthStencil {
    texture: rt::Texture,
    depth: bool,
    dimensions: [u32; 2],
}

impl Targets {
    /// Each color target of `case` as a Shared 2D texture of its format and extent, made for
    /// rendering at the next of `indices` and written from its bytes; a rasterization-disabled
    /// draw's one attachment is the oracle's 1x1 `R8Unorm` one at index 0, which no draw writes
    /// (a case of that draw authors no color target, `RasterizationDisabledRenderTarget`).
    fn make(
        device: &rt::Device,
        device_type: DeviceType,
        case: &RenderCase,
        indices: &mut Indices,
    ) -> Result<Targets, ExecutorError> {
        if case.fragment.is_none() {
            let (grant, at) =
                runtime_textures::texture_indices(device, indices, 1, &runtime_error)?;
            let sink = attachment(device, SINK, [1, 1], (0, at.start))?;
            return Ok(Targets {
                textures: BTreeMap::from([(0, sink)]),
                depth_stencil: None,
                _grant: grant,
            });
        }
        let aspects = usize::from(case.depth_stencil.is_some());
        let (grant, mut at) = runtime_textures::texture_indices(
            device,
            indices,
            case.color_targets.len() + aspects,
            &runtime_error,
        )?;
        let depth_stencil = case
            .depth_stencil
            .as_ref()
            .map(|authored| {
                let index = at.next().ok_or_else(|| {
                    ExecutorError::Failed(
                        "no texture index for the depth/stencil attachment".into(),
                    )
                })?;
                DepthStencil::make(device, device_type, authored, index)
            })
            .transpose()?;
        let mut textures = BTreeMap::new();
        for ((&index, target), at) in case.color_targets.iter().zip(at) {
            let format = resources::pixel_format(target.format);
            let [width, height] = target.dimensions;
            let texture = attachment(device, format, [width, height], (index, at))?;
            let row = width as usize * target.format.bytes_per_pixel();
            texture
                .replace_region(
                    whole([width, height]),
                    0,
                    0,
                    target.bytes.as_slice().into(),
                    row as u64,
                    (row * height as usize) as u64,
                )
                .map_err(|error| runtime_error(&format!("color target {index}"), error))?;
            textures.insert(index, texture);
        }
        Ok(Targets {
            textures,
            depth_stencil,
            _grant: grant,
        })
    }

    /// The pass: each attachment loaded and stored.
    fn pass(&self) -> rt::RenderPassDescriptor {
        let mut pass = rt::RenderPassDescriptor::default();
        for (&index, texture) in &self.textures {
            let attachment = &mut pass.color_attachments[index as usize];
            attachment.texture = Some(texture.clone());
            attachment.load_action = rt::LoadAction::Load;
            attachment.store_action = rt::StoreAction::Store;
        }
        match &self.depth_stencil {
            Some(made) if made.depth => {
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(made.texture.clone());
                attachment.load_action = rt::LoadAction::Load;
                attachment.store_action = rt::StoreAction::Store;
            }
            Some(made) => {
                let attachment = &mut pass.stencil_attachment;
                attachment.texture = Some(made.texture.clone());
                attachment.load_action = rt::LoadAction::Load;
                attachment.store_action = rt::StoreAction::Store;
            }
            None => {}
        }
        pass
    }

    /// The depth/stencil attachment's aspect with its bytes now, where the pass has one.
    fn depth_stencil_finals(&self) -> Result<Option<LiteralDepthStencil>, ExecutorError> {
        let Some(made) = &self.depth_stencil else {
            return Ok(None);
        };
        let [width, height] = made.dimensions;
        let row = width as usize * if made.depth { 4 } else { 1 };
        let image = row * height as usize;
        let mut bytes = vec![0u8; image];
        made.texture
            .get_bytes(
                &mut bytes,
                row as u64,
                image as u64,
                whole([width, height]),
                0,
                0,
            )
            .map_err(|error| runtime_error("the depth/stencil attachment's read-back", error))?;
        let (depth, stencil) = match made.depth {
            true => (Some(bytes), None),
            false => (None, Some(bytes)),
        };
        Ok(Some(LiteralDepthStencil {
            dimensions: made.dimensions,
            samples: 1,
            depth,
            stencil,
        }))
    }

    /// Each authored color target with its bytes now, by Metal index.
    fn finals(&self, case: &RenderCase) -> Result<BTreeMap<u32, ColorTarget>, ExecutorError> {
        case.color_targets
            .iter()
            .map(|(&index, authored)| {
                let texture = self.textures.get(&index).ok_or_else(|| {
                    ExecutorError::Failed(format!("color target {index} is not made"))
                })?;
                let [width, height] = authored.dimensions;
                let row = width as usize * authored.format.bytes_per_pixel();
                let image = row * height as usize;
                let mut bytes = vec![0u8; image];
                texture
                    .get_bytes(
                        &mut bytes,
                        row as u64,
                        image as u64,
                        whole([width, height]),
                        0,
                        0,
                    )
                    .map_err(|error| {
                        runtime_error(&format!("color target {index} read-back"), error)
                    })?;
                Ok((
                    index,
                    ColorTarget {
                        bytes,
                        ..authored.clone()
                    },
                ))
            })
            .collect()
    }
}

impl DepthStencil {
    /// `authored`'s one aspect at texture index `at` of the run's grant: a `Depth32Float` filled
    /// with its depths by a draw (`runtime_depth`), or a `Stencil8` written from its bytes.
    fn make(
        device: &rt::Device,
        device_type: DeviceType,
        authored: &LiteralDepthStencil,
        at: u32,
    ) -> Result<DepthStencil, ExecutorError> {
        let [width, height] = authored.dimensions;
        let what = "the depth/stencil attachment";
        match (&authored.depth, &authored.stencil) {
            (Some(depths), None) => {
                let texture = aspect(device, rt::PixelFormat::Depth32Float, [width, height], at)?;
                let fill = runtime_depth::Fill {
                    texture: texture.clone(),
                    level: 0,
                    slice: 0,
                    width,
                    depths: depths.clone(),
                };
                runtime_depth::filled(device, device_type, &[fill], &runtime_error)?;
                Ok(DepthStencil {
                    texture,
                    depth: true,
                    dimensions: authored.dimensions,
                })
            }
            (None, Some(stencil)) => {
                let texture = aspect(device, rt::PixelFormat::Stencil8, [width, height], at)?;
                texture
                    .replace_region(
                        whole([width, height]),
                        0,
                        0,
                        stencil.as_slice().into(),
                        u64::from(width),
                        u64::from(width) * u64::from(height),
                    )
                    .map_err(|error| runtime_error(what, error))?;
                Ok(DepthStencil {
                    texture,
                    depth: false,
                    dimensions: authored.dimensions,
                })
            }
            _ => Err(ExecutorError::Failed(format!(
                "{what} through the runtime holds one aspect"
            ))),
        }
    }
}

/// A Shared 2D depth or stencil attachment of `format` and `extent` at texture index `at` of the
/// run's grant.
fn aspect(
    device: &rt::Device,
    format: rt::PixelFormat,
    [width, height]: [u32; 2],
    at: u32,
) -> Result<rt::Texture, ExecutorError> {
    let mut descriptor = rt::TextureDescriptor::new(rt::TextureType::Type2D, format, width, height);
    descriptor.usage = rt::TextureUsage::RENDER_TARGET;
    descriptor.storage_mode = rt::StorageMode::Shared;
    device
        .new_texture(&descriptor, Some(at))
        .map_err(|error| runtime_error("the depth/stencil attachment", error))
}

/// A Shared 2D color attachment of `format` and `extent`, color attachment `index`'s, at texture
/// index `at` of the run's grant.
fn attachment(
    device: &rt::Device,
    format: rt::PixelFormat,
    [width, height]: [u32; 2],
    (index, at): (u32, u32),
) -> Result<rt::Texture, ExecutorError> {
    let mut descriptor = rt::TextureDescriptor::new(rt::TextureType::Type2D, format, width, height);
    descriptor.usage = rt::TextureUsage::RENDER_TARGET;
    descriptor.storage_mode = rt::StorageMode::Shared;
    device
        .new_texture(&descriptor, Some(at))
        .map_err(|error| runtime_error(&format!("color target {index}"), error))
}

/// The region of a whole 2D level of `extent`.
fn whole([width, height]: [u32; 2]) -> rt::Region {
    rt::Region {
        origin: [0, 0, 0],
        size: [width, height, 1],
    }
}

/// Encode `case`'s draw: its primitives, direct, or indexed through `index_buffer`.
fn drawn(
    encoder: &mut rt::RenderCommandEncoder,
    case: &RenderCase,
    index_buffer: Option<&Placed>,
) -> Result<(), ExecutorError> {
    let Assembly::Primitives(primitive) = case.assembly else {
        return Err(ExecutorError::Failed(
            "a draw through the runtime draws primitives".into(),
        ));
    };
    let primitive = crate::case::indirect_commands::metal_primitive_type(primitive);
    match case.draw {
        DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => encoder.draw_primitives(
            primitive,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        ),
        DrawCall::Indexed {
            index_type,
            index_count,
            index_offset,
            instance_count,
            base_vertex,
            base_instance,
            ..
        } => {
            let buffer = index_buffer.ok_or_else(|| {
                ExecutorError::Failed("an indexed draw without its index buffer".into())
            })?;
            let index_type = if index_type == metal2vulkan::vk::IndexType::UINT16 {
                rt::IndexType::UInt16
            } else if index_type == metal2vulkan::vk::IndexType::UINT32 {
                rt::IndexType::UInt32
            } else {
                return Err(ExecutorError::Failed(format!(
                    "an indexed draw of index type {index_type:?}"
                )));
            };
            encoder.draw_indexed_primitives(
                primitive,
                index_count,
                rt::IndexBuffer {
                    buffer: &buffer.buffer,
                    offset: index_offset,
                    index_type,
                },
                instance_count,
                base_vertex,
                base_instance,
            )
        }
        _ => {
            return Err(ExecutorError::Failed(
                "a draw through the runtime is direct or indexed".into(),
            ))
        }
    }
    .map_err(|error| runtime_error("the draw", error))
}
