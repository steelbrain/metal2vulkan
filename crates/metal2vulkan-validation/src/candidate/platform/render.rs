//! The engine render candidate on a Vulkan device.
//!
//! One draw through one `RenderPipeline`, as a consumer records it: the plan's descriptor writes
//! (`plan_resources`), vertex buffers at the plan's bindings and offsets, each stage's push bytes
//! with the root-table and status holes filled, the pipeline's `GraphicsState` (formats, write
//! masks, blend state, topology class, rasterizer discard, and every state it makes dynamic), and
//! the Metal render encoder's defaults for the rest:
//!
//! - the viewport is the whole target with a negative height, so framebuffer rows, `FragCoord`,
//!   and derivative signs match Metal's y-up clip space without touching the shader (spec
//!   "Y-flip"); the scissor is the whole target. A rate-mapped pass is drawn on the product's
//!   region path, on every device (Metal's pass over the identity, a near-equivalent over any other
//!   map): the draw is recorded once per region of the map's plan for the map's screen
//!   (`graphics::rate_map_regions`), each with its region's viewport, flipped the same way, and its
//!   scissor, into the targets, which hold the screen and so the compile's physical size;
//! - the front face is the case's front-facing winding (`graphics::RenderCase::
//!   depth_stencil_state`; Metal's default is clockwise, [`super::METAL_DEFAULT_FRONT_FACE`]), and
//!   culling is the case's (`graphics::RenderCase::cull_mode`: a tessellation case's authored one,
//!   and otherwise off);
//! - depth bias and blend constants are Metal's defaults, and each face's stencil masks and
//!   reference the case's depth-stencil state's;
//! - a state the pipeline makes dynamic through extended dynamic state is recorded at the value it
//!   would otherwise take statically: the case's culling and front face, the draw's topology, and
//!   the depth-stencil state below, through Vulkan 1.3's core commands in a context at 1.3 or later
//!   (one restricted to a profile of that version), else `VK_EXT_extended_dynamic_state`'s
//!   (`device::ExtendedDynamicState`); a per-bind vertex stride (`VERTEX_INPUT_BINDING_STRIDE`, a
//!   layout of `MTLBufferLayoutStrideDynamic`) is recorded as the vertex buffers are bound, every
//!   binding's stride the binder's (`DrawCommands::vertex_strides`);
//! - every color attachment, and a fragment case's depth/stencil attachment (one Vulkan
//!   attachment for Metal's separate depth and stencil ones, holding each authored aspect), is
//!   loaded before and stored after the draw, and the depth-stencil state is the one the Metal
//!   oracle sets: the case's authored state, or the one the fragment function's live outputs
//!   imply (`executor_contract::DrawDepthStencil::of_case`);
//! - a color attachment the fragment function reads (framebuffer fetch) is also the input
//!   attachment of its index: loaded with its authored contents as Metal's `MTLLoadActionLoad`
//!   loads it, kept in `GENERAL` through the subpass, read through the plan's input-attachment
//!   write, with the render pass's by-region self-dependency and, where the artifact states it,
//!   rasterization-order color access, so each fetch sees the earlier fragments of the draw that
//!   cover its pixel, as in Metal (in a multisampled pass the attachment of the pass's samples,
//!   which a function running once per sample reads at its own sample);
//! - a color attachment the fragment function reads in the interlocked storage form
//!   (`GraphicsState::storage_attachments`) is no attachment of the subpass (an unused reference)
//!   but a single-sample 2D storage image of its own format holding its authored contents, kept in
//!   `GENERAL` from its upload through the draw to its readback and bound where the plan's
//!   `WriteSource::TileAttachment` names it; the function reads and writes it inside the pixel
//!   interlock, and a render-target output of it is that image's readback. In a multisampled pass
//!   it is a target of the pass's samples with `STORAGE` usage, filled and read back sample by
//!   sample as the attachments are and kept in `GENERAL` through the draw, which a function running
//!   once per sample reads and writes at its own sample;
//! - a fragment function that tests and writes the stencil itself (the shader-side stencil,
//!   `GraphicsState::shadow_stencil`) runs as `ShadowStencilUse` asks of a consumer ([`shadow`]):
//!   the pipeline's stencil test is off, the bind state carries the case's depth/stencil
//!   descriptor, and the attachment's stencil is copied into a single-sample storage image before
//!   the render pass and back after it, outside the pass, the image bound in `GENERAL` where the
//!   plan's `WriteSource::ShadowStencil` names it;
//! - a patch draw binds its factor buffer where the plan's `WriteSource::TessellationFactors` names
//!   it, over the draw's range, and draws a patch list: the tessellation pipeline's synthesized
//!   vertex and control stages bind nothing of Metal's, and its evaluation stage binds the vertex
//!   function's resources;
//! - a fragment case's explicit imageblock is the pass's storage image: one coherent `R32Uint`
//!   2D-array image of the pass's extent with a layer per cell word (at least the pipeline's
//!   `words`), holding the cells the Metal oracle's initializer defines, created with `STORAGE`
//!   usage, kept in `GENERAL` from its upload through the draw to its readback, and bound where the
//!   plan's `WriteSource::Imageblock` names it. The pass has the one pipeline, so one master. A
//!   pipeline that reaches no imageblock (the product dropped every view) leaves the cells as the
//!   initializer left them, as Metal's pass does.
//!
//! The run obeys the compute executor's rules (status before output, the validation layer); a
//! pipeline whose modules loop draws once, as compiled, like any other.

use super::acceptance::{unmet_requirements, LayoutError, LayoutObjects};
use super::buffers::{create_host_buffer, HostBuffer};
use super::compute::{
    allocate, allocate_in, attempt, check_backend, command_replay, environment, executed_census,
    incompatible_binding, module_censuses, opened_device, refused_attempt, replay_commands,
    table_bytes, upload_root_table, Allocation, CompileChild, Live, OnDevice, Replay, ReplayPath,
};
use super::device::{with_validated_context, ExtendedDynamicState, VulkanContext};
use super::graphics_pipeline::{self, GraphicsError};
use super::plan_resources::{barrier, create_image, layers, Guest, PlanResources, TextureKey};
use crate::candidate::engine::argument_buffers::{
    field_words, owner_bytes, records_storage, validate_records,
};
use crate::candidate::engine::graphics::{
    self, stage_of, Assembly, ColorTarget, RenderCase, RenderFinals, RenderOutput,
};
use crate::candidate::engine::guards::{declared_reach, with_escaped, GuardCount, Placement};
use crate::candidate::engine::{
    bind_refusal, draw_out_of_bounds_run, engine_fingerprint, resources, status_verdict,
    BufferAddress, CompiledCase, ComputeRun, RunAttempt,
};
use crate::candidate::engine::{index_padding_agrees, ChildCommand, CompiledChild};
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::case::{ResourceRole, TextureFormat, TextureType};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::multisample::AttachmentTransfer;
use crate::executor_contract::out_of_bounds::OutOfBoundsRun;
use crate::gpu::wait::{with_submission_label, SubmissionLabel};
use crate::gpu::DeviceType;
use crate::hash::sha256_bytes;
use crate::literal::{LiteralDepthStencil, LiteralTexture};
use crate::observation::Backend;
use crate::worker::compile::WorkerDevice;
use ash::vk;
use metal2vulkan::bind::{
    BindShape, BindState, BindingPlan, DrawCommands, ExecutionStatus, PushHole, RateMapRegions,
    SamplerDescriptor, StatusReport, DRAW_ARGUMENT_BYTES,
};
use metal2vulkan::device::Device;
use metal2vulkan::device::VkFeature;
use metal2vulkan::pipeline::{GraphicsState, RenderPipeline, StageModule, StorageAttachmentView};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};
use std::collections::BTreeMap;
use std::sync::Arc;

fn not_implemented(what: impl Into<String>) -> ExecutorError {
    NotSupported::NotImplemented(format!("engine candidate: {}", what.into())).into()
}

/// Compile `case` for the device `device_type` opens with `compile` (given the opened device's
/// description and policy and the bind shape the draw binds, `graphics::render_shape`; the tools'
/// compile worker; it returns the pipeline and its visible-function tables' bytes), plan and
/// execute its draw, and read the status before the output. Every refusal is typed; only a device,
/// driver, or tool failure is `Failed`. `children` compiles the template each command the case
/// executes runs, in a compile worker.
pub fn execute_engine_render(
    device_type: DeviceType,
    case_id: &str,
    case: &RenderCase,
    compile: impl FnOnce(
        &WorkerDevice,
        &BindShape,
    ) -> Result<CompiledCase<RenderPipeline>, ExecutorError>,
    children: impl FnMut(
        &WorkerDevice,
        u64,
        &BindShape,
        &ChildCommand,
    ) -> Result<CompiledChild, ExecutorError>,
    backend: Backend,
) -> RunAttempt {
    if let Some(refused) = refused_attempt(device_type, "engine-render") {
        return refused;
    }
    with_validated_context(device_type, |context| {
        let result = run(context, case_id, case, (compile, children), backend);
        Ok(attempt(context, result, "engine-render"))
    })
    .unwrap_or_else(|error| RunAttempt::of(Err(ExecutorError::Failed(error))))
}

fn run(
    context: &VulkanContext,
    case_id: &str,
    case: &RenderCase,
    (compile, mut children): (
        impl FnOnce(&WorkerDevice, &BindShape) -> Result<CompiledCase<RenderPipeline>, ExecutorError>,
        impl FnMut(
            &WorkerDevice,
            u64,
            &BindShape,
            &ChildCommand,
        ) -> Result<CompiledChild, ExecutorError>,
    ),
    backend: Backend,
) -> Result<ComputeRun, ExecutorError> {
    check_backend(context, backend)?;
    let target = opened_device(context)?;
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let shape = graphics::render_shape(case, addressed).map_err(ExecutorError::Failed)?;
    let compiled = compile(&target, &shape)?;
    let pipeline = &compiled.pipeline;
    let device = target.device();
    let missing = unmet_requirements(context, &pipeline.requirements);
    if !missing.is_empty() {
        return Err(NotSupported::NotEnabled(missing).into());
    }
    // A rate-mapped pass's regions, planned for the device's viewport limits and this pipeline
    // before anything is allocated.
    let rate_map = case.rate_map_pass.as_ref();
    let regions = rate_map
        .map(|map| graphics::rate_map_regions(map, &context.region_limits(), &pipeline.graphics))
        .transpose()?;
    // How the Metal vertex function's module contains each out-of-bounds target on this device,
    // decided before anything runs; a form the device leaves uncontained is refused, never run.
    let out_of_bounds =
        draw_out_of_bounds_run(context.execution_class, device.profile(), pipeline, case)?;
    let spv_sha256 = sha256_bytes(
        &pipeline
            .stages
            .iter()
            .flat_map(StageModule::spirv_bytes)
            .collect::<Vec<_>>(),
    );
    let label = SubmissionLabel {
        case_id: case_id.into(),
        module_sha256: spv_sha256.clone(),
    };
    let mut compile_child = |token: u64, shape: &BindShape, command: &ChildCommand| {
        children(&target, token, shape, command)
    };
    let executed = with_submission_label(label, || {
        execute(
            context,
            device,
            (&compiled, &shape),
            (case, regions.as_ref()),
            out_of_bounds.as_ref(),
            &mut compile_child,
        )
    });
    // Every object of the run is destroyed by now, so destruction reports here too.
    let messages = context.take_messages();
    if !messages.is_empty() {
        let run = match &executed {
            Ok(_) => String::new(),
            Err(error) => format!(" (the run also failed: {error})"),
        };
        return Err(ExecutorError::Failed(format!(
            "VK_LAYER_KHRONOS_validation reported {} message(s); the run is not evidence{run}: {}",
            messages.len(),
            messages.join(" | ")
        )));
    }
    let Drawn {
        output,
        status,
        escaped,
        children,
        replayed,
    } = executed?;
    // Every stage module drawn, in the order `spv_sha256` hashes them, then each replayed
    // command's, and the guard bytes each out-of-bounds target's run changed.
    let planned = pipeline.plan(device, &shape).ok();
    let mut modules = module_censuses(&pipeline.stages, device.profile(), planned.as_ref())?;
    modules.extend(children);
    let mut census = executed_census(
        context,
        modules,
        out_of_bounds.map(|run| with_escaped(run, &escaped, Placement::Allocation)),
    );
    census.rate_map_pass = rate_map
        .zip(regions.as_ref())
        .map(|(map, plan)| graphics::rate_map_pass_census(map, plan));
    let mut environment = environment(context, device, "engine-render");
    command_replay(&mut environment, &replayed);
    Ok(ComputeRun {
        output,
        spv_sha256,
        translator_fingerprint: engine_fingerprint(device.profile()),
        environment,
        status,
        comparison: census.comparison_class(),
        census,
    })
}

/// Allocate the case's buffers (argument-buffer buffers first, so each owner is created with
/// their addresses, its textures' and samplers' resource ids, and its command buffers',
/// [`field_words`],
/// [`owner_bytes`]), visible-function tables (holding the compile's table bytes), and vertex
/// buffers, write the commands the host encodes into their record storage, plan and instantiate
/// its draw, create what every plan write names (argument-buffer textures, heap arrays, and
/// handle tables included), the color targets, and the pipeline, record and submit the draw,
/// wait, read the status, check the records, execute the commands the case executes (`compile`
/// compiling each child in a compile worker), and read the output; with the guard and padding bytes
/// the draw changed around each out-of-bounds target (`out_of_bounds`), by target, and the census
/// of each module a replayed command executed.
fn execute(
    context: &VulkanContext,
    device: &Device,
    (compiled, compiled_for): (&CompiledCase<RenderPipeline>, &BindShape),
    (case, regions): (&RenderCase, Option<&RateMapRegions>),
    out_of_bounds: Option<&OutOfBoundsRun>,
    compile: &mut CompileChild<'_>,
) -> Result<Drawn, ExecutorError> {
    let CompiledCase { pipeline, tables } = compiled;
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let slot = |stage, class, index| SlotRef {
        stage,
        class,
        index,
    };
    // With a device address an allocation's guest address lies a fixed offset above it
    // (`BufferAddress::addressed`), else at its synthetic base.
    let own_guest = |allocation: &Allocation, synthetic: BufferAddress| match allocation.address {
        Some(host) => BufferAddress::addressed(host).map_err(ExecutorError::Failed),
        None => Ok(synthetic),
    };
    let synthetic_used = graphics::synthetic_used_addresses(case).map_err(ExecutorError::Failed)?;
    let records = graphics::records_ids(case).map_err(ExecutorError::Failed)?;
    let mut embedded: BTreeMap<(Stage, u32, u32), Allocation> = BTreeMap::new();
    let mut embedded_addresses = BTreeMap::new();
    for (key, bytes) in case.argument_buffer_buffers() {
        let (stage, owner, field) = key;
        let label = format!("{stage:?} argument-buffer buffer {owner}+{field}");
        let allocation = allocate(
            context,
            &label,
            bytes,
            addressed,
            vk::BufferUsageFlags::empty(),
        )?;
        // A command buffer's record storage lies at its ID's records address.
        let address = match records.get(&key) {
            Some(&id) => records_storage(id, allocation.address),
            None => own_guest(&allocation, synthetic_used[&key])?,
        };
        embedded_addresses.insert(key, address);
        embedded.insert(key, allocation);
    }
    let tokens = graphics::texture_tokens(case).map_err(ExecutorError::Failed)?;
    let samplers = graphics::sampler_ids(case).map_err(ExecutorError::Failed)?;
    let words = field_words(
        &embedded_addresses,
        &records,
        &tokens,
        &samplers,
        case.objects(),
    )
    .map_err(ExecutorError::Failed)?;
    let mut buffers: BTreeMap<(Stage, u32), Allocation> = BTreeMap::new();
    for (stage, function) in case.stages() {
        let resources = &function.resources;
        let fields = words
            .iter()
            .filter(|((of, ..), _)| *of == stage)
            .map(|(&(_, owner, field), &word)| ((owner, field), word))
            .collect();
        for &index in resources.buffers.keys() {
            let bytes =
                owner_bytes(&resources.buffers, index, &fields).map_err(ExecutorError::Failed)?;
            let label = format!("{stage:?} buffer {index}");
            let allocation = allocate(
                context,
                &label,
                &bytes,
                addressed,
                vk::BufferUsageFlags::empty(),
            )?;
            buffers.insert((stage, index), allocation);
        }
    }
    let mut table_buffers: BTreeMap<(Stage, u32), Allocation> = BTreeMap::new();
    for (stage, function) in case.stages() {
        for (&index, table) in &function.resources.function_tables {
            let bytes = table_bytes(tables, stage, index, table.byte_len())?;
            let label = format!("{stage:?} visible function table {index}");
            let allocation = allocate(
                context,
                &label,
                bytes,
                addressed,
                vk::BufferUsageFlags::empty(),
            )?;
            table_buffers.insert((stage, index), allocation);
        }
    }
    // A patch draw's factors, read by the control stage as a storage buffer.
    let factors = match &case.assembly {
        Assembly::Patches { factors, .. } => Some(allocate(
            context,
            "tessellation factors",
            factors,
            addressed,
            vk::BufferUsageFlags::empty(),
        )?),
        Assembly::Primitives(_) => None,
    };
    // An out-of-bounds target is allocated inside its guard.
    for vertex in &case.vertex_buffers {
        let label = format!("vertex buffer {}", vertex.buffer_index);
        let allocation = allocate_in(
            context,
            &label,
            &vertex.bytes,
            addressed,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            declared_reach(
                &case.out_of_bounds,
                OutOfBoundsResource::AttributeBuffer {
                    location: vertex.location,
                },
            ),
        )?;
        if buffers
            .insert((Stage::Vertex, vertex.buffer_index), allocation)
            .is_some()
        {
            return Err(format!(
                "vertex buffer {} is also a bound buffer",
                vertex.buffer_index
            )
            .into());
        }
    }
    let addresses: BTreeMap<(Stage, u32), BufferAddress> = buffers
        .iter()
        .chain(&table_buffers)
        .map(|(&(stage, index), allocation)| {
            let address = own_guest(allocation, graphics::synthetic_address(stage, index))?;
            Ok(((stage, index), address))
        })
        .collect::<Result<_, ExecutorError>>()?;
    // The guest address of a case buffer of the stage that owns a command buffer.
    let case_address = |(stage, _, _): (Stage, u32, u32), buffer: u32| {
        addresses.get(&(stage, buffer)).map(|a| a.guest)
    };
    // The commands the host encodes, written into their record storage before the producer runs,
    // each buffer at its case buffer's guest address.
    for (key, command) in case.command_buffers() {
        if command.commands.is_empty() {
            continue;
        }
        let storage = embedded
            .get(&key)
            .ok_or_else(|| format!("command buffer {key:?} has no record storage"))?;
        // SAFETY: the storage was created for this run and nothing has been submitted yet.
        let mapped = unsafe { storage.buffer.map()? };
        command
            .encode_host_records(mapped.bytes, |buffer| case_address(key, buffer))
            .map_err(ExecutorError::Failed)?;
    }
    let state = graphics::bind_state(case, &addresses, &embedded_addresses)
        .map_err(ExecutorError::Failed)?;
    // The compile settled the plan's variant for this shape (`variants::compile_render_for_shape`):
    // planning is a function of the pipeline, the device's profile, and the shape, so the same
    // shape plans the variant the pipeline was compiled with.
    let shape = state.shape();
    if shape != *compiled_for {
        return Err(ExecutorError::Failed(format!(
            "the draw binds shape {shape:?}, not the {compiled_for:?} it was compiled for"
        )));
    }
    let plan = pipeline
        .plan(device, &shape)
        .map_err(|e| bind_refusal(&e))?;
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    // An indexed draw whose positions run past its index buffer reads a copy of its indices
    // padded with zeros, as the runtime's does.
    let (index_bytes, draw, padding) = match &case.index_buffer {
        None => (None, case.draw, None),
        Some(bytes) => {
            let (bytes, draw, padding) =
                graphics::padded_index_draw(case.draw, bytes).map_err(ExecutorError::Failed)?;
            (Some(bytes), draw, padding)
        }
    };
    let commands = plan
        .instantiate_draw(&state, &draw)
        .map_err(|e| bind_refusal(&e))?;
    if commands.indirect_needs_arguments {
        return Err(not_implemented("an indirect draw read on the CPU"));
    }
    // The draw contains the index positions past its index buffer as the target's containment
    // states.
    index_padding_agrees(out_of_bounds, padding)?;
    // An indexed draw's index buffer, inside its guard when it is an out-of-bounds target.
    let index_buffer = match &index_bytes {
        None => None,
        Some(bytes) => Some(allocate_in(
            context,
            "index buffer",
            bytes,
            addressed,
            vk::BufferUsageFlags::INDEX_BUFFER,
            declared_reach(&case.out_of_bounds, OutOfBoundsResource::IndexBuffer),
        )?),
    };
    let status = allocate(
        context,
        "execution status",
        &ExecutionStatus::initial(),
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let guest_buffers: BTreeMap<SlotRef, vk::Buffer> = buffers
        .iter()
        .chain(&table_buffers)
        .map(|(&(stage, index), allocation)| {
            (
                slot(stage, SlotClass::Buffer, index),
                allocation.buffer.buffer,
            )
        })
        .collect();
    let mut textures: BTreeMap<SlotRef, &LiteralTexture> = BTreeMap::new();
    let mut samplers: BTreeMap<SlotRef, SamplerDescriptor> = BTreeMap::new();
    for (stage, function) in case.stages() {
        for (&index, texture) in &function.resources.textures {
            textures.insert(slot(stage, SlotClass::Texture, index), texture);
        }
        for (&index, sampler) in &function.resources.samplers {
            samplers.insert(
                slot(stage, SlotClass::Sampler, index),
                resources::sampler_descriptor(sampler),
            );
        }
    }
    let used_textures: BTreeMap<u32, &LiteralTexture> = case
        .argument_buffer_textures()
        .map(|(key, texture)| (tokens[&key], texture))
        .collect();
    let mut constant_blobs: BTreeMap<Stage, Arc<[u8]>> = BTreeMap::new();
    let tessellated = pipeline.graphics.tessellation.is_some();
    for module in &pipeline.stages {
        if let Some(blob) = &module.constant_blob {
            let stage = stage_of(module, tessellated)?.ok_or_else(|| {
                format!(
                    "the synthesized {:?} stage carries a constant blob",
                    module.stage
                )
            })?;
            constant_blobs.insert(stage, blob.clone());
        }
    }
    // A pass of one sample's storage-form attachments: the plan's textures.
    let storage_attachments = storage_attachments(&pipeline.graphics, case)?;
    let observed = match case.output {
        RenderOutput::Texture { stage, binding, .. } => {
            Some(TextureKey::Slot(slot(stage, SlotClass::Texture, binding)))
        }
        RenderOutput::ArgumentBufferTexture {
            stage,
            owner,
            field_offset,
            ..
        } => tokens
            .get(&(stage, owner, field_offset))
            .copied()
            .map(TextureKey::Used),
        RenderOutput::Imageblock { .. } => Some(TextureKey::Imageblock),
        // A color attachment the fragment function reads in the interlocked storage form in a
        // pass of one sample is the storage texture the plan views; a multisampled pass's is a
        // target, read back as the others are.
        RenderOutput::RenderTarget { index, .. } if storage_attachments.contains_key(&index) => {
            Some(TextureKey::TileAttachment(index))
        }
        RenderOutput::None
        | RenderOutput::Buffer { .. }
        | RenderOutput::ArgumentBufferBuffer { .. }
        | RenderOutput::RenderTarget { .. }
        | RenderOutput::Depth { .. }
        | RenderOutput::Stencil { .. } => None,
    };
    // The attachments first: a framebuffer-fetch input's descriptor views its color attachment.
    let targets = Targets::create(context, &pipeline.graphics, case)?;
    let readers = multisampled::SampleReaders::create(context, &targets)?;
    let imageblock = imageblock_storage(&pipeline.graphics, case, targets.extent)?;
    // The shader-side stencil's shadow of the stencil attachment.
    let shadow = match (&pipeline.graphics.shadow_stencil, &targets.depth_stencil) {
        (Some(used), Some(target)) => Some(shadow::ShadowStencil::create(context, used, target)?),
        (Some(_), None) => {
            return Err(ExecutorError::Failed(
                "the pipeline keeps a shadow of a stencil attachment the draw lacks".into(),
            ))
        }
        (None, _) => None,
    };
    // The sample planes of each storage-form attachment the plan views through them.
    let planes = sample_planes(context, &pipeline.graphics, case, &targets)?;
    // Every target's view: a color attachment's, which an input-attachment write reads, and a
    // multisampled pass's storage-form attachment's, which a tile-attachment write views, or its
    // planes' 2D-array view where the plan views those.
    let attachments: BTreeMap<u32, vk::ImageView> = targets
        .targets
        .iter()
        .map(|(&index, target)| (index, target.view))
        .chain(planes.iter().map(|p| (p.color_index, p.view)))
        .collect();
    let mut resources = PlanResources::create(
        context,
        device,
        &plan.descriptors,
        &Guest {
            buffers: &guest_buffers,
            textures: &textures,
            // A render case's out-of-bounds targets are attribute and index buffers, none a texture
            // buffer.
            texel_guards: &BTreeMap::new(),
            used_textures: &used_textures,
            samplers: &samplers,
            argument_samplers: &state.argument_samplers,
            constant_blobs: &constant_blobs,
            descriptor_ranges: &commands.descriptor_ranges,
            descriptor_bytes: &commands.descriptor_bytes,
            // `graphics::render_case` refuses a case's acceleration structures.
            structures: &BTreeMap::new(),
            tessellation_factors: factors.as_ref().map(|factors| factors.buffer.buffer),
            attachments: &attachments,
            imageblock: imageblock.as_ref(),
            tile_attachments: &storage_attachments,
            shadow_stencil: shadow.as_ref().map(|shadow| shadow.view),
            status: Some(status.buffer.buffer),
        },
        observed,
    )?;
    // Each stage's resident root table, behind its `RootTableAddress` hole.
    let mut roots = Vec::new();
    for (stages, table) in &commands.root_tables {
        let (table, sink) = upload_root_table(context, table, addressed)?;
        roots.push((vk::ShaderStageFlags::from(*stages), table, sink));
    }
    let push = push_writes(&commands, &roots, &status, &|_| None)?;

    let layout = match LayoutObjects::create(&context.device, &pipeline.layout) {
        Ok(layout) => layout,
        Err(LayoutError::Shape(why) | LayoutError::Refused(why)) => {
            return Err(ExecutorError::Failed(why))
        }
    };
    let mut objects = Objects {
        device: &context.device,
        modules: Vec::new(),
        render_pass: vk::RenderPass::null(),
        framebuffer: vk::Framebuffer::null(),
        pipeline: vk::Pipeline::null(),
        command_pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
        layout,
    };
    create_render_pass(context, &mut objects, &pipeline.graphics, &targets)?;
    // The encoder state the pipeline takes statically, or the draw records where the pipeline
    // makes it dynamic: the draw's topology with Metal's primitive restart for it (a patch list
    // for patches), and the depth-stencil state the oracle sets.
    let (topology, primitive_restart) = graphics::assembly_input(&case.assembly);
    let topology: vk::PrimitiveTopology = topology.into();
    let encoder = EncoderState::of(case, &pipeline.graphics)?;
    create_pipeline(
        context,
        &mut objects,
        pipeline,
        &plan.specialization,
        (topology, primitive_restart),
        encoder,
    )?;
    let sets = resources.write_sets(context, &objects.layout)?;
    // Where the pipeline's strides are dynamic state, every binding is bound with its stride.
    let dynamic_strides = pipeline.graphics.dynamic_states.iter().any(|&state| {
        vk::DynamicState::from(state) == vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE
    });
    let vertex_buffers: Vec<VertexBinding> = plan
        .vertex_buffers
        .iter()
        .map(|bind| {
            let buffer = buffers
                .get(&(Stage::Vertex, bind.buffer_index))
                .ok_or_else(|| {
                    format!("the plan binds unbound vertex buffer {}", bind.buffer_index)
                })?;
            let offset = commands
                .buffer_offsets
                .iter()
                .find(|(binding, _)| *binding == bind.binding)
                .map(|&(_, offset)| offset)
                .ok_or_else(|| format!("no offset for vertex binding {}", bind.binding))?;
            let stride = if dynamic_strides {
                let stride = commands
                    .vertex_strides
                    .iter()
                    .find(|(binding, _)| *binding == bind.binding)
                    .map(|&(_, stride)| u64::from(stride))
                    .ok_or_else(|| format!("no stride for vertex binding {}", bind.binding))?;
                Some(stride)
            } else {
                None
            };
            Ok((bind.binding, buffer.buffer.buffer, offset, stride))
        })
        .collect::<Result<_, ExecutorError>>()?;
    let draws = draws(targets.extent, regions);
    record_and_submit(
        context,
        &mut objects,
        Recording {
            resources: &resources,
            targets: &targets,
            readers: readers.as_ref(),
            planes: &planes,
            draws: &draws,
            sets: &sets,
            push: &push,
            vertex_buffers: &vertex_buffers,
            index_buffer: index_buffer
                .as_ref()
                .map(|allocation| allocation.buffer.buffer),
            dynamic: &pipeline.graphics.dynamic_states,
            topology,
            primitive_restart,
            encoder,
            rasterizer_discard: pipeline.graphics.rasterizer_discard,
            draw: &commands,
            shadow: shadow.as_ref(),
            replayed: None,
        },
    )?;

    // The status first: nothing is read from a run whose indirect count was clamped or that
    // missed the resident set.
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .map_err(|_| "the status buffer is shorter than ExecutionStatus")?;
    let report = status_verdict(&status_bytes)?;
    let mut finals = RenderFinals::default();
    let mut escaped = BTreeMap::new();
    for (&(stage, index), allocation) in &buffers {
        let len = case
            .function(stage)
            .and_then(|f| f.resources.buffers.get(&index))
            .map_or(0, Vec::len);
        // SAFETY: as for the status buffer.
        let bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
        finals.buffers.insert((stage, index), bytes);
        // Only an out-of-bounds target's vertex buffer is guarded. SAFETY: as for the status
        // buffer.
        if let Some(count) = unsafe { allocation.buffer.escaped()? } {
            let location = case
                .vertex_buffers
                .iter()
                .find(|vertex| stage == Stage::Vertex && vertex.buffer_index == index)
                .map(|vertex| vertex.location)
                .ok_or_else(|| format!("a guarded {stage:?} buffer {index} holds no attribute"))?;
            escaped.insert(OutOfBoundsResource::AttributeBuffer { location }, count);
        }
    }
    if let Some(allocation) = &index_buffer {
        // SAFETY: as for the status buffer.
        if let Some(count) = unsafe { allocation.buffer.escaped()? } {
            escaped.insert(OutOfBoundsResource::IndexBuffer, count);
        }
    }
    for (&key, allocation) in &embedded {
        let (stage, owner, field) = key;
        let len = case
            .function(stage)
            .and_then(|f| f.resources.argument_buffer_buffers.get(&(owner, field)))
            .map_or(0, Vec::len);
        // SAFETY: as for the status buffer.
        let bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
        finals.argument_buffer_buffers.insert(key, bytes);
    }
    // Each command buffer's records as the host contract decodes them: an encoding fault fails the
    // run. Then the commands the case executes run, over the run's allocations, and the buffers are
    // read again.
    validate_records(case.command_buffers(), &finals.argument_buffer_buffers)
        .map_err(ExecutorError::Failed)?;
    let (children, replayed) = if case
        .command_buffers()
        .any(|(_, command)| command.execute.is_some())
    {
        // Each allocation at its guest address and the length the guest sees (not the padding).
        let lengths = |stage: Stage, index: u32| {
            case.function(stage).map_or(0, |function| {
                function
                    .resources
                    .buffers
                    .get(&index)
                    .map(|bytes| bytes.len() as u64)
                    .or_else(|| {
                        function
                            .resources
                            .function_tables
                            .get(&index)
                            .map(|table| table.byte_len())
                    })
                    .unwrap_or(0)
            })
        };
        let live = buffers
            .iter()
            .chain(&table_buffers)
            .filter(|(&(stage, index), _)| lengths(stage, index) != 0)
            .map(|(&(stage, index), allocation)| {
                (
                    addresses[&(stage, index)],
                    lengths(stage, index),
                    allocation,
                )
            })
            .chain(embedded.iter().map(|(key, allocation)| {
                (
                    embedded_addresses[key],
                    finals.argument_buffer_buffers[key].len() as u64,
                    allocation,
                )
            }))
            .map(|(address, length, allocation)| Live {
                buffer: allocation.buffer.buffer,
                address,
                length,
            })
            .collect::<Vec<_>>();
        // A render command buffer on the device where the runtime's GPU form draws it, from what
        // the host and this draw's functions may write into its records; every other one by the
        // host replay. A render case registers no compute replay variant.
        let replayed = replay_commands(
            context,
            device,
            case.command_buffers(),
            Replay {
                live: &live,
                records: &finals.argument_buffer_buffers,
                addresses: &case_address,
                compile,
            },
            OnDevice {
                registered: &[],
                storage: &|key| embedded.get(&key).and_then(|allocation| allocation.address),
                writes: pipeline.interface.command_writes,
            },
        )?;
        for (&(stage, index), allocation) in &buffers {
            if let Some(bytes) = finals.buffers.get_mut(&(stage, index)) {
                let len = bytes.len();
                // SAFETY: as for the status buffer; every replayed command completed too.
                *bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
            }
        }
        for (key, allocation) in &embedded {
            if let Some(bytes) = finals.argument_buffer_buffers.get_mut(key) {
                let len = bytes.len();
                // SAFETY: as for the status buffer.
                *bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
            }
        }
        replayed
    } else {
        (Vec::new(), Vec::new())
    };
    match (observed, case.output) {
        (Some(key @ TextureKey::Slot(observed)), _) => {
            if let Some(texture) = textures.get(&observed) {
                // SAFETY: as for the status buffer.
                let texture = unsafe { resources.final_texture(key, texture)? };
                finals
                    .textures
                    .insert((observed.stage, observed.index), texture);
            }
        }
        (
            Some(key @ TextureKey::Used(_)),
            RenderOutput::ArgumentBufferTexture {
                stage,
                owner,
                field_offset,
                ..
            },
        ) => {
            let texture = case.function(stage).and_then(|f| {
                f.resources
                    .argument_buffer_textures
                    .get(&(owner, field_offset))
            });
            if let Some(texture) = texture {
                // SAFETY: as for the status buffer.
                let texture = unsafe { resources.final_texture(key, texture)? };
                finals
                    .argument_buffer_textures
                    .insert((stage, owner, field_offset), texture);
            }
        }
        _ => {}
    }
    if let RenderOutput::RenderTarget { index, .. } = case.output {
        let target = match storage_attachments.get(&index) {
            // The storage image's readback, or its authored bytes where no write of the plan
            // reaches it.
            Some(attachment) => {
                // SAFETY: as for the status buffer.
                let texture = unsafe {
                    resources.final_texture(TextureKey::TileAttachment(index), attachment)?
                };
                let authored = case
                    .color_targets
                    .get(&index)
                    .ok_or_else(|| format!("render target {index} is not authored"))?;
                ColorTarget {
                    bytes: texture.bytes,
                    ..authored.clone()
                }
            }
            // SAFETY: as for the status buffer.
            None => unsafe { targets.final_target(index, case)? },
        };
        finals.targets.insert(index, target);
    }
    // SAFETY: as for the status buffer.
    finals.depth_stencil = unsafe { targets.final_depth_stencil()? };
    if let (RenderOutput::Imageblock { .. }, Some(cells)) = (case.output, &case.imageblock) {
        finals.imageblock = Some(match &imageblock {
            Some(storage) => {
                // SAFETY: as for the status buffer.
                let image = unsafe { resources.final_texture(TextureKey::Imageblock, storage)? };
                cells
                    .from_image(storage.dimensions[2], &image.bytes)
                    .map_err(ExecutorError::Failed)?
            }
            // No write reaches the storage: the cells are the initializer's.
            None => cells.clone(),
        });
    }
    let output =
        graphics::select_render_output(case.output, &finals).map_err(ExecutorError::Failed)?;
    drop(objects);
    drop(readers);
    drop(resources);
    drop(shadow);
    drop(planes);
    drop(targets);
    Ok(Drawn {
        output,
        status: report,
        escaped,
        children,
        replayed,
    })
}

/// What [`execute`] read after the draw: the output, the status, the guard and padding bytes
/// that changed around each guarded out-of-bounds target, by target, the census of each module
/// a replayed command executed, and which replay ran each executed command buffer.
struct Drawn {
    output: Vec<u8>,
    status: StatusReport,
    escaped: BTreeMap<OutOfBoundsResource, GuardCount>,
    children: Vec<crate::coverage::ModuleCensus>,
    replayed: Vec<ReplayPath>,
}

/// Draw one replayed render command: `pipeline`, compiled for `case` (a render template with the
/// command's draw, which the checker requires to be a rasterization-disabled vertex case and the
/// candidate a vertex function that binds buffers only, `engine::replay_templates`), planned
/// for `state` (the plan must name no other variant and no neutral), over the run's allocations
/// `guest` (by slot), with `primitive`'s topology, which must be within the class the template was
/// compiled for (`graphics::replayed_draw_in_class`; a draw of another class fails the run). Its
/// pass is the template's own: the oracle's 1x1 `R8Unorm` scaffolding, which rasterization-disabled
/// drawing never writes. Each draw is submitted and waited for, with a clean status required,
/// before the next command.
pub(super) fn submit_draw_child(
    context: &VulkanContext,
    device: &Device,
    case: &RenderCase,
    pipeline: &RenderPipeline,
    state: &metal2vulkan::bind::BindState,
    primitive: crate::case::Primitive,
    guest: &BTreeMap<SlotRef, vk::Buffer>,
) -> Result<(), ExecutorError> {
    graphics::replayed_draw_in_class(&case.assembly, primitive).map_err(ExecutorError::Failed)?;
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let shape = state.shape();
    let plan = pipeline
        .plan(device, &shape)
        .map_err(|e| bind_refusal(&e))?;
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    if !plan.vertex_buffers.is_empty() {
        return Err(not_implemented("a replayed draw's vertex inputs"));
    }
    let commands = plan
        .instantiate_draw(state, &case.draw)
        .map_err(|e| bind_refusal(&e))?;
    if commands.indirect_needs_arguments {
        return Err(not_implemented("an indirect draw read on the CPU"));
    }
    let status = allocate(
        context,
        "replayed draw status",
        &ExecutionStatus::initial(),
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    draw_child(
        context,
        device,
        (case, primitive),
        (pipeline, &plan, &commands),
        (state, guest),
        (&status, &|_| None),
        None,
    )?;
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .map_err(|_| "the status buffer is shorter than ExecutionStatus")?;
    status_verdict(&status_bytes)?;
    Ok(())
}

/// What a draw replay records around its variant's draw (`compute::draw_replay`): before the
/// render pass, each owned kernel's pass in order (the preprocess, then a merged replay's two
/// merge passes), the first after a barrier from every command earlier in submission order and
/// each after one from the pass before it; in the pass, in place of the draw's own, each of
/// `draws` in order.
pub(super) struct ReplayedDraws<'a> {
    pub(super) passes: &'a [OwnedPass],
    /// The scratch the draws read their arguments from.
    pub(super) scratch: vk::Buffer,
    pub(super) draws: &'a [ReplayedDraw],
}

/// The address a draw replay gives a push hole of its variant's draw (a slot table's, its merged
/// draw's header), where it fills that hole.
pub(super) type Holes<'a> = &'a dyn Fn(&PushHole) -> Option<u64>;

/// One owned kernel's dispatch: its pipeline and layout, push bytes (`COMPUTE`, offset 0), and
/// group count.
pub(super) struct OwnedPass {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) push: Vec<u8>,
    pub(super) group_count: [u32; 3],
}

/// One draw of a draw replay: `vkCmdDrawIndirect` of one draw at `arguments` in the scratch, after
/// the push writes that make it its command's (none for the first, whose writes the variant's
/// draw carries).
pub(super) struct ReplayedDraw {
    pub(super) push: Vec<(vk::ShaderStageFlags, u32, Vec<u8>)>,
    pub(super) arguments: u64,
}

/// Draw a replayed execution's commands on the device: `pipeline`, the draw replay variant of
/// `case` (a render template, as [`submit_draw_child`] states it), planned as `plan` and
/// instantiated as `commands` (one indirect draw) for `state`, over the run's allocations `guest`
/// (by slot), with `primitive`'s topology, which must be within the template's class; its push
/// holes the replay fills (`holes`: a slot table's, the merged draw's header) with their
/// addresses, `status` its `ExecutionStatus`, and `replayed`'s passes and draws in place of the
/// draw's own. Submitted and waited for; the caller reads the status.
pub(super) fn submit_replayed_draws(
    context: &VulkanContext,
    device: &Device,
    (case, primitive): (&RenderCase, crate::case::Primitive),
    instantiated: (&RenderPipeline, &BindingPlan, &DrawCommands),
    bound: (&BindState, &BTreeMap<SlotRef, vk::Buffer>),
    filled: (&Allocation, Holes<'_>),
    replayed: &ReplayedDraws<'_>,
) -> Result<(), ExecutorError> {
    graphics::replayed_draw_in_class(&case.assembly, primitive).map_err(ExecutorError::Failed)?;
    draw_child(
        context,
        device,
        (case, primitive),
        instantiated,
        bound,
        filled,
        Some(replayed),
    )
}

/// Record and submit one replayed command's draw (or a draw replay's, `replayed`) of `pipeline`,
/// planned as `plan` and instantiated as `commands` for `state`, over the run's allocations
/// `guest`, in its template's pass (as [`submit_draw_child`] states it) with `primitive`'s
/// topology, its push holes filled by `holes` first, then from what the draw names (its root
/// tables, `status`); wait for it. The caller reads the status.
fn draw_child(
    context: &VulkanContext,
    device: &Device,
    (case, primitive): (&RenderCase, crate::case::Primitive),
    (pipeline, plan, commands): (&RenderPipeline, &BindingPlan, &DrawCommands),
    (state, guest): (&BindState, &BTreeMap<SlotRef, vk::Buffer>),
    (status, holes): (&Allocation, Holes<'_>),
    replayed: Option<&ReplayedDraws<'_>>,
) -> Result<(), ExecutorError> {
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let mut constant_blobs: BTreeMap<Stage, Arc<[u8]>> = BTreeMap::new();
    for module in &pipeline.stages {
        if let Some(blob) = &module.constant_blob {
            let stage = stage_of(module, false)?
                .ok_or("a replayed draw's synthesized stage carries a constant blob")?;
            constant_blobs.insert(stage, blob.clone());
        }
    }
    let targets = Targets::create(context, &pipeline.graphics, case)?;
    // A multisampled template's attachments leave the initializer's layout the same way.
    let readers = multisampled::SampleReaders::create(context, &targets)?;
    let mut resources = PlanResources::create(
        context,
        device,
        &plan.descriptors,
        &Guest {
            buffers: guest,
            textures: &BTreeMap::new(),
            // A replayed command declares no out-of-bounds target.
            texel_guards: &BTreeMap::new(),
            used_textures: &BTreeMap::new(),
            samplers: &BTreeMap::new(),
            argument_samplers: &state.argument_samplers,
            constant_blobs: &constant_blobs,
            descriptor_ranges: &commands.descriptor_ranges,
            descriptor_bytes: &commands.descriptor_bytes,
            structures: &BTreeMap::new(),
            tessellation_factors: None,
            attachments: &BTreeMap::new(),
            imageblock: None,
            tile_attachments: &BTreeMap::new(),
            shadow_stencil: None,
            status: Some(status.buffer.buffer),
        },
        None,
    )?;
    let mut roots = Vec::new();
    for (stages, table) in &commands.root_tables {
        let (table, sink) = upload_root_table(context, table, addressed)?;
        roots.push((vk::ShaderStageFlags::from(*stages), table, sink));
    }
    let push = push_writes(commands, &roots, status, holes)?;
    let layout = match LayoutObjects::create(&context.device, &pipeline.layout) {
        Ok(layout) => layout,
        Err(LayoutError::Shape(why) | LayoutError::Refused(why)) => {
            return Err(ExecutorError::Failed(why))
        }
    };
    let mut objects = Objects {
        device: &context.device,
        modules: Vec::new(),
        render_pass: vk::RenderPass::null(),
        framebuffer: vk::Framebuffer::null(),
        pipeline: vk::Pipeline::null(),
        command_pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
        layout,
    };
    create_render_pass(context, &mut objects, &pipeline.graphics, &targets)?;
    let (topology, primitive_restart) = graphics::input_assembly(primitive);
    let topology: vk::PrimitiveTopology = topology.into();
    let encoder = EncoderState::of(case, &pipeline.graphics)?;
    create_pipeline(
        context,
        &mut objects,
        pipeline,
        &plan.specialization,
        (topology, primitive_restart),
        encoder,
    )?;
    let sets = resources.write_sets(context, &objects.layout)?;
    record_and_submit(
        context,
        &mut objects,
        Recording {
            resources: &resources,
            targets: &targets,
            readers: readers.as_ref(),
            // A replayed draw's template is a vertex case with rasterization disabled: no
            // framebuffer fetch.
            planes: &[],
            // A replayed draw's pass is its template's, which has no rate map.
            draws: &draws(targets.extent, None),
            sets: &sets,
            push: &push,
            vertex_buffers: &[],
            index_buffer: None,
            dynamic: &pipeline.graphics.dynamic_states,
            topology,
            primitive_restart,
            encoder,
            rasterizer_discard: pipeline.graphics.rasterizer_discard,
            draw: commands,
            shadow: None,
            replayed,
        },
    )?;
    drop(objects);
    drop(readers);
    drop(targets);
    drop(resources);
    Ok(())
}

/// The color attachments `graphics` reads in the interlocked storage form in a pass of one sample,
/// by index, as the storage textures the plan's `WriteSource::TileAttachment` writes view: each a
/// single-sample 2D texture of the case's authored target, holding its authored bytes, which
/// Metal's `MTLLoadActionLoad` loads. The pass leaves them out of the subpass (an unused reference,
/// `graphics_pipeline::create_render_pass`); their extent is the pass's ([`Targets::create`]). A
/// multisampled pass has none: its storage-form attachments are targets of its samples
/// ([`Bound::Storage`], or [`Bound::Planes`]), which the plan views through the draw's attachment
/// views (a target's own, or its planes').
fn storage_attachments(
    graphics: &GraphicsState,
    case: &RenderCase,
) -> Result<BTreeMap<u32, LiteralTexture>, ExecutorError> {
    let mut out = BTreeMap::new();
    if let AttachmentTransfer::PerSample { .. } = case.attachment_transfer() {
        return Ok(out);
    }
    for a in &graphics.storage_attachments {
        let authored = case.color_targets.get(&a.color_index).ok_or_else(|| {
            format!(
                "storage attachment {} has no authored target",
                a.color_index
            )
        })?;
        let [width, height] = authored.dimensions;
        out.insert(
            a.color_index,
            LiteralTexture {
                binding: a.color_index,
                role: ResourceRole::InOut,
                texture_type: TextureType::D2,
                format: authored.format,
                dimensions: [width, height, 1],
                sample_count: 1,
                bytes: authored.bytes.clone(),
                sparse: None,
                mip_levels: 1,
            },
        );
    }
    Ok(out)
}

/// The sample planes of each storage-form attachment of `graphics` the plan views through them
/// (`StorageAttachmentView::SamplePlanes`), over its target in `targets` (a multisampled pass's,
/// [`Bound::Planes`]), holding the case's authored target.
fn sample_planes<'a>(
    context: &'a VulkanContext,
    graphics: &GraphicsState,
    case: &RenderCase,
    targets: &Targets<'_>,
) -> Result<Vec<planes::SamplePlanes<'a>>, ExecutorError> {
    let mut out = Vec::new();
    for used in &graphics.storage_attachments {
        if used.view == StorageAttachmentView::Attachment {
            continue;
        }
        let index = used.color_index;
        let target = targets
            .targets
            .get(&index)
            .ok_or_else(|| format!("storage attachment {index} has no target"))?;
        let authored = case
            .color_targets
            .get(&index)
            .ok_or_else(|| format!("storage attachment {index} has no authored target"))?;
        out.push(planes::SamplePlanes::create(
            context,
            used,
            (target, authored),
            targets.extent,
            targets.samples,
        )?);
    }
    Ok(out)
}

/// The pass's fragment imageblock storage for a draw of `graphics` over a framebuffer of `extent`,
/// as a texture the plan's `WriteSource::Imageblock` views: an `R32Uint` 2D array of the pass's
/// extent with a layer per word of the case's cells, which must be at least the pipeline's `words`
/// (`imageblock::Cells::storage_layers`), holding the cells the Metal oracle's initializer defines
/// (`imageblock::Cells::image_bytes`). `None` when the pipeline has no fragment imageblock. The
/// case's imageblock covers the pass: the checker requires its extent to be the attachments' (the
/// oracle's 1x1 pass without any).
fn imageblock_storage(
    graphics: &GraphicsState,
    case: &RenderCase,
    extent: vk::Extent2D,
) -> Result<Option<LiteralTexture>, ExecutorError> {
    let Some(state) = &graphics.fragment_imageblock else {
        return Ok(None);
    };
    let cells = case
        .imageblock
        .as_ref()
        .ok_or("the pipeline reaches a fragment imageblock the case does not author")?;
    let [width, height] = cells.dimensions;
    if (width, height) != (extent.width, extent.height) {
        return Err(not_implemented(format!(
            "a {width}x{height} imageblock in a {}x{} pass",
            extent.width, extent.height
        )));
    }
    let layers = cells.storage_layers(state.words).map_err(not_implemented)?;
    Ok(Some(LiteralTexture {
        binding: 0,
        role: ResourceRole::InOut,
        texture_type: TextureType::D2Array,
        format: TextureFormat::R32Uint,
        dimensions: [width, height, layers],
        sample_count: 1,
        bytes: cells.image_bytes(layers),
        sparse: None,
        mip_levels: 1,
    }))
}

/// Each stage's push-constant writes with the consumer's holes filled: the address `filled` gives
/// a hole (a draw replay's slot table, its merged draw's header), else its root table's and the
/// status buffer's device addresses.
fn push_writes(
    commands: &DrawCommands,
    roots: &[(vk::ShaderStageFlags, Allocation, Allocation)],
    status: &Allocation,
    filled: Holes<'_>,
) -> Result<Vec<(vk::ShaderStageFlags, u32, Vec<u8>)>, ExecutorError> {
    let mut writes: Vec<(vk::ShaderStageFlags, u32, Vec<u8>)> = commands
        .push_constants
        .iter()
        .map(|w| (w.stages.into(), w.offset, w.bytes.clone()))
        .collect();
    for hole in &commands.push_holes {
        let (stages, offset, address) = match (*hole, filled(hole)) {
            (
                PushHole::RootTableAddress { stages, offset }
                | PushHole::MergedDrawAddress { stages, offset },
                Some(address),
            ) => (stages.into(), offset, address),
            (PushHole::RootTableAddress { stages, offset }, None) => {
                let stages = vk::ShaderStageFlags::from(stages);
                let root = roots
                    .iter()
                    .find(|(of, ..)| *of == stages)
                    .and_then(|(_, table, _)| table.address)
                    .ok_or("a root-table hole without its stage's root table")?;
                (stages, offset, root)
            }
            (PushHole::ExecutionStatusAddress { stages, offset }, _) => (
                stages.into(),
                offset,
                status
                    .address
                    .ok_or("a status-address hole without device addresses")?,
            ),
            (other, _) => return Err(not_implemented(format!("the push hole {other:?}"))),
        };
        let write = writes
            .iter_mut()
            .find(|(of, start, bytes)| {
                *of == stages
                    && *start <= offset
                    && offset as usize + 8 <= *start as usize + bytes.len()
            })
            .ok_or_else(|| {
                format!(
                    "no push write of stages {:#x} covers the hole at {offset}",
                    stages.as_raw()
                )
            })?;
        let at = (offset - write.1) as usize;
        write.2[at..at + 8].copy_from_slice(&address.to_le_bytes());
    }
    Ok(writes)
}

/// One color attachment's image, view, and how its bytes cross.
struct Target {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    transfer: Transfer,
    /// What the subpass makes of it.
    bound: Bound,
}

/// What the subpass makes of one color target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    /// A color attachment.
    Color,
    /// A color attachment a framebuffer-fetch input reads: also an input attachment, kept in
    /// `GENERAL` through the subpass, as the render pass describes it
    /// (`graphics_pipeline::create_render_pass`).
    Fetched,
    /// A multisampled pass's attachment the fragment function reads and writes in the
    /// interlocked storage form: no attachment of the subpass (the framebuffer attaches only the
    /// state's color attachments), but a storage image of the pass's samples the plan's
    /// `WriteSource::TileAttachment` views, in `GENERAL`. A pass of one sample keeps its
    /// storage-form attachments as the plan's textures ([`storage_attachments`]).
    Storage,
    /// A multisampled pass's attachment the fragment function reads and writes through its sample
    /// planes ([`planes`]): no attachment of the subpass, and no image the draw binds (the plan's
    /// `WriteSource::TileAttachment` views the planes), but split into the planes by a compute
    /// fetch of its samples before the pass and merged back after it by a sample-rate pass that
    /// writes it as a color attachment, in `GENERAL` throughout.
    Planes,
}

impl Bound {
    /// The image usage the subpass adds to a color target's own: input-attachment reads of a
    /// fetched one, storage reads and writes of a storage-form one.
    fn usage(self) -> vk::ImageUsageFlags {
        match self {
            Bound::Color => vk::ImageUsageFlags::empty(),
            Bound::Fetched => vk::ImageUsageFlags::INPUT_ATTACHMENT,
            Bound::Storage => vk::ImageUsageFlags::STORAGE,
            // The split fetches it as every readback does; the merge attaches it.
            Bound::Planes => vk::ImageUsageFlags::empty(),
        }
    }
}

/// How an attachment's (or aspect's) bytes reach it before the draw and leave it after
/// (`executor_contract::multisample::AttachmentTransfer`).
enum Transfer {
    /// One sample, as every case once drew: copied from its staging buffer before the draw and into
    /// its readback buffer after it.
    Copy(Box<Copies>),
    /// More samples: filled by the per-sample initializer when the attachment was created, and
    /// read sample by sample after the draw ([`multisampled`]).
    PerSample(Box<multisampled::SampleReadback>),
}

/// The host buffers a one-sample attachment (or aspect) is copied from and into.
struct Copies {
    staging: HostBuffer,
    readback: HostBuffer,
}

impl Transfer {
    /// The copies of `staging` before the draw and into `readback` after it.
    fn copies(staging: HostBuffer, readback: HostBuffer) -> Transfer {
        Transfer::Copy(Box::new(Copies { staging, readback }))
    }

    /// The staging buffer a copy uploads from, for one sample.
    fn staging(&self) -> Option<&HostBuffer> {
        match self {
            Transfer::Copy(copies) => Some(&copies.staging),
            Transfer::PerSample(_) => None,
        }
    }

    /// The buffer a copy reads back into, for one sample.
    fn readback(&self) -> Option<&HostBuffer> {
        match self {
            Transfer::Copy(copies) => Some(&copies.readback),
            Transfer::PerSample(_) => None,
        }
    }

    /// The final bytes: `len` bytes of a copy's readback, or every sample's bytes of a
    /// `dimensions` attachment of `samples` samples.
    ///
    /// # Safety
    ///
    /// The submission that read the attachment back has completed.
    unsafe fn bytes(
        &self,
        len: usize,
        dimensions: [u32; 2],
        samples: u32,
    ) -> Result<Vec<u8>, String> {
        match self {
            // SAFETY: the caller's contract.
            Transfer::Copy(copies) => Ok(unsafe { copies.readback.map()? }.bytes[..len].to_vec()),
            // SAFETY: the caller's contract.
            Transfer::PerSample(readback) => unsafe { readback.bytes(dimensions, samples) },
        }
    }

    /// The readback view a per-sample transfer reads through, destroyed with its attachment.
    fn readback_view(&self) -> Option<vk::ImageView> {
        match self {
            Transfer::Copy(_) => None,
            Transfer::PerSample(readback) => Some(readback.view),
        }
    }
}

impl Target {
    /// The layout the render pass keeps the attachment in, the access its subpass makes of it,
    /// and the stages that access it ([`Bound::attachment_use`]).
    fn attachment_use(&self) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
        self.bound.attachment_use()
    }
}

impl Bound {
    /// The layout the render pass keeps a target of this use in, the access its subpass makes of
    /// it, and the stages that access it: color output, and the fragment shader's
    /// input-attachment reads of a fetched one, or its storage reads and writes of a storage-form
    /// one; for one read through its sample planes, the split's compute reads before the pass
    /// and the merge's color-attachment accesses after it.
    fn attachment_use(self) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
        let color =
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        match self {
            Bound::Color => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                color,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            Bound::Fetched => (
                vk::ImageLayout::GENERAL,
                color | vk::AccessFlags::INPUT_ATTACHMENT_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            Bound::Storage => (
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            Bound::Planes => (
                vk::ImageLayout::GENERAL,
                color | vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
        }
    }
}

/// One authored aspect of the depth/stencil attachment and how its bytes cross.
struct Aspect {
    aspect: vk::ImageAspectFlags,
    transfer: Transfer,
}

/// The depth/stencil attachment: its image in the pass's format, a view of every aspect of that
/// format, and each authored aspect (depth first).
struct DepthStencilTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Every aspect of the format, which the view and the layout transitions cover.
    format_aspects: vk::ImageAspectFlags,
    aspects: Vec<Aspect>,
    dimensions: [u32; 2],
    /// The samples of each pixel.
    samples: u32,
}

/// The attachments of the draw: the color attachments by Metal index, the depth/stencil one, the
/// framebuffer extent, and the pass's sample count.
struct Targets<'a> {
    device: &'a ash::Device,
    targets: BTreeMap<u32, Target>,
    depth_stencil: Option<DepthStencilTarget>,
    extent: vk::Extent2D,
    samples: u32,
}

impl Drop for Targets<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run and is destroyed once, after
        // the submission that used it completed (or before any submission).
        unsafe {
            for target in self.targets.values() {
                if let Some(view) = target.transfer.readback_view() {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image_view(target.view, None);
                self.device.destroy_image(target.image, None);
                self.device.free_memory(target.memory, None);
            }
            if let Some(target) = &self.depth_stencil {
                for view in target
                    .aspects
                    .iter()
                    .filter_map(|aspect| aspect.transfer.readback_view())
                {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image_view(target.view, None);
                self.device.destroy_image(target.image, None);
                self.device.free_memory(target.memory, None);
            }
        }
    }
}

impl<'a> Targets<'a> {
    /// One image per attachment the graphics state names, of its format, the case's authored
    /// extent, and the pass's sample count, holding the authored initial bytes: each color
    /// attachment (also an input attachment where a framebuffer-fetch input reads it), and the
    /// one depth/stencil attachment of the pass (`graphics_pipeline::depth_stencil_format`)
    /// holding each aspect the case authors. A pass of one sample uploads and reads them by
    /// copies; a multisampled one ([`RenderCase::attachment_transfer`]) fills and reads them sample
    /// by sample ([`multisampled`]). A storage-form attachment is no attachment of the subpass, but
    /// its extent is the pass's: in a pass of one sample it is the plan's storage texture
    /// ([`storage_attachments`]); in a multisampled one a target of the pass's samples, filled and
    /// read sample by sample as the attachments are, which a fragment function running once per
    /// sample reads and writes at its own sample: itself ([`Bound::Storage`]), or through its
    /// sample planes ([`Bound::Planes`], [`planes`]).
    fn create(
        context: &'a VulkanContext,
        graphics: &GraphicsState,
        case: &RenderCase,
    ) -> Result<Targets<'a>, ExecutorError> {
        let transfer = case.attachment_transfer();
        let samples = match transfer {
            AttachmentTransfer::Copy => 1,
            AttachmentTransfer::PerSample { samples } => samples,
        };
        let rasterized = vk::SampleCountFlags::from(graphics.rasterization_samples).as_raw();
        if rasterized != samples {
            return Err(ExecutorError::Failed(format!(
                "the pipeline rasterizes {rasterized} sample(s) in a pass of {samples}"
            )));
        }
        let mut targets = Targets {
            device: &context.device,
            targets: BTreeMap::new(),
            depth_stencil: None,
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            samples,
        };
        let mut extent = None;
        // A storage-form attachment is no attachment of the subpass, but its image is one of the
        // pass's targets: the framebuffer's extent is its extent too. In a multisampled pass it is
        // a target of the pass's samples.
        for a in &graphics.storage_attachments {
            let authored = case.color_targets.get(&a.color_index).ok_or_else(|| {
                format!(
                    "storage attachment {} has no authored target",
                    a.color_index
                )
            })?;
            if *extent.get_or_insert(authored.dimensions) != authored.dimensions {
                return Err(not_implemented("color targets of different extents"));
            }
            if let AttachmentTransfer::PerSample { samples } = transfer {
                let bound = match a.view {
                    StorageAttachmentView::Attachment => Bound::Storage,
                    StorageAttachmentView::SamplePlanes { .. } => Bound::Planes,
                };
                // The draw views a multisampled target in its own format alone.
                if bound == Bound::Storage && a.view_alias.is_some() {
                    return Err(not_implemented(
                        "a multisampled storage-form attachment viewed in another format",
                    ));
                }
                let target = multisampled::create_target(
                    context,
                    a.color_index,
                    a.format.into(),
                    authored,
                    samples,
                    bound,
                )?;
                targets.targets.insert(a.color_index, target);
            }
        }
        for (index, state) in graphics.color_attachments.iter().enumerate() {
            let Some(state) = state else { continue };
            let index = index as u32;
            let authored = case
                .color_targets
                .get(&index)
                .ok_or_else(|| format!("color attachment {index} has no authored target"))?;
            let [width, height] = authored.dimensions;
            if *extent.get_or_insert([width, height]) != [width, height] {
                return Err(not_implemented("color targets of different extents"));
            }
            let bound = if graphics
                .input_attachments
                .iter()
                .any(|input| input.color_index == index)
            {
                Bound::Fetched
            } else {
                Bound::Color
            };
            let target = match transfer {
                AttachmentTransfer::Copy => {
                    create_target(context, index, state.format.into(), authored, bound)?
                }
                AttachmentTransfer::PerSample { samples } => multisampled::create_target(
                    context,
                    index,
                    state.format.into(),
                    authored,
                    samples,
                    bound,
                )?,
            };
            targets.targets.insert(index, target);
        }
        if let Some(format) = graphics_pipeline::attachable_depth_stencil_format(
            &context.instance,
            context.physical,
            graphics,
        )
        .map_err(graphics_error)?
        {
            let authored = case
                .depth_stencil
                .as_ref()
                .ok_or("the pipeline has a depth/stencil attachment the case does not author")?;
            if *extent.get_or_insert(authored.dimensions) != authored.dimensions {
                return Err(not_implemented(
                    "color and depth/stencil targets of different extents",
                ));
            }
            if let Some(depths) = &authored.depth {
                super::plan_resources::require_depth_range(context, format, depths)?;
            }
            targets.depth_stencil = Some(match transfer {
                AttachmentTransfer::Copy => create_depth_stencil(context, format, authored)?,
                AttachmentTransfer::PerSample { samples } => {
                    multisampled::create_depth_stencil(context, format, authored, samples)?
                }
            });
        }
        if let Some([width, height]) = extent {
            targets.extent = vk::Extent2D { width, height };
        }
        // A multisampled subpass without attachments (storage-form ones are none).
        let attached = graphics.color_attachments.iter().any(Option::is_some)
            || targets.depth_stencil.is_some();
        if let (AttachmentTransfer::PerSample { samples }, false) = (transfer, attached) {
            multisampled::attachmentless_support(context, samples)?;
        }
        Ok(targets)
    }

    /// The depth/stencil attachment with its final aspects.
    ///
    /// # Safety
    ///
    /// The submission that rendered to it has completed.
    unsafe fn final_depth_stencil(&self) -> Result<Option<LiteralDepthStencil>, String> {
        let Some(target) = &self.depth_stencil else {
            return Ok(None);
        };
        let mut finals = LiteralDepthStencil {
            dimensions: target.dimensions,
            samples: target.samples,
            depth: None,
            stencil: None,
        };
        let [width, height] = target.dimensions;
        for aspect in &target.aspects {
            let (slot, texel) = if aspect.aspect == vk::ImageAspectFlags::DEPTH {
                (&mut finals.depth, 4)
            } else {
                (&mut finals.stencil, 1)
            };
            let len = width as usize * height as usize * texel;
            // SAFETY: the caller's contract.
            *slot = Some(unsafe {
                aspect
                    .transfer
                    .bytes(len, target.dimensions, target.samples)?
            });
        }
        Ok(Some(finals))
    }

    /// Target `index` with its final contents.
    ///
    /// # Safety
    ///
    /// The submission that rendered to it has completed.
    unsafe fn final_target(&self, index: u32, case: &RenderCase) -> Result<ColorTarget, String> {
        let authored = case
            .color_targets
            .get(&index)
            .ok_or_else(|| format!("render target {index} is not authored"))?;
        let target = self
            .targets
            .get(&index)
            .ok_or_else(|| format!("render target {index} is not attached"))?;
        // SAFETY: the caller's contract.
        let bytes = unsafe {
            target
                .transfer
                .bytes(authored.bytes.len(), authored.dimensions, authored.samples)?
        };
        Ok(ColorTarget {
            bytes,
            ..authored.clone()
        })
    }
}

/// Color attachment `index` of a pass of one sample, of `format` holding `authored`, used as
/// `bound`: a color attachment, or one a framebuffer-fetch input reads (never a storage-form
/// one, which a pass of one sample keeps as the plan's texture, [`storage_attachments`]).
fn create_target(
    context: &VulkanContext,
    index: u32,
    format: vk::Format,
    authored: &ColorTarget,
    bound: Bound,
) -> Result<Target, ExecutorError> {
    let label = format!("color target {index}");
    let [width, height] = authored.dimensions;
    let texel = authored.format.bytes_per_pixel();
    if authored.bytes.len() != width as usize * height as usize * texel {
        return Err(format!(
            "{label} authors {} bytes for {width}x{height} texels of {texel} bytes",
            authored.bytes.len()
        )
        .into());
    }
    let (image, memory) = create_image(
        context,
        &label,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | bound.usage(),
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let destroy = || {
        // SAFETY: the image and memory were created above and are unused.
        unsafe {
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        }
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: `view_info` names the live image.
    let view = match unsafe { context.device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            destroy();
            return Err(format!("create {label} view: {error}").into());
        }
    };
    let host = |what: &str, usage| {
        create_host_buffer(context, &format!("{label} {what}"), &authored.bytes, usage)
    };
    let buffers = host("staging", vk::BufferUsageFlags::TRANSFER_SRC).and_then(|staging| {
        host("readback", vk::BufferUsageFlags::TRANSFER_DST).map(|readback| (staging, readback))
    });
    match buffers {
        Ok((staging, readback)) => Ok(Target {
            image,
            memory,
            view,
            transfer: Transfer::copies(staging, readback),
            bound,
        }),
        Err(error) => {
            // SAFETY: the view was created above and is unused.
            unsafe { context.device.destroy_image_view(view, None) };
            destroy();
            Err(error.into())
        }
    }
}

/// The depth/stencil attachment of `format`, holding each aspect `authored` gives (tightly packed:
/// little-endian `f32` depth, one byte of stencil per texel). Every authored aspect must be one
/// of the format's; an aspect of the format the case does not author (the stencil of a combined
/// format the product chose for a lone depth attachment) is neither loaded nor read, and no test
/// of the draw reads it.
fn create_depth_stencil(
    context: &VulkanContext,
    format: vk::Format,
    authored: &LiteralDepthStencil,
) -> Result<DepthStencilTarget, ExecutorError> {
    let label = "the depth/stencil attachment";
    let [width, height] = authored.dimensions;
    let texels = width as usize * height as usize;
    let format_aspects = (if graphics_pipeline::has_depth(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::empty()
    }) | if graphics_pipeline::has_stencil(format) {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::empty()
    };
    let mut wanted = Vec::new();
    for (aspect, bytes, texel, name) in [
        (vk::ImageAspectFlags::DEPTH, &authored.depth, 4, "depth"),
        (
            vk::ImageAspectFlags::STENCIL,
            &authored.stencil,
            1,
            "stencil",
        ),
    ] {
        let Some(bytes) = bytes else { continue };
        if !format_aspects.contains(aspect) {
            return Err(format!(
                "the case authors {name}, which VkFormat {} of the pass does not hold",
                format.as_raw()
            )
            .into());
        }
        if bytes.len() != texels * texel {
            return Err(format!(
                "the {name} aspect authors {} bytes for {width}x{height} texels of {texel} bytes",
                bytes.len()
            )
            .into());
        }
        wanted.push((aspect, bytes, name));
    }
    let (image, memory) = create_image(
        context,
        label,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let destroy = || {
        // SAFETY: the image and memory were created above and are unused.
        unsafe {
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        }
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: format_aspects,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: `view_info` names the live image.
    let view = match unsafe { context.device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            destroy();
            return Err(format!("create {label} view: {error}").into());
        }
    };
    let mut aspects = Vec::new();
    for (aspect, bytes, name) in wanted {
        let host = |what: &str, usage| {
            create_host_buffer(context, &format!("{label} {name} {what}"), bytes, usage)
        };
        let buffers = host("staging", vk::BufferUsageFlags::TRANSFER_SRC).and_then(|staging| {
            host("readback", vk::BufferUsageFlags::TRANSFER_DST).map(|readback| (staging, readback))
        });
        match buffers {
            Ok((staging, readback)) => aspects.push(Aspect {
                aspect,
                transfer: Transfer::copies(staging, readback),
            }),
            Err(error) => {
                // SAFETY: the view was created above and is unused.
                unsafe { context.device.destroy_image_view(view, None) };
                destroy();
                return Err(error.into());
            }
        }
    }
    Ok(DepthStencilTarget {
        image,
        memory,
        view,
        format_aspects,
        aspects,
        dimensions: authored.dimensions,
        samples: 1,
    })
}

/// Every object of one draw beyond its buffers, resources, targets, and layout, destroyed on
/// every exit path (before the layout, a field declared after them).
struct Objects<'a> {
    device: &'a ash::Device,
    modules: Vec<vk::ShaderModule>,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    layout: LayoutObjects<'a>,
}

impl Drop for Objects<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run, is destroyed once, and the
        // submission that used it has completed (or never happened).
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for module in &self.modules {
                self.device.destroy_shader_module(*module, None);
            }
        }
    }
}

/// The render pass of the pipeline's graphics state (`graphics_pipeline::create_render_pass`, the
/// one L4 acceptance creates) and its framebuffer over the targets, in the pass's attachment
/// order (the present color attachments by index).
fn create_render_pass(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    graphics: &GraphicsState,
    targets: &Targets<'_>,
) -> Result<(), ExecutorError> {
    objects.render_pass =
        graphics_pipeline::create_render_pass(&context.device, graphics).map_err(graphics_error)?;
    let views: Vec<vk::ImageView> = graphics
        .color_attachments
        .iter()
        .enumerate()
        .filter(|(_, state)| state.is_some())
        .map(|(index, _)| {
            targets
                .targets
                .get(&(index as u32))
                .map(|target| target.view)
                .ok_or_else(|| format!("color attachment {index} has no target"))
        })
        .chain(targets.depth_stencil.iter().map(|target| Ok(target.view)))
        .collect::<Result<_, String>>()?;
    let framebuffer = vk::FramebufferCreateInfo::default()
        .render_pass(objects.render_pass)
        .attachments(&views)
        .width(targets.extent.width)
        .height(targets.extent.height)
        .layers(1);
    // SAFETY: the render pass and every view are live.
    objects.framebuffer = unsafe { context.device.create_framebuffer(&framebuffer, None) }
        .map_err(|error| format!("create framebuffer: {error}"))?;
    Ok(())
}

/// A graphics-builder failure as the executor reports it: a state the builder does not create is
/// typed `NotImplemented`, what the device does not provide `NotEnabled`, and a driver refusal
/// fails the run.
fn graphics_error(error: GraphicsError) -> ExecutorError {
    match error {
        GraphicsError::Unmodelled(why) => not_implemented(why),
        GraphicsError::NotSupported(why) => NotSupported::NotEnabled(vec![why]).into(),
        GraphicsError::Refused(why) => ExecutorError::Failed(why),
    }
}

/// The graphics pipeline of `pipeline` for this draw (`graphics_pipeline::create_graphics_pipeline`,
/// the one L4 acceptance creates): its stages under the plan's specialization, the draw's
/// `topology` within the compiled class with its primitive restart, and `encoder`, the state the
/// Metal oracle sets for the case ([`EncoderState`]). Its creation must meet the 5-s ceiling. Every
/// state it makes dynamic is one the draw records.
fn create_pipeline(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    pipeline: &RenderPipeline,
    specialization: &metal2vulkan::cache::SpecializationKey,
    (topology, primitive_restart): (vk::PrimitiveTopology, bool),
    encoder: EncoderState,
) -> Result<(), ExecutorError> {
    let graphics = &pipeline.graphics;
    let compiled: vk::PrimitiveTopology = graphics.topology.into();
    let class = |t: vk::PrimitiveTopology| match t {
        vk::PrimitiveTopology::POINT_LIST => 0,
        vk::PrimitiveTopology::LINE_LIST | vk::PrimitiveTopology::LINE_STRIP => 1,
        _ => 2,
    };
    if class(topology) != class(compiled) {
        return Err(format!(
            "the draw's topology {} is outside the compiled class of {}",
            topology.as_raw(),
            compiled.as_raw()
        )
        .into());
    }
    for &state in &graphics.dynamic_states {
        let state = vk::DynamicState::from(state);
        let Some(record) = DynamicRecord::of(state) else {
            return Err(not_implemented(format!("dynamic state {}", state.as_raw())));
        };
        if record.needs_extended() && context.extended_dynamic_state.is_none() {
            return Err(format!(
                "dynamic state {} on a context below Vulkan 1.3 without \
                 VK_EXT_extended_dynamic_state",
                state.as_raw()
            )
            .into());
        }
        // Vulkan 1.3's core command, or `VK_EXT_extended_dynamic_state2`'s where the context
        // enabled it.
        if state == vk::DynamicState::PRIMITIVE_RESTART_ENABLE
            && context
                .extended_dynamic_state
                .is_none_or(|e| e.primitive_restart_enable.is_none())
        {
            return Err(
                "a dynamic primitive restart on a context below Vulkan 1.3 without \
                 VK_EXT_extended_dynamic_state2"
                    .into(),
            );
        }
    }
    for stage in &pipeline.stages {
        let info = vk::ShaderModuleCreateInfo::default().code(&stage.spirv);
        // SAFETY: `info` points at live SPIR-V words for the call.
        let module = unsafe { context.device.create_shader_module(&info, None) }
            .map_err(|error| format!("create shader module: {error}"))?;
        objects.modules.push(module);
    }
    let map: Vec<vk::SpecializationMapEntry> = specialization
        .entries
        .iter()
        .map(|&entry| entry.into())
        .collect();
    let specializations = vec![(map, specialization.data.clone()); pipeline.stages.len()];
    let (created, elapsed) = graphics_pipeline::create_graphics_pipeline(
        &context.device,
        &context.executing_driver_facts(),
        pipeline,
        &graphics_pipeline::Consumer {
            modules: &objects.modules,
            specializations: &specializations,
            layout: objects.layout.pipeline_layout,
            render_pass: objects.render_pass,
            topology,
            primitive_restart,
            depth_stencil: encoder.depth_stencil,
            cull_mode: encoder.cull_mode,
            front_face: encoder.front_face,
        },
    )
    .map_err(graphics_error)?;
    objects.pipeline = created;
    crate::budgets::creation_within_ceiling("the graphics pipeline", elapsed)?;
    Ok(())
}

/// How the draw records a dynamic state of its pipeline: the one table the pipeline's creation
/// (which refuses a state it has no row for, or one its context lacks the commands of) and the
/// recording ([`record_and_submit`]) share, so every state a created pipeline makes dynamic is
/// one each of its draws sets (`VUID-vkCmdDraw-None-04879` and its siblings). Each state is at
/// the Metal render encoder's default (the whole target for the viewport and scissor), or at the
/// draw's own state (its topology, primitive restart and depth-stencil state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DynamicRecord {
    /// The viewport and scissor, set for each recording of the draw ([`draws`]).
    PerDraw,
    /// The per-bind vertex strides (`VERTEX_INPUT_BINDING_STRIDE`, a vertex layout of
    /// `MTLBufferLayoutStrideDynamic` where the profile makes strides dynamic), recorded with the
    /// vertex buffers at the strides the case binds them with, through
    /// [`VulkanContext::extended_dynamic_state`].
    Strides,
    /// A Vulkan 1.0 command of the bound pipeline.
    Core,
    /// A state `VK_EXT_extended_dynamic_state` defines (core in Vulkan 1.3), or the primitive
    /// restart `VK_EXT_extended_dynamic_state2` does, through
    /// [`VulkanContext::extended_dynamic_state`] ([`record_extended_state`]).
    Extended,
}

impl DynamicRecord {
    /// How the draw records `state`; `None` for a state this executor does not record.
    fn of(state: vk::DynamicState) -> Option<DynamicRecord> {
        use vk::DynamicState as D;
        Some(match state {
            D::VIEWPORT | D::SCISSOR => DynamicRecord::PerDraw,
            D::VERTEX_INPUT_BINDING_STRIDE => DynamicRecord::Strides,
            D::DEPTH_BIAS
            | D::BLEND_CONSTANTS
            | D::STENCIL_COMPARE_MASK
            | D::STENCIL_WRITE_MASK
            | D::STENCIL_REFERENCE => DynamicRecord::Core,
            D::CULL_MODE
            | D::FRONT_FACE
            | D::PRIMITIVE_TOPOLOGY
            | D::PRIMITIVE_RESTART_ENABLE
            | D::DEPTH_TEST_ENABLE
            | D::DEPTH_WRITE_ENABLE
            | D::DEPTH_COMPARE_OP
            | D::STENCIL_TEST_ENABLE
            | D::STENCIL_OP => DynamicRecord::Extended,
            _ => return None,
        })
    }

    /// Whether the draw records it through [`VulkanContext::extended_dynamic_state`].
    fn needs_extended(self) -> bool {
        matches!(self, DynamicRecord::Strides | DynamicRecord::Extended)
    }
}

/// What one draw records.
/// The encoder state the Metal oracle sets for a case's draw, which the pipeline takes
/// statically or the draw records where the pipeline makes it dynamic: the depth-stencil state
/// with each face's stencil reference and the front face (`RenderCase::depth_stencil_state`), and
/// the cull mode (`RenderCase::cull_mode`).
#[derive(Clone, Copy)]
struct EncoderState {
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'static>,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
}

impl EncoderState {
    /// The encoder state of `case`'s draw through a pipeline of `graphics`: the stencil test is
    /// off where the fragment function tests the stencil itself (`graphics.shadow_stencil`), which
    /// takes the descriptor's stencil state from the bind state instead.
    fn of(case: &RenderCase, graphics: &GraphicsState) -> Result<EncoderState, ExecutorError> {
        let depth_stencil = graphics_pipeline::metal_depth_stencil(&case.depth_stencil_state);
        Ok(EncoderState {
            depth_stencil: depth_stencil.stencil_test_enable(
                depth_stencil.stencil_test_enable == vk::TRUE && graphics.shadow_stencil.is_none(),
            ),
            cull_mode: graphics::cull_mode_flags(case.cull_mode)
                .map_err(not_implemented)?
                .into(),
            front_face: graphics_pipeline::front_face(&case.depth_stencil_state),
        })
    }
}

struct Recording<'r> {
    resources: &'r PlanResources<'r>,
    targets: &'r Targets<'r>,
    /// The per-sample fill and readback of a multisampled pass's attachments; `None` for one
    /// sample, whose attachments `targets` copies.
    readers: Option<&'r multisampled::SampleReaders<'r>>,
    /// The sample planes split before the pass and merged after it.
    planes: &'r [planes::SamplePlanes<'r>],
    /// The draw's viewport and scissor, once per recording of it ([`draws`]).
    draws: &'r [(vk::Viewport, vk::Rect2D)],
    sets: &'r [vk::DescriptorSet],
    push: &'r [(vk::ShaderStageFlags, u32, Vec<u8>)],
    /// Each vertex buffer's binding.
    vertex_buffers: &'r [VertexBinding],
    /// An indexed draw's index buffer (`graphics::INDEX_BUFFER_TOKEN`), bound at the draw's
    /// index buffer offset.
    index_buffer: Option<vk::Buffer>,
    dynamic: &'r [metal2vulkan::vk::DynamicState],
    /// The draw's topology and Metal's primitive restart for it, for a dynamic
    /// `PRIMITIVE_TOPOLOGY` and `PRIMITIVE_RESTART_ENABLE`.
    topology: vk::PrimitiveTopology,
    primitive_restart: bool,
    /// The encoder state the pipeline would take statically, for its dynamic states.
    encoder: EncoderState,
    /// The pipeline has rasterization disabled, so it has no fragment shader or fragment output
    /// state for dynamic state to set.
    rasterizer_discard: bool,
    draw: &'r DrawCommands,
    /// The shader-side stencil's shadow, copied in before the render pass and out after it.
    shadow: Option<&'r shadow::ShadowStencil<'r>>,
    /// A draw replay's passes before the render pass and draws in place of `draw`'s own, which is
    /// then one indirect draw.
    replayed: Option<&'r ReplayedDraws<'r>>,
}

/// A vertex buffer's `(binding, buffer, offset, stride)`, the stride where the pipeline's
/// strides are dynamic state (`VERTEX_INPUT_BINDING_STRIDE`), every binding's then.
type VertexBinding = (u32, vk::Buffer, u64, Option<u64>);

/// The draw's `(viewport, scissor)`, once per recording of it: the whole target `extent` with a
/// negative height (the module docs), or, for a rate-mapped pass, each region of its plan, whose
/// Metal-convention viewport `[x, y, width, height]` (y down) is flipped the same way, `y + height`
/// and `-height`, with the region's scissor.
fn draws(
    extent: vk::Extent2D,
    regions: Option<&RateMapRegions>,
) -> Vec<(vk::Viewport, vk::Rect2D)> {
    let flipped = |[x, y, width, height]: [f32; 4]| vk::Viewport {
        x,
        y: y + height,
        width,
        height: -height,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let Some(plan) = regions else {
        return vec![(
            flipped([0.0, 0.0, extent.width as f32, extent.height as f32]),
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            },
        )];
    };
    plan.regions()
        .iter()
        .map(|region| {
            let [x, y] = region.scissor_offset();
            let [width, height] = region.scissor_extent();
            (
                flipped(region.viewport()),
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: x as i32,
                        y: y as i32,
                    },
                    extent: vk::Extent2D { width, height },
                },
            )
        })
        .collect()
}

/// Whether `state` belongs to the fragment shader or fragment output state, which a pipeline
/// with rasterization disabled does not include: setting it after binding such a pipeline is
/// `VUID-vkCmdDraw-None-08608`. The depth and stencil test states are fragment shader state.
fn fragment_state(state: vk::DynamicState) -> bool {
    matches!(
        state,
        vk::DynamicState::BLEND_CONSTANTS
            | vk::DynamicState::STENCIL_COMPARE_MASK
            | vk::DynamicState::STENCIL_WRITE_MASK
            | vk::DynamicState::STENCIL_REFERENCE
            | vk::DynamicState::DEPTH_TEST_ENABLE
            | vk::DynamicState::DEPTH_WRITE_ENABLE
            | vk::DynamicState::DEPTH_COMPARE_OP
            | vk::DynamicState::STENCIL_TEST_ENABLE
            | vk::DynamicState::STENCIL_OP
    )
}

/// Record extended dynamic `state` at the value the pipeline would take statically
/// (`graphics_pipeline::create_graphics_pipeline`): the case's culling and front face, the draw's
/// topology and primitive restart, and each face's depth and stencil test state of `recording.encoder`.
///
/// # Safety
/// `command` is recording on the device `extended`'s commands were loaded for, after the
/// pipeline was bound.
unsafe fn record_extended_state(
    extended: &ExtendedDynamicState,
    command: vk::CommandBuffer,
    state: vk::DynamicState,
    recording: &Recording<'_>,
) {
    let ds = &recording.encoder.depth_stencil;
    // SAFETY: the caller's contract; each command sets one state of the bound pipeline.
    unsafe {
        match state {
            vk::DynamicState::CULL_MODE => {
                (extended.cull_mode)(command, recording.encoder.cull_mode)
            }
            vk::DynamicState::FRONT_FACE => {
                (extended.front_face)(command, recording.encoder.front_face)
            }
            vk::DynamicState::PRIMITIVE_TOPOLOGY => {
                (extended.primitive_topology)(command, recording.topology)
            }
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                if let Some(restart) = extended.primitive_restart_enable {
                    restart(command, vk::Bool32::from(recording.primitive_restart));
                }
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => {
                (extended.depth_test_enable)(command, ds.depth_test_enable)
            }
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                (extended.depth_write_enable)(command, ds.depth_write_enable)
            }
            vk::DynamicState::DEPTH_COMPARE_OP => {
                (extended.depth_compare_op)(command, ds.depth_compare_op)
            }
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                (extended.stencil_test_enable)(command, ds.stencil_test_enable)
            }
            vk::DynamicState::STENCIL_OP => {
                for (face, ops) in [
                    (vk::StencilFaceFlags::FRONT, ds.front),
                    (vk::StencilFaceFlags::BACK, ds.back),
                ] {
                    (extended.stencil_op)(
                        command,
                        face,
                        ops.fail_op,
                        ops.pass_op,
                        ops.depth_fail_op,
                        ops.compare_op,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Record the uploads, the draw in its render pass, and the readbacks, submit, and wait under
/// the hang protocol.
fn record_and_submit(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    recording: Recording<'_>,
) -> Result<(), ExecutorError> {
    let device = &context.device;
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family);
    // SAFETY: the queue family is the context's.
    objects.command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .map_err(|error| format!("create command pool: {error}"))?;
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(objects.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // The pool is live; the buffer is freed with it.
    let command = context
        .allocate_command_buffers(&allocate_info)
        .map_err(|error| format!("allocate command buffer: {error}"))?[0];
    let targets = recording.targets;
    let color = vk::ImageAspectFlags::COLOR;
    let extent = targets.extent;
    let depth_stencil = targets.depth_stencil.as_ref();
    // Each attachment image a copy uploads and reads back (one sample; a multisampled one is
    // filled and read by `recording.readers`), with the aspects its layout transitions cover, and
    // the access and layout of its subpass (a framebuffer-fetch input's color attachment in
    // `GENERAL`).
    let images: Vec<(
        vk::Image,
        vk::ImageAspectFlags,
        vk::AccessFlags,
        vk::ImageLayout,
    )> = targets
        .targets
        .values()
        .filter(|t| t.transfer.staging().is_some())
        .map(|t| {
            let (layout, access, _) = t.attachment_use();
            (t.image, color, access, layout)
        })
        .chain(depth_stencil.filter(|t| t.samples == 1).map(|t| {
            (
                t.image,
                t.format_aspects,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
        }))
        .collect();
    let to_transfer: Vec<vk::ImageMemoryBarrier<'_>> = images
        .iter()
        .map(|&(image, aspects, _, _)| {
            barrier(
                image,
                aspects,
                1,
                (vk::AccessFlags::empty(), vk::ImageLayout::UNDEFINED),
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
            )
        })
        .collect();
    let to_attachment: Vec<vk::ImageMemoryBarrier<'_>> = images
        .iter()
        .map(|&(image, aspects, access, layout)| {
            barrier(
                image,
                aspects,
                1,
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                (access, layout),
            )
        })
        .collect();
    let to_source: Vec<vk::ImageMemoryBarrier<'_>> = images
        .iter()
        .map(|&(image, aspects, access, layout)| {
            barrier(
                image,
                aspects,
                1,
                (access, layout),
                (
                    vk::AccessFlags::TRANSFER_READ,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ),
            )
        })
        .collect();
    // The attachment stages: color output, the fragment tests that access depth/stencil, and the
    // fragment shader where it reads an input attachment.
    let attachment_stages = targets.targets.values().fold(
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        |stages, t| stages | t.attachment_use().2,
    );
    let region = |aspect| {
        vk::BufferImageCopy::default()
            .image_subresource(layers(aspect, 1))
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
    };
    let copy = region(color);
    let transfer_to_host = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };
    let draw = match (
        recording.draw.draw,
        recording.index_buffer,
        recording.replayed,
    ) {
        (metal2vulkan::vk::DrawCommand::Draw { .. }, None, None)
        | (metal2vulkan::vk::DrawCommand::DrawIndexed { .. }, Some(_), None)
        | (metal2vulkan::vk::DrawCommand::DrawIndirect { .. }, None, Some(_)) => {
            recording.draw.draw
        }
        (_, _, Some(_)) => return Err("a draw replay's draw is not one indirect draw".into()),
        (metal2vulkan::vk::DrawCommand::DrawIndexed { .. }, None, _) => {
            return Err("an indexed draw without an index buffer".into())
        }
        (metal2vulkan::vk::DrawCommand::Draw { .. }, Some(_), _) => {
            return Err("an index buffer for a draw that is not indexed".into())
        }
        _ => return Err(not_implemented("an indirect draw")),
    };
    // A draw replay's barriers. Its preprocess reads the records and the buffers they name, which
    // the producer's command buffer wrote: a pipeline barrier recorded outside a render pass
    // instance has every command earlier in submission order in its first synchronization scope
    // (the Vulkan specification's `vkCmdPipelineBarrier`), so one from every command makes those
    // writes visible. Each pass's writes (the slot tables, the arguments, a merged replay's
    // sections) before the next pass reads and writes them and the draws read them, as indirect
    // arguments and in their functions; and the draws' writes, which the host reads after the
    // wait, made visible to it.
    let earlier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    let passed = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::INDIRECT_COMMAND_READ,
        );
    let drawn = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    let after_pass = vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::DRAW_INDIRECT
        | vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER;
    // SAFETY: every handle recorded is live until the fence signals; the buffer is recorded once.
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| format!("begin command buffer: {error}"))?;
        recording.resources.record_uploads(command);
        if !to_transfer.is_empty() {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            for t in targets.targets.values() {
                if let Some(staging) = t.transfer.staging() {
                    device.cmd_copy_buffer_to_image(
                        command,
                        staging.buffer,
                        t.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
            }
            if let Some(t) = depth_stencil {
                for aspect in &t.aspects {
                    if let Some(staging) = aspect.transfer.staging() {
                        device.cmd_copy_buffer_to_image(
                            command,
                            staging.buffer,
                            t.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[region(aspect.aspect)],
                        );
                    }
                }
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                attachment_stages,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_attachment,
            );
        }
        if let Some(readers) = recording.readers {
            readers.record_initialized(command, attachment_stages);
        }
        for planes in recording.planes {
            planes.record_split(command);
        }
        if let (Some(shadow), Some(target)) = (recording.shadow, depth_stencil) {
            shadow.record_copy_in(command, target);
        }
        if let Some(replayed) = recording.replayed {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[earlier],
                &[],
                &[],
            );
            for pass in replayed.passes {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
                device.cmd_push_constants(
                    command,
                    pass.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &pass.push,
                );
                let [x, y, z] = pass.group_count;
                device.cmd_dispatch(command, x, y, z);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    after_pass,
                    vk::DependencyFlags::empty(),
                    &[passed],
                    &[],
                    &[],
                );
            }
        }
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(objects.render_pass)
            .framebuffer(objects.framebuffer)
            .render_area(render_area);
        device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, objects.pipeline);
        if !recording.sets.is_empty() {
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                objects.layout.pipeline_layout,
                0,
                recording.sets,
                &[],
            );
        }
        for (stages, offset, bytes) in recording.push {
            device.cmd_push_constants(
                command,
                objects.layout.pipeline_layout,
                *stages,
                *offset,
                bytes,
            );
        }
        for &(binding, buffer, offset, stride) in recording.vertex_buffers {
            match (stride, &context.extended_dynamic_state) {
                // `create_pipeline` refused dynamic strides without the extension's commands.
                (Some(stride), Some(extended)) => (extended.bind_vertex_buffers2)(
                    command,
                    binding,
                    1,
                    &buffer,
                    &offset,
                    std::ptr::null(),
                    &stride,
                ),
                _ => device.cmd_bind_vertex_buffers(command, binding, &[buffer], &[offset]),
            }
        }
        for state in recording
            .dynamic
            .iter()
            .map(|&state| vk::DynamicState::from(state))
        {
            if recording.rasterizer_discard && fragment_state(state) {
                continue;
            }
            match (DynamicRecord::of(state), state) {
                // Set per draw, below; the strides with the vertex buffers, above;
                // `create_pipeline` refused every state without a record.
                (None | Some(DynamicRecord::PerDraw | DynamicRecord::Strides), _) => {}
                // `create_pipeline` refused these without the extension's commands.
                (Some(DynamicRecord::Extended), state) => {
                    if let Some(extended) = &context.extended_dynamic_state {
                        record_extended_state(extended, command, state, &recording);
                    }
                }
                (Some(DynamicRecord::Core), vk::DynamicState::DEPTH_BIAS) => {
                    device.cmd_set_depth_bias(command, 0.0, 0.0, 0.0)
                }
                (Some(DynamicRecord::Core), vk::DynamicState::BLEND_CONSTANTS) => {
                    device.cmd_set_blend_constants(command, &[0.0; 4])
                }
                (Some(DynamicRecord::Core), _) => {
                    let ds = &recording.encoder.depth_stencil;
                    for (face, ops) in [
                        (vk::StencilFaceFlags::FRONT, ds.front),
                        (vk::StencilFaceFlags::BACK, ds.back),
                    ] {
                        match state {
                            vk::DynamicState::STENCIL_COMPARE_MASK => {
                                device.cmd_set_stencil_compare_mask(command, face, ops.compare_mask)
                            }
                            vk::DynamicState::STENCIL_WRITE_MASK => {
                                device.cmd_set_stencil_write_mask(command, face, ops.write_mask)
                            }
                            _ => device.cmd_set_stencil_reference(command, face, ops.reference),
                        }
                    }
                }
            }
        }
        let dynamic = |state| {
            recording
                .dynamic
                .iter()
                .any(|&s| vk::DynamicState::from(s) == state)
        };
        let (set_viewport, set_scissor) = (
            dynamic(vk::DynamicState::VIEWPORT),
            dynamic(vk::DynamicState::SCISSOR),
        );
        // One draw per recording of it: the whole target, or each region of a rate-mapped pass.
        for (viewport, scissor) in recording.draws {
            if set_viewport {
                device.cmd_set_viewport(command, 0, std::slice::from_ref(viewport));
            }
            if set_scissor {
                device.cmd_set_scissor(command, 0, std::slice::from_ref(scissor));
            }
            match (draw, recording.index_buffer) {
                (
                    metal2vulkan::vk::DrawCommand::DrawIndexed {
                        index_type,
                        index_buffer_offset,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    },
                    Some(index_buffer),
                ) => {
                    device.cmd_bind_index_buffer(
                        command,
                        index_buffer,
                        index_buffer_offset,
                        vk::IndexType::from_raw(index_type.as_raw()),
                    );
                    device.cmd_draw_indexed(
                        command,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    );
                }
                (
                    metal2vulkan::vk::DrawCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    },
                    _,
                ) => device.cmd_draw(
                    command,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ),
                // A draw replay's draws, each of one draw at its arguments.
                (metal2vulkan::vk::DrawCommand::DrawIndirect { .. }, _) => {
                    let Some(replayed) = recording.replayed else {
                        continue;
                    };
                    for draw in replayed.draws {
                        for (stages, offset, bytes) in &draw.push {
                            device.cmd_push_constants(
                                command,
                                objects.layout.pipeline_layout,
                                *stages,
                                *offset,
                                bytes,
                            );
                        }
                        device.cmd_draw_indirect(
                            command,
                            replayed.scratch,
                            draw.arguments,
                            1,
                            DRAW_ARGUMENT_BYTES as u32,
                        );
                    }
                }
                // Refused before recording began.
                _ => {}
            }
        }
        device.cmd_end_render_pass(command);
        if recording.replayed.is_some() {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[drawn],
                &[],
                &[],
            );
        }
        if let (Some(shadow), Some(target)) = (recording.shadow, depth_stencil) {
            shadow.record_copy_out(command, target);
        }
        for planes in recording.planes {
            planes.record_merge(command);
        }
        if !to_source.is_empty() {
            device.cmd_pipeline_barrier(
                command,
                attachment_stages,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_source,
            );
            for t in targets.targets.values() {
                if let Some(readback) = t.transfer.readback() {
                    device.cmd_copy_image_to_buffer(
                        command,
                        t.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        readback.buffer,
                        &[copy],
                    );
                }
            }
            if let Some(t) = depth_stencil {
                for aspect in &t.aspects {
                    if let Some(readback) = aspect.transfer.readback() {
                        device.cmd_copy_image_to_buffer(
                            command,
                            t.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            readback.buffer,
                            &[region(aspect.aspect)],
                        );
                    }
                }
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &transfer_to_host,
                &[],
                &[],
            );
        }
        if let Some(readers) = recording.readers {
            readers.record_reads(command, attachment_stages);
        }
        recording.resources.record_readback(command);
        device
            .end_command_buffer(command)
            .map_err(|error| format!("end command buffer: {error}"))?;
        objects.fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|error| format!("create completion fence: {error}"))?;
        let commands = [command];
        let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
        device
            .queue_submit(context.queue, &submit, objects.fence)
            .map_err(|error| format!("submit command buffer: {error}"))?;
    }
    context.wait_for_fence(objects.fence, "engine render candidate execution")?;
    Ok(())
}

mod multisampled;
mod planes;
mod shadow;
#[cfg(test)]
mod tests;
