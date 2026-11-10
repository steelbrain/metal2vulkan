use super::*;
use crate::check::interface::AirInterface;
use crate::gpu::compile;
use crate::gpu::wait::metal::commit_and_wait;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLCullMode, MTLDevice, MTLFunction, MTLIndexType, MTLLibrary, MTLLoadAction, MTLOrigin,
    MTLPipelineOption, MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLRegion,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions, MTLSize,
    MTLStageInputOutputDescriptor, MTLStepFunction, MTLStoreAction,
    MTLTessellationControlPointIndexType, MTLTessellationFactorFormat,
    MTLTessellationFactorStepFunction, MTLTessellationPartitionMode, MTLVertexDescriptor,
    MTLViewport, MTLWinding,
};
use std::cell::RefCell;

mod acceleration;
mod argument_buffers;
mod attachments;
mod buffers;
mod function_tables;
mod imageblocks;
mod indirect_commands;
mod libraries;
mod mesh;
mod pipelines;
mod readback;
mod sample_readback;
mod samplers;
mod textures;
mod tile;
mod vertex_inputs;
use function_tables::{
    bind_fragment_function_tables, bind_vertex_function_tables, make_compute_function_tables,
    make_render_function_tables, MetalFunctionTables,
};
pub(crate) use libraries::load_library;
pub(super) use libraries::reusing_tool_results;
pub(crate) use libraries::shader_air;
use libraries::{
    arm_fuel, check_fuel, load_linked_functions, load_probe_library, load_producer_library,
    make_function, MetalLinkedFunctions,
};

use pipelines::{
    companion_library, encode_pipeline_templates, make_compute_pipeline, make_pipeline_templates,
    make_render_functions, make_render_pipeline, metal_cull_mode, NativePipeline,
};

use readback::{selected_output, MetalOutputResources};

use acceleration::make_acceleration_structures;
use samplers::make_samplers;

use crate::executor_contract::buffer_slots::NativeBufferSlots;
use crate::executor_contract::render_target_array::render_target_array_length;

use vertex_inputs::{make_vertex_buffers, make_vertex_layout};

use indirect_commands::{
    encode_host_commands, encode_indirect_commands, execute_host_commands, make_indirect_commands,
    IndirectCommands,
};

use argument_buffers::{
    argument_buffer_resources_resident_on_compute, argument_buffer_resources_resident_on_render,
    argument_encoder, argument_member, encode_argument_buffer_buffers,
    encode_argument_buffer_function_tables, encode_argument_buffer_samplers,
    encode_argument_buffer_textures, make_argument_buffer_buffers, make_argument_buffer_textures,
    ArgumentBufferBuffer, ArgumentBufferTexture,
};

use textures::{
    make_texture, make_texture_arrays, make_textures, metal_pixel_format, texture_shader_usage,
    MetalTextureLiteral, Texture, TextureArray,
};

use attachments::{
    make_depth_stencil, make_depth_stencil_state, make_rasterization_sink, make_render_targets,
    DepthStencilTexture,
};

use tile::execute_tile;

use imageblocks::make_fragment_imageblock_execution;

use buffers::{
    compile_rate_map, make_buffers, new_buffer_from_slice, shadow_offset,
    shadow_out_of_bounds_targets, DeviceBufferArrayElement, NativeBuffers,
};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Function = Retained<ProtocolObject<dyn MTLFunction>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// The facts the oracle binds a shader's resources by: its AIR-layer interface for slots,
/// argument ids, and declared access, and the declared reflection ([`crate::air::reflect`]) for the
/// families read from it (imageblocks, tessellation, depth and stencil).
#[derive(Clone, Copy)]
struct ShaderFacts<'a> {
    reflection: &'a crate::air::reflect::ShaderInterface,
    interface: &'a AirInterface,
}

struct MetalContext {
    device: Device,
    queue: CommandQueue,
}

/// Open the system Metal device. The only `MTLDevice` creation in this crate: it refuses while
/// the hang tripwire exists. Public for the crate's binaries (the format scout's Metal side).
pub fn system_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>, String> {
    crate::gpu::tripwire::refuse_if_tripped()?;
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_string())
}

impl MetalContext {
    fn new() -> Result<Self, String> {
        let device = system_device()?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_string())?;
        Ok(Self { device, queue })
    }
}

/// Execute one case on the Metal device. The context is created on first use per thread and is
/// used only by that thread. Under
/// [`OracleBound::Fuel`](crate::oracle_fuel::OracleBound) the case's AIR runs fuel-instrumented
/// and the status buffer is checked before any output is read.
pub fn execute(
    case: &AuthoredCase,
    source: OracleSource<'_>,
    resources: &LiteralResources,
    reflection: &crate::air::reflect::ShaderInterface,
    interface: &AirInterface,
    function_tables: &crate::library_module::ResolvedLinkedFunctions,
    templates: &BTreeMap<String, OracleTemplate<'_>>,
) -> Result<Vec<u8>, MetalRunError> {
    debug_assert!(crate::executor_contract::require_case(case, "Metal executor").is_ok());
    thread_local! {
        static CONTEXT: RefCell<Option<MetalContext>> = const { RefCell::new(None) };
    }
    crate::gpu::tripwire::refuse_if_tripped()?;
    let label = crate::gpu::wait::SubmissionLabel {
        case_id: case.case_id.clone(),
        module_sha256: source.row.air_sha256.clone(),
    };
    crate::gpu::wait::with_submission_label(label, || {
        autoreleasepool(|_| {
            CONTEXT.with(|slot| {
                let mut slot = slot.borrow_mut();
                if slot.is_none() {
                    *slot = Some(MetalContext::new()?);
                }
                let context = slot.as_ref().expect("initialized Metal context");
                let pipelines = make_pipeline_templates(&context.device, templates)?;
                let facts = ShaderFacts {
                    reflection,
                    interface,
                };
                if case.out_of_bounds.is_some()
                    && (case.stage == crate::case::Stage::Fragment
                        || !reflection.implicit_imageblock_attachments.is_empty())
                {
                    // The case checker admits out-of-bounds targets in kernel cases and, of an
                    // attribute or index buffer, in vertex cases.
                    return Err(
                        "only a compute dispatch or a vertex draw binds out-of-bounds shadows"
                            .into(),
                    );
                }
                match case.stage {
                    crate::case::Stage::Kernel
                        if !reflection.implicit_imageblock_attachments.is_empty() =>
                    {
                        // A tile function's commands are neither host-encoded nor executed.
                        if case.command_buffers.iter().any(|resource| {
                            !resource.commands.is_empty() || resource.execute.is_some()
                        }) {
                            return Err("a tile kernel's host-encoded or executed indirect \
                                        commands are not modelled"
                                .into());
                        }
                        execute_tile(
                            context,
                            case,
                            source,
                            resources,
                            facts,
                            function_tables,
                            &pipelines,
                        )
                    }
                    crate::case::Stage::Kernel => execute_compute(
                        context,
                        case,
                        source,
                        resources,
                        facts,
                        function_tables,
                        &pipelines,
                    ),
                    crate::case::Stage::Fragment => execute_fragment(
                        context,
                        case,
                        source,
                        resources,
                        facts,
                        function_tables,
                        &pipelines,
                    ),
                    crate::case::Stage::Vertex => execute_vertex(
                        context,
                        case,
                        source,
                        resources,
                        facts,
                        function_tables,
                        &pipelines,
                    ),
                    crate::case::Stage::Object | crate::case::Stage::Mesh => {
                        mesh::execute_mesh(context, case, source, resources, facts)
                    }
                }
            })
        })
    })
}

/// The CPU-side part of [`execute`]'s libraries: for the case's own, slot allocation, the fuel
/// transform under [`OracleBound::Fuel`](crate::oracle_fuel::OracleBound), `metal-opt`, and
/// `metallib`; then `metal-as`, `metal-opt`, and `metallib` for its generated companion's, and
/// `metal-opt` and `metallib` for each pipeline template's (and its companion's) and each linked
/// module's. It opens no device and runs nothing.
pub(super) fn prepare(
    case: &AuthoredCase,
    source: OracleSource<'_>,
    interface: &AirInterface,
    linked: &crate::library_module::ResolvedLinkedFunctions,
    templates: &BTreeMap<String, OracleTemplate<'_>>,
) -> Result<libraries::PreparedLibrary, String> {
    let reserved = match case.stage {
        crate::case::Stage::Vertex => make_vertex_layout(case, interface)?
            .slots
            .iter()
            .map(|(slot, _)| *slot as u32)
            .collect(),
        _ => Vec::new(),
    };
    let mut prepared = libraries::prepare_producer_library(case, &source, interface, &reserved)?;
    prepared
        .library_sha256
        .extend(libraries::prepare_dependency_libraries(
            source.companion,
            linked,
            templates,
        )?);
    Ok(prepared)
}

/// Time compute pipeline creation for a kernel case's own module: as compiled (`loops` `None`) or
/// fuel-instrumented with only the selected loops checked ([`load_probe_library`]), with the case's
/// function constants and linked functions, under the compile bound. It opens the device and
/// creates the pipeline, and encodes and runs nothing, so a module whose loops are unbounded never
/// executes.
pub(super) fn probe_compile(
    case: &AuthoredCase,
    source: OracleSource<'_>,
    interface: &AirInterface,
    function_tables: &crate::library_module::ResolvedLinkedFunctions,
    constants: &[crate::literal::LiteralFunctionConstant],
    loops: Option<crate::oracle_fuel::LoopSelection>,
) -> Result<CompileProbe, String> {
    if case.stage != crate::case::Stage::Kernel || !case.command_buffers.is_empty() {
        return Err("the compile probe times a kernel case without command buffers".into());
    }
    autoreleasepool(|_| {
        let device = system_device()?;
        let (library, fuel) = load_probe_library(&device, case, &source, interface, loops)?;
        let function = make_function(&library, &NSString::from_str(&case.entry), constants)?;
        let linked = load_linked_functions(&device, function_tables, constants)?;
        let start = std::time::Instant::now();
        make_compute_pipeline(
            &device,
            &function,
            &case.kernel_stage_inputs,
            interface,
            &linked,
            false,
        )
        .map_err(|error| match error {
            MetalRunError::Failed(error) => error,
            MetalRunError::ReferenceGap(gap) => gap.message,
        })?;
        Ok(CompileProbe {
            seconds: start.elapsed().as_secs_f64(),
            fuel,
        })
    })
}

fn execute_compute(
    context: &MetalContext,
    case: &AuthoredCase,
    source: OracleSource<'_>,
    resources: &LiteralResources,
    facts: ShaderFacts<'_>,
    function_tables: &crate::library_module::ResolvedLinkedFunctions,
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<Vec<u8>, MetalRunError> {
    let ShaderFacts {
        reflection,
        interface,
    } = facts;
    let device = &context.device;
    let NativeBuffers {
        mut buffers,
        device_buffer_array_elements,
    } = make_buffers(device, resources, reflection, interface)?;
    shadow_out_of_bounds_targets(device, case, interface, &mut buffers)?;
    let indirect_commands = make_indirect_commands(device, case)?;
    let (library, mut fuel) = load_producer_library(
        device,
        &source,
        &indirect_commands,
        interface,
        &mut buffers,
        &[],
    )?;
    let entry = NSString::from_str(&case.entry);
    let function = make_function(&library, &entry, &resources.function_constants)?;
    let linked = load_linked_functions(device, function_tables, &resources.function_constants)?;
    let pipeline = make_compute_pipeline(
        device,
        &function,
        &case.kernel_stage_inputs,
        interface,
        &linked,
        false,
    )?;
    let metal_function_tables = make_compute_function_tables(&pipeline, function_tables, &linked)?;
    let argument_buffer_buffers = make_argument_buffer_buffers(device, case, resources)?;
    let textures = make_textures(device, &context.queue, resources, interface)?;
    let texture_arrays = make_texture_arrays(device, &context.queue, resources, interface)?;
    let argument_buffer_textures =
        make_argument_buffer_textures(device, &context.queue, resources, interface)?;
    encode_argument_buffer_buffers(
        &function,
        case,
        &buffers,
        &argument_buffer_buffers,
        interface,
    )?;
    encode_pipeline_templates(
        &function,
        &buffers,
        &case.pipeline_states,
        interface,
        pipelines,
    )?;
    encode_indirect_commands(&function, &buffers, &indirect_commands.commands, interface)?;
    encode_host_commands(case, &indirect_commands.commands, &buffers, pipelines)?;
    encode_argument_buffer_textures(&function, &buffers, &argument_buffer_textures, interface)?;
    let _argument_buffer_samplers =
        encode_argument_buffer_samplers(device, &function, case, &buffers, interface)?;
    encode_argument_buffer_function_tables(&function, &buffers, &metal_function_tables, interface)?;
    let samplers = make_samplers(device, case)?;
    let dispatch = case
        .dispatch
        .as_ref()
        .ok_or_else(|| "kernel case has no dispatch".to_string())?;
    let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| "computeCommandEncoder returned nil".to_string())?;
    encoder.setComputePipelineState(&pipeline);
    argument_buffer_resources_resident_on_compute(
        &encoder,
        &argument_buffer_textures,
        &argument_buffer_buffers,
        &indirect_commands.commands,
    );
    // A stage input of a dynamic stride binds its data with the stride form, at the case's
    // stride, as the engine candidate binds it.
    let strides = crate::executor_contract::stage_inputs::bound_strides(
        &crate::executor_contract::stage_inputs::authored(&case.kernel_stage_inputs, interface)?,
    );
    for (binding, buffer) in &buffers {
        let offset = shadow_offset(case, *binding);
        match strides.get(binding) {
            Some(&stride) => unsafe {
                encoder.setBuffer_offset_attributeStride_atIndex(
                    buffer,
                    offset,
                    stride as usize,
                    *binding as usize,
                )
            },
            None => unsafe {
                encoder.setBuffer_offset_atIndex(Some(&**buffer), offset, *binding as usize)
            },
        }
    }
    for (binding, acceleration_structure) in &acceleration_structures {
        unsafe {
            encoder.setAccelerationStructure_atBufferIndex(
                Some(&**acceleration_structure),
                *binding as usize,
            );
        }
    }
    for (binding, table) in &metal_function_tables.visible {
        unsafe {
            encoder.setVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize);
        }
    }
    for (location, table) in &metal_function_tables.intersection {
        if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
            binding,
        } = location
        {
            unsafe {
                encoder
                    .setIntersectionFunctionTable_atBufferIndex(Some(&**table), *binding as usize);
            }
        }
    }
    for (binding, texture) in &textures {
        unsafe { encoder.setTexture_atIndex(Some(&**texture), *binding as usize) };
    }
    for (binding, elements) in &texture_arrays {
        for (element, texture) in elements.iter().enumerate() {
            unsafe { encoder.setTexture_atIndex(Some(&**texture), *binding as usize + element) };
        }
    }
    for (binding, sampler) in &samplers {
        unsafe { encoder.setSamplerState_atIndex(Some(&**sampler), *binding as usize) };
    }
    for resource in &case.threadgroup_memory {
        // An out-of-bounds target's shadow past it.
        let shadow =
            crate::executor_contract::out_of_bounds::threadgroup_reach(case, resource.binding);
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(
                (resource.length + shadow) as usize,
                resource.binding as usize,
            );
        }
    }
    if let Some(imageblock) = &case.imageblock {
        encoder.setImageblockWidth_height(
            imageblock.dimensions[0] as usize,
            imageblock.dimensions[1] as usize,
        );
    }
    if !resources.kernel_stage_inputs.is_empty() {
        // The region the engine candidate binds too.
        let region =
            crate::executor_contract::stage_inputs::region(dispatch.stage_in_region, dispatch.grid);
        let [x, y, z] = region.origin.map(|v| v as usize);
        encoder.setStageInRegion(MTLRegion {
            origin: MTLOrigin { x, y, z },
            size: mtl_size(region.size),
        });
    }
    threads_per_threadgroup_within_limit(
        dispatch.threads_per_threadgroup,
        pipeline.maxTotalThreadsPerThreadgroup(),
        "Metal compute",
    )?;
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(dispatch.grid),
        mtl_size(dispatch.threads_per_threadgroup),
    );
    encoder.endEncoding();
    arm_fuel(fuel.as_mut(), &buffers);
    commit_and_wait(&command_buffer, "Metal compute dispatch")?;
    check_fuel(fuel.as_ref())?;
    indirect_commands
        .validate_readback()
        .map_err(|error| error.to_string())?;
    execute_host_commands(
        device,
        &context.queue,
        case,
        &indirect_commands.commands,
        &buffers,
        &argument_buffer_buffers,
        pipelines,
    )?;
    Ok(selected_output(
        case,
        &buffers,
        &argument_buffer_buffers,
        &textures,
        &texture_arrays,
        &argument_buffer_textures,
        MetalOutputResources {
            device_buffer_array_elements: &device_buffer_array_elements,
            colors: &[],
            depth_stencil: None,
            fragment_imageblock: None,
        },
    )?)
}

fn execute_fragment(
    context: &MetalContext,
    case: &AuthoredCase,
    source: OracleSource<'_>,
    resources: &LiteralResources,
    facts: ShaderFacts<'_>,
    function_tables: &crate::library_module::ResolvedLinkedFunctions,
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<Vec<u8>, MetalRunError> {
    let ShaderFacts {
        reflection,
        interface,
    } = facts;
    let device = &context.device;
    let NativeBuffers {
        mut buffers,
        device_buffer_array_elements,
    } = make_buffers(device, resources, reflection, interface)?;
    let indirect_commands = make_indirect_commands(device, case)?;
    let (fragment_library, mut fuel) = load_producer_library(
        device,
        &source,
        &indirect_commands,
        interface,
        &mut buffers,
        &[],
    )?;
    let entry = NSString::from_str(&case.entry);
    let fragment = make_function(&fragment_library, &entry, &resources.function_constants)?;
    let linked = load_linked_functions(device, function_tables, &resources.function_constants)?;
    // The pass's render target array length, the derivation the engine candidate binds: layered
    // at 1 when the function reads its slice.
    let render_target_array_length =
        render_target_array_length(crate::case::Stage::Fragment, source.entry_ll);
    let layered_rendering = render_target_array_length > 0;
    let functions = make_render_functions(device, case, source.companion, fragment.clone())?;
    let vertex = &functions.0;
    let render_targets = make_render_targets(device, &context.queue, resources, layered_rendering)?;
    let depth_stencil = make_depth_stencil(device, &context.queue, resources, layered_rendering)?;
    let pipeline = make_render_pipeline(device, case, &functions, &linked, None, false)?;
    let fragment_imageblock = make_fragment_imageblock_execution(
        device,
        vertex,
        &render_targets,
        depth_stencil.as_ref(),
        case,
        resources,
        reflection,
    )?;
    let metal_function_tables = make_render_function_tables(
        &pipeline,
        MTLRenderStages::Fragment,
        function_tables,
        &linked,
    )?;

    let argument_buffer_buffers = make_argument_buffer_buffers(device, case, resources)?;
    let textures = make_textures(device, &context.queue, resources, interface)?;
    let texture_arrays = make_texture_arrays(device, &context.queue, resources, interface)?;
    let argument_buffer_textures =
        make_argument_buffer_textures(device, &context.queue, resources, interface)?;
    encode_argument_buffer_buffers(
        &fragment,
        case,
        &buffers,
        &argument_buffer_buffers,
        interface,
    )?;
    encode_pipeline_templates(
        &fragment,
        &buffers,
        &case.pipeline_states,
        interface,
        pipelines,
    )?;
    encode_indirect_commands(&fragment, &buffers, &indirect_commands.commands, interface)?;
    encode_host_commands(case, &indirect_commands.commands, &buffers, pipelines)?;
    encode_argument_buffer_textures(&fragment, &buffers, &argument_buffer_textures, interface)?;
    let _argument_buffer_samplers =
        encode_argument_buffer_samplers(device, &fragment, case, &buffers, interface)?;
    encode_argument_buffer_function_tables(&fragment, &buffers, &metal_function_tables, interface)?;
    let samplers = make_samplers(device, case)?;
    let draw_depth_stencil = crate::executor_contract::DrawDepthStencil::of_case(
        case.depth_stencil
            .as_ref()
            .and_then(|attachment| attachment.state.as_ref()),
        interface,
    );
    let depth_stencil_state = make_depth_stencil_state(device, &draw_depth_stencil)?;
    let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let rate_map = set_render_pass_rate_map(device, case, &pass)?;
    if render_targets.is_empty() && depth_stencil.is_none() {
        // Metal needs explicit raster dimensions when a fragment pass has no attachments.
        // Buffer side effects can still be observed without a color or depth attachment. The
        // pass's sample count is the case's: 1 for a pass of one sample.
        pass.setDefaultRasterSampleCount(attachmentless_raster_sample_count(case));
        pass.setRenderTargetWidth(1);
        pass.setRenderTargetHeight(1);
    }
    if layered_rendering {
        pass.setRenderTargetArrayLength(render_target_array_length as usize);
    }
    let pass_attachments = pass.colorAttachments();
    for (index, texture) in &render_targets {
        let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setTexture(Some(&**texture));
        attachment.setLoadAction(MTLLoadAction::Load);
        attachment.setStoreAction(MTLStoreAction::Store);
    }
    if let Some(attachment) = &depth_stencil {
        if let Some(texture) = &attachment.depth {
            let depth = pass.depthAttachment();
            depth.setTexture(Some(&**texture));
            depth.setLoadAction(MTLLoadAction::Load);
            depth.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(texture) = &attachment.stencil {
            let stencil = pass.stencilAttachment();
            stencil.setTexture(Some(&**texture));
            stencil.setLoadAction(MTLLoadAction::Load);
            stencil.setStoreAction(MTLStoreAction::Store);
        }
    }
    if let Some(imageblock) = &fragment_imageblock {
        let sample_length = pipeline.imageblockSampleLength();
        if sample_length != imageblock.initialize.imageblockSampleLength()
            || imageblock
                .resolve
                .as_ref()
                .is_some_and(|resolve| resolve.imageblockSampleLength() != sample_length)
        {
            return Err("generated and target fragment imageblock layouts disagree".into());
        }
        pass.setImageblockSampleLength(sample_length);
        // Metal render-pass tile dimensions describe the hardware tile, not the attachment or
        // authored plane extent. Apple GPUs reject sub-tile values (including a 1x1 fixture),
        // so use the same portable 16x16 tile exercised by the existing tile executor.
        pass.setTileWidth(16);
        pass.setTileHeight(16);
    }
    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
    set_rate_map_viewport(&encoder, rate_map);
    if let Some(imageblock) = &fragment_imageblock {
        encoder.setRenderPipelineState(&imageblock.initialize);
        for (binding, buffer) in &imageblock.inputs {
            unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(&**buffer), 0, *binding) };
        }
        unsafe { encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3) };
    }
    encoder.setRenderPipelineState(&pipeline);
    encoder.setDepthStencilState(Some(&*depth_stencil_state));
    // The encoder's references and winding, set only where a case authors other than Metal's
    // defaults (references 0, clockwise).
    if draw_depth_stencil.front_reference != 0 || draw_depth_stencil.back_reference != 0 {
        encoder.setStencilFrontReferenceValue_backReferenceValue(
            draw_depth_stencil.front_reference,
            draw_depth_stencil.back_reference,
        );
    }
    if draw_depth_stencil.winding != metal2vulkan::metal::Winding::Clockwise {
        encoder.setFrontFacingWinding(objc2_metal::MTLWinding(
            draw_depth_stencil.winding.as_raw() as _
        ));
    }
    argument_buffer_resources_resident_on_render(
        &encoder,
        &argument_buffer_textures,
        &argument_buffer_buffers,
        &indirect_commands.commands,
        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
    );
    for (binding, buffer) in &buffers {
        unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize) };
    }
    for (binding, acceleration_structure) in &acceleration_structures {
        unsafe {
            encoder.setFragmentAccelerationStructure_atBufferIndex(
                Some(&**acceleration_structure),
                *binding as usize,
            );
        }
    }
    bind_fragment_function_tables(&encoder, &metal_function_tables);
    for (binding, texture) in &textures {
        unsafe { encoder.setFragmentTexture_atIndex(Some(&**texture), *binding as usize) };
    }
    for (binding, elements) in &texture_arrays {
        for (element, texture) in elements.iter().enumerate() {
            unsafe {
                encoder.setFragmentTexture_atIndex(Some(&**texture), *binding as usize + element);
            }
        }
    }
    for (binding, sampler) in &samplers {
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(&**sampler), *binding as usize);
        }
    }
    let draw = case.draw.as_ref().expect("validated fragment draw");
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
            metal_primitive(draw.primitive),
            draw.vertex_start as usize,
            draw.vertex_count as usize,
            draw.instance_count as usize,
        );
    }
    if let Some(imageblock) = &fragment_imageblock {
        if let (Some(resolve), Some(output)) = (&imageblock.resolve, &imageblock.output) {
            encoder.setRenderPipelineState(resolve);
            unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(&**output), 0, 0) };
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3)
            };
        }
    }
    encoder.endEncoding();
    arm_fuel(fuel.as_mut(), &buffers);
    commit_and_wait(&command_buffer, "Metal fragment draw")?;
    check_fuel(fuel.as_ref())?;
    indirect_commands
        .validate_readback()
        .map_err(|error| error.to_string())?;
    execute_host_commands(
        device,
        &context.queue,
        case,
        &indirect_commands.commands,
        &buffers,
        &argument_buffer_buffers,
        pipelines,
    )?;
    if let Some(output) =
        multisampled_output(context, case, &render_targets, depth_stencil.as_ref())?
    {
        return Ok(output);
    }
    Ok(selected_output(
        case,
        &buffers,
        &argument_buffer_buffers,
        &textures,
        &texture_arrays,
        &argument_buffer_textures,
        MetalOutputResources {
            device_buffer_array_elements: &device_buffer_array_elements,
            colors: &render_targets,
            depth_stencil: depth_stencil.as_ref(),
            fragment_imageblock: fragment_imageblock.as_ref().and_then(|imageblock| {
                imageblock
                    .output
                    .as_ref()
                    .zip(imageblock.output_pixel_size)
                    .map(|(output, pixel_size)| (output, imageblock.dimensions, pixel_size))
            }),
        },
    )?)
}

/// The default raster sample count of `case`'s pass without attachments: the case's, and 1 for a
/// pass of one sample, as every attachmentless pass was drawn before.
fn attachmentless_raster_sample_count(case: &AuthoredCase) -> usize {
    case.raster_samples() as usize
}

/// A multisampled pass's render-target, depth, or stencil output: every sample of the selected
/// region, read sample by sample ([`sample_readback::read_samples`]) and never resolved. `None` for
/// a pass of one sample, whose outputs [`selected_output`] reads as before, and for any other
/// output.
fn multisampled_output(
    context: &MetalContext,
    case: &AuthoredCase,
    render_targets: &[(u32, Texture)],
    depth_stencil: Option<&DepthStencilTexture>,
) -> Result<Option<Vec<u8>>, String> {
    use crate::executor_contract::multisample::{
        attachment_transfer, sample_read, AttachmentTransfer, SampleRead,
    };
    let AttachmentTransfer::PerSample { samples } = attachment_transfer(case) else {
        return Ok(None);
    };
    let (texture, read, extent, origin, dimensions) = match &case.output {
        crate::case::OutputSelection::RenderTarget {
            index,
            origin,
            dimensions,
        } => {
            let texture = render_targets
                .iter()
                .find_map(|(candidate, texture)| (candidate == index).then_some(texture))
                .ok_or_else(|| format!("output render target {index} was not bound"))?;
            let resource = case
                .render_targets
                .iter()
                .find(|resource| resource.index == *index)
                .ok_or_else(|| format!("output render target {index} is not declared"))?;
            (
                texture,
                sample_read(resource.format)?,
                resource.dimensions,
                *origin,
                *dimensions,
            )
        }
        crate::case::OutputSelection::Depth { origin, dimensions } => {
            let texture = depth_stencil
                .and_then(|attachment| attachment.depth.as_ref())
                .ok_or("the depth output's aspect was not allocated")?;
            let resource = case
                .depth_stencil
                .as_ref()
                .ok_or("the depth output's attachment is not declared")?;
            (
                texture,
                sample_read(crate::case::TextureFormat::Depth32Float)?,
                resource.dimensions,
                *origin,
                *dimensions,
            )
        }
        crate::case::OutputSelection::Stencil { origin, dimensions } => {
            let texture = depth_stencil
                .and_then(|attachment| attachment.stencil.as_ref())
                .ok_or("the stencil output's aspect was not allocated")?;
            let resource = case
                .depth_stencil
                .as_ref()
                .ok_or("the stencil output's attachment is not declared")?;
            (
                texture,
                SampleRead::Stencil,
                resource.dimensions,
                *origin,
                *dimensions,
            )
        }
        _ => return Ok(None),
    };
    let bytes = sample_readback::read_samples(
        &context.device,
        &context.queue,
        texture,
        read,
        extent,
        samples,
    )?;
    crate::literal::select_tightly_packed_2d(
        &bytes,
        extent,
        origin,
        dimensions,
        read.texel_bytes() * samples as usize,
    )
    .map(Some)
}

fn execute_vertex(
    context: &MetalContext,
    case: &AuthoredCase,
    source: OracleSource<'_>,
    resources: &LiteralResources,
    facts: ShaderFacts<'_>,
    function_tables: &crate::library_module::ResolvedLinkedFunctions,
    pipelines: &BTreeMap<String, NativePipeline>,
) -> Result<Vec<u8>, MetalRunError> {
    let ShaderFacts {
        reflection,
        interface,
    } = facts;
    let device = &context.device;
    let NativeBuffers {
        mut buffers,
        device_buffer_array_elements,
    } = make_buffers(device, resources, reflection, interface)?;
    let layout = make_vertex_layout(case, interface)?;
    let indirect_commands = make_indirect_commands(device, case)?;
    let (vertex_library, mut fuel) = load_producer_library(
        device,
        &source,
        &indirect_commands,
        interface,
        &mut buffers,
        &layout
            .slots
            .iter()
            .map(|(slot, _)| *slot as u32)
            .collect::<Vec<_>>(),
    )?;
    let entry = NSString::from_str(&case.entry);
    let vertex = make_function(&vertex_library, &entry, &resources.function_constants)?;
    let linked = load_linked_functions(device, function_tables, &resources.function_constants)?;
    // The pass's render target array length, the derivation the engine candidate binds: layered
    // at 1 when the function chooses its slice.
    let render_target_array_length =
        render_target_array_length(crate::case::Stage::Vertex, source.entry_ll);
    let layered_rendering = render_target_array_length > 0;
    let functions = make_render_functions(device, case, source.companion, vertex.clone())?;
    let mut render_targets =
        make_render_targets(device, &context.queue, resources, layered_rendering)?;
    if case.is_rasterization_disabled_vertex() {
        // Metal still requires a render-pass attachment to create an encoder. Rasterization is
        // disabled on the pipeline, so this private 1x1 sink is API scaffolding and cannot
        // participate in the authored observation.
        render_targets.push((0, make_rasterization_sink(device)?));
    }
    // A vertex draw shadows its attribute buffer and index buffer targets, and no other: the
    // checker admits no other target in a vertex case.
    if let Some(declaration) = &case.out_of_bounds {
        use crate::case::out_of_bounds::OutOfBoundsResource as R;
        if let Some(target) = declaration
            .targets
            .iter()
            .find(|target| !matches!(target.resource, R::AttributeBuffer { .. } | R::IndexBuffer))
        {
            return Err(format!("a vertex draw shadows no {}", target.resource.label()).into());
        }
    }
    let vertex_buffers = make_vertex_buffers(device, case, resources, &layout)?;
    // An indexed draw's index buffer, with its shadow past its indices when the case declares it
    // an out-of-bounds target.
    let index_buffer = match case
        .draw
        .as_ref()
        .and_then(|draw| draw.index_buffer.as_ref())
    {
        None => None,
        Some(index_buffer) => {
            let label = "index buffer";
            let bytes = index_buffer
                .bytes()
                .ok_or_else(|| format!("{label} bytes are not base64"))?;
            let reach = crate::executor_contract::out_of_bounds::reach(
                case,
                crate::case::out_of_bounds::OutOfBoundsResource::IndexBuffer,
            );
            let bytes = buffers::with_shadow_past(&bytes, reach, label)?;
            Some((index_buffer, buffers::upload_buffer(device, &bytes, label)?))
        }
    };
    let pipeline = make_render_pipeline(
        device,
        case,
        &functions,
        &linked,
        Some(&layout.descriptor),
        false,
    )?;
    let metal_function_tables =
        make_render_function_tables(&pipeline, MTLRenderStages::Vertex, function_tables, &linked)?;

    let argument_buffer_buffers = make_argument_buffer_buffers(device, case, resources)?;
    let textures = make_textures(device, &context.queue, resources, interface)?;
    let texture_arrays = make_texture_arrays(device, &context.queue, resources, interface)?;
    let argument_buffer_textures =
        make_argument_buffer_textures(device, &context.queue, resources, interface)?;
    encode_argument_buffer_buffers(&vertex, case, &buffers, &argument_buffer_buffers, interface)?;
    encode_pipeline_templates(
        &vertex,
        &buffers,
        &case.pipeline_states,
        interface,
        pipelines,
    )?;
    encode_indirect_commands(&vertex, &buffers, &indirect_commands.commands, interface)?;
    encode_host_commands(case, &indirect_commands.commands, &buffers, pipelines)?;
    encode_argument_buffer_textures(&vertex, &buffers, &argument_buffer_textures, interface)?;
    let _argument_buffer_samplers =
        encode_argument_buffer_samplers(device, &vertex, case, &buffers, interface)?;
    encode_argument_buffer_function_tables(&vertex, &buffers, &metal_function_tables, interface)?;
    let samplers = make_samplers(device, case)?;
    let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let rate_map = set_render_pass_rate_map(device, case, &pass)?;
    if layered_rendering {
        pass.setRenderTargetArrayLength(render_target_array_length as usize);
    }
    let pass_attachments = pass.colorAttachments();
    for (index, texture) in &render_targets {
        let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setTexture(Some(&**texture));
        attachment.setLoadAction(MTLLoadAction::Load);
        attachment.setStoreAction(MTLStoreAction::Store);
    }
    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
    set_rate_map_viewport(&encoder, rate_map);
    encoder.setRenderPipelineState(&pipeline);
    argument_buffer_resources_resident_on_render(
        &encoder,
        &argument_buffer_textures,
        &argument_buffer_buffers,
        &indirect_commands.commands,
        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
    );
    for (slot, buffer, stride) in &vertex_buffers {
        match stride {
            Some(stride) => unsafe {
                encoder.setVertexBuffer_offset_attributeStride_atIndex(
                    Some(&**buffer),
                    0,
                    *stride as usize,
                    *slot,
                )
            },
            None => unsafe { encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *slot) },
        }
    }
    for (binding, buffer) in &buffers {
        unsafe { encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize) };
    }
    for (binding, acceleration_structure) in &acceleration_structures {
        unsafe {
            encoder.setVertexAccelerationStructure_atBufferIndex(
                Some(&**acceleration_structure),
                *binding as usize,
            );
        }
    }
    bind_vertex_function_tables(&encoder, &metal_function_tables);
    for (binding, texture) in &textures {
        unsafe { encoder.setVertexTexture_atIndex(Some(&**texture), *binding as usize) };
    }
    for (binding, elements) in &texture_arrays {
        for (element, texture) in elements.iter().enumerate() {
            unsafe {
                encoder.setVertexTexture_atIndex(Some(&**texture), *binding as usize + element);
            }
        }
    }
    for (binding, sampler) in &samplers {
        unsafe { encoder.setVertexSamplerState_atIndex(Some(&**sampler), *binding as usize) };
    }
    if let (Some(tessellation), Some(draw)) = (&resources.tessellation, &case.tessellation) {
        // The factors, control points, patch start, base instance, and cull mode the engine
        // candidate draws too.
        use crate::executor_contract::tessellation as shared;
        let control_points = shared::control_points(interface)?;
        encoder.setCullMode(metal_cull_mode(shared::cull_mode(draw)));
        let factor_bytes = shared::factor_bytes(&tessellation.factors);
        let factor_pointer = NonNull::new(factor_bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| "tessellation factor pointer is null".to_string())?;
        let factor_buffer = unsafe {
            device.newBufferWithBytes_length_options(
                factor_pointer,
                factor_bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| "create Metal tessellation factor buffer".to_string())?;
        unsafe {
            encoder.setTessellationFactorBuffer_offset_instanceStride(
                Some(&*factor_buffer),
                shared::FACTOR_OFFSET as usize,
                shared::FACTOR_INSTANCE_STRIDE as usize,
            );
            if tessellation.amplification_count > 1 {
                encoder.setVertexAmplificationCount_viewMappings(
                    tessellation.amplification_count as usize,
                    core::ptr::null(),
                );
            }
            encoder.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance(
                    control_points as usize,
                    shared::PATCH_START as usize,
                    tessellation.factors.len(),
                    None,
                    0,
                    tessellation.instance_count as usize,
                    shared::BASE_INSTANCE as usize,
                );
        }
    } else {
        let draw = case.draw.as_ref().expect("validated vertex draw");
        match &index_buffer {
            // With the draw's base instance.
            None => unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    metal_primitive(draw.primitive),
                    draw.vertex_start as usize,
                    draw.vertex_count as usize,
                    draw.instance_count as usize,
                    draw.base_instance as usize,
                );
            },
            // `vertex_count` indices from the buffer's offset, with the draw's base vertex and
            // base instance.
            Some((literal, buffer)) => unsafe {
                encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    metal_primitive(draw.primitive),
                    draw.vertex_count as usize,
                    match literal.index_type {
                        crate::case::IndexType::Uint16 => MTLIndexType::UInt16,
                        crate::case::IndexType::Uint32 => MTLIndexType::UInt32,
                    },
                    buffer,
                    literal.offset as usize,
                    draw.instance_count as usize,
                    draw.base_vertex as isize,
                    draw.base_instance as usize,
                );
            },
        }
    }
    encoder.endEncoding();
    arm_fuel(fuel.as_mut(), &buffers);
    commit_and_wait(&command_buffer, "Metal vertex draw")?;
    check_fuel(fuel.as_ref())?;
    indirect_commands
        .validate_readback()
        .map_err(|error| error.to_string())?;
    execute_host_commands(
        device,
        &context.queue,
        case,
        &indirect_commands.commands,
        &buffers,
        &argument_buffer_buffers,
        pipelines,
    )?;
    Ok(selected_output(
        case,
        &buffers,
        &argument_buffer_buffers,
        &textures,
        &texture_arrays,
        &argument_buffer_textures,
        MetalOutputResources {
            device_buffer_array_elements: &device_buffer_array_elements,
            colors: &render_targets,
            depth_stencil: None,
            fragment_imageblock: None,
        },
    )?)
}

/// Set the case's render pass rate map on `pass`: Metal's compile of the literal, its screen size
/// returned for the viewport ([`set_rate_map_viewport`]).
fn set_render_pass_rate_map(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    pass: &MTLRenderPassDescriptor,
) -> Result<Option<[u32; 2]>, String> {
    let Some(literal) = &case.render_pass_rate_map else {
        return Ok(None);
    };
    let map = compile_rate_map(device, literal, "render pass rasterization-rate map")?;
    pass.setRasterizationRateMap(Some(&map));
    Ok(Some(literal.screen_size))
}

/// A rate-mapped pass's viewport is the map's screen, in screen coordinates: with a map Metal reads
/// the viewport as screen coordinates, and its default, the attachments' size, is not the screen's.
fn set_rate_map_viewport(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    screen: Option<[u32; 2]>,
) {
    if let Some([width, height]) = screen {
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: f64::from(width),
            height: f64::from(height),
            znear: 0.0,
            zfar: 1.0,
        });
    }
}

/// The Metal primitive type of an authored primitive: the one mapping
/// (`case::indirect_commands::metal_primitive_type`), whose raw value is Metal's enumerator.
fn metal_primitive(primitive: crate::case::Primitive) -> MTLPrimitiveType {
    let raw = crate::case::indirect_commands::metal_primitive_type(primitive).as_raw();
    MTLPrimitiveType(raw as usize)
}

/// The pipeline topology class of an authored primitive: the one mapping
/// (`case::Primitive::topology_class`), whose raw value is Metal's enumerator.
fn metal_primitive_topology_class(primitive: crate::case::Primitive) -> MTLPrimitiveTopologyClass {
    MTLPrimitiveTopologyClass(primitive.topology_class().as_raw() as usize)
}

#[cfg(test)]
mod primitive_topology_tests;

#[cfg(test)]
mod indirect_command_allocation_tests;

#[cfg(all(test, m2v_gpu))]
mod predication_tests;

/// Metal drops a dispatch whose threadgroup is larger than the pipeline permits without
/// reporting an error: the command buffer still reaches `Completed`, so every output reads
/// back untouched and the case looks like a kernel that chose to write nothing. Refuse the
/// dispatch instead, naming both the authored threadgroup and the limit Metal computed.
pub(super) fn threads_per_threadgroup_within_limit(
    threads_per_threadgroup: [u32; 3],
    limit: usize,
    what: &str,
) -> Result<(), String> {
    let total = threads_per_threadgroup_total(threads_per_threadgroup, what)?;
    if total > limit {
        return Err(format!(
            "{what} dispatch asks for {total} threads a threadgroup \
             ({threads_per_threadgroup:?}), but the pipeline permits {limit}"
        ));
    }
    Ok(())
}

/// The threads of a threadgroup of `threads_per_threadgroup`, refused where they overflow a count.
pub(super) fn threads_per_threadgroup_total(
    threads_per_threadgroup: [u32; 3],
    what: &str,
) -> Result<usize, String> {
    threads_per_threadgroup
        .iter()
        .try_fold(1usize, |total, extent| total.checked_mul(*extent as usize))
        .ok_or_else(|| {
            format!("{what} threadgroup {threads_per_threadgroup:?} overflows a thread count")
        })
}

fn mtl_size(size: [u32; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}

/// What a row's `environment` records: this host's [`identity`] and the architecture.
pub fn environment() -> Result<serde_json::Value, String> {
    let mut environment = identity()?.record();
    environment.insert("architecture".into(), std::env::consts::ARCH.into());
    Ok(serde_json::Value::Object(environment))
}

/// This host's Metal identity, read once a process: the device, its GPU core count, macOS's version
/// and build, the selected Xcode's build, and the Metal toolchain. Its tools run together, each
/// under the compile deadline.
pub fn identity() -> Result<MetalIdentity, String> {
    static IDENTITY: std::sync::OnceLock<Result<MetalIdentity, String>> =
        std::sync::OnceLock::new();
    IDENTITY.get_or_init(read_identity).clone()
}

fn read_identity() -> Result<MetalIdentity, String> {
    let device = system_device()?.name().to_string();
    let stdout = |what: &str, output: Result<compile::ToolOutput, compile::ToolError>| {
        let output = output.map_err(|error| format!("{what}: {error}"))?;
        if !output.success {
            return Err(format!("{what} failed: {}", output.stderr.trim()));
        }
        Ok(output.stdout)
    };
    let tool = |program: &str, args: &[&str]| {
        stdout(
            &format!("{program} {}", args.join(" ")),
            compile::host_tool(program, args),
        )
    };
    let (os, os_build, developer, metal, ioreg) = std::thread::scope(|scope| {
        let os = scope.spawn(|| tool("sw_vers", &["-productVersion"]));
        let os_build = scope.spawn(|| tool("sw_vers", &["-buildVersion"]));
        let developer = scope.spawn(|| tool("xcode-select", &["-p"]));
        let metal =
            scope.spawn(|| stdout("metal --version", compile::xcrun("metal", ["--version"])));
        let ioreg = scope.spawn(|| tool("ioreg", &["-r", "-c", "AGXAccelerator", "-d", "1"]));
        let join = |handle: std::thread::ScopedJoinHandle<'_, Result<String, String>>| {
            handle
                .join()
                .unwrap_or_else(|_| Err("a host identity probe panicked".into()))
        };
        (
            join(os),
            join(os_build),
            join(developer),
            join(metal),
            join(ioreg),
        )
    });
    // The Xcode whose tools `xcrun` runs: the selected developer directory's.
    let plist = std::path::Path::new(developer?.trim()).join("../version.plist");
    let plist = std::fs::read_to_string(&plist)
        .map_err(|error| format!("read {}: {error}", plist.display()))?;
    let nonempty = |what: &str, value: String| {
        let value = value.trim().to_string();
        if value.is_empty() {
            Err(format!("{what} printed nothing"))
        } else {
            Ok(value)
        }
    };
    Ok(MetalIdentity {
        device,
        gpu_cores: crate::c3::reference::parse_gpu_cores(&ioreg?)?,
        os: nonempty("sw_vers -productVersion", os?)?,
        os_build: nonempty("sw_vers -buildVersion", os_build?)?,
        xcode_build: crate::c3::reference::parse_xcode_build(&plist)?,
        metal_toolchain: crate::c3::reference::parse_metal_toolchain(&metal?)?,
    })
}
