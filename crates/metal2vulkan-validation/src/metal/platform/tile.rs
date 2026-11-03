//! Tile execution, coverage initialization, and threadgroup-memory layout.

use super::function_tables::bind_tile_function_tables;
use super::{
    argument_buffer_resources_resident_on_render, arm_fuel, check_fuel, commit_and_wait,
    encode_argument_buffer_buffers, encode_argument_buffer_function_tables,
    encode_argument_buffer_samplers, encode_argument_buffer_textures, encode_indirect_commands,
    encode_pipeline_templates, load_linked_functions, load_producer_library,
    make_acceleration_structures, make_argument_buffer_buffers, make_argument_buffer_textures,
    make_buffers, make_function, make_indirect_commands, make_render_function_tables,
    make_render_targets, make_samplers, make_texture_arrays, make_textures, mtl_size,
    selected_output, threads_per_threadgroup_total, threads_per_threadgroup_within_limit,
    MetalContext, MetalOutputResources, NativeBuffers, NativePipeline, ShaderFacts, Texture,
};
use crate::case::{AuthoredCase, ImplicitImageblockCoverage};
use crate::gpu::compile;
use crate::literal::LiteralResources;
use crate::metal::MetalRunError;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLColorWriteMask, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLLibrary,
    MTLLoadAction, MTLPipelineOption, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages,
    MTLStoreAction, MTLTexture, MTLTileRenderPipelineDescriptor,
};
use std::collections::BTreeMap;

/// Execute a kernel-based tile function whose implicit imageblock fields alias render-pass
/// color attachments. Metal exposes this ABI through a tile render pipeline, not a compute
/// pipeline; using the latter would compile the function but leave its attachment storage
/// unbound.
pub(super) fn execute_tile(
    context: &MetalContext,
    case: &AuthoredCase,
    source: crate::metal::OracleSource<'_>,
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
    let render_targets = make_render_targets(device, &context.queue, resources, false)?;
    let dispatch = case
        .dispatch
        .as_ref()
        .ok_or_else(|| "tile kernel case has no dispatch".to_string())?;

    let pipeline_descriptor = MTLTileRenderPipelineDescriptor::new();
    unsafe { pipeline_descriptor.setTileFunction(&function) };
    unsafe { pipeline_descriptor.setRasterSampleCount(1) };
    // `threadgroupSizeMatchesTileSize` lets Metal pick the threadgroup for the tile function
    // instead of taking the one the case authored. A tile function with an *explicit*
    // imageblock is sized from that imageblock rather than from the render pass, so Metal
    // caps `maxTotalThreadsPerThreadgroup` far below the authored tile -- 4 instead of 256 for
    // a 16x16 tile whose imageblock is 16 bytes a sample. `dispatchThreadsPerTile` past that
    // cap is dropped with no error and a `Completed` command buffer, so the case reads back
    // its own authored bytes. Pin the threadgroup to what the case dispatches and let Metal
    // size the pipeline from the function.
    pipeline_descriptor.setThreadgroupSizeMatchesTileSize(false);
    pipeline_descriptor
        .setRequiredThreadsPerThreadgroup(mtl_size(dispatch.threads_per_threadgroup));
    // The limit is the required size's threads, the limit Metal answers for such a pipeline it
    // compiles for the first time. Metal keeps a tile pipeline it has compiled and returns it for
    // a later descriptor of the same function that differs only in `requiredThreadsPerThreadgroup`,
    // in this process or a later one, with the first pipeline's limit (measured on an M3 Max,
    // macOS 27.0.1). A case's limit would then follow whichever case of its function ran first:
    // the 8x8 case of a fixture got 16 after its 4x4 case. The descriptor's own limit separates
    // the pipelines, so each threadgroup size gets a pipeline of its own.
    pipeline_descriptor.setMaxTotalThreadsPerThreadgroup(threads_per_threadgroup_total(
        dispatch.threads_per_threadgroup,
        "Metal tile",
    )?);
    if let Some(linked) = &linked.descriptor {
        pipeline_descriptor.setLinkedFunctions(Some(linked));
    }
    let pipeline_attachments = pipeline_descriptor.colorAttachments();
    for (index, texture) in &render_targets {
        let attachment = unsafe { pipeline_attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setPixelFormat(texture.pixelFormat());
    }
    let pipeline = compile::bounded("tile render pipeline creation", || {
        device.newRenderPipelineStateWithTileDescriptor_options_reflection_error(
            &pipeline_descriptor,
            MTLPipelineOption::None,
            None,
        )
    })
    .map_err(|error| {
        super::pipelines::pipeline_failure(device, "create tile render pipeline", &error)
    })?;
    // Refuse a dispatch past the pipeline's limit before any encoder exists: an encoder dropped
    // without `endEncoding` aborts the process.
    threads_per_threadgroup_within_limit(
        dispatch.threads_per_threadgroup,
        pipeline.maxTotalThreadsPerThreadgroup(),
        "Metal tile",
    )?;
    let coverage_pipeline = make_tile_coverage_pipeline(device, &render_targets)?;
    let metal_function_tables =
        make_render_function_tables(&pipeline, MTLRenderStages::Tile, function_tables, &linked)?;

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
    encode_argument_buffer_textures(&function, &buffers, &argument_buffer_textures, interface)?;
    let _argument_buffer_samplers =
        encode_argument_buffer_samplers(device, &function, case, &buffers, interface)?;
    encode_argument_buffer_function_tables(&function, &buffers, &metal_function_tables, interface)?;
    let samplers = make_samplers(device, case)?;
    let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;

    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let pass_attachments = pass.colorAttachments();
    for (index, texture) in &render_targets {
        let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setTexture(Some(&**texture));
        attachment.setLoadAction(MTLLoadAction::Load);
        attachment.setStoreAction(MTLStoreAction::Store);
    }
    let imageblock = case.imageblock.as_ref().ok_or_else(|| {
        "implicit imageblock tile kernel has no imageblock dimensions".to_string()
    })?;
    if imageblock.implicit_coverage != Some(ImplicitImageblockCoverage::FullSingleSample) {
        return Err("implicit imageblock tile kernel requires full_single_sample coverage".into());
    }
    pass.setTileWidth(imageblock.dimensions[0] as usize);
    pass.setTileHeight(imageblock.dimensions[1] as usize);
    // The tile memory a render pass reserves defaults to what its colour attachments need. A
    // tile function with an explicit imageblock can need more -- 24 bytes a sample against the
    // 18 that rgba16+rgba16+r16 attachments imply -- and Metal then refuses the pipeline on
    // the encoder. The pipeline knows the number it needs, so ask it.
    pass.setImageblockSampleLength(pipeline.imageblockSampleLength());
    pass.setRenderTargetWidth(dispatch.grid[0] as usize);
    pass.setRenderTargetHeight(dispatch.grid[1] as usize);
    let threadgroup_layout = tile_threadgroup_memory_layout(case)?;
    if let Some((_, total)) = threadgroup_layout.last() {
        pass.setThreadgroupMemoryLength(*total);
    }

    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
    encoder.setRenderPipelineState(&coverage_pipeline);
    unsafe { encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3) };
    encoder.setRenderPipelineState(&pipeline);
    for (binding, buffer) in &buffers {
        unsafe { encoder.setTileBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize) };
    }
    for (binding, acceleration_structure) in &acceleration_structures {
        unsafe {
            encoder.setTileAccelerationStructure_atBufferIndex(
                Some(&**acceleration_structure),
                *binding as usize,
            )
        };
    }
    bind_tile_function_tables(&encoder, &metal_function_tables);
    argument_buffer_resources_resident_on_render(
        &encoder,
        &argument_buffer_textures,
        &argument_buffer_buffers,
        &indirect_commands.commands,
        MTLRenderStages::Tile,
    );
    for (binding, texture) in &textures {
        unsafe { encoder.setTileTexture_atIndex(Some(&**texture), *binding as usize) };
    }
    for (binding, elements) in &texture_arrays {
        for (element, texture) in elements.iter().enumerate() {
            unsafe {
                encoder.setTileTexture_atIndex(Some(&**texture), *binding as usize + element)
            };
        }
    }
    for (binding, sampler) in &samplers {
        unsafe { encoder.setTileSamplerState_atIndex(Some(&**sampler), *binding as usize) };
    }
    for ((binding, length), (offset, _)) in case
        .threadgroup_memory
        .iter()
        .map(|resource| (resource.binding, resource.length))
        .zip(&threadgroup_layout)
    {
        unsafe {
            encoder.setThreadgroupMemoryLength_offset_atIndex(
                length as usize,
                *offset,
                binding as usize,
            )
        };
    }
    encoder.dispatchThreadsPerTile(mtl_size(dispatch.threads_per_threadgroup));
    encoder.endEncoding();
    arm_fuel(fuel.as_mut(), &buffers);
    commit_and_wait(&command_buffer, "Metal tile dispatch")?;
    check_fuel(fuel.as_ref())?;
    indirect_commands
        .validate_readback()
        .map_err(|error| error.to_string())?;
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

fn make_tile_coverage_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    render_targets: &[(u32, Texture)],
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let source = NSString::from_str(
        r#"
#include <metal_stdlib>
using namespace metal;
struct CoverageVertex { float4 position [[position]]; };
vertex CoverageVertex metal2vulkan_tile_coverage_vertex(uint id [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    return CoverageVertex { float4(positions[id], 0.0, 1.0) };
}
fragment half4 metal2vulkan_tile_coverage_fragment() { return half4(0.0h); }
"#,
    );
    let library = compile::bounded("tile coverage library compilation", || {
        device.newLibraryWithSource_options_error(&source, None)
    })
    .map_err(|error| format!("compile tile coverage pipeline: {error}"))?;
    let vertex = library
        .newFunctionWithName(&NSString::from_str("metal2vulkan_tile_coverage_vertex"))
        .ok_or_else(|| "generated tile coverage library has no vertex function".to_string())?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str("metal2vulkan_tile_coverage_fragment"))
        .ok_or_else(|| "generated tile coverage library has no fragment function".to_string())?;
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setVertexFunction(Some(&vertex));
    descriptor.setFragmentFunction(Some(&fragment));
    let attachments = descriptor.colorAttachments();
    for (index, texture) in render_targets {
        let attachment = unsafe { attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setPixelFormat(texture.pixelFormat());
        attachment.setWriteMask(MTLColorWriteMask::None);
    }
    compile::bounded("tile coverage pipeline creation", || {
        device.newRenderPipelineStateWithDescriptor_error(&descriptor)
    })
    .map_err(|error| format!("create tile coverage pipeline: {error}"))
}

fn tile_threadgroup_memory_layout(case: &AuthoredCase) -> Result<Vec<(usize, usize)>, String> {
    let mut end = 0usize;
    case.threadgroup_memory
        .iter()
        .map(|resource| {
            let offset = end;
            end = end
                .checked_add(resource.length as usize)
                .ok_or_else(|| "tile threadgroup-memory length overflows".to_string())?;
            let result = (offset, end);
            end = end
                .checked_add(15)
                .map(|value| value & !15)
                .ok_or_else(|| "tile threadgroup-memory alignment overflows".to_string())?;
            Ok(result)
        })
        .collect()
}
