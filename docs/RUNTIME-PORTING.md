# Porting Metal code to the runtime

This page maps Metal's calls, as Objective-C or `metal-rs` makes them, onto
`metal2vulkan-runtime`, the Metal API on Vulkan
([its README](../crates/metal2vulkan-runtime/README.md)). The tables follow the seam of one
consumer, a paravirtualized GPU that wraps Metal behind its own traits; the **Item** column names
that seam's item, and the section headings name the Metal class. A port from plain Metal code reads
the tables by the **Metal** column.

Each runtime item's rustdoc names its Metal selector and says what differs or is refused; read it
for the details a row leaves out. A row marked *(partial)* covers the item with a difference or a
narrow refusal that the call's rustdoc describes.

## Conventions

| Metal (objc or metal-rs) | Runtime |
|---|---|
| `MTLDevice`, `MTLBuffer`, ... | the same names without the prefix: `Device`, `Buffer`, ... |
| a selector, `newBufferWithLength:options:` | a snake_case method on the receiver: `Device::new_buffer(length, options)` |
| `retain` / `release` | `Clone` / `Drop`; object handles are `Send + Sync` |
| object identity (`==` on `id`) | `PartialEq`, `Eq` and `Hash` by identity |
| `nil` arguments and results | `Option` |
| an `NSError **` out-argument, or a nil result on failure | `Result<_, Error>`; `Error` is typed (the runtime's README, "Errors") |
| descriptor objects (`MTLTextureDescriptor`, `MTLRenderPassDescriptor`, ...) | plain values with public fields; a setter is a field write |
| pipeline descriptors | `metal2vulkan::pipeline` values, from the translator crate |
| encoders, closed by `endEncoding` | owned values; `end_encoding(self)` consumes them |
| completion and scheduled blocks | `Handler`, a boxed `FnOnce(&CommandBuffer) + Send` |
| `MTLSize`, `MTLOrigin`, `MTLRegion` | `Size`, `Origin`, `Region` |
| `MTLResourceOptions`, `MTLTextureUsage`, ... | typed values (`ResourceOptions`; `metal2vulkan::metal::TextureUsage`, ...) |
| Apple-private selectors (SPI) | where the runtime has the call, a method of the same name; `Error::AppleOnly` where Vulkan has no exact form |


## By object

### `Handle`: every object

| Item | Metal | Runtime |
|---|---|---|
| `clone_ref` |  | `Clone` (a retain); `Drop` releases |
| `id` |  | `PartialEq`, `Eq` and `Hash` by identity |

### `Platform`: the object types and their constructors

| Item | Metal | Runtime |
|---|---|---|
| `type Device` | `id<MTLDevice>` | `Device` |
| `type CommandQueue` | `id<MTLCommandQueue>` | `CommandQueue` |
| `type CommandBuffer` | `id<MTLCommandBuffer>` | `CommandBuffer` |
| `type BlitEncoder` | `id<MTLBlitCommandEncoder>` | `BlitCommandEncoder` |
| `type ComputeEncoder` | `id<MTLComputeCommandEncoder>` | `ComputeCommandEncoder` |
| `type RenderEncoder` | `id<MTLRenderCommandEncoder>` | `RenderCommandEncoder` |
| `type Buffer` | `id<MTLBuffer>` | `Buffer` |
| `type Texture` | `id<MTLTexture>` | `Texture` |
| `type Heap` | `id<MTLHeap>` | `Heap` |
| `type SamplerState` | `id<MTLSamplerState>` | `SamplerState` |
| `type DepthStencilState` | `id<MTLDepthStencilState>` | `DepthStencilState` |
| `type Fence` | `id<MTLFence>` | `Fence` |
| `type Event` | `id<MTLEvent>` | `Event` |
| `type SharedEvent` | `id<MTLSharedEvent>` | `SharedEvent` |
| `type SharedEventHandle` | `MTLSharedEventHandle *` | `SharedEventHandle`: in one process only *(partial)* |
| `type SharedEventListener` | `MTLSharedEventListener *` | `SharedEventListener` |
| `type SharedTextureHandle` | `MTLSharedTextureHandle *` | `SharedTextureHandle`: in one process only *(partial)* |
| `type Library` | `id<MTLLibrary>` | `Library` |
| `type Function` | `id<MTLFunction>` | `Function` |
| `type ComputePipeline` | `id<MTLComputePipelineState>` | `ComputePipelineState` |
| `type RenderPipeline` | `id<MTLRenderPipelineState>` | `RenderPipelineState` |
| `type RateMap` | `id<MTLRasterizationRateMap>` | `RasterizationRateMap` |
| `type Icb` | `id<MTLIndirectCommandBuffer>` | `IndirectCommandBuffer` |
| `type IndirectRenderCommand` | `id<MTLIndirectRenderCommand>` | `IndirectRenderCommand` |
| `type IndirectComputeCommand` | `id<MTLIndirectComputeCommand>` | `IndirectComputeCommand` |
| `type TextureDescriptor` | `MTLTextureDescriptor *` | `TextureDescriptor`, a plain value |
| `type RenderPassDescriptor` | `MTLRenderPassDescriptor *` | `RenderPassDescriptor`, a plain value |
| `type RenderPipelineDescriptor` | `MTLRenderPipelineDescriptor *` | `metal2vulkan::pipeline::RenderPipelineDescriptor` |
| `type ComputePipelineDescriptor` | `MTLComputePipelineDescriptor *` | `metal2vulkan::pipeline::ComputePipelineDescriptor` |
| `type TileRenderPipelineDescriptor` | `MTLTileRenderPipelineDescriptor *` | `metal2vulkan::pipeline::TileRenderPipelineDescriptor` |
| `type RateMapDescriptor` | `MTLRasterizationRateMapDescriptor *` | `metal2vulkan::bind::RateMapDescriptor` |
| `type RateMapLayerDescriptor` | `MTLRasterizationRateLayerDescriptor *` | `metal2vulkan::bind::RateMapLayerDescriptor` |
| `new_render_pass_descriptor` | `new` | `RenderPassDescriptor::new` |
| `new_shared_event_listener` | `new` | `SharedEventListener::new` |
| `render_pass_descriptor` | `renderPassDescriptor` | `RenderPassDescriptor::new` |
| `texture_2d_descriptor` | `texture2DDescriptorWithPixelFormat:width:height:mipmapped:` | `TextureDescriptor::texture_2d` |
| `new_rate_map_layer` | `initWithSampleCount:` | `metal2vulkan::bind::RateMapLayerDescriptor::new`; a refused sample count is a typed error |
| `texture_descriptor` |  | `TextureDescriptor`'s fields |
| `rate_map_descriptor` |  | `metal2vulkan::bind::RateMapDescriptor` |
| `open_device` |  | `Device::all`, matched by `Device::registry_id` |

### `BlitCmd`: `MTLBlitCommandEncoder` commands

| Item | Metal | Runtime |
|---|---|---|
| `CopyBufferToTexture` | `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:…` | `BlitCommandEncoder::copy_from_buffer_to_texture` *(partial)* |
| `CopyBufferToTextureOptions` | `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:…` | `BlitCommandEncoder::copy_from_buffer_to_texture` with `BlitOption` *(partial)* |
| `CopyBufferToBuffer` | `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` | `BlitCommandEncoder::copy_from_buffer_to_buffer` |
| `CopyTextureToBuffer` | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:…` | `BlitCommandEncoder::copy_from_texture_to_buffer` *(partial)* |
| `CopyTextureToBufferOptions` | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:…` | `BlitCommandEncoder::copy_from_texture_to_buffer` with `BlitOption` *(partial)* |
| `CopyTextureRegion` | `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:…` | `BlitCommandEncoder::copy_from_texture_to_texture` *(partial)* |
| `CopyTextureSlices` | `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:…` | `BlitCommandEncoder::copy_texture_slices`; `BlitCommandEncoder::copy_texture` is `copyFromTexture:toTexture:` *(partial)* |
| `CopyIndirectCommandBuffer` | `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:` | `BlitCommandEncoder::copy_indirect_command_buffer` |
| `EndEncoding` | `endEncoding` | `BlitCommandEncoder::end_encoding` |
| `FillBuffer` | `fillBuffer:range:value:` | `BlitCommandEncoder::fill_buffer` |
| `FillBufferPattern4` | `fillBuffer:range:pattern4:` (SPI) | `BlitCommandEncoder::fill_buffer_pattern4` |
| `FillTexture` | `fillTexture:level:slice:region:bytes:length:` (SPI) | `BlitCommandEncoder::fill_texture`: every texture type, placed as Metal places it, every format; a multisample texture's samples each take their own bytes of the pattern, as Metal's do; a texture buffer's fill writes its buffer's bytes, the pattern across them, as Metal's does (zeros where Metal's are its stack's, announced, `texture-buffer-fill-leftover`) *(partial)* |
| `FillTextureColor` | `fillTexture:level:slice:region:color:pixelFormat:` (SPI), and `...color:` | `BlitCommandEncoder::fill_texture_color`, packed as Metal's fills pack; a color packed in a format Metal packs none in fills the command buffer's last packed texel (announced, `unpacked-fill-color`) *(partial)* |
| `GenerateMipmaps` | `generateMipmapsForTexture:` | `BlitCommandEncoder::generate_mipmaps` *(partial)* |
| `InvalidateCompressedTexture` | `invalidateCompressedTexture:` (SPI) | `BlitCommandEncoder::invalidate_compressed_texture`: a no-op (Vulkan shows no compression) |
| `InvalidateCompressedTextureSlice` | `invalidateCompressedTexture:slice:level:` (SPI) | `BlitCommandEncoder::invalidate_compressed_texture_slice`: a no-op |
| `OptimizeForCpu` | `optimizeContentsForCPUAccess:` | `BlitCommandEncoder::optimize_contents_for_cpu_access` (a hint; no-op) |
| `OptimizeForCpuSlice` | `optimizeContentsForCPUAccess:slice:level:` | `BlitCommandEncoder::optimize_contents_for_cpu_access_slice_level` (a no-op) |
| `OptimizeForGpu` | `optimizeContentsForGPUAccess:` | `BlitCommandEncoder::optimize_contents_for_gpu_access` (no-op) |
| `OptimizeForGpuSlice` | `optimizeContentsForGPUAccess:slice:level:` | `BlitCommandEncoder::optimize_contents_for_gpu_access_slice_level` (a no-op) |
| `OptimizeIndirectCommandBuffer` | `optimizeIndirectCommandBuffer:withRange:` | `BlitCommandEncoder::optimize_indirect_command_buffer` |
| `ResetCommandsInBuffer` | `resetCommandsInBuffer:withRange:` | `BlitCommandEncoder::reset_commands_in_buffer` |
| `SynchronizeResource` | `synchronizeResource:` | `BlitCommandEncoder::synchronize_resource` (managed storage) |
| `SynchronizeTexture` | `synchronizeTexture:slice:level:` | `BlitCommandEncoder::synchronize_texture` |
| `UpdateFence` | `updateFence:` | `BlitCommandEncoder::update_fence` |
| `WaitForFence` | `waitForFence:` | `BlitCommandEncoder::wait_for_fence` |

### `BlitEncoder`: `MTLBlitCommandEncoder`

| Item | Metal | Runtime |
|---|---|---|
| `encode` |  | one `BlitCommandEncoder` method per `BlitCmd` (above) |

### `CommandQueue`: `MTLCommandQueue`

| Item | Metal | Runtime |
|---|---|---|
| `command_buffer` | `commandBuffer` | `CommandQueue::command_buffer` |
| `set_label` | `setLabel:` | `CommandQueue::set_label` |

### `CommandBuffer`: `MTLCommandBuffer`

| Item | Metal | Runtime |
|---|---|---|
| `add_completed_handler` | `addCompletedHandler:` | `CommandBuffer::add_completed_handler` |
| `blit_encoder` | `blitCommandEncoder` | `CommandBuffer::blit_command_encoder` |
| `commit` | `commit` | `CommandBuffer::commit` |
| `compute_encoder` | `computeCommandEncoder` | `CommandBuffer::compute_command_encoder` |
| `device` | `device` | `CommandBuffer::device` |
| `encode_signal` | `encodeSignalEvent:value:` | `CommandBuffer::encode_signal_event` |
| `encode_signal_when_scheduled` | `encodeSignalEventScheduled:value:` (SPI) | `CommandBuffer::encode_signal_event_scheduled` |
| `encode_wait` | `encodeWaitForEvent:value:` | `CommandBuffer::encode_wait_for_event` |
| `error` | `error` | `CommandBuffer::error` |
| `render_encoder` | `renderCommandEncoderWithDescriptor:` | `CommandBuffer::render_command_encoder` *(partial)* |
| `set_protection_options` | `setProtectionOptions:` (SPI) | `CommandBuffer::set_protection_options`: an encoder made under options other than 0 fails the command buffer from there (`CommandBufferError::ProtectedEncoder`), as Metal's GPU does; 1, 3 and 7 refused where they end Metal's process; changes Metal was not measured at refused `AppleOnly::ProtectedContent` *(partial)* |
| `status` | `status` | `CommandBuffer::status` |
| `wait_until_completed` | `waitUntilCompleted` | `CommandBuffer::wait_until_completed`, or `wait_until_completed_for` with a timeout |
| `wait_until_scheduled` | `waitUntilScheduled` | `CommandBuffer::wait_until_scheduled` |
| `compute_encoder_for_pass` | `computeCommandEncoderWithDescriptor:` | `CommandBuffer::compute_command_encoder_with_dispatch_type`; `compute_command_encoder_with_raw_dispatch_type` for a raw `MTLDispatchType` (every value but 1 serial, as Metal's) |
| `used_resources` | (none: a runtime extension) | `CommandBuffer::used_resources`: the buffers, textures and heaps its ended encoders' commands use, each as what owns its memory, until it completes; what `detach_backing` waits on |
| `ended_encoders` | (none: a runtime extension) | `CommandBuffer::ended_encoders`: each ended encoder's kind, the event waits and signals before it, and what its own commands use; with `commit_failed`'s `FailedEncoders::Set`, what fails the one submission that uses a detached resource |
| `add_submission_handler` | (none: a runtime extension) | `CommandBuffer::add_submission_handler`: a handler run once the command buffer's last batch reached its `VkQueue`, on the device's scheduler thread, which submitted it, before it is `Scheduled`; never for one that fails before |
| `set_submission_check` | (none: a runtime extension) | `CommandBuffer::set_submission_check`: a check of each of Metal's submissions of the command buffer, run once its event waits are met, just before it is submitted; a code it answers fails that submission alone, as Metal fails one that uses a resource detached by then |
| `object_id` | (none: Metal names an object by pointer) | `Buffer::object_id`, `Texture::object_id`, `Heap::object_id`: an `ObjectId` unique for the object's life, never reused (a sparse heap's `gpu_address` is 0) |

### `ComputeCmd`: `MTLComputeCommandEncoder` commands

| Item | Metal | Runtime |
|---|---|---|
| `DispatchThreadgroups` | `dispatchThreadgroups:threadsPerThreadgroup:` | `ComputeCommandEncoder::dispatch_threadgroups` |
| `DispatchThreadgroupsIndirect` | `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:…` | `ComputeCommandEncoder::dispatch_threadgroups_indirect`; at the Metal level `MacOS26_7`, a compute encoder run's first indirect dispatch in an encoder of another dispatch type is refused `Invalid` (`IndirectDispatchOfAnotherType`), where that Metal ends the process |
| `DispatchThreads` | `dispatchThreads:threadsPerThreadgroup:` | `ComputeCommandEncoder::dispatch_threads` |
| `DispatchThreadsIndirect` | `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` (SPI) | `ComputeCommandEncoder::dispatch_threads_indirect` (a host step reads the arguments); refused at `MacOS26_7` as `dispatch_threadgroups_indirect` is |
| `EncodeStartDoWhile` | `encodeStartDoWhile` (SPI) | `ComputeCommandEncoder::encode_start_do_while`, as Metal runs it, its body in its unit's later encoders included, nested to any depth, laid down as 64 copies of the loop's commands (16 inside another loop, fewer past 16384 steps and gated dispatches, at least one), submitted again from where the loop stopped until it ends; a loop without bound runs until the process ends, as Metal's does |
| `EncodeStartElse` | `encodeStartElse` (SPI) | `ComputeCommandEncoder::encode_start_else`, as Metal runs it; with no block open an unbalanced record, dropping its range of the unit's work as Metal's does, the unit's dispatches before its first record included |
| `EncodeStartIf` | `encodeStartIf:offset:comparison:referenceValue:` (SPI) | `ComputeCommandEncoder::encode_start_if`, as Metal runs it, on a counter the runtime's step kernel moves; a live block open where its unit ends fails the command buffer where its commands skip there, as Metal's GPU recovery does (`PredicationOpenAtUnitEnd`); a word past its buffer reads as Metal reads it (the heap's bytes, or zero; zero announced where Metal reads unrelated memory) *(partial)* |
| `EncodeStartWhile` | `encodeStartWhile:offset:comparison:referenceValue:` (SPI) | `ComputeCommandEncoder::encode_start_while`, as Metal runs it, laid down as `encode_start_do_while`'s loops are; its word read as `encode_start_if` reads it *(partial)* |
| `EndEncoding` | `endEncoding` | `ComputeCommandEncoder::end_encoding` |
| `ExecuteCommandsIndirect` | `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` | `ComputeCommandEncoder::execute_commands_in_buffer_indirect`; `_checked` adds an `IndirectCommandCheck` its host step calls on each command first, and the `IndirectCommandCheckNeeds` it looks for |
| `ExecuteCommandsInBuffer` | `executeCommandsInBuffer:withRange:` | `ComputeCommandEncoder::execute_commands_in_buffer`; `_checked` adds an `IndirectCommandCheck` that `commit` (for the CPU's records) or a host step calls on each command first, and the `IndirectCommandCheckNeeds` it looks for, which may rule it out (then it runs on the GPU alone where it can) |
| `CompressedTextureReinterpretationFlush` | `insertCompressedTextureReinterpretationFlush` (SPI) | `ComputeCommandEncoder::insert_compressed_texture_reinterpretation_flush`: a no-op |
| `MemoryBarrierWithResources` | `memoryBarrierWithResources:count:` | `ComputeCommandEncoder::memory_barrier_with_resources` |
| `MemoryBarrierWithScope` | `memoryBarrierWithScope:` | `ComputeCommandEncoder::memory_barrier_with_scope` |
| `SetBufferOffset` | `setBufferOffset:atIndex:` | `ComputeCommandEncoder::set_buffer_offset` |
| `SetBufferOffsetAttributeStride` | `setBufferOffset:attributeStride:atIndex:` | `ComputeCommandEncoder::set_buffer_offset_with_stride` |
| `SetBuffersAttributeStrides` | `setBuffers:offsets:attributeStrides:withRange:` | `ComputeCommandEncoder::set_buffers_with_strides` |
| `SetBuffers` | `setBuffers:offsets:withRange:` | `ComputeCommandEncoder::set_buffers` |
| `SetComputePipelineState` | `setComputePipelineState:` | `ComputeCommandEncoder::set_compute_pipeline_state` |
| `SetImageblockWidth` | `setImageblockWidth:height:` | `ComputeCommandEncoder::set_imageblock_size` |
| `SetSamplerStatesLodClamps` | `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` | `ComputeCommandEncoder::set_sampler_states_with_lod_clamps` |
| `SetSamplerStates` | `setSamplerStates:withRange:` | `ComputeCommandEncoder::set_sampler_states` |
| `SetStageInRegion` | `setStageInRegion:` | `ComputeCommandEncoder::set_stage_in_region` |
| `SetStageInRegionIndirect` | `setStageInRegionWithIndirectBuffer:indirectBufferOffset:` | `ComputeCommandEncoder::set_stage_in_region_indirect` (the region read by a host step, as `dispatch_threads_indirect` reads its arguments) |
| `SetTexture` | `setTexture:atIndex:` | `ComputeCommandEncoder::set_texture` |
| `SetTextures` | `setTextures:withRange:` | `ComputeCommandEncoder::set_textures` |
| `SetThreadgroupMemoryLength` | `setThreadgroupMemoryLength:atIndex:` | `ComputeCommandEncoder::set_threadgroup_memory_length` |
| `UpdateFence` | `updateFence:` | `ComputeCommandEncoder::update_fence` |
| `UseHeaps` | `useHeaps:count:` | `ComputeCommandEncoder::use_heaps` |
| `UseResources` | `useResources:count:usage:` | `ComputeCommandEncoder::use_resources` |
| `WaitForFence` | `waitForFence:` | `ComputeCommandEncoder::wait_for_fence` |

### `ComputeEncoder`: `MTLComputeCommandEncoder`

| Item | Metal | Runtime |
|---|---|---|
| `encode` |  | one `ComputeCommandEncoder` method per `ComputeCmd` (above) |
| `encode_end_do_while` | `encodeEndDoWhile:offset:comparison:referenceValue:` (SPI) | `ComputeCommandEncoder::encode_end_do_while`, answering Metal's `BOOL`; with no block open answering `NO` and dropping its range as Metal's does; its word read as `encode_start_if` reads it *(partial)* |
| `encode_end_if` | `encodeEndIf` (SPI) | `ComputeCommandEncoder::encode_end_if`, answering Metal's `BOOL`; with no block open answering `NO` and dropping its range as Metal's does |
| `encode_end_while` | `encodeEndWhile` (SPI) | `ComputeCommandEncoder::encode_end_while`, answering Metal's `BOOL`; with no block open answering `NO` and dropping its range as Metal's does |

### `TextureDescriptor`: `MTLTextureDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `set_resource_options` | `setResourceOptions:` | field `TextureDescriptor::resource_options` |
| `set_rotation` | `setRotation:` (SPI) | field `TextureDescriptor::rotation`: 0 to 4, with field `framebuffer_only`, for a texture over an IOSurface plane; past 4 or without `framebuffer_only` refused `Invalid` (`TextureProblem::Rotation`, `RotationWithoutFramebufferOnly`), as Metal asserts |
| `set_usage` | `setUsage:` | field `TextureDescriptor::usage` |
| `set_width` | `setWidth:` | field `TextureDescriptor::width` |
| `set_height` | `setHeight:` | field `TextureDescriptor::height` |
| `set_depth` | `setDepth:` | field `TextureDescriptor::depth` |
| `set_texture_type` | `setTextureType:` | field `TextureDescriptor::texture_type` |
| `set_pixel_format` | `setPixelFormat:` | field `TextureDescriptor::pixel_format` (a format the device lacks is refused, typed, at creation) |
| `rotation` | `rotation` (SPI) | field `TextureDescriptor::rotation`; `Texture::rotation` |
| `storage_mode` | `storageMode` | `ResourceOptions::storage_mode` of field `TextureDescriptor::resource_options` |
| `pixel_format` | `pixelFormat` | field `TextureDescriptor::pixel_format` |
| `texture_type` | `textureType` | field `TextureDescriptor::texture_type` |

### `RenderPassDescriptor`: `MTLRenderPassDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `set_sample_positions` | `setSamplePositions:count:` | field `RenderPassDescriptor::sample_positions` (`VK_EXT_sample_locations`; each coordinate rounded as Metal rounds it) *(partial)* |
| `set_pass` |  | `RenderPassDescriptor`'s fields |
| `set_color_attachment` |  | field `RenderPassDescriptor::color_attachments` |
| `set_depth_attachment` |  | field `RenderPassDescriptor::depth_attachment` |
| `set_stencil_attachment` |  | field `RenderPassDescriptor::stencil_attachment` |
| `color_attachment_state` |  | the fields of `RenderPassDescriptor::color_attachments[i]` |
| `depth_attachment_state` |  | the fields of `RenderPassDescriptor::depth_attachment` |
| `stencil_attachment_state` |  | the fields of `RenderPassDescriptor::stencil_attachment` |

### `RenderPipelineDescriptor`: `MTLRenderPipelineDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `set_resource_index` | `setResourceIndex:` (SPI) | `PipelineOptions::placement`: the low 32 bits of the index, below 16384 |

### `ComputePipelineDescriptor`: `MTLComputePipelineDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `set_resource_index` | `setResourceIndex:` (SPI) | `PipelineOptions::placement`: the low 32 bits of the index, below 16384 |

### `GpuDevice`: `MTLDevice`

| Item | Metal | Runtime |
|---|---|---|
| `heap_texture_size_and_align` | `heapTextureSizeAndAlignWithDescriptor:` | `Device::heap_texture_size_and_align`: Metal's answer wherever the runtime models it, the device's images' own footprint where it does not |
| `new_buffer_over_host_memory` | `newBufferWithBytesNoCopy:length:options:deallocator:` | `unsafe Device::new_buffer_with_bytes_no_copy` over a `HostMemory` (`VK_EXT_external_memory_host`; where the device refuses the pointer, the dma-bufs of `HostMemory::with_dma_buf`'s source, `VK_EXT_external_memory_dma_buf`) *(partial)* |
| `new_buffer_of_length` | `newBufferWithLength:options:` | `Device::new_buffer` |
| `new_command_queue` | `newCommandQueue` | `Device::new_command_queue` |
| `new_compute_pipeline_state` | `newComputePipelineStateWithDescriptor:options:reflection:error:` | `Device::new_compute_pipeline_state` |
| `new_compute_pipeline_state_for_function` | `newComputePipelineStateWithFunction:error:` | `Device::new_compute_pipeline_state_with_function` |
| `new_shader_library` | `newLibraryWithSource:options:error:` | `Device::new_library_with_data` over the consumer's prebuilt metallib |
| `new_event` | `newEvent` | `Device::new_event` |
| `new_fence` | `newFence` | `Device::new_fence` |
| `new_library` | `newLibraryWithData:error:` | `Device::new_library_with_data` |
| `new_rate_map` | `newRasterizationRateMapWithDescriptor:` | `Device::new_rasterization_rate_map` |
| `new_render_pipeline_state_from_descriptor` | `newRenderPipelineStateWithDescriptor:options:reflection:error:` | `Device::new_render_pipeline_state` |
| `new_tile_render_pipeline_state` | `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` | `Device::new_tile_render_pipeline_state` |
| `new_shared_event` | `newSharedEvent` | `Device::new_shared_event` |
| `open_shared_event` | `newSharedEventWithHandle:` | `Device::new_shared_event_with_handle`: in one process only *(partial)* |
| `new_shared_texture` | `newSharedTextureWithDescriptor:` | `Device::new_shared_texture`: in one process only *(partial)* |
| `open_shared_texture` | `newSharedTextureWithHandle:withResourceIndex:` (SPI) | `Device::new_shared_texture_with_handle`, the index as an `IdPlacement` |
| `new_texture` | `newTextureWithDescriptor:` | `Device::new_texture` |
| `reserve_resource_indices` | `reserveResourceIndicesForResourceType:indices:indexCount:` (SPI) | `Device::reserve_resource_ids` |
| `caps` |  | `Device::properties()`: Metal's public properties (see `DeviceCaps` below) |
| `new_buffer` | `newBufferWithDescriptor:` (SPI) | `Device::new_buffer_with_descriptor` (`BufferDescriptor` by value, with host contents and an `AddressPlacement`; contents to copy from memory that may change while it is read as `BufferContents::CopyHost` over a `HostBytes`) |
| `new_heap` | `newHeapWithDescriptor:` | `Device::new_heap` (`HeapDescriptor::placement` pins its address range) |
| `reserve_address_pool` | (none: a runtime extension; Metal's own pools of buffers shorter than a page) | `Device::reserve_address_pool` -> `AddressPool`, holding a pin's span; `BufferDescriptor::with_pool_slot(&pool, offset)` places a buffer at the pool's address plus a 256-byte-aligned offset, holding its length rounded up to 256 bytes |
| `new_heap_over_host_memory` | (none: a runtime extension) | `unsafe Device::new_heap_with_bytes_no_copy`: a heap over the caller's page-aligned memory, a `HostMemory` (`VK_EXT_external_memory_host`, or dma-bufs as for a buffer), the heap twin of `new_buffer_with_bytes_no_copy` |
| `rebind_host_memory` | (none: a runtime extension) | `unsafe Buffer::rebind_bytes_no_copy`, `unsafe Heap::rebind_bytes_no_copy`: another range of the same length, a `HostMemory`, under the same objects and GPU addresses, in commit order (`RebindProblem::EncodedUses` while uncommitted work or an open encoder used them; a render encoder uses its attachments' bytes from its open) |
| `new_sampler_state` | `newSamplerStateWithDescriptor:` | `Device::new_sampler_state`, or `Device::new_sampler_state_with_options` (argument-buffer support, placement) |
| `new_depth_stencil_state` | `newDepthStencilStateWithDescriptor:` | `Device::new_depth_stencil_state` |
| `new_indirect_command_buffer` | `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` | `Device::new_indirect_command_buffer` (placement as an option) |
| `new_render_pipeline_state` | `newRenderPipelineStateWithDescriptor:error:` | `Device::new_render_pipeline_state` |

### `SharedEvent`: `MTLSharedEvent`

| Item | Metal | Runtime |
|---|---|---|
| `new_shared_event_handle` | `newSharedEventHandle` | `SharedEvent::new_shared_event_handle` |
| `set_signaled_value` | `setSignaledValue:` | `SharedEvent::set_signaled_value` |
| `wait_until_signaled_value` | `waitUntilSignaledValue:timeoutMS:` | `SharedEvent::wait_until_signaled_value` (timeout) |

### `IndirectCommandBuffer`: `MTLIndirectCommandBuffer`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_resource_id` | `gpuResourceID` | `IndirectCommandBuffer::gpu_resource_id` |
| `compute_command` | `indirectComputeCommandAtIndex:` | `IndirectCommandBuffer::indirect_compute_command_at` |
| `render_command` | `indirectRenderCommandAtIndex:` | `IndirectCommandBuffer::indirect_render_command_at` |
| `size` | `size` | `IndirectCommandBuffer::size` |

### `IcbRenderCmd`: `MTLIndirectRenderCommand` commands

| Item | Metal | Runtime |
|---|---|---|
| `ClearBarrier` | `clearBarrier` | `IndirectRenderCommand::clear_barrier`: refused `Invalid`, as `set_barrier` is |
| `DrawIndexedPatches` | `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:…` | `IndirectRenderCommand::draw_indexed_patches` (the control-point index buffer an `Option`) *(partial)* |
| `DrawIndexedPrimitives` | `drawIndexedPrimitives:indexCount:indexType:indexBuffer:…` | `IndirectRenderCommand::draw_indexed_primitives` |
| `DrawPatches` | `drawPatches:patchStart:patchCount:patchIndexBuffer:…` | `IndirectRenderCommand::draw_patches` |
| `DrawPrimitives` | `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:` | `IndirectRenderCommand::draw_primitives` |
| `Reset` | `reset` | `IndirectRenderCommand::reset` |
| `SetBarrier` | `setBarrier` | `IndirectRenderCommand::set_barrier`: refused `Invalid`, as Metal's validation refuses it on a buffer without mesh draws |
| `SetFragmentBuffer` | `setFragmentBuffer:offset:atIndex:` | `IndirectRenderCommand::set_fragment_buffer` |
| `SetRenderPipelineState` | `setRenderPipelineState:` | `IndirectRenderCommand::set_render_pipeline_state` |
| `SetVertexBuffer` | `setVertexBuffer:offset:atIndex:` | `IndirectRenderCommand::set_vertex_buffer` |
| `SetVertexBufferAttributeStride` | `setVertexBuffer:offset:attributeStride:atIndex:` | `IndirectRenderCommand::set_vertex_buffer_with_stride` |

### `IndirectRenderCommand`: `MTLIndirectRenderCommand`

| Item | Metal | Runtime |
|---|---|---|
| `encode` |  | one `IndirectRenderCommand` method per `IcbRenderCmd` (above) |
| `command_type`, `fragment_buffer_address`, `vertex_attribute_stride`, `vertex_buffer_address`, `has_barrier`, `draw_args`, `draw_indexed_args`, `draw_patches_args`, `draw_indexed_patches_args`, `tessellation_factor_args` | the getters (SPI) | `IndirectRenderCommand::command_type`, `vertex_buffer`, `fragment_buffer`, `has_barrier`, `draw`: each reads its field of the record alone; or `IndirectCommandBuffer::commands`, each render command decoded as an `EncodedRenderCommand`; a patch draw's buffer addresses as Metal answers them, `NIL_PATCH_BUFFER_ADDRESS` for nil, from `IndirectDraw::patch_index_buffer_address` and `control_point_index_buffer_address` |
| `pipeline_state_unique_identifier` | `getPipelineStateUniqueIdentifier` (SPI) | `IndirectRenderCommand::pipeline_unique_identifier` (with `pipeline_state` and `pipeline_id`), or that of the command read back (`IndirectCommandBuffer::commands`) *(partial)* |

### `IcbComputeCmd`: `MTLIndirectComputeCommand` commands

| Item | Metal | Runtime |
|---|---|---|
| `ClearBarrier` | `clearBarrier` | `IndirectComputeCommand::clear_barrier` |
| `ConcurrentDispatchThreadgroups` | `concurrentDispatchThreadgroups:threadsPerThreadgroup:` | `IndirectComputeCommand::concurrent_dispatch_threadgroups` |
| `ConcurrentDispatchThreads` | `concurrentDispatchThreads:threadsPerThreadgroup:` | `IndirectComputeCommand::concurrent_dispatch_threads` |
| `Reset` | `reset` | `IndirectComputeCommand::reset` |
| `SetBarrier` | `setBarrier` | `IndirectComputeCommand::set_barrier` |
| `SetComputePipelineState` | `setComputePipelineState:` | `IndirectComputeCommand::set_compute_pipeline_state` |
| `SetImageblockWidth` | `setImageblockWidth:height:` | `IndirectComputeCommand::set_imageblock_size` |
| `SetKernelBuffer` | `setKernelBuffer:offset:atIndex:` | `IndirectComputeCommand::set_kernel_buffer` |
| `SetKernelBufferAttributeStride` | `setKernelBuffer:offset:attributeStride:atIndex:` | `IndirectComputeCommand::set_kernel_buffer_with_stride` |
| `SetStageInRegion` | `setStageInRegion:` | `IndirectComputeCommand::set_stage_in_region` |
| `SetThreadgroupMemoryLength` | `setThreadgroupMemoryLength:atIndex:` | `IndirectComputeCommand::set_threadgroup_memory_length` |

### `IndirectComputeCommand`: `MTLIndirectComputeCommand`

| Item | Metal | Runtime |
|---|---|---|
| `encode` |  | one `IndirectComputeCommand` method per `IcbComputeCmd` (above) |
| `command_type`, `kernel_attribute_stride`, `kernel_buffer_address`, `stage_in_region`, `threadgroup_memory_length`, `has_barrier`, `dispatch_threadgroups_args`, `dispatch_threads_args`, `image_block_args` | the getters (SPI) | `IndirectComputeCommand::command_type`, `kernel_buffer`, `stage_in_region`, `threadgroup_memory_length`, `has_barrier`, `dispatch`, `imageblock_size`: each reads its field of the record alone; or `IndirectCommandBuffer::commands`, each compute command decoded as an `EncodedComputeCommand` |
| `pipeline_state_unique_identifier` | `getPipelineStateUniqueIdentifier` (SPI) | `IndirectComputeCommand::pipeline_unique_identifier` (with `pipeline_state` and `pipeline_id`), or that of the command read back (`IndirectCommandBuffer::commands`) *(partial)* |

### `RateMap`: `MTLRasterizationRateMap`

| Item | Metal | Runtime |
|---|---|---|
| `copy_parameter_data` | `copyParameterDataToBuffer:offset:` | `RasterizationRateMap::copy_parameter_data_to_buffer` |
| `layer_count` | `layerCount` | `RasterizationRateMap::layer_count` |
| `mutability` | `mutability` (SPI) | `RasterizationRateMap::mutability` |
| `physical_to_screen` | `mapPhysicalToScreenCoordinates:forLayer:` | `RasterizationRateMap::map_physical_to_screen_coordinates`; a map of no layers, and a mutable map not yet reset, are refused (Metal's process ends) |
| `reset` | `resetUsingDescriptor:` (SPI) | `RasterizationRateMap::reset_using_descriptor`; a pass not yet submitted draws by the map as reset |
| `screen_to_physical` | `mapScreenToPhysicalCoordinates:forLayer:` | `RasterizationRateMap::map_screen_to_physical_coordinates`; a map of no layers is refused (Metal's process ends) |
| `parameter_buffer_size_and_align` | `parameterBufferSizeAndAlign` | `RasterizationRateMap::parameter_buffer_size_and_align` |
| `physical_granularity` | `physicalGranularity` | `RasterizationRateMap::physical_granularity` |
| `physical_size` | `physicalSizeForLayer:` | `RasterizationRateMap::physical_size_for_layer` |

### `RateMapDescriptor`: `MTLRasterizationRateMapDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `set_layer` | `setLayer:atIndex:` | field `metal2vulkan::bind::RateMapDescriptor::layers` |
| `set_mutability` | `setMutability:` (SPI) | field `metal2vulkan::bind::RateMapDescriptor::mutability` |
| `layer_count` | `layerCount` | `layers.len()` |
| `layer` |  | `layers[i]` |

### `RateMapLayerDescriptor`: `MTLRasterizationRateLayerDescriptor`

| Item | Metal | Runtime |
|---|---|---|
| `horizontal_sample_storage` | `horizontalSampleStorage` | field `metal2vulkan::bind::RateMapLayerDescriptor::horizontal` |
| `sample_count` | `sampleCount` | `horizontal.len()` and `vertical.len()` |
| `vertical_sample_storage` | `verticalSampleStorage` | field `metal2vulkan::bind::RateMapLayerDescriptor::vertical` |

### `RenderCmd`: `MTLRenderCommandEncoder` commands

| Item | Metal | Runtime |
|---|---|---|
| `DispatchThreadsPerTile` | `dispatchThreadsPerTile:` | `RenderCommandEncoder::dispatch_threads_per_tile` |
| `DispatchThreadsPerTileInRegion` | `dispatchThreadsPerTile:inRegion:` (SPI) | `RenderCommandEncoder::dispatch_threads_per_tile_in_region` with no index: the tiles the region touches |
| `DispatchThreadsPerTileInRegionIndex` | `dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` (SPI) | `RenderCommandEncoder::dispatch_threads_per_tile_in_region` with `Some(index)` |
| `DrawIndexedPatchesIndirect` | `drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:…` | `RenderCommandEncoder::draw_indexed_patches_indirect` |
| `DrawIndexedPatches` | `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:…` | `RenderCommandEncoder::draw_indexed_patches` |
| `DrawIndexedPrimitives` | `drawIndexedPrimitives:indexCount:indexType:indexBuffer:…` | `RenderCommandEncoder::draw_indexed_primitives` (Metal's defaults for the rest) *(partial)* |
| `DrawIndexedPrimitivesInstances` | `drawIndexedPrimitives:indexCount:indexType:indexBuffer:…` | `RenderCommandEncoder::draw_indexed_primitives` *(partial)* |
| `DrawIndexedPrimitivesBase` | `drawIndexedPrimitives:indexCount:indexType:indexBuffer:…` | `RenderCommandEncoder::draw_indexed_primitives` *(partial)* |
| `DrawIndexedPrimitivesIndirect` | `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:…` | `RenderCommandEncoder::draw_indexed_primitives_indirect` *(partial)* |
| `DrawPatchesIndirect` | `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:…` | `RenderCommandEncoder::draw_patches_indirect`, a patch index buffer included |
| `DrawPatches` | `drawPatches:patchStart:patchCount:patchIndexBuffer:…` | `RenderCommandEncoder::draw_patches`, a patch index buffer included |
| `DrawPrimitivesIndirect` | `drawPrimitives:indirectBuffer:indirectBufferOffset:` | `RenderCommandEncoder::draw_primitives_indirect` *(partial)* |
| `DrawPrimitives` | `drawPrimitives:vertexStart:vertexCount:` | `RenderCommandEncoder::draw_primitives` (Metal's defaults for the rest) *(partial)* |
| `DrawPrimitivesInstances` | `drawPrimitives:vertexStart:vertexCount:instanceCount:` | `RenderCommandEncoder::draw_primitives` *(partial)* |
| `DrawPrimitivesBase` | `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:` | `RenderCommandEncoder::draw_primitives` *(partial)* |
| `EndEncoding` | `endEncoding` | `RenderCommandEncoder::end_encoding` |
| `ExecuteCommandsIndirect` | `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` | `RenderCommandEncoder::execute_commands_in_buffer_indirect`: the pass is recorded at a host step (announced once); `_checked` adds an `IndirectCommandCheck` the step calls on each command first, and the `IndirectCommandCheckNeeds` it looks for *(partial)* |
| `ExecuteCommandsInBuffer` | `executeCommandsInBuffer:withRange:` | `RenderCommandEncoder::execute_commands_in_buffer`: the records as they stand at commit, on the GPU alone where the commit can tell every command draws one primitive type under the inherited state (one merged draw for points, lines or triangles), or at a host step where the GPU writes them otherwise; `_checked` adds an `IndirectCommandCheck` that `commit` (for the CPU's records) or that step calls on each command first, and the `IndirectCommandCheckNeeds` it looks for, which may rule it out *(partial)* |
| `MemoryBarrierWithScope` | `memoryBarrierWithScope:afterStages:beforeStages:` | `RenderCommandEncoder::memory_barrier_with_scope` |
| `SetAlphaTestReferenceValue` | `setAlphaTestReferenceValue:` (SPI) | `RenderCommandEncoder::set_alpha_test_reference_value`: the reference a pipeline with the private alpha test compares against |
| `SetBlendColorRed` | `setBlendColorRed:green:blue:alpha:` | `RenderCommandEncoder::set_blend_color` |
| `SetColorResolveTexture` | `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` (SPI) | `RenderCommandEncoder::set_color_resolve_target`: the last target set holds for every draw of the pass *(partial)* |
| `SetColorStoreAction` | `setColorStoreAction:atIndex:` | `RenderCommandEncoder::set_color_store_action` |
| `SetColorStoreActionOptions` | `setColorStoreActionOptions:atIndex:` | `RenderCommandEncoder::set_color_store_action_options`: custom sample positions refused where Metal refuses them |
| `SetCullMode` | `setCullMode:` | `RenderCommandEncoder::set_cull_mode` |
| `SetDepthBias` | `setDepthBias:slopeScale:clamp:` | `RenderCommandEncoder::set_depth_bias` *(partial)* |
| `SetDepthCleared` | `setDepthCleared` (SPI) | `RenderCommandEncoder::set_depth_cleared`: changes nothing, as on Metal |
| `SetDepthResolveTexture` | `setDepthResolveTexture:slice:depthPlane:level:yInvert:` (SPI) | `RenderCommandEncoder::set_depth_resolve_target`, as the color one *(partial)* |
| `SetDepthStencilState` | `setDepthStencilState:` | `RenderCommandEncoder::set_depth_stencil_state` (`None` for nil, Metal's default state) |
| `SetDepthStoreAction` | `setDepthStoreAction:` | `RenderCommandEncoder::set_depth_store_action` |
| `SetFragmentBuffer` | `setFragmentBuffer:offset:atIndex:` | `RenderCommandEncoder::set_fragment_buffer` |
| `SetFragmentBufferOffset` | `setFragmentBufferOffset:atIndex:` | `RenderCommandEncoder::set_fragment_buffer_offset` |
| `SetFragmentBuffers` | `setFragmentBuffers:offsets:withRange:` | `RenderCommandEncoder::set_fragment_buffers` |
| `SetFragmentSamplerStatesLodClamps` | `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:` | `RenderCommandEncoder::set_fragment_sampler_states` with clamps |
| `SetFragmentSamplerStates` | `setFragmentSamplerStates:withRange:` | `RenderCommandEncoder::set_fragment_sampler_states` |
| `SetFragmentTexture` | `setFragmentTexture:atIndex:` | `RenderCommandEncoder::set_fragment_texture` |
| `SetFragmentTextures` | `setFragmentTextures:withRange:` | `RenderCommandEncoder::set_fragment_textures` |
| `SetFrontFacingWinding` | `setFrontFacingWinding:` | `RenderCommandEncoder::set_front_facing_winding` |
| `SetLineWidth` | `setLineWidth:` (SPI) | `RenderCommandEncoder::set_line_width`: 1 accepted; anything else refused `AppleOnly::LineWidth` *(partial)* |
| `SetPointSize` | `setPointSize:` (SPI) | `RenderCommandEncoder::set_point_size`: a no-op, as on Metal |
| `SetProvokingVertexMode` | `setProvokingVertexMode:` (SPI) | `RenderCommandEncoder::set_provoking_vertex_mode`: modes 0 and 1 drawn as Metal draws them; every other mode, which Metal draws a third way, drawn as 1 (patches as 0), each draw Metal's third way draws otherwise announced (`provoking-vertex-unmatched`) *(partial)* |
| `SetRenderPipelineState` | `setRenderPipelineState:` | `RenderCommandEncoder::set_render_pipeline_state` |
| `SetScissorRect` | `setScissorRect:` | `RenderCommandEncoder::set_scissor_rect` |
| `SetScissorRects` | `setScissorRects:count:` | `RenderCommandEncoder::set_scissor_rects` |
| `SetStencilCleared` | `setStencilCleared` (SPI) | `RenderCommandEncoder::set_stencil_cleared`: changes nothing, as on Metal |
| `SetStencilFrontReferenceValue` | `setStencilFrontReferenceValue:backReferenceValue:` | `RenderCommandEncoder::set_stencil_reference_values` |
| `SetStencilResolveTexture` | `setStencilResolveTexture:slice:depthPlane:level:yInvert:` (SPI) | `RenderCommandEncoder::set_stencil_resolve_target`, as the color one *(partial)* |
| `SetStencilStoreAction` | `setStencilStoreAction:` | `RenderCommandEncoder::set_stencil_store_action` |
| `SetStencilStoreActionOptions` | `setStencilStoreActionOptions:` | `RenderCommandEncoder::set_stencil_store_action_options`, as the color one |
| `SetTessellationFactorBuffer` | `setTessellationFactorBuffer:offset:instanceStride:` | `RenderCommandEncoder::set_tessellation_factor_buffer` |
| `SetTessellationFactorScale` | `setTessellationFactorScale:` | `RenderCommandEncoder::set_tessellation_factor_scale` |
| `SetThreadgroupMemoryLength` | `setThreadgroupMemoryLength:offset:atIndex:` | `RenderCommandEncoder::set_threadgroup_memory_length` (a tile function's memory, kept across the pass's tile dispatches) |
| `SetTileBufferOffset` | `setTileBufferOffset:atIndex:` | `RenderCommandEncoder::set_tile_buffer_offset` |
| `SetTileBuffers` | `setTileBuffers:offsets:withRange:` | `RenderCommandEncoder::set_tile_buffers` |
| `SetTileSamplerStatesLodClamps` | `setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:` | `RenderCommandEncoder::set_tile_sampler_states` with clamps |
| `SetTileSamplerStates` | `setTileSamplerStates:withRange:` | `RenderCommandEncoder::set_tile_sampler_states` |
| `SetTileTextures` | `setTileTextures:withRange:` | `RenderCommandEncoder::set_tile_textures` |
| `SetTriangleFillMode` | `setTriangleFillMode:` | `RenderCommandEncoder::set_triangle_fill_mode` |
| `SetVertexAmplificationCount` | `setVertexAmplificationCount:viewMappings:` | `RenderCommandEncoder::set_vertex_amplification_count`: fewer view mappings than the count refused `Invalid` |
| `SetVertexAmplificationMode` | `setVertexAmplificationMode:value:` (SPI) | `RenderCommandEncoder::set_vertex_amplification_mode`: sets the amplification count, the view mappings kept; a count of 0, past 8 or past the mappings refused `AppleOnly::VertexAmplificationMode` *(partial)* |
| `SetVertexBuffer` | `setVertexBuffer:offset:atIndex:` | `RenderCommandEncoder::set_vertex_buffer` |
| `SetVertexBufferOffset` | `setVertexBufferOffset:atIndex:` | `RenderCommandEncoder::set_vertex_buffer_offset` |
| `SetVertexBufferOffsetAttributeStride` | `setVertexBufferOffset:attributeStride:atIndex:` | `RenderCommandEncoder::set_vertex_buffer_offset_with_stride` |
| `SetVertexBuffersAttributeStrides` | `setVertexBuffers:offsets:attributeStrides:withRange:` | `RenderCommandEncoder::set_vertex_buffers_with_strides` |
| `SetVertexBuffers` | `setVertexBuffers:offsets:withRange:` | `RenderCommandEncoder::set_vertex_buffers` |
| `SetVertexSamplerStatesLodClamps` | `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:` | `RenderCommandEncoder::set_vertex_sampler_states` with clamps |
| `SetVertexSamplerStates` | `setVertexSamplerStates:withRange:` | `RenderCommandEncoder::set_vertex_sampler_states` |
| `SetVertexTextures` | `setVertexTextures:withRange:` | `RenderCommandEncoder::set_vertex_textures` |
| `SetViewport` | `setViewport:` | `RenderCommandEncoder::set_viewport` *(partial)* |
| `SetViewportTransformEnabled` | `setViewportTransformEnabled:` (SPI) | `RenderCommandEncoder::set_viewport_transform_enabled`: on accepted; off refused `AppleOnly::ViewportTransformDisabled` *(partial)* |
| `SetViewports` | `setViewports:count:` | `RenderCommandEncoder::set_viewports` *(partial)* |
| `SetVisibilityResultMode` | `setVisibilityResultMode:offset:` | `RenderCommandEncoder::set_visibility_result_mode` (occlusion queries) |
| `UpdateFence` | `updateFence:afterStages:` | `RenderCommandEncoder::update_fence` |
| `UseHeaps` | `useHeaps:count:` | `RenderCommandEncoder::use_heaps`, which always names the stages |
| `UseHeapsStages` | `useHeaps:count:stages:` | `RenderCommandEncoder::use_heaps` |
| `UseResources` | `useResources:count:usage:` | `RenderCommandEncoder::use_resources`, which always names the stages |
| `UseResourcesStages` | `useResources:count:usage:stages:` | `RenderCommandEncoder::use_resources` |
| `WaitForFence` | `waitForFence:beforeStages:` | `RenderCommandEncoder::wait_for_fence` |

### `RenderEncoder`: `MTLRenderCommandEncoder`

| Item | Metal | Runtime |
|---|---|---|
| `encode` |  | one `RenderCommandEncoder` method per `RenderCmd` (above) |
| `tile_height` | `tileHeight` | `RenderCommandEncoder::tile_height` |
| `tile_width` | `tileWidth` | `RenderCommandEncoder::tile_width` |

### `GpuResource`: `MTLResource`

| Item | Metal | Runtime |
|---|---|---|
| `device` | `device` | `Buffer::device`, `Texture::device` |
| `resource_options` | `resourceOptions` | `Buffer::resource_options`, `Texture::resource_options` |
| `set_label` | `setLabel:` | `Buffer::set_label`, `Texture::set_label` |
| `set_purgeable_state` | `setPurgeableState:` | `Buffer::set_purgeable_state`, `Texture::set_purgeable_state`: Metal's answers, the state kept from 16 KiB or for a shareable texture; refused for a heap's resource |
| `storage_mode` | `storageMode` | `Buffer::storage_mode`, `Texture::storage_mode` |

### `Buffer`: `MTLBuffer`

| Item | Metal | Runtime |
|---|---|---|
| `contents` | `contents` | `Buffer::contents` |
| `did_modify_range` | `didModifyRange:` | `Buffer::did_modify_range` |
| `gpu_address` | `gpuAddress` (SPI) | `Buffer::gpu_address`, in the runtime's Metal address space |
| `length` | `length` | `Buffer::length` |
| `new_linear_texture` | `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` (SPI) | `Buffer::new_linear_texture` (an aliasing image, or a shadow image where Vulkan cannot alias) *(partial)* |
| `new_texture` | `newTextureWithDescriptor:offset:bytesPerRow:` | `Buffer::new_texture`, as `new_linear_texture` *(partial)* |

### `Texture`: `MTLTexture`

| Item | Metal | Runtime |
|---|---|---|
| `array_length` | `arrayLength` | `Texture::array_length` |
| `depth` | `depth` | `Texture::depth` |
| `gpu_resource_id` | `gpuResourceID` | `Texture::gpu_resource_id` |
| `height` | `height` | `Texture::height` |
| `is_shareable` | `isShareable` | `Texture::is_shareable` |
| `mipmap_level_count` | `mipmapLevelCount` | `Texture::mipmap_level_count` |
| `new_shared_texture_handle` | `newSharedTextureHandle` | `Texture::new_shared_texture_handle` |
| `new_texture_view` | `newTextureViewWithPixelFormat:` | `Texture::new_texture_view` |
| `new_texture_view_at_index` | `newTextureViewWithPixelFormat:resourceIndex:` (SPI) | `Texture::new_texture_view_with_placement` |
| `new_texture_subview` | `newTextureViewWithPixelFormat:textureType:levels:slices:` | `Texture::new_texture_view_with_levels_slices` |
| `new_swizzled_texture_subview` | `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:` | `Texture::new_texture_view_with_swizzle` |
| `new_swizzled_texture_subview_at_index` | `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:…` | `Texture::new_texture_subview_with_placement` (a swizzle) |
| `new_texture_subview_at_index` | `newTextureViewWithPixelFormat:textureType:levels:slices:…` | `Texture::new_texture_subview_with_placement` (no swizzle) |
| `pixel_format` | `pixelFormat` | `Texture::pixel_format` |
| `protection_options` | `protectionOptions` (SPI) | `Texture::protection_options`: always 0 |
| `replace_region` | `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:` | `Texture::replace_region`; from memory that may change while it is read (a guest's), `Texture::replace_region_with_host_bytes` over a `HostBytes`, which the call copies once through the pointer, with no copy of the caller's first |
| `sample_count` | `sampleCount` | `Texture::sample_count` |
| `texture_type` | `textureType` | `Texture::texture_type` |
| `usage` | `usage` | `Texture::usage` |
| `width` | `width` | `Texture::width` |

### `Heap`: `MTLHeap`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_address` | `gpuAddress` (SPI) | `Heap::gpu_address`: the heap's Metal address range |
| `new_buffer_of_length` | `newBufferWithLength:options:offset:` | `Heap::new_buffer`, or `Heap::new_buffer_at_offset` in a placement heap |
| `new_texture` | `newTextureWithDescriptor:offset:` | `Heap::new_texture_at_offset` |
| `resource_options` | `resourceOptions` | `Heap::resource_options` |
| `set_purgeable_state` | `setPurgeableState:` | `Heap::set_purgeable_state` |
| `storage_mode` | `storageMode` | `Heap::storage_mode` |

### `SamplerState`: `MTLSamplerState`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_resource_id` | `gpuResourceID` | `SamplerState::gpu_resource_id` |

### `DepthStencilState`: `MTLDepthStencilState`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_resource_id` | `gpuResourceID` | `DepthStencilState::gpu_resource_id`: `Invalid(DepthStencilResourceId)` below Apple10, as Metal's validation stops it |

### `ComputePipeline`: `MTLComputePipelineState`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_resource_id` | `gpuResourceID` | `ComputePipelineState::gpu_resource_id` |
| `imageblock_memory_length` | `imageblockMemoryLengthForDimensions:` | `ComputePipelineState::imageblock_memory_length_for_dimensions` |
| `max_total_threads_per_threadgroup` | `maxTotalThreadsPerThreadgroup` | `ComputePipelineState::max_total_threads_per_threadgroup` |
| `static_threadgroup_memory_length` | `staticThreadgroupMemoryLength` | `ComputePipelineState::static_threadgroup_memory_length` |
| `support_indirect_command_buffers` | `supportIndirectCommandBuffers` | `ComputePipelineState::support_indirect_command_buffers` |
| `thread_execution_width` | `threadExecutionWidth` | `ComputePipelineState::thread_execution_width` (32) |

### `RenderPipeline`: `MTLRenderPipelineState`

| Item | Metal | Runtime |
|---|---|---|
| `gpu_resource_id` | `gpuResourceID` | `RenderPipelineState::gpu_resource_id` |
| `imageblock_memory_length` | `imageblockMemoryLengthForDimensions:` | `RenderPipelineState::imageblock_memory_length_for_dimensions` |
| `imageblock_sample_length` | `imageblockSampleLength` | `RenderPipelineState::imageblock_sample_length` |
| `max_total_threads_per_threadgroup` | `maxTotalThreadsPerThreadgroup` | `RenderPipelineState::max_total_threads_per_threadgroup` |
| `support_indirect_command_buffers` | `supportIndirectCommandBuffers` | `RenderPipelineState::support_indirect_command_buffers` |
| `threadgroup_size_matches_tile_size` | `threadgroupSizeMatchesTileSize` | `RenderPipelineState::threadgroup_size_matches_tile_size` |

### `Library`: `MTLLibrary`

| Item | Metal | Runtime |
|---|---|---|
| `function_names` | `functionNames` | `Library::function_names` |
| `new_function` | `newFunctionWithName:` | `Library::new_function_with_name` |

### `DeviceCaps`: `MTLDevice` queries

| Item | Metal | Runtime |
|---|---|---|
| `name` | `name` | `DeviceProperties::name` |
| `registry_id` | `registryID` | `Device::registry_id`, folded from `deviceUUID` |
| `max_threads_per_threadgroup` | `maxThreadsPerThreadgroup` | `DeviceProperties::max_threads_per_threadgroup` |
| `limits.max_threadgroup_memory_length` | `maxThreadgroupMemoryLength` | `DeviceProperties::max_threadgroup_memory_length` |
| `flags.are_raster_order_groups_supported` | `areRasterOrderGroupsSupported` | `DeviceProperties::are_raster_order_groups_supported`: yes where the device enables `fragmentShaderPixelInterlock` *(partial)* |
| `flags.is_depth24_stencil8_pixel_format_supported` | `isDepth24Stencil8PixelFormatSupported` | `DeviceProperties::is_depth24_stencil8_pixel_format_supported`; where no, making the format is refused |
| `sample_counts` | `supportsTextureSampleCount:` | `DeviceProperties::supports_texture_sample_count` |
| `families` | `supportsFamily:` | `Device::supports_family`; `Device::family_blockers` names what blocks a family |
| `heap_buffer_page_private` | `heapBufferSizeAndAlignWithLength:options:` | `Device::heap_buffer_size_and_align` |
| `min_linear_alignment_bgra8` | `minimumLinearTextureAlignmentForPixelFormat:` | `DeviceProperties::minimum_linear_texture_alignment_for_pixel_format` |

## Composed from runtime calls

These items have no single runtime call. The consumer builds them from the calls named.

| Item | How |
|---|---|
| `BlitCmd::CopyTextureRegionOptions` | options 0 is `BlitCommandEncoder::copy_from_texture_to_texture`; a public `BlitOption` goes through a buffer (`copy_from_texture_to_buffer`, then `copy_from_buffer_to_texture`) |
| `RenderCmd::SetDepthClipModeSpi` | the public `RenderCommandEncoder::set_depth_clip_mode` |
| `RenderCmd::SetPrimitiveRestartEnabled` | `RenderCommandEncoder::set_primitive_restart_enabled`: indexed strips restart at the index as Metal takes it for their index type, and only there |
| `GpuDevice::allow_foreign_libraries` | nothing to call: `Device::new_library_with_data` reads a metallib of any Apple platform |
| `GpuDevice::new_buffer_on_surface`, `GpuDevice::new_texture_on_surface` | `Device::new_buffer_with_bytes_no_copy` over the surface's memory, then, for a texture, `Buffer::new_texture_with_iosurface_plane` with the plane's offset, `bytesPerRow` and whether the surface has planes (`Buffer::new_texture` applies the buffer rules, which refuse a surface plane's pitches off 16 that Metal takes) |
| `DeviceCaps::max_vertex_amplification_count` | the largest count `DeviceProperties::supports_vertex_amplification_count` accepts |
| `DeviceCaps::limits.max_rasterization_rate_layer_count` | the largest count `DeviceProperties::supports_rasterization_rate_map_with_layer_count` accepts |
| `DeviceCaps::flags.supports_programmable_sample_positions` | `DeviceProperties::are_programmable_sample_positions_supported` |
| `DeviceCaps::limits.max_viewport_count` | `DeviceProperties::max_viewports` |
| `DeviceCaps::flags.supports_shared_texture_handles` | the runtime's `SharedTextureHandle`, in one process |

## Not the runtime's

These stay with the consumer. The rule: what a plain Metal program needs belongs to the runtime;
what only a paravirtualized device needs belongs to its consumer.

- **IOSurface:** the `Surface` and `IoSurfaces` traits, `Buffer::iosurface`, `Texture::iosurface`,
  `GpuDevice::texture_level_info`, and the `supports_buffer_with_iosurface` flag.
- **Operating-system services:** the `Clock`, `TimerClock`, `Threads`, `Preferences` and `HostVm`
  traits, and `Platform`'s `clock`, `threads`, `vm`, `globals` and `launcher`.
- **Scattered backing**: `Platform::type AddressRangeArray`,
  `new_address_range_array`, and `detach_backing` and `replace_backing` on buffers and heaps.
- **The guest's own numbering:** the runtime answers `resource_index` and `unique_identifier` on
  pipeline states and indirect command buffers by Metal's rules where they were measured, and
  places an object's resource ID where the consumer asks (`IdPlacement`,
  `Device::reserve_resource_ids`); a guest's own numbering of other objects stays the consumer's.
- **Apple's serialized pipeline descriptors:** the three `deserialize_*_pipeline_descriptor`
  items.
- **Apple's descriptions of formats and devices:** `GpuDevice::pixel_format_info` (Apple's flag
  words; `metal2vulkan::metal::format` holds the block facts), `DeviceCaps::gpu_core_count`, and
  the private limits and flags no measurement covers.
