//! The error every fallible call returns: one typed variant per case, never text to parse.

use crate::arrays::IndexClass;
use crate::floor::BelowFloor;
use crate::import::ImportHandle;
use crate::memory::{BlockId, MemoryClass};
use ash::vk;
use metal2vulkan::metal::PixelFormat;
use std::{fmt, io};

/// What a failing call returns.
///
/// Metal's counterpart is the `NSError` a creation call fills in, or the nil it returns. Here each
/// case is a variant of its own, so a caller decides by matching, never by reading a message.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// [`Instance::new`](crate::Instance::new) found no Vulkan loader under the platform's
    /// standard names, or [`Instance::with_drivers`](crate::Instance::with_drivers) none at the
    /// consumer's path. Metal has no counterpart: Metal is part of the system.
    Loader(ash::LoadingError),
    /// [`Instance::with_drivers`](crate::Instance::with_drivers): the consumer gave its own
    /// drivers, and the Vulkan loader does not offer `VK_LUNARG_direct_driver_loading`, by which
    /// it would load them. Metal has no counterpart: Metal is part of the system.
    NoDirectDriverLoading,
    /// [`Device::new`](crate::Device::new): the device is below the runtime's floor, with every
    /// item it lacks. Metal has no counterpart: every Metal device runs every program its family
    /// supports.
    BelowFloor(BelowFloor),
    /// A Vulkan call failed: `call` names the Vulkan command and `result` is its error (out of
    /// host or device memory, a lost device, an extension the loader does not offer, ...).
    Vulkan {
        /// The Vulkan command, such as `vkCreateDevice`.
        call: &'static str,
        /// What it returned.
        result: vk::Result,
    },
    /// A call Metal refuses: its validation stops it, or it answers nil.
    Refused(Refusal),
    /// One of the device's or the runtime's fixed limits.
    Limit(Limit),
    /// A block the consumer holds no handle of: never taken, or its last handle gone. Metal has
    /// no counterpart: blocks are the consumer's memory, not Metal objects.
    NoSuchBlock(BlockId),
    /// The runtime could not start one of its threads. Metal has no counterpart.
    Thread(io::Error),
    /// [`Device::new_compute_pipeline_state`](crate::Device::new_compute_pipeline_state): the
    /// translator made no pipeline of the kernel: a typed refusal of what it does not translate,
    /// a descriptor Metal refuses, or a translation past its 20-second budget.
    /// [`Device::new_sampler_state`](crate::Device::new_sampler_state): the translator made no
    /// device sampler of the descriptor. Metal's
    /// counterpart is `newComputePipelineStateWithFunction:error:` failing, which Metal does only
    /// for a descriptor it refuses. [`Device::new`](crate::Device::new) gives it too where the
    /// translator takes no description of the device, which the floor leaves no device to do.
    Translation(metal2vulkan::error::Error),
    /// A Metal feature the runtime does not build yet: a typed known gap.
    Unsupported(Gap),
    /// A call that needs one of the runtime's threads, after
    /// [`Device::quiesce`](crate::Device::quiesce) stopped them. Metal has no counterpart.
    Quiesced,
}

/// A call Metal refuses, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A buffer of no bytes ([`Device::new_buffer`](crate::Device::new_buffer) with a length of
    /// 0). Metal answers nil.
    ///
    /// Basis: metal-measured (an Apple M3 Max: newBufferWithLength:0 answers nil).
    EmptyBuffer,
    /// A buffer longer than [`MAX_BUFFER_LENGTH`](crate::MAX_BUFFER_LENGTH)
    /// ([`Device::new_buffer`](crate::Device::new_buffer)): a shader reaches a buffer argument's
    /// bytes by 32-bit offsets from its bind address, which the translator may read as signed.
    /// Metal answers nil past a device's `maxBufferLength`, which the consumer states no larger
    /// than this.
    ///
    /// Basis: ours (this crate's cap: a buffer's every byte lies within 2^31 bytes of each bind
    /// address in it).
    BufferTooLong {
        /// The buffer's length.
        length: u64,
        /// [`MAX_BUFFER_LENGTH`](crate::MAX_BUFFER_LENGTH).
        max: u64,
    },
    /// A buffer whose range leaves its block ([`Device::new_buffer`](crate::Device::new_buffer)).
    /// Metal answers nil for a buffer placed past the end of its heap.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: a placement heap of 1 MiB answers nil
    /// for a 4096-byte buffer at offset 1,047,552).
    BufferPastBlock {
        /// The buffer's offset in the block.
        offset: u64,
        /// The buffer's length.
        length: u64,
        /// The block's length.
        block_len: u64,
    },
    /// A Managed buffer whose bytes are in a block the host does not map, a Private block
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)).
    ///
    /// Basis: ours (a Managed buffer's bytes are the consumer's, which the host reaches through
    /// the block's mapping; a Private block has none).
    ManagedBlockNotShared {
        /// The block named.
        block: BlockId,
    },
    /// A Managed buffer given no place for its device copy on a device that keeps one, where
    /// [`Device::has_unified_memory`](crate::Device::has_unified_memory) is false
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)).
    ///
    /// Basis: ours (Metal keeps a Managed buffer's second copy on a Mac with a discrete GPU; the
    /// consumer places it, as it places every buffer).
    ManagedCopyMissing,
    /// A Managed buffer given a place for a device copy on a device that keeps none, where
    /// [`Device::has_unified_memory`](crate::Device::has_unified_memory) is true
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)).
    ///
    /// Basis: ours (on unified memory a Managed buffer is its bytes, as Metal keeps one copy of a
    /// Managed buffer on Apple silicon).
    ManagedCopyOnUnified,
    /// A Managed buffer whose device copy is placed in a block that is not Private
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)).
    ///
    /// Basis: ours (shaders reach the device copy in Private memory, device-local where it has
    /// room, as a discrete GPU's Metal keeps the copy in its own memory).
    ManagedCopyNotPrivate {
        /// The block named.
        block: BlockId,
    },
    /// A zero fill whose range leaves its block
    /// ([`Device::zero_block_range`](crate::Device::zero_block_range)), as a buffer placed there
    /// would.
    ///
    /// Basis: ours (the fill stands for the zeroed contents of a buffer placed in the range, and
    /// Metal answers nil for a buffer placed past the end of its heap).
    ZeroRangePastBlock {
        /// The range's offset in the block.
        offset: u64,
        /// The range's length.
        length: u64,
        /// The block's length.
        block_len: u64,
    },
    /// A zero fill of a Shared block ([`Device::zero_block_range`](crate::Device::zero_block_range)).
    ///
    /// Basis: ours (the host maps a Shared block, and its bytes are zeroed through the mapping, as
    /// a zeroed Shared buffer's are; the fill is for the memory the host does not map).
    ZeroRangeShared {
        /// The block named.
        block: BlockId,
    },
    /// A changed range that leaves its buffer ([`Buffer::did_modify_range`](crate::Buffer::did_modify_range)).
    ///
    /// Basis: ours (the copy of such a range would write past the buffer's device copy).
    ModifiedRange {
        /// The range's first byte in the buffer.
        start: u64,
        /// The range's length.
        length: u64,
        /// The buffer's length.
        buffer_length: u64,
    },
    /// A call on a command buffer already committed: encoding, adding a handler, enqueueing or
    /// committing again.
    ///
    /// Basis: metal-documented (Metal's validation asserts "commit an already committed command
    /// buffer", "Completed handler provided after commit call" and "Scheduled handler provided
    /// after commit call").
    Committed,
    /// A call a scheduled handler may not make
    /// ([`CommandBuffer::add_scheduled_handler`](crate::CommandBuffer::add_scheduled_handler)):
    /// [`CommandBuffer::commit`](crate::CommandBuffer::commit), the waits
    /// ([`CommandBuffer::wait_until_scheduled`](crate::CommandBuffer::wait_until_scheduled),
    /// [`CommandBuffer::wait_until_completed`](crate::CommandBuffer::wait_until_completed),
    /// [`SharedEvent::wait_until_signaled_value`](crate::SharedEvent::wait_until_signaled_value)),
    /// the uploads and readbacks ([`Texture::replace_region`](crate::Texture::replace_region),
    /// [`Texture::upload_region`](crate::Texture::upload_region),
    /// [`Texture::get_bytes`](crate::Texture::get_bytes),
    /// [`Device::zero_block_range`](crate::Device::zero_block_range)), and the objects that take
    /// an upload ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer), and
    /// [`Device::new_buffer`](crate::Device::new_buffer) zeroing a buffer in a Private block).
    /// Refused on every device and whatever the runtime holds: a scheduled handler runs on the
    /// submit thread, before its command buffer's submission, and each of these calls may wait
    /// for that thread to submit. Metal runs a scheduled handler on a thread of its own, where
    /// each of these calls runs.
    ///
    /// Basis: ours (the consumer's scheduled handlers run on the submit thread, and the one
    /// consumer's make none of these calls; a guest's scheduled handler runs in the guest).
    InScheduledHandler,
    /// A wait, in a handler or a listener, for what the runtime thread that runs it would have to
    /// do first, so it would never end: on the completion thread (a completed handler or a
    /// listener), [`CommandBuffer::wait_until_completed`](crate::CommandBuffer::wait_until_completed)
    /// for a command buffer not yet completed,
    /// [`CommandBuffer::wait_until_scheduled`](crate::CommandBuffer::wait_until_scheduled) for one
    /// not yet scheduled, and
    /// [`SharedEvent::wait_until_signaled_value`](crate::SharedEvent::wait_until_signaled_value)
    /// for a value not yet reached; in any handler or listener,
    /// [`CommandQueue::command_buffer`](crate::CommandQueue::command_buffer) on a queue whose
    /// maximum count of command buffers are made and not yet completed. A wait already met
    /// answers at once there. The condition is the program's own state: what it committed, and
    /// what has completed. Metal runs handlers on a thread of its own, where the wait would end.
    ///
    /// Basis: ours (no wait of the runtime's hangs by construction).
    WaitOnOwnThread,
    /// An event, a buffer, a texture or a pipeline state of another device in an encode call
    /// ([`CommandBuffer::encode_wait_for_event`](crate::CommandBuffer::encode_wait_for_event),
    /// [`CommandBuffer::encode_signal_event`](crate::CommandBuffer::encode_signal_event), or a
    /// [`ComputeCommandEncoder`](crate::ComputeCommandEncoder) or
    /// [`BlitCommandEncoder`](crate::BlitCommandEncoder) call), or a block of another device
    /// given to [`Device::release_block`](crate::Device::release_block).
    ///
    /// Basis: ours (an event's semaphore, a buffer's block, a texture's image, a pipeline and a
    /// block belong to the `VkDevice` that made them).
    OtherDevice,
    /// A queue that may hold no command buffer
    /// ([`Device::new_command_queue`](crate::Device::new_command_queue) with a count of 0).
    ///
    /// Basis: ours (its first `commandBuffer` call would wait forever).
    NoCommandBuffers,
    /// A second encoder, an encode call, or a commit or enqueue, while an encoder of the command
    /// buffer is open.
    ///
    /// Basis: metal-documented (a command buffer has one active encoder at a time, ended by
    /// `endEncoding` before another is made or the command buffer is committed).
    EncoderOpen,
    /// A dispatch or a draw with no pipeline state set.
    ///
    /// Basis: metal-documented (a compute dispatch needs a compute pipeline state set first, and a
    /// draw a render pipeline state).
    NoPipeline,
    /// A draw of patches with a render pipeline state whose vertex function is not a
    /// post-tessellation one, or a draw of primitives with one whose vertex function is.
    ///
    /// Basis: metal-measured (an M4: plain Metal stops its process over a patch draw with a
    /// pipeline that is not a tessellation one, its validation off; Metal's validation stops a
    /// draw of primitives with a tessellation pipeline).
    PatchDraw,
    /// A draw of patches while the encoder has no tessellation factor buffer
    /// (`setTessellationFactorBuffer:offset:instanceStride:`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal ends the process, SIGSEGV,
    /// at a quad patch draw with no tessellation factor buffer, where the same draw with one set
    /// runs and paints the target, the API validation layer off).
    NoTessellationFactorBuffer,
    /// A buffer argument index past Metal's 31 entries (`setBuffer`, `setBytes`,
    /// `setBufferOffset`, `setThreadgroupMemoryLength`).
    ///
    /// Basis: metal-documented (the buffer argument table has 31 entries per stage).
    BufferIndex {
        /// The index asked for.
        index: u32,
    },
    /// A texture argument index past Metal's 128 entries (`setTexture`).
    ///
    /// Basis: metal-documented (the texture argument table has 128 entries per stage on Apple
    /// GPUs).
    TextureArgumentIndex {
        /// The index asked for.
        index: u32,
    },
    /// A sampler argument index past Metal's 16 entries (`setSamplerState`).
    ///
    /// Basis: metal-documented (the sampler argument table has 16 entries per stage).
    SamplerArgumentIndex {
        /// The index asked for.
        index: u32,
    },
    /// A consumer's answer to
    /// [`Device::new_compute_pipeline_state`](crate::Device::new_compute_pipeline_state)'s
    /// numbered samplers with another count of indices. Metal has no counterpart: Metal numbers
    /// the samplers itself.
    ///
    /// Basis: ours (each numbered sampler takes one index).
    SamplerIds {
        /// The samplers the state numbers.
        numbered: usize,
        /// The indices given.
        given: usize,
    },
    /// `setBufferOffset` at an index where no buffer is bound, or a draw whose vertex descriptor
    /// reads a vertex buffer index where none is bound.
    ///
    /// Basis: ours (there is no buffer whose offset to change, and none whose vertices the draw
    /// reads; Metal's validation stops such a draw).
    NoBufferAtIndex {
        /// The index asked for.
        index: u32,
    },
    /// A threadgroup of no threads, or of more than the pipeline state's
    /// `maxTotalThreadsPerThreadgroup`.
    ///
    /// Basis: metal-documented (a threadgroup's threads total at most the state's
    /// `maxTotalThreadsPerThreadgroup`).
    ThreadgroupSize {
        /// The threadgroup size asked for.
        size: [u32; 3],
        /// The state's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
    /// An indirect dispatch's argument offset that is not a multiple of 4, or that leaves fewer
    /// than 12 bytes (three 32-bit counts) in the buffer.
    ///
    /// Basis: ours (`vkCmdDispatchIndirect` reads three 32-bit counts at a 4-byte aligned offset
    /// inside the buffer).
    IndirectOffset {
        /// The offset asked for.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
    /// A dispatch whose kernel, or a draw whose vertex or fragment function, reaches a buffer
    /// bound where its address, the buffer's `gpuAddress` plus the bind offset, is not a multiple
    /// of the argument's alignment (`air.arg_type_align_size`, at most 16). Plain Metal runs such a dispatch at addresses of
    /// its own: an Apple M3 Max reaches each element at its address rounded down to the smaller of
    /// the element's alignment and 4 (`probes/misaligned_buffer_binds.m` in the conformance
    /// crate).
    ///
    /// Basis: metal-documented (Metal's API validation refuses the bind offset: "must be a
    /// multiple of 4 but was set to 2" for a `uint *` argument bound at offset 2, an Apple M3 Max,
    /// macOS 27.0.1, `probes/vector_access_placement.m` in the conformance crate).
    BufferOffsetAlignment {
        /// The buffer argument index.
        index: u32,
        /// The bind offset.
        offset: u64,
        /// The argument's alignment in bytes.
        alignment: u64,
    },
    /// A grant of an index already granted ([`Device::grant_indices`](crate::Device::grant_indices)).
    /// Metal has no counterpart: Metal chooses its resource IDs itself.
    ///
    /// Basis: ours (each index has one grant, so each resource ID names one object).
    IndicesGranted {
        /// The class of the index.
        class: IndexClass,
        /// The first index of the range that is granted already.
        index: u32,
    },
    /// A texture, a sampler state or a pipeline state's constexpr sampler at an index no live
    /// grant of its class holds ([`Device::new_texture`](crate::Device::new_texture),
    /// [`Device::new_sampler_state`](crate::Device::new_sampler_state),
    /// [`Device::new_compute_pipeline_state`](crate::Device::new_compute_pipeline_state)), or a
    /// release of an index that is not granted
    /// ([`Device::release_indices`](crate::Device::release_indices)). Metal has no counterpart.
    ///
    /// Basis: ours (an object takes an index the consumer was granted, and a grant is returned
    /// once).
    IndicesNotGranted {
        /// The class of the index.
        class: IndexClass,
        /// The first index of the range that is not granted.
        index: u32,
    },
    /// A release of a grant while a texture, a sampler state or a constexpr sampler holds one of
    /// its indices ([`Device::release_indices`](crate::Device::release_indices)). Metal has no
    /// counterpart.
    ///
    /// Basis: ours (an object's index stays granted while the object holds it).
    IndicesInUse {
        /// The class of the index.
        class: IndexClass,
        /// The first index of the grant where an object lives.
        index: u32,
    },
    /// A texture whose usage shaders reach (read or write) made without an index, or one of no
    /// such usage made with one ([`Device::new_texture`](crate::Device::new_texture)). Metal has
    /// no counterpart: Metal gives every texture a resource ID itself.
    ///
    /// Basis: ours (an index is an element of set 1's texture arrays, which only shaders read).
    TextureIndex {
        /// The index given.
        index: Option<u32>,
    },
    /// A texture with a width, height or depth of 0.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor has width of zero", and the same for a height of zero).
    TextureZeroExtent,
    /// A texture with a height other than 1 of a 1D type, or a depth other than 1 of a type that
    /// is not 3D.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "the texture type
    /// MTLTextureType1D requires that height is 1", and the same for a 2D texture of depth 2).
    TextureExtentForType,
    /// A cube texture whose width and height differ.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor width must equal height for cube textures").
    TextureCubeNotSquare,
    /// A texture with an array length of 0, or other than 1 of a type that is not an array.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts
    /// "MTLTextureDescriptor has arrayLength of zero", where an array length of 2 is made).
    TextureArrayLength {
        /// The array length asked for.
        array_length: u32,
    },
    /// A texture with a mipmap level count of 0, more than its full chain's, or more than 1 of a
    /// multisample or 1D type.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "requests 5
    /// mipmap levels, but the dimensions (8, 8, 1) can only support a maxiumum of 4 levels", and
    /// the same for 0).
    TextureLevelCount {
        /// The levels asked for.
        levels: u32,
        /// The most the type and extent take.
        max: u32,
    },
    /// A texture with a sample count of 0, or other than 1 of a type that is not multisample.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "has sampleCount
    /// set but is using a type that does not allow sampleCount" for 0 and for 2 of a 2D texture).
    TextureSampleCount {
        /// The sample count asked for.
        sample_count: u32,
    },
    /// A texture with a swizzle other than each channel from itself and shader-write or
    /// render-target usage.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Texture swizzling
    /// is not compatable with MTLTextureUsageShaderWrite", where a swizzle of read usage alone is
    /// made).
    TextureSwizzleWithWrites,
    /// [`Texture::replace_region`](crate::Texture::replace_region) or
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) of a Private or Memoryless texture.
    ///
    /// Basis: metal-documented (Apple's documentation of `MTLStorageModePrivate`: "The resource is
    /// accessible only to the GPU.").
    TextureBytesPrivate,
    /// [`Texture::replace_region`](crate::Texture::replace_region),
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) or a blit copy
    /// ([`BlitCommandEncoder::copy_from_texture`](crate::BlitCommandEncoder::copy_from_texture))
    /// of a level or slice past the texture's, or of a region that leaves the level or is off the
    /// format's block grid.
    ///
    /// Basis: ours (no texel of the texture is there to copy).
    TextureRegion {
        /// The level asked for.
        level: u32,
        /// The slice asked for.
        slice: u32,
    },
    /// [`Texture::replace_region`](crate::Texture::replace_region) or
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) with rows closer than a row of the
    /// region, images closer than an image of it, or bytes too few for it.
    ///
    /// Basis: ours (the call would read or write past the bytes it is given, or one texel's
    /// bytes over another's).
    TextureBytesLayout {
        /// The bytes from one row's start to the next's.
        bytes_per_row: u64,
        /// The bytes from one image's start to the next's.
        bytes_per_image: u64,
        /// The bytes given.
        len: u64,
    },
    /// [`BlitCommandEncoder::copy_from_buffer`](crate::BlitCommandEncoder::copy_from_buffer) or
    /// [`BlitCommandEncoder::copy_from_texture_to_buffer`](crate::BlitCommandEncoder::copy_from_texture_to_buffer)
    /// whose offset or row stride, for a format of single texels, is no whole number of its
    /// texels; whose rows or images are closer than the region's; or whose region runs past the
    /// buffer's end. A row stride of a region one row high and an image stride of one one image
    /// deep are not read.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1, probe `blit_buffer_copy_rules.m` in
    /// the conformance crate, jobs m3-max-43541 and m3-max-43545: Metal's validation layer stops an
    /// RGBA8Unorm offset of 2, `sourceOffset (2) must be a multiple of 4 bytes`, a row stride of
    /// 18, `sourceBytesPerRow (18) must be a multiple of 4 bytes`, and the same of a copy into a
    /// buffer, a row stride of 12 for rows of 16 bytes, an image stride of 12 for images of 16,
    /// and a region 16 bytes past the buffer's end, `totalBytesUsed(4112) must be <= [sourceBuffer
    /// length](4096)`; it takes an R8Unorm offset of 2, a row stride of 0 for one row, image
    /// strides of no whole number of rows, and an image stride on a 2D texture and on a slice of a
    /// 1D array).
    BufferCopyLayout {
        /// The region's first byte in the buffer.
        offset: u64,
        /// The bytes from one row's start to the next's.
        bytes_per_row: u64,
        /// The bytes from one image's start to the next's.
        bytes_per_image: u64,
        /// The buffer's length.
        len: u64,
    },
    /// [`BlitCommandEncoder::fill_buffer`](crate::BlitCommandEncoder::fill_buffer) of an empty
    /// range, or of one past the buffer's end.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1, probe `blit_fill_rules.m` in the
    /// conformance crate, job m3-max-43573: Metal's validation layer stops both,
    /// `range.length(0) must not be 0` and
    /// `(range.location + range.length)(68) must be <= [buffer length](64)`).
    FillRange {
        /// The range's first byte.
        start: u64,
        /// The byte after its last.
        end: u64,
        /// The buffer's length.
        len: u64,
    },
    /// [`Texture::new_texture_view`](crate::Texture::new_texture_view) of an empty range of levels
    /// or slices, or of one past the texture's.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1, probe `texture_view_rules.m` in the
    /// conformance crate, job m3-max-43677: Metal's validation layer stops each,
    /// `(newLevelRange.location + newLevelRange.length)(4) must be <= [texture mipmapLevelCount](3)`,
    /// `newLevelRange.length must not be 0` and
    /// `(newSliceRange.location + newSliceRange.length)(4) must be <= (3)`).
    TextureViewRange {
        /// The view's first level in the texture.
        level: u32,
        /// Its levels.
        level_count: u32,
        /// The view's first slice in the texture: an array's element, or a cube's face.
        slice: u32,
        /// Its slices.
        slice_count: u32,
    },
    /// [`Texture::new_texture_view`](crate::Texture::new_texture_view) of a pixel format Metal does
    /// not view the texture as: without `PixelFormatView` usage, any but its own, its sRGB sibling
    /// and a combined depth and stencil format's stencil
    /// (`metal2vulkan::metal::format::views_without_usage`); with it, any but those and the
    /// uncompressed colour formats of the same bytes a texel
    /// (`metal2vulkan::metal::format::views_with_usage`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1, probe `texture_view_rules.m` in the
    /// conformance crate, job m3-max-43677: Metal's validation layer stops an `R32Uint` view of an
    /// `RGBA8Unorm` texture of `ShaderRead` usage, `Texture View Validation`, and makes it where
    /// the usage adds `PixelFormatView`; Apple's `makeTextureView` and
    /// `MTLTextureUsage.pixelFormatView` pages state the rest).
    TextureViewFormat {
        /// The texture's pixel format.
        texture: PixelFormat,
        /// The view's.
        view: PixelFormat,
    },
    /// [`BlitCommandEncoder::generate_mipmaps`](crate::BlitCommandEncoder::generate_mipmaps) of a
    /// texture of more than one level whose pixel format is not a colour-renderable, filterable
    /// one: a depth or stencil, compressed, YCbCr or integer format.
    ///
    /// Basis: metal-measured (Apple's `generateMipmaps(for:)` page: mipmap generation works only
    /// for textures with colour-renderable and colour-filterable pixel formats; Metal's API
    /// validation stops the others, measured on an Apple M3 Max and an M4 for
    /// `metal2vulkan-runtime`'s mipmap plan).
    MipmapFormat {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// [`BlitCommandEncoder::copy_from_texture`](crate::BlitCommandEncoder::copy_from_texture)
    /// between textures of two pixel formats, or of two sample counts.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1, probe `blit_copy_format_pairs.m` in
    /// the conformance crate: Metal's validation layer stops every pair of two formats it was
    /// given, sRGB, integer, float, channel order, depth and compressed pairs among them, with
    /// `[sourceTexture pixelFormat](...) must equal [destinationTexture pixelFormat](...)`, and a
    /// source of four samples into one of one with
    /// `[sourceTexture sampleCount](4) must equal [destinationTexture sampleCount](1)`).
    TextureCopyMismatch {
        /// The source texture's pixel format.
        source: PixelFormat,
        /// The destination texture's pixel format.
        destination: PixelFormat,
    },
    /// A Managed texture's bytes the device cannot copy in place
    /// ([`Device::new_managed_texture`](crate::Device::new_managed_texture)): a place for each
    /// level and slice, in order, not given; a buffer the host does not reach; a place whose rows
    /// are not a whole number of texel blocks apart, whose images are not a whole number of rows
    /// apart, or whose first byte in the buffer's block is not a multiple of the texel block's
    /// bytes (of 4 for a depth or stencil format, its rows too); or a place whose level's bytes
    /// leave the buffer. Also [`Texture::upload_region`](crate::Texture::upload_region) on a
    /// texture without named bytes, or of a depth or stencil region whose first byte is not a
    /// multiple of 4.
    ///
    /// Basis: ours (a buffer-to-image copy reads a place only so laid out, and the consumer's
    /// bytes are the place the copy reads).
    ManagedTextureBytes {
        /// The level of the first place refused; the texture's level count where the count of
        /// places is wrong.
        level: u32,
        /// The slice of the first place refused; 0 where the count of places is wrong.
        slice: u32,
    },
    /// A texture over a buffer ([`Buffer::new_texture`](crate::Buffer::new_texture)) whose rows
    /// are not a whole number of texels apart or are closer than a row, whose offset is not a
    /// whole number of texels, or whose texels run past the buffer.
    ///
    /// Basis: ours (a linear image and a copy read rows whole texels apart from a texel's start,
    /// and texels past the buffer would lie in memory it does not hold).
    BufferTextureLayout {
        /// The texture's offset in the buffer.
        offset: u64,
        /// Its rows' pitch.
        bytes_per_row: u64,
        /// The buffer's length.
        len: u64,
    },
    /// A heap texture whose image does not lie in its heap
    /// ([`Device::new_heap_texture`](crate::Device::new_heap_texture)): past the heap's end at
    /// its offset, at an offset in the block that is not a multiple of the image's alignment, or
    /// in a heap range that leaves its block. The size and alignment are the device's image's,
    /// as [`Device::heap_texture_size_and_align`](crate::Device::heap_texture_size_and_align)
    /// answers them, which may exceed what another device answered for the same descriptor.
    ///
    /// Basis: ours (an image bound past its heap's range would take memory the heap does not
    /// hold, and Vulkan binds an image only at its alignment).
    HeapTexturePlace {
        /// The texture's offset in the heap.
        offset: u64,
        /// The bytes the texture takes.
        size: u64,
        /// What its offset in the block is to be a multiple of.
        align: u64,
        /// The heap's bytes.
        heap_len: u64,
    },
    /// A copy between buffers
    /// ([`BlitCommandEncoder::copy_from_buffer_to_buffer`](crate::BlitCommandEncoder::copy_from_buffer_to_buffer))
    /// whose range, `size` bytes from `offset`, leaves the source or the destination.
    ///
    /// Basis: metal-documented (Metal's validation stops a copy past either buffer's length).
    BufferCopyRange {
        /// The range's offset in the buffer it leaves.
        offset: u64,
        /// The bytes copied.
        size: u64,
        /// That buffer's length.
        len: u64,
    },
    /// A copy between buffers whose source offset, destination offset or size is not a multiple
    /// of 4.
    ///
    /// Basis: metal-documented (`copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`:
    /// on macOS both offsets and the size are multiples of 4).
    BufferCopyUnaligned {
        /// The offset in the source.
        source_offset: u64,
        /// The offset in the destination.
        destination_offset: u64,
        /// The bytes copied.
        size: u64,
    },
    /// A copy between buffers whose source and destination share bytes: one buffer's
    /// overlapping ranges, or two buffers that overlap in one block.
    ///
    /// Basis: ours (Metal states no result for such a copy, and `vkCmdCopyBuffer` reads no byte
    /// it also writes, `VUID-vkCmdCopyBuffer-pRegions-00117`; the runtime does not emulate one).
    BufferCopyOverlap {
        /// The offset in the source.
        source_offset: u64,
        /// The offset in the destination.
        destination_offset: u64,
        /// The bytes copied.
        size: u64,
    },
    /// A render pass attachment
    /// ([`CommandBuffer::render_command_encoder`](crate::CommandBuffer::render_command_encoder))
    /// whose texture was made without `RenderTarget` usage.
    ///
    /// Basis: metal-documented (a texture a render pass renders into needs
    /// `MTLTextureUsageRenderTarget`; Metal's validation stops the descriptor).
    AttachmentUsage {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// A render pass attachment of a level or slice past its texture's.
    ///
    /// Basis: ours (no texel of the texture is there to render into).
    AttachmentPlace {
        /// The level asked for.
        level: u32,
        /// The slice asked for.
        slice: u32,
    },
    /// A render pass color attachment of a depth or stencil pixel format, a depth attachment of a
    /// format without depth, or a stencil attachment of a format without stencil.
    ///
    /// Basis: metal-documented (a color attachment takes a color-renderable format, a depth
    /// attachment a format with depth and a stencil attachment a format with stencil; Metal's
    /// validation stops the descriptor).
    AttachmentFormat {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// A draw whose vertex descriptor reads a vertex buffer at an offset at or past the buffer's
    /// length.
    ///
    /// Basis: metal-documented (a buffer's offset lies inside it; Metal's validation stops the
    /// bind).
    VertexBufferOffset {
        /// The vertex buffer index.
        index: u32,
        /// The offset bound.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
    /// An indexed draw whose index buffer offset is not a multiple of the index size, or whose
    /// indices leave the buffer.
    ///
    /// Basis: metal-documented (the offset is a multiple of the index size and the indices lie in
    /// the buffer; Metal's validation stops the draw).
    IndexBufferOffset {
        /// The offset asked for.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
    /// A tessellation factor buffer whose factors, at its address plus the offset, start at an
    /// odd byte: a factor is a `half`. An offset 2 past a multiple of 4 is read at that byte, as
    /// plain Metal reads it (Metal probe case `factor_offset_2` on an M4).
    ///
    /// Basis: metal-documented (`setTessellationFactorBuffer:offset:instanceStride:`'s offset is a
    /// multiple of 4; Metal's validation stops a patch draw with any other).
    TessellationFactorOffset {
        /// The offset asked for.
        offset: u64,
    },
    /// An indirect dispatch's arguments, or an indexed draw's indices, at a byte of the
    /// allocation the buffer's block lies in (its device copy's, for a Managed buffer with one)
    /// that is not a multiple of 4, or of the index size.
    ///
    /// Metal's offset rules keep both at such a multiple of the buffer's start, so only a buffer
    /// that starts off it meets this: a no-copy buffer
    /// (`newBufferWithBytesNoCopy:length:options:deallocator:`) over a pointer that is not a
    /// multiple of 4 (indirect arguments) or of the index size (indexed draws), which documented
    /// Metal does not allow, its pointer being page-aligned. The consumer places every other
    /// buffer at an aligned byte. Such a buffer is still made, and every other use of it runs:
    /// only the dispatch or draw that reads its arguments or indices is refused.
    ///
    /// Basis: ours (`vkCmdDispatchIndirect` reads its arguments at a multiple of 4 in the
    /// `VkBuffer`, and `vkCmdBindIndexBuffer` takes indices at a multiple of their size in the
    /// memory; a Metal buffer starts at an aligned address, so Metal's offset rule alone keeps
    /// both aligned there).
    ArgumentPlace {
        /// The byte of the allocation the arguments or indices start at: their block's offset
        /// there ([`BlockInfo::offset`](crate::BlockInfo::offset)) plus their place in the
        /// block.
        place: u64,
        /// The multiple Vulkan reads them at: 4, or the index size.
        alignment: u64,
    },
    /// A viewport with a field that is NaN or infinite (but its origin's x, which draws nothing).
    ///
    /// Basis: metal-measured (an Apple M4, macOS 27.0.1: Metal's validation stops a NaN or
    /// infinite origin; the other fields are taken by the same rule).
    ViewportNotFinite,
    /// A draw under a depth-stencil state that tests the stencil, in a pass without a stencil
    /// attachment.
    ///
    /// Basis: metal-measured (an Apple M4, macOS 27.0.1: Metal's validation stops the draw; plain
    /// Metal tests it against a hidden stencil of 0 that the runtime does not keep).
    StencilAttachment,
}

/// A Metal feature the runtime does not build yet, with what it was asked. Each is refused until a
/// replay stream needs it, never run in a near form. Metal has no counterpart: Metal runs each.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// A compute pipeline state for indirect command buffers
    /// (`supportIndirectCommandBuffers`): the runtime has no indirect command buffers yet.
    ///
    /// Basis: ours (indirect command buffers are not built).
    IndirectCommandBuffers,
    /// A kernel that reads something the compute encoder does not supply yet: a function table,
    /// a `[[stage_in]]` fetch, a parameter the runtime does not fill, or a specialization the
    /// runtime does not set. Buffers, `setBytes` data, textures and sampler states by their
    /// resource IDs, constexpr samplers by their indices, an integer write's channel widths,
    /// threadgroup memory lengths, the dispatch's grid, the stage-in region and a pinned subgroup
    /// size are supplied.
    ///
    /// Basis: ours (the encoder supplies only those).
    KernelInput,
    /// A dispatch whose threadgroup is not whole SIMD groups of a kernel whose tier runs each SIMD
    /// group as one subgroup of `width` threads (a pinned subgroup of 32, or the device's own
    /// 32), where the state has no emulated-SIMD-group module to run it on: Metal runs the last
    /// SIMD group of such a threadgroup partly filled. A kernel the consumer compiled
    /// (`StageSource::Compiled`) keeps the variant it was compiled under, and the translator may
    /// refuse a kernel's emulated variant (where its static threadgroup memory and one SIMD
    /// group's scratch are past the device's). Refused at the dispatch; the state's other sizes
    /// run.
    ///
    /// Basis: ours (no emulated-SIMD-group module).
    PartialSimdgroups {
        /// The threadgroup size asked for.
        threads_per_threadgroup: [u32; 3],
        /// The threads every threadgroup must be a multiple of.
        width: u32,
    },
    /// `dispatchThreads` with a grid that is not a whole number of threadgroups on some axis, of
    /// a kernel that observes its threadgroup partition and whose partial-threadgroups variant the
    /// translator refused, or at a threadgroup that is not whole subgroups of a kernel whose tier
    /// runs only whole ones: Metal runs the last threadgroup of such an axis with the threads
    /// left. The translator refuses the variant of a kernel whose SIMD groups it emulates
    /// (SIMD-group operations where the device's compute subgroups are neither 32 lanes nor 64 or
    /// 128 split into 32-lane clusters, or a lockstep pair in divergent flow on a device without
    /// `VK_KHR_shader_maximal_reconvergence`), and of one whose barriers in a loop need a vote
    /// word its workgroup memory cannot hold. Refused at the dispatch; the state's whole grids
    /// run.
    ///
    /// Basis: ours (the translator's partial-threadgroups variant does not take the emulated
    /// SIMD-group tier, nor a vote word past the workgroup memory).
    PartialThreadgroups {
        /// The grid asked for.
        threads_per_grid: [u32; 3],
        /// The threadgroup size asked for.
        threads_per_threadgroup: [u32; 3],
    },
    /// A dispatch of a kernel that reads the stage-in region (`[[grid_origin]]`, `[[grid_size]]`)
    /// on an encoder that never set one
    /// ([`ComputeCommandEncoder::set_stage_in_region`](crate::ComputeCommandEncoder::set_stage_in_region)):
    /// what such a kernel reads on Metal is not established, and is not invented.
    ///
    /// Basis: ours (no replay stream dispatches such a kernel without setting a region).
    NoStageInRegion,
    /// A texture of `Memoryless` storage, which only render passes use.
    ///
    /// Basis: ours (render passes are not built).
    MemorylessTexture,
    /// A texture buffer (`MTLTextureTypeTextureBuffer`).
    ///
    /// Basis: ours (no texel-buffer array; no replay stream uses one).
    TextureBuffer,
    /// A cube array texture (`MTLTextureTypeCubeArray`), and so a view of one: its image view
    /// needs Vulkan's `imageCubeArray`, which the floor does not require.
    ///
    /// Basis: ours (no replay stream makes a cube array texture or view; built when one does).
    CubeArrayTexture,
    /// A texture of a pixel format the runtime makes no image of on this device for the
    /// descriptor's usage: the device's format query lacks the format or one of the usages, or
    /// the format's texels need a conversion in the shader that the translator's identity domains
    /// do not make (an extended-range, bfloat, YCbCr, decoded or copies-only format, and 24-bit
    /// depth held as 32-bit float depth).
    ///
    /// Basis: ours (the device's format query, or the identity domains' shaders read every
    /// texture through its view alone).
    TextureFormat {
        /// The pixel format asked for.
        pixel_format: PixelFormat,
    },
    /// [`BlitCommandEncoder::generate_mipmaps`](crate::BlitCommandEncoder::generate_mipmaps) of a
    /// texture whose image the device's blits do not filter: its format query lacks blits from
    /// or to the image's format, or its linear filtering, or the image holds the pixel format
    /// through a conversion (an extended-range or YCbCr format), or the texture is a view in
    /// another format than its image's, whose blit would filter the image's.
    ///
    /// Basis: ours (the device's format query; the levels are the device's blit chain alone).
    MipmapFilter {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// A sampler state whose minification or magnification filter is cubic
    /// ([`Device::new_sampler_state`](crate::Device::new_sampler_state)).
    ///
    /// Basis: ours (the translator's identity domains do not filter bicubically, and Vulkan's
    /// cubic filter is not Metal's).
    CubicSampler,
    /// A sampler state bound with LOD clamps its samples read otherwise through
    /// ([`ComputeCommandEncoder::set_sampler_state_with_lod_clamps`](crate::ComputeCommandEncoder::set_sampler_state_with_lod_clamps)
    /// and the render encoder's vertex and fragment forms): a bound that differs from the state's
    /// own once both are held to the levels a sample can read
    /// ([`lod_clamps_read_alike`](metal2vulkan::metal::sampler::lod_clamps_read_alike)).
    ///
    /// Basis: ours (no replay stream binds such clamps: every bind-time clamp the streams make
    /// reads as its state's own).
    LodClamps,
    /// [`Texture::replace_region`](crate::Texture::replace_region),
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) or a blit copy between a buffer and a
    /// texture
    /// ([`BlitCommandEncoder::copy_from_buffer`](crate::BlitCommandEncoder::copy_from_buffer)) of a
    /// texture whose image is no plain copy of Metal's bytes in one aspect: a format of depth and
    /// stencil both, or a multisample texture.
    ///
    /// Basis: ours (no copy between such an image and bytes is built).
    TextureBytes {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// [`Texture::replace_region`](crate::Texture::replace_region) or
    /// [`BlitCommandEncoder::copy_from_buffer`](crate::BlitCommandEncoder::copy_from_buffer) into a
    /// 32-bit float depth (`Depth32Float`). Metal keeps any `float` there; Vulkan copies no depth
    /// outside `[0, 1]` into a depth aspect without `VK_EXT_depth_range_unrestricted`, which the
    /// floor does not require (the Intel devices measured lack it), and no recorded stream uploads
    /// one.
    ///
    /// Basis: ours (no upload of a 32-bit float depth is built).
    FloatDepthUpload {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// [`BlitCommandEncoder::copy_from_texture`](crate::BlitCommandEncoder::copy_from_texture)
    /// that no single `vkCmdCopyImage` in the floor makes: a multisample texture; a format of
    /// depth and stencil both; one pixel format held in two Vulkan formats, which the two
    /// textures' usages chose; a 1D texture with one of another type; or a region of one
    /// texture's level and slice into the same level and slice.
    ///
    /// Basis: ours (Vulkan 1.3 copies none of these with one command; each is built when a replay
    /// stream makes one).
    TextureCopy {
        /// The source texture's pixel format.
        source: PixelFormat,
        /// The destination texture's pixel format.
        destination: PixelFormat,
    },
    /// A texture over a buffer ([`Buffer::new_texture`](crate::Buffer::new_texture)) the runtime
    /// does not make: other than 2D of one level, sample and slice; of a format of other than
    /// one color texel an element; or whose first texel in the bytes shaders reach is not a whole
    /// number of texels in, which no copy addresses.
    ///
    /// Basis: ours (the in-place linear image and the copy form take one 2D color image).
    TextureOverBuffer {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// A heap texture the runtime does not place
    /// ([`Device::new_heap_texture`](crate::Device::new_heap_texture) and
    /// [`Device::heap_texture_size_and_align`](crate::Device::heap_texture_size_and_align)): in a
    /// Shared block, or one whose image the device requires an allocation of its own for, or
    /// that does not take the Private memory type. No recorded stream makes a Shared heap, and
    /// every optimal image measured on the lab's devices takes the Private type.
    ///
    /// Basis: ours (heap textures are built for Private heaps of placeable images).
    HeapTexture {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// [`BlitCommandEncoder::copy_from_buffer`](crate::BlitCommandEncoder::copy_from_buffer) or
    /// [`BlitCommandEncoder::copy_from_texture_to_buffer`](crate::BlitCommandEncoder::copy_from_texture_to_buffer)
    /// whose bytes no copy command in the floor addresses, though Metal takes them: an image's
    /// offset in the `VkBuffer` shaders reach that is no multiple of a texel block's bytes (of 4
    /// for a depth or stencil aspect: a 16-bit depth at an offset of 2), or a row stride of no
    /// whole number of a compressed format's blocks.
    ///
    /// Basis: ours (Vulkan 1.3 places a copy's rows in a buffer by a count of texels from an
    /// aligned offset; each is built when a replay stream makes one).
    BufferCopyAlignment {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
        /// The region's first byte in the buffer.
        offset: u64,
        /// The bytes from one row's start to the next's.
        bytes_per_row: u64,
        /// The bytes from one image's start to the next's.
        bytes_per_image: u64,
    },
    /// [`BlitCommandEncoder::fill_buffer`](crate::BlitCommandEncoder::fill_buffer) of a range
    /// that does not start and end on whole 4-byte words of the bytes shaders reach, which Metal
    /// fills byte by byte (probe `blit_fill_rules.m` in the conformance crate, an Apple M3 Max,
    /// macOS 27.0.1, job m3-max-43571).
    ///
    /// Basis: ours (`vkCmdFillBuffer` writes whole words; a byte fill is built when a replay
    /// stream makes one).
    FillAlignment {
        /// The range's first byte.
        start: u64,
        /// The byte after its last.
        end: u64,
    },
    /// A render pipeline state of something the runtime does not build yet: what `kind` names.
    ///
    /// Basis: ours (the replay streams' render pipelines draw into colour, depth and stencil
    /// attachments of one sample, with no tessellation and no framebuffer fetch; each other kind
    /// is built when a replay stream makes one).
    RenderPipeline {
        /// What the pipeline asks for.
        kind: RenderPipelineKind,
    },
    /// A [`Fence`](crate::Fence) updated or waited on by an encoder of a queue other than the first
    /// that used it: the runtime orders the encoders of one queue, so a fence would order nothing
    /// across two. Metal's fence orders the encoders of one queue; `MTLEvent` orders queues.
    ///
    /// Basis: ours (a fence is built for one queue; the replay streams use each on one queue).
    FenceOtherQueue,
    /// A draw or dispatch whose function declares `access::read_write` at texture slot `slot` and
    /// reads it, where the texture bound there has a swizzle other than each channel's own (a view made
    /// with one, [`Texture::new_texture_view_with_swizzle`](crate::Texture::new_texture_view_with_swizzle)):
    /// such a slot reads through the texture's storage element, which maps each channel to
    /// itself as Vulkan requires of a storage image
    /// (VUID-VkWriteDescriptorSet-descriptorType-00336), so its `read` would answer the texel
    /// unswizzled, where Metal reads it through the swizzle. Found when the pipeline state or a
    /// binding changes, for the read-write slots a function reads alone
    /// (`TextureSlot::storage` with a read). A write is not swizzled: Apple defines the swizzle
    /// for reads and samples.
    ///
    /// Basis: ours (built when a replay stream counts one).
    SwizzledReadWrite {
        /// The function's texture slot.
        slot: u32,
    },
    /// [`Texture::new_texture_view`](crate::Texture::new_texture_view) of a kind Metal makes and
    /// the runtime does not build: what `kind` names.
    ///
    /// Basis: ours (the replay streams' views that compute binds are same-format views of every
    /// slice, 2D views of one slice of a 2D array, and level subranges, swizzled or not; each
    /// other kind is built when a replay stream binds one).
    TextureView {
        /// What the view changes that is not built.
        kind: TextureViewKind,
    },
    /// A render pass of something the runtime does not build yet
    /// ([`CommandBuffer::render_command_encoder`](crate::CommandBuffer::render_command_encoder)):
    /// what `kind` names.
    ///
    /// Basis: ours (the replay streams' passes render into one to two color attachments and a
    /// stencil or a depth attachment, each a level of a 2D texture of one sample, with no resolve
    /// and a clear depth of 1, none naming two textures for its depth and stencil; each other kind
    /// is built when a replay stream makes one; `DepthStencilViews` and `ClearDepth` are what
    /// Vulkan's rendering under the floor takes no form of).
    RenderPass {
        /// What the pass asks for.
        kind: RenderPassKind,
    },
    /// A draw or render encoder state the runtime does not build yet
    /// ([`RenderCommandEncoder`](crate::RenderCommandEncoder)): what `kind` names.
    ///
    /// Basis: ours (the replay streams draw lines and triangles through one viewport Vulkan's is,
    /// with pipeline states of their pass's formats, under depth biases with no clamp, filled and
    /// clipped; each other kind is built when a replay stream draws one), and device for
    /// [`DrawKind::DividedBaseInstance`] (a device that does not report
    /// `supportsNonZeroFirstInstance`).
    Draw {
        /// What the draw asks for.
        kind: DrawKind,
    },
}

/// What a render pipeline state asks that the runtime does not build ([`Gap::RenderPipeline`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPipelineKind {
    /// A depth attachment format and a stencil attachment format that differ
    /// (`depthAttachmentPixelFormat` and `stencilAttachmentPixelFormat`, as of separate depth and
    /// stencil textures): Vulkan's pipeline takes one format for both
    /// (VUID-VkGraphicsPipelineCreateInfo-renderPass-06589), as its rendering takes one view for
    /// both ([`RenderPassKind::DepthStencilViews`]).
    DepthStencilFormats,
    /// A `rasterSampleCount` that is not a power of two from 1 to 64: Vulkan's rasterization
    /// takes one of its sample count bits (`VkSampleCountFlagBits`).
    SampleCount,
    /// Vertex amplification (`maxVertexAmplificationCount` above 1).
    VertexAmplification,
    /// A post-tessellation vertex function's tessellation state that the runtime does not build:
    /// a per-instance factor step function (`MTLTessellationFactorStepFunctionPerInstance` or
    /// `PerPatchAndPerInstance`) or an enabled factor scale (`tessellationFactorScaleEnabled`),
    /// which no replay stream's states use. The translator refuses a control-point index type
    /// (`tessellationControlPointIndexType`) itself. A descriptor's tessellation state with a vertex
    /// function that is not post-tessellation is ignored, as Metal ignores it.
    Tessellation,
    /// Stages the translator synthesizes beside the functions' but a tessellation's: geometry
    /// stages, lines drawn as bands of triangles, a rate map's fan-out, or mesh records.
    SynthesizedStages,
    /// A fragment function that reads its colour attachments (framebuffer fetch).
    FramebufferFetch,
    /// A fragment function that reaches the render pass's imageblock.
    Imageblock,
    /// A fragment function's stencil output applied by the shader.
    ShaderStencil,
    /// `alphaToOneEnabled`, which needs a feature the floor does not require.
    AlphaToOne,
    /// A function that reads constant data the binder supplies.
    ConstantData,
    /// A specialization the draw would set, beside the sample and amplification counts.
    Specialization,
    /// Dynamic state the translator asks for beyond what Vulkan 1.3 leaves dynamic.
    DynamicState,
    /// A Vulkan feature or extension the translator's pipeline needs (its `Requirements`) that
    /// device creation does not enable.
    DeviceFeature,
    /// A function that reads what a draw does not supply yet: a function table, a table the
    /// binder builds, a parameter block past the push range, a push field or bind-time fact the
    /// render encoder does not fill, or an unnormalized constexpr sampler.
    DrawInput,
}

/// What a draw asks that the runtime does not build ([`Gap::Draw`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawKind {
    /// A draw of points (`MTLPrimitiveTypePoint`).
    Points,
    /// A depth bias clamp other than 0 (`setDepthBias:slopeScale:clamp:`'s `clamp`), which no
    /// replay stream sets.
    DepthBiasClamp,
    /// A triangle fill mode but `Fill` (`setTriangleFillMode:`), which no replay stream sets.
    FillMode,
    /// The depth clip mode `Clamp` (`setDepthClipMode:`), which no replay stream sets.
    DepthClamp,
    /// More than one viewport or scissor rectangle, or none (`setViewports:count:`,
    /// `setScissorRects:count:`), which no replay stream sets.
    Viewports,
    /// A viewport Vulkan's is not: a negative width (Metal mirrors the draw), a depth range past
    /// `[0, 1]`, or bounds past the device's viewport limits.
    Viewport,
    /// A pipeline state whose colour or stencil formats are not its pass's attachments', or
    /// whose `rasterSampleCount` is not their sample count: Metal's validation stops most such
    /// draws, and plain Metal zeroes or drops the colours. Vulkan draws no other count
    /// (VUID-vkCmdDraw-multisampledRenderToSingleSampled-07285).
    AttachmentFormats,
    /// A draw that binds a texture whose image the pass renders, where neither the attachment
    /// feedback loop nor a copy into the image's shadow reads it: on a device without the
    /// attachment reads (`floor::Capabilities::attachment_reads`), from the vertex function at the
    /// rendered level and layer, written, of a depth or stencil attachment, of an image with a
    /// storage view or over a buffer in place, or of an image two attachments of the pass render.
    /// Metal reads a render target through a texture.
    AttachmentRead,
    /// A draw at a base instance other than 0 through a pipeline state that steps a vertex layout
    /// per instance at a rate above 1, on a device that does not report
    /// `supportsNonZeroFirstInstance`: Vulkan's divisor takes `firstInstance` 0 alone there
    /// (VUID-vkCmdDraw-pNext-09461). Metal counts the step from any base instance. The same
    /// draw from base instance 0 runs.
    DividedBaseInstance,
    /// A draw of patches through a patch index buffer (`drawPatches:`'s `patchIndexBuffer`), which
    /// no replay stream's patch draws name: the control stage takes each patch's index from its
    /// position.
    PatchIndexBuffer,
}

/// What a render pass asks that the runtime does not build ([`Gap::RenderPass`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPassKind {
    /// A depth attachment and a stencil attachment that are not one texture's same level and
    /// slice: Vulkan's rendering takes one image view for both
    /// (VUID-VkRenderingInfo-pDepthAttachment-06085).
    DepthStencilViews,
    /// A `Clear` load of a depth attachment whose `clearDepth` lies outside `[0, 1]`, or is NaN:
    /// Vulkan clears no such depth without `VK_EXT_depth_range_unrestricted`
    /// (VUID-VkClearDepthStencilValue-depth-00022), which the floor does not require.
    ClearDepth,
    /// Attachments of two sample counts: Vulkan renders one count into every attachment of a
    /// pass (VUID-VkRenderingInfo-multisampledRenderToSingleSampled-06857).
    SampleCounts,
    /// A resolve texture, or a store action that resolves.
    Resolve,
    /// An attachment of a texture type but a 2D texture, a 2D array, a cube or a cube array: a 1D
    /// texture's, or a 3D texture's depth plane.
    TextureType,
    /// `MTLStoreActionCustomSampleDepthStore`, set on the pass or by the encoder's
    /// `setColorStoreAction:atIndex:` family.
    StoreAction,
    /// A `Clear` load action on an attachment whose format the runtime clears only through a
    /// shader: an extended-range, bfloat or YCbCr format.
    ClearFormat,
    /// A pass with no attachment, its size from `renderTargetWidth` and `renderTargetHeight` alone.
    NoAttachment,
}

/// What a texture view changes that the runtime does not build ([`Gap::TextureView`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureViewKind {
    /// Another pixel format Metal takes that the texture's image does not list: the device holds
    /// it in no viewed form of the image format's compatibility class.
    PixelFormat,
    /// Another texture type, but a 2D view of one slice of a 2D array.
    TextureType,
    /// Some of the slices of a texture that is not a 1D or 2D array: a cube's faces, or a cube
    /// array's cubes.
    Slices,
}

/// A limit a call ran into, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// A block longer than one allocation and one buffer hold on the device
    /// ([`Device::take_block`](crate::Device::take_block) past `maxMemoryAllocationSize` or
    /// `maxBufferSize`). Metal's counterpart is `maxBufferLength`, past which it answers nil.
    ///
    /// Basis: device (the device's `maxMemoryAllocationSize` and `maxBufferSize`).
    BlockLen {
        /// The length asked for, rounded up to the allocation granule.
        len: u64,
        /// The most the device holds in one block.
        max: u64,
    },
    /// A block after every block number has been used
    /// ([`Device::take_block`](crate::Device::take_block)): numbers are never reused during a
    /// device's life. Metal has no counterpart.
    ///
    /// Basis: ours (block numbers are 32 bits).
    BlockIds,
    /// A dispatch whose threadgroup memory, the kernel's own and the lengths set (each rounded up
    /// to 16 bytes), is past the device's `maxComputeSharedMemorySize`; for a threadgroup that
    /// runs the emulated-SIMD-group module, with the scratch of its SIMD groups. Metal's
    /// counterpart is `maxThreadgroupMemoryLength`, past which its validation stops the dispatch.
    ///
    /// Basis: device (the device's `maxComputeSharedMemorySize`).
    ThreadgroupMemory {
        /// The threadgroup memory the dispatch takes, in bytes.
        length: u64,
        /// The device's most.
        max: u32,
    },
    /// `setBytes` data longer than one parameter ring chunk. Metal documents the call for data
    /// under 4 KB.
    ///
    /// Basis: ours (a ring chunk is 64 KiB).
    BytesLength {
        /// The length asked for.
        len: u64,
        /// The most one call takes.
        max: u64,
    },
    /// A grant of no indices, or one past its class's range
    /// ([`Device::grant_indices`](crate::Device::grant_indices), [`IndexClass::range`]). Metal
    /// has no counterpart.
    ///
    /// Basis: ours (the descriptor arrays are 65,536 sampled and 65,536 storage textures, one
    /// index of both a texture's, and 4,000 samplers of each use, and index 0 is nil).
    IndexRange {
        /// The class asked for.
        class: IndexClass,
        /// The first index asked for.
        first: u32,
        /// How many indices.
        count: u32,
    },
    /// A texture whose extent or array layers are past what the device makes of its format,
    /// type, usage and sample count. A cube's layers are six a cube.
    ///
    /// Basis: device (the image format query's `maxExtent` and `maxArrayLayers`).
    TextureSize {
        /// The extent asked for.
        extent: [u32; 3],
        /// The array layers asked for.
        layers: u32,
        /// The device's most.
        max_extent: [u32; 3],
        /// The device's most layers.
        max_layers: u32,
    },
    /// A sampler state whose device samplers would take the device past its
    /// `maxSamplerAllocationCount`, less the sampler arrays' two defaults
    /// ([`Device::new_sampler_state`](crate::Device::new_sampler_state)). Equal create infos share
    /// one sampler, so it bounds distinct samplers, not sampler states.
    ///
    /// Basis: device (the device's `maxSamplerAllocationCount`).
    DeviceSamplers {
        /// The most samplers the device's states hold.
        max: u32,
    },
    /// A `VkDeviceMemory` allocation past the device's `maxMemoryAllocationCount` live ones,
    /// refused before the driver is called: each block, each imported block, each texture's own
    /// memory, each staging buffer of [`Texture::replace_region`](crate::Texture::replace_region)
    /// and [`Texture::get_bytes`](crate::Texture::get_bytes), the parameter ring's blocks and the
    /// zero source count one while they live ([`Gauges::blocks`](crate::Gauges::blocks)). Vulkan
    /// leaves an allocation past the count undefined, so the runtime counts its own. Metal has no
    /// counterpart: its allocations are bounded by memory alone.
    ///
    /// Basis: device (the device's `maxMemoryAllocationCount`).
    MemoryAllocations {
        /// The most allocations the device holds at once.
        max: u32,
    },
    /// A take that no free range fits made an allocation that the consumer's
    /// [`BlockMappings::map`](crate::BlockMappings::map) did not map: the runtime freed it, and
    /// hands out none of its blocks. Metal has no counterpart: its memory has no second mapping.
    ///
    /// Basis: ours (the consumer's hook gave the allocation no place in its own address range).
    AllocationNotMapped {
        /// The class of the allocation.
        class: MemoryClass,
        /// The length of the block the take asked for.
        len: u64,
    },
    /// An import of a handle the device does not take
    /// ([`Device::import_dma_buf`](crate::Device::import_dma_buf),
    /// [`Device::import_host_memory`](crate::Device::import_host_memory)): it does not list the
    /// handle's extensions (`VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf` for a
    /// dma-buf, `VK_EXT_external_memory_host` for host memory), or its buffers do not import the
    /// handle (`vkGetPhysicalDeviceExternalBufferProperties`). Metal's counterpart,
    /// `newBufferWithBytesNoCopy:length:options:deallocator:`, takes any page-aligned memory.
    ///
    /// Basis: device (the device's extensions and its external buffer properties).
    NoImport {
        /// The handle asked for.
        handle: ImportHandle,
    },
    /// An import whose handle's memory type query offers no type the import takes, so nothing is
    /// imported: for a dma-buf, the first type the fd's query offers is not host-visible and
    /// host-coherent, or the buffer's memory requirements do not list it; for host memory, the
    /// pointer's query offers no type that is both and that the requirements list. Metal's
    /// counterpart, `newBufferWithBytesNoCopy:length:options:deallocator:`, takes any page-aligned
    /// memory.
    ///
    /// Basis: device (the handle's memory type query and the buffer's memory requirements).
    ImportType {
        /// The handle asked for.
        handle: ImportHandle,
        /// The types the handle's query offered, one bit each.
        offered: u32,
        /// The types the buffer's memory requirements list, one bit each.
        listed: u32,
    },
    /// An import of no bytes, or one whose address or length is not a multiple of `alignment`:
    /// for host memory the device's `minImportedHostPointerAlignment`, and for either handle the
    /// alignment of the buffer's memory requirements where their size is past the length. Metal's
    /// counterpart, `newBufferWithBytesNoCopy:length:options:deallocator:`, takes memory of a
    /// whole number of pages at a page's first byte.
    ///
    /// Basis: device (`minImportedHostPointerAlignment` and the buffer's memory requirements).
    ImportAlignment {
        /// The handle asked for.
        handle: ImportHandle,
        /// The length asked for.
        len: u64,
        /// The multiple the device takes.
        alignment: u64,
    },
    /// A multisample texture of a sample count the device does not make for its format, type and
    /// usage.
    ///
    /// Basis: device (the image format query's `sampleCounts`).
    TextureSamples {
        /// The sample count asked for.
        sample_count: u32,
        /// The counts the device makes, one bit each (`VkSampleCountFlags`).
        counts: u32,
    },
}

impl Error {
    /// The error of a failed Vulkan `call`, for `map_err`.
    pub(crate) fn vulkan(call: &'static str) -> impl FnOnce(vk::Result) -> Error {
        move |result| Error::Vulkan { call, result }
    }

    /// An allocation's out-of-device-memory error.
    pub(crate) fn out_of_device_memory() -> Error {
        Error::Vulkan {
            call: "vkAllocateMemory",
            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        }
    }

    /// Whether it is an allocation's out-of-device-memory error.
    pub(crate) fn is_out_of_device_memory(&self) -> bool {
        matches!(
            self,
            Error::Vulkan {
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                ..
            }
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(error) => write!(f, "no Vulkan loader: {error}"),
            Error::NoDirectDriverLoading => f.write_str(
                "the Vulkan loader does not offer VK_LUNARG_direct_driver_loading, so it takes no driver of the consumer's",
            ),
            Error::BelowFloor(below) => write!(f, "{below}"),
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
            Error::Refused(refusal) => write!(f, "{refusal}"),
            Error::Limit(limit) => write!(f, "{limit}"),
            Error::NoSuchBlock(block) => {
                write!(f, "block {} is not one the consumer holds", block.get())
            }
            Error::Thread(error) => write!(f, "a runtime thread did not start: {error}"),
            Error::Quiesced => f.write_str("the device is quiesced"),
            Error::Translation(error) => write!(f, "the translator made no pipeline: {error}"),
            Error::Unsupported(gap) => write!(f, "{gap}"),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::EmptyBuffer => f.write_str("Metal makes no buffer of length 0"),
            Refusal::BufferTooLong { length, max } => {
                write!(f, "a buffer of {length} bytes is longer than {max}")
            }
            Refusal::BufferPastBlock {
                offset,
                length,
                block_len,
            } => write!(
                f,
                "a buffer of {length} bytes at offset {offset} leaves its block of {block_len} bytes"
            ),
            Refusal::ManagedBlockNotShared { block } => write!(
                f,
                "a Managed buffer's bytes are in block {}, which is Private",
                block.get()
            ),
            Refusal::ManagedCopyMissing => write!(
                f,
                "a Managed buffer on a device without unified memory needs a place for its device copy"
            ),
            Refusal::ManagedCopyOnUnified => write!(
                f,
                "a Managed buffer on a device with unified memory keeps no device copy to place"
            ),
            Refusal::ManagedCopyNotPrivate { block } => write!(
                f,
                "a Managed buffer's device copy is placed in block {}, which is not Private",
                block.get()
            ),
            Refusal::ZeroRangePastBlock {
                offset,
                length,
                block_len,
            } => write!(
                f,
                "a zero fill of {length} bytes at offset {offset} leaves its block of {block_len} bytes"
            ),
            Refusal::ZeroRangeShared { block } => write!(
                f,
                "a zero fill of block {}, which is Shared: the host zeroes its bytes",
                block.get()
            ),
            Refusal::ModifiedRange {
                start,
                length,
                buffer_length,
            } => write!(
                f,
                "a changed range of {length} bytes from {start} leaves its buffer of \
                 {buffer_length} bytes"
            ),
            Refusal::Committed => f.write_str("the command buffer is already committed"),
            Refusal::InScheduledHandler => f.write_str(
                "a scheduled handler runs on the submit thread, and this call may wait for it",
            ),
            Refusal::WaitOnOwnThread => f.write_str(
                "this wait, in a handler or a listener, waits for what its own thread does next",
            ),
            Refusal::OtherDevice => f.write_str("the object is another device's"),
            Refusal::NoCommandBuffers => {
                f.write_str("a command queue must hold at least one command buffer")
            }
            Refusal::EncoderOpen => f.write_str("an encoder of the command buffer is open"),
            Refusal::NoPipeline => f.write_str("no pipeline state is set"),
            Refusal::PatchDraw => f.write_str(
                "patches are drawn with a post-tessellation vertex function's pipeline state, and \
                 only with one",
            ),
            Refusal::NoTessellationFactorBuffer => {
                f.write_str("a draw of patches needs a tessellation factor buffer")
            }
            Refusal::BufferIndex { index } => {
                write!(f, "buffer index {index} is past the 31 entries")
            }
            Refusal::NoBufferAtIndex { index } => write!(f, "no buffer is bound at index {index}"),
            Refusal::TextureArgumentIndex { index } => {
                write!(f, "texture index {index} is past the 128 entries")
            }
            Refusal::SamplerArgumentIndex { index } => {
                write!(f, "sampler index {index} is past the 16 entries")
            }
            Refusal::SamplerIds { numbered, given } => write!(
                f,
                "{given} sampler indices were given for the {numbered} samplers the state numbers"
            ),
            Refusal::ThreadgroupSize { size, max } => write!(
                f,
                "a threadgroup of {size:?} threads is empty or past the state's {max}"
            ),
            Refusal::IndirectOffset { offset, length } => write!(
                f,
                "indirect arguments at offset {offset} of a {length}-byte buffer are misaligned or past its end"
            ),
            Refusal::BufferOffsetAlignment {
                index,
                offset,
                alignment,
            } => write!(
                f,
                "the buffer at index {index}, bound at offset {offset}, is not at a multiple of its argument's {alignment}-byte alignment"
            ),
            Refusal::IndicesGranted { class, index } => {
                write!(f, "{class:?} index {index} is granted already")
            }
            Refusal::IndicesNotGranted { class, index } => {
                write!(f, "{class:?} index {index} is not granted")
            }
            Refusal::IndicesInUse { class, index } => {
                write!(f, "an object lives at {class:?} index {index} of the grant")
            }
            Refusal::TextureIndex { index: Some(index) } => write!(
                f,
                "a texture no shader reaches takes no index, and {index} was given"
            ),
            Refusal::TextureIndex { index: None } => {
                f.write_str("a texture shaders read or write takes an index, and none was given")
            }
            Refusal::TextureZeroExtent => f.write_str("a texture has a width, height or depth of 0"),
            Refusal::TextureExtentForType => {
                f.write_str("the texture's height or depth is not 1 where its type needs 1")
            }
            Refusal::TextureCubeNotSquare => f.write_str("a cube texture's width and height differ"),
            Refusal::TextureArrayLength { array_length } => write!(
                f,
                "an array length of {array_length} does not suit the texture's type"
            ),
            Refusal::TextureLevelCount { levels, max } => write!(
                f,
                "{levels} mipmap levels are none or past the texture's {max}"
            ),
            Refusal::TextureSampleCount { sample_count } => write!(
                f,
                "a sample count of {sample_count} does not suit the texture's type"
            ),
            Refusal::TextureSwizzleWithWrites => {
                f.write_str("a swizzled texture takes no shader-write or render-target usage")
            }
            Refusal::TextureBytesPrivate => {
                f.write_str("the CPU reaches no Private or Memoryless texture")
            }
            Refusal::TextureRegion { level, slice } => write!(
                f,
                "the region of level {level}, slice {slice} is not inside the texture"
            ),
            Refusal::TextureBytesLayout {
                bytes_per_row,
                bytes_per_image,
                len,
            } => write!(
                f,
                "rows {bytes_per_row} and images {bytes_per_image} bytes apart in {len} bytes do not hold the region"
            ),
            Refusal::BufferCopyLayout {
                offset,
                bytes_per_row,
                bytes_per_image,
                len,
            } => write!(
                f,
                "a region at {offset} with rows {bytes_per_row} and images {bytes_per_image} bytes apart is no layout Metal takes in a buffer of {len} bytes"
            ),
            Refusal::FillRange { start, end, len } => write!(
                f,
                "a fill of {start}..{end} is empty or past the buffer's {len} bytes"
            ),
            Refusal::TextureViewRange {
                level,
                level_count,
                slice,
                slice_count,
            } => write!(
                f,
                "a view of {level_count} levels from {level} and {slice_count} slices from {slice} is empty or leaves the texture"
            ),
            Refusal::TextureViewFormat { texture, view } => write!(
                f,
                "a {view:?} view of a {texture:?} texture, which Metal does not view it as with \
                 its usage"
            ),
            Refusal::MipmapFormat { pixel_format } => write!(
                f,
                "mipmaps of a {pixel_format:?} texture, which is not a colour-renderable, \
                 filterable format"
            ),
            Refusal::TextureCopyMismatch {
                source,
                destination,
            } => write!(
                f,
                "a copy from {source:?} to {destination:?} is between two formats or two sample counts, which Metal refuses"
            ),
            Refusal::ManagedTextureBytes { level, slice } => write!(
                f,
                "a Managed texture's bytes of level {level}, slice {slice} are not where a copy reads them in place"
            ),
            Refusal::BufferTextureLayout {
                offset,
                bytes_per_row,
                len,
            } => write!(
                f,
                "a texture at offset {offset}, rows {bytes_per_row} bytes apart, does not lie in whole texels inside a buffer of {len} bytes"
            ),
            Refusal::HeapTexturePlace {
                offset,
                size,
                align,
                heap_len,
            } => write!(
                f,
                "a heap texture of {size} bytes aligned to {align} does not lie at offset {offset} of a heap of {heap_len} bytes"
            ),
            Refusal::BufferCopyRange { offset, size, len } => write!(
                f,
                "a copy of {size} bytes at {offset} leaves a buffer of {len} bytes"
            ),
            Refusal::BufferCopyUnaligned {
                source_offset,
                destination_offset,
                size,
            } => write!(
                f,
                "a copy of {size} bytes from {source_offset} to {destination_offset} is not in multiples of 4"
            ),
            Refusal::BufferCopyOverlap {
                source_offset,
                destination_offset,
                size,
            } => write!(
                f,
                "a copy of {size} bytes from {source_offset} to {destination_offset} reads bytes it writes"
            ),
            Refusal::AttachmentUsage { pixel_format } => write!(
                f,
                "an attachment's texture of {pixel_format:?} was made without RenderTarget usage"
            ),
            Refusal::AttachmentPlace { level, slice } => write!(
                f,
                "an attachment's level {level}, slice {slice} is past its texture's"
            ),
            Refusal::AttachmentFormat { pixel_format } => write!(
                f,
                "an attachment of {pixel_format:?} is not of its kind's formats"
            ),
            Refusal::VertexBufferOffset {
                index,
                offset,
                length,
            } => write!(
                f,
                "vertex buffer {index} at offset {offset} lies past its {length} bytes"
            ),
            Refusal::IndexBufferOffset { offset, length } => write!(
                f,
                "indices at offset {offset} of a {length}-byte buffer are misaligned or past its end"
            ),
            Refusal::TessellationFactorOffset { offset } => write!(
                f,
                "tessellation factors at offset {offset} start at an odd byte"
            ),
            Refusal::ArgumentPlace { place, alignment } => write!(
                f,
                "arguments at byte {place} of their block are not at a multiple of {alignment}"
            ),
            Refusal::ViewportNotFinite => f.write_str("a viewport field is NaN or infinite"),
            Refusal::StencilAttachment => {
                f.write_str("a draw tests the stencil in a pass without a stencil attachment")
            }
        }
    }
}

impl fmt::Display for Gap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gap::IndirectCommandBuffers => {
                f.write_str("pipeline states for indirect command buffers are not built")
            }
            Gap::KernelInput => {
                f.write_str("the kernel reads an input the compute encoder does not supply yet")
            }
            Gap::NoStageInRegion => {
                f.write_str("the kernel reads the stage-in region, which the encoder never set")
            }
            Gap::PartialSimdgroups {
                threads_per_threadgroup,
                width,
            } => write!(
                f,
                "a threadgroup of {threads_per_threadgroup:?} is not whole SIMD groups of {width} threads, and the state has no emulated module for it"
            ),
            Gap::PartialThreadgroups {
                threads_per_grid,
                threads_per_threadgroup,
            } => write!(
                f,
                "a grid of {threads_per_grid:?} threads in threadgroups of {threads_per_threadgroup:?} ends in partial threadgroups, which this kernel's variant does not run"
            ),
            Gap::MemorylessTexture => f.write_str("Memoryless textures are not built"),
            Gap::CubeArrayTexture => f.write_str("cube array textures are not built"),
            Gap::CubicSampler => f.write_str("cubic sampler filters are not built"),
            Gap::LodClamps => f.write_str(
                "a sampler state's bind-time LOD clamps that read otherwise than its own are not built",
            ),
            Gap::TextureBuffer => f.write_str("texture buffers are not built"),
            Gap::TextureFormat { pixel_format } => write!(
                f,
                "no texture of {pixel_format:?} is made on this device for the usage"
            ),
            Gap::MipmapFilter { pixel_format } => write!(
                f,
                "mipmaps of a {pixel_format:?} texture, whose image this device's blits do not filter"
            ),
            Gap::TextureBytes { pixel_format } => write!(
                f,
                "bytes are not copied to or from this texture of {pixel_format:?}"
            ),
            Gap::FloatDepthUpload { pixel_format } => write!(
                f,
                "bytes are not uploaded into the 32-bit float depth of this texture of {pixel_format:?}"
            ),
            Gap::TextureCopy {
                source,
                destination,
            } => write!(
                f,
                "this copy from a texture of {source:?} to one of {destination:?} is not built"
            ),
            Gap::TextureOverBuffer { pixel_format } => write!(
                f,
                "this texture of {pixel_format:?} over a buffer is not held in place on this device"
            ),
            Gap::HeapTexture { pixel_format } => write!(
                f,
                "a heap texture of {pixel_format:?} is not placed in this heap on this device"
            ),
            Gap::BufferCopyAlignment {
                pixel_format,
                offset,
                bytes_per_row,
                bytes_per_image,
            } => write!(
                f,
                "a copy of {pixel_format:?} at {offset} with rows {bytes_per_row} and images {bytes_per_image} bytes apart in a buffer is not built"
            ),
            Gap::FillAlignment { start, end } => write!(
                f,
                "a fill of {start}..{end}, not whole 4-byte words, is not built"
            ),
            Gap::FenceOtherQueue => f.write_str("a fence used on a second queue is not built"),
            Gap::SwizzledReadWrite { slot } => write!(
                f,
                "a read_write texture slot {slot} bound to a swizzled texture is not built"
            ),
            Gap::RenderPipeline { kind } => {
                write!(f, "a render pipeline state with {kind:?} is not built")
            }
            Gap::TextureView { kind } => write!(f, "a texture view that changes {kind:?} is not built"),
            Gap::RenderPass { kind } => write!(f, "a render pass with {kind:?} is not built"),
            Gap::Draw { kind } => write!(f, "a draw with {kind:?} is not built"),
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Limit::BlockLen { len, max } => write!(
                f,
                "a block of {len} bytes is past the device's largest, {max} bytes"
            ),
            Limit::BlockIds => f.write_str("every block number has been used"),
            Limit::ThreadgroupMemory { length, max } => write!(
                f,
                "{length} bytes of threadgroup memory are past the device's {max}"
            ),
            Limit::BytesLength { len, max } => {
                write!(f, "{len} bytes are past the most one call takes, {max}")
            }
            Limit::IndexRange {
                class,
                first,
                count,
            } => write!(
                f,
                "{count} {class:?} indices from {first} are none or leave the class's range"
            ),
            Limit::TextureSize {
                extent,
                layers,
                max_extent,
                max_layers,
            } => write!(
                f,
                "a texture of {extent:?} in {layers} layers is past the device's {max_extent:?} in {max_layers}"
            ),
            Limit::DeviceSamplers { max } => {
                write!(f, "the device holds {max} samplers for sampler states already")
            }
            Limit::MemoryAllocations { max } => {
                write!(f, "the device holds {max} memory allocations already")
            }
            Limit::AllocationNotMapped { class, len } => write!(
                f,
                "the consumer did not map the {class:?} allocation made for a block of {len} bytes"
            ),
            Limit::NoImport { handle } => write!(f, "the device imports no {handle:?}"),
            Limit::ImportType {
                handle,
                offered,
                listed,
            } => write!(
                f,
                "the {handle:?}'s query offers types {offered:#x}, and no host-visible, \
                 host-coherent type the import takes of the buffer's {listed:#x}"
            ),
            Limit::ImportAlignment {
                handle,
                len,
                alignment,
            } => write!(
                f,
                "an import of {len} bytes of {handle:?} at no multiple of {alignment} bytes"
            ),
            Limit::TextureSamples {
                sample_count,
                counts,
            } => write!(
                f,
                "a sample count of {sample_count} is not among the device's ({counts:#x})"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Loader(error) => Some(error),
            Error::BelowFloor(below) => Some(below),
            Error::Vulkan { result, .. } => Some(result),
            Error::Thread(error) => Some(error),
            Error::Translation(error) => Some(error),
            Error::NoDirectDriverLoading
            | Error::Refused(_)
            | Error::Limit(_)
            | Error::NoSuchBlock(_)
            | Error::Quiesced
            | Error::Unsupported(_) => None,
        }
    }
}
