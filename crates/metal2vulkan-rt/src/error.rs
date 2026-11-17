//! The error every fallible call returns: one typed variant per case, never text to parse.

use crate::arrays::IndexClass;
use crate::floor::BelowFloor;
use crate::memory::BlockId;
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
    /// [`Instance::new`](crate::Instance::new) found no Vulkan loader to load. Metal has no
    /// counterpart: Metal is part of the system.
    Loader(ash::LoadingError),
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
    /// A block the consumer does not hold: never taken, or already released. Metal has no
    /// counterpart: blocks are the consumer's memory, not Metal objects.
    NoSuchBlock(BlockId),
    /// The runtime could not start one of its threads. Metal has no counterpart.
    Thread(io::Error),
    /// [`Device::new_compute_pipeline_state`](crate::Device::new_compute_pipeline_state): the
    /// translator made no pipeline of the kernel: a typed refusal of what it does not translate,
    /// a descriptor Metal refuses, or a translation past its 20-second budget. Metal's
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
    /// A call on a command buffer already committed: encoding, adding a handler, enqueueing or
    /// committing again.
    ///
    /// Basis: metal-documented (Metal's validation asserts "commit an already committed command
    /// buffer", "Completed handler provided after commit call" and "Scheduled handler provided
    /// after commit call").
    Committed,
    /// An event, a buffer or a pipeline state of another device in an encode call
    /// ([`CommandBuffer::encode_wait_for_event`](crate::CommandBuffer::encode_wait_for_event),
    /// [`CommandBuffer::encode_signal_event`](crate::CommandBuffer::encode_signal_event), or a
    /// [`ComputeCommandEncoder`](crate::ComputeCommandEncoder) call).
    ///
    /// Basis: ours (an event's semaphore, a buffer's block and a pipeline belong to the
    /// `VkDevice` that made them).
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
    /// A dispatch with no pipeline state set.
    ///
    /// Basis: metal-documented (a compute dispatch needs a compute pipeline state set first).
    NoPipeline,
    /// A buffer argument index past Metal's 31 entries (`setBuffer`, `setBytes`,
    /// `setBufferOffset`, `setThreadgroupMemoryLength`).
    ///
    /// Basis: metal-documented (the buffer argument table has 31 entries per stage).
    BufferIndex {
        /// The index asked for.
        index: u32,
    },
    /// `setBufferOffset` at an index where no buffer is bound.
    ///
    /// Basis: ours (there is no buffer whose offset to change).
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
    /// A release of an index that is not granted
    /// ([`Device::release_indices`](crate::Device::release_indices)). Metal has no counterpart.
    ///
    /// Basis: ours (a grant is returned once).
    IndicesNotGranted {
        /// The class of the index.
        class: IndexClass,
        /// The first index of the range that is not granted.
        index: u32,
    },
    /// A texture at an index where a texture lives already
    /// ([`Device::new_texture`](crate::Device::new_texture)). Metal has no counterpart: Metal
    /// chooses its resource IDs itself.
    ///
    /// Basis: ours (an index names one object at a time, and its elements are that object's).
    IndexInUse {
        /// The class of the index.
        class: IndexClass,
        /// The index.
        index: u32,
    },
    /// A release of a grant while a texture lives at one of its indices
    /// ([`Device::release_indices`](crate::Device::release_indices)). Metal has no counterpart.
    ///
    /// Basis: ours (an object's index stays granted while the object lives).
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
    /// [`Texture::replace_region`](crate::Texture::replace_region) or
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) of a level or slice past the texture's,
    /// or of a region that leaves the level or is off the format's block grid.
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
    /// A kernel that reads something the compute encoder does not supply yet: a texture or a
    /// sampler slot, a function table, a `[[stage_in]]` fetch, a parameter the runtime does not
    /// fill, or a specialization the runtime does not set. Buffers, `setBytes` data, threadgroup
    /// memory lengths, the dispatch's grid and a pinned subgroup size are supplied.
    ///
    /// Basis: ours (the encoder binds buffers alone).
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
    /// A texture of `Memoryless` storage, which only render passes use.
    ///
    /// Basis: ours (render passes are not built).
    MemorylessTexture,
    /// A texture buffer (`MTLTextureTypeTextureBuffer`).
    ///
    /// Basis: ours (no texel-buffer array; no replay stream uses one).
    TextureBuffer,
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
    /// [`Texture::replace_region`](crate::Texture::replace_region) or
    /// [`Texture::get_bytes`](crate::Texture::get_bytes) of a texture whose image is no plain
    /// copy of Metal's bytes in one aspect: a format of depth and stencil both, or a multisample
    /// texture.
    ///
    /// Basis: ours (no copy between such an image and bytes is built).
    TextureBytes {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
    /// [`Texture::replace_region`](crate::Texture::replace_region) into a 32-bit float depth
    /// (`Depth32Float`). Metal keeps any `float` there; Vulkan copies no depth outside `[0, 1]`
    /// into a depth aspect without `VK_EXT_depth_range_unrestricted`, which the floor does not
    /// require (the Intel devices measured lack it), and no recorded stream uploads one.
    ///
    /// Basis: ours (no upload of a 32-bit float depth is built).
    FloatDepthUpload {
        /// The texture's pixel format.
        pixel_format: PixelFormat,
    },
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
    /// A threadgroup longer on an axis than the device's `maxComputeWorkGroupSize` (the floor holds
    /// 1,024 by 1,024 by 64). Metal's go to 1,024 on every axis.
    ///
    /// Basis: device (the device's `maxComputeWorkGroupSize`).
    ThreadgroupShape {
        /// The threadgroup size asked for.
        size: [u32; 3],
        /// The device's most.
        max: [u32; 3],
    },
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
    /// Basis: ours (the descriptor arrays are 65,536 sampled textures, 4,096 storage textures and
    /// 4,000 samplers of each use, and index 0 is nil).
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(error) => write!(f, "no Vulkan loader: {error}"),
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
            Refusal::BufferPastBlock {
                offset,
                length,
                block_len,
            } => write!(
                f,
                "a buffer of {length} bytes at offset {offset} leaves its block of {block_len} bytes"
            ),
            Refusal::Committed => f.write_str("the command buffer is already committed"),
            Refusal::OtherDevice => f.write_str("the object is another device's"),
            Refusal::NoCommandBuffers => {
                f.write_str("a command queue must hold at least one command buffer")
            }
            Refusal::EncoderOpen => f.write_str("an encoder of the command buffer is open"),
            Refusal::NoPipeline => f.write_str("no compute pipeline state is set"),
            Refusal::BufferIndex { index } => {
                write!(f, "buffer index {index} is past the 31 entries")
            }
            Refusal::NoBufferAtIndex { index } => write!(f, "no buffer is bound at index {index}"),
            Refusal::ThreadgroupSize { size, max } => write!(
                f,
                "a threadgroup of {size:?} threads is empty or past the state's {max}"
            ),
            Refusal::IndirectOffset { offset, length } => write!(
                f,
                "indirect arguments at offset {offset} of a {length}-byte buffer are misaligned or past its end"
            ),
            Refusal::IndicesGranted { class, index } => {
                write!(f, "{class:?} index {index} is granted already")
            }
            Refusal::IndicesNotGranted { class, index } => {
                write!(f, "{class:?} index {index} is not granted")
            }
            Refusal::IndexInUse { class, index } => {
                write!(f, "an object lives at {class:?} index {index} already")
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
            Gap::TextureBuffer => f.write_str("texture buffers are not built"),
            Gap::TextureFormat { pixel_format } => write!(
                f,
                "no texture of {pixel_format:?} is made on this device for the usage"
            ),
            Gap::TextureBytes { pixel_format } => write!(
                f,
                "bytes are not copied to or from this texture of {pixel_format:?}"
            ),
            Gap::FloatDepthUpload { pixel_format } => write!(
                f,
                "bytes are not uploaded into the 32-bit float depth of this texture of {pixel_format:?}"
            ),
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
            Limit::ThreadgroupShape { size, max } => {
                write!(f, "a threadgroup of {size:?} is past the device's {max:?}")
            }
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
            Error::Refused(_)
            | Error::Limit(_)
            | Error::NoSuchBlock(_)
            | Error::Quiesced
            | Error::Unsupported(_) => None,
        }
    }
}
