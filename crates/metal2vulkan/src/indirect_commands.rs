//! The indirect-command record format: what a translated kernel's command encoders write, what the
//! CPU's do, and what a replay reads (`bind::indirect_commands` is the public path).
//!
//! The records are ordinary buffer storage, a header and one fixed-size record per command; they
//! are not Vulkan command buffers or device-generated-command tokens.

mod descriptor;
pub use descriptor::{CommandDescriptor, ComputeDispatch, RenderDraws};
mod host;
pub use host::{BindingError, CommandBuffer, CommandBufferError, RecordView, ResolvedBindings};

use std::fmt;
use std::num::NonZeroU64;

/// Buffer namespace of an encoded command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferStage {
    Kernel,
    Vertex,
    Fragment,
    Object,
    /// A render command's mesh buffers, which take the words of a compute command's kernel
    /// buffers: a record holds one family's stages, as its descriptor allows one family's
    /// commands ([`BufferStage::of`]).
    Mesh,
}

/// Recognized command-encoder operation; recognition does not establish lowering support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandEncoderOperation {
    Buffer {
        stage: BufferStage,
        address_space: u32,
    },
    Dispatch {
        exact_threads: bool,
    },
    Pipeline {
        render: bool,
    },
    Barrier {
        render: bool,
        enabled: bool,
    },
    ThreadgroupMemory,
    Draw,
    /// `set_stage_in_region`: the six words of the region.
    StageInRegion,
    /// `set_imageblock_size`: the two `u16`s of the size.
    ImageblockSize,
    /// `reset`: the record restored to [`CommandRecord::default`], of a render command where
    /// `render`, else a compute command.
    Reset {
        render: bool,
    },
    /// `draw_indexed_primitives`: an indexed draw's arguments.
    DrawIndexed,
    /// `draw_patches`: a patch draw's arguments.
    DrawPatches,
    /// `copy_command`: a compute command's record from a source command, whose buffer's header
    /// the lowering checks against the destination's (`lower::access::encode`).
    Copy,
    /// `draw_indexed_patches`: an indexed patch draw's arguments, a patch draw's and its
    /// control-point index buffer's address ([`Action::DrawIndexedPatches`]).
    DrawIndexedPatches,
    /// `draw_mesh_threadgroups` (`threads` false) or `draw_mesh_threads`: a mesh draw's grid and
    /// its object and mesh threadgroups' threads ([`Action::DrawMeshThreadgroups`],
    /// [`Action::DrawMeshThreads`]).
    DrawMesh {
        threads: bool,
    },
}

/// Number of buffer slots represented in a command.
pub const BINDING_COUNT: usize = 32;
/// Number of threadgroup-buffer argument-table entries exposed by the Metal ABI
/// (`air.max_threadgroup_buffers` is 31, so indices run 0..=30): the threadgroup-memory lengths a
/// command carries. Threadgroup buffers take no descriptor.
pub const THREADGROUP_BUFFER_ARGUMENT_COUNT: u32 = 31;
/// [`THREADGROUP_BUFFER_ARGUMENT_COUNT`] as a length.
pub const THREADGROUP_BUFFER_ARGUMENT_COUNT_USIZE: usize =
    THREADGROUP_BUFFER_ARGUMENT_COUNT as usize;
const THREADGROUP_COUNT: usize = THREADGROUP_BUFFER_ARGUMENT_COUNT_USIZE;
const HEADER_WORDS: usize = 16;
const BUFFER_WORDS: usize = 4;
const BUFFER_STAGES: usize = 4;
const THREADGROUP_WORD: usize = HEADER_WORDS + BUFFER_STAGES * BINDING_COUNT * BUFFER_WORDS;
/// Size of one command record: its words through the threadgroup lengths, then the imageblock
/// size's and the pipeline state's (`layout::IMAGEBLOCK`, `layout::STATE`). The storage base must
/// be aligned to eight bytes.
pub const COMMAND_BYTES: usize =
    ((THREADGROUP_WORD + THREADGROUP_COUNT + 2) * 4).next_multiple_of(8);
/// Host-initialized command count, atomic encoding-fault flag, descriptor flags, and packed binding counts.
pub const COMMAND_BUFFER_HEADER_BYTES: usize = 16;
/// Bytes of one command's written flag (a `u32`), past every record ([`written_flag_offset`]).
pub const WRITTEN_FLAG_BYTES: usize = 4;

/// The byte offset of command `index`'s written flag in the storage of a buffer of `capacity`
/// commands: the flags follow the records, one `u32` a command. Each encoder operation that
/// writes a command's record (one the header's policy takes, `lower::access::encode`) stores 1
/// at its flag, contained as the record's fields are: storage that ends at its records takes no
/// flag, and its records are written as before. A consumer whose storage holds the flags, and
/// whose own writes of the records set them as well, knows a record whose flag is clear to hold
/// what it last reset it to, and may reset only the flagged ones, clearing each flag with its
/// record.
pub const fn written_flag_offset(capacity: u32, index: u32) -> u64 {
    COMMAND_BUFFER_HEADER_BYTES as u64
        + capacity as u64 * COMMAND_BYTES as u64
        + index as u64 * WRITTEN_FLAG_BYTES as u64
}

/// The guest address of the records of the indirect command buffer of resource ID 0, the null
/// one, which no records lie at ([`records_address`]).
pub const RECORDS_ADDRESS_BASE: u64 = 1 << 63;
/// How far an indirect command buffer's resource ID is shifted to give the guest address its
/// records start at past [`RECORDS_ADDRESS_BASE`] ([`records_address`]): 2^31 bytes from one ID's
/// start to the next's. Records of more bytes run past the next IDs' starts, which a search for
/// those IDs' records still tells apart: it finds an ID's records only at their start.
pub(crate) const RECORDS_ADDRESS_SHIFT: u32 = 31;
/// The first resource ID whose records [`records_address`] cannot place, 2^31: the IDs of indirect
/// command buffers run from 1 below it, as Metal's do (on an M3 Max, a reservation of 2^31 - 2
/// handed out 2^31 - 1 next, and one of 2^31 to 2^32 - 1 never returned). The last ID's records
/// start 2^62 bytes below 2^64, so records of any length a device holds fit after any ID's start.
pub const RECORDS_ID_LIMIT: u64 = 1 << 31;

/// The guest address the records of the indirect command buffer whose resource ID is `id` (below
/// [`RECORDS_ID_LIMIT`]) start at: `2^63 + id * 2^31`. An argument buffer's command-buffer member
/// holds the ID, as Metal's `gpuResourceID` and `setIndirectCommandBuffer:atIndex:` give it (small
/// integers from 1, measured on an M3 Max and an M4, or the IDs a reservation leaves next, up to
/// 2^31 - 1 on an M3 Max), and a translated kernel that encodes through it reaches the records
/// that start at this address in its resident set, and no other: records of more than 2^31 bytes
/// run past the next IDs' starts, and the resident set keeps each records range whole
/// (`ResidentSet`). So a consumer that makes an indirect command buffer with ID `id` makes its
/// record buffer resident at `records_address(id)`, a range no Metal buffer reaches (each lies
/// below 2^63). The null ID 0 gives an address no records start at, so an encoder operation
/// through it is contained as one through an ID whose records are not resident is.
pub const fn records_address(id: u64) -> u64 {
    RECORDS_ADDRESS_BASE | (id << RECORDS_ADDRESS_SHIFT)
}

/// The byte offsets the lowering of the encoder operations writes through device addresses
/// (`lower::access::encode`), with the header checks of [`Policy`].
pub(crate) mod layout {
    use super::{
        BufferStage, BINDING_COUNT, BUFFER_STAGES, BUFFER_WORDS, HEADER_WORDS, THREADGROUP_COUNT,
        THREADGROUP_WORD,
    };

    /// The header's command capacity (`u32`).
    pub(crate) const CAPACITY: u64 = 0;
    /// The header's sticky encoding-fault flag (`u32`, written atomically).
    pub(crate) const FAULT: u64 = 4;
    /// The header's descriptor policy bits (`u32`).
    pub(crate) const POLICY: u64 = 8;
    /// The header's packed binding counts (`u32`, one byte per count): a compute descriptor's
    /// kernel buffers in byte 0 and threadgroup lengths in byte 3; a render descriptor's object,
    /// vertex, fragment and mesh buffers in bytes 0 to 3.
    pub(crate) const COUNTS: u64 = 12;
    /// The first record.
    pub(crate) const RECORDS: u64 = super::COMMAND_BUFFER_HEADER_BYTES as u64;
    /// Bytes of one record.
    pub(crate) const RECORD: u64 = super::COMMAND_BYTES as u64;
    /// A record's action tag (`u32`).
    pub(crate) const ACTION: u64 = 0;
    /// A record's barrier flag (`u32`).
    pub(crate) const BARRIER: u64 = 4;
    /// A record's pipeline token (`u64`).
    pub(crate) const PIPELINE: u64 = 8;
    /// A record's six action arguments (`u32` each): a dispatch's grid and local size, or a
    /// draw's primitive type, vertex start, vertex count, instance count, and base instance.
    pub(crate) const ARGUMENTS: u64 = 16;
    /// Bytes of one buffer binding: its address (`u64`), then its attribute stride (`u64`).
    pub(crate) const BINDING: u64 = (BUFFER_WORDS * 4) as u64;
    /// A buffer binding's attribute stride, within the binding.
    pub(crate) const STRIDE: u64 = 8;
    /// A record's threadgroup-memory lengths (`u32` each).
    pub(crate) const THREADGROUP: u64 = (THREADGROUP_WORD * 4) as u64;

    /// The first buffer binding of `stage` in a record.
    pub(crate) const fn buffers(stage: BufferStage) -> u64 {
        ((HEADER_WORDS + stage.index() * BINDING_COUNT * BUFFER_WORDS) * 4) as u64
    }

    /// Every buffer binding of a record, all four stages' words (`BINDING` bytes each, from
    /// `buffers(BufferStage::Kernel)`).
    pub(crate) const ALL_BINDINGS: u64 = (BUFFER_STAGES * BINDING_COUNT) as u64;
    /// A record's stage-in region (`u32` each, after the six action arguments): origin x, y and
    /// z, then size x, y and z; all zero when the command sets none.
    pub(crate) const STAGE_IN: u64 = ARGUMENTS + 24;
    /// The words a draw's arguments may take from [`ARGUMENTS`]: the six action arguments and the
    /// stage-in region's six, which a render command never sets. An indexed draw takes eight
    /// (primitive type, index count, index type, instance count, the index buffer's address as a
    /// `u64`, base vertex, base instance), a patch draw eleven (control points, patch start, patch
    /// count, instance count, the patch index buffer's and the factor buffer's addresses as
    /// `u64`s, base instance, the factors' instance stride, the factor scale's `f32` bits), a mesh
    /// draw nine (its grid, the threads of an object threadgroup and of a mesh threadgroup, three
    /// each); each address lies 8-byte aligned.
    pub(crate) const DRAW_ARGUMENT_WORDS: usize = 12;
    /// An indexed patch draw's control-point index buffer's address (`u64`): it takes the first
    /// two threadgroup lengths' words, which a render command never sets, as its other arguments
    /// take a patch draw's [`DRAW_ARGUMENT_WORDS`].
    pub(crate) const CONTROL_POINT_INDICES: u64 = THREADGROUP;
    /// A record's imageblock size (`u16` each, after the threadgroup lengths): width, then height;
    /// zero when the command sets none.
    pub(crate) const IMAGEBLOCK: u64 = ((THREADGROUP_WORD + THREADGROUP_COUNT) * 4) as u64;
    /// A record's pipeline state word (`u32`, after the imageblock size, 8-byte aligned and alone
    /// in its `u64`): [`RETAINED`] where the pipeline word holds the pipeline a compute command's
    /// reset left it executing with, which its read-back reports cleared, as Metal's getter does;
    /// [`KERNEL_PIPELINE`] where a kernel's encoder wrote the pipeline word; 0 where the pipeline
    /// word holds no pipeline. Each bit lies only beside a pipeline.
    pub(crate) const STATE: u64 = IMAGEBLOCK + 4;
    /// The state word's bit of a retained pipeline.
    pub(crate) const RETAINED: u32 = 1;
    /// The state word's bit of a pipeline a kernel's encoder wrote: its pipeline setter, or its
    /// copy from a command whose pipeline one wrote ([`CommandRecord::kernel_pipeline`]).
    ///
    /// [`CommandRecord::kernel_pipeline`]: super::CommandRecord::kernel_pipeline
    pub(crate) const KERNEL_PIPELINE: u32 = 2;
    /// Both bits of the state word.
    pub(crate) const STATE_BITS: u32 = RETAINED | KERNEL_PIPELINE;
}

/// The descriptor policy an encoder operation's write requires of the record header: its
/// policy bits masked with `required | forbidden` equal `required`, and hold one of `any` where
/// `any` is not 0, the command index is below the capacity, a binding index is below its count,
/// and a dynamic stride is all ones unless the descriptor enables dynamic strides. A write that
/// fails any of them sets the fault flag instead and writes nothing, but a draw on a buffer of
/// `dropped_on` alone, which writes nothing and sets no fault. A threadgroup length is kept as
/// set, a multiple of 16 or not, as Metal keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    pub(crate) required: u32,
    pub(crate) forbidden: u32,
    /// The kinds of which the descriptor must allow one (0: no such check): those whose Metal
    /// record holds the operation's draw ([`DrawKind::holders`]).
    pub(crate) any: u32,
    /// The byte of the header's packed counts that bounds the operation's binding index, where
    /// it binds one.
    pub(crate) binding: Option<u8>,
    pub(crate) dynamic_stride: bool,
    /// The kinds on a buffer of which, allowing none of `any`, a draw that is otherwise valid is
    /// dropped: it writes nothing and sets no fault (0: none). A kernel's draw of primitives on
    /// a buffer of mesh draws alone draws nothing on Metal, without a fault (a Metal probe).
    pub(crate) dropped_on: u32,
}

/// A draw's kind, as the command types Metal's records are made for order them: a buffer's
/// records hold a draw of a kind its types allow, or of a kind whose record a kind it allows
/// holds. Draws of primitives fit every render buffer's records but those of mesh draws alone,
/// indexed draws an indexed or a patch buffer's, and patch and indexed patch draws a patch or an
/// indexed patch buffer's (a kernel's draw on a buffer of such a kind runs as on its own on the M3
/// Max and the M4; on a buffer of a smaller kind the GPU faults, or on the M4 a patch draw on an
/// indexed buffer draws nothing). Mesh draws of either kind fit a buffer of either mesh kind's
/// records, and no other's (a Metal probe, the M3 Max and an M4: a kernel's mesh draw of either
/// kind on a buffer of either mesh kind runs as on its own, and on a buffer of draws of primitives
/// the GPU faults). Dispatches of either kind fit every compute buffer's records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawKind {
    Primitives,
    Indexed,
    Patches,
    /// A mesh draw of threadgroups or of threads.
    Mesh,
}

impl DrawKind {
    /// The header bits of the draw kinds whose records hold a draw of this kind.
    pub(crate) const fn holders(self) -> u32 {
        match self {
            DrawKind::Primitives => DRAW | DRAW_INDEXED | DRAW_PATCHES | DRAW_INDEXED_PATCHES,
            DrawKind::Indexed => DRAW_INDEXED | DRAW_PATCHES | DRAW_INDEXED_PATCHES,
            DrawKind::Patches => DRAW_PATCHES | DRAW_INDEXED_PATCHES,
            DrawKind::Mesh => DRAW_MESH_THREADGROUPS | DRAW_MESH_THREADS,
        }
    }
}

// The header's policy bits, which a command-buffer descriptor sets at initialization and the
// encoder operations' lowering checks ([`Policy`]).
pub(crate) const COMPUTE: u32 = 1;
pub(crate) const RENDER: u32 = 2;
pub(crate) const GROUPS: u32 = 4;
pub(crate) const THREADS: u32 = 8;
pub(crate) const DRAW: u32 = 16;
/// The header bit that enables dynamic attribute strides.
pub(crate) const DYNAMIC_STRIDE: u32 = 32;
pub(crate) const INHERIT_PIPELINE: u32 = 64;
pub(crate) const INHERIT_BUFFERS: u32 = 128;
/// The header bit of a render descriptor that allows indexed draws (`DRAW` allows draws of
/// primitives).
pub(crate) const DRAW_INDEXED: u32 = 256;
/// The header bit of a render descriptor that allows patch draws.
pub(crate) const DRAW_PATCHES: u32 = 512;
/// The header bit of a render descriptor that allows indexed patch draws.
pub(crate) const DRAW_INDEXED_PATCHES: u32 = 1024;
/// The header bit of a render descriptor that allows mesh draws of threadgroups.
pub(crate) const DRAW_MESH_THREADGROUPS: u32 = 2048;
/// The header bit of a render descriptor that allows mesh draws of threads.
pub(crate) const DRAW_MESH_THREADS: u32 = 4096;

impl BufferStage {
    /// The stage's words in a record: a render command's mesh buffers take a compute command's
    /// kernel buffers'.
    const fn index(self) -> usize {
        match self {
            Self::Kernel | Self::Mesh => 0,
            Self::Vertex => 1,
            Self::Fragment => 2,
            Self::Object => 3,
        }
    }

    /// The stages a record of a compute (`render` false) or a render command holds, one for
    /// each of its words' stages, in their order.
    pub const fn of(render: bool) -> [BufferStage; BUFFER_STAGES] {
        match render {
            false => [Self::Kernel, Self::Vertex, Self::Fragment, Self::Object],
            true => [Self::Mesh, Self::Vertex, Self::Fragment, Self::Object],
        }
    }
}

/// A buffer slot a setter set: the address Metal's getter answers and the attribute stride form.
/// A slot no setter set, or one a reset cleared, is none ([`Bindings::buffers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    /// The buffer's address plus the low 32 bits of the offset set, or, for a nil buffer, the low
    /// 32 bits of the offset alone, as Metal keeps and answers it. Below
    /// [`BufferBinding::MIN_BUFFER_ADDRESS`] it names no buffer ([`BufferBinding::names_buffer`]);
    /// at or above it, it must identify registered device storage.
    pub address: u64,
    /// None uses the pipeline's static stride. The all-bits-set value is reserved for None.
    pub attribute_stride: Option<u64>,
}

/// A live host resource and its device-address interval, which
/// [`BufferBinding::resolve_buffer`] resolves a record's buffer address against. Registration
/// does not transfer ownership.
pub struct RegisteredBuffer<'a, T> {
    resource: &'a T,
    address: u64,
    length: u64,
}

/// The selected allocation and exact interior offset; no memory is copied or rebound.
#[non_exhaustive]
pub struct ResolvedBuffer<'a, T> {
    /// The registered resource whose interval holds the address.
    pub resource: &'a T,
    /// The address's offset from the start of the resource's interval.
    pub offset: u64,
    /// The bytes of the interval from that offset to its end.
    pub available_bytes: u64,
}

/// Why a buffer address does not resolve ([`RegisteredBuffer::new`],
/// [`BufferBinding::resolve_buffer`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressError {
    /// A registration whose address is below [`BufferBinding::MIN_BUFFER_ADDRESS`], which no
    /// binding names, whose length is zero, or whose interval passes the end of the address
    /// space.
    InvalidAllocation {
        /// The registered address.
        address: u64,
        /// The registered length.
        length: u64,
    },
    /// More than one registered interval holds the requested bytes at the address.
    Ambiguous {
        /// The binding's address.
        address: u64,
        /// The bytes requested from it.
        required_bytes: u64,
    },
    /// No registered interval holds the requested bytes at the address.
    Unavailable {
        /// The binding's address.
        address: u64,
        /// The bytes requested from it.
        required_bytes: u64,
    },
}

impl fmt::Display for AddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAllocation { address, length } => write!(
                f,
                "invalid registered buffer interval at {address:#x} with {length} bytes"
            ),
            Self::Ambiguous {
                address,
                required_bytes,
            } => write!(
                f,
                "multiple registered buffers cover {required_bytes} bytes at {address:#x}"
            ),
            Self::Unavailable {
                address,
                required_bytes,
            } => write!(
                f,
                "no registered buffer covers {required_bytes} bytes at {address:#x}"
            ),
        }
    }
}

impl std::error::Error for AddressError {}

impl<'a, T> RegisteredBuffer<'a, T> {
    /// Registers `resource` as holding `length` bytes from device address `address`.
    pub fn new(resource: &'a T, address: u64, length: u64) -> Result<Self, AddressError> {
        if address < BufferBinding::MIN_BUFFER_ADDRESS
            || length == 0
            || address.checked_add(length - 1).is_none()
        {
            return Err(AddressError::InvalidAllocation { address, length });
        }
        Ok(Self {
            resource,
            address,
            length,
        })
    }
}

impl BufferBinding {
    /// The lowest address that names a buffer, 2^32: every Metal buffer address lies at or
    /// above it (Apple silicon's, and the runtime's address space, `[2^32, 2^47)`), so a lower
    /// one is the offset a nil buffer was set at.
    pub const MIN_BUFFER_ADDRESS: u64 = 1 << 32;

    /// Whether the address names a buffer: not a nil buffer's offset.
    pub const fn names_buffer(&self) -> bool {
        self.address >= Self::MIN_BUFFER_ADDRESS
    }

    /// Find a unique live allocation covering the requested span without descriptor alignment rules.
    /// Multiple covering registrations are rejected rather than resolved by registration order.
    /// A nil buffer's binding stays unbound; the caller owns nullability, access, and shader
    /// alignment policy.
    /// A zero-byte span at the allocation end grants no permission to dereference that address.
    pub fn resolve_buffer<'a, T>(
        self,
        buffers: &[RegisteredBuffer<'a, T>],
        required_bytes: u64,
    ) -> Result<Option<ResolvedBuffer<'a, T>>, AddressError> {
        if !self.names_buffer() {
            return Ok(None);
        }
        let mut resolved = None;
        for buffer in buffers {
            let Some(offset) = self.address.checked_sub(buffer.address) else {
                continue;
            };
            let Some(available_bytes) = buffer.length.checked_sub(offset) else {
                continue;
            };
            if required_bytes <= available_bytes {
                if resolved.is_some() {
                    return Err(AddressError::Ambiguous {
                        address: self.address,
                        required_bytes,
                    });
                }
                resolved = Some(ResolvedBuffer {
                    resource: buffer.resource,
                    offset,
                    available_bytes,
                });
            }
        }
        resolved.map(Some).ok_or(AddressError::Unavailable {
            address: self.address,
            required_bytes,
        })
    }
}

/// Action to execute after applying the command's buffer, pipeline, and memory state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Action {
    /// No action: a reset command, or one that sets state only, which runs nothing.
    #[default]
    Empty,
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`: `grid` groups of `local_size`.
    DispatchThreadgroups {
        grid: [u32; 3],
        local_size: [u32; 3],
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`: `grid` threads in groups of
    /// `local_size`.
    DispatchThreads {
        grid: [u32; 3],
        local_size: [u32; 3],
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    DrawPrimitives {
        /// Metal primitive-type encoding; the replay backend must validate its supported topology.
        primitive_type: u32,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`. Its arguments take the stage-in region's words, which a render
    /// command never sets (`layout::DRAW_ARGUMENT_WORDS`).
    DrawIndexedPrimitives {
        /// Metal primitive-type encoding, as [`Action::DrawPrimitives`]'s.
        primitive_type: u32,
        index_count: u32,
        /// Metal's `MTLIndexType`: 0 for 16-bit indices, 1 for 32-bit ones; the replay backend
        /// must validate it.
        index_type: u32,
        /// The index buffer's address plus the offset of the first index.
        index_buffer: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`. Its arguments take the stage-in region's words,
    /// as [`Action::DrawIndexedPrimitives`]'s do.
    DrawPatches {
        control_points: u32,
        patch_start: u32,
        patch_count: u32,
        /// The patch index buffer's address plus its offset; zero: no patch index buffer.
        patch_index_buffer: u64,
        instance_count: u32,
        base_instance: u32,
        /// The tessellation factor buffer's address plus its offset.
        factor_buffer: u64,
        /// The factors' stride from one instance to the next, in bytes.
        factor_instance_stride: u32,
        /// The `f32` bits of the scale the factors are multiplied by: a kernel's `draw_patches`
        /// gives one, which Metal applies whatever the pipeline; the CPU's setter takes none, so
        /// 1.0.
        factor_scale: u32,
    },
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:
    /// tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: [`Action::DrawPatches`]'s arguments, in its
    /// words, and the control-point index buffer's address in the first two threadgroup lengths'
    /// (`layout::CONTROL_POINT_INDICES`), which a render command never sets.
    DrawIndexedPatches {
        control_points: u32,
        patch_start: u32,
        patch_count: u32,
        /// The patch index buffer's address plus its offset; zero: no patch index buffer.
        patch_index_buffer: u64,
        /// The control-point index buffer's address plus its offset; zero: none.
        control_point_index_buffer: u64,
        instance_count: u32,
        base_instance: u32,
        /// The tessellation factor buffer's address plus its offset.
        factor_buffer: u64,
        /// The factors' stride from one instance to the next, in bytes.
        factor_instance_stride: u32,
        /// The `f32` bits of the scale the factors are multiplied by; the CPU's setter takes
        /// none, so 1.0.
        factor_scale: u32,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: the grid's
    /// threadgroups (of the object function where the pipeline has one, else of the mesh
    /// function), then the threads of an object and of a mesh threadgroup. Its arguments take the
    /// stage-in region's words, as [`Action::DrawIndexedPrimitives`]'s do.
    DrawMeshThreadgroups {
        threadgroups: [u32; 3],
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: the grid's
    /// threads, then the threads of an object and of a mesh threadgroup, as
    /// [`Action::DrawMeshThreadgroups`]'s.
    DrawMeshThreads {
        threads: [u32; 3],
        object_threads: [u32; 3],
        mesh_threads: [u32; 3],
    },
}

impl Action {
    /// What a record holding the action, and no barrier, holds of what a command may write
    /// ([`CommandRecord::writes`] of such a record): its operation, and a draw's primitive type
    /// (every type for a private one, past Metal's five).
    pub fn writes(&self) -> crate::pipeline::CommandWrites {
        writes_of(self, false)
    }

    /// Whether the action's arguments take the stage-in region's words
    /// (`layout::DRAW_ARGUMENT_WORDS`): an indexed draw's, a patch draw's, an indexed patch
    /// draw's or a mesh draw's. A render command sets no stage-in region.
    pub(crate) fn takes_stage_in_words(&self) -> bool {
        matches!(
            self,
            Action::DrawIndexedPrimitives { .. }
                | Action::DrawPatches { .. }
                | Action::DrawIndexedPatches { .. }
                | Action::DrawMeshThreadgroups { .. }
                | Action::DrawMeshThreads { .. }
        )
    }
}

/// Pipeline and buffer state held by a command or by its parent encoder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bindings {
    /// Host-registered pipeline token. No token means no pipeline was set on this state source.
    pub pipeline: Option<NonZeroU64>,
    buffers: [[Option<BufferBinding>; BINDING_COUNT]; BUFFER_STAGES],
}

impl Bindings {
    /// The buffer slots of `stage`: each one a setter set, none where no setter did or a reset
    /// cleared it. A record holds none as an address and a stride of 0 and a set slot's stride
    /// form as all ones (the pipeline's) or the stride, the answers of Metal's getters on a buffer
    /// of dynamic attribute strides.
    pub fn buffers(&self, stage: BufferStage) -> &[Option<BufferBinding>; BINDING_COUNT] {
        &self.buffers[stage.index()]
    }

    pub fn buffers_mut(
        &mut self,
        stage: BufferStage,
    ) -> &mut [Option<BufferBinding>; BINDING_COUNT] {
        &mut self.buffers[stage.index()]
    }
}

/// The descriptor-selected source of one category of execution state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateSource {
    Command,
    Parent,
}

/// Explicit descriptor policy; pipeline and buffer inheritance are independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inheritance {
    pub pipeline: StateSource,
    pub buffers: StateSource,
}

/// Complete persistent state of one indirect command.
/// Reset restores this value's default; execution must not implicitly reset it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandRecord {
    pub action: Action,
    /// Prior commands must complete before this command executes.
    pub barrier: bool,
    pub bindings: Bindings,
    /// The threadgroup-memory lengths, as set: Metal keeps any length, a multiple of 16 or not,
    /// and past the device's maximum (the probe 1).
    pub threadgroup_memory_lengths: [u32; THREADGROUP_COUNT],
    /// The stage-in region (`setStageInRegion:`): origin x, y and z, then size x, y and z; all
    /// zero when the command sets none.
    pub stage_in_region: [u32; 6],
    /// The imageblock size (`setImageblockWidth:height:`): width, then height; zero when the
    /// command sets none.
    pub imageblock_size: [u16; 2],
    /// The pipeline a compute command's reset left it executing with: Metal's reset reports the
    /// pipeline cleared (`bindings.pipeline` is `None`) while the command's later dispatch runs
    /// it (measured on the M3 Max and the M4). A pipeline setter, or a copy from a command that
    /// reports one, replaces it. Never beside `bindings.pipeline`.
    pub retained_pipeline: Option<NonZeroU64>,
    /// Whether a kernel's encoder wrote the pipeline the command executes with (its pipeline
    /// setter, or its copy from a command whose pipeline one wrote), rather than a CPU setter:
    /// Metal's kernel resolves the ID it writes to the state holding it as it writes, where the
    /// CPU's setter names the state it is handed, so a replay that tells the two apart needs it.
    /// A CPU pipeline setter clears it; a compute command's reset keeps it with the pipeline it
    /// retains. Never without a pipeline.
    pub kernel_pipeline: bool,
}

/// Why a record was refused: its length, an unknown action, a barrier flag other than 0 or 1,
/// the all-ones attribute stride, which stands for the pipeline's static stride, a stage-in
/// region beside a draw whose arguments take its words, or a field's index past what a record
/// holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordError {
    Length {
        actual: usize,
    },
    // Only decoding finds these.
    Action {
        value: u32,
    },
    Barrier {
        value: u32,
    },
    ReservedStride,
    /// A stage-in region set beside an indexed or patch draw, whose arguments take its words
    /// (`layout::DRAW_ARGUMENT_WORDS`): a render command sets none.
    StageInRegion,
    /// A retained pipeline beside a set one, or a pipeline state word with a bit other than
    /// `layout::RETAINED` and `layout::KERNEL_PIPELINE`, or either bit without a pipeline.
    PipelineState,
    /// A threadgroup length set beside an indexed patch draw, whose control-point index buffer
    /// takes the first two lengths' words (`layout::CONTROL_POINT_INDICES`): a render command
    /// sets none.
    ThreadgroupMemoryLength,
    /// A field's index past what a record holds: a buffer slot
    /// at or past [`BINDING_COUNT`], or a threadgroup length at or past
    /// [`THREADGROUP_BUFFER_ARGUMENT_COUNT`].
    Index {
        index: usize,
        count: usize,
    },
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { actual } => write!(
                f,
                "indirect command record has {actual} bytes; expected {COMMAND_BYTES}"
            ),
            Self::Action { value } => write!(f, "unknown indirect command action {value}"),
            Self::Barrier { value } => write!(f, "invalid indirect command barrier flag {value}"),
            Self::ReservedStride => f.write_str(
                "all-bits-set attribute stride is reserved for the pipeline's static stride",
            ),
            Self::StageInRegion => {
                f.write_str("an indexed or patch draw's arguments take the stage-in region's words")
            }
            Self::PipelineState => f.write_str(
                "an indirect command retains a pipeline beside a set one, or its pipeline state \
                 word is malformed",
            ),
            Self::ThreadgroupMemoryLength => f.write_str(
                "an indexed patch draw's control-point index buffer takes the threadgroup \
                 lengths' first words",
            ),
            Self::Index { index, count } => write!(
                f,
                "indirect command field index {index} is past the record's {count}"
            ),
        }
    }
}

impl std::error::Error for RecordError {}

impl CommandRecord {
    /// What the record holds, as a route that wrote it states it (`pipeline::CommandWrites`): its
    /// action's operation and a draw's primitive type (a value past Metal's five counting as all
    /// of them), and its barrier. The pipeline state it names is the writer's to add: a CPU setter
    /// names one by the state itself.
    pub fn writes(&self) -> crate::pipeline::CommandWrites {
        writes_of(&self.action, self.barrier)
    }

    /// Encode a complete record without allocating. Invalid input leaves the destination unchanged.
    pub fn encode_into(&self, bytes: &mut [u8]) -> Result<(), RecordError> {
        require_length(bytes)?;
        if self
            .bindings
            .buffers
            .iter()
            .flatten()
            .flatten()
            .any(|binding| binding.attribute_stride == Some(u64::MAX))
        {
            return Err(RecordError::ReservedStride);
        }
        // A draw whose arguments take the stage-in region's words has none.
        let wide = self.action.takes_stage_in_words();
        if wide && self.stage_in_region != [0; 6] {
            return Err(RecordError::StageInRegion);
        }
        // An indexed patch draw's control-point index buffer takes the first two threadgroup
        // lengths' words.
        let control_point_indices = match self.action {
            Action::DrawIndexedPatches {
                control_point_index_buffer,
                ..
            } => Some(control_point_index_buffer),
            _ => None,
        };
        if control_point_indices.is_some() && self.threadgroup_memory_lengths[..2] != [0; 2] {
            return Err(RecordError::ThreadgroupMemoryLength);
        }
        if self.bindings.pipeline.is_some() && self.retained_pipeline.is_some() {
            return Err(RecordError::PipelineState);
        }
        let pipeline = self.bindings.pipeline.or(self.retained_pipeline);
        if self.kernel_pipeline && pipeline.is_none() {
            return Err(RecordError::PipelineState);
        }
        bytes.fill(0);
        let (tag, arguments) = action_words(&self.action);
        put_word(bytes, 0, tag);
        put_word(bytes, 1, u32::from(self.barrier));
        // The pipeline word holds the pipeline the command executes with; the state word says
        // whether the read-back reports it, and whether a kernel wrote it.
        put_u64(bytes, 2, pipeline.map_or(0, NonZeroU64::get));
        let retained = match self.retained_pipeline {
            Some(_) => layout::RETAINED,
            None => 0,
        };
        let kernel = match self.kernel_pipeline {
            true => layout::KERNEL_PIPELINE,
            false => 0,
        };
        put_word(bytes, layout::STATE as usize / 4, retained | kernel);
        let first = layout::ARGUMENTS as usize / 4;
        for (index, value) in arguments.into_iter().enumerate() {
            put_word(bytes, first + index, value);
        }
        for (index, binding) in self.bindings.buffers.iter().flatten().enumerate() {
            let word = HEADER_WORDS + index * BUFFER_WORDS;
            let (address, stride) = binding.map_or((0, 0), |binding| {
                (
                    binding.address,
                    binding.attribute_stride.unwrap_or(u64::MAX),
                )
            });
            put_u64(bytes, word, address);
            put_u64(bytes, word + 2, stride);
        }
        for (index, &length) in self.threadgroup_memory_lengths.iter().enumerate() {
            put_word(bytes, THREADGROUP_WORD + index, length);
        }
        if let Some(address) = control_point_indices {
            put_u64(bytes, layout::CONTROL_POINT_INDICES as usize / 4, address);
        }
        if !wide {
            for (index, &value) in self.stage_in_region.iter().enumerate() {
                put_word(bytes, layout::STAGE_IN as usize / 4 + index, value);
            }
        }
        let imageblock = layout::IMAGEBLOCK as usize;
        for (index, &value) in self.imageblock_size.iter().enumerate() {
            bytes[imageblock + 2 * index..imageblock + 2 * index + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

/// What a record of `action` and `barrier` holds ([`CommandRecord::writes`]): its action's
/// command kind too, a dispatch's compute and a draw's render; a record of no action names none.
pub(crate) fn writes_of(action: &Action, barrier: bool) -> crate::pipeline::CommandWrites {
    use crate::pipeline::CommandWrites;
    let draw = |operations: u32, primitive_type: u32| CommandWrites {
        operations,
        primitive_types: if primitive_type < 5 {
            1 << primitive_type
        } else {
            CommandWrites::ALL_PRIMITIVE_TYPES
        },
        kinds: CommandWrites::RENDER_COMMANDS,
    };
    let only = |operations: u32, kinds: u8| CommandWrites {
        operations,
        primitive_types: 0,
        kinds,
    };
    let mut writes = match *action {
        Action::Empty => CommandWrites::NONE,
        Action::DispatchThreadgroups { .. } | Action::DispatchThreads { .. } => {
            only(CommandWrites::DISPATCH, CommandWrites::COMPUTE_COMMANDS)
        }
        Action::DrawPrimitives { primitive_type, .. } => draw(CommandWrites::DRAW, primitive_type),
        Action::DrawIndexedPrimitives { primitive_type, .. } => {
            draw(CommandWrites::DRAW_INDEXED, primitive_type)
        }
        Action::DrawPatches { .. } | Action::DrawIndexedPatches { .. } => {
            only(CommandWrites::DRAW_PATCHES, CommandWrites::RENDER_COMMANDS)
        }
        Action::DrawMeshThreadgroups { .. } | Action::DrawMeshThreads { .. } => {
            only(CommandWrites::DRAW_MESH, CommandWrites::RENDER_COMMANDS)
        }
    };
    if barrier {
        writes.operations |= CommandWrites::BARRIER;
    }
    writes
}

/// One field of a command's record, as one CPU setter writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordField {
    /// The action ([`CommandRecord::action`]): a dispatch or a draw.
    Action(Action),
    /// The barrier flag ([`CommandRecord::barrier`]).
    Barrier(bool),
    /// The pipeline the command names ([`Bindings::pipeline`]), set: a pipeline a reset retained
    /// goes ([`CommandRecord::retained_pipeline`]).
    Pipeline(NonZeroU64),
    /// Slot `slot` of `stage`'s buffers ([`Bindings::buffers`]), set, or unset with `None`.
    Buffer {
        stage: BufferStage,
        slot: usize,
        binding: Option<BufferBinding>,
    },
    /// Threadgroup-memory length `index` ([`CommandRecord::threadgroup_memory_lengths`]).
    ThreadgroupMemoryLength { index: usize, length: u32 },
    /// The stage-in region ([`CommandRecord::stage_in_region`]).
    StageInRegion([u32; 6]),
    /// The imageblock size ([`CommandRecord::imageblock_size`]).
    ImageblockSize([u16; 2]),
}

impl CommandRecord {}

fn require_length(bytes: &[u8]) -> Result<(), RecordError> {
    if bytes.len() == COMMAND_BYTES {
        Ok(())
    } else {
        Err(RecordError::Length {
            actual: bytes.len(),
        })
    }
}

fn put_word(bytes: &mut [u8], word: usize, value: u32) {
    bytes[word * 4..word * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], word: usize, value: u64) {
    bytes[word * 4..word * 4 + 8].copy_from_slice(&value.to_le_bytes());
}

/// The tag and the argument words ([`layout::DRAW_ARGUMENT_WORDS`], zero past the action's) a
/// record of `action` holds.
fn action_words(action: &Action) -> (u32, [u32; layout::DRAW_ARGUMENT_WORDS]) {
    match *action {
        Action::Empty => (0, arguments([])),
        Action::DispatchThreadgroups { grid, local_size } => (
            1,
            arguments([
                grid[0],
                grid[1],
                grid[2],
                local_size[0],
                local_size[1],
                local_size[2],
            ]),
        ),
        Action::DispatchThreads { grid, local_size } => (
            2,
            arguments([
                grid[0],
                grid[1],
                grid[2],
                local_size[0],
                local_size[1],
                local_size[2],
            ]),
        ),
        Action::DrawPrimitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => (
            3,
            arguments([
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            ]),
        ),
        Action::DrawIndexedPrimitives {
            primitive_type,
            index_count,
            index_type,
            index_buffer,
            instance_count,
            base_vertex,
            base_instance,
        } => (
            4,
            arguments([
                primitive_type,
                index_count,
                index_type,
                instance_count,
                index_buffer as u32,
                (index_buffer >> 32) as u32,
                base_vertex as u32,
                base_instance,
            ]),
        ),
        Action::DrawPatches {
            control_points,
            patch_start,
            patch_count,
            patch_index_buffer,
            instance_count,
            base_instance,
            factor_buffer,
            factor_instance_stride,
            factor_scale,
        } => (
            5,
            arguments([
                control_points,
                patch_start,
                patch_count,
                instance_count,
                patch_index_buffer as u32,
                (patch_index_buffer >> 32) as u32,
                factor_buffer as u32,
                (factor_buffer >> 32) as u32,
                base_instance,
                factor_instance_stride,
                factor_scale,
            ]),
        ),
        Action::DrawIndexedPatches {
            control_points,
            patch_start,
            patch_count,
            patch_index_buffer,
            control_point_index_buffer: _,
            instance_count,
            base_instance,
            factor_buffer,
            factor_instance_stride,
            factor_scale,
        } => (
            6,
            arguments([
                control_points,
                patch_start,
                patch_count,
                instance_count,
                patch_index_buffer as u32,
                (patch_index_buffer >> 32) as u32,
                factor_buffer as u32,
                (factor_buffer >> 32) as u32,
                base_instance,
                factor_instance_stride,
                factor_scale,
            ]),
        ),
        Action::DrawMeshThreadgroups {
            threadgroups: grid,
            object_threads,
            mesh_threads,
        }
        | Action::DrawMeshThreads {
            threads: grid,
            object_threads,
            mesh_threads,
        } => (
            match action {
                Action::DrawMeshThreadgroups { .. } => 7,
                _ => 8,
            },
            arguments([
                grid[0],
                grid[1],
                grid[2],
                object_threads[0],
                object_threads[1],
                object_threads[2],
                mesh_threads[0],
                mesh_threads[1],
                mesh_threads[2],
            ]),
        ),
    }
}

/// An action's argument words, zero past them ([`layout::DRAW_ARGUMENT_WORDS`]).
fn arguments<const N: usize>(words: [u32; N]) -> [u32; layout::DRAW_ARGUMENT_WORDS] {
    let mut all = [0; layout::DRAW_ARGUMENT_WORDS];
    all[..N].copy_from_slice(&words);
    all
}

impl CommandEncoderOperation {
    /// The descriptor policy this operation's write requires of the header ([`Policy`]).
    pub(crate) fn policy(self) -> Policy {
        let family = |render: bool| if render { RENDER } else { COMPUTE };
        let plain = |required| Policy {
            required,
            forbidden: 0,
            any: 0,
            binding: None,
            dynamic_stride: false,
            dropped_on: 0,
        };
        // A draw needs a render descriptor whose records hold its kind.
        let draw = |kind: DrawKind| Policy {
            any: kind.holders(),
            ..plain(RENDER)
        };
        match self {
            Self::Buffer { stage, .. } => Policy {
                required: family(stage != BufferStage::Kernel),
                forbidden: INHERIT_BUFFERS,
                any: 0,
                binding: Some(match stage {
                    BufferStage::Kernel => 0,
                    BufferStage::Vertex => 1,
                    BufferStage::Fragment => 2,
                    // A render descriptor's object and mesh counts take the bytes of the kernel
                    // and threadgroup counts, which a compute descriptor's alone hold.
                    BufferStage::Object => 0,
                    BufferStage::Mesh => 3,
                }),
                dynamic_stride: matches!(stage, BufferStage::Kernel | BufferStage::Vertex),
                dropped_on: 0,
            },
            // Every compute descriptor allows a dispatch kind, and its records hold either.
            Self::Dispatch { .. } => plain(COMPUTE),
            Self::Pipeline { render } => Policy {
                forbidden: INHERIT_PIPELINE,
                ..plain(family(render))
            },
            Self::Barrier { render, .. } => plain(family(render)),
            Self::ThreadgroupMemory => Policy {
                binding: Some(3),
                ..plain(COMPUTE)
            },
            Self::Draw => Policy {
                dropped_on: DRAW_MESH_THREADGROUPS | DRAW_MESH_THREADS,
                ..draw(DrawKind::Primitives)
            },
            Self::DrawIndexed => draw(DrawKind::Indexed),
            Self::DrawPatches | Self::DrawIndexedPatches => draw(DrawKind::Patches),
            Self::DrawMesh { .. } => draw(DrawKind::Mesh),
            Self::Reset { render } => plain(family(render)),
            Self::StageInRegion | Self::ImageblockSize | Self::Copy => plain(COMPUTE),
        }
    }
}

#[cfg(test)]
mod tests;
