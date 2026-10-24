//! Indirect command buffers (`MTLIndirectCommandBuffer`, `MTLIndirectComputeCommand`).
//!
//! - **Storage.** An indirect command buffer is one runtime buffer in the record format of
//!   `bind::indirect_commands`: a header, then one record per command, so a kernel encodes the same
//!   records the CPU does. Its `gpu_resource_id` is the ID Metal's pool hands out, of the indirect
//!   command buffers' own space, or the one its descriptor places it at (an M3 Max and an M4,
//!   compute and render buffers in one space beside textures', samplers', tables' and pipeline
//!   states'; the pool and the placement). The buffer holds the ID until it and every command that
//!   holds it are gone, a blit's command holding only its records. The records lie at
//!   `records_address` of the ID, outside the Metal address space (`address.rs`), where a
//!   translated kernel that reads the ID from an argument buffer reaches the buffer last made with
//!   the ID. The buffer is host-visible whatever storage mode the program asks for; the object
//!   answers the mode asked for, and the CPU encodes and reads back a `Private` buffer's commands
//!   as a `Shared` one's, as plain Metal does.
//! - **CPU encoding.** [`IndirectCommandBuffer::indirect_compute_command_at`] gives a compute
//!   command's setters, which write its record as Metal's do (an M3 Max on macOS 26.7 and an M4 on
//!   27.0.1): a buffer's address plus its offset's low 32 bits as one address; threadgroup lengths
//!   as set; sizes and stage-in regions to their low 32 bits.
//!   [`IndirectCommandBuffer::indirect_render_command_at`] gives a render command's: its render
//!   pipeline state, vertex and fragment buffers, and a draw of primitives, each value's low 32
//!   bits; and an indexed draw, its index buffer as one address, and a patch draw, its patch index
//!   and tessellation factor buffers so; and (probes) an indexed patch draw, its control-point
//!   index buffer so too; and (a Metal probe) its object and mesh buffers, a mesh draw of
//!   threadgroups or threads, and its barrier. What Metal's validation stops, or ends the process
//!   for, is refused `Invalid` ([`Problem::IndirectCommandBuffer`]). A command names the pipeline
//!   state set on it without holding it, as Metal's keeps no reference: the program's last
//!   release of the state gives its ID back while the record still names it. The command keeps
//!   what the state runs, and runs that while its record names the ID, as Metal's does.
//! - **Render states a command does not inherit** (an M4 with macOS 27.0.1, each case plain and
//!   under Metal's validation). macOS 26 lets a descriptor leave a render command's depth/stencil
//!   state, depth bias, depth clip mode, cull mode, front-facing winding and triangle fill mode to
//!   the command (`inherit_depth_stencil_state` and the rest, `false`), which its setters then set.
//!   On the M4 none of it is supported: Metal's validation stops the creation of such a buffer but
//!   for the winding ("must be inherited on this device"), plain Metal makes it and its commands
//!   draw with the encoder's states as an inheriting buffer's do, and every setter raises
//!   `APINotSupported`, ending the process, whatever the buffer inherits (its validation stops the
//!   setter first on a buffer that inherits the state). Its Apple10 family alone lists the feature
//!   ("Indirect command buffer support for raster and depth/stencil state"). So on a device that
//!   does not claim Apple10 the runtime makes such a buffer, whose commands draw with the encoder's
//!   states, and refuses each setter (`Invalid`, [`IndirectCommandBufferProblem::RenderState`]); on
//!   one that claims it, whose commands would draw with their own, both are
//!   `NotImplemented(IndirectRenderCommands)`. A buffer that supports color attachment mapping
//!   executes as any other in an encoder that maps none, the runtime's every encoder, as Metal's
//!   does. An object threadgroup length a render command sets is taken at any index, as plain Metal
//!   takes one past the descriptor's count, and kept by none: an executed command whose mesh draw's
//!   object function reads one is `NotImplemented(IndirectRenderCommands)`. Apple's private
//!   properties (the same M4): stencil reference values, depth test bounds, scissor rectangles,
//!   viewports and the blend color left to the command are taken as the render states above
//!   (validation stops each "must be inherited on this device"; plain Metal draws with the
//!   encoder's viewport, scissor, cull mode and winding); the scissor rectangle and viewport counts
//!   and `allowOverrideRenderStates` are kept, and change nothing a command draws.
//! - **Read-back.** [`IndirectCommandBuffer::commands`] decodes a range of records into typed
//!   commands, and each command's getters ([`IndirectComputeCommand::command_type`],
//!   [`IndirectRenderCommand::draw`] and the rest) read one field of its record: runtime
//!   extensions standing for Metal's private getters.
//! - **Execution.** A compute encoder's `execute_commands_in_buffer` runs a range of the records
//!   as they stand when the execution's batch is submitted, or for records the GPU may write, once
//!   the work before it completed ([`indirect_execute`](super::indirect_execute)); it refuses a
//!   buffer of render commands, which a render encoder's runs (`render/executions.rs`).
//! - **Creation.** The records of every command must fit one buffer of the device's
//!   `max_buffer_length`; a larger buffer is refused, naming the limit. The record format holds
//!   every render command type, mesh draws among them, whose object and mesh buffers a command sets
//!   as its vertex and fragment buffers; a buffer of either mesh kind holds both mesh draws, as
//!   Metal's does.

use super::address::{AddressPlacement, Place, RecordsId};
use super::buffer::{Buffer, Modes};
use super::degradation::Degradation;
use super::depth_stencil::DepthStencilState;
use super::device::lock;
use super::error::{internal, invalid, Error, MemoryKind, Problem, RuntimeGap};
use super::families::GpuFamily;
use super::indirect_execute::Unforeseen;
use super::memory::MemoryClass;
use super::pipeline::{
    ComputePipelineState, ComputeProgram, RenderPipelineState, RenderProgram, RenderStateKind,
    WeakComputePipelineState, WeakRenderPipelineState,
};
use super::released_programs::{KeptProgram, KeptPrograms};
use super::render::{index_size, DrawProblem};
use super::render_values::DrawPrimitiveType;
use super::resource_ids::{records_placement, HolderThen, IdPlacement, ResourceId, StateKind};
use super::unique_identifier::{degrade_identifier, state_value_degradation, UniqueIdentifier};
use super::values::{Origin, Region, Size};
use super::written_commands::{Named, WrittenCommands};
use super::{
    CpuCacheMode, Device, GpuAddress, HazardTrackingMode, PurgeableState, ResourceOptions,
    StorageMode,
};
use metal2vulkan::bind::indirect_commands::{
    Action, BufferBinding, BufferStage, CommandBuffer, CommandBufferError, CommandDescriptor,
    CommandRecord, ComputeDispatch, Inheritance, RecordError, RecordField, RecordView, RenderDraws,
    StateSource, BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
    THREADGROUP_BUFFER_ARGUMENT_COUNT,
};
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::{
    CullMode, DepthClipMode, IndexType, IndirectCommandType, PrimitiveType, TriangleFillMode,
    Winding,
};
use metal2vulkan::pipeline::CommandWrites;
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The threadgroup lengths a fresh descriptor binds, as Metal's answers.
const DEFAULT_THREADGROUP_LENGTHS: u32 = 31;

/// The most kernel, vertex or fragment buffers, or threadgroup lengths, a descriptor may bind.
/// Metal's validation stops more than 31, but plain Metal makes a buffer of more, whose commands
/// keep a buffer set at any index below the count and whose functions bind none past 30 (32, 33 and
/// 64 kernel buffers, 32 threadgroup lengths, 32 and 33 vertex and 33 fragment buffers, each set at
/// its last index and run). Its storage grew with the count to 65536 kernel buffers (1024 vertex
/// and fragment buffers and threadgroup lengths, as kernel buffers grow), but a buffer of 2^31
/// was made as small as one of few, and 2^32 - 1 ended its process: past 65536 its storage is not
/// defined.
const MOST_BINDS: u32 = 65536;

/// `count` bindings, of which the record holds `capacity`: the ones a command's record holds.
fn held(count: u32, capacity: u32) -> u32 {
    count.min(capacity)
}

/// The buffer bindings a command's record holds of each stage.
const RECORD_BUFFERS: u32 = BINDING_COUNT as u32;

/// The draw command types.
const DRAW_TYPES: IndirectCommandType = IndirectCommandType::DRAW
    .union(IndirectCommandType::DRAW_INDEXED)
    .union(IndirectCommandType::DRAW_PATCHES)
    .union(IndirectCommandType::DRAW_INDEXED_PATCHES);

/// The mesh draw command types.
const MESH_TYPES: IndirectCommandType =
    IndirectCommandType::DRAW_MESH_THREADGROUPS.union(IndirectCommandType::DRAW_MESH_THREADS);

/// The render command types.
const RENDER_TYPES: IndirectCommandType = DRAW_TYPES.union(MESH_TYPES);

/// The compute command types.
const COMPUTE_TYPES: IndirectCommandType = IndirectCommandType::CONCURRENT_DISPATCH
    .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);

/// `MTLIndirectCommandBufferDescriptor`, its fields Metal's defaults until set: no command
/// types, neither the pipeline state nor the buffers inherited, no buffers of any stage, 31
/// threadgroup lengths, no dynamic attribute strides (as a fresh descriptor answers on an M3 Max
/// and an M4), and macOS 26's render states inherited, without color attachment mapping (the
/// SDK's defaults); Apple's private render states inherited, one scissor rectangle and one
/// viewport, and no render state overrides (a fresh descriptor's answers on an M4 under macOS
/// 27.0.1, plain and validated alike).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    /// `commandTypes`.
    pub command_types: IndirectCommandType,
    /// `inheritPipelineState`: commands run the executing encoder's pipeline state.
    pub inherit_pipeline_state: bool,
    /// `inheritBuffers`: commands run with the executing encoder's buffers.
    pub inherit_buffers: bool,
    /// `maxVertexBufferBindCount`.
    pub max_vertex_buffer_bind_count: u32,
    /// `maxFragmentBufferBindCount`.
    pub max_fragment_buffer_bind_count: u32,
    /// `maxObjectBufferBindCount`.
    pub max_object_buffer_bind_count: u32,
    /// `maxMeshBufferBindCount`.
    pub max_mesh_buffer_bind_count: u32,
    /// `maxObjectThreadgroupMemoryBindCount`: the object threadgroup lengths a render command
    /// sets ([`IndirectRenderCommand::set_object_threadgroup_memory_length`]).
    pub max_object_threadgroup_memory_bind_count: u32,
    /// `maxKernelBufferBindCount`.
    pub max_kernel_buffer_bind_count: u32,
    /// `maxKernelThreadgroupMemoryBindCount`.
    pub max_kernel_threadgroup_memory_bind_count: u32,
    /// `supportDynamicAttributeStride`.
    pub support_dynamic_attribute_stride: bool,
    /// `inheritDepthStencilState`: commands draw with the executing encoder's depth/stencil
    /// state. `false` changes nothing a command draws on a device that does not claim Apple10:
    /// its commands draw with the encoder's state still, as plain Metal's do on an M4, where
    /// Metal's validation stops the buffer's creation ("must be inherited on
    /// this device"). A device that claims Apple10 refuses it (the module docs).
    pub inherit_depth_stencil_state: bool,
    /// `inheritDepthBias`: as `inherit_depth_stencil_state`, for the depth bias.
    pub inherit_depth_bias: bool,
    /// `inheritDepthClipMode`: as `inherit_depth_stencil_state`, for the depth clip mode.
    pub inherit_depth_clip_mode: bool,
    /// `inheritCullMode`: as `inherit_depth_stencil_state`, for the cull mode.
    pub inherit_cull_mode: bool,
    /// `inheritFrontFacingWinding`: as `inherit_depth_stencil_state`, for the front-facing
    /// winding, whose `false` Metal's validation takes.
    pub inherit_front_facing_winding: bool,
    /// `inheritTriangleFillMode`: as `inherit_depth_stencil_state`, for the triangle fill mode.
    pub inherit_triangle_fill_mode: bool,
    /// `supportColorAttachmentMapping`: whether the buffer may execute in an encoder that maps
    /// its color attachments. The runtime's encoders map none
    /// (`RuntimeGap::ColorAttachmentMapping`), and a buffer that supports it executes in one that
    /// maps none as any other does, as Metal's does on an M4 with and without its validation.
    pub support_color_attachment_mapping: bool,
    /// `inheritStencilReferenceValues`, Apple's private property: as
    /// `inherit_depth_stencil_state`, for the stencil reference values. On an M4 plain Metal
    /// makes a buffer that does not inherit them, whose commands draw with the encoder's states,
    /// and its validation stops the creation ("must be inherited on this device"), as for the four
    /// below.
    pub inherit_stencil_reference_values: bool,
    /// `inheritDepthTestBounds`, private: as `inherit_stencil_reference_values`, for the depth
    /// test bounds.
    pub inherit_depth_test_bounds: bool,
    /// `inheritScissorRects`, private: as `inherit_stencil_reference_values`, for the scissor
    /// rectangles.
    pub inherit_scissor_rects: bool,
    /// `inheritViewports`, private: as `inherit_stencil_reference_values`, for the viewports.
    pub inherit_viewports: bool,
    /// `inheritBlendColor`, private: as `inherit_stencil_reference_values`, for the blend color.
    pub inherit_blend_color: bool,
    /// `maxScissorRectCount`, private: kept. A buffer made with another count draws with the
    /// encoder's scissor rectangles, plain and validated.
    pub max_scissor_rect_count: u64,
    /// `maxViewportCount`, private: as `max_scissor_rect_count`, for the viewports.
    pub max_viewport_count: u64,
    /// `allowOverrideRenderStates`, private (an `NSInteger`): as `max_scissor_rect_count`.
    pub allow_override_render_states: i64,
    /// Where the buffer goes in the indirect command buffers' space of IDs: an ID Metal's pool
    /// hands out, or (`resourceIndex`, Apple's private property, a runtime extension as for
    /// textures and sampler states) the index's low 32 bits, plain allocation where those are 0.
    /// A placed buffer's ID is not the pool's, so plain allocation may hand it out again; the
    /// buffer made last with an ID is the one a kernel encoding through the ID reaches
    /// (`address.rs`).
    pub placement: IdPlacement,
}

impl Default for IndirectCommandBufferDescriptor {
    fn default() -> IndirectCommandBufferDescriptor {
        IndirectCommandBufferDescriptor {
            command_types: IndirectCommandType::empty(),
            inherit_pipeline_state: false,
            inherit_buffers: false,
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            max_object_threadgroup_memory_bind_count: 0,
            max_kernel_buffer_bind_count: 0,
            max_kernel_threadgroup_memory_bind_count: DEFAULT_THREADGROUP_LENGTHS,
            support_dynamic_attribute_stride: false,
            inherit_depth_stencil_state: true,
            inherit_depth_bias: true,
            inherit_depth_clip_mode: true,
            inherit_cull_mode: true,
            inherit_front_facing_winding: true,
            inherit_triangle_fill_mode: true,
            support_color_attachment_mapping: false,
            inherit_stencil_reference_values: true,
            inherit_depth_test_bounds: true,
            inherit_scissor_rects: true,
            inherit_viewports: true,
            inherit_blend_color: true,
            max_scissor_rect_count: 1,
            max_viewport_count: 1,
            allow_override_render_states: 0,
            placement: IdPlacement::Any,
        }
    }
}

impl IndirectCommandBufferDescriptor {
    /// A descriptor with Metal's defaults.
    pub fn new() -> IndirectCommandBufferDescriptor {
        IndirectCommandBufferDescriptor::default()
    }

    /// Whether commands inherit every render state macOS 26 lets a buffer leave to them, and
    /// every one Apple's private properties do.
    fn inherits_render_states(&self) -> bool {
        self.inherit_depth_stencil_state
            && self.inherit_depth_bias
            && self.inherit_depth_clip_mode
            && self.inherit_cull_mode
            && self.inherit_front_facing_winding
            && self.inherit_triangle_fill_mode
            && self.inherit_stencil_reference_values
            && self.inherit_depth_test_bounds
            && self.inherit_scissor_rects
            && self.inherit_viewports
            && self.inherit_blend_color
    }

    /// The record format's descriptor and inheritance of this descriptor, or what refuses it
    /// for `call`.
    fn records(&self, call: &'static str) -> Result<(CommandDescriptor, Inheritance), Error> {
        let problem = |problem| invalid(call, Problem::IndirectCommandBuffer(problem));
        let types = self.command_types;
        // No command types make a buffer of compute records no command reaches (the buffer's
        // calls refuse them, `check_command_types`).
        // Draws of any kind beside dispatches make a buffer of compute commands, as plain
        // Metal's: its compute commands encode and run, and its process ends at a render
        // command's first setter.
        let mixed = types.intersects(RENDER_TYPES) && types.intersects(COMPUTE_TYPES);
        let source = |inherited| {
            if inherited {
                StateSource::Parent
            } else {
                StateSource::Command
            }
        };
        let inheritance = Inheritance {
            pipeline: source(self.inherit_pipeline_state),
            buffers: source(self.inherit_buffers),
        };
        if !mixed && types.intersects(RENDER_TYPES) {
            for count in [
                self.max_vertex_buffer_bind_count,
                self.max_fragment_buffer_bind_count,
                self.max_object_buffer_bind_count,
                self.max_mesh_buffer_bind_count,
            ] {
                if count > MOST_BINDS {
                    return Err(problem(IndirectCommandBufferProblem::BindCount {
                        count,
                        limit: MOST_BINDS,
                    }));
                }
            }
            return Ok((
                CommandDescriptor::Render {
                    draws: RenderDraws {
                        primitives: types.contains(IndirectCommandType::DRAW),
                        indexed: types.contains(IndirectCommandType::DRAW_INDEXED),
                        patches: types.contains(IndirectCommandType::DRAW_PATCHES),
                        indexed_patches: types.contains(IndirectCommandType::DRAW_INDEXED_PATCHES),
                        mesh_threadgroups: types
                            .contains(IndirectCommandType::DRAW_MESH_THREADGROUPS),
                        mesh_threads: types.contains(IndirectCommandType::DRAW_MESH_THREADS),
                    },
                    max_vertex_buffer_bind_count: held(
                        self.max_vertex_buffer_bind_count,
                        RECORD_BUFFERS,
                    ),
                    max_fragment_buffer_bind_count: held(
                        self.max_fragment_buffer_bind_count,
                        RECORD_BUFFERS,
                    ),
                    max_object_buffer_bind_count: held(
                        self.max_object_buffer_bind_count,
                        RECORD_BUFFERS,
                    ),
                    max_mesh_buffer_bind_count: held(
                        self.max_mesh_buffer_bind_count,
                        RECORD_BUFFERS,
                    ),
                    support_dynamic_attribute_stride: self.support_dynamic_attribute_stride,
                },
                inheritance,
            ));
        }
        for count in [
            self.max_kernel_buffer_bind_count,
            self.max_kernel_threadgroup_memory_bind_count,
        ] {
            if count > MOST_BINDS {
                return Err(problem(IndirectCommandBufferProblem::BindCount {
                    count,
                    limit: MOST_BINDS,
                }));
            }
        }
        let dispatch = match (
            types.contains(IndirectCommandType::CONCURRENT_DISPATCH),
            types.contains(IndirectCommandType::CONCURRENT_DISPATCH_THREADS),
        ) {
            (true, true) => ComputeDispatch::Both,
            (true, false) => ComputeDispatch::Threadgroups,
            _ => ComputeDispatch::Threads,
        };
        Ok((
            CommandDescriptor::Compute {
                dispatch,
                max_buffer_bind_count: held(self.max_kernel_buffer_bind_count, RECORD_BUFFERS),
                max_threadgroup_memory_bind_count: held(
                    self.max_kernel_threadgroup_memory_bind_count,
                    THREADGROUP_BUFFER_ARGUMENT_COUNT,
                ),
                support_dynamic_attribute_stride: self.support_dynamic_attribute_stride,
            },
            inheritance,
        ))
    }
}

/// What an indirect command buffer call did wrong (`Problem::IndirectCommandBuffer`): each is
/// what Metal's API validation stops, or what ends Metal's process.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectCommandBufferProblem {
    /// A command, a CPU reset, a read-back or an argument buffer's encoding of a buffer of no
    /// command types, or its blit copy to or from a buffer of command types. Plain Metal makes
    /// such a buffer, where its validation stops it, and executes, resets, optimizes and copies
    /// it to another such as nothing, as the runtime does, but its process ends at a command's
    /// first setter and at its CPU reset; the rest was not measured.
    NoCommandTypes,
    /// A descriptor with more kernel, vertex, fragment, object or mesh buffers, or kernel
    /// threadgroup lengths, than 65536, past which plain Metal's storage is not defined (a buffer of 2^31 kernel
    /// buffers was made as small as one of few, 2^32 - 1 ended its process).
    /// Metal's validation stops more than 31, but plain Metal makes a buffer of up to 65536, as
    /// the runtime does.
    BindCount {
        /// The count the descriptor asks for.
        count: u32,
        /// The most the runtime binds.
        limit: u32,
    },
    /// A command at or past the buffer's size.
    CommandIndex {
        /// The command's index.
        index: u64,
        /// The buffer's size, in commands.
        size: u64,
    },
    /// A range of commands not inside the buffer: for `reset_with_range`, one that starts at or
    /// past its size (an empty one too, but the empty one at the size) or ends past it; for the
    /// read-back, one that ends past it.
    Range {
        /// The range's first command.
        location: u64,
        /// The range's commands.
        length: u64,
        /// The buffer's size, in commands.
        size: u64,
    },
    /// A pipeline state set on a command of a buffer that inherits it (this ends Metal's
    /// process).
    InheritedPipelineState,
    /// A pipeline state made without `supportIndirectCommandBuffers` (this ends Metal's
    /// process).
    PipelineWithoutSupport,
    /// A draw of a kind (`DRAW_INDEXED`, `DRAW_PATCHES` or `DRAW_INDEXED_PATCHES`, a probe) whose
    /// record the buffer's records cannot hold: an indexed draw on a buffer of draws of primitives
    /// alone (plain Metal's GPU faults), a patch draw of either kind on one of neither patch kind
    /// (plain Metal ends its process), a draw of primitives or an indexed one on a buffer of mesh
    /// draws alone (Metal's process ends, with and without its validation), a mesh draw on one of
    /// no mesh kind (a kernel's faults Metal's GPU on a buffer of draws of primitives, a Metal
    /// probe). Metal's validation stops any kind the command types lack ("drawIndexedPrimitives is
    /// not enabled on this indirect command buffer"), but plain Metal runs one their records hold,
    /// as the runtime does.
    CommandType(IndirectCommandType),
    /// An execution range read from a buffer (`execute_commands_in_buffer_indirect`) at an
    /// `offset` that is not a multiple of 4 bytes, or whose 8 bytes run past the buffer's
    /// `length`. Metal's validation stops neither, but what it reads is not defined: its GPU
    /// faulted the command buffer at an offset of 2.
    RangeBuffer {
        /// The range's offset in the buffer.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
    /// A compute command of a buffer of render command types, or a render command of one of
    /// compute types (the first ends Metal's process, validation stops the second).
    CommandKind,
    /// An executed command's kernel or vertex buffer at `index` set without an attribute stride
    /// (the plain setter, the stride form at `MTLAttributeStrideStatic`, or a kernel's plain
    /// `set_kernel_buffer`), where its pipeline state's layout of `MTLBufferLayoutStrideDynamic`
    /// reads the stride. Metal reads such a buffer at stride 2^32 - 1, with no bounds check, so
    /// past element 0 the fetch reads whatever memory lies 4 GiB on (measured on an M3 Max and an
    /// M4, with and without the debug layer, which stops neither).
    UnstridedBuffer {
        /// The buffer's index.
        index: u32,
    },
    /// A render state set on a render command (`set_depth_stencil_state`, `set_depth_bias`,
    /// `set_depth_clip_mode`, `set_cull_mode`, `set_front_facing_winding`,
    /// `set_triangle_fill_mode`) on a device that does not claim Apple10. Plain Metal raises
    /// `APINotSupported` at each, ending its process, on an M4 (macOS 27.0.1), whether the
    /// buffer inherits the state or not; its validation stops the setter on a buffer that
    /// inherits the state, and the creation of one that does not.
    RenderState,
    /// An object threadgroup length set on a render command of a buffer whose command types
    /// hold no mesh draw (`set_object_threadgroup_memory_length`): Metal's process ends, with and
    /// without its validation, on an M4.
    ObjectThreadgroupMemory,
    /// A render pipeline state set on a command of a buffer whose command types hold no draw of
    /// its kind (`set_render_pipeline_state`): one of a vertex function on a buffer of mesh draws
    /// alone, a mesh one on a buffer of no mesh draw. Metal's process ends in the setter, with
    /// and without its validation, on an M4 (buffers of either mesh kind and of both, Metal's
    /// driver faulting: buffers of draws of primitives, of indexed draws and of patch draws).
    PipelineKind,
}

impl Device {
    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`: a buffer of
    /// `max_count` reset commands of `descriptor`, `max_count` 0 included.
    ///
    /// Refused `Invalid`: what plain Metal makes no defined storage of (more than 65536 kernel,
    /// vertex or fragment buffers or threadgroup lengths, `Memoryless` storage). Plain Metal
    /// makes what its validation stops of the rest, and so does the runtime: no command types (a
    /// buffer that executes as nothing and whose commands are refused), render types beside
    /// compute types (a buffer of compute commands), more than 31 buffers or lengths. A placement
    /// at 16384 or more is refused `Invalid` (`Problem::ResourceId`): Metal writes past its table
    /// of IDs there. Refused `NotImplemented`: records past one buffer of the device's
    /// `max_buffer_length` (`IndirectCommandBufferPastMaxBufferLength`). The device's memory can
    /// refuse it too (`OutOfMemory`), and so can the IDs, once every ID below 2^31 is held.
    pub fn new_indirect_command_buffer(
        &self,
        descriptor: &IndirectCommandBufferDescriptor,
        max_count: u64,
        options: ResourceOptions,
    ) -> Result<IndirectCommandBuffer, Error> {
        const CALL: &str = "new_indirect_command_buffer";
        let options = self.take_options(CALL, options);
        let modes = Modes::of(options)?;
        if modes.storage == StorageMode::Memoryless {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        let (records, inheritance) = descriptor.records(CALL)?;
        // A device that claims Apple10 would draw a command with its own render states where the
        // buffer does not inherit them: not built.
        if !descriptor.inherits_render_states() && self.supports_family(GpuFamily::Apple10.raw()) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        let placed = records_placement(CALL, descriptor.placement)?;
        let max_buffer_length = self.properties().max_buffer_length;
        let past_length = || {
            Error::NotImplemented(RuntimeGap::IndirectCommandBufferPastMaxBufferLength {
                count: max_count,
                max_buffer_length,
            })
        };
        let size = u32::try_from(max_count).map_err(|_| past_length())?;
        let length = storage_length(size);
        if length > max_buffer_length {
            return Err(past_length());
        }
        // The records are host-visible whatever the storage mode asked for: the CPU initializes
        // them, and reads them back.
        let backing = Modes {
            storage: StorageMode::Shared,
            ..modes
        };
        // Its ID, held until the buffer is gone (`RecordsHold`): `OutOfMemory` once every ID
        // below the records' limit is held. The records lie at the ID's address.
        let held = lock(&self.shared().addresses)
            .take_records_id(placed)
            .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
        let id = RecordsHold {
            device: self.clone(),
            held,
        };
        let identifier = UniqueIdentifier::buffer(
            &self.shared().identifiers,
            metal_allocated_size(descriptor, size),
        )?;
        let storage = self.zeroed_buffer(
            CALL,
            length,
            backing,
            MemoryClass::Shared,
            false,
            Place::Records(held.id),
        )?;
        // The slots past the record's, of their own buffer, as long as Metal's storage grows.
        let overflow = match OverflowLayout::of(descriptor) {
            Some(layout) => {
                let bytes = layout.bytes(size);
                if bytes > max_buffer_length {
                    return Err(past_length());
                }
                let storage = self.zeroed_buffer(
                    CALL,
                    bytes.max(SLOT_BYTES),
                    backing,
                    MemoryClass::Shared,
                    false,
                    AddressPlacement::Outside,
                )?;
                Some(Overflow {
                    storage,
                    past: layout.past,
                    lengths: layout.lengths,
                    pipeline_words: layout.pipeline_words,
                    commands: size,
                })
            }
            None => None,
        };
        let options =
            ResourceOptions::new(modes.storage, modes.cache, storage.hazard_tracking_mode());
        let icb = IndirectCommandBuffer(Arc::new(IcbShared {
            storage,
            id,
            identifier,
            descriptor: *descriptor,
            records,
            inheritance,
            size,
            options,
            pipelines: Mutex::new(vec![None; size as usize]),
            gpu_written: AtomicBool::new(false),
            unforeseen: Mutex::new(Vec::new()),
            cpu_writes: AtomicU64::new(0),
            barriers: AtomicBool::new(false),
            nines: AtomicBool::new(false),
            overflow,
            tail: Mutex::default(),
            gpu_writes: Arc::default(),
            written: Arc::default(),
            kept: Arc::default(),
            keeping: AtomicBool::new(false),
        }));
        icb.initialize()?;
        self.take_device_sampler();
        self.make_texture_pool();
        Ok(icb)
    }
}

/// The pipeline state the CPU set on a command: a compute command's, or a render command's,
/// with the ID it had then. The command names the state without holding it, as Metal's keeps
/// no reference: the program's last release of the state gives its ID back while the command
/// still names it (the consumer's G323). It holds what the state runs, its program: while the
/// record names the ID the command runs that program, whoever holds the ID since, as Metal's
/// command the CPU encoded runs the released state's. With it, what the state may do to indirect
/// command buffers (`Named`), read as it was set.
#[derive(Clone)]
enum CommandState {
    Compute(WeakComputePipelineState, ComputeProgram, IdStamp, Named),
    Render(WeakRenderPipelineState, RenderProgram, IdStamp, Named),
}

/// The ID a state had when the CPU set it on a command, and the serial of its hold of the ID,
/// which tells whether the state was the ID's last holder (`ResourceIds::released_pipeline`);
/// and its `uniqueIdentifier`, which Metal's getter answers after the state's release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IdStamp {
    id: u64,
    serial: u64,
    identifier: u64,
}

impl CommandState {
    /// A compute command's state: the state while it lives, and its stamp.
    fn compute(&self) -> Option<(Option<ComputePipelineState>, IdStamp)> {
        match self {
            CommandState::Compute(named, _, stamp, _) => Some((named.upgrade(), *stamp)),
            CommandState::Render(..) => None,
        }
    }

    /// A render command's state, as `compute` a compute command's.
    fn render(&self) -> Option<(Option<RenderPipelineState>, IdStamp)> {
        match self {
            CommandState::Render(named, _, stamp, _) => Some((named.upgrade(), *stamp)),
            CommandState::Compute(..) => None,
        }
    }

    /// What the state may do to indirect command buffers, of the program it runs.
    fn named(&self) -> Named {
        match self {
            CommandState::Compute(.., named) | CommandState::Render(.., named) => *named,
        }
    }

    /// Whether the program released the state: its program runs for a record naming the ID it
    /// had (`IndirectCommandBuffer::name_state`).
    fn released(&self) -> bool {
        match self {
            CommandState::Compute(named, ..) => named.upgrade().is_none(),
            CommandState::Render(named, ..) => named.upgrade().is_none(),
        }
    }

    /// A compute command's state's program, as a state of no ID (`ComputeProgram::state`).
    fn compute_program(&self) -> Option<ComputePipelineState> {
        match self {
            CommandState::Compute(_, program, ..) => Some(program.state()),
            CommandState::Render(..) => None,
        }
    }

    /// A render command's state's program, as a state of no ID (`RenderProgram::state`).
    fn render_program(&self) -> Option<RenderPipelineState> {
        match self {
            CommandState::Render(_, program, ..) => Some(program.state()),
            CommandState::Compute(..) => None,
        }
    }

    /// Its ID's hold and program, where the program released the state (`released_programs.rs`).
    fn kept(&self) -> Option<(IdStamp, KeptProgram)> {
        if !self.released() {
            return None;
        }
        Some(match self {
            CommandState::Compute(_, program, stamp, _) => {
                (*stamp, KeptProgram::Compute(program.clone()))
            }
            CommandState::Render(_, program, stamp, _) => {
                (*stamp, KeptProgram::Render(program.clone()))
            }
        })
    }
}

/// One kind of pipeline state a command's record names: compute (`S` a
/// [`ComputePipelineState`]) or render.
struct Of<S> {
    /// The kind, as the device's IDs keep it.
    kind: StateKind,
    /// The state the CPU set on a command, where of this kind (`CommandState::compute`).
    own: fn(&CommandState) -> Option<(Option<S>, IdStamp)>,
    /// Its program, as a state (`CommandState::compute_program`).
    program: fn(&CommandState) -> Option<S>,
    /// A kept program of the kind, as a state (`KeptProgram::compute`).
    kept: fn(&KeptProgram) -> Option<S>,
    /// A state's ID.
    id: fn(&S) -> u64,
    /// A state's `uniqueIdentifier`.
    identifier: fn(&S) -> u64,
}

/// Compute states (`Of`).
const COMPUTE: Of<ComputePipelineState> = Of {
    kind: StateKind::Compute,
    own: CommandState::compute,
    program: CommandState::compute_program,
    kept: KeptProgram::compute,
    id: compute_id,
    identifier: ComputePipelineState::identifier_value,
};

/// Render states (`Of`).
const RENDER: Of<RenderPipelineState> = Of {
    kind: StateKind::Render,
    own: CommandState::render,
    program: CommandState::render_program,
    kept: KeptProgram::render,
    id: render_id,
    identifier: RenderPipelineState::identifier_value,
};

/// A compute state's ID.
fn compute_id(state: &ComputePipelineState) -> u64 {
    state.gpu_resource_id().0
}

/// A render state's ID.
fn render_id(state: &RenderPipelineState) -> u64 {
    state.gpu_resource_id().0
}

/// Who reads the states commands name, which decides what a released one reads as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reader {
    /// An execution of the CPU's records, which `commit` records.
    Commit,
    /// A host step's execution of the records as the GPU and the CPU left them.
    Step,
    /// The getter ([`IndirectCommandBuffer::commands`]), which runs nothing: the state the CPU
    /// set reads as none once the program released it, unless a state of its kind took its ID
    /// since, which it reads as.
    Getter,
}

/// How a command an execution runs differs from Metal's, where it names the ID of a state the
/// program released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unmatched {
    /// The record names another ID than the state the CPU set (the GPU wrote it), whose holder
    /// as the GPU last could write it was a state of the command's kind the program released
    /// since, whose program the buffer does not keep (`released_programs.rs`): Metal runs that
    /// state's program, so the command runs nothing.
    Unrun,
    /// The record names the ID of the state the CPU set, which the program released, and the
    /// GPU may have written it while another state held the ID: the runtime runs that state's
    /// program, as Metal runs a command the CPU encoded, where Metal runs the other state for a
    /// record the GPU wrote (nothing, for a state of the other kind).
    Unsure,
}

/// The commands of a range an execution runs otherwise than Metal's may, each with the ID its
/// record names and how: what an execution announces (`IndirectCommandBuffer::degrade_released`),
/// and the getter reads without announcing.
#[derive(Debug, Default)]
pub(super) struct Released(Vec<(u32, u64, Unmatched)>);

/// The bytes of the records of `size` commands, header included.
fn storage_length(size: u32) -> u64 {
    COMMAND_BUFFER_HEADER_BYTES as u64 + u64::from(size) * COMMAND_BYTES as u64
}

/// Metal's `allocatedSize` of a buffer of `count` commands of `descriptor`, whatever the
/// runtime's own records take: the bytes Metal lays them out in, in whole 16-KiB pages, at least
/// one. Its compute and render rules are the consumer's, fitted on an M3 Max; the M4 matched
/// them on every row measured, and the mesh and no-type rules are fitted on the M4, the M3 Max
/// matching them too where measured:
///
/// - Compute types, and render types beside compute types (a buffer of compute commands): a
///   run of arrays, each at its own alignment, of 48, 8 a kernel buffer, 280, 24 (unless the
///   pipeline is inherited), 4 a kernel buffer's stride, 352 (344 inheriting the pipeline) and
///   33 bytes a command, after 112 bytes.
/// - Any other: a header of `380 + (count + 1) * first + count * second` bytes at a multiple
///   of 64, then `command` bytes a command ([`RenderLayout`]).
///
/// Buffers inherited count none of any stage; with dynamic strides each counted vertex buffer
/// (kernel buffer, for compute) takes 4 bytes more a command.
pub(super) fn metal_allocated_size(
    descriptor: &IndirectCommandBufferDescriptor,
    count: u32,
) -> u64 {
    let n = u64::from(count);
    let types = descriptor.command_types;
    let bytes = if types.intersects(COMPUTE_TYPES) {
        compute_bytes(descriptor, n)
    } else {
        RenderLayout::of(descriptor).bytes(n)
    };
    bytes.div_ceil(super::address::PAGE).max(1) * super::address::PAGE
}

/// A stage's buffer count as the layout counts it: none when buffers are inherited.
fn counted(descriptor: &IndirectCommandBufferDescriptor, count: u32) -> u64 {
    if descriptor.inherit_buffers {
        0
    } else {
        u64::from(count)
    }
}

/// The bytes Metal lays out `n` compute commands of `descriptor` in ([`metal_allocated_size`]).
fn compute_bytes(descriptor: &IndirectCommandBufferDescriptor, n: u64) -> u64 {
    let kernel = counted(descriptor, descriptor.max_kernel_buffer_bind_count);
    let strides = u64::from(descriptor.support_dynamic_attribute_stride);
    let inherited = descriptor.inherit_pipeline_state;
    let mut bytes = (48 * n + 112).next_multiple_of(32);
    bytes = (bytes + 8 * kernel * n).next_multiple_of(32) + 280 * n;
    if !inherited {
        bytes = bytes.next_multiple_of(32) + 24 * n;
    }
    bytes = (bytes + 4 * kernel * strides * n).next_multiple_of(64);
    let state = if inherited { 344 } else { 352 };
    (bytes + state * n).next_multiple_of(32) + 33 * n
}

/// Metal's layout of a buffer of render or mesh commands, or of no command types: a header of
/// `380 + (n + 1) * first + n * second` bytes at a multiple of 64, then `command` bytes a
/// command ([`metal_allocated_size`]). `second` is 84 bytes, 28 inheriting the pipeline.
/// `first` and `command`, "inheriting" the pipeline:
///
/// | Command types | `first` | inheriting | `command` | inheriting |
/// |---|---|---|---|---|
/// | draws, neither indexed nor of patches | 64 | 52 | 800 + b | 592 + b |
/// | draws, indexed, not of patches | 84 | 72 | 800 + b | 592 + b |
/// | draws of patches | 104 | 80 | 808 + b | 600 + b |
/// | a mesh type alone | 104 | 88 | 1472 + b | 1160 + b |
/// | a mesh type beside draws | 104 | 88 | 1872 + b | 1456 + b |
/// | a mesh type beside draws of patches | 112 | 96 | 1880 + b | 1464 + b |
/// | none | 24 | 24 | 400 + b | 296 + b |
///
/// `b` is 8 bytes a buffer it counts, and 4 a vertex buffer's stride with dynamic strides. Draws
/// count vertex and fragment buffers; a mesh type alone counts fragment, object and mesh
/// buffers, and beside draws all four; no command types count fragment buffers. With dynamic
/// strides and buffers of its own, a buffer of a mesh type takes 4 bytes more for an odd count
/// (`words`).
struct RenderLayout {
    first: u64,
    second: u64,
    command: u64,
    /// Whether one of the commands' arrays is of 4-byte words at a multiple of 8 bytes, so an
    /// odd count takes 4 bytes more: a mesh type's, with dynamic strides and its own buffers.
    words: bool,
}

impl RenderLayout {
    fn of(descriptor: &IndirectCommandBufferDescriptor) -> RenderLayout {
        let types = descriptor.command_types;
        let inherited = descriptor.inherit_pipeline_state;
        let vertex = counted(descriptor, descriptor.max_vertex_buffer_bind_count);
        let fragment = counted(descriptor, descriptor.max_fragment_buffer_bind_count);
        let object = counted(descriptor, descriptor.max_object_buffer_bind_count);
        let mesh = counted(descriptor, descriptor.max_mesh_buffer_bind_count);
        let strides = 4 * vertex * u64::from(descriptor.support_dynamic_attribute_stride);
        let patches = types.intersects(
            IndirectCommandType::DRAW_PATCHES.union(IndirectCommandType::DRAW_INDEXED_PATCHES),
        );
        let patch = if patches { 8 } else { 0 };
        let draws = types.intersects(DRAW_TYPES);
        let pick = |own: u64, parent: u64| if inherited { parent } else { own };
        let (first, command) = if types.intersects(MESH_TYPES) {
            if draws {
                (
                    pick(104, 88) + patch,
                    pick(1872, 1456) + 8 * (vertex + fragment + object + mesh) + patch,
                )
            } else {
                (
                    pick(104, 88),
                    pick(1472, 1160) + 8 * (fragment + object + mesh),
                )
            }
        } else if draws {
            let first = if patches {
                pick(104, 80)
            } else if types.contains(IndirectCommandType::DRAW_INDEXED) {
                pick(84, 72)
            } else {
                pick(64, 52)
            };
            (first, pick(800, 592) + 8 * (vertex + fragment) + patch)
        } else {
            (24, pick(400, 296) + 8 * fragment)
        };
        RenderLayout {
            first,
            second: pick(84, 28),
            command: command + strides,
            words: types.intersects(MESH_TYPES)
                && descriptor.support_dynamic_attribute_stride
                && !descriptor.inherit_buffers,
        }
    }

    /// The bytes of `n` commands.
    fn bytes(&self, n: u64) -> u64 {
        let padding = if self.words && n % 2 == 1 { 4 } else { 0 };
        (380 + (n + 1) * self.first + n * self.second).next_multiple_of(64)
            + n * self.command
            + padding
    }
}

/// `MTLIndirectCommandBuffer`. Cheap to clone; `Send + Sync`; two are equal when they are the
/// same buffer.
#[derive(Clone)]
pub struct IndirectCommandBuffer(Arc<IcbShared>);

struct IcbShared {
    /// The header and the records.
    storage: Buffer,
    /// Its ID, held while the buffer lives.
    id: RecordsHold,
    /// Its `uniqueIdentifier` (`unique_identifier.rs`), held while the buffer lives.
    identifier: UniqueIdentifier,
    descriptor: IndirectCommandBufferDescriptor,
    /// The record format's descriptor and inheritance of `descriptor`.
    records: CommandDescriptor,
    inheritance: Inheritance,
    size: u32,
    /// The options asked for, the hazard tracking mode resolved.
    options: ResourceOptions,
    /// The pipeline state the CPU set on each command (`CommandState`). Held while the CPU
    /// writes a record, so writes are one at a time.
    pipelines: Mutex<Vec<Option<CommandState>>>,
    /// An encoder made the buffer resident for writing: the GPU may write its records from then
    /// on (`written_by_gpu`). Set, and read where it decides, under `unforeseen`'s lock.
    gpu_written: AtomicBool,
    /// The executions `commit` recorded from the CPU's records before the flag was set, held
    /// weakly until it is (`watch`, `written_by_gpu`).
    unforeseen: Mutex<Vec<Weak<Unforeseen>>>,
    /// The CPU's writes of the records and of the slots past them so far, counted
    /// ([`cpu_writes`](IndirectCommandBuffer::cpu_writes)). Each adds under `pipelines`, after it
    /// wrote.
    cpu_writes: AtomicU64,
    /// The CPU set a render command's barrier: an execution's render pass instance may split at
    /// it (`render/executions.rs`), for good.
    barriers: AtomicBool,
    /// The CPU drew primitive type 9 in a render command: an execution looks for it among the
    /// commands it runs ([`nine_among`](IndirectCommandBuffer::nine_among)), for good.
    nines: AtomicBool,
    /// The buffer slots past the record's, where the descriptor counts more, and a compute
    /// buffer's threadgroup lengths 31 to 37 ([`Overflow`]).
    overflow: Option<Overflow>,
    /// The threadgroup lengths set past the last command's 38 ([`LENGTH_STRIDE`]) and past the
    /// commands' words there ([`Overflow::tail_at`]), by their place past the last command's 38,
    /// none 0. Metal keeps them in its storage's slack for its getter alone (an M3 Max, 3
    /// commands: the last command's lengths at 100, 300 and 1000 read back as set, changing no
    /// other command's state); no command's reset reaches them, and a GPU copy reaches the
    /// first two alone, writing there a value of its own, which the runtime does not (it keeps
    /// them). Past about 4000 places Metal's setter ends the process (`SIGSEGV`, `SIGBUS`); the
    /// runtime keeps them all.
    tail: Mutex<BTreeMap<u64, u32>>,
    /// The command buffers that may write its records on the GPU ([`GpuWrite`]): how many have
    /// not yet completed, and the stamp of the last one's completion.
    gpu_writes: Arc<Mutex<GpuWrites>>,
    /// What its records may hold, from every route that writes them (`written_commands.rs`).
    written: Arc<WrittenCommands>,
    /// The programs of the released pipeline states its records the GPU wrote may name, kept
    /// from the first command buffer that may write them (`released_programs.rs`).
    kept: Arc<Mutex<KeptPrograms>>,
    /// The device's keepers list `kept` (`Keepers::add`).
    keeping: AtomicBool,
}

/// The buffer slots of each command past the 32 of a stage its record holds, below the
/// descriptor's count of the stage, and a compute command's threadgroup lengths 31 to 37, past
/// the 31 its record holds, and its words past the last command's lengths, in a host-visible
/// buffer of their own outside the Metal address space. A command's block holds its buffer
/// slots, 16 bytes a slot, the address word and the stride word as a record's slot holds them
/// (both 0 where none is set), its stages in [`OVERFLOW_STAGES`]' order, then, in a buffer of
/// compute commands, its lengths 31 to 37 ([`LENGTHS_BYTES`]), its pipeline words where the
/// buffer does not inherit its pipeline state ([`PIPELINE_WORDS`]) and its dispatch words
/// ([`DISPATCH_WORDS`]); each command's block follows the one before. Metal keeps such a slot or
/// length for its getters alone (no function binds past index 30), its storage growing with the
/// buffer count; its getters answer every slot below the count and every length, its GPU copy
/// carries them (but lengths 36 and 37 where the commands hold pipeline words, which it clears),
/// its GPU reset clears them, and its CPU resets clear them but the dispatch words.
pub(super) struct Overflow {
    /// Made zeroed.
    storage: Buffer,
    /// The slots past the record's of each of [`OVERFLOW_STAGES`].
    past: [u32; 5],
    /// Whether each command's block holds its threadgroup lengths 31 to 37 and its dispatch
    /// words: the buffer holds compute commands.
    lengths: bool,
    /// Whether each command's block holds its pipeline words: the buffer holds compute commands
    /// and does not inherit its pipeline state.
    pipeline_words: bool,
    /// The commands whose blocks it holds: the buffer's size.
    commands: u32,
}

/// What [`Overflow`] holds of each command, of a descriptor: the slots past the record's of
/// each stage, whether it holds the threadgroup lengths 31 to 37 and the dispatch words, and
/// whether it holds the pipeline words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OverflowLayout {
    past: [u32; 5],
    lengths: bool,
    pipeline_words: bool,
}

/// How many threadgroup lengths Metal keeps a compute command, at `command * 38 + index` of one
/// array of the buffer's, whatever the descriptor's count (an M3 Max, macOS 27.0.1, counts of 8,
/// 31, 32, 40, 64 and 100: command 1's length set at 38 is command 2's at 0, at 50 command 2's
/// at 12, and command 0's getter at 68 answers command 1's at 30). Neither the setter nor the
/// getter checks the index against 38 or against the count; both take its low 32 bits.
const LENGTH_STRIDE: u32 = 38;

/// The bytes of a command's threadgroup lengths 31 to 37 in its [`Overflow`] block: 4 a length,
/// padded to 8 bytes.
const LENGTHS_BYTES: u64 = 32;

/// The bytes of a command's threadgroup lengths 31 to 35 in its [`Overflow`] block, which
/// Metal's GPU copy carries whatever the descriptor; lengths 36 and 37 follow.
const LENGTHS_BEFORE_36: u64 = 20;

/// The words Metal keeps a compute command past the last command's threadgroup lengths, at
/// `2 * command` of them, where the buffer does not inherit its pipeline state (an M3 Max, macOS
/// 27.0.1, buffers of 1 to 8 commands): its pipeline setter writes them, and so do its getter and
/// setter of a length there. Metal's CPU and GPU resets of the command clear them and its GPU copy
/// carries them.
const PIPELINE_WORDS: u64 = 2;

/// The words Metal keeps a compute command past the last command's threadgroup lengths and the
/// [`PIPELINE_WORDS`], at `8 * command` of them (the probes of [`PIPELINE_WORDS`]'s): its dispatch
/// writes the threadgroup counts and size in the first six and its imageblock setter the width
/// and height in the last two, and so do its getter and setter of a length there. Metal's GPU
/// reset of the command clears them and its GPU copy carries them; its CPU resets leave them.
const DISPATCH_WORDS: u64 = 8;

/// The stages whose slots [`Overflow`] holds, in its order.
const OVERFLOW_STAGES: [BufferStage; 5] = [
    BufferStage::Kernel,
    BufferStage::Vertex,
    BufferStage::Fragment,
    BufferStage::Object,
    BufferStage::Mesh,
];

/// The bytes of one buffer slot: the address word and the stride word.
const SLOT_BYTES: u64 = 16;

impl OverflowLayout {
    /// What `descriptor`'s commands hold past their records, `None` where it counts no more than
    /// a record holds of any stage and holds no compute command.
    fn of(descriptor: &IndirectCommandBufferDescriptor) -> Option<OverflowLayout> {
        let counts = [
            descriptor.max_kernel_buffer_bind_count,
            descriptor.max_vertex_buffer_bind_count,
            descriptor.max_fragment_buffer_bind_count,
            descriptor.max_object_buffer_bind_count,
            descriptor.max_mesh_buffer_bind_count,
        ];
        let past = counts.map(|count| count.saturating_sub(RECORD_BUFFERS));
        // Any compute command may hold a length past its record: Metal's setter keeps one at
        // any index, whatever the count.
        let lengths = descriptor.command_types.intersects(COMPUTE_TYPES);
        let pipeline_words = lengths && !descriptor.inherit_pipeline_state;
        (lengths || past.iter().any(|&slots| slots > 0)).then_some(OverflowLayout {
            past,
            lengths,
            pipeline_words,
        })
    }

    /// The bytes of one command's buffer slots.
    fn slot_bytes(&self) -> u64 {
        self.past.iter().map(|&slots| u64::from(slots)).sum::<u64>() * SLOT_BYTES
    }

    /// The bytes of one command's pipeline words: none where the buffer holds no compute
    /// commands or inherits its pipeline state.
    fn pipeline_bytes(&self) -> u64 {
        if self.pipeline_words {
            PIPELINE_WORDS * 4
        } else {
            0
        }
    }

    /// The bytes of one command's lengths 31 to 37, pipeline words and dispatch words: none where
    /// the buffer holds no compute commands.
    fn lengths_bytes(&self) -> u64 {
        if self.lengths {
            LENGTHS_BYTES + self.pipeline_bytes() + DISPATCH_WORDS * 4
        } else {
            0
        }
    }

    /// The bytes of one command's block: its buffer slots, then its lengths 31 to 37, pipeline
    /// words and dispatch words.
    fn command_bytes(&self) -> u64 {
        self.slot_bytes() + self.lengths_bytes()
    }

    /// The bytes an overflow of `commands` commands holds.
    fn bytes(&self, commands: u32) -> u64 {
        u64::from(commands) * self.command_bytes()
    }
}

impl Overflow {
    /// Its layout.
    fn layout(&self) -> OverflowLayout {
        OverflowLayout {
            past: self.past,
            lengths: self.lengths,
            pipeline_words: self.pipeline_words,
        }
    }

    /// The bytes of commands `range`'s blocks.
    pub(super) fn bytes(&self, range: Range<u32>) -> Range<u64> {
        let each = self.layout().command_bytes();
        u64::from(range.start) * each..u64::from(range.end) * each
    }

    /// The offset of threadgroup length `slot` (31 to 37) of command `index`, `None` where the
    /// blocks hold no lengths or `slot` lies past them.
    fn length_at(&self, index: u32, slot: u32) -> Option<u64> {
        let past = slot.checked_sub(THREADGROUP_BUFFER_ARGUMENT_COUNT)?;
        (self.lengths && index < self.commands && slot < LENGTH_STRIDE).then(|| {
            self.bytes(index..index).start + self.layout().slot_bytes() + u64::from(past) * 4
        })
    }

    /// The offset of the word `past` places past the last command's threadgroup lengths, where
    /// Metal keeps a command's words there: each command's [`PIPELINE_WORDS`] first, where the
    /// commands hold them, then each command's [`DISPATCH_WORDS`]. `None` past them, or where
    /// the blocks hold no lengths.
    fn tail_at(&self, past: u64) -> Option<u64> {
        if !self.lengths {
            return None;
        }
        let commands = u64::from(self.commands);
        let pipeline = if self.pipeline_words {
            PIPELINE_WORDS
        } else {
            0
        };
        let (command, at) = match past.checked_sub(pipeline * commands) {
            None => (past / pipeline, LENGTHS_BYTES + past % pipeline * 4),
            Some(past) if past < DISPATCH_WORDS * commands => (
                past / DISPATCH_WORDS,
                LENGTHS_BYTES + pipeline * 4 + past % DISPATCH_WORDS * 4,
            ),
            Some(_) => return None,
        };
        let command = command as u32;
        Some(self.bytes(command..command).start + self.layout().slot_bytes() + at)
    }

    /// The bytes of command `index`'s block a CPU reset clears, as Metal's: all but its dispatch
    /// words.
    fn cpu_reset_bytes(&self, index: u32) -> Range<u64> {
        let block = self.bytes(index..index + 1);
        let kept = if self.lengths { DISPATCH_WORDS * 4 } else { 0 };
        block.start..block.end - kept
    }

    /// The runs of commands `range`'s blocks a GPU copy into them clears after it carried the
    /// source's blocks, as `(offset, bytes)`: lengths 36 and 37 where the commands hold
    /// pipeline words, as Metal's copy clears them; none otherwise.
    pub(super) fn copy_clears(&self, range: Range<u32>) -> Vec<(u64, u64)> {
        if !self.pipeline_words {
            return Vec::new();
        }
        let lengths = self.layout().slot_bytes() + LENGTHS_BEFORE_36;
        range
            .map(|index| {
                let at = self.bytes(index..index).start + lengths;
                (at, LENGTHS_BYTES - LENGTHS_BEFORE_36)
            })
            .collect()
    }

    /// The offset of slot `slot` (at or past the record's) of `stage` of command `index`, `None`
    /// past the descriptor's count.
    fn at(&self, index: u32, stage: BufferStage, slot: u32) -> Option<u64> {
        let of = OVERFLOW_STAGES.iter().position(|&s| s == stage)?;
        let past = slot.checked_sub(RECORD_BUFFERS)?;
        let before: u64 = self.past[..of].iter().map(|&slots| u64::from(slots)).sum();
        (past < self.past[of])
            .then(|| self.bytes(index..index).start + (before + u64::from(past)) * SLOT_BYTES)
    }

    /// The per-stage runs of slots a GPU copy of command `from` of `self` into command `to` of
    /// `into` carries, as `(source offset, destination offset, bytes)`, and the runs of `into`'s
    /// slots it clears, as `(offset, bytes)`: of each stage, the slots both hold, and the
    /// destination's past the source's, which the source holds unset.
    pub(super) fn runs(&self, from: u32, into: &Overflow, to: u32) -> OverflowRuns {
        let mut runs = OverflowRuns::default();
        let (mut source, mut destination) =
            (self.bytes(from..from).start, into.bytes(to..to).start);
        for (&held, &taken) in self.past.iter().zip(&into.past) {
            let both = u64::from(held.min(taken)) * SLOT_BYTES;
            if both > 0 {
                runs.copies.push((source, destination, both));
            }
            let cleared = u64::from(taken.saturating_sub(held)) * SLOT_BYTES;
            if cleared > 0 {
                runs.clears.push((destination + both, cleared));
            }
            source += u64::from(held) * SLOT_BYTES;
            destination += u64::from(taken) * SLOT_BYTES;
        }
        // The threadgroup lengths 31 to 37, pipeline words and dispatch words, as Metal's copy
        // carries a command's: lengths 36 and 37 cleared where the destination's commands hold
        // pipeline words.
        if !into.lengths {
            return runs;
        }
        if !self.lengths {
            runs.clears
                .push((destination, into.layout().lengths_bytes()));
            return runs;
        }
        runs.copies.push((source, destination, LENGTHS_BEFORE_36));
        let (after, rest) = (LENGTHS_BEFORE_36, LENGTHS_BYTES - LENGTHS_BEFORE_36);
        match into.pipeline_words {
            true => runs.clears.push((destination + after, rest)),
            false => runs
                .copies
                .push((source + after, destination + after, rest)),
        }
        let (source, destination) = (source + LENGTHS_BYTES, destination + LENGTHS_BYTES);
        let pipeline = PIPELINE_WORDS * 4;
        match (self.pipeline_words, into.pipeline_words) {
            (true, true) => runs.copies.push((source, destination, pipeline)),
            (false, true) => runs.clears.push((destination, pipeline)),
            _ => {}
        }
        let (source, destination) = (
            source + self.layout().pipeline_bytes(),
            destination + into.layout().pipeline_bytes(),
        );
        runs.copies.push((source, destination, DISPATCH_WORDS * 4));
        runs
    }

    /// The buffer that holds the slots.
    pub(super) fn storage(&self) -> &Buffer {
        &self.storage
    }

    /// Whether `other` lays its commands' blocks out as this one does.
    pub(super) fn same_layout(&self, other: &Overflow) -> bool {
        self.layout() == other.layout()
    }
}

/// The command buffers that may write a buffer's records on the GPU (`IcbShared::gpu_writes`).
#[derive(Debug, Default)]
struct GpuWrites {
    /// Those not yet completed.
    pending: u32,
    /// The device's ID stamp (`ResourceIds::stamp`) as the last one completed.
    end: u64,
}

/// A command buffer's note of a buffer whose records its work may write on the GPU (made
/// resident for writing, or the destination of a blit's copy or reset of commands), from the
/// encoding call until the command buffer completes (an encoder's retained object): its drop
/// stamps the buffer's [`GpuWrites`]. It holds the buffer's writes alone, not the buffer, whose
/// ID a blit's command does not hold, as Metal's does not.
pub(super) struct GpuWrite {
    device: Device,
    writes: Arc<Mutex<GpuWrites>>,
}

impl Drop for GpuWrite {
    fn drop(&mut self) {
        let stamp = lock(&self.device.shared().resource_ids).stamp();
        let mut writes = lock(&self.writes);
        writes.pending = writes.pending.saturating_sub(1);
        writes.end = writes.end.max(stamp);
    }
}

/// Where a threadgroup length is kept (`IndirectCommandBuffer::length_place`): command and
/// index of a record, an offset of the [`Overflow`] (a length 31 to 37 or a command's word past
/// the last command's lengths), or a place past the last command's lengths and the commands'
/// words there (`IcbShared::tail`).
enum LengthPlace {
    Record(u32, u32),
    Overflow(u64),
    Tail(u64),
}

/// What a GPU copy of commands does to the destination's [`Overflow`]: the runs it copies from
/// the source's and the runs it clears.
#[derive(Default)]
pub(super) struct OverflowRuns {
    pub(super) copies: Vec<(u64, u64, u64)>,
    pub(super) clears: Vec<(u64, u64)>,
}

/// An indirect command buffer's records ID, given back as the buffer goes (`address.rs`).
struct RecordsHold {
    device: Device,
    held: RecordsId,
}

impl Drop for RecordsHold {
    fn drop(&mut self) {
        lock(&self.device.shared().addresses).give_records_id(self.held);
    }
}

impl std::fmt::Debug for IndirectCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndirectCommandBuffer")
            .field("size", &self.0.size)
            .field("descriptor", &self.0.descriptor)
            .finish()
    }
}

impl PartialEq for IndirectCommandBuffer {
    fn eq(&self, other: &IndirectCommandBuffer) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for IndirectCommandBuffer {}

impl std::hash::Hash for IndirectCommandBuffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl IndirectCommandBuffer {
    /// The records' bytes: the header, then each command's record.
    fn bytes(&self) -> Result<*mut u8, Error> {
        self.0
            .storage
            .contents()
            .map(|contents| contents.as_ptr())
            .ok_or_else(|| internal("an indirect command buffer's records are not mapped"))
    }

    /// Write the header and reset every command.
    fn initialize(&self) -> Result<(), Error> {
        let _writing = lock(&self.0.pipelines);
        let length = storage_length(self.0.size) as usize;
        // SAFETY: the storage is host-visible and mapped for its life, `length` bytes from its
        // start; nothing else reaches it before the buffer is returned.
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.bytes()?, length) };
        CommandBuffer::initialize(bytes, self.0.size, &self.0.records, self.0.inheritance)
            .map_err(|_| internal("an indirect command buffer's header was refused"))
    }

    /// The bytes of command `index`'s record (below the size), as they are now.
    fn record_now(&self, index: u32) -> Result<[u8; COMMAND_BYTES], Error> {
        let mut bytes = [0; COMMAND_BYTES];
        // SAFETY: the record of a command below the size lies inside the mapped storage. A GPU
        // write to it meanwhile is the program's race, as on Metal.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bytes()?.add(record_offset(index)),
                bytes.as_mut_ptr(),
                COMMAND_BYTES,
            );
        }
        Ok(bytes)
    }

    /// The record of command `index` (below the size).
    fn record(&self, index: u32) -> Result<CommandRecord, Error> {
        CommandRecord::decode(&self.record_now(index)?).map_err(|_| undecodable())
    }

    /// `read` of command `index`'s record (below the size) as it is now, field by field
    /// (`RecordView`), as [`commands`](Self::commands) decodes it: one getter's read, counted
    /// as a record read.
    fn read_record<T>(
        &self,
        index: u32,
        read: impl FnOnce(&RecordView<'_>) -> T,
    ) -> Result<T, Error> {
        self.device().shared().counts.indirect_records_read(1);
        let bytes = self.record_now(index)?;
        let view = RecordView::new(&bytes).map_err(|_| undecodable())?;
        Ok(read(&view))
    }

    /// Write `record` as command `index` (below the size). The caller holds `pipelines`.
    fn write(&self, index: u32, record: &CommandRecord) -> Result<(), Error> {
        let mut bytes = [0; COMMAND_BYTES];
        record
            .encode_into(&mut bytes)
            .map_err(|_| internal("an indirect command's record does not encode"))?;
        // SAFETY: as `record`'s; writes are one at a time (`pipelines`).
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.bytes()?.add(record_offset(index)),
                COMMAND_BYTES,
            );
        }
        self.wrote();
        Ok(())
    }

    /// Count a write of the CPU's ([`cpu_writes`](Self::cpu_writes)), made. The caller holds
    /// `pipelines`.
    fn wrote(&self) {
        self.0.cpu_writes.fetch_add(1, Ordering::Release);
    }

    /// The CPU's writes of the records, and of the slots past them, so far: setters, resets and
    /// the CPU's range resets. An execution `commit` recorded from the CPU's records reads it
    /// before it reads them; where it moved by the time the execution's batch is ready to submit,
    /// a host step records the execution again from the records as they stand then
    /// (`indirect_execute.rs`, `render/executions.rs`), as Metal's execution reads them as it
    /// runs.
    pub(super) fn cpu_writes(&self) -> u64 {
        self.0.cpu_writes.load(Ordering::Acquire)
    }

    /// Reset commands `range` (inside the size).
    fn reset_commands(&self, range: Range<u32>) -> Result<(), Error> {
        let mut pipelines = lock(&self.0.pipelines);
        for index in range.clone() {
            self.write(index, &CommandRecord::default())?;
            pipelines[index as usize] = None;
        }
        self.clear_overflow(range)
    }

    /// `range` as commands, refused unless it starts below the size and ends at or before it, as
    /// Metal's validation requires of `resetWithRange:`, or is the empty one at the size, which
    /// plain Metal takes and which resets nothing. Plain Metal's process ended at a range far past
    /// the size, its GPU faulted executing a buffer reset from past its size, and a range across
    /// the end reset the commands inside it but left the buffer running nothing
    /// (`icb_reset_far_past`, `icb_reset_start_past`, `icb_reset_cross_end`), so those stay
    /// refused.
    fn inside(&self, range: &Range<u64>) -> Result<Range<u32>, Error> {
        let size = u64::from(self.0.size);
        let at_size = range.start == size && range.end == size;
        if (range.start >= size && !at_size) || range.end > size || range.end < range.start {
            return Err(problem(
                "reset_with_range",
                IndirectCommandBufferProblem::Range {
                    location: range.start,
                    length: range.end.saturating_sub(range.start),
                    size,
                },
            ));
        }
        Ok(range.start as u32..range.end as u32)
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        self.0.storage.device()
    }

    /// `size`: the `max_count` the buffer was made with.
    pub fn size(&self) -> u64 {
        u64::from(self.0.size)
    }

    /// `gpuResourceID`: the ID Metal's pool handed out when the buffer was made, of the
    /// indirect command buffers' own space (compute and render buffers share it), or the one its
    /// descriptor placed it at; free again once the buffer and every command that holds it are
    /// gone, a blit's command holding only its records, as Metal's is. A translated kernel that
    /// reads it from an argument buffer encodes into the records of the buffer made last with it,
    /// which lie at `records_address` of it.
    pub fn gpu_resource_id(&self) -> ResourceId {
        ResourceId(self.0.id.held.id)
    }

    /// `resourceIndex`, Apple's private property:
    /// [`gpu_resource_id`](Self::gpu_resource_id)'s value, as Metal answers it.
    pub fn resource_index(&self) -> u64 {
        self.gpu_resource_id().0
    }

    /// `uniqueIdentifier`, Apple's private property: 2^36 plus the buffer's place in Metal's
    /// range for indirect command buffers, as Metal answers it: the lowest gap from `0x18000` that
    /// holds its [`allocated_size`](Self::allocated_size) and one 16-KiB page more, at a multiple
    /// of 32 KiB, free again once the buffer is gone and no command buffer that executes it may
    /// run. Metal's own encoders take ranges there too, which the runtime takes as an M3 Max's
    /// do (macOS 27.0.1): a blit encoder's and a render encoder's at their opening, a dispatch's
    /// or a blit command's as its command buffer is scheduled. A render pass's run takes more, by
    /// its targets' sizes and what the device drew before, which the runtime does not model: a
    /// buffer made after a command buffer that encoded a render encoder was scheduled, or after a
    /// resource state encoder opened, announces
    /// [`Degradation::UnmeasuredUniqueIdentifier`](crate::Degradation::UnmeasuredUniqueIdentifier)
    /// once a device where this answers.
    pub fn unique_identifier(&self) -> u64 {
        self.0
            .identifier
            .answer(self.device(), "an indirect command buffer")
    }

    /// Whether the buffer's [`unique_identifier`](Self::unique_identifier) is a number not
    /// measured to be Metal's: a runtime query, with no Metal counterpart. True for a buffer made
    /// after a command buffer of its device that encoded a render encoder was scheduled, after a
    /// resource state encoder of its device opened, or after Metal's own encoders' ranges passed
    /// what the runtime models, where `unique_identifier` announces
    /// [`Degradation::UnmeasuredUniqueIdentifier`](crate::Degradation::UnmeasuredUniqueIdentifier);
    /// false where it answers Metal's number. Fixed at the buffer's making; it announces nothing.
    pub fn identifier_unmeasured(&self) -> bool {
        self.0.identifier.degradation().is_some()
    }

    /// Whether the buffer holds its ID's slot: whether a kernel encoding through the ID reaches
    /// its records, which it does only while no buffer made later holds the ID.
    pub(super) fn holds_id_slot(&self) -> bool {
        lock(&self.device().shared().addresses).holds_records_slot(self.0.id.held)
    }

    /// `indirectComputeCommandAtIndex:`: command `index`'s setters. Refused (`Invalid`): an index
    /// at or past the size, as Metal's validation stops it; a buffer of render command types
    /// (`CommandKind`), which ends Metal's process. A `Private` buffer's commands are encoded
    /// here too, as plain Metal encodes, reads back and runs them where its validation stops the
    /// call (an M3 Max and an M4).
    pub fn indirect_compute_command_at(&self, index: u64) -> Result<IndirectComputeCommand, Error> {
        const CALL: &str = "indirect_compute_command_at";
        let index = self.command_at(CALL, index, false)?;
        Ok(IndirectComputeCommand {
            icb: self.clone(),
            index,
        })
    }

    /// `indirectRenderCommandAtIndex:`: command `index`'s setters. Refused (`Invalid`) as
    /// [`indirect_compute_command_at`](Self::indirect_compute_command_at) is, and for a buffer of
    /// dispatches beside draws (`CommandKind`), which ends Metal's process at the command's
    /// first setter or breaks its other commands. A buffer of dispatches alone gives one, as plain
    /// Metal does where its validation stops the call, whose setters do what Metal's do there.
    pub fn indirect_render_command_at(&self, index: u64) -> Result<IndirectRenderCommand, Error> {
        const CALL: &str = "indirect_render_command_at";
        let index = self.command_at(CALL, index, true)?;
        Ok(IndirectRenderCommand {
            icb: self.clone(),
            index,
        })
    }

    /// Command `index` for `call`, of a buffer whose commands are render ones where `render`:
    /// refused (`Invalid`) for a buffer of the other kind's commands (but a render command of a
    /// buffer of dispatches alone), a buffer of no command types, and an index at or past the
    /// size. A `Private` buffer's commands are the CPU's as a `Shared` one's are, as plain Metal
    /// encodes, reads back and runs them.
    fn command_at(&self, call: &'static str, index: u64, render: bool) -> Result<u32, Error> {
        self.check_command_types(call)?;
        self.check_kind(call, render)?;
        if index >= u64::from(self.0.size) {
            return Err(problem(
                call,
                IndirectCommandBufferProblem::CommandIndex {
                    index,
                    size: self.size(),
                },
            ));
        }
        Ok(index as u32)
    }

    /// Whether its commands are render commands: its descriptor's command types are render ones.
    pub(super) fn is_render(&self) -> bool {
        matches!(self.0.records, CommandDescriptor::Render { .. })
    }

    /// Refuse, for `call`, a buffer whose commands are not of the kind `render` says
    /// (`Invalid`, `IndirectCommandBufferProblem::CommandKind`), but a render command of a
    /// buffer of dispatches alone, which plain Metal gives (`IndirectRenderCommand`'s
    /// `of_compute`).
    fn check_kind(&self, call: &'static str, render: bool) -> Result<(), Error> {
        let dispatches_alone = !self.0.descriptor.command_types.intersects(RENDER_TYPES);
        if self.is_render() != render && !(render && dispatches_alone) {
            return Err(problem(call, IndirectCommandBufferProblem::CommandKind));
        }
        Ok(())
    }

    /// Whether its descriptor has no command types.
    pub(super) fn has_no_command_types(&self) -> bool {
        self.0.descriptor.command_types.is_empty()
    }

    /// Refuse, for `call`, a buffer of no command types (`Invalid`,
    /// `IndirectCommandBufferProblem::NoCommandTypes`): plain Metal makes one, and executes,
    /// resets, optimizes and copies it as nothing, but its process ends at a command's first setter
    /// and at its CPU reset; what else it does with one was not measured.
    pub(super) fn check_command_types(&self, call: &'static str) -> Result<(), Error> {
        if self.has_no_command_types() {
            return Err(problem(call, IndirectCommandBufferProblem::NoCommandTypes));
        }
        Ok(())
    }

    /// Whether a render pass's execution of the buffer runs nothing, as plain Metal's runs a
    /// buffer of no command types, or of dispatches beside draws of primitives. One of dispatches
    /// beside other draws was not measured in a pass.
    pub(super) fn runs_nothing_in_a_pass(&self) -> bool {
        let types = self.0.descriptor.command_types;
        types.is_empty()
            || (types.intersection(RENDER_TYPES) == IndirectCommandType::DRAW
                && types.intersects(COMPUTE_TYPES))
    }

    /// Write `field` into command `index`'s record (below the size), and where the field names a
    /// pipeline state, `set` as the state the CPU set on it: the field's words alone, as Metal's
    /// setter writes its field, the rest of the record as the GPU and the CPU left it
    /// (`CommandRecord::write_field`); then what the record holds joins what its records may.
    fn edit(&self, index: u32, field: RecordField, set: Option<CommandState>) -> Result<(), Error> {
        let mut pipelines = lock(&self.0.pipelines);
        let mut bytes = self.record_now(index)?;
        let written =
            CommandRecord::write_field(&mut bytes, field).map_err(|error| match error {
                RecordError::Length { .. }
                | RecordError::Action { .. }
                | RecordError::Barrier { .. }
                | RecordError::PipelineState => undecodable(),
                _ => internal("an indirect command's record does not take a setter's field"),
            })?;
        // SAFETY: as `record_now`'s; `written` lies inside the record, and writes are one at a
        // time (`pipelines`).
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[written.clone()].as_ptr(),
                self.bytes()?.add(record_offset(index) + written.start),
                written.len(),
            );
        }
        self.wrote();
        let pipeline = &mut pipelines[index as usize];
        if let Some(set) = set {
            *pipeline = Some(set);
        }
        let named = pipeline.as_ref().map(CommandState::named);
        let writes = RecordView::new(&bytes).map_err(|_| undecodable())?.writes();
        self.0.written.add_record(writes, named);
        Ok(())
    }

    /// Reset command `index` (below the size) by `reset`, which changes its whole record and the
    /// pipeline state the CPU set on it, and clear its slots past the record's, as Metal's reset
    /// clears every slot.
    fn reset_command(
        &self,
        index: u32,
        reset: impl FnOnce(&mut CommandRecord, &mut Option<CommandState>),
    ) -> Result<(), Error> {
        let mut pipelines = lock(&self.0.pipelines);
        let mut record = self.record(index)?;
        reset(&mut record, &mut pipelines[index as usize]);
        self.write(index, &record)?;
        self.clear_overflow(index..index + 1)
    }

    /// Set `set`'s buffer on command `index`, for `call`, at the buffer's address plus the offset's
    /// low 32 bits, as Metal keeps it: on an M4 under macOS 27.0.1, a kernel or vertex buffer set
    /// at 2^40 + 16 or 2^32 + 32, plainly or with an attribute stride, read and answered 16 or 32
    /// past the buffer's address, and so did a fragment buffer's getter; one set at 2^32 + 16 in a
    /// buffer past 4 GiB read at 16, where a compute encoder's own buffer reads at the whole offset
    /// (with and without API validation; the consumer measured the getters so on an M3 Max), and so
    /// did object and mesh buffers' getters and reads. Refused (`Invalid`): a buffer of another
    /// device. On a buffer that inherits its buffers the record keeps it and reads it back, and an
    /// execution binds the encoder's buffers alone, none where the encoder's slot is unbound, as
    /// plain Metal's, where its validation stops the call.
    ///
    /// A slot at or past the set's limit (the descriptor's count of the stage) is **taken** and
    /// kept nowhere. Plain Metal takes it: on an M3 Max under macOS 27.0.1 a buffer of
    /// `maxKernelBufferBindCount` 2 took `setKernelBuffer:offset:atIndex:` at index 2 and at
    /// index 4 with no error and its `size` still read 3 commands, where Metal's own API
    /// validation stops the call ("index(2) must be < 2."). Nothing of either side binds such a
    /// slot to a function: an execution binds the slots below the count alone (`bindings` in
    /// `indirect_execute.rs`, `buffers` in `render/executions.rs`), as Metal's count-sized
    /// argument table does, and [`read_buffer`](Self::read_buffer) answers none at or past the
    /// count. So the binding is dropped rather than written: the record's own slots are held to
    /// the descriptor's counts by the engine (`indirect_commands::descriptor`'s `validate_record`,
    /// which an execution's records and the records the GPU writes both go through), and what
    /// Metal keeps such a slot in for its private getters is unmeasured.
    fn set_buffer(&self, index: u32, call: &'static str, set: BufferSet<'_>) -> Result<(), Error> {
        let BufferSet {
            stage,
            limit,
            slot,
            buffer,
            offset,
            attribute_stride,
        } = set;
        if let Some(buffer) = buffer {
            if !buffer.device().same(self.device()) {
                return Err(invalid(call, Problem::ForeignObject));
            }
        }
        // At or past the descriptor's count nothing binds the slot and no getter reads it, so
        // the binding is kept nowhere; the call is still taken, as plain Metal takes it.
        if slot >= limit {
            return Ok(());
        }
        // Metal keeps the offset's low 32 bits: its getters answer them and the GPU reads
        // there, inside a buffer past 4 GiB too. A nil buffer keeps them too, as its address:
        // Metal's getters answer them and its function sees them as the buffer's address.
        let base = buffer.map_or(0, |buffer| buffer.gpu_address().0);
        let binding = BufferBinding {
            address: base + u64::from(offset as u32),
            attribute_stride,
        };
        // Past the slots the record holds no function binds a buffer (Metal's argument tables
        // hold 31): plain Metal keeps it for its private getters, and so does the overflow.
        if slot >= RECORD_BUFFERS {
            let _writing = lock(&self.0.pipelines);
            return self.write_overflow(index, stage, slot, Some(binding));
        }
        let field = RecordField::Buffer {
            stage,
            slot: slot as usize,
            binding: Some(binding),
        };
        self.edit(index, field, None)
    }

    /// Where Metal keeps threadgroup length `slot` of command `index` of a compute buffer: at
    /// `index * 38 + slot` of one array of the buffer's ([`LENGTH_STRIDE`]), the length of
    /// command `(index * 38 + slot) / 38` at the remainder: in its record below 31, in its
    /// [`Overflow`] block at 31 to 37; past the last command's, the commands' words there
    /// ([`Overflow::tail_at`]), and past those in [`IcbShared::tail`].
    fn length_place(&self, index: u32, slot: u32) -> Result<LengthPlace, Error> {
        let stride = u64::from(LENGTH_STRIDE);
        let flat = u64::from(index) * stride + u64::from(slot);
        let (command, slot) = (flat / stride, (flat % stride) as u32);
        let size = u64::from(self.0.size);
        if command >= size {
            let past = flat - size * stride;
            let word = self
                .0
                .overflow
                .as_ref()
                .and_then(|overflow| overflow.tail_at(past));
            return Ok(word.map_or(LengthPlace::Tail(past), LengthPlace::Overflow));
        }
        let command = command as u32;
        if slot < THREADGROUP_BUFFER_ARGUMENT_COUNT {
            return Ok(LengthPlace::Record(command, slot));
        }
        self.0
            .overflow
            .as_ref()
            .and_then(|overflow| overflow.length_at(command, slot))
            .map(LengthPlace::Overflow)
            .ok_or_else(|| internal("an indirect command's threadgroup length has no place"))
    }

    /// Set threadgroup length `slot` of command `index`, where Metal keeps it
    /// ([`length_place`](Self::length_place)). A length past a record is no execution's: it
    /// counts as no write of the records ([`cpu_writes`](Self::cpu_writes)).
    fn set_length(&self, index: u32, slot: u32, length: u32) -> Result<(), Error> {
        match self.length_place(index, slot)? {
            LengthPlace::Record(command, slot) => {
                let field = RecordField::ThreadgroupMemoryLength {
                    index: slot as usize,
                    length,
                };
                self.edit(command, field, None)
            }
            LengthPlace::Overflow(at) => {
                let _pipelines = lock(&self.0.pipelines);
                self.write_overflow_at(at, &length.to_le_bytes())
            }
            LengthPlace::Tail(at) => {
                let mut tail = lock(&self.0.tail);
                match length {
                    0 => tail.remove(&at),
                    _ => tail.insert(at, length),
                };
                Ok(())
            }
        }
    }

    /// Threadgroup length `slot` of command `index` as it is now, where Metal keeps it
    /// ([`length_place`](Self::length_place)): 0 where none was set.
    fn length(&self, index: u32, slot: u32) -> Result<u32, Error> {
        match self.length_place(index, slot)? {
            LengthPlace::Record(command, slot) => self
                .read_record(command, |view| {
                    view.threadgroup_memory_length(slot as usize)
                })
                .map(Option::unwrap_or_default),
            LengthPlace::Overflow(at) => {
                let mut bytes = [0u8; 4];
                match self.read_overflow_at(at, &mut bytes) {
                    true => Ok(u32::from_le_bytes(bytes)),
                    false => Err(internal(
                        "an indirect command buffer's overflow is not mapped",
                    )),
                }
            }
            LengthPlace::Tail(at) => Ok(lock(&self.0.tail).get(&at).copied().unwrap_or(0)),
        }
    }

    /// Write `binding` (none: unset) at `slot` of `stage` of command `index`, past the record's
    /// slots and below the descriptor's count. The caller holds `pipelines`.
    fn write_overflow(
        &self,
        index: u32,
        stage: BufferStage,
        slot: u32,
        binding: Option<BufferBinding>,
    ) -> Result<(), Error> {
        let at = self
            .0
            .overflow
            .as_ref()
            .and_then(|overflow| overflow.at(index, stage, slot))
            .ok_or_else(|| internal("an indirect command's slot past its record has no place"))?;
        let (address, stride) = binding.map_or((0, 0), |binding| {
            (
                binding.address,
                binding.attribute_stride.unwrap_or(u64::MAX),
            )
        });
        let mut bytes = [0u8; SLOT_BYTES as usize];
        bytes[..8].copy_from_slice(&address.to_le_bytes());
        bytes[8..].copy_from_slice(&stride.to_le_bytes());
        self.write_overflow_at(at, &bytes)?;
        self.wrote();
        Ok(())
    }

    /// Write `bytes` at `at` of the overflow, which holds them (`Overflow::at`,
    /// `Overflow::length_at`). The caller holds `pipelines`.
    fn write_overflow_at(&self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let contents = self
            .0
            .overflow
            .as_ref()
            .and_then(|overflow| overflow.storage.contents())
            .ok_or_else(|| internal("an indirect command buffer's overflow is not mapped"))?;
        // SAFETY: the overflow is host-visible and mapped for its life, and `at` and `bytes` lie
        // inside it (the caller's); writes are one at a time (`pipelines`). A GPU write meanwhile
        // is the program's race, as on Metal.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.as_ptr().add(at as usize),
                bytes.len(),
            );
        }
        Ok(())
    }

    /// Read `bytes` from `at` of the overflow, which holds them (as
    /// [`write_overflow_at`](Self::write_overflow_at)'s); false where it is not mapped.
    fn read_overflow_at(&self, at: u64, bytes: &mut [u8]) -> bool {
        let Some(contents) = self
            .0
            .overflow
            .as_ref()
            .and_then(|overflow| overflow.storage.contents())
        else {
            return false;
        };
        // SAFETY: as `write_overflow_at`'s; a write meanwhile is the program's race.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr().add(at as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        true
    }

    /// The slots of `stage` of command `index` past the record's, below `count`, as set: none
    /// where no setter set one or a reset cleared it.
    fn overflow_slots(
        &self,
        index: u32,
        stage: BufferStage,
        count: u32,
    ) -> Vec<Option<BufferBinding>> {
        let Some(overflow) = self.0.overflow.as_ref() else {
            return Vec::new();
        };
        (RECORD_BUFFERS..count)
            .take_while(|&slot| overflow.at(index, stage, slot).is_some())
            .map(|slot| self.overflow_slot(index, stage, slot))
            .collect()
    }

    /// Slot `slot` of `stage` of command `index`, past the record's, as set: none where no setter
    /// set it, a reset cleared it, or the descriptor counts none there.
    fn overflow_slot(&self, index: u32, stage: BufferStage, slot: u32) -> Option<BufferBinding> {
        let at = self.0.overflow.as_ref()?.at(index, stage, slot)?;
        let mut bytes = [0u8; SLOT_BYTES as usize];
        if !self.read_overflow_at(at, &mut bytes) {
            return None;
        }
        let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default());
        let (address, stride) = (word(0), word(8));
        // Both 0 is a slot no setter set, as a record's (`CommandRecord::decode`).
        (address != 0 || stride != 0).then_some(BufferBinding {
            address,
            attribute_stride: (stride != u64::MAX).then_some(stride),
        })
    }

    /// Clear the slots past the record's of commands `range`, as a CPU reset does: all but
    /// their dispatch words ([`Overflow::cpu_reset_bytes`]). The caller holds `pipelines`.
    fn clear_overflow(&self, range: Range<u32>) -> Result<(), Error> {
        let Some(overflow) = self.0.overflow.as_ref() else {
            return Ok(());
        };
        let contents = overflow
            .storage
            .contents()
            .ok_or_else(|| internal("an indirect command buffer's overflow is not mapped"))?;
        for index in range {
            let bytes = overflow.cpu_reset_bytes(index);
            // SAFETY: as `write_overflow`'s; `bytes` lies inside the overflow.
            unsafe {
                std::ptr::write_bytes(
                    contents.as_ptr().add(bytes.start as usize),
                    0,
                    (bytes.end - bytes.start) as usize,
                );
            }
        }
        self.wrote();
        Ok(())
    }

    /// The slots past the record's, where the descriptor counts more than a record holds.
    pub(super) fn overflow(&self) -> Option<&Overflow> {
        self.0.overflow.as_ref()
    }

    /// The buffers of `stage` below `count` that command `index`, of `record`, sets, nil
    /// buffers included: those its record holds, then those past them ([`Overflow`]); none
    /// where no setter set one or a reset cleared it.
    fn buffers_of(
        &self,
        (index, record): (u32, &CommandRecord),
        stage: BufferStage,
        count: u32,
    ) -> Vec<Option<IndirectBufferBinding>> {
        let held = record.bindings.buffers(stage)[..held(count, RECORD_BUFFERS) as usize]
            .iter()
            .copied();
        held.chain(self.overflow_slots(index, stage, count))
            .map(|binding| {
                binding.map(|binding| {
                    IndirectBufferBinding::new(
                        GpuAddress(binding.address),
                        binding.attribute_stride,
                    )
                })
            })
            .collect()
    }

    /// `resetWithRange:`: reset commands `range`, as a fresh buffer's. Refused (`Invalid`) unless
    /// the range starts below the size and ends at or before it, an empty range included, as
    /// Metal's validation refuses it, but for the empty range at the size, which resets nothing
    /// as plain Metal's does; a `Private` buffer's are reset too.
    pub fn reset_with_range(&self, range: Range<u64>) -> Result<(), Error> {
        self.check_command_types("reset_with_range")?;
        let range = self.inside(&range)?;
        self.reset_commands(range)
    }

    /// The commands of `range`, decoded (a runtime extension standing for Metal's private
    /// getters): a compute command's dispatch, barrier, pipeline state and the ID its record names,
    /// kernel buffers, threadgroup lengths, stage-in region and imageblock size; a render command's
    /// draw, barrier, pipeline state and its ID, and vertex and fragment buffers. Of a descriptor's
    /// count past what a record holds (32 buffers of a stage, 31 threadgroup lengths), the slots no
    /// function binds, it reads every buffer slot, as Metal's getters do, and the lengths the
    /// record holds. Refused (`Invalid`): a range that ends past the size; an empty one anywhere up
    /// to the size reads nothing.
    pub fn commands(&self, range: Range<u64>) -> Result<Vec<IndirectCommand>, Error> {
        self.check_command_types("commands")?;
        let range = self.command_range("commands", range)?;
        let (records, sets) = self.read_records(range.clone())?;
        let read = range
            .zip(&records)
            .zip(&sets)
            .map(|((index, record), set)| (index, record, set.as_ref()));
        Ok(self.getter_commands(read))
    }

    /// The commands `read` (each one's index, record and the state the CPU set on it), as
    /// [`commands`](Self::commands) reads them, announcing their identifiers' kinds as it does.
    fn getter_commands<'r>(
        &self,
        read: impl Iterator<Item = (u32, &'r CommandRecord, Option<&'r CommandState>)> + Clone,
    ) -> Vec<IndirectCommand> {
        // A getter runs nothing, so it announces none of the commands it reads that name a
        // released state: an execution of them does.
        let commands: Vec<IndirectCommand> = if self.is_render() {
            let live = || RenderPipelineState::indirect_states(self.device());
            let (named, _) = self.name_records(read.clone(), Reader::Getter, &RENDER, live);
            read.zip(named)
                .map(|((index, record, set), pipeline)| {
                    let runs = record.bindings.pipeline.or(record.retained_pipeline);
                    let (pipeline, identifier) = self.render_reading(runs, set, pipeline);
                    IndirectCommand::Render(self.render_command(
                        (index, record),
                        pipeline,
                        identifier,
                    ))
                })
                .collect()
        } else {
            let live = || ComputePipelineState::indirect_states(self.device());
            let (named, _) = self.name_records(read.clone(), Reader::Getter, &COMPUTE, live);
            read.zip(named)
                .map(|((index, record, set), pipeline)| {
                    let reported = record.bindings.pipeline;
                    let (pipeline, identifier) = self.compute_reading(reported, set, pipeline);
                    IndirectCommand::Compute(self.compute_command(
                        (index, record),
                        pipeline,
                        identifier,
                    ))
                })
                .collect()
        };
        self.announce_identifiers(&commands);
        commands
    }

    /// Call `check` on each command whose record `recorded` holds, commands from `start` (inside
    /// the size), in order, as [`commands`](Self::commands) reads them now: the host step's check
    /// of the commands it will record, before it records any ([`IndirectCommandCheck`]).
    /// `recorded` is the step's records, `None` for one that faulted; an empty list (a faulted
    /// header) checks nothing. The check reads the step's records, which the step read once the
    /// work before it completed, as the getter would read them then: none is read again.
    pub(super) fn check_recorded(
        &self,
        check: &Mutex<IndirectCommandCheck>,
        start: u32,
        recorded: &[Option<CommandRecord>],
    ) {
        let held = (start..)
            .zip(recorded)
            .filter_map(|(index, record)| Some((index, record.as_ref()?)))
            .collect();
        self.check_held(check, held);
    }

    /// Call `check` on each command of `read`, the records `commit` read of commands from
    /// `start` (inside the size), in order, as [`commands`](Self::commands) reads them now:
    /// `commit`'s check of the commands it records from the CPU's records, before it records any
    /// ([`IndirectCommandCheck`]). None is read again.
    pub(super) fn check_read<'r>(
        &self,
        check: &Mutex<IndirectCommandCheck>,
        start: u32,
        read: impl Iterator<Item = &'r CommandRecord>,
    ) {
        self.check_held(check, (start..).zip(read).collect());
    }

    /// Call `check` on each of `held`, a command's index and its record, in order, as
    /// [`commands`](Self::commands) reads them now, the state the CPU set on each as it is now.
    fn check_held(&self, check: &Mutex<IndirectCommandCheck>, held: Vec<(u32, &CommandRecord)>) {
        if held.is_empty() {
            return;
        }
        let sets: Vec<Option<CommandState>> = {
            let pipelines = lock(&self.0.pipelines);
            held.iter()
                .map(|&(index, _)| pipelines.get(index as usize).cloned().flatten())
                .collect()
        };
        let read = held
            .iter()
            .zip(&sets)
            .map(|(&(index, record), set)| (index, record, set.as_ref()));
        let commands = self.getter_commands(read);
        let mut check = lock(check);
        for (&(index, _), command) in held.iter().zip(&commands) {
            check(u64::from(index), command);
        }
    }

    /// Announce the kinds of the states' identifiers `commands` answer, none of which is measured
    /// to be Metal's number (`unique_identifier.rs`); a command that names no state answers 0, as
    /// Metal's does. Each kind once a call, in the order the commands first answer it.
    fn announce_identifiers(&self, commands: &[IndirectCommand]) {
        let mut kinds: [Option<Degradation>; 2] = [None; 2];
        let answered = commands.iter().map(|command| match command {
            IndirectCommand::Compute(command) => command.pipeline_unique_identifier,
            IndirectCommand::Render(command) => command.pipeline_unique_identifier,
        });
        for kind in answered.filter_map(|answer| answer.and_then(state_value_degradation)) {
            if let Some(free) = kinds
                .iter_mut()
                .find(|held| held.is_none_or(|held| held == kind))
            {
                *free = Some(kind);
            }
        }
        for kind in kinds.into_iter().flatten() {
            degrade_identifier(self.device(), kind, "an indirect command's pipeline state");
        }
    }

    /// What the getter reads of a compute command's pipeline state, where its record reports
    /// `reported` (`bindings.pipeline`), the CPU set `set` on it and the getter names `state`
    /// for the pipeline it executes with (`name_state`): the state, which a pipeline a reset
    /// retained runs but reads as none, as Metal's getter reports it; and its identifier
    /// (`pipeline_identifier`).
    fn compute_reading(
        &self,
        reported: Option<NonZeroU64>,
        set: Option<&CommandState>,
        state: Option<ComputePipelineState>,
    ) -> (Option<ComputePipelineState>, Option<u64>) {
        let state = state.filter(|_| reported.is_some());
        let identifier = self.pipeline_identifier(reported, set, &COMPUTE, state.as_ref());
        (state, identifier)
    }

    /// What the getter reads of a render command's pipeline state, as `compute_reading` a compute
    /// one's, where its record executes with `runs`, which it reports, a retained one too.
    fn render_reading(
        &self,
        runs: Option<NonZeroU64>,
        set: Option<&CommandState>,
        state: Option<RenderPipelineState>,
    ) -> (Option<RenderPipelineState>, Option<u64>) {
        let identifier = self.pipeline_identifier(runs, set, &RENDER, state.as_ref());
        (state, identifier)
    }

    /// Command `index`'s (below the size) pipeline words as its record holds them now, the
    /// pipeline it reports (`bindings.pipeline`) and the one it executes with (a retained one
    /// too), and the state the CPU set on it, read together: one getter's read.
    fn read_pipeline(
        &self,
        index: u32,
    ) -> Result<(Option<NonZeroU64>, Option<NonZeroU64>, Option<CommandState>), Error> {
        let pipelines = lock(&self.0.pipelines);
        let (reported, runs) = self.read_record(index, |view| {
            (
                view.pipeline(),
                view.pipeline().or(view.retained_pipeline()),
            )
        })?;
        Ok((reported, runs, pipelines[index as usize].clone()))
    }

    /// What [`commands`](Self::commands) reads of compute command `index`'s pipeline state
    /// (`compute_reading`), read alone.
    fn compute_pipeline_now(
        &self,
        index: u32,
    ) -> Result<(Option<ComputePipelineState>, Option<u64>), Error> {
        let (reported, runs, set) = self.read_pipeline(index)?;
        let live = || ComputePipelineState::indirect_states(self.device());
        let read = std::iter::once((index, runs, set.as_ref()));
        let (mut named, _) = self.name_pipelines(read, Reader::Getter, &COMPUTE, live);
        Ok(self.compute_reading(reported, set.as_ref(), named.pop().flatten()))
    }

    /// What [`commands`](Self::commands) reads of render command `index`'s pipeline state
    /// (`render_reading`), read alone.
    fn render_pipeline_now(
        &self,
        index: u32,
    ) -> Result<(Option<RenderPipelineState>, Option<u64>), Error> {
        let (_, runs, set) = self.read_pipeline(index)?;
        let live = || RenderPipelineState::indirect_states(self.device());
        let read = std::iter::once((index, runs, set.as_ref()));
        let (mut named, _) = self.name_pipelines(read, Reader::Getter, &RENDER, live);
        Ok(self.render_reading(runs, set.as_ref(), named.pop().flatten()))
    }

    /// Announce the kind of a state's identifier a getter answered, as
    /// [`announce_identifiers`](Self::announce_identifiers) does.
    fn announce_identifier(&self, identifier: Option<u64>) {
        if let Some(kind) = identifier.and_then(state_value_degradation) {
            degrade_identifier(self.device(), kind, "an indirect command's pipeline state");
        }
    }

    /// The buffer slot `slot` of `stage` of command `index` (below the size), below `count`, as
    /// [`commands`](Self::commands) reads it: the record's, or past the slots it holds, the
    /// overflow's ([`Overflow`]); none at or past `count`, and where no setter set it or a reset
    /// cleared it. One getter's read.
    fn read_buffer(
        &self,
        index: u32,
        stage: BufferStage,
        count: u32,
        slot: u64,
    ) -> Result<Option<IndirectBufferBinding>, Error> {
        let Some(slot) = u32::try_from(slot).ok().filter(|&slot| slot < count) else {
            return Ok(None);
        };
        let held = self.read_record(index, |view| view.buffer(stage, slot as usize))?;
        let binding = match slot < RECORD_BUFFERS {
            true => held,
            false => self.overflow_slot(index, stage, slot),
        };
        Ok(binding.map(|binding| {
            IndirectBufferBinding::new(GpuAddress(binding.address), binding.attribute_stride)
        }))
    }

    /// `getPipelineStateUniqueIdentifier` of a command whose record names pipeline state ID
    /// `named`, the CPU having set `set` on it and the getter reading `state` of kind `of` for
    /// it (`name_state`): 0 where it names none; the identifier of the state the CPU set where
    /// the record names its ID, after the state's release too, as Metal answers it (measured on
    /// an M3 Max by the consumer), unless the GPU may have written the record while another state
    /// held the ID ([`may_name_another`](Self::may_name_another)); for a record the GPU wrote, the
    /// identifier of the state that held the ID as the GPU last could write it
    /// ([`written_identifier`](Self::written_identifier)), else the live state of the ID's. `None`
    /// where the runtime cannot tell which state the record names.
    fn pipeline_identifier<S>(
        &self,
        named: Option<NonZeroU64>,
        set: Option<&CommandState>,
        of: &Of<S>,
        state: Option<&S>,
    ) -> Option<u64> {
        let Some(named) = named.map(NonZeroU64::get) else {
            return Some(0);
        };
        match set.and_then(of.own).filter(|(_, stamp)| stamp.id == named) {
            Some((Some(_), stamp)) => Some(stamp.identifier),
            Some((None, stamp)) => {
                (!self.may_name_another(named, of.kind, stamp.serial)).then_some(stamp.identifier)
            }
            None => self
                .written_identifier(named, of.kind)
                .or_else(|| state.map(of.identifier)),
        }
    }

    /// The descriptor the buffer was made with.
    pub(super) fn descriptor(&self) -> &IndirectCommandBufferDescriptor {
        &self.0.descriptor
    }

    /// What its records inherit from the encoder that executes them (the policy).
    pub(super) fn inheritance(&self) -> Inheritance {
        self.0.inheritance
    }

    /// `range` as commands for `call`: refused (`Invalid`) when it ends past the size, as Metal's
    /// validation refuses `executeCommandsInBuffer:withRange:`; an empty one anywhere up to the
    /// size names none.
    pub(super) fn command_range(
        &self,
        call: &'static str,
        range: Range<u64>,
    ) -> Result<Range<u32>, Error> {
        let size = u64::from(self.0.size);
        if range.end > size || range.end < range.start {
            return Err(problem(
                call,
                IndirectCommandBufferProblem::Range {
                    location: range.start,
                    length: range.end.saturating_sub(range.start),
                    size,
                },
            ));
        }
        Ok(range.start as u32..range.end as u32)
    }

    /// The records' buffer: what an encoder makes resident for the buffer, and where a function's
    /// command encoders write (`gpu_resource_id`).
    pub(super) fn storage(&self) -> &Buffer {
        &self.0.storage
    }

    /// An encoder made the buffer resident for writing, so the GPU may write its records from
    /// now on: an execution's records are known only once the work before it completed, and it
    /// is replayed on the host then (`indirect_execute.rs`). For good: whatever the GPU wrote may
    /// stay in records the CPU did not write again. The first call settles, on its thread, the host
    /// steps of the executions `commit` recorded from the CPU's records and whose batches wait
    /// (`watch`), and only then sets the flag, which the scheduler reads.
    pub(super) fn written_by_gpu(&self) {
        if self.gpu_written() {
            return;
        }
        loop {
            let pending: Vec<Arc<Unforeseen>> = {
                let mut watched = lock(&self.0.unforeseen);
                watched.retain(|execution| execution.strong_count() > 0);
                let pending: Vec<_> = watched
                    .iter()
                    .filter_map(Weak::upgrade)
                    .filter(|execution| !execution.settled())
                    .collect();
                if pending.is_empty() {
                    watched.clear();
                    self.0.gpu_written.store(true, Ordering::Release);
                    return;
                }
                pending
            };
            // Settled without the lock: a commit meanwhile adds its executions, settled next.
            for execution in &pending {
                execution.settle(self.device());
            }
        }
    }

    /// A hold on the buffer for a command buffer whose work may write its records on the GPU,
    /// which the encoder retains until it completes ([`GpuWrite`]). The first one has the buffer
    /// keep, from now on, the programs of the states the program releases
    /// (`released_programs.rs`).
    pub(super) fn gpu_write(&self) -> GpuWrite {
        if !self.0.keeping.swap(true, Ordering::AcqRel) {
            self.device().shared().keepers.add(&self.0.kept);
        }
        lock(&self.0.gpu_writes).pending += 1;
        GpuWrite {
            device: self.device().clone(),
            writes: Arc::clone(&self.0.gpu_writes),
        }
    }

    /// For a blit's copy of `source`'s commands `range` into this buffer, which the copy's
    /// [`gpu_write`](Self::gpu_write) has keep programs already: keep what `source` keeps, and the
    /// programs of the released states the CPU set on the commands copied, whose records name
    /// them as Metal's copied records do (`released_programs.rs`).
    pub(super) fn keep_copied(&self, source: &IndirectCommandBuffer, range: Range<u32>) {
        let mut copied = lock(&source.0.kept).clone();
        let sets: Vec<CommandState> = {
            let pipelines = lock(&source.0.pipelines);
            range
                .filter_map(|index| pipelines.get(index as usize).cloned().flatten())
                .collect()
        };
        // Upgraded with no lock of either buffer held: a state dropped here releases its ID.
        for (stamp, program) in sets.iter().filter_map(CommandState::kept) {
            copied.keep(stamp.id, stamp.serial, program);
        }
        lock(&self.0.kept).keep_all(copied);
    }

    /// The stamp of the device's IDs (`ResourceIds::stamp`) as the GPU last could write the
    /// buffer's records: at the completion of the last command buffer that could
    /// ([`GpuWrite`]), or now (`u64::MAX`) while one has not completed; none where the GPU could
    /// not write them.
    fn window(&self) -> Option<u64> {
        if !self.gpu_written() {
            return None;
        }
        let writes = lock(&self.0.gpu_writes);
        Some(match writes.pending {
            0 => writes.end,
            _ => u64::MAX,
        })
    }

    /// The state that held pipeline state ID `named` as the GPU last could write the buffer's
    /// records ([`window`](Self::window)), released since or not: the state a GPU write of the ID
    /// resolved it to, as Metal's write resolves the ID as it runs and its record runs that state
    /// and answers its identifier after (`released_programs.rs`). None where the GPU could not
    /// write the records, or the ID's holder then is past the ones the device keeps.
    fn window_holder(&self, named: u64) -> Option<HolderThen> {
        let window = self.window()?;
        lock(&self.device().shared().resource_ids).pipeline_holder_by(named, window)
    }

    /// How many programs of released states the buffer keeps (`released_programs.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn kept_count(&self) -> usize {
        lock(&self.0.kept).count()
    }

    /// The `uniqueIdentifier` of the state of `kind` that held pipeline state ID `named` as the
    /// GPU last could write the buffer's records ([`window_holder`](Self::window_holder)), as
    /// Metal's record the GPU wrote answers (measured on an M3 Max by the consumer). None where
    /// that is none, or one of another kind.
    fn written_identifier(&self, named: u64, kind: StateKind) -> Option<u64> {
        self.window_holder(named)
            .filter(|then| then.kind == kind)
            .map(|then| then.identifier)
    }

    /// Whether the GPU may have written ID `named` into the record of a command the CPU set the
    /// state of kind `kind` whose hold of the ID had `serial` on, so that Metal's record names
    /// another state than the CPU's: the GPU could write the buffer's records after the state
    /// took the ID, and the ID's holder as it last could ([`window_holder`](Self::window_holder))
    /// is another state.
    fn may_name_another(&self, named: u64, kind: StateKind, serial: u64) -> bool {
        let Some(window) = self.window() else {
            return false;
        };
        serial < window
            && lock(&self.device().shared().resource_ids)
                .pipeline_holder_by(named, window)
                .map(|then| (then.kind, then.serial))
                != Some((kind, serial))
    }

    /// The program of the released state of kind `of` whose hold of ID `named` had `serial`, as a
    /// state, where the buffer keeps it (`released_programs.rs`).
    fn kept<S>(&self, named: u64, serial: u64, of: &Of<S>) -> Option<S> {
        let program = lock(&self.0.kept).program(named, serial).cloned()?;
        (of.kept)(&program)
    }

    /// For an execution of commands `range` on the GPU alone, which runs a record by its ID among
    /// the compute states `live` lists, each with its ID: those states, but that where the GPU
    /// last could write the records the ID's holder was a state the program released since, whose
    /// program the buffer keeps, the ID names that program instead, as a record the GPU wrote then
    /// runs it (`window_holder`), and no record the GPU writes later can name another, as no state
    /// took the ID before the window. `None` where the CPU set on a command of `range` the live
    /// state that took such an ID since: the ID names two states, which the host step tells apart
    /// (`name_state`).
    pub(super) fn gpu_states(
        &self,
        range: Range<u32>,
        live: Vec<ComputePipelineState>,
    ) -> Option<Vec<(u64, ComputePipelineState)>> {
        let mut states: Vec<(u64, ComputePipelineState)> = live
            .into_iter()
            .map(|state| (compute_id(&state), state))
            .collect();
        let ids: Vec<u64> = lock(&self.0.kept).ids().collect();
        let mut kept = Vec::new();
        for id in ids {
            let Some(then) = self.window_holder(id) else {
                continue;
            };
            if then.holds || then.kind != StateKind::Compute {
                continue;
            }
            if let Some(program) = self.kept(id, then.serial, &COMPUTE) {
                kept.push((id, program));
            }
        }
        if kept.is_empty() {
            return Some(states);
        }
        let sets: Vec<CommandState> = {
            let pipelines = lock(&self.0.pipelines);
            range
                .filter_map(|index| pipelines.get(index as usize).cloned().flatten())
                .collect()
        };
        // Upgraded with no lock of the buffer held: a state dropped here releases its ID.
        let set_live = sets
            .iter()
            .filter_map(CommandState::compute)
            .filter(|(alive, _)| alive.is_some())
            .map(|(_, stamp)| stamp.id)
            .collect::<std::collections::BTreeSet<u64>>();
        if kept.iter().any(|(id, _)| set_live.contains(id)) {
            return None;
        }
        states.retain(|(id, _)| kept.iter().all(|(taken, _)| taken != id));
        states.extend(kept);
        Some(states)
    }

    /// Watch `execution`, which `commit` recorded from the CPU's records: its host step is
    /// settled when an encoder makes the buffer resident for writing, or now, on the committing
    /// thread, where one did since the commit looked.
    pub(super) fn watch(&self, execution: &Arc<Unforeseen>) {
        {
            let mut watched = lock(&self.0.unforeseen);
            if !self.0.gpu_written.load(Ordering::Acquire) {
                watched.retain(|execution| execution.strong_count() > 0);
                watched.push(Arc::downgrade(execution));
                return;
            }
        }
        execution.settle(self.device());
    }

    /// Whether the GPU may have written the buffer's records: a runtime extension, with no Metal
    /// counterpart. It answers `true`, for good, from the first of these calls on: an encoder's
    /// `use_resource` or `use_resources` of the buffer with a usage holding
    /// `ResourceUsage::WRITE` (compute or render, whatever its pipeline states can do: a kernel
    /// or function that encodes, resets or copies commands is not asked), a blit encoder's
    /// `copy_indirect_command_buffer` into it of a range that is not empty, and its
    /// `reset_commands_in_buffer` of a range that is not empty. It is set at the call, before
    /// any of that work runs or is committed, and stays set though the work never runs (its
    /// command buffer is never committed, or fails). So it is broader than "work whose state can
    /// encode indirect commands wrote it": a buffer made resident for writing in an encoder whose
    /// states can encode none answers `true` too.
    ///
    /// It decides when the runtime reads an execution's records (`indirect_execute.rs`,
    /// `render/executions.rs`): while it answers `false`, an execution runs the records as they
    /// stand at its command buffer's `commit`, as [`commands`](Self::commands) reads them then;
    /// once it answers `true`, an execution committed since runs them as they stand once the
    /// work before it completed, read in its host step on the scheduler's thread, and so does an
    /// execution committed before whose batch was not yet submitted. What `commands` reads
    /// before then may not be what runs.
    pub fn gpu_written(&self) -> bool {
        self.0.gpu_written.load(Ordering::Acquire)
    }

    /// What the buffer's records may hold, from every route that writes them encoded so far: a
    /// runtime extension, with no Metal counterpart. The CPU's setters add what each command
    /// holds after the call (its draw or dispatch, a draw's primitive type among it, its barrier,
    /// and [`CommandWrites::PIPELINE`] where it names a state); an encoder that made the buffer
    /// resident for writing adds what its states may write
    /// ([`ComputePipelineState::indirect_command_writes`],
    /// [`RenderPipelineState::indirect_command_writes`]): the state set when it made the buffer
    /// resident, each set after, and the states its executions' commands name; a blit encoder's
    /// `copy_indirect_command_buffer` into it adds the source buffer's, now and as the source
    /// gains more. Resets add nothing, and nothing is taken away: a record written again keeps
    /// the earlier writes here, as a GPU write may stay. A clear bit is proven of every write
    /// encoded so far: no command holds such a draw, dispatch, barrier or primitive type (a
    /// point draw, where [`CommandWrites::may_draw_primitive_type`] of
    /// `PrimitiveType::Point` answers `false`). A route encoded later adds its own then.
    pub fn command_writes(&self) -> CommandWrites {
        self.0.written.writes()
    }

    /// Whether a command of the buffer may name a pipeline state whose functions may encode,
    /// reset or copy indirect commands, from every route that writes its records encoded so far:
    /// a runtime extension, with no Metal counterpart. The CPU's setters add the states they set
    /// (`can_encode_indirect_commands` of each); a function that may set a command's pipeline
    /// state or copy a command (`CommandWrites::PIPELINE`, `CommandWrites::COPY`) in an encoder
    /// that made the buffer resident for writing may name any state; a blit copy into it adds the
    /// source buffer's answer, now and later. `false` is proven of every write encoded so far, as
    /// [`command_writes`](Self::command_writes)' clear bits are.
    pub fn may_name_encoding_states(&self) -> bool {
        self.0.written.named().encodes
    }

    /// What its records may hold (`written_commands.rs`).
    pub(super) fn written(&self) -> &Arc<WrittenCommands> {
        &self.0.written
    }

    /// Whether the CPU ever set a render command's barrier, so an execution may split its render
    /// pass instance at one (`render/executions.rs`).
    pub(super) fn held_a_barrier(&self) -> bool {
        self.0.barriers.load(Ordering::Acquire)
    }

    /// How the first draw of primitive type 9 among commands `range` (inside the size), as their
    /// records stand, fails the command buffer that executes them, as Metal's GPU fails it there
    /// (`DrawPrimitiveType::Nine`; measured through indirect command buffers too): `None` where
    /// none draws one that fails it, and at once where the CPU never drew one. A kernel writes no
    /// such draw (it writes points).
    pub(super) fn nine_among(&self, range: Range<u32>) -> Result<Option<NineDraw>, Error> {
        if !self.0.nines.load(Ordering::Acquire) {
            return Ok(None);
        }
        let nine = DrawPrimitiveType::Nine.raw() as u32;
        for index in range {
            match self.record(index)?.action {
                Action::DrawPrimitives {
                    primitive_type,
                    vertex_count,
                    instance_count,
                    ..
                } if primitive_type == nine && vertex_count >= 3 && instance_count > 0 => {
                    return Ok(Some(NineDraw::Hangs));
                }
                Action::DrawIndexedPrimitives { primitive_type, .. } if primitive_type == nine => {
                    return Ok(Some(NineDraw::Faults));
                }
                _ => {}
            }
        }
        Ok(None)
    }

    /// Whether a render command's record may hold a barrier: the CPU set one
    /// ([`held_a_barrier`](Self::held_a_barrier)), or the GPU may have written the records (a
    /// kernel's `set_barrier`, or a blit copy of a command that holds one).
    pub(super) fn may_hold_barriers(&self) -> bool {
        self.held_a_barrier() || self.gpu_written()
    }

    /// The records of commands `range` (inside the size) as the GPU and the CPU left them, read
    /// once the work before an execution completed, checked as `CommandBuffer::decode` checks a
    /// producer's storage but command by command: a header that holds the producers' sticky
    /// encoding fault (`ReplayFaults::ENCODING`), or that a write overran, runs none of them; a
    /// record that does not decode, or that sets what its descriptor does not allow, is `None`
    /// (`ReplayFaults::RECORD`).
    pub(super) fn written_records(&self, range: Range<u32>) -> Result<WrittenRecords, Error> {
        let mut header = [0; COMMAND_BUFFER_HEADER_BYTES];
        // SAFETY: the header lies at the start of the mapped storage; the GPU's writes to it
        // completed (the caller's contract).
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bytes()?,
                header.as_mut_ptr(),
                COMMAND_BUFFER_HEADER_BYTES,
            );
        }
        let checked = self
            .0
            .records
            .validate_header(&header, self.0.size, self.0.inheritance);
        let header_fault = match checked {
            Ok(()) => None,
            Err(CommandBufferError::EncodingFault(_)) => Some(ReplayFaults::ENCODING),
            Err(_) => Some(ReplayFaults::RECORD),
        };
        if let Some(fault) = header_fault {
            return Ok(WrittenRecords {
                faults: fault,
                records: Vec::new(),
            });
        }
        self.device()
            .shared()
            .counts
            .indirect_records_read(range.len() as u64);
        let mut faults = ReplayFaults::default();
        let mut records = Vec::with_capacity(range.len());
        for index in range {
            let mut bytes = [0; COMMAND_BYTES];
            // SAFETY: as `record`'s.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.bytes()?.add(record_offset(index)),
                    bytes.as_mut_ptr(),
                    COMMAND_BYTES,
                );
            }
            let record = CommandRecord::decode(&bytes)
                .ok()
                .filter(|record| self.allows(record));
            if record.is_none() {
                faults = ReplayFaults(faults.0 | ReplayFaults::RECORD.0);
            }
            records.push(record);
        }
        Ok(WrittenRecords { faults, records })
    }

    /// Whether the descriptor allows what `record` sets: its action and buffers, and a pipeline
    /// state or buffers only where the buffer does not inherit them.
    fn allows(&self, record: &CommandRecord) -> bool {
        let inheritance = self.0.inheritance;
        let sets_buffers = || {
            BufferStage::of(self.is_render())
                .into_iter()
                .flat_map(|stage| record.bindings.buffers(stage))
                .any(Option::is_some)
        };
        self.0.records.validate_record(record).is_ok()
            && !(inheritance.pipeline == StateSource::Parent
                && (record.bindings.pipeline.is_some() || record.retained_pipeline.is_some()))
            && !(inheritance.buffers == StateSource::Parent && sets_buffers())
    }

    /// The records of commands `range` (inside the size) as they are now, each with the compute
    /// pipeline state it names for `reader` (`name_state`): the one the CPU set, or, for a
    /// record the GPU wrote since (a kernel, or the blit encoder's copy or reset), the state of its
    /// ID among those made with `supportIndirectCommandBuffers` that live now; `None` where it
    /// names none or none lives. With them, the commands an execution runs otherwise than Metal's
    /// may (`Released`).
    pub(super) fn records(
        &self,
        range: Range<u32>,
        reader: Reader,
    ) -> Result<(Vec<(CommandRecord, Option<ComputePipelineState>)>, Released), Error> {
        self.named_records(range, reader, &COMPUTE, || {
            ComputePipelineState::indirect_states(self.device())
        })
    }

    /// The records of commands `range` (inside the size) as they are now, each with the render
    /// pipeline state it names, as `records` names a compute one.
    pub(super) fn render_records(
        &self,
        range: Range<u32>,
        reader: Reader,
    ) -> Result<(Vec<(CommandRecord, Option<RenderPipelineState>)>, Released), Error> {
        self.named_records(range, reader, &RENDER, || {
            RenderPipelineState::indirect_states(self.device())
        })
    }

    /// The records of commands `range` as they are now, each with the state of kind `of` it
    /// names for `reader` (`name_state`), the live states of its ID listed by `live`, once.
    fn named_records<S: Clone>(
        &self,
        range: Range<u32>,
        reader: Reader,
        of: &Of<S>,
        live: impl Fn() -> Vec<S>,
    ) -> Result<(Vec<(CommandRecord, Option<S>)>, Released), Error> {
        let (records, sets) = self.read_records(range.clone())?;
        let read = range
            .zip(&records)
            .zip(&sets)
            .map(|((index, record), set)| (index, record, set.as_ref()));
        let (named, released) = self.name_records(read, reader, of, live);
        Ok((records.into_iter().zip(named).collect(), released))
    }

    /// The records of commands `range` (inside the size) as they are now, and the state the CPU
    /// set on each, read together.
    fn read_records(
        &self,
        range: Range<u32>,
    ) -> Result<(Vec<CommandRecord>, Vec<Option<CommandState>>), Error> {
        self.device()
            .shared()
            .counts
            .indirect_records_read(range.len() as u64);
        let pipelines = lock(&self.0.pipelines);
        range
            .map(|index| Ok((self.record(index)?, pipelines[index as usize].clone())))
            .collect()
    }

    /// The state of kind `of` each of `read` (a command's index, record and the state the CPU
    /// set on it) names for `reader` (`name_state`), the live states of its ID listed by `live`,
    /// once; and the commands an execution runs otherwise than Metal's may (`Released`).
    fn name_records<'r, S: Clone>(
        &self,
        read: impl Iterator<Item = (u32, &'r CommandRecord, Option<&'r CommandState>)>,
        reader: Reader,
        of: &Of<S>,
        live: impl Fn() -> Vec<S>,
    ) -> (Vec<Option<S>>, Released) {
        // The pipeline each command executes with: a retained one too.
        let runs = read.map(|(index, record, set)| {
            (
                index,
                record.bindings.pipeline.or(record.retained_pipeline),
                set,
            )
        });
        self.name_pipelines(runs, reader, of, live)
    }

    /// [`name_records`](Self::name_records) of each command's index, the pipeline ID its
    /// record executes with, and the state the CPU set on it.
    fn name_pipelines<'r, S: Clone>(
        &self,
        read: impl Iterator<Item = (u32, Option<NonZeroU64>, Option<&'r CommandState>)>,
        reader: Reader,
        of: &Of<S>,
        live: impl Fn() -> Vec<S>,
    ) -> (Vec<Option<S>>, Released) {
        // The live states are listed with no lock of the buffer held, once.
        let mut listed: Option<Vec<S>> = None;
        let mut unmatched = Vec::new();
        let named = read
            .map(|(index, runs, set)| {
                let named = runs?;
                let holder = || {
                    listed
                        .get_or_insert_with(&live)
                        .iter()
                        .find(|state| (of.id)(state) == named.get())
                        .cloned()
                };
                let at = (index, named.get());
                self.name_state(at, reader, set, of, holder, &mut unmatched)
            })
            .collect();
        (named, Released(unmatched))
    }

    /// For a host step: the compute state each of `records` names (commands from `start`, as
    /// the GPU and the CPU left them, `None` where a record runs nothing for a fault), among
    /// `states` by ID unless it names the state the CPU set (`name_state`); and the commands it
    /// runs otherwise than Metal's may (`Released`).
    pub(super) fn step_states(
        &self,
        start: u32,
        records: &[Option<CommandRecord>],
        states: &[ComputePipelineState],
    ) -> (Vec<Option<ComputePipelineState>>, Released) {
        self.written_states(start, records, states, &COMPUTE)
    }

    /// For a host step: the render state each of `records` names, as `step_states` a compute
    /// one.
    pub(super) fn step_render_states(
        &self,
        start: u32,
        records: &[Option<CommandRecord>],
        states: &[RenderPipelineState],
    ) -> (Vec<Option<RenderPipelineState>>, Released) {
        self.written_states(start, records, states, &RENDER)
    }

    /// Whether the CPU set a state on a command of `range` that the program released since:
    /// the command runs the state's program where its record names the ID the state had
    /// (`name_state`), which an execution the GPU alone runs, by the states alive at its commit
    /// (`gpu_execute.rs`), does not hold.
    pub(super) fn sets_a_released_state(&self, range: Range<u32>) -> bool {
        let sets: Vec<CommandState> = {
            let pipelines = lock(&self.0.pipelines);
            range
                .filter_map(|index| pipelines.get(index as usize).cloned().flatten())
                .collect()
        };
        // Upgraded with no lock of the buffer held: a state dropped here releases its ID.
        sets.iter().any(CommandState::released)
    }

    /// The state of kind `of` each of `records`, commands from `start`, names for a host step.
    fn written_states<S: Clone>(
        &self,
        start: u32,
        records: &[Option<CommandRecord>],
        states: &[S],
        of: &Of<S>,
    ) -> (Vec<Option<S>>, Released) {
        let sets: Vec<Option<CommandState>> = {
            let pipelines = lock(&self.0.pipelines);
            (start..)
                .take(records.len())
                .map(|index| pipelines.get(index as usize).cloned().flatten())
                .collect()
        };
        let mut unmatched = Vec::new();
        let named = records
            .iter()
            .zip(sets)
            .zip(start..)
            .map(|((record, set), index)| {
                let named = record
                    .as_ref()?
                    .bindings
                    .pipeline
                    .or(record.as_ref()?.retained_pipeline)?;
                let holder = || {
                    states
                        .iter()
                        .find(|state| (of.id)(state) == named.get())
                        .cloned()
                };
                let at = (index, named.get());
                self.name_state(at, Reader::Step, set.as_ref(), of, holder, &mut unmatched)
            })
            .collect();
        (named, Released(unmatched))
    }

    /// The state of kind `of` command `index`'s record runs for `reader`, where it names
    /// pipeline state ID `named` and the CPU set `set` on the command, as Metal's runs it:
    ///
    /// - Where `set` had the ID: the state while it lives; once the program released it, its
    ///   program (as a state of no ID), whoever holds the ID since, as Metal's command the CPU
    ///   encoded runs it. A host step cannot tell a record the GPU wrote naming that ID from
    ///   the CPU's: where the GPU may have written it while another state held the ID
    ///   ([`may_name_another`](Self::may_name_another)), Metal's run of a GPU write would differ
    ///   (`Unmatched::Unsure`). The getter reads it as the state holding the ID instead
    ///   (`holder`), none where none lives.
    /// - Else (the GPU wrote the ID): [`name_written`](Self::name_written).
    ///
    /// The commands an execution runs otherwise than Metal's may are added to `unmatched`.
    fn name_state<S>(
        &self,
        (index, named): (u32, u64),
        reader: Reader,
        set: Option<&CommandState>,
        of: &Of<S>,
        holder: impl FnOnce() -> Option<S>,
        unmatched: &mut Vec<(u32, u64, Unmatched)>,
    ) -> Option<S> {
        let own = set.and_then(of.own).filter(|(_, stamp)| stamp.id == named);
        let Some((alive, stamp)) = own else {
            return self.name_written((index, named), reader, of, holder, unmatched);
        };
        if alive.is_some() {
            return alive;
        }
        match reader {
            Reader::Getter => holder(),
            Reader::Commit => set.and_then(of.program),
            Reader::Step => {
                if self.may_name_another(named, of.kind, stamp.serial) {
                    unmatched.push((index, named, Unmatched::Unsure));
                }
                set.and_then(of.program)
            }
        }
    }

    /// The state of kind `of` command `index`'s record the GPU wrote runs for `reader`, where it
    /// names pipeline state ID `named`, as Metal's runs it: the state that held the ID as the GPU
    /// last could write the records ([`window_holder`](Self::window_holder)), which Metal's write
    /// resolved the ID to (`released_programs.rs`):
    ///
    /// - one of the other kind: none, as on Metal;
    /// - one that holds the ID still: the state of the kind holding it (`holder`);
    /// - one the program released since: its program, as a state of no ID, where the buffer keeps
    ///   it; else none (`Unmatched::Unrun`). The getter reads it as none.
    /// - none the device keeps (the records were the CPU's alone, or the ID's holders since are
    ///   more than it keeps): the state of the kind holding the ID (`holder`), else none, the
    ///   program having released a state of the kind that held it last (`Unmatched::Unrun`).
    fn name_written<S>(
        &self,
        (index, named): (u32, u64),
        reader: Reader,
        of: &Of<S>,
        holder: impl FnOnce() -> Option<S>,
        unmatched: &mut Vec<(u32, u64, Unmatched)>,
    ) -> Option<S> {
        let state = match self.window_holder(named) {
            Some(then) if then.kind != of.kind => return None,
            Some(then) if then.holds => return holder(),
            Some(_) if reader == Reader::Getter => return None,
            Some(then) => self.kept(named, then.serial, of),
            None => match holder() {
                Some(state) => return Some(state),
                None if self
                    .released_holder(named)
                    .is_some_and(|(kind, _)| kind == of.kind) =>
                {
                    None
                }
                None => return None,
            },
        };
        if state.is_none() && reader != Reader::Getter {
            unmatched.push((index, named, Unmatched::Unrun));
        }
        state
    }

    /// The kind and hold serial of the state that last held pipeline state ID `named`, where
    /// the program released it and no state took the ID since (`ResourceIds::released_pipeline`).
    fn released_holder(&self, named: u64) -> Option<(StateKind, u64)> {
        lock(&self.device().shared().resource_ids).released_pipeline(named)
    }

    /// Announce `Degradation::ReleasedCommandState` for an execution of commands of which
    /// `released` run otherwise than Metal's may, from the call that records the execution:
    /// `commit`, or the host step, on the scheduler's thread, for one it records.
    pub(super) fn degrade_released(&self, released: &Released) {
        let Some(&(index, named, unmatched)) = released.0.first() else {
            return;
        };
        self.device()
            .degrade(Degradation::ReleasedCommandState, || {
                (
                    format!(
                        "{} command {index} of an indirect command buffer, naming pipeline state \
                         ID {named}{}",
                        if self.is_render() {
                            "render"
                        } else {
                            "compute"
                        },
                        match released.0.len() {
                            1 => String::new(),
                            n => format!(" (and {} more commands)", n - 1),
                        }
                    ),
                    match unmatched {
                        Unmatched::Unrun => {
                            "the program released a state of the command's kind holding the ID, \
                             and no state took it since"
                        }
                        Unmatched::Unsure => {
                            "the program released the state set on the command, which was not \
                             the ID's last holder, and the GPU may have written the record"
                        }
                    }
                    .to_owned(),
                )
            });
    }

    /// The bytes of commands `range`'s records in the storage.
    pub(super) fn record_bytes(range: Range<u32>) -> Range<u64> {
        record_offset(range.start) as u64..record_offset(range.end) as u64
    }

    /// The typed command of `record`, its pipeline state `pipeline`.
    fn compute_command(
        &self,
        (index, record): (u32, &CommandRecord),
        pipeline: Option<ComputePipelineState>,
        pipeline_unique_identifier: Option<u64>,
    ) -> EncodedComputeCommand {
        let dispatch = dispatch_of(&record.action);
        let descriptor = &self.0.descriptor;
        let [x, y, z, width, height, depth] = record.stage_in_region.map(u64::from);
        EncodedComputeCommand {
            dispatch,
            command_type: command_type(&record.action),
            barrier: record.barrier,
            pipeline,
            pipeline_id: record.bindings.pipeline.map(|id| ResourceId(id.get())),
            pipeline_unique_identifier,
            kernel_buffers: self.buffers_of(
                (index, record),
                BufferStage::Kernel,
                descriptor.max_kernel_buffer_bind_count,
            ),
            getters: Getters::of(descriptor),
            threadgroup_memory_lengths: record.threadgroup_memory_lengths[..held(
                descriptor.max_kernel_threadgroup_memory_bind_count,
                THREADGROUP_BUFFER_ARGUMENT_COUNT,
            ) as usize]
                .to_vec(),
            stage_in_region: Region {
                origin: Origin::new(x, y, z),
                size: Size::new(width, height, depth),
            },
            imageblock_width: record.imageblock_size[0].into(),
            imageblock_height: record.imageblock_size[1].into(),
        }
    }

    /// The typed render command of `record`, its pipeline state `pipeline`. A draw of a primitive
    /// type or an index type Metal has no value for (a kernel's write) reads as none: it runs
    /// nothing.
    fn render_command(
        &self,
        (index, record): (u32, &CommandRecord),
        pipeline: Option<RenderPipelineState>,
        pipeline_unique_identifier: Option<u64>,
    ) -> EncodedRenderCommand {
        let draw = draw_of(&record.action);
        let descriptor = &self.0.descriptor;
        EncodedRenderCommand {
            draw,
            command_type: command_type(&record.action),
            barrier: record.barrier,
            pipeline,
            pipeline_id: record
                .bindings
                .pipeline
                .or(record.retained_pipeline)
                .map(|id| ResourceId(id.get())),
            pipeline_unique_identifier,
            vertex_buffers: self.buffers_of(
                (index, record),
                BufferStage::Vertex,
                descriptor.max_vertex_buffer_bind_count,
            ),
            fragment_buffers: self.buffers_of(
                (index, record),
                BufferStage::Fragment,
                descriptor.max_fragment_buffer_bind_count,
            ),
            object_buffers: self.buffers_of(
                (index, record),
                BufferStage::Object,
                descriptor.max_object_buffer_bind_count,
            ),
            mesh_buffers: self.buffers_of(
                (index, record),
                BufferStage::Mesh,
                descriptor.max_mesh_buffer_bind_count,
            ),
            getters: Getters::of(descriptor),
        }
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<String> {
        self.0.storage.attributes().label()
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        self.0.storage.attributes().set_label(label);
    }

    /// `storageMode`: the mode asked for.
    pub fn storage_mode(&self) -> StorageMode {
        self.0
            .options
            .storage_mode()
            .expect("checked when the buffer was made")
    }

    /// `cpuCacheMode`: the mode asked for.
    pub fn cpu_cache_mode(&self) -> CpuCacheMode {
        self.0
            .options
            .cpu_cache_mode()
            .expect("checked when the buffer was made")
    }

    /// `hazardTrackingMode`: `Tracked` or `Untracked`, `Default` resolved to `Tracked`, as Metal
    /// resolves it.
    pub fn hazard_tracking_mode(&self) -> HazardTrackingMode {
        self.0.storage.hazard_tracking_mode()
    }

    /// `resourceOptions`: the options asked for, the hazard tracking mode resolved.
    pub fn resource_options(&self) -> ResourceOptions {
        self.0.options
    }

    /// `allocatedSize`: Metal's, the bytes its layout of the descriptor's commands takes, in
    /// whole 16-KiB pages, whatever the runtime's own records take (the rule matched every row
    /// measured on an M4, over a million, and on an M3 Max).
    pub fn allocated_size(&self) -> u64 {
        metal_allocated_size(&self.0.descriptor, self.0.size)
    }

    /// `heap`: none (an indirect command buffer is never made in a heap).
    pub fn heap(&self) -> Option<&super::Heap> {
        None
    }

    /// `heapOffset`: 0.
    pub fn heap_offset(&self) -> u64 {
        0
    }

    /// `isAliasable`: false.
    pub fn is_aliasable(&self) -> bool {
        false
    }

    /// `setPurgeableState:`: the state before the call, or the state now for `KeepCurrent`, as a
    /// buffer's. `Empty` resets every command, as Metal's commands read after it.
    pub fn set_purgeable_state(&self, state: PurgeableState) -> Result<PurgeableState, Error> {
        let answer = self.0.storage.attributes().set_purgeable_state(state);
        if state == PurgeableState::Empty {
            self.reset_commands(0..self.0.size)?;
        }
        Ok(answer)
    }
}

/// The records of an execution's commands as the GPU and the CPU left them
/// ([`IndirectCommandBuffer::written_records`]).
pub(super) struct WrittenRecords {
    /// Why commands run nothing: the header's (then `records` is empty), or the union of the
    /// records' that are `None`.
    pub(super) faults: ReplayFaults,
    /// Each command's record, in range order; `None` where it does not decode or sets what the
    /// descriptor does not allow.
    pub(super) records: Vec<Option<CommandRecord>>,
}

/// A buffer a command's setter sets: at `slot` of `stage`, below the descriptor's `limit`, at
/// `offset` and `attribute_stride` (none: the pipeline's own); `None` unsets the slot.
struct BufferSet<'a> {
    stage: BufferStage,
    limit: u32,
    slot: u32,
    buffer: Option<&'a Buffer>,
    offset: u64,
    attribute_stride: Option<u64>,
}

/// What Metal's address getter of a stage answers at `index` of `buffers`, the slots below the
/// descriptor's `count` of the stage: the address set (a nil buffer's offset), 0 where none is
/// set; none at or past `count`, where Metal's answer was not measured.
fn address_at(buffers: &[Option<IndirectBufferBinding>], count: u32, index: u64) -> Option<u64> {
    (index < u64::from(count)).then(|| {
        let slot = usize::try_from(index).ok().and_then(|at| buffers.get(at));
        slot.copied()
            .flatten()
            .map_or(0, |binding| binding.address.0)
    })
}

/// What Metal's attribute stride getter of a stage answers at `index` of `buffers`, as
/// [`address_at`] reads them: all ones on a buffer without dynamic attribute strides
/// (`dynamic` false); else the stride set, all ones for the pipeline's own, and 0 where none
/// is set.
fn stride_at(
    buffers: &[Option<IndirectBufferBinding>],
    (count, dynamic): (u32, bool),
    index: u64,
) -> Option<u64> {
    (index < u64::from(count)).then(|| {
        let slot = usize::try_from(index).ok().and_then(|at| buffers.get(at));
        match (dynamic, slot.copied().flatten()) {
            (false, _) => u64::MAX,
            (true, None) => 0,
            (true, Some(binding)) => binding.attribute_stride.unwrap_or(u64::MAX),
        }
    })
}

/// `getCommandType`'s answer for a record of `action`: the command type of its dispatch or draw,
/// none for an empty record (on an M4: `0x20` and `0x40` for the dispatches, 1
/// and 2 for draws and indexed draws, 0 fresh and after a reset), and none for an indexed draw of
/// primitive type 9 (the M3 Max, macOS 27.0.1: 0 there, 2 for indexed draws of types 0 to 8, and 1
/// for direct draws of types 0 to 9).
fn command_type(action: &Action) -> IndirectCommandType {
    match *action {
        Action::Empty => IndirectCommandType::empty(),
        Action::DrawIndexedPrimitives { primitive_type, .. }
            if u64::from(primitive_type) == DrawPrimitiveType::Nine.raw() =>
        {
            IndirectCommandType::empty()
        }
        Action::DispatchThreadgroups { .. } => IndirectCommandType::CONCURRENT_DISPATCH,
        Action::DispatchThreads { .. } => IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
        Action::DrawPrimitives { .. } => IndirectCommandType::DRAW,
        Action::DrawIndexedPrimitives { .. } => IndirectCommandType::DRAW_INDEXED,
        Action::DrawPatches { .. } => IndirectCommandType::DRAW_PATCHES,
        Action::DrawIndexedPatches { .. } => IndirectCommandType::DRAW_INDEXED_PATCHES,
        Action::DrawMeshThreadgroups { .. } => IndirectCommandType::DRAW_MESH_THREADGROUPS,
        Action::DrawMeshThreads { .. } => IndirectCommandType::DRAW_MESH_THREADS,
    }
}

/// A record's dispatch, of `action`; none for an action that dispatches nothing.
fn dispatch_of(action: &Action) -> Option<IndirectDispatch> {
    let size =
        |[width, height, depth]: [u32; 3]| Size::new(width.into(), height.into(), depth.into());
    match *action {
        Action::DispatchThreadgroups { grid, local_size } => Some(IndirectDispatch::Threadgroups {
            threadgroups_per_grid: size(grid),
            threads_per_threadgroup: size(local_size),
        }),
        Action::DispatchThreads { grid, local_size } => Some(IndirectDispatch::Threads {
            threads_per_grid: size(grid),
            threads_per_threadgroup: size(local_size),
        }),
        _ => None,
    }
}

/// A record's draw as Metal's private getters answer it (`drawArguments`, `drawIndexedArguments`):
/// [`draw_of`]'s, a draw of primitive type 7, 8 or 9 read as one of points, 0 (the M3 Max, macOS
/// 27.0.1: each of the three, set by the CPU, direct or indexed, read back 0; 0 to 6 read back as
/// set).
fn answered_draw(action: &Action) -> Option<IndirectDraw> {
    let mut draw = draw_of(action)?;
    if let IndirectDraw::Primitives { primitive_type, .. }
    | IndirectDraw::IndexedPrimitives { primitive_type, .. } = &mut draw
    {
        if matches!(
            primitive_type,
            DrawPrimitiveType::Quads | DrawPrimitiveType::QuadStrip | DrawPrimitiveType::Nine
        ) {
            *primitive_type = PrimitiveType::Point.into();
        }
    }
    Some(draw)
}

/// A record's draw, of `action`: none for an action that draws nothing, and for a draw of a
/// primitive type or an index type Metal has no value for, which runs nothing.
fn draw_of(action: &Action) -> Option<IndirectDraw> {
    let size =
        |[width, height, depth]: [u32; 3]| Size::new(width.into(), height.into(), depth.into());
    match *action {
        Action::DrawPrimitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => DrawPrimitiveType::from_raw(primitive_type.into())
            .ok()
            .map(|primitive_type| IndirectDraw::Primitives {
                primitive_type,
                vertex_start: vertex_start.into(),
                vertex_count: vertex_count.into(),
                instance_count: instance_count.into(),
                base_instance: base_instance.into(),
            }),
        Action::DrawIndexedPrimitives {
            primitive_type,
            index_count,
            index_type,
            index_buffer,
            instance_count,
            base_vertex,
            base_instance,
        } => DrawPrimitiveType::from_raw(primitive_type.into())
            .ok()
            .zip(IndexType::try_from(index_type).ok())
            .map(
                |(primitive_type, index_type)| IndirectDraw::IndexedPrimitives {
                    primitive_type,
                    index_count: index_count.into(),
                    index_type,
                    index_buffer: GpuAddress(index_buffer),
                    instance_count: instance_count.into(),
                    base_vertex: base_vertex.into(),
                    base_instance: base_instance.into(),
                },
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
            factor_scale: _,
        } => Some(IndirectDraw::Patches {
            control_points: control_points.into(),
            patch_start: patch_start.into(),
            patch_count: patch_count.into(),
            patch_index_buffer: (patch_index_buffer != 0).then_some(GpuAddress(patch_index_buffer)),
            instance_count: instance_count.into(),
            base_instance: base_instance.into(),
            factor_buffer: GpuAddress(factor_buffer),
            factor_instance_stride: factor_instance_stride.into(),
        }),
        Action::DrawIndexedPatches {
            control_points,
            patch_start,
            patch_count,
            patch_index_buffer,
            control_point_index_buffer,
            instance_count,
            base_instance,
            factor_buffer,
            factor_instance_stride,
            factor_scale: _,
        } => Some(IndirectDraw::IndexedPatches {
            control_points: control_points.into(),
            patch_start: patch_start.into(),
            patch_count: patch_count.into(),
            patch_index_buffer: (patch_index_buffer != 0).then_some(GpuAddress(patch_index_buffer)),
            control_point_index_buffer: (control_point_index_buffer != 0)
                .then_some(GpuAddress(control_point_index_buffer)),
            instance_count: instance_count.into(),
            base_instance: base_instance.into(),
            factor_buffer: GpuAddress(factor_buffer),
            factor_instance_stride: factor_instance_stride.into(),
        }),
        Action::DrawMeshThreadgroups {
            threadgroups,
            object_threads,
            mesh_threads,
        } => Some(IndirectDraw::MeshThreadgroups {
            threadgroups_per_grid: size(threadgroups),
            threads_per_object_threadgroup: size(object_threads),
            threads_per_mesh_threadgroup: size(mesh_threads),
        }),
        Action::DrawMeshThreads {
            threads,
            object_threads,
            mesh_threads,
        } => Some(IndirectDraw::MeshThreads {
            threads_per_grid: size(threads),
            threads_per_object_threadgroup: size(object_threads),
            threads_per_mesh_threadgroup: size(mesh_threads),
        }),
        Action::Empty | Action::DispatchThreadgroups { .. } | Action::DispatchThreads { .. } => {
            None
        }
    }
}

/// The byte offset of command `index`'s record in the storage.
fn record_offset(index: u32) -> usize {
    COMMAND_BUFFER_HEADER_BYTES + index as usize * COMMAND_BYTES
}

/// The runtime's error for a record that does not decode (a GPU write, the program's race).
fn undecodable() -> Error {
    internal("an indirect command's record does not decode")
}

fn problem(call: &'static str, problem: IndirectCommandBufferProblem) -> Error {
    invalid(call, Problem::IndirectCommandBuffer(problem))
}

/// `MTLIndirectComputeCommand`: the setters of one command of an indirect command buffer. Each
/// writes the command's record at once; a command encoded for execution reads the record as it
/// is when the execution runs.
#[derive(Clone, Debug)]
pub struct IndirectComputeCommand {
    icb: IndirectCommandBuffer,
    index: u32,
}

impl IndirectComputeCommand {
    /// Write `field` into the record, with `set` as the state the CPU set where it names one
    /// (`IndirectCommandBuffer::edit`).
    fn edit(&self, field: RecordField, set: Option<CommandState>) -> Result<(), Error> {
        self.icb.edit(self.index, field, set)
    }

    fn descriptor(&self) -> &IndirectCommandBufferDescriptor {
        &self.icb.0.descriptor
    }

    /// `setComputePipelineState:`. Refused (`Invalid`): a buffer that inherits its pipeline
    /// state, a state made without `supportIndirectCommandBuffers` (both end Metal's process), a
    /// state of another device.
    pub fn set_compute_pipeline_state(&self, state: &ComputePipelineState) -> Result<(), Error> {
        const CALL: &str = "set_compute_pipeline_state";
        if !state.device().same(self.icb.device()) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        if self.descriptor().inherit_pipeline_state {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::InheritedPipelineState,
            ));
        }
        let (Some(id), Some(serial)) = (
            NonZeroU64::new(state.gpu_resource_id().0),
            state.id_serial(),
        ) else {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::PipelineWithoutSupport,
            ));
        };
        let stamp = IdStamp {
            id: id.get(),
            serial,
            identifier: state.identifier_value(),
        };
        let named = Named {
            encodes: state.can_encode_indirect_commands(),
            writes: state.indirect_command_writes(),
        };
        let set = CommandState::Compute(state.downgrade(), state.program(), stamp, named);
        self.edit(RecordField::Pipeline(id), Some(set))
    }

    /// `setKernelBuffer:offset:atIndex:`: buffer `index` at `buffer`'s address plus `offset`'s
    /// low 32 bits, as Metal keeps it, at the pipeline's own attribute stride; `None` binds no
    /// buffer and keeps the offset's low 32 bits as the slot's address, as Metal's nil does: its
    /// getter answers them, and a function the command runs sees them as the buffer's address (an
    /// execution on the host binds no buffer there, its function seeing none, announced as
    /// `indirect-nil-buffer-offset`). Taken at an index at or past the descriptor's
    /// `max_kernel_buffer_bind_count` and kept nowhere, as plain Metal takes it (where its own
    /// validation stops the call): no execution binds such a slot and no getter answers it.
    /// Measured on an M3 Max under macOS 27.0.1: a buffer of `maxKernelBufferBindCount` 2 took
    /// the call at index 2 and at index 4 with no error, and its `size` still read 3 commands.
    /// Refused (`Invalid`): a buffer of another device. On a buffer that inherits its buffers it
    /// is kept and read back, and executions bind the encoder's.
    pub fn set_kernel_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer("set_kernel_buffer", buffer, offset, None, index)
    }

    /// `setKernelBuffer:offset:attributeStride:atIndex:`: as
    /// [`set_kernel_buffer`](Self::set_kernel_buffer), at attribute stride `stride` (all ones is
    /// the pipeline's own, as Metal reads it back). Where the descriptor does not support dynamic
    /// attribute strides the stride is dropped and the buffer set as `set_kernel_buffer` sets it,
    /// as plain Metal keeps it (its stride reads back all ones, the pipeline's own; an M3 Max and
    /// an M4; Metal's API validation stops the call). `None` sets a nil buffer at the stride, which
    /// the stride getter answers.
    pub fn set_kernel_buffer_with_stride(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_kernel_buffer_with_stride";
        let dynamic = self.descriptor().support_dynamic_attribute_stride;
        let stride = (dynamic && stride != u64::MAX).then_some(stride);
        self.set_buffer(CALL, buffer, offset, stride, index)
    }

    fn set_buffer(
        &self,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        attribute_stride: Option<u64>,
        index: u32,
    ) -> Result<(), Error> {
        let set = BufferSet {
            stage: BufferStage::Kernel,
            limit: self.descriptor().max_kernel_buffer_bind_count,
            slot: index,
            buffer,
            offset,
            attribute_stride,
        };
        self.icb.set_buffer(self.index, call, set)
    }

    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`: the command dispatches
    /// `threadgroups` of `threads_per_threadgroup`, each axis's low 32 bits, as Metal keeps them.
    /// Every compute buffer's records hold it: on a buffer of dispatches of threads it runs as
    /// on its own, as plain Metal runs it, where Metal's validation stops the call
    /// ("MTLIndirectCommandTypeConcurrentDispatch is not enabled").
    pub fn concurrent_dispatch_threadgroups(
        &self,
        threadgroups: Size,
        threads_per_threadgroup: Size,
    ) -> Result<(), Error> {
        self.set_dispatch(Action::DispatchThreadgroups {
            grid: low_bits(threadgroups),
            local_size: low_bits(threads_per_threadgroup),
        })
    }

    /// `concurrentDispatchThreads:threadsPerThreadgroup:`: the command dispatches `threads` in
    /// groups of `threads_per_threadgroup`, each axis's low 32 bits, as Metal keeps them. Every
    /// compute buffer's records hold it: on a buffer of dispatches of threadgroups it runs as on
    /// its own, as plain Metal runs it, where Metal's validation stops the call.
    pub fn concurrent_dispatch_threads(
        &self,
        threads: Size,
        threads_per_threadgroup: Size,
    ) -> Result<(), Error> {
        self.set_dispatch(Action::DispatchThreads {
            grid: low_bits(threads),
            local_size: low_bits(threads_per_threadgroup),
        })
    }

    /// Set the record's dispatch `action`, which every compute buffer's records hold
    /// (`CommandDescriptor::holds`).
    fn set_dispatch(&self, action: Action) -> Result<(), Error> {
        self.edit(RecordField::Action(action), None)
    }

    /// `setBarrier`: the command waits for the commands before it.
    pub fn set_barrier(&self) -> Result<(), Error> {
        self.edit(RecordField::Barrier(true), None)
    }

    /// `clearBarrier`.
    pub fn clear_barrier(&self) -> Result<(), Error> {
        self.edit(RecordField::Barrier(false), None)
    }

    /// `setThreadgroupMemoryLength:atIndex:`: as set, a multiple of 16 or not and past the
    /// device's maximum alike, where Metal keeps it, at any index, the descriptor's
    /// `max_kernel_threadgroup_memory_bind_count` or not: 38 lengths a command, of one array, so
    /// an index of 38 or more sets a later command's length (`index - 38` of the next command's,
    /// and so on), and one past the last command's is kept beside the buffer for the getter. No
    /// function binds one past index 30: Metal keeps lengths 31 to 37 for its private getter
    /// alone, as the runtime does. Past the last command's lengths Metal keeps each command's
    /// words: 2 a command where the buffer does not inherit its pipeline state, then 8 a command;
    /// a length set there is reset and copied with that command, as on Metal: a CPU reset clears
    /// the 2, a GPU reset both, and a GPU copy carries both (and clears lengths 36 and 37 where
    /// the commands hold the 2). Differs: Metal's pipeline setter writes the 2 words and clears
    /// lengths 36 and 37, and its dispatch and imageblock setters write the 8 (the threadgroup
    /// counts and size, the imageblock's width and height), so its getter answers those there;
    /// the runtime keeps the lengths set there alone, and its pipeline setter leaves lengths 36
    /// and 37. A GPU copy writes a value of its own two places past the commands' words, which
    /// the runtime does not. Far past the last command's lengths (about 4000 places on an M3
    /// Max), Metal's setter ends the process; the runtime keeps the length.
    pub fn set_threadgroup_memory_length(&self, length: u32, index: u32) -> Result<(), Error> {
        self.icb.set_length(self.index, index, length)
    }

    /// `setStageInRegion:`: each origin and size value's low 32 bits, as Metal keeps them.
    pub fn set_stage_in_region(&self, region: Region) -> Result<(), Error> {
        let Region { origin, size } = region;
        let values = [
            origin.x,
            origin.y,
            origin.z,
            size.width,
            size.height,
            size.depth,
        ];
        let region = values.map(|value| value as u32);
        self.edit(RecordField::StageInRegion(region), None)
    }

    /// `setImageblockWidth:height:`: as set. A side past 65535, which Metal keeps, is refused
    /// (`NotImplemented(IndirectImageblockSize)`): the record holds 16 bits a side, as a
    /// kernel's `set_imageblock_size` writes it.
    pub fn set_imageblock_size(&self, width: u64, height: u64) -> Result<(), Error> {
        let (Ok(w), Ok(h)) = (u16::try_from(width), u16::try_from(height)) else {
            return Err(Error::NotImplemented(RuntimeGap::IndirectImageblockSize {
                width,
                height,
            }));
        };
        self.edit(RecordField::ImageblockSize([w, h]), None)
    }

    /// `reset`: the command as a fresh buffer's, but for the pipeline state it executes with,
    /// which Metal's reset keeps while its getter reports none (measured on the M3 Max and the M4):
    /// a dispatch encoded after the reset runs it. [`IndirectCommandBuffer:: commands`] reports
    /// none, as Metal's getter does; a pipeline setter replaces it. The imageblock size is kept
    /// too, as plain Metal's reads back after `reset` (the size set before it, measured by the
    /// integration consumer on an M3 without API validation).
    pub fn reset(&self) -> Result<(), Error> {
        self.icb.reset_command(self.index, |record, _| {
            let retained = record.bindings.pipeline.or(record.retained_pipeline);
            *record = CommandRecord {
                retained_pipeline: retained,
                imageblock_size: record.imageblock_size,
                ..CommandRecord::default()
            };
        })
    }
    /// `getCommandType`, Apple's private getter: the command's
    /// [`command_type`](EncodedComputeCommand::command_type) as
    /// [`IndirectCommandBuffer::commands`] reads it now, its record's action alone read. Each of
    /// the command's getters reads its field of the record as it is at the call (a runtime
    /// extension standing for Metal's private getters, as `commands` is), so a field read after
    /// the GPU may have written the records may not be what an execution committed earlier
    /// runs ([`IndirectCommandBuffer::gpu_written`]).
    pub fn command_type(&self) -> Result<IndirectCommandType, Error> {
        self.icb
            .read_record(self.index, |view| command_type(&view.action()))
    }

    /// `hasBarrier`, Apple's private getter: the command's
    /// [`barrier`](EncodedComputeCommand::barrier), read alone.
    pub fn has_barrier(&self) -> Result<bool, Error> {
        self.icb.read_record(self.index, |view| view.barrier())
    }

    /// `dispatchThreadgroupsArguments` and `dispatchThreadsArguments`, Apple's private getters:
    /// the command's [`dispatch`](EncodedComputeCommand::dispatch), read alone.
    pub fn dispatch(&self) -> Result<Option<IndirectDispatch>, Error> {
        self.icb
            .read_record(self.index, |view| dispatch_of(&view.action()))
    }

    /// The kernel buffer set at `index`: the command's
    /// [`kernel_buffers`](EncodedComputeCommand::kernel_buffers)`[index]`, read alone, none at or
    /// past the descriptor's `max_kernel_buffer_bind_count`. Metal's private getters of the slot
    /// (`getKernelBufferAddressAtIndex:`, `getKernelAttributeStrideAtIndex:`) answer it as
    /// [`EncodedComputeCommand::kernel_buffer_address`] and
    /// [`EncodedComputeCommand::kernel_attribute_stride`] do.
    pub fn kernel_buffer(&self, index: u64) -> Result<Option<IndirectBufferBinding>, Error> {
        let count = self.descriptor().max_kernel_buffer_bind_count;
        self.icb
            .read_buffer(self.index, BufferStage::Kernel, count, index)
    }

    /// `getThreadgroupMemoryLengthAtIndex:`, Apple's private getter: the length at `index`'s low
    /// 32 bits, as Metal's takes them, where Metal keeps it, as
    /// [`set_threadgroup_memory_length`](Self::set_threadgroup_memory_length) sets it (below 31,
    /// the command's
    /// [`threadgroup_memory_lengths`](EncodedComputeCommand::threadgroup_memory_lengths)`[index]`;
    /// at 38 or more, a later command's; past the last command's, the commands' words there),
    /// read alone: 0 where none was set, at any index, the descriptor's
    /// `max_kernel_threadgroup_memory_bind_count` or not; always some. Differs: past the last
    /// command's lengths Metal's getter answers what its pipeline, dispatch and imageblock
    /// setters wrote there, the runtime's only lengths set there (the setter's docs); far past
    /// them (an M3 Max: 100000 places past command 0's of 3), Metal's getter ends the process,
    /// and the runtime's answers.
    pub fn threadgroup_memory_length(&self, index: u64) -> Result<Option<u32>, Error> {
        self.icb.length(self.index, index as u32).map(Some)
    }

    /// `getStageInRegion`, Apple's private getter: the command's
    /// [`stage_in_region`](EncodedComputeCommand::stage_in_region), read alone.
    pub fn stage_in_region(&self) -> Result<Region, Error> {
        let [x, y, z, width, height, depth] = self
            .icb
            .read_record(self.index, |view| view.stage_in_region())?
            .map(u64::from);
        Ok(Region {
            origin: Origin::new(x, y, z),
            size: Size::new(width, height, depth),
        })
    }

    /// The imageblock size Apple's private getter answers: the command's
    /// [`imageblock_width`](EncodedComputeCommand::imageblock_width) and
    /// [`imageblock_height`](EncodedComputeCommand::imageblock_height), read alone.
    pub fn imageblock_size(&self) -> Result<(u64, u64), Error> {
        let [width, height] = self
            .icb
            .read_record(self.index, |view| view.imageblock_size())?;
        Ok((width.into(), height.into()))
    }

    /// The command's [`pipeline`](EncodedComputeCommand::pipeline), read alone: the state set on
    /// it, none when unset, reset, or when no state of the ID it names lives.
    pub fn pipeline_state(&self) -> Result<Option<ComputePipelineState>, Error> {
        Ok(self.icb.compute_pipeline_now(self.index)?.0)
    }

    /// The command's [`pipeline_id`](EncodedComputeCommand::pipeline_id), read alone: the ID of
    /// the state its record names, none where it names none.
    pub fn pipeline_id(&self) -> Result<Option<ResourceId>, Error> {
        self.icb.read_record(self.index, |view| {
            view.pipeline().map(|id| ResourceId(id.get()))
        })
    }

    /// `getPipelineStateUniqueIdentifier`, Apple's private getter: the command's
    /// [`pipeline_unique_identifier`](EncodedComputeCommand::pipeline_unique_identifier), read
    /// alone, whose kind the device announces as `commands` announces it.
    pub fn pipeline_unique_identifier(&self) -> Result<Option<u64>, Error> {
        let (_, identifier) = self.icb.compute_pipeline_now(self.index)?;
        self.icb.announce_identifier(identifier);
        Ok(identifier)
    }
}

/// Each axis of `size`'s low 32 bits.
fn low_bits(size: Size) -> [u32; 3] {
    [size.width, size.height, size.depth].map(|axis| axis as u32)
}

/// A check of the commands an execution runs (a runtime extension; Metal has none), given with
/// what it looks for ([`IndirectCommandCheckNeeds`]) to `execute_commands_in_buffer_checked` or
/// `execute_commands_in_buffer_indirect_checked` of a render or a compute encoder. Where the
/// runtime reads the records of the commands it will record for the execution, it calls the check
/// once for each of them, in order, before it records any command of the range, with the
/// command's index in the buffer and the command as [`IndirectCommandBuffer::commands`] reads it
/// then: at `commit`, on the thread that commits, where it records the execution from the CPU's
/// records as they stand then; in a host step, on the thread that runs the step, where the step
/// records it from the records as the GPU and the CPU left them. An execution `commit` recorded
/// that a host step records again before its batch is submitted (an encoder made its buffer
/// resident for writing since, or a route outgrew its render pass's form on the GPU) is checked
/// again there. Neither calls it
/// where what the records may hold leaves it nothing to see ([`IndirectCommandCheckNeeds`]). A
/// command whose record faulted runs nothing and is not checked; nor is any where the buffer's
/// header faulted. The check may end the process, and must not unwind; returning lets the commands
/// run. A command buffer dropped before its host step never calls the step's check.
pub type IndirectCommandCheck = Box<dyn FnMut(u64, &IndirectCommand) + Send + 'static>;

/// What an [`IndirectCommandCheck`] looks for among an execution's commands (a runtime extension;
/// Metal has none), given with it to `execute_commands_in_buffer_checked` and
/// `execute_commands_in_buffer_indirect_checked`. Where what the buffer's records may hold, as
/// the runtime last decides on before it would read them for the check, rules out every kind it
/// names, the check has nothing to see and is not called: an execution may then run on the GPU
/// alone, and neither `commit` nor a host step calls the check, or reads the records for it. A
/// record that holds no draw and no dispatch runs nothing, so a buffer into whose records no route
/// wrote one leaves every check nothing to see. Where what the records may hold grows past what
/// left the check nothing to see by the time an execution that runs on the GPU alone is ready to
/// submit, a host step records it instead and calls the check. Where it does not rule them out,
/// the check is called with every command, as for [`IndirectCommandCheckNeeds::ALL`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IndirectCommandCheckNeeds {
    /// Draws of points (`MTLPrimitiveTypePoint`): ruled out where no record may hold one
    /// ([`IndirectCommandBuffer::command_writes`]; a primitive type the writes do not state, or
    /// a value Metal has no type for, counts as a point).
    pub point_draws: bool,
    /// Commands that run a pipeline state that may encode, reset or copy indirect commands
    /// (`can_encode_indirect_commands`): ruled out where no state the commands may run can (the
    /// encoder's state where the buffer inherits it, and those its records may name,
    /// [`IndirectCommandBuffer::may_name_encoding_states`]).
    pub encoding_states: bool,
}

impl IndirectCommandCheckNeeds {
    /// Every kind: draws of points and commands under states that may encode. Where what the
    /// records may hold rules out both, the check still has nothing to see.
    pub const ALL: IndirectCommandCheckNeeds = IndirectCommandCheckNeeds {
        point_draws: true,
        encoding_states: true,
    };

    /// A check that looks for draws of points where `point_draws`, and for commands under states
    /// that may encode indirect commands where `encoding_states`.
    pub const fn new(point_draws: bool, encoding_states: bool) -> IndirectCommandCheckNeeds {
        IndirectCommandCheckNeeds {
            point_draws,
            encoding_states,
        }
    }

    /// What a check of these needs may see among commands whose records may hold `writes`, where
    /// `encodes` answers whether a state they may run may encode indirect commands (asked only
    /// where it decides): draws of points (a primitive type `writes` do not state counts as one)
    /// and commands under such states, each where it looks for them; neither where no record may
    /// hold a draw or a dispatch, as such a record runs nothing.
    pub(super) fn sees(
        self,
        writes: CommandWrites,
        encodes: impl FnOnce() -> bool,
    ) -> (bool, bool) {
        if !writes.may(RUNS) {
            return (false, false);
        }
        let points =
            self.point_draws && writes.may_draw_primitive_type(PrimitiveType::Point.as_raw());
        (points, self.encoding_states && encodes())
    }
}

/// The operations of a record that runs something: a draw of any kind, or a dispatch
/// ([`IndirectCommandCheckNeeds::sees`]).
const RUNS: u32 = CommandWrites::DRAW
    | CommandWrites::DRAW_INDEXED
    | CommandWrites::DRAW_PATCHES
    | CommandWrites::DRAW_MESH
    | CommandWrites::DISPATCH;

/// One command of an indirect command buffer, as [`IndirectCommandBuffer::commands`] reads it: a
/// runtime extension standing for Metal's private getters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum IndirectCommand {
    /// A compute command.
    Compute(EncodedComputeCommand),
    /// A render command.
    Render(EncodedRenderCommand),
}

/// A compute command's state, as its setters left it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncodedComputeCommand {
    /// The dispatch it runs; none for a reset or state-only command, which runs nothing.
    pub dispatch: Option<IndirectDispatch>,
    /// `getCommandType`, Apple's private getter: the type of the dispatch the record holds
    /// (`CONCURRENT_DISPATCH` or `CONCURRENT_DISPATCH_THREADS`), none for a fresh or reset
    /// command, as Metal answers.
    pub command_type: IndirectCommandType,
    /// `setBarrier`.
    pub barrier: bool,
    /// The pipeline state set on it; none when unset, or when no state of the ID it names lives.
    pub pipeline: Option<ComputePipelineState>,
    /// The resource ID of the pipeline state its record names, whether or not a state of that
    /// ID still lives (`pipeline`); none where it names none (unset, or reset, which keeps the
    /// state for the command's runs but names none). Metal's getter answers a released state's
    /// identifier, and 0 for none (measured on an M3 Max by the consumer).
    pub pipeline_id: Option<ResourceId>,
    /// `getPipelineStateUniqueIdentifier`, Apple's private getter: the
    /// [`unique_identifier`](ComputePipelineState::unique_identifier) of the state its record
    /// names, the one the CPU set after its release too, as Metal answers it (measured on an M3
    /// Max by the consumer); 0 where it names none (`pipeline_id`). For a record the GPU wrote,
    /// the state that held its ID as the GPU last could write the record (the completion of the
    /// last command buffer that made the buffer resident for writing, or copied or reset commands
    /// into it; now, while one has not completed), released since or not and after another state
    /// took the ID, as Metal's GPU write resolves the ID as it runs and its record names that
    /// state after (measured on an M3 Max by the consumer). Differs: a record written by an
    /// earlier command buffer, its ID's holder changed before the last one completed, answers
    /// the later holder's. `None` where the runtime cannot tell which state the record names:
    /// the ID's holder then was a state of the other kind, or is past the last four holders of
    /// the ID the device keeps, and no live state of this kind holds it.
    pub pipeline_unique_identifier: Option<u64>,
    /// Each kernel buffer below the descriptor's `max_kernel_buffer_bind_count`, past the 32 a
    /// record holds too, as set, a nil buffer included (its address the offset's low 32 bits): none
    /// where no setter set it or a reset cleared it. Metal's getters of the slots are
    /// [`kernel_buffer_address`](Self::kernel_buffer_address) and
    /// [`kernel_attribute_stride`](Self::kernel_attribute_stride).
    pub kernel_buffers: Vec<Option<IndirectBufferBinding>>,
    /// Each threadgroup length below the descriptor's
    /// `max_kernel_threadgroup_memory_bind_count`, as set; 0 when unset.
    pub threadgroup_memory_lengths: Vec<u32>,
    /// The stage-in region; all zero when unset.
    pub stage_in_region: Region,
    /// The imageblock width; 0 when unset.
    pub imageblock_width: u64,
    /// The imageblock height; 0 when unset.
    pub imageblock_height: u64,
    getters: Getters,
}

impl EncodedComputeCommand {
    /// `getKernelBufferAddressAtIndex:`, Apple's private getter: the address of the buffer set
    /// at `index` plus the offset's low 32 bits, the offset's low 32 bits alone for a nil
    /// buffer, 0 where none is set, as Metal answers, past the 32 slots a record holds too. None at
    /// or past the descriptor's `max_kernel_buffer_bind_count`, where Metal's answer was not
    /// measured.
    pub fn kernel_buffer_address(&self, index: u64) -> Option<u64> {
        address_at(&self.kernel_buffers, self.getters.kernel, index)
    }

    /// `getKernelAttributeStrideAtIndex:`, Apple's private getter: all ones on a buffer without
    /// dynamic attribute strides; with them, the stride the stride form set, all ones after the
    /// plain form, a nil buffer's too, and 0 where none is set, as Metal answers. Past the count,
    /// as [`kernel_buffer_address`](Self::kernel_buffer_address).
    pub fn kernel_attribute_stride(&self, index: u64) -> Option<u64> {
        let of = (self.getters.kernel, self.getters.dynamic_strides);
        stride_at(&self.kernel_buffers, of, index)
    }
}

/// The descriptor's facts a command's buffer getters answer by: its count of each stage's
/// buffers, and whether it supports dynamic attribute strides.
#[derive(Clone, Copy, Debug)]
struct Getters {
    dynamic_strides: bool,
    kernel: u32,
    vertex: u32,
    fragment: u32,
    object: u32,
    mesh: u32,
}

impl Getters {
    fn of(descriptor: &IndirectCommandBufferDescriptor) -> Getters {
        Getters {
            dynamic_strides: descriptor.support_dynamic_attribute_stride,
            kernel: descriptor.max_kernel_buffer_bind_count,
            vertex: descriptor.max_vertex_buffer_bind_count,
            fragment: descriptor.max_fragment_buffer_bind_count,
            object: descriptor.max_object_buffer_bind_count,
            mesh: descriptor.max_mesh_buffer_bind_count,
        }
    }
}

/// A compute command's dispatch.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectDispatch {
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups {
        /// `threadgroupsPerGrid`.
        threadgroups_per_grid: Size,
        /// `threadsPerThreadgroup`.
        threads_per_threadgroup: Size,
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`.
    Threads {
        /// `threadsPerGrid`.
        threads_per_grid: Size,
        /// `threadsPerThreadgroup`.
        threads_per_threadgroup: Size,
    },
}

/// A buffer set on a command.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectBufferBinding {
    /// The buffer's address plus the low 32 bits of the offset set; for a nil buffer, the
    /// offset's low 32 bits alone, as Metal keeps it (below 2^32, which no buffer's address is).
    pub address: GpuAddress,
    /// The attribute stride set, or none for the pipeline's own.
    pub attribute_stride: Option<u64>,
}

impl IndirectBufferBinding {
    /// A buffer at `address`, at `attribute_stride` (none: the pipeline's own).
    pub fn new(address: GpuAddress, attribute_stride: Option<u64>) -> IndirectBufferBinding {
        IndirectBufferBinding {
            address,
            attribute_stride,
        }
    }
}

/// `MTLIndirectRenderCommand`: the setters of one command of an indirect command buffer of
/// render command types. Each writes the command's record at once, as a compute command's do; a
/// command encoded for execution reads the record as it is when the execution runs.
#[derive(Clone, Debug)]
pub struct IndirectRenderCommand {
    icb: IndirectCommandBuffer,
    index: u32,
}

impl IndirectRenderCommand {
    fn descriptor(&self) -> &IndirectCommandBufferDescriptor {
        &self.icb.0.descriptor
    }

    /// Whether it is a command of a buffer of dispatches, which plain Metal gives where its
    /// validation stops `indirectRenderCommandAtIndex:`: the setters it ignores there do nothing
    /// (`draw_primitives`, `set_vertex_buffer`, `set_fragment_buffer`, `set_barrier`,
    /// `clear_barrier`; the buffer's commands read back and run as before), and the others are
    /// refused (`CommandKind`): `set_render_pipeline_state`, `draw_indexed_primitives` and `reset`
    /// end its process, and the rest were not measured.
    fn of_compute(&self) -> bool {
        !self.icb.is_render()
    }

    /// Write `field` into the record, with `set` as the state the CPU set where it names one
    /// (`IndirectCommandBuffer::edit`), for `call`: refused (`CommandKind`) for a command of a
    /// buffer of dispatches (`of_compute`).
    fn edit(
        &self,
        call: &'static str,
        field: RecordField,
        set: Option<CommandState>,
    ) -> Result<(), Error> {
        if self.of_compute() {
            return Err(problem(call, IndirectCommandBufferProblem::CommandKind));
        }
        self.icb.edit(self.index, field, set)
    }

    /// `setRenderPipelineState:`. Refused (`Invalid`): a buffer that inherits its pipeline
    /// state, a state made without `supportIndirectCommandBuffers` (a tile pipeline's among
    /// them), a state of another device, and a state whose kind of draw the buffer's command
    /// types lack (`PipelineKind`: a vertex function's on a buffer of mesh draws alone, a mesh
    /// one's on a buffer of no mesh draw), where Metal's process ends.
    pub fn set_render_pipeline_state(&self, state: &RenderPipelineState) -> Result<(), Error> {
        const CALL: &str = "set_render_pipeline_state";
        if !state.device().same(self.icb.device()) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        if self.descriptor().inherit_pipeline_state {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::InheritedPipelineState,
            ));
        }
        let Some(id) = NonZeroU64::new(state.gpu_resource_id().0) else {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::PipelineWithoutSupport,
            ));
        };
        // The draws of the state's kind: a mesh state's mesh draws, another's the rest.
        let drawn = match state.kind() {
            RenderStateKind::Graphics(graphics) if graphics.descriptor().vertex().is_none() => {
                MESH_TYPES
            }
            RenderStateKind::Graphics(_) | RenderStateKind::Tile(_) => {
                RENDER_TYPES.difference(MESH_TYPES)
            }
        };
        if !self.of_compute() && !self.descriptor().command_types.intersects(drawn) {
            return Err(problem(CALL, IndirectCommandBufferProblem::PipelineKind));
        }
        // A tile pipeline's state has no ID, refused above.
        let (Some(named), Some(program), Some(serial)) =
            (state.downgrade(), state.program(), state.id_serial())
        else {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::PipelineWithoutSupport,
            ));
        };
        let stamp = IdStamp {
            id: id.get(),
            serial,
            identifier: state.identifier_value(),
        };
        let can = Named {
            encodes: state.can_encode_indirect_commands(),
            writes: state.indirect_command_writes(),
        };
        let set = CommandState::Render(named, program, stamp, can);
        self.edit(CALL, RecordField::Pipeline(id), Some(set))
    }

    /// `setVertexBuffer:offset:atIndex:`: vertex buffer `index` at `buffer`'s address plus
    /// `offset`'s low 32 bits, as Metal keeps it, at the pipeline's own attribute stride; `None`
    /// sets a nil buffer at the offset's low 32 bits, as
    /// [`IndirectComputeCommand::set_kernel_buffer`] does, an index at or past the descriptor's
    /// `max_vertex_buffer_bind_count` taken and kept nowhere as that call's is. Refused
    /// (`Invalid`): a buffer of another device. On a buffer that inherits its buffers it is kept
    /// and read back, and executions bind the encoder's.
    pub fn set_vertex_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        if self.of_compute() {
            return Ok(());
        }
        self.set_buffer(
            "set_vertex_buffer",
            BufferStage::Vertex,
            buffer,
            offset,
            None,
            index,
        )
    }

    /// `setVertexBuffer:offset:attributeStride:atIndex:`: as
    /// [`set_vertex_buffer`](Self::set_vertex_buffer), at attribute stride `stride` (all ones is
    /// the pipeline's own). On a buffer that does not support dynamic attribute strides the
    /// stride is dropped and the buffer set, as plain Metal sets it where its validation stops
    /// the call: the stride reads back all ones and the draw reads the buffer at its offset, as a
    /// kernel buffer's is. `None` sets a nil buffer at the stride, as
    /// [`IndirectComputeCommand::set_kernel_buffer_with_stride`] does.
    pub fn set_vertex_buffer_with_stride(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_vertex_buffer_with_stride";
        let dynamic = self.descriptor().support_dynamic_attribute_stride;
        let stride = (dynamic && stride != u64::MAX).then_some(stride);
        self.set_buffer(CALL, BufferStage::Vertex, buffer, offset, stride, index)
    }

    /// `setFragmentBuffer:offset:atIndex:`: as [`set_vertex_buffer`](Self::set_vertex_buffer),
    /// against the descriptor's `max_fragment_buffer_bind_count`.
    pub fn set_fragment_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        if self.of_compute() {
            return Ok(());
        }
        self.set_buffer(
            "set_fragment_buffer",
            BufferStage::Fragment,
            buffer,
            offset,
            None,
            index,
        )
    }

    fn set_buffer(
        &self,
        call: &'static str,
        stage: BufferStage,
        buffer: Option<&Buffer>,
        offset: u64,
        attribute_stride: Option<u64>,
        index: u32,
    ) -> Result<(), Error> {
        if self.of_compute() {
            return Err(problem(call, IndirectCommandBufferProblem::CommandKind));
        }
        let descriptor = self.descriptor();
        let limit = match stage {
            BufferStage::Fragment => descriptor.max_fragment_buffer_bind_count,
            BufferStage::Object => descriptor.max_object_buffer_bind_count,
            BufferStage::Mesh => descriptor.max_mesh_buffer_bind_count,
            BufferStage::Kernel | BufferStage::Vertex => descriptor.max_vertex_buffer_bind_count,
        };
        let set = BufferSet {
            stage,
            limit,
            slot: index,
            buffer,
            offset,
            attribute_stride,
        };
        self.icb.set_buffer(self.index, call, set)
    }

    /// Refuse a draw `action` of `kind` for `call` where the buffer's records cannot hold it
    /// (`IndirectCommandBufferProblem::CommandType`, `CommandDescriptor::holds`). Metal's
    /// validation stops a draw of any kind the command types lack ("drawPrimitives is not
    /// enabled on this indirect command buffer"), but plain Metal runs one whose record a kind
    /// they allow holds as on a buffer of its own kind, and faults the GPU or ends the process
    /// for the others (the M3 Max and the M4 alike), as it runs a kernel's.
    fn check_draw(
        &self,
        call: &'static str,
        kind: IndirectCommandType,
        action: &Action,
    ) -> Result<(), Error> {
        if self.of_compute() {
            return Err(problem(call, IndirectCommandBufferProblem::CommandKind));
        }
        if !self.icb.0.records.holds(action) {
            return Err(problem(
                call,
                IndirectCommandBufferProblem::CommandType(kind),
            ));
        }
        Ok(())
    }

    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`: the command draws
    /// `vertex_count` vertices of `primitive_type` from `vertex_start`, `instance_count` times
    /// from `base_instance`, each value's low 32 bits, which the record holds (Metal takes larger
    /// values; what it keeps of them is not measured: its getter answers an object). A type is a
    /// public one or one of Metal's private ones ([`DrawPrimitiveType`]), which an execution draws
    /// as an encoder's draw of it draws, type 9 failing the command buffer so where an execution
    /// of a range given at its call finds it there (the M3 Max, macOS 27.0.1: types 5 to 9 drew
    /// and failed as the encoder's draws of them, direct and indexed). A draw of no vertices or
    /// no instances runs nothing, as plain Metal runs it where its validation stops the call
    /// ("instanceCount(0) must be non-zero"). The records of every render buffer but one
    /// of mesh draws alone hold it: on a buffer whose command types lack `DRAW` it runs as on its
    /// own, as plain Metal runs it, where Metal's validation stops the call ("drawPrimitives is not
    /// enabled on this indirect command buffer"). Refused (`Invalid`): a buffer of mesh draws alone
    /// (`CommandType`), where Metal's process ends, with and without its validation (an M4: buffers
    /// of either mesh kind and of both, the encoder's pipeline inherited).
    pub fn draw_primitives(
        &self,
        primitive_type: impl Into<DrawPrimitiveType>,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_primitives";
        if self.of_compute() {
            return Ok(());
        }
        let primitive_type = primitive_type.into();
        let action = Action::DrawPrimitives {
            primitive_type: primitive_type.raw() as u32,
            vertex_start: vertex_start as u32,
            vertex_count: vertex_count as u32,
            instance_count: instance_count as u32,
            base_instance: base_instance as u32,
        };
        self.check_draw(CALL, IndirectCommandType::DRAW, &action)?;
        self.edit(CALL, RecordField::Action(action), None)?;
        self.drew(primitive_type);
        Ok(())
    }

    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`: the command draws `index_count` indices of `index_type` from
    /// `index_buffer_offset` of `index_buffer`, `instance_count` times from `base_instance`, each
    /// index plus `base_vertex`, of a public primitive type or a private one, as
    /// [`draw_primitives`](Self::draw_primitives) says (type 9 faults the command buffer, of any
    /// index or instance count, as the encoder's draw does). The record holds the buffer as one
    /// address (its `gpu_address` plus the offset's low 32 bits, as Metal keeps it; its validation
    /// checks the whole offset, as the refusal below does), which an execution resolves against
    /// the encoder's resident resources (`render/executions.rs`), and each value's low 32 bits,
    /// `base_vertex` as a signed one: Metal keeps a base vertex's so (a count past 32 bits always
    /// runs past the buffer, which is refused). It runs as on its own on a buffer of patch draws,
    /// whose records hold it, as plain Metal runs it, where Metal's validation stops a buffer
    /// whose command types lack `DRAW_INDEXED`. A draw of no indices or no instances runs nothing
    /// (but type 9's), and the indices are read from the offset rounded down to a multiple of the
    /// index size, as plain Metal runs them where its validation stops the call
    /// (`idx_count_zero`, `idx_instances_zero`; 32-bit indices at offset 2 and 16-bit ones at 1
    /// drew those at 0). Refused (`Invalid`): a buffer of draws of primitives alone, whose records
    /// cannot hold it (`CommandType`; plain Metal's GPU faults), or of mesh draws alone
    /// (`CommandType`; Metal's process ends), and, as Metal's validation stops it, indices past
    /// the buffer's length from that offset (`IndexBufferRange`, of the whole count; plain Metal
    /// draws from whatever memory follows), and an index buffer of another device.
    pub fn draw_indexed_primitives(
        &self,
        primitive_type: impl Into<DrawPrimitiveType>,
        index_count: u64,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_indexed_primitives";
        let primitive_type = primitive_type.into();
        let action = Action::DrawIndexedPrimitives {
            primitive_type: primitive_type.raw() as u32,
            index_count: index_count as u32,
            index_type: index_type.as_raw(),
            index_buffer: index_buffer.gpu_address().0 + u64::from(index_buffer_offset as u32),
            instance_count: instance_count as u32,
            base_vertex: base_vertex as i32,
            base_instance: base_instance as u32,
        };
        self.check_draw(CALL, IndirectCommandType::DRAW_INDEXED, &action)?;
        if !index_buffer.device().same(self.icb.device()) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let size = index_size(index_type);
        let offset = index_buffer_offset - index_buffer_offset % size;
        let past = index_count
            .checked_mul(size)
            .and_then(|bytes| bytes.checked_add(offset))
            .is_none_or(|end| end > index_buffer.length());
        if past {
            return Err(invalid(
                CALL,
                Problem::Draw(DrawProblem::IndexBufferRange {
                    offset,
                    count: index_count,
                    length: index_buffer.length(),
                }),
            ));
        }
        self.edit(CALL, RecordField::Action(action), None)?;
        self.drew(primitive_type);
        Ok(())
    }

    /// The command now draws `primitive_type`: a draw of type 9 marks its buffer for the
    /// executions that look for one ([`IndirectCommandBuffer::nine_among`]).
    fn drew(&self, primitive_type: DrawPrimitiveType) {
        if primitive_type == DrawPrimitiveType::Nine {
            self.icb.0.nines.store(true, Ordering::Release);
        }
    }

    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: the command draws `patch_count` patches from
    /// `patch_start` (each patch's index from `patch_index_buffer` where it has one),
    /// `instance_count` times from `base_instance`, their factors read from
    /// `tessellation_factor_buffer` at the offset and the instance stride. The record holds each
    /// buffer as one address, at its offset's low 32 bits as Metal keeps it (the patch index and
    /// factor buffers), which an execution resolves against the encoder's resident resources, and
    /// each value's low 32 bits (Metal keeps a patch count's so). The control points are kept,
    /// though Metal's execution draws with its pipeline's (measured: 3 and 0 drew as a quad
    /// pipeline's 4), as the execution does, more than 32 included (a Metal probe `icb_cp33` drew
    /// as 3 do, where Metal's validation stops the call); a draw of no patches or no instances runs
    /// nothing, and the patch indices are read from the offset rounded down to a multiple of 4, as
    /// plain Metal runs them where its validation stops the call (`patch_instances_zero`; indices 9
    /// and 3 at offset 2 drew patches 9 and 3). A factor offset and instance stride that are not
    /// multiples of 4, which Metal's validation stops for the encoder's, are read as plain Metal
    /// reads them, as the encoder's are: the factors from the offset rounded down to a multiple of
    /// 2, each instance's at the stride rounded down to a multiple of 4 (Metal probes
    /// `icb_factor_offset2`, 4 and 5: strides 2 and 14 drew what 0 and 12 draw, 6 and 10 what 4 and
    /// 8 draw). It runs as on its own on a buffer of indexed patch draws, whose records hold it, as
    /// plain Metal runs it, where Metal's validation stops a buffer whose command types lack
    /// `DRAW_PATCHES`. Refused (`Invalid`): a buffer of neither patch kind, whose records cannot
    /// hold it (`CommandType`; plain Metal ends its process), and, as Metal's validation stops
    /// each, buffers of another device. Refused as the encoder's draw refuses them, which Metal's
    /// validation of a command was not measured to: patch indices past the patch index buffer
    /// from its offset's low 32 bits, which the command keeps (`PatchIndexBufferRange`; plain
    /// Metal read 0 past it, the buffer's page padding, which stays undefined). An odd factor
    /// offset is read from the byte below, and an odd instance stride
    /// rounded down to a multiple of 4, as plain Metal reads a command's and the encoder's alike,
    /// with and without its validation.
    pub fn draw_patches(
        &self,
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor_buffer: &Buffer,
        tessellation_factor_buffer_offset: u64,
        tessellation_factor_buffer_instance_stride: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_patches";
        let action = Action::DrawPatches {
            control_points: number_of_patch_control_points as u32,
            patch_start: patch_start as u32,
            patch_count: patch_count as u32,
            patch_index_buffer: patch_index_buffer.map_or(0, |buffer| {
                buffer.gpu_address().0 + u64::from(patch_index_buffer_offset as u32)
            }),
            instance_count: instance_count as u32,
            base_instance: base_instance as u32,
            factor_buffer: tessellation_factor_buffer.gpu_address().0
                + u64::from(tessellation_factor_buffer_offset as u32),
            factor_instance_stride: tessellation_factor_buffer_instance_stride as u32,
            factor_scale: 1.0f32.to_bits(),
        };
        self.check_draw(CALL, IndirectCommandType::DRAW_PATCHES, &action)?;
        self.check_patches(
            CALL,
            patch_start,
            patch_count,
            (patch_index_buffer, patch_index_buffer_offset),
            None,
            tessellation_factor_buffer,
        )?;
        self.edit(CALL, RecordField::Action(action), None)
    }

    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:
    /// tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: the command draws as
    /// [`draw_patches`](Self::draw_patches) does, but control point `k` of the patch whose index is
    /// `p` is the index (of its pipeline's `tessellationControlPointIndexType`) at position
    /// `p · N + k` of `control_point_index_buffer` from its offset, `N` its pipeline's control
    /// points, whatever the command names, as the encoder's `draw_indexed_patches` reads them
    /// (a probe, an M4: the command's 0, 1 and 4 drew as the pipeline's 3). The record holds each
    /// buffer as one address, at its offset's low 32 bits (the control-point index buffer's too),
    /// and each value's low 32 bits (probe 9b: Metal keeps a patch start's so; its getters answer
    /// each count's). `None` for the control-point index buffer, which Metal takes, makes the
    /// command a plain patch draw of the same values, as Metal records it: its `getCommandType`
    /// answers `DrawPatches` on an M4 (probe 9b, a buffer of indexed patch draws alone) and on the
    /// consumer's M3 Max, and it draws as a plain patch draw on that pipeline draws, every
    /// control-point index 0 on a pipeline with an index type (probe 9b: the same marks as its
    /// plain patch draw on the same pipeline). It runs as on its own on a buffer of patch draws,
    /// whose records hold it, as plain Metal runs it, where Metal's validation stops a buffer whose
    /// command types lack `DRAW_INDEXED_PATCHES` ("drawIndexedPatches is not enabled on this
    /// indirect command buffer"). Refused (`Invalid`) as `draw_patches` is, for a buffer of neither
    /// patch kind (`CommandType`), and for a control-point index buffer of another device. Metal's
    /// validation does not stop a factor offset or instance stride that is not a multiple of 4 here
    /// (probe 9), which the runtime reads as `draw_patches` does; nor does it check the
    /// control-point index buffer, whose index size the pipeline set at execution states (an
    /// execution runs what Metal's does, `render/executions.rs`).
    pub fn draw_indexed_patches(
        &self,
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: Option<&Buffer>,
        control_point_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor_buffer: &Buffer,
        tessellation_factor_buffer_offset: u64,
        tessellation_factor_buffer_instance_stride: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_indexed_patches";
        // Each buffer at its offset's low 32 bits, as Metal keeps it.
        let patch_index = patch_index_buffer.map_or(0, |buffer| {
            buffer.gpu_address().0 + u64::from(patch_index_buffer_offset as u32)
        });
        let factors = tessellation_factor_buffer.gpu_address().0
            + u64::from(tessellation_factor_buffer_offset as u32);
        // Without a control-point index buffer Metal records a plain patch draw.
        let action = match control_point_index_buffer {
            Some(buffer) => Action::DrawIndexedPatches {
                control_points: number_of_patch_control_points as u32,
                patch_start: patch_start as u32,
                patch_count: patch_count as u32,
                patch_index_buffer: patch_index,
                control_point_index_buffer: buffer.gpu_address().0
                    + u64::from(control_point_index_buffer_offset as u32),
                instance_count: instance_count as u32,
                base_instance: base_instance as u32,
                factor_buffer: factors,
                factor_instance_stride: tessellation_factor_buffer_instance_stride as u32,
                factor_scale: 1.0f32.to_bits(),
            },
            None => Action::DrawPatches {
                control_points: number_of_patch_control_points as u32,
                patch_start: patch_start as u32,
                patch_count: patch_count as u32,
                patch_index_buffer: patch_index,
                instance_count: instance_count as u32,
                base_instance: base_instance as u32,
                factor_buffer: factors,
                factor_instance_stride: tessellation_factor_buffer_instance_stride as u32,
                factor_scale: 1.0f32.to_bits(),
            },
        };
        self.check_draw(CALL, IndirectCommandType::DRAW_INDEXED_PATCHES, &action)?;
        self.check_patches(
            CALL,
            patch_start,
            patch_count,
            (patch_index_buffer, patch_index_buffer_offset),
            control_point_index_buffer,
            tessellation_factor_buffer,
        )?;
        // Its control-point index buffer takes the words of the object threadgroup lengths 0 and
        // 1 (`set_object_threadgroup_memory_length`).
        if matches!(action, Action::DrawIndexedPatches { .. })
            && self.descriptor().command_types.intersects(MESH_TYPES)
        {
            for index in 0..2 {
                let field = RecordField::ThreadgroupMemoryLength { index, length: 0 };
                self.edit(CALL, field, None)?;
            }
        }
        self.edit(CALL, RecordField::Action(action), None)
    }

    /// Refuse a patch draw's values for `call` as [`draw_patches`](Self::draw_patches) states:
    /// its buffers (the patch index buffer and its offset, a control-point index buffer, the
    /// factor buffer) of another device, and patch indices past the patch index buffer from its
    /// offset's low 32 bits, which the record keeps, rounded down to a multiple of 4.
    fn check_patches(
        &self,
        call: &'static str,
        patch_start: u64,
        patch_count: u64,
        (patch_index_buffer, patch_index_buffer_offset): (Option<&Buffer>, u64),
        control_point_index_buffer: Option<&Buffer>,
        factor_buffer: &Buffer,
    ) -> Result<(), Error> {
        let device = self.icb.device();
        let foreign = patch_index_buffer
            .into_iter()
            .chain(control_point_index_buffer)
            .chain([factor_buffer])
            .any(|buffer| !buffer.device().same(device));
        if foreign {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let draw = |problem| Err(invalid(call, Problem::Draw(problem)));
        if let Some(buffer) = patch_index_buffer {
            // The record keeps the offset's low 32 bits, as Metal's (its getter answers them).
            let kept = u64::from(patch_index_buffer_offset as u32);
            let offset = kept - kept % 4;
            let indices = u64::from(patch_start as u32) + u64::from(patch_count as u32);
            let end = offset.checked_add(4 * indices);
            if end.is_none_or(|end| end > buffer.length()) {
                return draw(DrawProblem::PatchIndexBufferRange {
                    offset,
                    indices,
                    length: buffer.length(),
                });
            }
        }
        Ok(())
    }

    /// `setObjectBuffer:offset:atIndex:`: object buffer `index` at `buffer`'s address plus
    /// `offset`'s low 32 bits, as Metal keeps it; `None` sets a nil buffer at the
    /// offset's low 32 bits, as [`IndirectComputeCommand::set_kernel_buffer`] does, an index at
    /// or past the descriptor's `max_object_buffer_bind_count` taken and kept nowhere as that
    /// call's is. Refused (`Invalid`): a buffer of another device, a command of a buffer of
    /// dispatches (`CommandKind`).
    pub fn set_object_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer(
            "set_object_buffer",
            BufferStage::Object,
            buffer,
            offset,
            None,
            index,
        )
    }

    /// `setMeshBuffer:offset:atIndex:`: as [`set_object_buffer`](Self::set_object_buffer),
    /// against the descriptor's `max_mesh_buffer_bind_count`.
    pub fn set_mesh_buffer(
        &self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer(
            "set_mesh_buffer",
            BufferStage::Mesh,
            buffer,
            offset,
            None,
            index,
        )
    }

    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: the
    /// command draws `threadgroups_per_grid` threadgroups (of the object function where its
    /// pipeline has one, else of the mesh function) of the threads given, each axis's low 32
    /// bits, as the encoder's `draw_mesh_threadgroups` draws them (`render/mesh.rs`). Refused
    /// (`Invalid`) for a buffer whose records do not hold it (`CommandType`) and a command of a
    /// buffer of dispatches (`CommandKind`).
    pub fn draw_mesh_threadgroups(
        &self,
        threadgroups_per_grid: Size,
        threads_per_object_threadgroup: Size,
        threads_per_mesh_threadgroup: Size,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_mesh_threadgroups";
        let action = Action::DrawMeshThreadgroups {
            threadgroups: low_bits(threadgroups_per_grid),
            object_threads: low_bits(threads_per_object_threadgroup),
            mesh_threads: low_bits(threads_per_mesh_threadgroup),
        };
        self.check_draw(CALL, IndirectCommandType::DRAW_MESH_THREADGROUPS, &action)?;
        self.edit(CALL, RecordField::Action(action), None)
    }

    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: the command
    /// draws `threads_per_grid` threads, as the encoder's `draw_mesh_threads` draws them, each
    /// axis's low 32 bits. Refused as
    /// [`draw_mesh_threadgroups`](Self::draw_mesh_threadgroups) is.
    pub fn draw_mesh_threads(
        &self,
        threads_per_grid: Size,
        threads_per_object_threadgroup: Size,
        threads_per_mesh_threadgroup: Size,
    ) -> Result<(), Error> {
        const CALL: &str = "draw_mesh_threads";
        let action = Action::DrawMeshThreads {
            threads: low_bits(threads_per_grid),
            object_threads: low_bits(threads_per_object_threadgroup),
            mesh_threads: low_bits(threads_per_mesh_threadgroup),
        };
        self.check_draw(CALL, IndirectCommandType::DRAW_MESH_THREADS, &action)?;
        self.edit(CALL, RecordField::Action(action), None)
    }

    /// `setBarrier`: the command's draw runs after the earlier draws of its execution and of the
    /// encoder, their writes visible to it, as Metal orders it on a buffer of mesh draws (a Metal
    /// probe, the M3 Max and an M4: a reader's object function reads a writer's mesh function's
    /// words) and plain Metal on a buffer without mesh draws, where Metal's validation stops the
    /// call ("setBarrier is not enabled on this indirect command buffer (requires
    /// DrawMeshThreadgroups or DrawMeshThreads)"; an M4). An execution draws such a command in a
    /// render pass instance of its own (`render/executions.rs`), a memoryless attachment kept
    /// across it, whether the barrier was set before or after the encoder that executes the
    /// command ended. A command of a buffer of dispatches does nothing, as plain Metal's.
    pub fn set_barrier(&self) -> Result<(), Error> {
        if self.of_compute() {
            return Ok(());
        }
        self.edit("set_barrier", RecordField::Barrier(true), None)?;
        self.icb.0.barriers.store(true, Ordering::Release);
        Ok(())
    }

    /// `clearBarrier`: the command keeps no barrier, as plain Metal's reads back where its
    /// validation stops the call on a buffer without mesh draws. A command of a buffer of
    /// dispatches does nothing, as plain Metal's.
    pub fn clear_barrier(&self) -> Result<(), Error> {
        if self.of_compute() {
            return Ok(());
        }
        self.edit("clear_barrier", RecordField::Barrier(false), None)
    }

    /// `reset`: the command as a fresh buffer's.
    pub fn reset(&self) -> Result<(), Error> {
        if self.of_compute() {
            return Err(problem("reset", IndirectCommandBufferProblem::CommandKind));
        }
        self.icb.reset_command(self.index, |record, pipeline| {
            *record = CommandRecord::default();
            *pipeline = None;
        })
    }

    /// The refusal of a render state set on the command by `call` (the module docs): `Invalid`
    /// (`RenderState`) on a device that does not claim Apple10, where Metal's process ends;
    /// `NotImplemented(IndirectRenderCommands)` on one that does, whose commands would draw with
    /// it.
    fn render_state(&self, call: &'static str) -> Error {
        if self.icb.device().supports_family(GpuFamily::Apple10.raw()) {
            return Error::NotImplemented(RuntimeGap::IndirectRenderCommands);
        }
        problem(call, IndirectCommandBufferProblem::RenderState)
    }

    /// `setDepthStencilState:`: refused on a device that does not claim Apple10 (`Invalid`,
    /// `RenderState`), where Metal's process ends, and `NotImplemented(IndirectRenderCommands)` on
    /// one that does, whose commands would draw with it (the module docs). Refused (`Invalid`)
    /// first: a state of another device.
    pub fn set_depth_stencil_state(&self, state: Option<&DepthStencilState>) -> Result<(), Error> {
        const CALL: &str = "set_depth_stencil_state";
        if state.is_some_and(|state| !state.device().same(self.icb.device())) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        Err(self.render_state(CALL))
    }

    /// `setDepthBias:slopeScale:clamp:`: refused, as `set_depth_stencil_state` is.
    pub fn set_depth_bias(
        &self,
        _depth_bias: f32,
        _slope_scale: f32,
        _clamp: f32,
    ) -> Result<(), Error> {
        Err(self.render_state("set_depth_bias"))
    }

    /// `setDepthClipMode:`: refused, as `set_depth_stencil_state` is.
    pub fn set_depth_clip_mode(&self, _mode: DepthClipMode) -> Result<(), Error> {
        Err(self.render_state("set_depth_clip_mode"))
    }

    /// `setCullMode:`: refused, as `set_depth_stencil_state` is.
    pub fn set_cull_mode(&self, _mode: CullMode) -> Result<(), Error> {
        Err(self.render_state("set_cull_mode"))
    }

    /// `setFrontFacingWinding:`: refused, as `set_depth_stencil_state` is.
    pub fn set_front_facing_winding(&self, _winding: Winding) -> Result<(), Error> {
        Err(self.render_state("set_front_facing_winding"))
    }

    /// `setTriangleFillMode:`: refused, as `set_depth_stencil_state` is.
    pub fn set_triangle_fill_mode(&self, _mode: TriangleFillMode) -> Result<(), Error> {
        Err(self.render_state("set_triangle_fill_mode"))
    }

    /// `setObjectThreadgroupMemoryLength:atIndex:`: the command's mesh draw runs its object
    /// function with the lengths the command set, 0 where it set none, never the encoder's (the
    /// M3 Max: an argument past a length of 16 reached the next argument, whatever the encoder
    /// set). Taken at any index, as plain Metal takes one at or past the descriptor's
    /// `max_object_threadgroup_memory_bind_count`, where its validation stops it (an M4); the
    /// record holds indices below 31, which is every one a function binds, and one past them
    /// changes nothing. Differs: the record's first two lengths are an indexed patch draw's
    /// control-point index buffer, so such a draw on the command clears them, and a length set
    /// there while the command holds one is dropped. Refused (`Invalid`): a command of a buffer
    /// whose command types hold no mesh draw (`ObjectThreadgroupMemory`), where Metal's process
    /// ends, and of a buffer of dispatches (`CommandKind`).
    pub fn set_object_threadgroup_memory_length(
        &self,
        length: u32,
        index: u32,
    ) -> Result<(), Error> {
        const CALL: &str = "set_object_threadgroup_memory_length";
        if self.of_compute() {
            return Err(problem(CALL, IndirectCommandBufferProblem::CommandKind));
        }
        if !self.descriptor().command_types.intersects(MESH_TYPES) {
            return Err(problem(
                CALL,
                IndirectCommandBufferProblem::ObjectThreadgroupMemory,
            ));
        }
        if index >= THREADGROUP_BUFFER_ARGUMENT_COUNT {
            return Ok(());
        }
        let patch =
            |view: &RecordView<'_>| matches!(view.action(), Action::DrawIndexedPatches { .. });
        if index < 2 && self.icb.read_record(self.index, patch)? {
            return Ok(());
        }
        let field = RecordField::ThreadgroupMemoryLength {
            index: index as usize,
            length,
        };
        self.edit(CALL, field, None)
    }

    /// Refuse, for `call`, a getter of a command of a buffer of dispatches (`CommandKind`,
    /// `of_compute`), whose answers were not measured.
    fn check_getter(&self, call: &'static str) -> Result<(), Error> {
        match self.of_compute() {
            true => Err(problem(call, IndirectCommandBufferProblem::CommandKind)),
            false => Ok(()),
        }
    }

    /// `getCommandType`, Apple's private getter: the command's
    /// [`command_type`](EncodedRenderCommand::command_type) as
    /// [`IndirectCommandBuffer::commands`] reads it now, its record's action alone read, as a
    /// compute command's getters read theirs ([`IndirectComputeCommand::command_type`]). Each
    /// getter is refused (`Invalid`, `CommandKind`) for a command of a buffer of dispatches,
    /// where Metal's answer was not measured.
    pub fn command_type(&self) -> Result<IndirectCommandType, Error> {
        self.check_getter("command_type")?;
        self.icb
            .read_record(self.index, |view| command_type(&view.action()))
    }

    /// `hasBarrier`, Apple's private getter: the command's
    /// [`barrier`](EncodedRenderCommand::barrier), read alone.
    pub fn has_barrier(&self) -> Result<bool, Error> {
        self.check_getter("has_barrier")?;
        self.icb.read_record(self.index, |view| view.barrier())
    }

    /// `drawArguments`, `drawIndexedArguments`, `drawPatchesArguments`,
    /// `drawIndexedPatchesArguments` and `getTessellationFactorArguments`, Apple's private
    /// getters: the command's [`draw`](EncodedRenderCommand::draw), read alone, as Metal answers
    /// it: a draw of primitive type 7, 8 or 9 reads as one of points, Metal's 0 (the M3 Max,
    /// macOS 27.0.1), though it draws as set.
    pub fn draw(&self) -> Result<Option<IndirectDraw>, Error> {
        self.check_getter("draw")?;
        self.icb
            .read_record(self.index, |view| answered_draw(&view.action()))
    }

    /// The vertex buffer set at `index`: the command's
    /// [`vertex_buffers`](EncodedRenderCommand::vertex_buffers)`[index]`, read alone, none at or
    /// past the descriptor's `max_vertex_buffer_bind_count`, which Metal's private getters of the
    /// slot answer as [`EncodedRenderCommand::vertex_buffer_address`] and
    /// [`EncodedRenderCommand::vertex_attribute_stride`] do.
    pub fn vertex_buffer(&self, index: u64) -> Result<Option<IndirectBufferBinding>, Error> {
        self.check_getter("vertex_buffer")?;
        let count = self.descriptor().max_vertex_buffer_bind_count;
        self.icb
            .read_buffer(self.index, BufferStage::Vertex, count, index)
    }

    /// The fragment buffer set at `index`, as [`vertex_buffer`](Self::vertex_buffer) reads a
    /// vertex buffer, below the descriptor's `max_fragment_buffer_bind_count`
    /// ([`EncodedRenderCommand::fragment_buffers`]).
    pub fn fragment_buffer(&self, index: u64) -> Result<Option<IndirectBufferBinding>, Error> {
        self.check_getter("fragment_buffer")?;
        let count = self.descriptor().max_fragment_buffer_bind_count;
        self.icb
            .read_buffer(self.index, BufferStage::Fragment, count, index)
    }

    /// The object buffer set at `index`, as [`vertex_buffer`](Self::vertex_buffer) reads a
    /// vertex buffer, below the descriptor's `max_object_buffer_bind_count`
    /// ([`EncodedRenderCommand::object_buffers`]).
    pub fn object_buffer(&self, index: u64) -> Result<Option<IndirectBufferBinding>, Error> {
        self.check_getter("object_buffer")?;
        let count = self.descriptor().max_object_buffer_bind_count;
        self.icb
            .read_buffer(self.index, BufferStage::Object, count, index)
    }

    /// The mesh buffer set at `index`, as [`vertex_buffer`](Self::vertex_buffer) reads a vertex
    /// buffer, below the descriptor's `max_mesh_buffer_bind_count`
    /// ([`EncodedRenderCommand::mesh_buffers`]).
    pub fn mesh_buffer(&self, index: u64) -> Result<Option<IndirectBufferBinding>, Error> {
        self.check_getter("mesh_buffer")?;
        let count = self.descriptor().max_mesh_buffer_bind_count;
        self.icb
            .read_buffer(self.index, BufferStage::Mesh, count, index)
    }

    /// The command's [`pipeline`](EncodedRenderCommand::pipeline), read alone.
    pub fn pipeline_state(&self) -> Result<Option<RenderPipelineState>, Error> {
        self.check_getter("pipeline_state")?;
        Ok(self.icb.render_pipeline_now(self.index)?.0)
    }

    /// The command's [`pipeline_id`](EncodedRenderCommand::pipeline_id), read alone.
    pub fn pipeline_id(&self) -> Result<Option<ResourceId>, Error> {
        self.check_getter("pipeline_id")?;
        self.icb.read_record(self.index, |view| {
            let runs = view.pipeline().or(view.retained_pipeline());
            runs.map(|id| ResourceId(id.get()))
        })
    }

    /// `getPipelineStateUniqueIdentifier`, Apple's private getter: the command's
    /// [`pipeline_unique_identifier`](EncodedRenderCommand::pipeline_unique_identifier), read
    /// alone, whose kind the device announces as `commands` announces it.
    pub fn pipeline_unique_identifier(&self) -> Result<Option<u64>, Error> {
        self.check_getter("pipeline_unique_identifier")?;
        let (_, identifier) = self.icb.render_pipeline_now(self.index)?;
        self.icb.announce_identifier(identifier);
        Ok(identifier)
    }
}

/// A render command's state, as its setters left it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncodedRenderCommand {
    /// The draw it runs; none for a reset or state-only command, which runs nothing. Its
    /// primitive type is the one set, which Metal's private getters answer as 0 for types 7 to 9
    /// ([`IndirectRenderCommand::draw`]).
    pub draw: Option<IndirectDraw>,
    /// `getCommandType`, Apple's private getter: the type of the draw the record holds (`DRAW`,
    /// `DRAW_INDEXED`, `DRAW_PATCHES`, `DRAW_INDEXED_PATCHES`, `DRAW_MESH_THREADGROUPS` or
    /// `DRAW_MESH_THREADS`; an indexed patch draw without a control-point index buffer is a
    /// patch draw), none for a fresh or reset command and for an indexed draw of primitive type
    /// 9, as Metal answers (the M3 Max, macOS 27.0.1).
    pub command_type: IndirectCommandType,
    /// `setBarrier`.
    pub barrier: bool,
    /// The pipeline state set on it; none when unset, or when no state of the ID it names lives.
    pub pipeline: Option<RenderPipelineState>,
    /// The resource ID of the pipeline state its record names, whether or not a state of that
    /// ID still lives (`pipeline`), a reset's retained one included; none where it names none.
    /// Metal's getter answers a released state's identifier, and 0 for none (measured on an M3 Max
    /// by the consumer).
    pub pipeline_id: Option<ResourceId>,
    /// `getPipelineStateUniqueIdentifier`, Apple's private getter, as a compute command's
    /// ([`EncodedComputeCommand::pipeline_unique_identifier`]), of the state `pipeline_id` names.
    pub pipeline_unique_identifier: Option<u64>,
    /// Each vertex buffer below the descriptor's `max_vertex_buffer_bind_count`, as a compute
    /// command's [`kernel_buffers`](EncodedComputeCommand::kernel_buffers) are.
    pub vertex_buffers: Vec<Option<IndirectBufferBinding>>,
    /// Each fragment buffer below the descriptor's `max_fragment_buffer_bind_count`, as
    /// `vertex_buffers` are.
    pub fragment_buffers: Vec<Option<IndirectBufferBinding>>,
    /// Each object buffer below the descriptor's `max_object_buffer_bind_count`, as
    /// `vertex_buffers` are.
    pub object_buffers: Vec<Option<IndirectBufferBinding>>,
    /// Each mesh buffer below the descriptor's `max_mesh_buffer_bind_count`, as
    /// `vertex_buffers` are.
    pub mesh_buffers: Vec<Option<IndirectBufferBinding>>,
    getters: Getters,
}

/// Apple's private getters of a render command's buffer slots, each as a compute command's
/// ([`EncodedComputeCommand::kernel_buffer_address`],
/// [`EncodedComputeCommand::kernel_attribute_stride`]) of its stage's slots and count.
impl EncodedRenderCommand {
    /// `getVertexBufferAddressAtIndex:`.
    pub fn vertex_buffer_address(&self, index: u64) -> Option<u64> {
        address_at(&self.vertex_buffers, self.getters.vertex, index)
    }

    /// `getVertexAttributeStrideAtIndex:`.
    pub fn vertex_attribute_stride(&self, index: u64) -> Option<u64> {
        let of = (self.getters.vertex, self.getters.dynamic_strides);
        stride_at(&self.vertex_buffers, of, index)
    }

    /// `getFragmentBufferAddressAtIndex:`.
    pub fn fragment_buffer_address(&self, index: u64) -> Option<u64> {
        address_at(&self.fragment_buffers, self.getters.fragment, index)
    }

    /// `getObjectBufferAddressAtIndex:`.
    pub fn object_buffer_address(&self, index: u64) -> Option<u64> {
        address_at(&self.object_buffers, self.getters.object, index)
    }

    /// `getMeshBufferAddressAtIndex:`.
    pub fn mesh_buffer_address(&self, index: u64) -> Option<u64> {
        address_at(&self.mesh_buffers, self.getters.mesh, index)
    }
}

/// How a draw of primitive type 9 among an execution's commands fails its command buffer
/// ([`IndirectCommandBuffer::nine_among`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NineDraw {
    /// A direct draw of 3 vertices or more: Metal's GPU hangs at it.
    Hangs,
    /// An indexed draw: Metal's GPU faults at it.
    Faults,
}

/// A render command's draw.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectDraw {
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Primitives {
        /// `primitiveType`: a public type or one of Metal's private ones.
        primitive_type: DrawPrimitiveType,
        /// `vertexStart`.
        vertex_start: u64,
        /// `vertexCount`.
        vertex_count: u64,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseInstance`.
        base_instance: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`: the index buffer as the address of its first index.
    IndexedPrimitives {
        /// `primitiveType`: a public type or one of Metal's private ones.
        primitive_type: DrawPrimitiveType,
        /// `indexCount`.
        index_count: u64,
        /// `indexType`.
        index_type: IndexType,
        /// `indexBuffer` plus `indexBufferOffset`: the address of the first index.
        index_buffer: GpuAddress,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseVertex`.
        base_vertex: i64,
        /// `baseInstance`.
        base_instance: u64,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: each buffer as the address its reads start at.
    Patches {
        /// `numberOfPatchControlPoints`.
        control_points: u64,
        /// `patchStart`.
        patch_start: u64,
        /// `patchCount`.
        patch_count: u64,
        /// `patchIndexBuffer` plus `patchIndexBufferOffset`; `None` for none.
        patch_index_buffer: Option<GpuAddress>,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseInstance`.
        base_instance: u64,
        /// `tessellationFactorBuffer` plus `tessellationFactorBufferOffset`.
        factor_buffer: GpuAddress,
        /// `tessellationFactorBufferInstanceStride`.
        factor_instance_stride: u64,
    },
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:
    /// tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: each buffer as the address its reads start at
    /// (Metal's `drawIndexedPatchesArguments` answers the patch index and control-point index
    /// buffers so, a probe).
    IndexedPatches {
        /// `numberOfPatchControlPoints`.
        control_points: u64,
        /// `patchStart`.
        patch_start: u64,
        /// `patchCount`.
        patch_count: u64,
        /// `patchIndexBuffer` plus `patchIndexBufferOffset`; `None` for none.
        patch_index_buffer: Option<GpuAddress>,
        /// `controlPointIndexBuffer` plus `controlPointIndexBufferOffset`; `None` for none.
        control_point_index_buffer: Option<GpuAddress>,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseInstance`.
        base_instance: u64,
        /// `tessellationFactorBuffer` plus `tessellationFactorBufferOffset`.
        factor_buffer: GpuAddress,
        /// `tessellationFactorBufferInstanceStride`.
        factor_instance_stride: u64,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    MeshThreadgroups {
        /// `threadgroupsPerGrid`: of the object function where the pipeline has one, else of
        /// the mesh function.
        threadgroups_per_grid: Size,
        /// `threadsPerObjectThreadgroup`.
        threads_per_object_threadgroup: Size,
        /// `threadsPerMeshThreadgroup`.
        threads_per_mesh_threadgroup: Size,
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    MeshThreads {
        /// `threadsPerGrid`.
        threads_per_grid: Size,
        /// `threadsPerObjectThreadgroup`.
        threads_per_object_threadgroup: Size,
        /// `threadsPerMeshThreadgroup`.
        threads_per_mesh_threadgroup: Size,
    },
}

/// The address Metal's private patch draw getters (`drawPatchesArguments`,
/// `drawIndexedPatchesArguments`) answer for a nil patch index or control-point index buffer,
/// whatever its offset (the M3 Max under macOS 27.0.1: offsets 0, 0x10, 0xfffffff0,
/// 0x100000010 and 0x1eeee1111; and the consumer's M3 Max under macOS 26.7, on a fresh command
/// and on one that held a buffer).
pub const NIL_PATCH_BUFFER_ADDRESS: u64 = 0x1_eeee_1111;

/// Apple's private getters of a patch draw's buffers, as Metal answers them.
impl IndirectDraw {
    /// `patchIndexBufferGPUVirtualAddress` of `drawPatchesArguments` and
    /// `drawIndexedPatchesArguments` (SPI): a patch draw's patch index buffer's address plus the
    /// offset's low 32 bits, or [`NIL_PATCH_BUFFER_ADDRESS`] for none; `None` for a draw of
    /// another kind, which the runtime does not read as a patch draw.
    pub fn patch_index_buffer_address(&self) -> Option<u64> {
        match *self {
            IndirectDraw::Patches {
                patch_index_buffer, ..
            }
            | IndirectDraw::IndexedPatches {
                patch_index_buffer, ..
            } => Some(patch_index_buffer.map_or(NIL_PATCH_BUFFER_ADDRESS, |address| address.0)),
            _ => None,
        }
    }

    /// `controlPointIndexBufferGPUVirtualAddress` of `drawIndexedPatchesArguments` (SPI): an
    /// indexed patch draw's control-point index buffer's address plus the offset's low 32 bits,
    /// and [`NIL_PATCH_BUFFER_ADDRESS`] for a patch draw, which an indexed one of no control-point
    /// index buffer is, as Metal answers it; `None` for a draw of another kind.
    pub fn control_point_index_buffer_address(&self) -> Option<u64> {
        match *self {
            IndirectDraw::Patches { .. } => Some(NIL_PATCH_BUFFER_ADDRESS),
            IndirectDraw::IndexedPatches {
                control_point_index_buffer,
                ..
            } => Some(
                control_point_index_buffer.map_or(NIL_PATCH_BUFFER_ADDRESS, |address| address.0),
            ),
            _ => None,
        }
    }
}
