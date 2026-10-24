//! The indirect-command-buffer vocabulary: what a case asks an `MTLIndirectCommandBuffer` to
//! hold, and what it asks each of its commands to be.
//!
//! One buffer is **a descriptor and a list of commands** ([`IcbDescriptor`], [`EncodedCommand`]),
//! and a command is a list of the setter calls the host makes on it, in order. The steps are a
//! list rather than a struct of fields for the reason the render vocabulary's are: setting a
//! binding twice, resetting a command after encoding it, and setting a barrier and then clearing
//! it are *cases*, and a case states one by listing both calls. A struct of final values could
//! not say which call came last.
//!
//! Each step names its own Metal selector ([`ComputeCommandStep::call`],
//! [`RenderCommandStep::call`]), so a refusal of ours reads as the Metal call we refused, and
//! carries the parameters its call takes ([`ComputeCommandStep::parameters`]) so a case's input
//! fingerprint covers them without the handles, whose numbers are a side's own.
//!
//! # What this vocabulary deliberately cannot say
//!
//! **A render command's own render states.** `setDepthStencilState:`, `setDepthBias:slopeScale:
//! clamp:`, `setDepthClipMode:`, `setCullMode:`, `setFrontFacingWinding:` and
//! `setTriangleFillMode:` have no step here and must not get one. The runtime's own measurement of
//! an Apple M4 on macOS 27.0.1 is that **each of those setters raises `APINotSupported` and ends
//! Metal's process** on a device that does not claim the `Apple10` family — which this suite's
//! oracle, an Apple M3 Max, does not either
//! (`metal2vulkan_runtime::indirect_command_buffer`'s module documentation, "Render states a
//! command does not inherit"). A case that called one would end the recorder, so the calls are
//! named here and left out.
//!
//! The **descriptor** side of the same feature is a different matter and is here: plain Metal
//! *makes* a buffer that leaves those states to its commands, and a case may observe that it did
//! ([`IcbDescriptor::inherit_depth_stencil_state`] and the five beside it). Metal's own API
//! validation stops that creation, which is a fact the recording's second oracle keeps.
//!
//! Nothing else is left out on purpose. A patch draw is **here**
//! ([`RenderCommandStep::DrawPatches`], [`RenderCommandStep::DrawIndexedPatches`]): it needs a
//! tessellation pipeline, which [`crate::render::RenderPipelineSpec::tessellating`] builds, and
//! `supportIndirectCommandBuffers` beside it.

use crate::case::{Failure, Handle};
use metal2vulkan::metal::{IndexType, IndirectCommandType, PrimitiveType};

/// `MTLIndirectCommandBufferDescriptor` as a case states it.
///
/// [`IcbDescriptor::new`] is Metal's own default for every field, so a case names only what it
/// means to change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IcbDescriptor {
    /// `commandTypes`.
    pub command_types: IndirectCommandType,
    /// `inheritPipelineState`.
    pub inherit_pipeline_state: bool,
    /// `inheritBuffers`.
    pub inherit_buffers: bool,
    /// `maxVertexBufferBindCount`.
    pub max_vertex_buffer_bind_count: u32,
    /// `maxFragmentBufferBindCount`.
    pub max_fragment_buffer_bind_count: u32,
    /// `maxKernelBufferBindCount`.
    pub max_kernel_buffer_bind_count: u32,
    /// `maxKernelThreadgroupMemoryBindCount`. Metal's default is 31, which is what
    /// [`IcbDescriptor::new`] holds.
    pub max_kernel_threadgroup_memory_bind_count: u32,
    /// `maxObjectBufferBindCount`.
    pub max_object_buffer_bind_count: u32,
    /// `maxMeshBufferBindCount`.
    pub max_mesh_buffer_bind_count: u32,
    /// `maxObjectThreadgroupMemoryBindCount`.
    pub max_object_threadgroup_memory_bind_count: u32,
    /// `supportDynamicAttributeStride`: whether a command may bind a buffer at an attribute
    /// stride of its own.
    pub support_dynamic_attribute_stride: bool,
    /// `supportColorAttachmentMapping`.
    pub support_color_attachment_mapping: bool,
    /// `inheritDepthStencilState`. `false` leaves the state to the command, which **no step of
    /// this vocabulary can then set** — the setter ends Metal's process on this oracle (the module
    /// documentation). What a case observes of `false` is that the buffer was made at all.
    pub inherit_depth_stencil_state: bool,
    /// `inheritDepthBias`, as [`IcbDescriptor::inherit_depth_stencil_state`].
    pub inherit_depth_bias: bool,
    /// `inheritDepthClipMode`, as [`IcbDescriptor::inherit_depth_stencil_state`].
    pub inherit_depth_clip_mode: bool,
    /// `inheritCullMode`, as [`IcbDescriptor::inherit_depth_stencil_state`].
    pub inherit_cull_mode: bool,
    /// `inheritFrontFacingWinding`, as [`IcbDescriptor::inherit_depth_stencil_state`].
    pub inherit_front_facing_winding: bool,
    /// `inheritTriangleFillMode`, as [`IcbDescriptor::inherit_depth_stencil_state`].
    pub inherit_triangle_fill_mode: bool,
}

impl IcbDescriptor {
    /// A descriptor of Metal's own defaults: no command type, nothing inherited of the pipeline
    /// state or the buffers, no buffer of any stage, 31 kernel threadgroup lengths, no dynamic
    /// attribute stride, no color attachment mapping, and every render state inherited.
    pub const fn new() -> Self {
        Self {
            command_types: IndirectCommandType::empty(),
            inherit_pipeline_state: false,
            inherit_buffers: false,
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_kernel_buffer_bind_count: 0,
            max_kernel_threadgroup_memory_bind_count: 31,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            max_object_threadgroup_memory_bind_count: 0,
            support_dynamic_attribute_stride: false,
            support_color_attachment_mapping: false,
            inherit_depth_stencil_state: true,
            inherit_depth_bias: true,
            inherit_depth_clip_mode: true,
            inherit_cull_mode: true,
            inherit_front_facing_winding: true,
            inherit_triangle_fill_mode: true,
        }
    }

    /// The descriptor's own words, for a case's input fingerprint: every field in declaration
    /// order, so a case that changes one feeds a different run.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.command_types.as_raw().to_le_bytes());
        for flag in [
            self.inherit_pipeline_state,
            self.inherit_buffers,
            self.support_dynamic_attribute_stride,
            self.support_color_attachment_mapping,
            self.inherit_depth_stencil_state,
            self.inherit_depth_bias,
            self.inherit_depth_clip_mode,
            self.inherit_cull_mode,
            self.inherit_front_facing_winding,
            self.inherit_triangle_fill_mode,
        ] {
            out.push(u8::from(flag));
        }
        for count in [
            self.max_vertex_buffer_bind_count,
            self.max_fragment_buffer_bind_count,
            self.max_kernel_buffer_bind_count,
            self.max_kernel_threadgroup_memory_bind_count,
            self.max_object_buffer_bind_count,
            self.max_mesh_buffer_bind_count,
            self.max_object_threadgroup_memory_bind_count,
        ] {
            out.extend(count.to_le_bytes());
        }
        out
    }
}

impl Default for IcbDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

/// One call the host makes on an `MTLIndirectComputeCommand`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeCommandStep {
    /// `setComputePipelineState:` of a state made with `supportIndirectCommandBuffers`.
    Pipeline(Handle),
    /// `setKernelBuffer:offset:atIndex:`.
    KernelBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setKernelBuffer:offset:attributeStride:atIndex:`, which the buffer's descriptor must
    /// support ([`IcbDescriptor::support_dynamic_attribute_stride`]).
    KernelBufferWithStride {
        buffer: Handle,
        offset: u64,
        stride: u64,
        index: u32,
    },
    /// `setThreadgroupMemoryLength:atIndex:`.
    ThreadgroupMemoryLength { length: u32, index: u32 },
    /// `setStageInRegion:`.
    StageInRegion { origin: [u32; 3], size: [u32; 3] },
    /// `setImageblockWidth:height:`.
    ImageblockSize { width: u32, height: u32 },
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`.
    DispatchThreadgroups { groups: [u32; 3], threads: [u32; 3] },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`.
    DispatchThreads {
        threads: [u32; 3],
        per_group: [u32; 3],
    },
    /// `setBarrier`.
    Barrier,
    /// `clearBarrier`.
    ClearBarrier,
    /// `reset`.
    Reset,
}

impl ComputeCommandStep {
    /// The Metal selector the step makes, so a refusal reads as the call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            ComputeCommandStep::Pipeline(_) => "setComputePipelineState",
            ComputeCommandStep::KernelBuffer { .. } => "setKernelBuffer",
            ComputeCommandStep::KernelBufferWithStride { .. } => "setKernelBufferAttributeStride",
            ComputeCommandStep::ThreadgroupMemoryLength { .. } => "setThreadgroupMemoryLength",
            ComputeCommandStep::StageInRegion { .. } => "setStageInRegion",
            ComputeCommandStep::ImageblockSize { .. } => "setImageblockWidthHeight",
            ComputeCommandStep::DispatchThreadgroups { .. } => "concurrentDispatchThreadgroups",
            ComputeCommandStep::DispatchThreads { .. } => "concurrentDispatchThreads",
            ComputeCommandStep::Barrier => "setBarrier",
            ComputeCommandStep::ClearBarrier => "clearBarrier",
            ComputeCommandStep::Reset => "reset",
        }
    }

    /// The parameters the step's own call carries, for a case's input fingerprint: everything but
    /// the handles, whose numbers are a side's own.
    pub fn parameters(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ComputeCommandStep::Pipeline(_)
            | ComputeCommandStep::Barrier
            | ComputeCommandStep::ClearBarrier
            | ComputeCommandStep::Reset => {}
            ComputeCommandStep::KernelBuffer { offset, index, .. } => {
                out.extend(offset.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            ComputeCommandStep::KernelBufferWithStride {
                offset,
                stride,
                index,
                ..
            } => {
                out.extend(offset.to_le_bytes());
                out.extend(stride.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            ComputeCommandStep::ThreadgroupMemoryLength { length, index } => {
                out.extend(length.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            ComputeCommandStep::StageInRegion { origin, size } => {
                for axis in origin.iter().chain(size) {
                    out.extend(axis.to_le_bytes());
                }
            }
            ComputeCommandStep::ImageblockSize { width, height } => {
                out.extend(width.to_le_bytes());
                out.extend(height.to_le_bytes());
            }
            ComputeCommandStep::DispatchThreadgroups { groups, threads } => {
                for axis in groups.iter().chain(threads) {
                    out.extend(axis.to_le_bytes());
                }
            }
            ComputeCommandStep::DispatchThreads { threads, per_group } => {
                for axis in threads.iter().chain(per_group) {
                    out.extend(axis.to_le_bytes());
                }
            }
        }
        out
    }
}

/// One call the host makes on an `MTLIndirectRenderCommand`.
///
/// The six render-state setters are **absent on purpose**; the module documentation says why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderCommandStep {
    /// `setRenderPipelineState:` of a state made with `supportIndirectCommandBuffers`.
    Pipeline(Handle),
    /// `setVertexBuffer:offset:atIndex:`.
    VertexBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setVertexBuffer:offset:attributeStride:atIndex:`, which the buffer's descriptor must
    /// support ([`IcbDescriptor::support_dynamic_attribute_stride`]).
    VertexBufferWithStride {
        buffer: Handle,
        offset: u64,
        stride: u64,
        index: u32,
    },
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setObjectBuffer:offset:atIndex:`.
    ObjectBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setMeshBuffer:offset:atIndex:`.
    MeshBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setObjectThreadgroupMemoryLength:atIndex:`.
    ObjectThreadgroupMemoryLength { length: u32, index: u32 },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Draw {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
        instances: u32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:`.
    DrawIndexed {
        primitive: PrimitiveType,
        count: u32,
        index_type: IndexType,
        indices: Handle,
        offset: u64,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`.
    ///
    /// A command's patch draw carries its **own** factor buffer, where a render encoder's draw
    /// reads the one `setTessellationFactorBuffer:offset:instanceStride:` bound before it. So two
    /// commands of one buffer can draw the same patches at different factors with no state
    /// between them, which is the shape a case over a buffer of patch draws states.
    DrawPatches {
        control_points: u32,
        /// `patchStart`.
        first: u32,
        /// `patchCount`.
        count: u32,
        /// `patchIndexBuffer`; `None` is Metal's nil, each patch drawn at its own index.
        patch_indices: Option<Handle>,
        patch_index_offset: u64,
        instances: u32,
        base_instance: u32,
        /// `tessellationFactorBuffer`, which the call takes and never nil.
        factors: Handle,
        factor_offset: u64,
        factor_instance_stride: u64,
    },
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:
    /// tessellationFactorBuffer:tessellationFactorBufferOffset:
    /// tessellationFactorBufferInstanceStride:`: [`RenderCommandStep::DrawPatches`] whose control
    /// point `k` of patch `p` is the index at position `p · N + k` of the control-point index
    /// buffer, `N` the pipeline's control points.
    DrawIndexedPatches {
        control_points: u32,
        first: u32,
        count: u32,
        patch_indices: Option<Handle>,
        patch_index_offset: u64,
        /// `controlPointIndexBuffer`. Not optional, though Metal's own selector takes nil:
        /// `objc2`'s binding of it declares the parameter non-null, so no case of this suite can
        /// state the nil that makes the call a plain patch draw.
        control_point_indices: Handle,
        control_point_index_offset: u64,
        instances: u32,
        base_instance: u32,
        factors: Handle,
        factor_offset: u64,
        factor_instance_stride: u64,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawMeshThreadgroups {
        groups: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawMeshThreads {
        threads: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `setBarrier`.
    Barrier,
    /// `clearBarrier`.
    ClearBarrier,
    /// `reset`.
    Reset,
}

impl RenderCommandStep {
    /// The Metal selector the step makes.
    pub fn call(&self) -> &'static str {
        match self {
            RenderCommandStep::Pipeline(_) => "setRenderPipelineState",
            RenderCommandStep::VertexBuffer { .. } => "setVertexBuffer",
            RenderCommandStep::VertexBufferWithStride { .. } => "setVertexBufferAttributeStride",
            RenderCommandStep::FragmentBuffer { .. } => "setFragmentBuffer",
            RenderCommandStep::ObjectBuffer { .. } => "setObjectBuffer",
            RenderCommandStep::MeshBuffer { .. } => "setMeshBuffer",
            RenderCommandStep::ObjectThreadgroupMemoryLength { .. } => {
                "setObjectThreadgroupMemoryLength"
            }
            RenderCommandStep::Draw { .. } => "drawPrimitives",
            RenderCommandStep::DrawIndexed { .. } => "drawIndexedPrimitives",
            RenderCommandStep::DrawPatches { .. } => "drawPatches",
            RenderCommandStep::DrawIndexedPatches { .. } => "drawIndexedPatches",
            RenderCommandStep::DrawMeshThreadgroups { .. } => "drawMeshThreadgroups",
            RenderCommandStep::DrawMeshThreads { .. } => "drawMeshThreads",
            RenderCommandStep::Barrier => "setBarrier",
            RenderCommandStep::ClearBarrier => "clearBarrier",
            RenderCommandStep::Reset => "reset",
        }
    }

    /// The parameters the step's own call carries, for a case's input fingerprint.
    pub fn parameters(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            RenderCommandStep::Pipeline(_)
            | RenderCommandStep::Barrier
            | RenderCommandStep::ClearBarrier
            | RenderCommandStep::Reset => {}
            RenderCommandStep::VertexBuffer { offset, index, .. }
            | RenderCommandStep::FragmentBuffer { offset, index, .. }
            | RenderCommandStep::ObjectBuffer { offset, index, .. }
            | RenderCommandStep::MeshBuffer { offset, index, .. } => {
                out.extend(offset.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            RenderCommandStep::VertexBufferWithStride {
                offset,
                stride,
                index,
                ..
            } => {
                out.extend(offset.to_le_bytes());
                out.extend(stride.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            RenderCommandStep::ObjectThreadgroupMemoryLength { length, index } => {
                out.extend(length.to_le_bytes());
                out.extend(index.to_le_bytes());
            }
            RenderCommandStep::Draw {
                primitive,
                first,
                count,
                instances,
                base_instance,
            } => {
                out.extend(primitive.as_raw().to_le_bytes());
                out.extend(first.to_le_bytes());
                out.extend(count.to_le_bytes());
                out.extend(instances.to_le_bytes());
                out.extend(base_instance.to_le_bytes());
            }
            RenderCommandStep::DrawIndexed {
                primitive,
                count,
                index_type,
                offset,
                instances,
                base_vertex,
                base_instance,
                ..
            } => {
                out.extend(primitive.as_raw().to_le_bytes());
                out.extend(count.to_le_bytes());
                out.extend(index_type.as_raw().to_le_bytes());
                out.extend(offset.to_le_bytes());
                out.extend(instances.to_le_bytes());
                out.extend(base_vertex.to_le_bytes());
                out.extend(base_instance.to_le_bytes());
            }
            RenderCommandStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
                factor_offset,
                factor_instance_stride,
                ..
            } => {
                out.extend(control_points.to_le_bytes());
                out.extend(first.to_le_bytes());
                out.extend(count.to_le_bytes());
                out.push(u8::from(patch_indices.is_some()));
                out.extend(patch_index_offset.to_le_bytes());
                out.extend(instances.to_le_bytes());
                out.extend(base_instance.to_le_bytes());
                out.extend(factor_offset.to_le_bytes());
                out.extend(factor_instance_stride.to_le_bytes());
            }
            RenderCommandStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_index_offset,
                instances,
                base_instance,
                factor_offset,
                factor_instance_stride,
                ..
            } => {
                out.extend(control_points.to_le_bytes());
                out.extend(first.to_le_bytes());
                out.extend(count.to_le_bytes());
                out.push(u8::from(patch_indices.is_some()));
                out.extend(patch_index_offset.to_le_bytes());
                out.extend(control_point_index_offset.to_le_bytes());
                out.extend(instances.to_le_bytes());
                out.extend(base_instance.to_le_bytes());
                out.extend(factor_offset.to_le_bytes());
                out.extend(factor_instance_stride.to_le_bytes());
            }
            RenderCommandStep::DrawMeshThreadgroups {
                groups: first,
                object,
                mesh,
            }
            | RenderCommandStep::DrawMeshThreads {
                threads: first,
                object,
                mesh,
            } => {
                for axis in first.iter().chain(object).chain(mesh) {
                    out.extend(axis.to_le_bytes());
                }
            }
        }
        out
    }
}

/// One command of a buffer, as the host encodes it: which kind it is, and the calls in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodedCommand {
    /// `indirectComputeCommandAtIndex:`, and the calls on it.
    Compute(Vec<ComputeCommandStep>),
    /// `indirectRenderCommandAtIndex:`, and the calls on it.
    Render(Vec<RenderCommandStep>),
}

impl EncodedCommand {
    /// The getter the command is reached through, so a case's fingerprint says which kind it is.
    pub fn call(&self) -> &'static str {
        match self {
            EncodedCommand::Compute(_) => "indirectComputeCommandAtIndex",
            EncodedCommand::Render(_) => "indirectRenderCommandAtIndex",
        }
    }

    /// How many setter calls the command makes, so a side can answer one entry per call without
    /// the case reaching into the variant.
    pub fn steps(&self) -> usize {
        match self {
            EncodedCommand::Compute(steps) => steps.len(),
            EncodedCommand::Render(steps) => steps.len(),
        }
    }

    /// The name of step `at`, or `"call"` past the end.
    pub fn step_call(&self, at: usize) -> &'static str {
        match self {
            EncodedCommand::Compute(steps) => {
                steps.get(at).map_or("call", ComputeCommandStep::call)
            }
            EncodedCommand::Render(steps) => steps.get(at).map_or("call", RenderCommandStep::call),
        }
    }

    /// The command's own words, for a case's input fingerprint: the getter's name, then each
    /// step's call and its parameters.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.call().as_bytes());
        match self {
            EncodedCommand::Compute(steps) => {
                for step in steps {
                    out.push(b'|');
                    out.extend(step.call().as_bytes());
                    out.push(b':');
                    out.extend(step.parameters());
                }
            }
            EncodedCommand::Render(steps) => {
                for step in steps {
                    out.push(b'|');
                    out.extend(step.call().as_bytes());
                    out.push(b':');
                    out.extend(step.parameters());
                }
            }
        }
        out
    }
}

/// What making a buffer and encoding its commands answered.
///
/// Metal's setters answer nothing, so its side answers `None` for every step and a refusal of ours
/// compares against that and reads as a refusal rather than a divergence — the same arrangement
/// [`crate::case::PassOutcome`] uses for a pass's steps.
#[derive(Clone, Debug)]
pub struct CommandsMade {
    /// The buffer.
    pub commands: Handle,
    /// What each command's steps answered, one list a command, in the order the case gave them.
    pub refused: Vec<Vec<Option<Failure>>>,
}
