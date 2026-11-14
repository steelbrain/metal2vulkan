//! A command's setters called in turn in one call ([`IndirectRenderCommand::set_each`],
//! [`IndirectComputeCommand::set_each`], and without a command's handle
//! [`IndirectCommandBuffer::set_render_command`] and
//! [`IndirectCommandBuffer::set_compute_command`]): a runtime extension with no Metal
//! counterpart, whose `MTLIndirectRenderCommand` and `MTLIndirectComputeCommand` have the setters
//! alone. Each call does what its setter does, refusals included; the first refused stops the
//! rest, its place answered with its error, and the calls before it stay made, as they would have
//! one setter call at a time. Their writes are one [`Batch`]: the storage's bytes found once, and
//! the writes made without the buffer's lock counted once.

use super::{
    Batch, ComputeAt, IndirectCommandBuffer, IndirectComputeCommand, IndirectRenderCommand,
    RenderAt,
};
use crate::depth_stencil::DepthStencilState;
use crate::render_values::DrawPrimitiveType;
use crate::values::{Region, Size};
use crate::{Buffer, ComputePipelineState, Error, RenderPipelineState};
use metal2vulkan::metal::{CullMode, DepthClipMode, IndexType, TriangleFillMode, Winding};

/// One call of an [`IndirectRenderCommand`]'s setters, for
/// [`set_each`](IndirectRenderCommand::set_each): a variant a setter, named after it, its fields
/// the setter's arguments, so each does what its setter's Metal counterpart does there (its
/// docs). A runtime extension: Metal has no such value.
#[derive(Clone, Copy, Debug)]
pub enum IndirectRenderSet<'a> {
    /// [`set_render_pipeline_state`](IndirectRenderCommand::set_render_pipeline_state)
    /// (`setRenderPipelineState:`).
    SetRenderPipelineState(&'a RenderPipelineState),
    /// [`set_vertex_buffer`](IndirectRenderCommand::set_vertex_buffer)
    /// (`setVertexBuffer:offset:atIndex:`).
    SetVertexBuffer {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_vertex_buffer_with_stride`](IndirectRenderCommand::set_vertex_buffer_with_stride)
    /// (`setVertexBuffer:offset:attributeStride:atIndex:`).
    SetVertexBufferWithStride {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `attributeStride`.
        stride: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_fragment_buffer`](IndirectRenderCommand::set_fragment_buffer)
    /// (`setFragmentBuffer:offset:atIndex:`).
    SetFragmentBuffer {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_object_buffer`](IndirectRenderCommand::set_object_buffer)
    /// (`setObjectBuffer:offset:atIndex:`).
    SetObjectBuffer {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_mesh_buffer`](IndirectRenderCommand::set_mesh_buffer)
    /// (`setMeshBuffer:offset:atIndex:`).
    SetMeshBuffer {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_object_threadgroup_memory_length`](IndirectRenderCommand::set_object_threadgroup_memory_length)
    /// (`setObjectThreadgroupMemoryLength:atIndex:`).
    SetObjectThreadgroupMemoryLength {
        /// `length`.
        length: u32,
        /// `atIndex`.
        index: u32,
    },
    /// [`draw_primitives`](IndirectRenderCommand::draw_primitives)
    /// (`drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`).
    DrawPrimitives {
        /// `primitiveType`.
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
    /// [`draw_indexed_primitives`](IndirectRenderCommand::draw_indexed_primitives)
    /// (`drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`).
    DrawIndexedPrimitives {
        /// `primitiveType`.
        primitive_type: DrawPrimitiveType,
        /// `indexCount`.
        index_count: u64,
        /// `indexType`.
        index_type: IndexType,
        /// `indexBuffer`.
        index_buffer: &'a Buffer,
        /// `indexBufferOffset`.
        index_buffer_offset: u64,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseVertex`.
        base_vertex: i64,
        /// `baseInstance`.
        base_instance: u64,
    },
    /// [`draw_patches`](IndirectRenderCommand::draw_patches)
    /// (`drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:`).
    DrawPatches {
        /// `numberOfPatchControlPoints`.
        number_of_patch_control_points: u64,
        /// `patchStart`.
        patch_start: u64,
        /// `patchCount`.
        patch_count: u64,
        /// `patchIndexBuffer`.
        patch_index_buffer: Option<&'a Buffer>,
        /// `patchIndexBufferOffset`.
        patch_index_buffer_offset: u64,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseInstance`.
        base_instance: u64,
        /// `tessellationFactorBuffer`.
        tessellation_factor_buffer: &'a Buffer,
        /// `tessellationFactorBufferOffset`.
        tessellation_factor_buffer_offset: u64,
        /// `tessellationFactorBufferInstanceStride`.
        tessellation_factor_buffer_instance_stride: u64,
    },
    /// [`draw_indexed_patches`](IndirectRenderCommand::draw_indexed_patches)
    /// (`drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:`).
    DrawIndexedPatches {
        /// `numberOfPatchControlPoints`.
        number_of_patch_control_points: u64,
        /// `patchStart`.
        patch_start: u64,
        /// `patchCount`.
        patch_count: u64,
        /// `patchIndexBuffer`.
        patch_index_buffer: Option<&'a Buffer>,
        /// `patchIndexBufferOffset`.
        patch_index_buffer_offset: u64,
        /// `controlPointIndexBuffer`.
        control_point_index_buffer: Option<&'a Buffer>,
        /// `controlPointIndexBufferOffset`.
        control_point_index_buffer_offset: u64,
        /// `instanceCount`.
        instance_count: u64,
        /// `baseInstance`.
        base_instance: u64,
        /// `tessellationFactorBuffer`.
        tessellation_factor_buffer: &'a Buffer,
        /// `tessellationFactorBufferOffset`.
        tessellation_factor_buffer_offset: u64,
        /// `tessellationFactorBufferInstanceStride`.
        tessellation_factor_buffer_instance_stride: u64,
    },
    /// [`draw_mesh_threadgroups`](IndirectRenderCommand::draw_mesh_threadgroups)
    /// (`drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`).
    DrawMeshThreadgroups {
        /// `threadgroupsPerGrid`.
        threadgroups_per_grid: Size,
        /// `threadsPerObjectThreadgroup`.
        threads_per_object_threadgroup: Size,
        /// `threadsPerMeshThreadgroup`.
        threads_per_mesh_threadgroup: Size,
    },
    /// [`draw_mesh_threads`](IndirectRenderCommand::draw_mesh_threads)
    /// (`drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`).
    DrawMeshThreads {
        /// `threadsPerGrid`.
        threads_per_grid: Size,
        /// `threadsPerObjectThreadgroup`.
        threads_per_object_threadgroup: Size,
        /// `threadsPerMeshThreadgroup`.
        threads_per_mesh_threadgroup: Size,
    },
    /// [`set_barrier`](IndirectRenderCommand::set_barrier) (`setBarrier`).
    SetBarrier,
    /// [`clear_barrier`](IndirectRenderCommand::clear_barrier) (`clearBarrier`).
    ClearBarrier,
    /// [`reset`](IndirectRenderCommand::reset) (`reset`).
    Reset,
    /// [`set_depth_stencil_state`](IndirectRenderCommand::set_depth_stencil_state)
    /// (`setDepthStencilState:`).
    SetDepthStencilState(Option<&'a DepthStencilState>),
    /// [`set_depth_bias`](IndirectRenderCommand::set_depth_bias)
    /// (`setDepthBias:slopeScale:clamp:`).
    SetDepthBias {
        /// `depthBias`.
        depth_bias: f32,
        /// `slopeScale`.
        slope_scale: f32,
        /// `clamp`.
        clamp: f32,
    },
    /// [`set_depth_clip_mode`](IndirectRenderCommand::set_depth_clip_mode)
    /// (`setDepthClipMode:`).
    SetDepthClipMode(DepthClipMode),
    /// [`set_cull_mode`](IndirectRenderCommand::set_cull_mode) (`setCullMode:`).
    SetCullMode(CullMode),
    /// [`set_front_facing_winding`](IndirectRenderCommand::set_front_facing_winding)
    /// (`setFrontFacingWinding:`).
    SetFrontFacingWinding(Winding),
    /// [`set_triangle_fill_mode`](IndirectRenderCommand::set_triangle_fill_mode)
    /// (`setTriangleFillMode:`).
    SetTriangleFillMode(TriangleFillMode),
}

/// One call of an [`IndirectComputeCommand`]'s setters, for
/// [`set_each`](IndirectComputeCommand::set_each), as [`IndirectRenderSet`] is a render
/// command's. A runtime extension: Metal has no such value.
#[derive(Clone, Copy, Debug)]
pub enum IndirectComputeSet<'a> {
    /// [`set_compute_pipeline_state`](IndirectComputeCommand::set_compute_pipeline_state)
    /// (`setComputePipelineState:`).
    SetComputePipelineState(&'a ComputePipelineState),
    /// [`set_kernel_buffer`](IndirectComputeCommand::set_kernel_buffer)
    /// (`setKernelBuffer:offset:atIndex:`).
    SetKernelBuffer {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_kernel_buffer_with_stride`](IndirectComputeCommand::set_kernel_buffer_with_stride)
    /// (`setKernelBuffer:offset:attributeStride:atIndex:`).
    SetKernelBufferWithStride {
        /// `buffer`.
        buffer: Option<&'a Buffer>,
        /// `offset`.
        offset: u64,
        /// `attributeStride`.
        stride: u64,
        /// `atIndex`.
        index: u32,
    },
    /// [`concurrent_dispatch_threadgroups`](IndirectComputeCommand::concurrent_dispatch_threadgroups)
    /// (`concurrentDispatchThreadgroups:threadsPerThreadgroup:`).
    ConcurrentDispatchThreadgroups {
        /// `threadgroupsPerGrid`.
        threadgroups: Size,
        /// `threadsPerThreadgroup`.
        threads_per_threadgroup: Size,
    },
    /// [`concurrent_dispatch_threads`](IndirectComputeCommand::concurrent_dispatch_threads)
    /// (`concurrentDispatchThreads:threadsPerThreadgroup:`).
    ConcurrentDispatchThreads {
        /// `threadsPerGrid`.
        threads: Size,
        /// `threadsPerThreadgroup`.
        threads_per_threadgroup: Size,
    },
    /// [`set_barrier`](IndirectComputeCommand::set_barrier) (`setBarrier`).
    SetBarrier,
    /// [`clear_barrier`](IndirectComputeCommand::clear_barrier) (`clearBarrier`).
    ClearBarrier,
    /// [`set_threadgroup_memory_length`](IndirectComputeCommand::set_threadgroup_memory_length)
    /// (`setThreadgroupMemoryLength:atIndex:`).
    SetThreadgroupMemoryLength {
        /// `length`.
        length: u32,
        /// `atIndex`.
        index: u32,
    },
    /// [`set_stage_in_region`](IndirectComputeCommand::set_stage_in_region)
    /// (`setStageInRegion:`).
    SetStageInRegion(Region),
    /// [`set_imageblock_size`](IndirectComputeCommand::set_imageblock_size)
    /// (`setImageblockWidth:height:`).
    SetImageblockSize {
        /// `width`.
        width: u64,
        /// `height`.
        height: u64,
    },
    /// [`reset`](IndirectComputeCommand::reset) (`reset`).
    Reset,
}

impl IndirectCommandBuffer {
    /// `indirectRenderCommandAtIndex:` and the setters `sets` name on its command, called in turn
    /// ([`IndirectRenderSet`]), as [`IndirectRenderCommand::set_each`] calls them: a runtime
    /// extension, the calls Metal makes one at a time made in one, with no command's handle (the
    /// buffer's is not cloned). Command `index`'s refusals, as
    /// [`indirect_render_command_at`](Self::indirect_render_command_at)'s, are answered at place
    /// 0, before any setter is called, `sets` empty or not; a setter's refusal at its place, those
    /// before it made and those after it not taken.
    pub fn set_render_command<'a>(
        &self,
        index: u64,
        sets: impl IntoIterator<Item = IndirectRenderSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        const CALL: &str = "set_render_command";
        let index = (self.command_at(CALL, index, true)).map_err(|error| (0, error))?;
        RenderAt { icb: self, index }.set_each(sets)
    }

    /// `indirectComputeCommandAtIndex:` and the setters `sets` name on its command, called in turn
    /// ([`IndirectComputeSet`]), as [`IndirectComputeCommand::set_each`] calls them: a runtime
    /// extension with no command's handle, as [`set_render_command`](Self::set_render_command)
    /// is. Command `index`'s refusals, as
    /// [`indirect_compute_command_at`](Self::indirect_compute_command_at)'s (a buffer of render
    /// command types among them), are answered at place 0, before any setter is called.
    pub fn set_compute_command<'a>(
        &self,
        index: u64,
        sets: impl IntoIterator<Item = IndirectComputeSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        const CALL: &str = "set_compute_command";
        let index = (self.command_at(CALL, index, false)).map_err(|error| (0, error))?;
        ComputeAt { icb: self, index }.set_each(sets)
    }
}

impl IndirectRenderCommand {
    /// The setters `sets` name, called in turn ([`IndirectRenderSet`]): a runtime extension with
    /// no Metal counterpart, each call doing what its setter does. The first a setter refuses
    /// stops them, answered with its place among them and the setter's error; those before it
    /// stay made, and those after it are not taken, as one setter call at a time would leave the
    /// command.
    pub fn set_each<'a>(
        &self,
        sets: impl IntoIterator<Item = IndirectRenderSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        self.at().set_each(sets)
    }
}

impl IndirectComputeCommand {
    /// The setters `sets` name, called in turn ([`IndirectComputeSet`]), as
    /// [`IndirectRenderCommand::set_each`] calls a render command's: a runtime extension with no
    /// Metal counterpart; the first refused stops them, answered with its place and error, those
    /// before it made.
    pub fn set_each<'a>(
        &self,
        sets: impl IntoIterator<Item = IndirectComputeSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        self.at().set_each(sets)
    }
}

impl RenderAt<'_> {
    /// [`IndirectRenderCommand::set_each`] of `sets`, their writes one [`Batch`].
    fn set_each<'a>(
        &self,
        sets: impl IntoIterator<Item = IndirectRenderSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        let mut batch = self.icb.batch(self.index);
        for (at, set) in sets.into_iter().enumerate() {
            self.set_one(&mut batch, set).map_err(|error| (at, error))?;
        }
        Ok(())
    }

    /// The setter `set` names: in `batch` where the setter writes a field, else the setter.
    #[inline(always)]
    fn set_one(&self, batch: &mut Batch<'_>, set: IndirectRenderSet<'_>) -> Result<(), Error> {
        use IndirectRenderSet as S;
        match set {
            S::SetRenderPipelineState(state) => self.pipeline_in(batch, state),
            S::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => self.vertex_buffer_in(batch, buffer, offset, index),
            S::SetVertexBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => self.vertex_buffer_with_stride_in(batch, buffer, offset, stride, index),
            S::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => self.fragment_buffer_in(batch, buffer, offset, index),
            S::SetObjectBuffer {
                buffer,
                offset,
                index,
            } => self.set_object_buffer(buffer, offset, index),
            S::SetMeshBuffer {
                buffer,
                offset,
                index,
            } => self.set_mesh_buffer(buffer, offset, index),
            S::SetObjectThreadgroupMemoryLength { length, index } => {
                self.set_object_threadgroup_memory_length(length, index)
            }
            S::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => self.draw_primitives_in(
                batch,
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            ),
            S::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => self.draw_indexed_primitives_in(
                batch,
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            ),
            S::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                tessellation_factor_buffer,
                tessellation_factor_buffer_offset,
                tessellation_factor_buffer_instance_stride,
            } => self.draw_patches(
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                tessellation_factor_buffer,
                tessellation_factor_buffer_offset,
                tessellation_factor_buffer_instance_stride,
            ),
            S::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
                tessellation_factor_buffer,
                tessellation_factor_buffer_offset,
                tessellation_factor_buffer_instance_stride,
            } => self.draw_indexed_patches(
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
                tessellation_factor_buffer,
                tessellation_factor_buffer_offset,
                tessellation_factor_buffer_instance_stride,
            ),
            S::DrawMeshThreadgroups {
                threadgroups_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            } => self.draw_mesh_threadgroups(
                threadgroups_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            ),
            S::DrawMeshThreads {
                threads_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            } => self.draw_mesh_threads(
                threads_per_grid,
                threads_per_object_threadgroup,
                threads_per_mesh_threadgroup,
            ),
            S::SetBarrier => self.barrier_in(batch, true),
            S::ClearBarrier => self.barrier_in(batch, false),
            S::Reset => self.reset(),
            S::SetDepthStencilState(state) => self.set_depth_stencil_state(state),
            S::SetDepthBias {
                depth_bias,
                slope_scale,
                clamp,
            } => self.set_depth_bias(depth_bias, slope_scale, clamp),
            S::SetDepthClipMode(mode) => self.set_depth_clip_mode(mode),
            S::SetCullMode(mode) => self.set_cull_mode(mode),
            S::SetFrontFacingWinding(winding) => self.set_front_facing_winding(winding),
            S::SetTriangleFillMode(mode) => self.set_triangle_fill_mode(mode),
        }
    }
}

impl ComputeAt<'_> {
    /// [`IndirectComputeCommand::set_each`] of `sets`, their writes one [`Batch`].
    fn set_each<'a>(
        &self,
        sets: impl IntoIterator<Item = IndirectComputeSet<'a>>,
    ) -> Result<(), (usize, Error)> {
        let mut batch = self.icb.batch(self.index);
        for (at, set) in sets.into_iter().enumerate() {
            self.set_one(&mut batch, set).map_err(|error| (at, error))?;
        }
        Ok(())
    }

    /// The setter `set` names: in `batch` where the setter writes a field, else the setter.
    #[inline(always)]
    fn set_one(&self, batch: &mut Batch<'_>, set: IndirectComputeSet<'_>) -> Result<(), Error> {
        use IndirectComputeSet as S;
        match set {
            S::SetComputePipelineState(state) => self.pipeline_in(batch, state),
            S::SetKernelBuffer {
                buffer,
                offset,
                index,
            } => self.kernel_buffer_in(batch, buffer, offset, index),
            S::SetKernelBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => self.kernel_buffer_with_stride_in(batch, buffer, offset, stride, index),
            S::ConcurrentDispatchThreadgroups {
                threadgroups,
                threads_per_threadgroup,
            } => self.dispatch_threadgroups_in(batch, threadgroups, threads_per_threadgroup),
            S::ConcurrentDispatchThreads {
                threads,
                threads_per_threadgroup,
            } => self.dispatch_threads_in(batch, threads, threads_per_threadgroup),
            S::SetBarrier => self.barrier_in(batch, true),
            S::ClearBarrier => self.barrier_in(batch, false),
            S::SetThreadgroupMemoryLength { length, index } => {
                self.set_threadgroup_memory_length(length, index)
            }
            S::SetStageInRegion(region) => self.set_stage_in_region(region),
            S::SetImageblockSize { width, height } => self.set_imageblock_size(width, height),
            S::Reset => self.reset(),
        }
    }
}
