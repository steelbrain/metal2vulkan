//! A render encoder's mesh draws: a mesh render pipeline's state
//! ([`Device::new_mesh_render_pipeline_state`](crate::Device::new_mesh_render_pipeline_state))
//! draws through `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`,
//! `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:` and
//! `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:
//! threadsPerMeshThreadgroup:`, with the object and mesh stages' tables the `setObject` and
//! `setMesh` setters fill.
//!
//! - **What Metal draws**, measured by probes on an M3 Max and an M4, the two alike: each mesh
//!   threadgroup's primitives in its threadgroup's order. `drawMeshThreads` without an object
//!   function launches the whole mesh threadgroups the threads fill, rounded down along each axis,
//!   and those threadgroups' threads are the grid the mesh function's builtins report
//!   (`threads_per_grid`, `thread_position_in_grid`): 10 by 3 threads in threadgroups of 4 by 2
//!   draw 2 by 1 threadgroups of an 8 by 2 grid. With an object function it launches the object
//!   threadgroups the threads fill, rounded up, the last along an axis running only the threads
//!   left (6 threads in threadgroups of 4: a threadgroup of 4 and one of 2, whose
//!   `threads_per_threadgroup` is 2). Each object threadgroup launches the mesh grid it sets, whose
//!   threadgroups read the payload it wrote. A grid with a zero axis, and a threadgroup of no
//!   thread, draw nothing; a threadgroup of more than 1024 threads, and indirect arguments at an
//!   offset that is not a multiple of 4, draw nothing too, though Metal's validation stops them
//!   ("threadsPerMeshThreadgroup (1025 x 1 x 1 = 1025) cannot exceed
//!   maxTotalThreadsPerMeshThreadgroup (1024)"), and they draw nothing here, refused only as a draw
//!   of the encoder's pipeline state is before it reads anything (no state, a tile state, a state
//!   that is not a mesh render pipeline's). A draw of primitives with a mesh render pipeline's
//!   state, and a mesh draw with another state, stop Metal with or without its validation
//!   (`DrawProblem::DrawKind`), but for a mesh draw of a grid with a zero axis, which plain Metal
//!   draws nothing of with a drawing state too, and so does the encoder (an M4;
//!   its validation stops it).
//! - **The draw** is the binder's (`DrawCall::MeshThreadgroups`, `MeshThreads`,
//!   `MeshThreadgroupsIndirect`): `vkCmdDrawMeshTasksEXT` of its first stage's threadgroups
//!   (the object function's, where the pipeline has one), each stage's threadgroup size a
//!   specialization of the draw's own `VkPipeline` (the draw's `DrawCommands::specialization`, as
//!   a kernel's threadgroup size is a dispatch's), through the
//!   render encoder's steps (`render.rs`). A mesh pipeline has no vertex input and no input
//!   assembly; its key takes a triangle list, under which Vulkan biases the depth of a mesh
//!   function's triangles and of none of its lines or points, as Metal does.
//! - **A partial object threadgroup** of `drawMeshThreads` runs as Metal runs it: the draw
//!   launches the whole object threadgroups the threads fill, rounded up, and the last along an
//!   axis answers the threads left as its `threads_per_threadgroup` (and its simdgroup and
//!   quadgroup counts and thread positions), its invocations past them returning before the
//!   object function's body, or, where the body holds a `threadgroup_barrier`, walking it to each
//!   barrier without running anything else of it (the binder's region flag,
//!   `RuntimeFact::PartialObjectThreadgroups`), on a device with `VK_EXT_mesh_shader` and emulated
//!   alike.
//! - **Refused**, as the binder refuses them: a threadgroup or a grid past the device's limits
//!   for its stage (`Unsupported{MeshShading}` naming the limit; an object grid only on a device
//!   that does not fold object grids), and an object
//!   function's arena with the lengths `setObjectThreadgroupMemoryLength:atIndex:` set
//!   ([`RenderCommandEncoder::set_object_threadgroup_memory_length`]) past the device's workgroup
//!   memory for the object stage (`Unsupported{ThreadgroupMemorySize}` naming the limit).
//! - **Metal's grids**, measured on the M3 Max (macOS 27.0.1), direct and indirect alike: without
//!   an object function a grid with an axis past 65,535 draws nothing, as does one of more than
//!   1,048,575 threadgroups (`bind::mesh_arguments`); with one, every object grid its GPU finishes
//!   within its watchdog's 5 seconds draws (2^29 threadgroups of an object function that does next
//!   to nothing in 4.98 seconds), and Metal's GPU discards a slower one. Every device draws an
//!   object grid of up to 2^29 threadgroups (`bind::METAL_OBJECT_GRID_TOTAL`), and a larger one
//!   draws nothing and fails the command buffer as Metal's GPU discards it
//!   (`CommandBufferError::GpuRecoveryVictim`): a direct draw from its encoder on, as Metal's
//!   (`DrawCommands::discards_command_buffer`), an indirect one as its arguments pass reports it.
//!   On a device with `VK_EXT_mesh_shader` an indirect draw reads the counts its arguments pass
//!   writes (`render/mesh_arguments.rs`): Metal's, or none where Metal draws nothing. Where the
//!   device folds object grids (device addresses, `shaderInt64`, `multiDrawIndirect`,
//!   `drawIndirectCount`: every committed profile with the extension), an object grid runs in
//!   pieces within the task stage's limits, its object function compiled folded. Elsewhere an
//!   indirect draw's grid past the device's task limits, which Metal draws, draws nothing and
//!   fails the command buffer (`CommandBufferError::MeshGridPastLimits`), and a direct one is
//!   refused as it is encoded.
//! - **Emulated** on a device without `VK_EXT_mesh_shader`, whose pipeline's object and mesh
//!   functions are compute stages and whose mesh function writes records
//!   (`GraphicsState::mesh_records`): each draw has a work buffer ([`MeshWork`], uploaded per draw)
//!   and its records, which the draw's set 0 binds for its dispatches and the draw alike; the draw
//!   is the binder's draw of the synthesized vertex stage, as a list of the mesh's topology. A
//!   direct draw without an object function knows its grid: its work buffer is complete, its
//!   records an exact region of the command buffer's arena, and it runs its mesh threadgroups as
//!   one dispatch ([`MeshSteps`]). A draw with an object function, or an indirect draw, runs the
//!   object threadgroups (or nothing), then the crate's mesh scan (`bind::mesh_scan_module`), which
//!   places the mesh threadgroups each object threadgroup launched (or Metal's indirect arguments'
//!   grid), reserves their records in the command buffer's records pool, and writes the mesh
//!   dispatch's and the draw's indirect arguments; between barriers, the mesh dispatch and the draw
//!   read them. The pool is a sixty-fourth of the device's largest device-local heap, within one
//!   binding and one allocation, its queue's block, which the command buffer leases from the
//!   device (`gpu_pools.rs`); a draw whose records the rest of it
//!   cannot hold draws nothing and fails the command buffer
//!   (`CommandBufferError::MeshRecordsOverflow`, naming the first such draw's number, its object
//!   threadgroups and its mesh threadgroups). The dispatches record before the pass, as a draw's
//!   copies do (`draw_source.rs`), or, where an earlier draw or tile dispatch of the encoder wrote
//!   what they read behind a memory barrier, or touched what they write, before the draw's own
//!   render pass instance, after theirs: between a barrier from everything before and one to
//!   everything after. An indirect command's record before its pass too, or, behind the command's
//!   barrier where an earlier draw's stages before its fragments touched what they use, before the
//!   run of the execution's that the barrier begins (`render/executions.rs`). Such draws announce
//!   `emulated-mesh-draws` once. An indirect draw of a pipeline with an object function first runs
//!   the object arguments pass (`bind::mesh_object_arguments_module`), which reads Metal's
//!   arguments, reserves the object threadgroups' slots in the records pool and writes the object
//!   dispatch, which then launches indirectly: a grid Metal's GPU discards fails the command
//!   buffer as Metal's (`CommandBufferError::GpuRecoveryVictim`), one past the compute limits
//!   fails it as one the device cannot launch
//!   (`CommandBufferError::MeshGridPastLimits`), and one whose slots the pool cannot hold as a
//!   records overflow (`CommandBufferError::MeshRecordsOverflow`, naming no mesh threadgroup).

use super::replay::Samplers;
use super::RenderCommandEncoder;
use super::RenderLowering;
use super::{check_arguments, draw_problem, Assembly, DrawBuffers, DrawProblem};
use super::{RenderCall, Table};
use crate::binding::{MeshRegions, PushWrite};
use crate::buffer::Buffer;
use crate::degradation::Degradation;
use crate::device::functions::Vk;
use crate::draw_source::DrawSource;
use crate::encoder::EncoderCore;
use crate::error::{internal, invalid, Error, Problem};
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use crate::pipeline::RawPipeline;
use crate::resource::ResourceKey;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use crate::tracker::{Barrier, Use};
use crate::upload::Upload;
use crate::values::Size;
use ash::vk;
use metal2vulkan::bind::{
    BufferRange, DrawCall, ExecutionStatus, IndirectRef, MeshDispatch, MeshRecordsRegion,
    MeshScanModule,
};
use std::sync::Arc;

/// Metal's most threads of an object or mesh threadgroup (`maxTotalThreadsPerObjectThreadgroup`,
/// `maxTotalThreadsPerMeshThreadgroup`, measured).
const METAL_THREADGROUP_THREADS: u64 = 1024;

/// The bytes of `MTLDispatchThreadgroupsIndirectArguments`, which are
/// `VkDrawMeshTasksIndirectCommandEXT`'s.
const INDIRECT_BYTES: u64 = 12;

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_object_buffer`].
    pub(crate) fn set_object_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::Buffer(Table::Object, buffer, offset, index)
        });
        self.object
            .set_buffer(&mut self.core, "set_object_buffer", buffer, offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_object_buffers`].
    pub(crate) fn set_object_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            RenderCall::Buffers(
                Table::Object,
                log.buffer_span(buffers),
                log.word_span(offsets),
                first,
            )
        });
        self.object.set_buffers(
            &mut self.core,
            "set_object_buffers",
            buffers,
            offsets,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_object_buffer_offset`].
    pub(crate) fn set_object_buffer_offset(
        &mut self,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::BufferOffset(Table::Object, offset, index));
        self.object
            .set_buffer_offset(&mut self.core, "set_object_buffer_offset", offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_object_bytes`].
    pub(crate) fn set_object_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.log(|log| RenderCall::Bytes(Table::Object, log.byte_span(bytes), index));
        self.object
            .set_bytes(&mut self.core, "set_object_bytes", bytes, None, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_object_texture`].
    pub(crate) fn set_object_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let texture = texture.map(|texture| log.hold_texture(texture));
            RenderCall::Texture(Table::Object, texture, index)
        });
        self.object
            .set_textures(&mut self.core, "set_object_texture", &[texture], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_object_textures`].
    pub(crate) fn set_object_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::Textures(Table::Object, log.texture_span(textures), first));
        self.object
            .set_textures(&mut self.core, "set_object_textures", textures, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_object_sampler_state`].
    pub(crate) fn set_object_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
            RenderCall::Sampler(Table::Object, sampler, index)
        });
        self.object.set_samplers(
            &mut self.core,
            "set_object_sampler_state",
            &[sampler],
            None,
            index,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_object_sampler_states`].
    pub(crate) fn set_object_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Samplers(Table::Object, Samplers::of(samplers, clamps, first)));
        self.object.set_samplers(
            &mut self.core,
            "set_object_sampler_states",
            samplers,
            clamps,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_object_threadgroup_memory_length`].
    pub(crate) fn set_object_threadgroup_memory_length(
        &mut self,
        length: u32,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::ObjectThreadgroupMemoryLength(length, index));
        const CALL: &str = "set_object_threadgroup_memory_length";
        crate::argument_table::check_index(CALL, index, crate::compute::THREADGROUP_SLOTS)?;
        self.object_threadgroup_lengths.insert(index, length);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_buffer`].
    pub(crate) fn set_mesh_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::Buffer(Table::Mesh, buffer, offset, index)
        });
        self.mesh
            .set_buffer(&mut self.core, "set_mesh_buffer", buffer, offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_buffers`].
    pub(crate) fn set_mesh_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            RenderCall::Buffers(
                Table::Mesh,
                log.buffer_span(buffers),
                log.word_span(offsets),
                first,
            )
        });
        self.mesh
            .set_buffers(&mut self.core, "set_mesh_buffers", buffers, offsets, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_buffer_offset`].
    pub(crate) fn set_mesh_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        self.log(|_| RenderCall::BufferOffset(Table::Mesh, offset, index));
        self.mesh
            .set_buffer_offset(&mut self.core, "set_mesh_buffer_offset", offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_bytes`].
    pub(crate) fn set_mesh_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.log(|log| RenderCall::Bytes(Table::Mesh, log.byte_span(bytes), index));
        self.mesh
            .set_bytes(&mut self.core, "set_mesh_bytes", bytes, None, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_texture`].
    pub(crate) fn set_mesh_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let texture = texture.map(|texture| log.hold_texture(texture));
            RenderCall::Texture(Table::Mesh, texture, index)
        });
        self.mesh
            .set_textures(&mut self.core, "set_mesh_texture", &[texture], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_textures`].
    pub(crate) fn set_mesh_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::Textures(Table::Mesh, log.texture_span(textures), first));
        self.mesh
            .set_textures(&mut self.core, "set_mesh_textures", textures, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_sampler_state`].
    pub(crate) fn set_mesh_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
            RenderCall::Sampler(Table::Mesh, sampler, index)
        });
        self.mesh.set_samplers(
            &mut self.core,
            "set_mesh_sampler_state",
            &[sampler],
            None,
            index,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_mesh_sampler_states`].
    pub(crate) fn set_mesh_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Samplers(Table::Mesh, Samplers::of(samplers, clamps, first)));
        self.mesh.set_samplers(
            &mut self.core,
            "set_mesh_sampler_states",
            samplers,
            clamps,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::draw_mesh_threadgroups`].
    pub(crate) fn draw_mesh_threadgroups(
        &mut self,
        groups: Size,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::MeshThreadgroups(Box::new((
                groups,
                threads_per_object_group,
                threads_per_mesh_group,
            )))
        });
        const CALL: &str = "draw_mesh_threadgroups";
        let grid = MeshGrid::Threadgroups(axes(CALL, groups)?);
        let object = axes(CALL, threads_per_object_group)?;
        let mesh = axes(CALL, threads_per_mesh_group)?;
        self.draw_mesh(CALL, grid, object, mesh)
    }

    /// The lowering of [`RenderCommandEncoder::draw_mesh_threads`].
    pub(crate) fn draw_mesh_threads(
        &mut self,
        threads: Size,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::MeshThreads(Box::new((
                threads,
                threads_per_object_group,
                threads_per_mesh_group,
            )))
        });
        const CALL: &str = "draw_mesh_threads";
        let grid = MeshGrid::Threads(axes(CALL, threads)?);
        let object = axes(CALL, threads_per_object_group)?;
        let mesh = axes(CALL, threads_per_mesh_group)?;
        self.draw_mesh(CALL, grid, object, mesh)
    }

    /// The direct mesh draw of `grid` for `call`, its object and mesh threadgroups of `object`
    /// and `mesh` threads: the encoder's `draw_mesh_threadgroups` and `draw_mesh_threads`, and
    /// an indirect command buffer's mesh command (`render/executions.rs`).
    pub(super) fn draw_mesh(
        &mut self,
        call: &'static str,
        grid: MeshGrid,
        object: [u32; 3],
        mesh: [u32; 3],
    ) -> Result<(), Error> {
        if !self.mesh_threadgroups_run(object, mesh) {
            return self.mesh_draws_nothing(call);
        }
        // A grid with an axis of 0 launches no threadgroup: an empty draw (`RenderCommandEncoder::
        // draw`, which checks it only as plain Metal checks one).
        let empty = match grid {
            MeshGrid::Threadgroups(axes) | MeshGrid::Threads(axes) => axes.contains(&0),
        };
        let draw = match grid {
            MeshGrid::Threadgroups(groups) => DrawCall::MeshThreadgroups {
                groups,
                threads_per_object_group: object,
                threads_per_mesh_group: mesh,
            },
            MeshGrid::Threads(threads) => DrawCall::MeshThreads {
                threads,
                threads_per_object_group: object,
                threads_per_mesh_group: mesh,
            },
        };
        self.draw(call, Assembly::Meshes, draw, DrawBuffers::default(), empty)
    }

    /// The lowering of [`RenderCommandEncoder::draw_mesh_threadgroups_indirect`].
    pub(crate) fn draw_mesh_threadgroups_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::MeshThreadgroupsIndirect(Box::new((
                indirect_buffer.clone(),
                indirect_buffer_offset,
                threads_per_object_group,
                threads_per_mesh_group,
            )))
        });
        const CALL: &str = "draw_mesh_threadgroups_indirect";
        self.core.check_device(CALL, indirect_buffer.device())?;
        // Arguments at an offset that is not a multiple of 4 draw nothing (the module docs).
        let aligned = indirect_buffer_offset.is_multiple_of(4);
        check_arguments(
            CALL,
            indirect_buffer.length(),
            indirect_buffer_offset,
            INDIRECT_BYTES,
        )?;
        let (object, mesh) = (
            axes(CALL, threads_per_object_group)?,
            axes(CALL, threads_per_mesh_group)?,
        );
        if !aligned || !self.mesh_threadgroups_run(object, mesh) {
            return self.mesh_draws_nothing(CALL);
        }
        let indirect = DrawSource::new(
            indirect_buffer,
            indirect_buffer_offset,
            INDIRECT_BYTES,
            4,
            false,
            false,
        )?;
        let draw = DrawCall::MeshThreadgroupsIndirect {
            indirect: IndirectRef::new(0, indirect.at),
            threads_per_object_group: object,
            threads_per_mesh_group: mesh,
        };
        let buffers = DrawBuffers {
            indirect: Some(indirect),
            ..DrawBuffers::default()
        };
        self.draw(CALL, Assembly::Meshes, draw, buffers, false)
    }

    /// Whether a mesh draw of `object` and `mesh` threads in its object and mesh threadgroups
    /// runs: not where it draws nothing, though Metal's validation stops it, more than
    /// [`METAL_THREADGROUP_THREADS`] in the mesh threadgroup, or in the object threadgroup of a
    /// pipeline with an object function (a probe).
    fn mesh_threadgroups_run(&self, object: [u32; 3], mesh: [u32; 3]) -> bool {
        let has_object = self.pipeline.as_ref().is_some_and(|state| {
            matches!(
                state.kind(),
                super::RenderStateKind::Graphics(graphics)
                    if graphics.descriptor().stage(metal2vulkan::pipeline::Stage::Object).is_some()
            )
        });
        let checked = [(false, mesh)]
            .into_iter()
            .chain(has_object.then_some((true, object)));
        let over = checked.into_iter().any(|(_, size)| {
            let threads: u64 = size.iter().map(|&axis| u64::from(axis)).product();
            threads > METAL_THREADGROUP_THREADS
        });
        !over
    }

    /// A mesh draw that draws nothing (the module docs): refused as a mesh draw of the encoder's
    /// pipeline state is before it reads anything, without a state, with a tile state, or with
    /// a state that is not a mesh render pipeline's (`DrawProblem::DrawKind`; each stops Metal).
    fn mesh_draws_nothing(&self, call: &'static str) -> Result<(), Error> {
        match self
            .pipeline
            .as_ref()
            .map(crate::pipeline::RenderPipelineState::kind)
        {
            None => Err(invalid(call, Problem::NoPipelineState)),
            Some(super::RenderStateKind::Tile(_)) => {
                Err(draw_problem(call, DrawProblem::TileState))
            }
            Some(super::RenderStateKind::Graphics(state))
                if state.descriptor().vertex().is_some() =>
            {
                Err(draw_problem(call, DrawProblem::DrawKind))
            }
            Some(super::RenderStateKind::Graphics(_)) => Ok(()),
        }
    }
}

impl RenderCommandEncoder {
    /// `setObjectBuffer:offset:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_buffer`](Self::set_vertex_buffer) is.
    pub fn set_object_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_buffer(buffer, offset, index)),
            false => self.defer(|log| {
                let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                RenderCall::Buffer(Table::Object, buffer, offset, index)
            }),
        }
    }

    /// `setObjectBuffers:offsets:withRange:`. Refused as
    /// [`set_vertex_buffers`](Self::set_vertex_buffers) is.
    pub fn set_object_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_buffers(buffers, offsets, first)),
            false => self.defer(|log| {
                RenderCall::Buffers(
                    Table::Object,
                    log.buffer_span(buffers),
                    log.word_span(offsets),
                    first,
                )
            }),
        }
    }

    /// `setObjectBufferOffset:atIndex:`. Refused as
    /// [`set_vertex_buffer_offset`](Self::set_vertex_buffer_offset) is.
    pub fn set_object_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_buffer_offset(offset, index)),
            false => self.defer(|_| RenderCall::BufferOffset(Table::Object, offset, index)),
        }
    }

    /// `setObjectBytes:length:atIndex:`: as [`set_vertex_bytes`](Self::set_vertex_bytes), in
    /// the object stage's table. Refused as that is.
    pub fn set_object_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_bytes(bytes, index)),
            false => {
                self.defer(|log| RenderCall::Bytes(Table::Object, log.byte_span(bytes), index))
            }
        }
    }

    /// `setObjectTexture:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_texture`](Self::set_vertex_texture) is.
    pub fn set_object_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_texture(texture, index)),
            false => self.defer(|log| {
                let texture = texture.map(|texture| log.hold_texture(texture));
                RenderCall::Texture(Table::Object, texture, index)
            }),
        }
    }

    /// `setObjectTextures:withRange:`. Refused as
    /// [`set_vertex_textures`](Self::set_vertex_textures) is.
    pub fn set_object_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_textures(textures, first)),
            false => self.defer(|log| {
                RenderCall::Textures(Table::Object, log.texture_span(textures), first)
            }),
        }
    }

    /// `setObjectSamplerState:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_sampler_state`](Self::set_vertex_sampler_state) is.
    pub fn set_object_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_object_sampler_state(sampler, index)),
            false => self.defer(|log| {
                let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                RenderCall::Sampler(Table::Object, sampler, index)
            }),
        }
    }

    /// `setObjectSamplerStates:withRange:`, or with `clamps`
    /// `setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. Refused as
    /// [`set_vertex_sampler_states`](Self::set_vertex_sampler_states) is.
    pub fn set_object_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_object_sampler_states(samplers, clamps, first))
            }
            false => self.defer(|_| {
                RenderCall::Samplers(Table::Object, Samplers::of(samplers, clamps, first))
            }),
        }
    }

    /// `setObjectThreadgroupMemoryLength:atIndex:`: the length of the object function's
    /// `[[threadgroup(index)]]` argument in the encoder's later mesh draws, whichever pipeline
    /// state they use; until it is set an argument's length is 0. A length that is not a multiple
    /// of 16 bytes is rounded up to 16 at the draw, as a kernel's
    /// (`ComputeCommandEncoder::set_threadgroup_memory_length`) is. Refused: an index past the
    /// table (`Invalid`). The object function's arena with these lengths must fit the device's
    /// workgroup memory for the object stage, checked at the draw
    /// (`Unsupported{ThreadgroupMemorySize}` naming the limit). An indirect command's mesh draw
    /// takes none of these, as Metal's takes the command's own
    /// (`IndirectRenderCommand::set_object_threadgroup_memory_length`).
    pub fn set_object_threadgroup_memory_length(
        &mut self,
        length: u32,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self
                .at_call(|lowering| lowering.set_object_threadgroup_memory_length(length, index)),
            false => self.defer(|_| RenderCall::ObjectThreadgroupMemoryLength(length, index)),
        }
    }

    /// `setMeshBuffer:offset:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_buffer`](Self::set_vertex_buffer) is.
    pub fn set_mesh_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_buffer(buffer, offset, index)),
            false => self.defer(|log| {
                let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                RenderCall::Buffer(Table::Mesh, buffer, offset, index)
            }),
        }
    }

    /// `setMeshBuffers:offsets:withRange:`. Refused as
    /// [`set_vertex_buffers`](Self::set_vertex_buffers) is.
    pub fn set_mesh_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_buffers(buffers, offsets, first)),
            false => self.defer(|log| {
                RenderCall::Buffers(
                    Table::Mesh,
                    log.buffer_span(buffers),
                    log.word_span(offsets),
                    first,
                )
            }),
        }
    }

    /// `setMeshBufferOffset:atIndex:`. Refused as
    /// [`set_vertex_buffer_offset`](Self::set_vertex_buffer_offset) is.
    pub fn set_mesh_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_buffer_offset(offset, index)),
            false => self.defer(|_| RenderCall::BufferOffset(Table::Mesh, offset, index)),
        }
    }

    /// `setMeshBytes:length:atIndex:`: as [`set_vertex_bytes`](Self::set_vertex_bytes), in the
    /// mesh stage's table. Refused as that is.
    pub fn set_mesh_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_bytes(bytes, index)),
            false => self.defer(|log| RenderCall::Bytes(Table::Mesh, log.byte_span(bytes), index)),
        }
    }

    /// `setMeshTexture:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_texture`](Self::set_vertex_texture) is.
    pub fn set_mesh_texture(&mut self, texture: Option<&Texture>, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_texture(texture, index)),
            false => self.defer(|log| {
                let texture = texture.map(|texture| log.hold_texture(texture));
                RenderCall::Texture(Table::Mesh, texture, index)
            }),
        }
    }

    /// `setMeshTextures:withRange:`. Refused as
    /// [`set_vertex_textures`](Self::set_vertex_textures) is.
    pub fn set_mesh_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_textures(textures, first)),
            false => self
                .defer(|log| RenderCall::Textures(Table::Mesh, log.texture_span(textures), first)),
        }
    }

    /// `setMeshSamplerState:atIndex:` (`None` unbinds the index). Refused as
    /// [`set_vertex_sampler_state`](Self::set_vertex_sampler_state) is.
    pub fn set_mesh_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_mesh_sampler_state(sampler, index)),
            false => self.defer(|log| {
                let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                RenderCall::Sampler(Table::Mesh, sampler, index)
            }),
        }
    }

    /// `setMeshSamplerStates:withRange:`, or with `clamps`
    /// `setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. Refused as
    /// [`set_vertex_sampler_states`](Self::set_vertex_sampler_states) is.
    pub fn set_mesh_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_mesh_sampler_states(samplers, clamps, first))
            }
            false => self.defer(|_| {
                RenderCall::Samplers(Table::Mesh, Samplers::of(samplers, clamps, first))
            }),
        }
    }

    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: `groups`
    /// mesh threadgroups of `threads_per_mesh_group` threads (the module docs). Refused as the
    /// module docs say, and for a size past 32 bits on an axis (`Invalid(SizeOutOfRange)`).
    pub fn draw_mesh_threadgroups(
        &mut self,
        groups: Size,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_mesh_threadgroups(
                    groups,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )
            }),
            false => self.defer(|_| {
                RenderCall::MeshThreadgroups(Box::new((
                    groups,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )))
            }),
        }
    }

    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: the whole mesh
    /// threadgroups of `threads_per_mesh_group` threads that `threads` fill (the module docs).
    /// Refused as [`draw_mesh_threadgroups`](Self::draw_mesh_threadgroups) is.
    pub fn draw_mesh_threads(
        &mut self,
        threads: Size,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_mesh_threads(
                    threads,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )
            }),
            false => self.defer(|_| {
                RenderCall::MeshThreads(Box::new((
                    threads,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )))
            }),
        }
    }

    /// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:
    /// threadsPerMeshThreadgroup:`: Metal's `MTLDispatchThreadgroupsIndirectArguments` at
    /// `indirect_buffer_offset`, the threadgroups' counts. Refused as
    /// [`draw_mesh_threadgroups`](Self::draw_mesh_threadgroups) is, as an indirect draw's
    /// arguments are (`render.rs`'s module docs), and for an indirect buffer of another device.
    /// On a device with `VK_EXT_mesh_shader` the draw's counts are an arguments pass's: Metal's
    /// where Metal draws them, an object grid in pieces where the device folds object grids; an
    /// object grid Metal's GPU discards fails the command buffer as Metal's does
    /// (`CommandBufferError::GpuRecoveryVictim`), and one past a device's task limits where it
    /// does not fold them (`CommandBufferError::MeshGridPastLimits`, the module docs). On one
    /// without it the emulation reads the counts on the GPU: a draw with an object function
    /// through its object arguments pass (the module docs).
    pub fn draw_mesh_threadgroups_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
        threads_per_object_group: Size,
        threads_per_mesh_group: Size,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_mesh_threadgroups_indirect(
                    indirect_buffer,
                    indirect_buffer_offset,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )
            }),
            false => self.defer(|_| {
                RenderCall::MeshThreadgroupsIndirect(Box::new((
                    indirect_buffer.clone(),
                    indirect_buffer_offset,
                    threads_per_object_group,
                    threads_per_mesh_group,
                )))
            }),
        }
    }
}

/// A direct mesh draw's grid: of threadgroups (`drawMeshThreadgroups`) or of threads
/// (`drawMeshThreads`).
#[derive(Clone, Copy, Debug)]
pub(super) enum MeshGrid {
    Threadgroups([u32; 3]),
    Threads([u32; 3]),
}

/// `size`'s axes, refused past 32 bits (`Invalid(SizeOutOfRange)`; Metal's answer is not
/// measured).
fn axes(call: &'static str, size: Size) -> Result<[u32; 3], Error> {
    size.axes()
        .ok_or_else(|| invalid(call, Problem::SizeOutOfRange))
}

/// An emulated mesh draw's work buffer and records (`DrawCommands::mesh_dispatch`), made for the
/// draw: the work buffer a region of the command buffer's arena holding the binder's words with
/// the draw's number and, for records in the command buffer's pool, the pool's words.
pub(crate) struct MeshWork {
    pub(crate) work: Upload,
    /// The records' buffer, offset and bytes.
    records: (vk::Buffer, u64, u64),
}

impl MeshWork {
    /// The work buffer and records of `dispatch`, in `core`'s command buffer. Refused as uploads
    /// and buffers are.
    pub(crate) fn new(core: &EncoderCore, dispatch: &MeshDispatch) -> Result<MeshWork, Error> {
        let device = core.device().clone();
        let queue = core.command_buffer().command_queue().id();
        let mut state = core.command_buffer().state();
        let mut words = dispatch.work.clone();
        let mut set = |at: usize, value: u32| match words.get_mut(at) {
            Some(word) => {
                *word = value;
                Ok(())
            }
            None => Err(internal("a mesh work buffer without its header")),
        };
        set(MeshDispatch::DRAW_ID, state.next_mesh_draw())?;
        let records = match dispatch.records {
            MeshRecordsRegion::Exact(bytes) => {
                let region = state.uploads.upload(&device, &[], bytes)?;
                (region.raw(), region.offset(), region.len())
            }
            MeshRecordsRegion::Pool => {
                let (raw, bytes) = state.mesh_pool(&device, queue)?;
                let words = u32::try_from(bytes / 4)
                    .map_err(|_| internal("a mesh records pool past a u32's words"))?;
                set(MeshDispatch::POOL_WORDS, words)?;
                (raw, 0, bytes)
            }
            _ => return Err(internal("mesh records of another kind")),
        };
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let work = state.uploads.upload(&device, &bytes, dispatch.work_bytes)?;
        Ok(MeshWork { work, records })
    }

    /// The work buffer's and the records' descriptors, each bound whole.
    pub(crate) fn regions(&self) -> MeshRegions {
        let (buffer, offset, range) = self.records;
        MeshRegions {
            work: vk::DescriptorBufferInfo::default()
                .buffer(self.work.raw())
                .offset(self.work.offset())
                .range(self.work.len()),
            records: vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(offset)
                .range(range),
        }
    }

    /// The work buffer's `VkBuffer` and the byte of it at `offset` into the work buffer.
    pub(crate) fn at(&self, offset: u64) -> (vk::Buffer, u64) {
        (self.work.raw(), self.work.offset() + offset)
    }
}

/// An emulated mesh draw's scan (`bind::MeshScanModule`), or its object arguments pass
/// (`bind::mesh_object_arguments_module`): the device's kernel and its set over the draw's work
/// buffer, the command buffer's status words, Metal's arguments, and the records.
pub(crate) struct Scan {
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
}

impl Scan {
    /// The scan of `work` (`OwnedModule::MeshScan`), or its object arguments pass
    /// (`OwnedModule::MeshObjectArguments`), as `which`, in `core`'s command buffer (which holds
    /// its set until it completes), reading Metal's arguments at `arguments` (the `VkBuffer` and
    /// the binder's range in it) for an indirect draw. Refused as kernels, uploads and sets are.
    pub(crate) fn new(
        core: &mut EncoderCore,
        which: OwnedModule,
        work: &MeshWork,
        arguments: Option<(vk::Buffer, BufferRange)>,
    ) -> Result<Scan, Error> {
        let device = core.device().clone();
        let kernel = device.owned_kernel(which)?;
        let status = core.command_buffer().state().status_words(&device)?;
        let status = (status.raw(), status.offset(), ExecutionStatus::BYTES as u64);
        let mut set = [(vk::Buffer::null(), 0, 0); 4];
        set[MeshScanModule::WORK_BINDING as usize] =
            (work.work.raw(), work.work.offset(), work.work.len());
        set[MeshScanModule::STATUS_BINDING as usize] = status;
        // Without arguments nothing reads the binding, which takes the status words.
        set[MeshScanModule::ARGUMENTS_BINDING as usize] = match arguments {
            Some((raw, range)) => (raw, range.offset, range.range),
            None => status,
        };
        set[MeshScanModule::RECORDS_BINDING as usize] = work.records;
        let owned = OwnedSets::write(&device, &kernel, &[set.to_vec()])?;
        let sets = owned.sets.clone();
        core.retain(owned);
        Ok(Scan { kernel, sets })
    }
}

/// How an emulated mesh draw launches its object or mesh threadgroups.
pub(crate) enum Launch {
    /// `vkCmdDispatch` of these counts.
    Direct([u32; 3]),
    /// `vkCmdDispatchIndirect` from this `VkBuffer` and byte: the object arguments pass's, or
    /// the scan's.
    Indirect(vk::Buffer, u64),
}

impl Launch {
    /// Record the dispatch.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass with a compute pipeline bound, and an
    /// indirect launch's arguments are written and visible to `DRAW_INDIRECT` reads.
    unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states.
        unsafe {
            match *self {
                Launch::Direct([x, y, z]) => raw.cmd_dispatch(command_buffer, x, y, z),
                Launch::Indirect(buffer, offset) => {
                    raw.cmd_dispatch_indirect(command_buffer, buffer, offset)
                }
            }
        }
    }
}

/// An emulated mesh draw's compute work (`DrawCommands::mesh_dispatch`), ready to record before
/// the draw's render pass instance (the module docs): its object dispatch, its scan, and its mesh
/// dispatch, each with the draw's sets and compute-stage push writes, between barriers.
pub(crate) struct MeshSteps {
    pub(crate) layout: vk::PipelineLayout,
    /// Set 0, the draw's, where the layout has one.
    pub(crate) sets: Vec<vk::DescriptorSet>,
    /// The dynamic offsets the draw binds set 0 with (`binding::dynamic_writes`).
    pub(crate) offsets: Vec<u32>,
    /// Set 1, the command buffer's heap set of the variant, where it has one.
    pub(crate) heap: Option<vk::DescriptorSet>,
    /// The draw's push writes of the compute stage.
    pub(crate) push: Vec<PushWrite>,
    /// The object arguments pass, of an indirect draw with an object function.
    pub(crate) arguments: Option<Scan>,
    /// The object function's pipeline and its launch, where the pipeline has one: indirect from
    /// the work buffer where the object arguments pass writes it.
    pub(crate) object: Option<(Arc<RawPipeline>, Launch)>,
    /// The scan, of a draw whose mesh threadgroups the GPU decides.
    pub(crate) scan: Option<Scan>,
    /// The mesh function's pipeline.
    pub(crate) pipeline: Arc<RawPipeline>,
    pub(crate) launch: Launch,
    /// The object and mesh functions' uses, and the scan's read of Metal's arguments: the
    /// draw's compute-stage uses.
    pub(crate) uses: Vec<Use<ResourceKey>>,
}

impl MeshSteps {
    /// The steps' uses.
    pub(crate) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Hold what the steps record until `core`'s command buffer completes, and announce the
    /// emulation once. Call it once the draw records.
    pub(crate) fn retain(&mut self, core: &mut EncoderCore) {
        core.retain(Arc::clone(&self.pipeline));
        if let Some((pipeline, _)) = &self.object {
            core.retain(Arc::clone(pipeline));
        }
        core.device().degrade(Degradation::EmulatedMeshDraws, || {
            (
                "a mesh draw".to_owned(),
                "the device does not enable VK_EXT_mesh_shader, so the draw's object and mesh \
                 functions run as compute dispatches before its render pass instance"
                    .to_owned(),
            )
        });
    }

    /// Record the steps into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
    pub(crate) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatches(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the steps into `command_buffer` between a barrier from everything before and one
    /// to everything after: before a render pass instance of the encoder's pass, after the
    /// instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(crate) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; barriers need nothing more.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.dispatches(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The object arguments pass, the object dispatch, the scan and the mesh dispatch, each after
    /// a barrier from the one before.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it; the
    /// pipelines, the sets and what they name are live until it completes (the draw retains
    /// them, and the command buffer the scan's set).
    unsafe fn dispatches(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let point = vk::PipelineBindPoint::COMPUTE;
        // A pipeline of the draw's layout, its sets and its push writes.
        let bind = |pipeline: vk::Pipeline| {
            // SAFETY: as the caller states.
            unsafe {
                raw.cmd_bind_pipeline(command_buffer, point, pipeline);
                if !self.sets.is_empty() {
                    raw.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        self.layout,
                        0,
                        &self.sets,
                        &self.offsets,
                    );
                }
                if let Some(heap) = self.heap {
                    raw.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        self.layout,
                        1,
                        &[heap],
                        &[],
                    );
                }
                for (stages, offset, bytes) in &self.push {
                    raw.cmd_push_constants(command_buffer, self.layout, *stages, *offset, bytes);
                }
            }
        };
        // One workgroup of an owned kernel, then a barrier.
        let owned = |scan: &Scan| {
            // SAFETY: as the caller states; the kernel is the device's.
            unsafe {
                raw.cmd_bind_pipeline(command_buffer, point, scan.kernel.pipeline);
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    point,
                    scan.kernel.layout,
                    0,
                    &scan.sets,
                    &[],
                );
                raw.cmd_dispatch(command_buffer, 1, 1, 1);
                Barrier::everything().record(raw, command_buffer);
            }
        };
        if let Some(arguments) = &self.arguments {
            owned(arguments);
        }
        if let Some((pipeline, launch)) = &self.object {
            bind(pipeline.raw);
            // SAFETY: as the caller states; an indirect launch's arguments lie in the work
            // buffer, which the object arguments pass wrote before the barrier.
            unsafe {
                launch.record(raw, command_buffer);
                Barrier::everything().record(raw, command_buffer);
            }
        }
        if let Some(scan) = &self.scan {
            owned(scan);
        }
        bind(self.pipeline.raw);
        // SAFETY: as the caller states; an indirect launch's arguments lie in the work buffer,
        // which the scan wrote before the barrier.
        unsafe { self.launch.record(raw, command_buffer) }
    }
}
