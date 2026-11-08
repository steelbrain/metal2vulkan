//! A render encoder's patch draws on a device without `tessellationShader`: a tessellation
//! pipeline whose post-tessellation vertex function the compile made a plain vertex function
//! (`GraphicsState::emulated_tessellation`) draws each patch draw as the binder's
//! `DrawCommands::tessellation_dispatch` states it.
//!
//! - **The work.** Each draw has a work buffer ([`TessellationWork`], uploaded per draw: the
//!   binder's header, the draw's number, for records in the pool the pool's words, and one slot
//!   per chunk of the draw) and its records, a region of the command buffer's arena for a direct
//!   draw, whose `patchCount · instanceCount` the host knows, or the command buffer's records
//!   pool for an indirect one (the pool emulated mesh draws reserve from, `render/mesh.rs`).
//! - **The scan** ([`TessellationSteps`]): for an indirect draw, a copy of Metal's 16 bytes of
//!   arguments into the work buffer's first four words (`vkCmdCopyBuffer`), then one workgroup
//!   of the device's tessellation scan for the pipeline's tessellation state
//!   (`bind::tessellation_scan_module`), which processes each patch's factors and writes the
//!   records and every chunk's `VkDrawIndirectCommand`. It binds four storage buffers, Vulkan's
//!   least `maxPerStageDescriptorStorageBuffers`, which is why the arguments are copied rather
//!   than bound. It records before the draw's render pass instance, as an emulated mesh draw's
//!   dispatches do (`render/mesh.rs`), the copy and the dispatch each between a barrier from
//!   everything before and one to everything after (`Barrier::everything`: from `ALL_COMMANDS` and
//!   `MEMORY_WRITE` to `ALL_COMMANDS` and `MEMORY_READ | MEMORY_WRITE`). The Vulkan specification's
//!   synchronization chapter makes such a barrier's first scope every command before it and its
//!   second every command after it ("the first synchronization scope includes all commands that
//!   occur earlier in submission order", "the second synchronization scope includes all commands
//!   that occur later in submission order"), and `MEMORY_READ` includes `TRANSFER_READ`,
//!   `INDIRECT_COMMAND_READ` and `SHADER_STORAGE_READ` ("`VK_ACCESS_2_MEMORY_READ_BIT` specifies
//!   all read accesses"): the arguments' writes are available to the copy, the copy's and the
//!   factors' writes to the scan, and its writes to the draws' indirect commands and vertex
//!   shaders.
//! - **The draws.** `K` chunk draws (`TessellationDispatch::chunks`), each a `vkCmdDrawIndirect`
//!   of one command from its slot of the work buffer after the push write that names its chunk,
//!   in order, so a draw of more than `2^30` triangles stays within a `u32` of vertices; a chunk
//!   past the draw's triangles draws no vertex.
//! - **Refused or failed.** An indirect draw whose records the rest of the pool cannot hold draws
//!   nothing and fails the command buffer (`CommandBufferError::TessellationRecordsOverflow`,
//!   naming the first records fault's draw number and records). A direct draw whose records pass
//!   `maxStorageBufferRange` is the binder's refusal (`Unsupported{Tessellation}`).
//!
//! Such draws announce `emulated-tessellation-draws` once.

use crate::binding::{MeshRegions, PushWrite};
use crate::degradation::Degradation;
use crate::device::functions::Vk;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use crate::resource::ResourceKey;
use crate::tracker::{Barrier, Use};
use crate::upload::Upload;
use ash::vk;
use metal2vulkan::bind::{
    ExecutionStatus, TessellationDispatch, TessellationRecords, TessellationScanModule,
};

/// An emulated tessellation draw's work buffer and records (`DrawCommands::tessellation_dispatch`),
/// made for the draw.
pub(crate) struct TessellationWork {
    pub(crate) work: Upload,
    /// The records' buffer, offset and bytes.
    records: (vk::Buffer, u64, u64),
    /// `K`, the draw's chunks.
    chunks: u32,
}

impl TessellationWork {
    /// The work buffer and records of `dispatch`, in `core`'s command buffer. Refused as uploads
    /// and pools are.
    pub(crate) fn new(
        core: &EncoderCore,
        dispatch: &TessellationDispatch,
    ) -> Result<TessellationWork, Error> {
        let device = core.device().clone();
        let queue = core.command_buffer().queue_place().id();
        let mut state = core.command_buffer().state();
        let (records, pool) = match dispatch.records {
            TessellationRecords::Exact { .. } => {
                let region = state
                    .uploads
                    .upload(&device, &[], dispatch.records.bytes())?;
                ((region.raw(), region.offset(), region.len()), None)
            }
            TessellationRecords::Pool => {
                let (raw, bytes) = state.mesh_pool(&device, queue)?;
                ((raw, 0, bytes), Some(bytes))
            }
            _ => return Err(internal("tessellation records of another kind")),
        };
        let chunks = dispatch.chunks(pool.unwrap_or(0));
        let mut words = dispatch.work(chunks);
        let mut set = |at: usize, value: u32| match words.get_mut(at) {
            Some(word) => {
                *word = value;
                Ok(())
            }
            None => Err(internal("a tessellation work buffer without its header")),
        };
        set(TessellationDispatch::DRAW_ID, state.next_mesh_draw())?;
        if let Some(bytes) = pool {
            let pool_words = u32::try_from(bytes / 4)
                .map_err(|_| internal("a records pool past a u32's words"))?;
            set(TessellationDispatch::POOL_WORDS, pool_words)?;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let work = state.uploads.upload(&device, &bytes, bytes.len() as u64)?;
        Ok(TessellationWork {
            work,
            records,
            chunks,
        })
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

    /// The chunk draws of `dispatch`, in order: each one's push writes, after `push` (the draw's
    /// own), and its `vkCmdDrawIndirect` from the work buffer.
    pub(crate) fn draws(
        &self,
        dispatch: &TessellationDispatch,
        push: &[PushWrite],
    ) -> Result<Vec<(Vec<PushWrite>, vk::Buffer, metal2vulkan::vk::DrawCommand)>, Error> {
        (0..self.chunks)
            .map(|k| {
                let (writes, command) = dispatch.draw(k);
                let mut all = push.to_vec();
                all.extend(crate::binding::push_writes(writes, &[])?);
                let metal2vulkan::vk::DrawCommand::DrawIndirect {
                    offset,
                    draw_count,
                    stride,
                } = command
                else {
                    return Err(internal("a chunk draw that is not indirect"));
                };
                let command = metal2vulkan::vk::DrawCommand::DrawIndirect {
                    offset: self.work.offset() + offset,
                    draw_count,
                    stride,
                };
                Ok((all, self.work.raw(), command))
            })
            .collect()
    }
}

/// An emulated tessellation draw's scan (the module docs), ready to record before the draw's
/// render pass instance: for an indirect draw the copy of Metal's arguments into the work buffer,
/// and the device's kernel for the pipeline's tessellation state and its set over the draw's work
/// buffer and records, the command buffer's status words and the factors.
pub(crate) struct TessellationSteps {
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    /// For an indirect draw, the copy of its arguments into the work buffer's first 16 bytes.
    copy: Option<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
    /// The scan's reads of the factors and the arguments: the draw's compute-stage uses.
    uses: Vec<Use<ResourceKey>>,
}

impl TessellationSteps {
    /// The scan of `dispatch` over `work`, in `core`'s command buffer (which holds its set until
    /// it completes), reading the factors at `factors` (the `VkBuffer`, offset and range; the
    /// zero buffer where `None`) and, for an indirect draw, Metal's arguments at `arguments` (the
    /// `VkBuffer` and the binder's offset in it, whose 16 bytes the encoder's checks held inside
    /// the Metal buffer). Refused as kernels, uploads and sets are.
    pub(crate) fn new(
        core: &mut EncoderCore,
        dispatch: &TessellationDispatch,
        work: &TessellationWork,
        (factors, arguments): (Option<(vk::Buffer, u64, u64)>, Option<(vk::Buffer, u64)>),
        uses: Vec<Use<ResourceKey>>,
    ) -> Result<TessellationSteps, Error> {
        let device = core.device().clone();
        let kernel = device.owned_kernel(OwnedModule::TessellationScan(dispatch.kernel))?;
        let (status, zeros) = {
            let mut state = core.command_buffer().state();
            let status = state.status_words(&device)?;
            // Zero factors cull every patch; an upload reads zero past its data.
            let zeros = match factors {
                Some(_) => None,
                None => Some(state.uploads.upload(&device, &[], 16)?),
            };
            (status, zeros)
        };
        let status = (status.raw(), status.offset(), ExecutionStatus::BYTES as u64);
        let mut set = [(vk::Buffer::null(), 0, 0); 4];
        set[TessellationScanModule::WORK_BINDING as usize] =
            (work.work.raw(), work.work.offset(), work.work.len());
        set[TessellationScanModule::RECORDS_BINDING as usize] = work.records;
        set[TessellationScanModule::STATUS_BINDING as usize] = status;
        set[TessellationScanModule::FACTORS_BINDING as usize] = match (factors, zeros) {
            (Some(factors), _) => factors,
            (None, Some(zeros)) => (zeros.raw(), zeros.offset(), zeros.len()),
            (None, None) => return Err(internal("a tessellation scan without its factors")),
        };
        let owned = OwnedSets::write(&device, &kernel, &[set.to_vec()])?;
        let sets = owned.sets.clone();
        core.retain(owned);
        let copy = arguments.map(|(raw, offset)| {
            let region = vk::BufferCopy {
                src_offset: offset,
                dst_offset: work.work.offset(),
                size: 16,
            };
            (raw, work.work.raw(), region)
        });
        Ok(TessellationSteps {
            kernel,
            sets,
            copy,
            uses,
        })
    }

    /// The scan's uses.
    pub(crate) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Announce the emulation once. Call it once the draw records.
    pub(crate) fn retain(&self, core: &mut EncoderCore) {
        core.device()
            .degrade(Degradation::EmulatedTessellationDraws, || {
                (
                    "a patch draw".to_owned(),
                    "the device does not enable tessellationShader, so a scan dispatch before \
                     the draw's render pass instance processes its patches' factors, and its \
                     post-tessellation vertex function draws the tessellator's triangles as a \
                     plain vertex function"
                        .to_owned(),
                )
            });
    }

    /// Record the scan into `core`'s command buffer before the pass: the arguments' copy and the
    /// dispatch, each between a barrier from everything before and one to everything after.
    pub(crate) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatch(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the scan into `command_buffer` between a barrier from everything before and one to
    /// everything after: before a render pass instance of the encoder's pass, after the
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
            self.dispatch(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The arguments' copy, if any, then a barrier from everything before and one to everything
    /// after, and the scan's one workgroup: recorded after a barrier from everything before.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it; the
    /// kernel is the device's, and the set, the copy's buffers and what the set names are live
    /// until it completes (the command buffer holds them); the copy's region lies in both of its
    /// buffers (the encoder's checks held the arguments inside the Metal buffer, and the work
    /// buffer's header is 16 words).
    unsafe fn dispatch(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let point = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: as the caller states.
        unsafe {
            if let Some((from, into, region)) = self.copy {
                raw.cmd_copy_buffer(command_buffer, from, into, &[region]);
                Barrier::everything().record(raw, command_buffer);
            }
            raw.cmd_bind_pipeline(command_buffer, point, self.kernel.pipeline);
            raw.cmd_bind_descriptor_sets(
                command_buffer,
                point,
                self.kernel.layout,
                0,
                &self.sets,
                &[],
            );
            raw.cmd_dispatch(command_buffer, 1, 1, 1);
        }
    }
}
