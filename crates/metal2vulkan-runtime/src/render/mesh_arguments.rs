//! The arguments pass of an indirect mesh draw on a device with `VK_EXT_mesh_shader`
//! (`DrawCommands::mesh_arguments`; the module and its rule are `bind::MeshArguments`'s), and a
//! folded object function's direct draw's fold (`DrawCommands::object_fold`).
//!
//! `vkCmdDrawMeshTasksIndirectEXT` requires every group count within the device's limits for the
//! draw's first stage, which Metal's GPU-written arguments need not be, and Metal draws nothing
//! for some grids the device would launch. So the draw reads the counts this pass writes:
//!
//! - **Before the pass** (hoisted, as an emulated mesh draw's dispatches are, `render/mesh.rs`):
//!   one invocation of the device's module of the draw's form reads Metal's arguments and writes
//!   the draw's scratch in the command buffer's upload arena: the mesh form's 12 bytes of counts,
//!   Metal's where Metal draws them and the device launches them, else zero; or a folded object
//!   function's region record, pieces and commands (`bind::fold_object_grid`). A barrier from
//!   everything before, and one to everything after.
//! - **In the pass**, the draw's `vkCmdDrawMeshTasksIndirectEXT` from those bytes, or a folded
//!   object function's `vkCmdDrawMeshTasksIndirectCountEXT` of its pieces, its object stage reading
//!   the record at the scratch's address (`binding::Tables::region_record`).
//! - **Where it records**: as an emulated mesh draw's steps (`render.rs`): before the render pass,
//!   or, where an earlier draw or tile dispatch of the encoder wrote the arguments behind a memory
//!   barrier, before the draw's own render pass instance, after the writer's. A write since the
//!   last barrier, which Metal does not order before the draw, is read as it was before the pass.
//! - **A folded object function's direct draw** uploads its fold ([`FoldUpload`]): the record its
//!   object stage reads, and its pieces' commands where it has more than one.
//! - **An object grid Metal's GPU discards**: one of more than 2^29 threadgroups
//!   (`bind::METAL_OBJECT_GRID_TOTAL`, past what Metal runs within its watchdog's 5 seconds). It
//!   draws nothing and fails the command buffer with Metal's error
//!   (`CommandBufferError::GpuRecoveryVictim`): a direct draw from its encoder on, as Metal's
//!   (`render.rs`); an indirect one as the pass reports it (`MESH_GRID_DISCARDED`), the rest of
//!   its encoder and the encoders after it running.
//! - **A grid the device cannot launch**: on a device that does not fold object grids, an
//!   indirect object grid past the device's task limits, which Metal draws. It draws nothing and
//!   fails the command buffer (`CommandBufferError::MeshGridPastLimits`, the runtime's failure):
//!   nothing is drawn short silently.

use crate::device::functions::Vk;
use crate::draw_source::DrawSource;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use crate::resource::ResourceKey;
use crate::tracker::{Access, Barrier, Use, UseRange};
use crate::upload::Upload;
use ash::vk;
use metal2vulkan::bind::{
    ExecutionStatus, IndirectRef, MeshArguments, MeshArgumentsForm, MeshArgumentsModule, ObjectFold,
};

/// One draw's pass, planned: its scratch, kernel and set, and its read of Metal's arguments.
pub(crate) struct ArgumentsPass {
    scratch: Upload,
    form: MeshArgumentsForm,
    kernel: OwnedKernel,
    sets: OwnedSets,
    push: [u8; 8],
    uses: Vec<Use<ResourceKey>>,
}

impl ArgumentsPass {
    /// The pass of the indirect mesh draw of `form`, of `threads` threads per object threadgroup,
    /// that reads its arguments from `indirect`, in `core`'s command buffer. Refused as uploads,
    /// kernels and descriptor sets are.
    pub(crate) fn new(
        core: &mut EncoderCore,
        indirect: &DrawSource<'_>,
        form: MeshArgumentsForm,
        threads: [u32; 3],
    ) -> Result<ArgumentsPass, Error> {
        let device = core.device().clone();
        let alignment = device
            .properties()
            .description
            .limits
            .min_storage_buffer_offset_alignment;
        let pass = MeshArguments::new(IndirectRef::new(0, indirect.at), form, alignment, threads)
            .ok_or_else(|| {
            internal("a mesh arguments pass at an offset not a multiple of 4")
        })?;
        let kernel = device.owned_kernel(OwnedModule::MeshArguments(form))?;
        let (scratch, status) = {
            let mut state = core.command_buffer().state();
            let bytes = u64::from(MeshArguments::scratch_bytes(form));
            let scratch = state.uploads.upload(&device, &[], bytes)?;
            (scratch, state.status_words(&device)?)
        };
        let mut set = [(vk::Buffer::null(), 0, 0); 3];
        set[MeshArgumentsModule::SOURCE_BINDING as usize] = (
            indirect.raw,
            pass.source_range.offset,
            pass.source_range.range,
        );
        set[MeshArgumentsModule::SCRATCH_BINDING as usize] =
            (scratch.raw(), scratch.offset(), scratch.len());
        set[MeshArgumentsModule::STATUS_BINDING as usize] =
            (status.raw(), status.offset(), ExecutionStatus::BYTES as u64);
        let sets = OwnedSets::write(&device, &kernel, &[set])?;
        Ok(ArgumentsPass {
            scratch,
            form,
            kernel,
            sets,
            push: pass.push_constants,
            uses: reads(indirect),
        })
    }

    /// The scratch's `VkBuffer` and offset: the draw's arguments.
    pub(crate) fn scratch(&self) -> (vk::Buffer, u64) {
        (self.scratch.raw(), self.scratch.offset())
    }

    /// The device address of a folded object function's region record in the scratch, which its
    /// object stage reads; `None` for every other form.
    pub(crate) fn record(&self) -> Result<Option<u64>, Error> {
        if self.form != MeshArgumentsForm::FoldedObject {
            return Ok(None);
        }
        let address = self
            .scratch
            .device_address()
            .ok_or_else(|| internal("a folded object grid without device addresses"))?;
        Ok(Some(address + u64::from(MeshArguments::OBJECT_RECORD)))
    }

    /// Hold the pass's set in `core`'s command buffer until it completes; what is left records
    /// its dispatch (`render.rs`). Call it once the draw records.
    pub(crate) fn into_steps(self, core: &mut EncoderCore) -> ArgumentsSteps {
        let steps = ArgumentsSteps {
            kernel: self.kernel,
            sets: self.sets.sets.clone(),
            push: self.push,
            uses: self.uses,
        };
        core.retain(self.sets);
        steps
    }
}

/// The pass's read of Metal's 12 bytes of arguments, in the compute stage.
fn reads(indirect: &DrawSource<'_>) -> Vec<Use<ResourceKey>> {
    let read = Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    let (arguments, offset) = indirect.metal();
    let (identity, range) = arguments.tracked(offset..offset + 12);
    vec![Use::reads(
        identity.key,
        identity.tracked,
        UseRange::Bytes(range),
        read,
    )]
}

/// A planned pass's dispatch, its set held by the command buffer
/// ([`ArgumentsPass::into_steps`]).
pub(crate) struct ArgumentsSteps {
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    push: [u8; 8],
    uses: Vec<Use<ResourceKey>>,
}

impl ArgumentsSteps {
    /// The pass's read of Metal's arguments.
    pub(crate) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the dispatch into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
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

    /// Record the dispatch into `command_buffer` between a barrier from everything before and
    /// one to everything after: before a render pass instance of the encoder's pass, after the
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

    /// The module's one invocation.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn dispatch(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; the kernel is the device's, its set and the buffers it
        // names live until the command buffer completes (retained by `into_steps`), and the push
        // range is the kernel's.
        unsafe {
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            raw.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &self.sets,
                &[],
            );
            raw.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.push,
            );
            raw.cmd_dispatch(command_buffer, 1, 1, 1);
        }
    }
}

/// A folded object function's direct draw's fold, uploaded into `core`'s command buffer's arena
/// (`DrawCommands::object_fold`): the record its object stage reads, and its pieces' commands.
pub(crate) struct FoldUpload {
    upload: Upload,
}

impl FoldUpload {
    /// The upload of `fold`. Refused as uploads are.
    pub(crate) fn new(core: &mut EncoderCore, fold: &ObjectFold) -> Result<FoldUpload, Error> {
        let device = core.device().clone();
        let mut state = core.command_buffer().state();
        let bytes = fold.bytes();
        let upload = state.uploads.upload(&device, &bytes, bytes.len() as u64)?;
        Ok(FoldUpload { upload })
    }

    /// The device address of the record its object stage reads.
    pub(crate) fn record(&self) -> Result<u64, Error> {
        let address = self
            .upload
            .device_address()
            .ok_or_else(|| internal("a folded object grid without device addresses"))?;
        Ok(address + u64::from(MeshArguments::OBJECT_RECORD))
    }

    /// The upload's `VkBuffer` and offset: its pieces' commands, at the draw's offset past it.
    pub(crate) fn commands(&self) -> (vk::Buffer, u64) {
        (self.upload.raw(), self.upload.offset())
    }
}
