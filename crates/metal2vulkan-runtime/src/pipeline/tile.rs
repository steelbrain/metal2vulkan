//! Tile pipeline states (`MTLRenderPipelineState` of
//! `newRenderPipelineStateWithTileDescriptor:options:reflection:error:`):
//! `Device::new_tile_render_pipeline_state`.
//!
//! - **Creation.** A tile pipeline's state is a compute state of its tile function
//!   ([`Kernel::Tile`]): the function compiles under the descriptor's color attachment formats,
//!   as `compile_tile_pipeline` compiles it, with a compute state's variants, stand-ins, memo,
//!   archives and disk cache. What the compile refuses is refused (`Compile`, `Unsupported`):
//!   among it a function that reaches an attachment without a format in the descriptor
//!   (`InvalidDescriptor(ColorAttachment)`), where Metal's creation answers "Invalid pixel format
//!   for render target 0", and a member whose type the attachment's format does not hold, which
//!   Metal's API validation stops (an M3 Max and an M4).
//! - **Answers.** A tile pipeline's state answers its descriptor's
//!   `threadgroup_size_matches_tile_size`, no indirect command buffers, and the compiled
//!   pipeline's `max_total_threads_per_threadgroup`: Metal's 1024, or the descriptor's limit, where
//!   the device runs a threadgroup of 1024 threads, else the device's lower limit.
//!   `imageblock_sample_length` is the aliased explicit imageblock's size, which the compile
//!   records (`RuntimeFact::AliasedImageblock`), else the attachments' texels packed as Metal packs
//!   them, after an explicit imageblock that stands alone as a fragment function's
//!   (`RuntimeFact::TileImageblock`, 16 for two `uint`s beside an `R32Uint` attachment, measured on
//!   an M4 by our probe); its memory length for a tile is that times the tile's cells,
//!   unrounded, as Metal's.

use super::render::RenderPipelineState;
use super::texture_access::{TextureAccessMemo, TextureAccesses};
use super::{ComputePipelineState, Kernel, PipelineOptions};
use crate::binary_archive::Used;
use crate::error::Error;
use crate::Device;
use metal2vulkan::metal::format::{explicit_imageblock_sample_length, imageblock_sample_length};
use metal2vulkan::pipeline::TileRenderPipelineDescriptor;
use metal2vulkan::reflection::RenderPipelineReflection;
use std::sync::Arc;

/// A tile pipeline's state: the compute state of its tile function, and its descriptor.
#[derive(Clone)]
pub(crate) struct TileState(Arc<TileShared>);

struct TileShared {
    kernel: ComputePipelineState,
    descriptor: TileRenderPipelineDescriptor,
    /// The tile function's texture slots and their accesses (`texture_access.rs`), read when
    /// first asked.
    texture_accesses: TextureAccessMemo,
}

impl Device {
    /// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` with no reflection:
    /// compile the tile pipeline `descriptor` into a render pipeline state under `options`
    /// (`PipelineOptions::default()` is Metal's `newRenderPipelineStateWithTileDescriptor:...`
    /// without options). Its tile function is a function's descriptor
    /// ([`Function::function_descriptor`](crate::Function::function_descriptor)) with
    /// the functions it links; a function the compile layer already compiled
    /// (`StageSource::Compiled`) is refused (`Invalid(CompiledKernel)`), and so are linked
    /// functions named twice and an archive of another device, as a compute state's are. Under
    /// `FAIL_ON_BINARY_ARCHIVE_MISS`, a pipeline no listed archive holds is `BinaryArchiveMiss`
    /// (a tile pipeline's archive key). What the compile refuses is `Compile` or `Unsupported`. The
    /// tile function's imports come from its dynamic libraries, as a compute state's do
    /// ([`Device::new_compute_pipeline_state`]).
    pub fn new_tile_render_pipeline_state(
        &self,
        descriptor: &TileRenderPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<RenderPipelineState, Error> {
        TileState::create(self, descriptor, options, None).map(RenderPipelineState::of_tile)
    }

    /// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` with reflection:
    /// [`Device::new_tile_render_pipeline_state`], and the pipeline's reflection when `options`
    /// asks for it (any word but `NONE`, which Metal answers alike),
    /// `None` when it does not, as Metal leaves `reflection` nil. The reflection is
    /// [`TileRenderPipelineDescriptor::reflection`]'s (its tile bindings), under the creation's
    /// budget; what it refuses refuses the call (`Unsupported`, `Compile`).
    pub fn new_tile_render_pipeline_state_with_reflection(
        &self,
        descriptor: &TileRenderPipelineDescriptor,
        options: &PipelineOptions,
    ) -> Result<(RenderPipelineState, Option<RenderPipelineReflection>), Error> {
        let state = TileState::create(self, descriptor, options, None)?;
        // The descriptor the state was made from, with the dynamic libraries its creation loaded
        // (`dynamic_library.rs`).
        let reflection = match options.options.reflects() {
            true => Some(state.kernel().tile_reflection(&options.budget())?),
            false => None,
        };
        Ok((RenderPipelineState::of_tile(state), reflection))
    }
}

impl TileState {
    /// Whether `other` is this state (a clone of it), not another state alike.
    pub(crate) fn is(&self, other: &TileState) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// `new_tile_render_pipeline_state` of `descriptor` under `options` on `device`, noting what
    /// it uses in `used` (a binary archive's add). The tile function's dynamic libraries are
    /// loaded first (`dynamic_library.rs`), and the state keeps the descriptor with them.
    pub(crate) fn create(
        device: &Device,
        descriptor: &TileRenderPipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
    ) -> Result<TileState, Error> {
        TileState::create_adding(device, descriptor, options, used, 0)
    }

    /// [`create`](Self::create), of a descriptor whose tile function's last `added_unsupported`
    /// `binary_functions` were added without its `support_adding_binary_functions`.
    pub(super) fn create_adding(
        device: &Device,
        descriptor: &TileRenderPipelineDescriptor,
        options: &PipelineOptions,
        used: Option<&Used>,
        added_unsupported: usize,
    ) -> Result<TileState, Error> {
        // Metal answers nil for a tile state of no samples (`raster_samples.rs`).
        super::raster_samples::check_tile(
            "new_tile_render_pipeline_state",
            descriptor.raster_sample_count,
        )?;
        let requested = descriptor;
        let mut descriptor = descriptor.clone();
        descriptor.tile = crate::dynamic_library::stage_with_dependencies(&descriptor.tile);
        let kernel = ComputePipelineState::of_kernel(
            device,
            Kernel::Tile(descriptor.clone()),
            options,
            used,
            "new_tile_render_pipeline_state",
            added_unsupported,
        )?;
        // The creation's record, as the program asked for it (`warm/record.rs`); a binary
        // archive's add makes none.
        if used.is_none() {
            device.record_pipeline(|| {
                crate::warm::Pipeline::Tile(requested.clone(), added_unsupported)
            });
        }
        Ok(TileState(Arc::new(TileShared {
            kernel,
            descriptor,
            texture_accesses: TextureAccessMemo::new(),
        })))
    }

    /// The compute state of the tile function, which a tile dispatch runs.
    pub(crate) fn kernel(&self) -> &ComputePipelineState {
        &self.0.kernel
    }

    /// The descriptor the state was made from.
    pub(crate) fn descriptor(&self) -> &TileRenderPipelineDescriptor {
        &self.0.descriptor
    }

    /// The tile function's texture slots and their accesses, from the pipeline's slot writes the
    /// first time. They read the tile function's roles, declared accesses and slots, so an
    /// argument whose type a reflection cannot describe does not refuse them
    /// (`pipeline::texture_access`).
    pub(crate) fn texture_accesses(&self) -> Result<&TextureAccesses, Error> {
        TextureAccesses::remembered(&self.0.texture_accesses, |budget| {
            self.descriptor()
                .slot_writes(budget)
                .map_err(crate::device::device_error)
        })
    }

    /// `imageblockSampleLength`: the explicit imageblock's size where the tile function aliases
    /// one onto the implicit imageblock (`RuntimeFact::AliasedImageblock`), else the color
    /// attachments' texels as [`imageblock_sample_length`] packs them, after an explicit imageblock
    /// that stands alone as [`explicit_imageblock_sample_length`] places them; `None` for a
    /// descriptor giving a color format without a texel size, which no pass has.
    pub(crate) fn imageblock_sample_length(&self) -> Option<u64> {
        let kernel = &self.0.kernel;
        if let Some(bytes) = kernel.aliased_imageblock_bytes() {
            return Some(bytes.into());
        }
        let colors = imageblock_sample_length(&self.0.descriptor.color_attachment_formats)?;
        let length = match kernel.tile_imageblock() {
            Some((data_size, alignment)) => {
                explicit_imageblock_sample_length(data_size, alignment, colors)?
            }
            None => colors,
        };
        Some(length.into())
    }
}
