//! The runtime's own compute work: the crate-owned modules of the compile-and-plan layer
//! (`bind`'s resolve and prepass modules) the runtime dispatches for Metal commands that have no
//! single Vulkan command.
//!
//! - **Kernels.** Each [`OwnedModule`] is made once per device, when a command first needs it:
//!   its shader module, a set layout of set 0 (sampled images first, at bindings `0..i`, where
//!   the module has them, then storage images, then storage buffers), a pipeline layout with its
//!   push range, and the compute pipeline. The device keeps them
//!   ([`OwnedKernels`]) and destroys them with itself; a command buffer holds its device, so a
//!   kernel outlives every command that records it. The mipmap filter is one kernel per
//!   `MipmapFilterKernel` a chain dispatches (a format's 2D and 3D steps, fetching or not;
//!   bounded by the formats), and
//!   the index rewrite one per `RewriteProgram` and `RewritePhase` (closed enums of a few).
//! - **Descriptor sets** ([`OwnedSets`]): one set per dispatch of a command, each written with the
//!   command's image views and buffer ranges, from a fitted arena of their own
//!   (`descriptor_arena.rs`) that takes pools of the device's spares (`spares.rs`); the command
//!   buffer, or the resource whose moves they are, holds them until it no longer uses them, and
//!   their pools then go back to the device's spares, which reset a pool before it serves again
//!   (`WorkCounts::owned_descriptor_pools`). So a pass's command, or a texture made, opens no pool
//!   where a spare has room: on MoltenVK a pool made is a Metal buffer made. A kernel whose set 0
//!   is a push descriptor set's ([`OwnedKernel::pushed`]) takes no pool: each dispatch pushes its
//!   descriptors as it records ([`OwnedBinds`], which binds either way by that field alone).
//!   Vulkan forbids a set from such a layout ("Each element of `pSetLayouts` must not have been
//!   created with `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT` set",
//!   `VkDescriptorSetAllocateInfo`), so `OwnedSets` refuses its kernel (internal).

use super::binding::{self, Info, SetWrite};
use super::descriptor_arena::{DescriptorArena, SetNeeds};
use super::device::lock;
use super::error::{internal, vulkan, Error};
use super::release;
use super::texture::decoded_twin::TwinDecode;
use super::ycbcr_device::YCbCrSampler;
use super::Device;
use crate::device::functions::Vk;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    MipmapFilterKernel, RewritePhase, RewriteProgram, ShadowMergeRead, ShadowMergeStores,
    ShadowTexelView, TessellationScanKernel, YCbCrCodes, YCbCrConversion,
};
use metal2vulkan::metal::format::ExtendedRange;
use std::collections::HashMap;

/// A crate-owned module the runtime dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OwnedModule {
    /// `bind::visibility_resolve_module`: a pass's `Boolean` visibility results, made Metal's 1
    /// or 0 in place.
    VisibilityResolve,
    /// `bind::visibility_combine_module`: a split pass's continuation queries, ORed into their
    /// results.
    VisibilityCombine,
    /// `bind::index_bounds_plan_module`: an indexed indirect draw's index bounds plan
    /// (`index_bounds.rs`).
    IndexBoundsPlan,
    /// `bind::index_bounds_fill_module`: the plan's reservation in the index pool, filled.
    IndexBoundsFill,
    /// `bind::index_bounds_plan_run_module`: a run of indexed indirect draws' index bounds plans,
    /// by one invocation (`index_bounds.rs`).
    IndexBoundsPlanRun,
    /// `bind::index_bounds_fill_run_module`: a run's reservations in the index pool, filled by
    /// one dispatch.
    IndexBoundsFillRun,
    /// `bind::index_gather_module`: the lists of a run of draws, by one dispatch
    /// (`index_rewrite.rs`).
    IndexGather,
    /// `bind::index_gather_plan_module`: an indirect run's lists placed in the index pool from
    /// Metal's arguments, before its gather (`index_rewrite.rs`).
    IndexGatherPlan,
    /// `bind::replay_preprocess_module`: a device replay's preprocess, which reads every buffer
    /// at its device address (no storage buffer; its push bytes, `gpu_execute.rs`).
    ReplayPreprocess,
    /// `bind::draw_replay_module`: a draw replay's preprocess, which reads every buffer at its
    /// device address (no storage buffer; its push bytes, `render/executions/gpu.rs`).
    DrawReplayPreprocess,
    /// `bind::draw_merge_module`: a merged draw replay's merge pass of `phase`, after its
    /// preprocess (no descriptor; its push bytes, `render/executions/gpu.rs`).
    DrawMerge(metal2vulkan::bind::MergePhase),
    /// `bind::flat_dispatch_module`: an indirect dispatch's region record and group counts on its
    /// flat variant, from Metal's arguments (`indirect_dispatch.rs`). No descriptor: it reaches
    /// memory by device address.
    FlatDispatch,
    /// `bind::shadow_merge_module` storing in one form (`ShadowMergeStores`): a shadowed
    /// texture's writes merged into its buffer's bytes (`shadow_merge.rs`), or a linear
    /// attachment's pass into its surface's (`linear_merge.rs`).
    ShadowMerge(ShadowMergeStores),
    /// `bind::shadow_copy_back_module` of one view of a texel (`ShadowTexelView`): a shadowed
    /// texture's texels stored back into its buffer's rows by a dispatch, where the device copies
    /// back faster so (`shadow_copy_back.rs`). Its shadow's unsigned-integer view is its sampled
    /// image.
    ShadowCopyBack(ShadowTexelView),
    /// `bind::rows_copy_module`: many equal rows of words copied from one buffer into another by
    /// a dispatch, in place of a copy's region per row (`row_copies.rs`).
    RowsCopy,
    /// `bind::shadow_merge_image_module` of one read of a texel (`ShadowMergeRead`): a shadowed
    /// texture's writes merged into its buffer's bytes, reading its shadow's texels through its
    /// unsigned-integer view or its own 8-bit unsigned normalized view, its sampled image
    /// (`shadow_merge.rs`).
    ShadowMergeImage(ShadowMergeRead, ShadowMergeStores),
    /// `bind::shadow_merge_kept_image_module` of one view of a texel (`ShadowTexelView`): as
    /// [`OwnedModule::ShadowMergeImage`], the texels before the commands read from an image copy
    /// of the shadow, its second sampled image (`shadow_merge.rs`).
    ShadowMergeKeptImage(ShadowTexelView, ShadowMergeStores),
    /// `bind::shadow_copy_in_module` of one view of a texel (`ShadowTexelView`): a merged
    /// shadowed texture's texels copied in from its buffer's rows and kept in its merge's scratch
    /// by one dispatch (`shadow_copy_in.rs`). Its shadow's unsigned-integer view is its storage
    /// image.
    ShadowCopyIn(ShadowTexelView),
    /// `bind::mipmap_filter_module` of one kernel: a level of a mipmap chain from the level
    /// above it, by the compute filter of a format the device's blits cannot filter
    /// (`mipmap_filter.rs`); a kernel that fetches the first level binds its sampled view first.
    MipmapFilter(MipmapFilterKernel),
    /// `bind::mesh_scan_module`: a scanned emulated mesh draw's mesh threadgroups, records and
    /// draw (`render/mesh.rs`).
    MeshScan,
    /// `bind::mesh_object_arguments_module`: an emulated indirect draw's object grid, its slots
    /// reserved and its object dispatch written, under the scan's bindings (`render/mesh.rs`).
    MeshObjectArguments,
    /// `bind::mesh_arguments_module` of one form: an indirect mesh draw's group counts, as Metal
    /// draws them within the device's limits, or a folded object function's pieces
    /// (`render/mesh_arguments.rs`).
    MeshArguments(metal2vulkan::bind::MeshArgumentsForm),
    /// `bind::plane_words_module`: a blit's color words of a two-plane extended-range texture
    /// moved between its texels and a buffer (`two_planes.rs`).
    PlaneWords,
    /// `bind::plane_copy_module`: a blit between two two-plane extended-range textures, both
    /// planes' bytes moved as Metal's blit moves them (`two_planes.rs`).
    PlaneCopy,
    /// `bind::index_rewrite_module` of one program's dispatch of one phase: an indexed draw's
    /// primitives as a list (`index_rewrite.rs`).
    IndexRewrite(RewriteProgram, RewritePhase),
    /// `bind::index_rewrite_run_module` of one program: a run of direct indexed draws'
    /// primitives as lists, one workgroup per draw (`index_rewrite.rs`).
    IndexRewriteRun(RewriteProgram),
    /// `bind::rotate_texels_module`: a rotated texture's texels moved between its pass's frame
    /// and its bytes (`rotated.rs`).
    RotateTexels,
    /// `bind::ycbcr_convert_module` of one conversion (`YCbCrConversion`: the layout of a
    /// texture's codes, their range and their matrix): a YCbCr texture's texels converted from
    /// its surface's two planes into its image (`ycbcr.rs`), which it binds as a storage image:
    /// converted values, or (`YCbCrOutput::Codes`) the packed codes; or, with `true`,
    /// `bind::ycbcr_convert_floats_module`: the texels as floats into a render pass's render
    /// shadow of the texture, which the pass loads (`ycbcr.rs`). One per conversion a texture of
    /// the device reads through, within the two-plane formats' layouts and the ranges and
    /// matrices they convert under.
    YCbCrConvert(YCbCrConversion, bool),
    /// `bind::ycbcr_sample_module`: a YCbCr texture's planes' image sampled through the device's
    /// own Y′CbCr conversion into its image, the output of the codes; its layout bakes the
    /// conversion's sampler, made with the first kernel of it and released with the device
    /// (`ycbcr_device.rs`). One per sampler and output a texture of the device reads through.
    YCbCrSample(YCbCrSampler, YCbCrCodes),
    /// `bind::chroma_sample_module`: a 4:2:2 texture's pairs' image sampled through the device's
    /// own Y′CbCr conversion into a band of its decoded twin's texels
    /// (`texture/decoded_twin.rs`); its layout bakes the conversion's sampler, as
    /// [`YCbCrSample`](Self::YCbCrSample)'s does. One per sampler (a format, two at most).
    ChromaSample(YCbCrSampler),
    /// `bind::ycbcr_refresh_module`: a YCbCr texture's planes compared with the copy its last
    /// conversion read, the changed words stored over it, and the conversion's indirect group
    /// count written where any changed (`ycbcr.rs`).
    YCbCrRefresh,
    /// `bind::two_plane_texels_module`: a heap's two-plane extended-range texture's texels moved
    /// between its image and its two planes in the heap's bytes (`texture/heap_two_planes.rs`).
    TwoPlaneTexels,
    /// `bind::plane_bytes_copy_module`: a blit into a heap's two-plane extended-range texture,
    /// the heap's bytes written as Metal's blit writes them (`two_planes.rs`).
    PlaneBytesCopy,
    /// `bind::ycbcr_write_module`: a 4:4:4 YCbCr texture's written texels encoded into its
    /// surface's two planes (`ycbcr.rs`), its write shadow bound as a storage image.
    YCbCrWrite,
    /// `bind::tiled_texels_module`: a heap texture's elements moved between its image's tight
    /// copy and Metal's tiles in the heap's bytes (`texture/heap_tiled.rs`).
    TiledTexels,
    /// `bind::extended_shadow_module` of one form: an extended-range attachment's `half` shadow
    /// set from its codes before the instance whose draws first hold it
    /// (`render/extended_shadow.rs`). Two storage images: the codes, then the shadow.
    ExtendedShadowLoad(ExtendedRange),
    /// `bind::predication_step_module`: one command-predication record's step, its skip counter
    /// moved and its dispatches gated (`predication.rs`). On a device with device addresses no
    /// descriptor: it reaches memory by device address; elsewhere five storage buffers, its
    /// record's region, its unit's counter, Metal's word, its loop's flag and the overflow word.
    PredicationStep,
    /// The GPU decode of one decoded twin's blocks (`TwinDecode::module`:
    /// `bind::astc_decode_module` and the other carriers' modules): a carrier's written region
    /// decoded as Metal decodes it into its decoded twin (`texture/decoded_twin.rs`).
    TwinDecode(TwinDecode),
    /// `bind::tessellation_scan_module` of one pipeline's tessellation state: an emulated patch
    /// draw's records and chunk draws (`render/tessellation_scan.rs`). One per state a pipeline
    /// of the device draws with (its domain, partition, winding, maximum factor, step function
    /// and scale: a few).
    TessellationScan(TessellationScanKernel),
}

impl OwnedModule {
    /// The module's sampled-image bindings, which come first in its set.
    fn images(self) -> u32 {
        match self {
            OwnedModule::MipmapFilter(kernel) => u32::from(kernel.fetches()),
            OwnedModule::ShadowCopyBack(_) | OwnedModule::ShadowMergeImage(..) => 1,
            OwnedModule::ShadowMergeKeptImage(..) => 2,
            // The combined image sampler of the planes' (or pairs') image.
            OwnedModule::YCbCrSample(..) | OwnedModule::ChromaSample(_) => 1,
            _ => 0,
        }
    }

    /// Whether the module's set 0 is pushed on `device` ([`OwnedKernel::pushed`]): the rows
    /// copy's and a shadowed texture's copies' and merges', on a device with
    /// `VK_KHR_push_descriptor`, so their commands make no pool on the encoding thread
    /// (`row_copies.rs`, `shadow_copy_back.rs`, `shadow_copy_in.rs`, `shadow_merge.rs`).
    fn pushed(self, device: &Device) -> bool {
        matches!(
            self,
            OwnedModule::RowsCopy
                | OwnedModule::ShadowCopyBack(_)
                | OwnedModule::ShadowCopyIn(_)
                | OwnedModule::ShadowMerge(_)
                | OwnedModule::ShadowMergeImage(..)
                | OwnedModule::ShadowMergeKeptImage(..)
        ) && device.shared().vk().push_descriptor().is_some()
    }

    /// The module's storage-image bindings, which come after its sampled images.
    fn storage_images(self) -> u32 {
        match self {
            OwnedModule::YCbCrConvert(..)
            | OwnedModule::YCbCrSample(..)
            | OwnedModule::YCbCrWrite
            | OwnedModule::ShadowCopyIn(_) => 1,
            OwnedModule::ExtendedShadowLoad(_) => 2,
            _ => 0,
        }
    }

    /// The module's SPIR-V words for `device`, its storage-buffer bindings (after its
    /// [`images`](Self::images)), and its push bytes.
    fn build(self, device: &Device) -> Result<(Vec<u32>, u32, u32), Error> {
        match self {
            OwnedModule::VisibilityResolve => {
                let module = metal2vulkan::bind::visibility_resolve_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 2, 0))
            }
            OwnedModule::VisibilityCombine => {
                let module = metal2vulkan::bind::visibility_combine_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::IndexBoundsPlan => {
                let module = metal2vulkan::bind::index_bounds_plan_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
            OwnedModule::IndexBoundsFill => {
                let module = metal2vulkan::bind::index_bounds_fill_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::IndexBoundsPlanRun => {
                let module =
                    metal2vulkan::bind::index_bounds_plan_run_module(device.compile_device())
                        .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
            OwnedModule::IndexBoundsFillRun => {
                let module =
                    metal2vulkan::bind::index_bounds_fill_run_module(device.compile_device())
                        .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::IndexGather => {
                let module = metal2vulkan::bind::index_gather_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
            OwnedModule::IndexGatherPlan => {
                let module = metal2vulkan::bind::index_gather_plan_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 5, 0))
            }
            OwnedModule::ReplayPreprocess => {
                let module = metal2vulkan::bind::replay_preprocess_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((
                    module.spirv.to_vec(),
                    0,
                    metal2vulkan::bind::ReplayPreprocessModule::PUSH_BYTES,
                ))
            }
            OwnedModule::DrawReplayPreprocess => {
                let module = metal2vulkan::bind::draw_replay_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((
                    module.spirv.to_vec(),
                    0,
                    metal2vulkan::bind::DrawReplayModule::PUSH_BYTES,
                ))
            }
            OwnedModule::DrawMerge(phase) => {
                let module = metal2vulkan::bind::draw_merge_module(device.compile_device(), phase)
                    .map_err(Error::Compile)?;
                Ok((
                    module.spirv.to_vec(),
                    0,
                    metal2vulkan::bind::DrawMergeModule::PUSH_BYTES,
                ))
            }
            OwnedModule::FlatDispatch => {
                let module = metal2vulkan::bind::flat_dispatch_module(device.compile_device())
                    .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::FlatDispatchModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 0, push_bytes))
            }
            OwnedModule::ShadowMerge(stores) => {
                let module =
                    metal2vulkan::bind::shadow_merge_module(device.compile_device(), stores)
                        .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::ShadowMergeModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 3, push_bytes))
            }
            OwnedModule::ShadowCopyBack(view) => {
                let module =
                    metal2vulkan::bind::shadow_copy_back_module(device.compile_device(), view)
                        .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::ShadowCopyBackModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 1, push_bytes))
            }
            OwnedModule::RowsCopy => {
                let module = metal2vulkan::bind::rows_copy_module(device.compile_device())
                    .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::RowsCopyModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 2, push_bytes))
            }
            OwnedModule::ShadowMergeImage(read, stores) => {
                let module = metal2vulkan::bind::shadow_merge_image_module(
                    device.compile_device(),
                    read,
                    stores,
                )
                .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::ShadowMergeImageModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 2, push_bytes))
            }
            OwnedModule::ShadowMergeKeptImage(view, stores) => {
                let module = metal2vulkan::bind::shadow_merge_kept_image_module(
                    device.compile_device(),
                    view,
                    stores,
                )
                .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::ShadowMergeImageModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 1, push_bytes))
            }
            OwnedModule::ShadowCopyIn(view) => {
                let module =
                    metal2vulkan::bind::shadow_copy_in_module(device.compile_device(), view)
                        .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::ShadowCopyInModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 2, push_bytes))
            }
            OwnedModule::MipmapFilter(kernel) => {
                let module =
                    metal2vulkan::bind::mipmap_filter_module(device.compile_device(), kernel)
                        .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3 - u32::from(kernel.fetches()), 0))
            }
            OwnedModule::MeshScan => {
                let module = metal2vulkan::bind::mesh_scan_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
            OwnedModule::MeshObjectArguments => {
                let module =
                    metal2vulkan::bind::mesh_object_arguments_module(device.compile_device())
                        .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
            OwnedModule::MeshArguments(form) => {
                let module =
                    metal2vulkan::bind::mesh_arguments_module(device.compile_device(), form)
                        .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::MeshArgumentsModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 3, push_bytes))
            }
            OwnedModule::PlaneWords => {
                let module = metal2vulkan::bind::plane_words_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::PlaneCopy => {
                let module = metal2vulkan::bind::plane_copy_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::IndexRewrite(program, phase) => {
                let device = device.compile_device();
                let module = metal2vulkan::bind::index_rewrite_module(device, program, phase)
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 5, 0))
            }
            OwnedModule::IndexRewriteRun(program) => {
                let device = device.compile_device();
                let module = metal2vulkan::bind::index_rewrite_run_module(device, program)
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 5, 0))
            }
            OwnedModule::YCbCrConvert(conversion, floats) => {
                let device = device.compile_device();
                let module = match floats {
                    true => metal2vulkan::bind::ycbcr_convert_floats_module(device, conversion),
                    false => metal2vulkan::bind::ycbcr_convert_module(device, conversion),
                }
                .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 2, 0))
            }
            OwnedModule::YCbCrSample(_, codes) => {
                let module =
                    metal2vulkan::bind::ycbcr_sample_module(device.compile_device(), codes)
                        .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::YCbCrSampleModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 0, push_bytes))
            }
            OwnedModule::ChromaSample(_) => {
                let module = metal2vulkan::bind::chroma_sample_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 2, 0))
            }
            OwnedModule::RotateTexels => {
                let module = metal2vulkan::bind::rotate_texels_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::YCbCrRefresh => {
                let module = metal2vulkan::bind::ycbcr_refresh_module(device.compile_device())
                    .map_err(Error::Compile)?;
                let push_bytes = metal2vulkan::bind::YCbCrRefreshModule::PUSH_BYTES;
                Ok((module.spirv.to_vec(), 3, push_bytes))
            }
            OwnedModule::TwoPlaneTexels => {
                let module = metal2vulkan::bind::two_plane_texels_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::PlaneBytesCopy => {
                let module = metal2vulkan::bind::plane_bytes_copy_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::YCbCrWrite => {
                let module = metal2vulkan::bind::ycbcr_write_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 2, 0))
            }
            OwnedModule::TiledTexels => {
                let module = metal2vulkan::bind::tiled_texels_module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::ExtendedShadowLoad(form) => {
                let device = device.compile_device();
                let module = metal2vulkan::bind::extended_shadow_module(device, form)
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 0, 0))
            }
            OwnedModule::PredicationStep => {
                use metal2vulkan::bind::{PredicationStepForm, PredicationStepModule};
                let module = metal2vulkan::bind::predication_step_module(device.compile_device())
                    .map_err(Error::Compile)?;
                let (buffers, push_bytes) = match module.form {
                    PredicationStepForm::Bound => (PredicationStepModule::BINDINGS, 0),
                    _ => (0, PredicationStepModule::PUSH_BYTES),
                };
                Ok((module.spirv.to_vec(), buffers, push_bytes))
            }
            OwnedModule::TwinDecode(decode) => {
                let module = decode
                    .module(device.compile_device())
                    .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 3, 0))
            }
            OwnedModule::TessellationScan(kernel) => {
                let module =
                    metal2vulkan::bind::tessellation_scan_module(device.compile_device(), &kernel)
                        .map_err(Error::Compile)?;
                Ok((module.spirv.to_vec(), 4, 0))
            }
        }
    }
}

/// The Vulkan objects of one kernel. The handles are the device's until it is dropped.
#[derive(Clone, Copy, Debug)]
pub(super) struct OwnedKernel {
    module: vk::ShaderModule,
    pub(super) set_layout: vk::DescriptorSetLayout,
    pub(super) layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
    /// Sampled-image bindings of set 0, its first.
    pub(super) images: u32,
    /// Storage-image bindings of set 0, after the sampled images.
    pub(super) storage_images: u32,
    /// Storage-buffer bindings of set 0, after the images.
    pub(super) buffers: u32,
    /// Whether set 0's layout is a push descriptor set's: its dispatches push their descriptors
    /// (`vkCmdPushDescriptorSetKHR`) in place of a set from a pool.
    pub(super) pushed: bool,
    /// The descriptors its first binding takes from a pool where that binding is the combined
    /// image sampler of a sampler of the device's Y′CbCr conversion its layout bakes
    /// (`combinedImageSamplerDescriptorCount`, `ycbcr_device.rs`); 0 for every other kernel.
    combined: u8,
}

impl OwnedKernel {
    /// Destroy the objects; no pending command uses them.
    fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the device is idle or the objects were never used; destroying a null handle
        // is a no-op.
        unsafe {
            release::object(raw, self.pipeline);
            release::object(raw, self.layout);
            release::object(raw, self.set_layout);
            release::object(raw, self.module);
        }
    }
}

/// Every kernel one device has made.
#[derive(Default)]
pub(super) struct OwnedKernels {
    visibility_resolve: Option<OwnedKernel>,
    visibility_combine: Option<OwnedKernel>,
    index_bounds_plan: Option<OwnedKernel>,
    index_bounds_fill: Option<OwnedKernel>,
    index_bounds_plan_run: Option<OwnedKernel>,
    index_bounds_fill_run: Option<OwnedKernel>,
    index_gather: Option<OwnedKernel>,
    index_gather_plan: Option<OwnedKernel>,
    replay_preprocess: Option<OwnedKernel>,
    draw_replay_preprocess: Option<OwnedKernel>,
    draw_merge_blocks: Option<OwnedKernel>,
    draw_merge_starts: Option<OwnedKernel>,
    flat_dispatch: Option<OwnedKernel>,
    shadow_merge: Option<OwnedKernel>,
    shadow_merge_blocks: Option<OwnedKernel>,
    shadow_copy_backs: HashMap<ShadowTexelView, OwnedKernel>,
    shadow_merge_images: HashMap<(ShadowMergeRead, ShadowMergeStores), OwnedKernel>,
    shadow_merge_kept_images: HashMap<(ShadowTexelView, ShadowMergeStores), OwnedKernel>,
    shadow_copy_ins: HashMap<ShadowTexelView, OwnedKernel>,
    mipmap_filters: HashMap<MipmapFilterKernel, OwnedKernel>,
    mesh_scan: Option<OwnedKernel>,
    mesh_object_arguments: Option<OwnedKernel>,
    mesh_arguments_mesh: Option<OwnedKernel>,
    mesh_arguments_object: Option<OwnedKernel>,
    mesh_arguments_folded: Option<OwnedKernel>,
    plane_words: Option<OwnedKernel>,
    plane_copy: Option<OwnedKernel>,
    index_rewrites: HashMap<(RewriteProgram, RewritePhase), OwnedKernel>,
    index_rewrite_runs: HashMap<RewriteProgram, OwnedKernel>,
    rotate_texels: Option<OwnedKernel>,
    rows_copy: Option<OwnedKernel>,
    ycbcr_refresh: Option<OwnedKernel>,
    ycbcr_converts: HashMap<(YCbCrConversion, bool), OwnedKernel>,
    ycbcr_samples: HashMap<(YCbCrSampler, YCbCrCodes), OwnedKernel>,
    chroma_samples: HashMap<YCbCrSampler, OwnedKernel>,
    /// The conversions and samplers the YCbCr and 4:2:2 sample kernels' layouts bake, made with
    /// the first kernel of each (or the first planes image or layout of a planes sampler,
    /// [`Device::made_ycbcr_conversion`]) and released with the device.
    ycbcr_samplers: HashMap<YCbCrSampler, (vk::SamplerYcbcrConversion, vk::Sampler)>,
    two_plane_texels: Option<OwnedKernel>,
    plane_bytes_copy: Option<OwnedKernel>,
    ycbcr_write: Option<OwnedKernel>,
    tiled_texels: Option<OwnedKernel>,
    extended_shadow_loads: HashMap<ExtendedRange, OwnedKernel>,
    predication_step: Option<OwnedKernel>,
    twin_decodes: HashMap<TwinDecode, OwnedKernel>,
    tessellation_scans: HashMap<TessellationScanKernel, OwnedKernel>,
}

impl OwnedKernels {
    /// Destroy every kernel; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        for kernel in [
            self.visibility_resolve,
            self.visibility_combine,
            self.index_bounds_plan,
            self.index_bounds_fill,
            self.index_bounds_plan_run,
            self.index_bounds_fill_run,
            self.index_gather,
            self.index_gather_plan,
            self.replay_preprocess,
            self.draw_replay_preprocess,
            self.draw_merge_blocks,
            self.draw_merge_starts,
            self.flat_dispatch,
            self.shadow_merge,
            self.shadow_merge_blocks,
            self.mesh_scan,
            self.mesh_object_arguments,
            self.mesh_arguments_mesh,
            self.mesh_arguments_object,
            self.mesh_arguments_folded,
            self.plane_words,
            self.plane_copy,
            self.rotate_texels,
            self.rows_copy,
            self.ycbcr_refresh,
            self.two_plane_texels,
            self.plane_bytes_copy,
            self.ycbcr_write,
            self.tiled_texels,
            self.predication_step,
        ]
        .into_iter()
        .flatten()
        .chain(self.shadow_copy_backs.values().copied())
        .chain(self.shadow_merge_images.values().copied())
        .chain(self.shadow_merge_kept_images.values().copied())
        .chain(self.shadow_copy_ins.values().copied())
        .chain(self.mipmap_filters.values().copied())
        .chain(self.index_rewrites.values().copied())
        .chain(self.index_rewrite_runs.values().copied())
        .chain(self.ycbcr_converts.values().copied())
        .chain(self.ycbcr_samples.values().copied())
        .chain(self.chroma_samples.values().copied())
        .chain(self.extended_shadow_loads.values().copied())
        .chain(self.twin_decodes.values().copied())
        .chain(self.tessellation_scans.values().copied())
        {
            kernel.destroy(raw);
        }
        for &(conversion, sampler) in self.ycbcr_samplers.values() {
            // SAFETY: as above; the kernels whose layouts bake them are destroyed.
            unsafe {
                release::object(raw, sampler);
                release::object(raw, conversion);
            }
        }
    }

    /// The device's conversion of `sampler` and the sampler of it that the YCbCr sample kernels
    /// of `sampler` bake, made on first use; the sampler counted against the device's samplers.
    fn ycbcr_sampler(
        &mut self,
        device: &Device,
        sampler: YCbCrSampler,
    ) -> Result<(vk::SamplerYcbcrConversion, vk::Sampler), Error> {
        if let Some(&made) = self.ycbcr_samplers.get(&sampler) {
            return Ok(made);
        }
        let made = sampler.make(device)?;
        self.ycbcr_samplers.insert(sampler, made);
        Ok(made)
    }

    /// The kernel of `which`, where the device has made it.
    fn get(&mut self, which: OwnedModule) -> Option<OwnedKernel> {
        match which {
            OwnedModule::ShadowCopyBack(view) => self.shadow_copy_backs.get(&view).copied(),
            OwnedModule::ShadowMergeImage(read, stores) => {
                self.shadow_merge_images.get(&(read, stores)).copied()
            }
            OwnedModule::ShadowMergeKeptImage(view, stores) => {
                self.shadow_merge_kept_images.get(&(view, stores)).copied()
            }
            OwnedModule::ShadowCopyIn(view) => self.shadow_copy_ins.get(&view).copied(),
            OwnedModule::MipmapFilter(kernel) => self.mipmap_filters.get(&kernel).copied(),
            OwnedModule::IndexRewrite(program, phase) => {
                self.index_rewrites.get(&(program, phase)).copied()
            }
            OwnedModule::IndexRewriteRun(program) => self.index_rewrite_runs.get(&program).copied(),
            OwnedModule::YCbCrConvert(conversion, floats) => {
                self.ycbcr_converts.get(&(conversion, floats)).copied()
            }
            OwnedModule::YCbCrSample(sampler, codes) => {
                self.ycbcr_samples.get(&(sampler, codes)).copied()
            }
            OwnedModule::ChromaSample(sampler) => self.chroma_samples.get(&sampler).copied(),
            OwnedModule::ExtendedShadowLoad(form) => self.extended_shadow_loads.get(&form).copied(),
            OwnedModule::TwinDecode(decode) => self.twin_decodes.get(&decode).copied(),
            OwnedModule::TessellationScan(kernel) => self.tessellation_scans.get(&kernel).copied(),
            _ => self.slot(which).and_then(|slot| *slot),
        }
    }

    /// Keep `kernel`, the kernel of `which`.
    fn put(&mut self, which: OwnedModule, kernel: OwnedKernel) {
        match which {
            OwnedModule::ShadowCopyBack(view) => {
                self.shadow_copy_backs.insert(view, kernel);
            }
            OwnedModule::ShadowMergeImage(read, stores) => {
                self.shadow_merge_images.insert((read, stores), kernel);
            }
            OwnedModule::ShadowMergeKeptImage(view, stores) => {
                self.shadow_merge_kept_images.insert((view, stores), kernel);
            }
            OwnedModule::ShadowCopyIn(view) => {
                self.shadow_copy_ins.insert(view, kernel);
            }
            OwnedModule::MipmapFilter(key) => {
                self.mipmap_filters.insert(key, kernel);
            }
            OwnedModule::IndexRewrite(program, phase) => {
                self.index_rewrites.insert((program, phase), kernel);
            }
            OwnedModule::IndexRewriteRun(program) => {
                self.index_rewrite_runs.insert(program, kernel);
            }
            OwnedModule::YCbCrConvert(conversion, floats) => {
                self.ycbcr_converts.insert((conversion, floats), kernel);
            }
            OwnedModule::YCbCrSample(sampler, codes) => {
                self.ycbcr_samples.insert((sampler, codes), kernel);
            }
            OwnedModule::ChromaSample(sampler) => {
                self.chroma_samples.insert(sampler, kernel);
            }
            OwnedModule::ExtendedShadowLoad(form) => {
                self.extended_shadow_loads.insert(form, kernel);
            }
            OwnedModule::TwinDecode(decode) => {
                self.twin_decodes.insert(decode, kernel);
            }
            OwnedModule::TessellationScan(key) => {
                self.tessellation_scans.insert(key, kernel);
            }
            _ => {
                if let Some(slot) = self.slot(which) {
                    *slot = Some(kernel);
                }
            }
        }
    }

    /// The slot of a kernel the device makes once (none for the shadow copy back's, the image
    /// merge's, the shadow copy in's, the mipmap filter's, the index rewrite's and its run's, the YCbCr conversion's, the unused bits', the compressed
    /// blit's, the extended shadow load's, the twin decode's or the tessellation scan's, one
    /// per key).
    fn slot(&mut self, which: OwnedModule) -> Option<&mut Option<OwnedKernel>> {
        Some(match which {
            OwnedModule::VisibilityResolve => &mut self.visibility_resolve,
            OwnedModule::VisibilityCombine => &mut self.visibility_combine,
            OwnedModule::IndexBoundsPlan => &mut self.index_bounds_plan,
            OwnedModule::IndexBoundsFill => &mut self.index_bounds_fill,
            OwnedModule::IndexBoundsPlanRun => &mut self.index_bounds_plan_run,
            OwnedModule::IndexBoundsFillRun => &mut self.index_bounds_fill_run,
            OwnedModule::IndexGather => &mut self.index_gather,
            OwnedModule::IndexGatherPlan => &mut self.index_gather_plan,
            OwnedModule::ReplayPreprocess => &mut self.replay_preprocess,
            OwnedModule::DrawReplayPreprocess => &mut self.draw_replay_preprocess,
            OwnedModule::DrawMerge(metal2vulkan::bind::MergePhase::Blocks) => {
                &mut self.draw_merge_blocks
            }
            OwnedModule::DrawMerge(metal2vulkan::bind::MergePhase::Starts) => {
                &mut self.draw_merge_starts
            }
            OwnedModule::FlatDispatch => &mut self.flat_dispatch,
            OwnedModule::ShadowMerge(ShadowMergeStores::Words) => &mut self.shadow_merge,
            OwnedModule::ShadowMerge(ShadowMergeStores::Blocks) => &mut self.shadow_merge_blocks,
            OwnedModule::ShadowCopyBack(_)
            | OwnedModule::ShadowMergeImage(..)
            | OwnedModule::ShadowMergeKeptImage(..)
            | OwnedModule::ShadowCopyIn(_)
            | OwnedModule::MipmapFilter(_)
            | OwnedModule::IndexRewrite(..)
            | OwnedModule::IndexRewriteRun(_)
            | OwnedModule::YCbCrConvert(..)
            | OwnedModule::YCbCrSample(..)
            | OwnedModule::ChromaSample(_)
            | OwnedModule::ExtendedShadowLoad(_)
            | OwnedModule::TwinDecode(_)
            | OwnedModule::TessellationScan(_) => return None,
            OwnedModule::MeshScan => &mut self.mesh_scan,
            OwnedModule::MeshObjectArguments => &mut self.mesh_object_arguments,
            OwnedModule::MeshArguments(metal2vulkan::bind::MeshArgumentsForm::Mesh) => {
                &mut self.mesh_arguments_mesh
            }
            OwnedModule::MeshArguments(metal2vulkan::bind::MeshArgumentsForm::Object) => {
                &mut self.mesh_arguments_object
            }
            OwnedModule::MeshArguments(metal2vulkan::bind::MeshArgumentsForm::FoldedObject) => {
                &mut self.mesh_arguments_folded
            }
            OwnedModule::PlaneWords => &mut self.plane_words,
            OwnedModule::PlaneCopy => &mut self.plane_copy,
            OwnedModule::RotateTexels => &mut self.rotate_texels,
            OwnedModule::RowsCopy => &mut self.rows_copy,
            OwnedModule::YCbCrRefresh => &mut self.ycbcr_refresh,

            OwnedModule::TwoPlaneTexels => &mut self.two_plane_texels,
            OwnedModule::PlaneBytesCopy => &mut self.plane_bytes_copy,
            OwnedModule::YCbCrWrite => &mut self.ycbcr_write,
            OwnedModule::TiledTexels => &mut self.tiled_texels,
            OwnedModule::PredicationStep => &mut self.predication_step,
        })
    }
}

impl Device {
    /// The device's Y′CbCr conversion of `sampler` that its YCbCr sample kernels bake, where a
    /// kernel of it was made ([`OwnedKernels::ycbcr_sampler`]): an image view such a kernel
    /// samples is made with it.
    pub(super) fn ycbcr_conversion(
        &self,
        sampler: YCbCrSampler,
    ) -> Option<vk::SamplerYcbcrConversion> {
        let kernels = lock(&self.shared().owned_kernels);
        let made = kernels.ycbcr_samplers.get(&sampler);
        made.map(|&(conversion, _)| conversion)
    }

    /// The device's Y′CbCr conversion of `sampler` ([`OwnedKernels::ycbcr_sampler`]), made on
    /// first use: a planes image's view (`texture/ycbcr_planes.rs`) and every immutable sampler
    /// of it a pipeline's layout bakes (`pipeline/objects.rs`'s `sampler_set`) are made with this
    /// one conversion, so each sampler's `VkSamplerYcbcrConversionInfo` is identically defined to
    /// the view's (`VUID-VkWriteDescriptorSet-descriptorType-02738`).
    pub(super) fn made_ycbcr_conversion(
        &self,
        sampler: YCbCrSampler,
    ) -> Result<vk::SamplerYcbcrConversion, Error> {
        let mut kernels = lock(&self.shared().owned_kernels);
        let (conversion, _) = kernels.ycbcr_sampler(self, sampler)?;
        Ok(conversion)
    }

    /// The kernel of `which`, made on first use.
    pub(super) fn owned_kernel(&self, which: OwnedModule) -> Result<OwnedKernel, Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().owned_kernels, |kernels| {
            kernels.get(which).is_some()
        });
        let mut kernels = lock(&self.shared().owned_kernels);
        if let Some(kernel) = kernels.get(which) {
            return Ok(kernel);
        }
        let (spirv, buffers, push_bytes) = which.build(self)?;
        let bindings = (which.images(), which.storage_images(), buffers);
        let baked = match which {
            OwnedModule::YCbCrSample(sampler, ..) | OwnedModule::ChromaSample(sampler) => {
                let (_, made) = kernels.ycbcr_sampler(self, sampler)?;
                Some((made, sampler.descriptors))
            }
            _ => None,
        };
        let kernel = make(
            self,
            &spirv,
            bindings,
            (push_bytes, which.pushed(self)),
            baked,
        )?;
        // Counted here, so the build attributes to the call that first needed the kernel rather
        // than to whichever path the kernel serves (`WorkCounts::owned_kernels_made`).
        self.shared().counts.owned_kernel_made(spirv.len());
        kernels.put(which, kernel);
        Ok(kernel)
    }
}

/// The objects of a kernel of `spirv` (entry point `main`) with `images` sampled images, then
/// `storage_images` storage images and then `buffers` storage buffers in set 0, a push descriptor
/// set's where `pushed`, and `push_bytes` of `COMPUTE` push constants. With `baked`, a sampler
/// of a Y′CbCr conversion and the descriptors a combined image sampler of it takes, its first
/// binding is that combined image sampler, the sampler immutable.
fn make(
    device: &Device,
    spirv: &[u32],
    (images, storage_images, buffers): (u32, u32, u32),
    (push_bytes, pushed): (u32, bool),
    baked: Option<(vk::Sampler, u32)>,
) -> Result<OwnedKernel, Error> {
    let shared = device.shared();
    let raw = shared.vk();
    let (sampler, combined) = baked.unwrap_or((vk::Sampler::null(), 0));
    let combined =
        u8::try_from(combined).map_err(|_| internal("a YCbCr format of too many planes"))?;
    let mut kernel = OwnedKernel {
        module: vk::ShaderModule::null(),
        set_layout: vk::DescriptorSetLayout::null(),
        layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
        images,
        storage_images,
        buffers,
        pushed,
        combined,
    };
    let immutable = [sampler];
    let made = (|| {
        let info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the code is live SPIR-V words for the call.
        kernel.module = unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)?;
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> =
            (0..images + storage_images + buffers)
                .map(|binding| {
                    let info = vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(kernel.descriptor_type(binding))
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE);
                    match kernel.baked(binding) {
                        true => info.immutable_samplers(&immutable),
                        false => info,
                    }
                })
                .collect();
        let mut info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        if pushed {
            info = info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
        }
        // SAFETY: `info` and its array are live for the call.
        kernel.set_layout =
            unsafe { raw.create_descriptor_set_layout(&info, None) }.map_err(vulkan)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_bytes)];
        let set_layouts = [kernel.set_layout];
        let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if push_bytes > 0 {
            info = info.push_constant_ranges(&push);
        }
        // SAFETY: as above.
        kernel.layout = unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(kernel.module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(kernel.layout);
        let created = shared
            .make_pipelines(
                Making::new("owned kernel").words(spirv).text(&(
                    images,
                    storage_images,
                    buffers,
                    push_bytes,
                    pushed,
                )),
                |cache| {
                    // SAFETY: every pointer in `info` outlives the call; the pipeline cache is the
                    // making's own (`shader_cache/driver.rs`).
                    unsafe { raw.create_compute_pipelines(cache, &[info], None) }
                },
            )
            .map_err(|(_, error)| vulkan(error))?;
        kernel.pipeline = created
            .first()
            .copied()
            .ok_or_else(|| internal("the driver made no pipeline"))?;
        Ok(())
    })();
    match made {
        Ok(()) => Ok(kernel),
        Err(error) => {
            kernel.destroy(raw);
            Err(error)
        }
    }
}

/// The type of binding `binding` of a set whose first `images` bindings are sampled images, the
/// next `storage_images` storage images, and the rest storage buffers.
fn descriptor_type(binding: u32, images: u32, storage_images: u32) -> vk::DescriptorType {
    if binding < images {
        vk::DescriptorType::SAMPLED_IMAGE
    } else if binding < images + storage_images {
        vk::DescriptorType::STORAGE_IMAGE
    } else {
        vk::DescriptorType::STORAGE_BUFFER
    }
}

impl OwnedKernel {
    /// Whether binding `binding` of its set 0 is the combined image sampler of a baked Y′CbCr
    /// conversion's sampler.
    fn baked(&self, binding: u32) -> bool {
        binding == 0 && self.combined > 0
    }

    /// The type of binding `binding` of its set 0: [`descriptor_type`]'s, but the combined image
    /// sampler of a baked sampler.
    fn descriptor_type(&self, binding: u32) -> vk::DescriptorType {
        match self.baked(binding) {
            true => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            false => descriptor_type(binding, self.images, self.storage_images),
        }
    }
}

/// What one set of an [`OwnedKernel`] binds: its sampled images' views, then its storage images'
/// views, each in the `GENERAL` layout, then its storage buffers' `(buffer, offset, range)`.
pub(super) struct OwnedSet {
    pub(super) images: Vec<vk::ImageView>,
    pub(super) storage_images: Vec<vk::ImageView>,
    pub(super) buffers: Vec<(vk::Buffer, u64, u64)>,
}

/// The descriptor sets of one command's dispatches of its kernels, from pools of the device's
/// spares (the module docs).
pub(super) struct OwnedSets {
    arena: DescriptorArena,
    pub(super) sets: Vec<vk::DescriptorSet>,
}

impl Drop for OwnedSets {
    fn drop(&mut self) {
        // No pending command uses the sets (a command buffer retains them until it completes):
        // their pools go back to the device's spares, which reset a pool before it serves again.
        self.arena.give_back();
    }
}

impl OwnedKernel {
    /// What one set of its set 0 takes from a pool: its sampled images, storage images and
    /// storage buffers (`SetNeeds`, by raw type ascending, each count above 0).
    fn set_needs(&self) -> SetNeeds {
        let baked = u32::from(self.combined > 0);
        let counts = [
            (
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                u32::from(self.combined),
            ),
            (vk::DescriptorType::SAMPLED_IMAGE, self.images - baked),
            (vk::DescriptorType::STORAGE_IMAGE, self.storage_images),
            (vk::DescriptorType::STORAGE_BUFFER, self.buffers),
        ];
        SetNeeds {
            counts: (counts.into_iter())
                .filter(|&(_, count)| count > 0)
                .map(|(ty, count)| (ty.as_raw(), count))
                .collect(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
        }
    }
}

impl OwnedSets {
    /// One set of `kernel` (of storage buffers only) per entry of `ranges`, each binding `i`
    /// written with the `i`-th `(buffer, offset, range)`.
    pub(super) fn write(
        device: &Device,
        kernel: &OwnedKernel,
        ranges: &[Vec<(vk::Buffer, u64, u64)>],
    ) -> Result<OwnedSets, Error> {
        let sets: Vec<OwnedSet> = ranges
            .iter()
            .map(|buffers| OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: buffers.clone(),
            })
            .collect();
        OwnedSets::write_each(device, &vec![*kernel; ranges.len()], &sets)
    }

    /// One set per entry of `sets`, of the kernel at its index in `kernels`, each written with
    /// its entry's views and then its buffer ranges, in binding order. Refused (internal) for a
    /// kernel whose set 0 is pushed, which no set is allocated from (the module docs): its
    /// dispatches bind through [`OwnedBinds`].
    pub(super) fn write_each(
        device: &Device,
        kernels: &[OwnedKernel],
        sets: &[OwnedSet],
    ) -> Result<OwnedSets, Error> {
        let raw = device.shared().vk();
        let count = u32::try_from(sets.len()).map_err(|_| internal("too many owned sets"))?;
        if kernels.len() != sets.len()
            || sets.iter().zip(kernels).any(|(set, kernel)| {
                set.images.len() != kernel.images as usize
                    || set.storage_images.len() != kernel.storage_images as usize
                    || set.buffers.len() != kernel.buffers as usize
            })
            || count == 0
        {
            return Err(internal("owned sets that do not match their kernel"));
        }
        if kernels.iter().any(|kernel| kernel.pushed) {
            return Err(internal("owned sets of a kernel whose set 0 is pushed"));
        }
        device.shared().counts.owned_descriptor_pool();
        let mut owned = OwnedSets {
            arena: DescriptorArena::fitted(),
            sets: Vec::with_capacity(count as usize),
        };
        for kernel in kernels {
            let set = (owned.arena).allocate(device, kernel.set_layout, &kernel.set_needs())?;
            owned.sets.push(set);
        }
        let image_info = |view: &vk::ImageView| {
            vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        let image_infos: Vec<Vec<vk::DescriptorImageInfo>> = sets
            .iter()
            .map(|set| set.images.iter().map(image_info).collect())
            .collect();
        let storage_infos: Vec<Vec<vk::DescriptorImageInfo>> = sets
            .iter()
            .map(|set| set.storage_images.iter().map(image_info).collect())
            .collect();
        let buffer_infos: Vec<Vec<vk::DescriptorBufferInfo>> = sets
            .iter()
            .map(|set| {
                set.buffers
                    .iter()
                    .map(|&(buffer, offset, range)| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer)
                            .offset(offset)
                            .range(range)
                    })
                    .collect()
            })
            .collect();
        let mut writes = Vec::new();
        let infos = image_infos.iter().zip(&storage_infos).zip(&buffer_infos);
        for ((set, ((images, storage), buffers)), kernel) in
            owned.sets.iter().zip(infos).zip(kernels)
        {
            for (binding, info) in images.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(binding as u32)
                        .descriptor_type(kernel.descriptor_type(binding as u32))
                        .image_info(std::slice::from_ref(info)),
                );
            }
            for (binding, info) in (images.len()..).zip(storage) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info)),
                );
            }
            for (binding, info) in (images.len() + storage.len()..).zip(buffers) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info)),
                );
            }
        }
        // SAFETY: the sets were just allocated and no command uses them; every view and buffer
        // written is live (the caller's command retains them).
        unsafe { raw.update_descriptor_sets(&writes, &[]) };
        Ok(owned)
    }
}

/// How one command's dispatches of a kernel bind set 0 (the module docs): pushed as each
/// records where the kernel's set 0 is a push descriptor set's ([`OwnedKernel::pushed`]), so the
/// command makes no pool; else one set per dispatch from a pool of their own ([`OwnedSets`]),
/// which the command buffer retains with the command until it completes.
pub(super) enum OwnedBinds {
    /// Each dispatch's writes, in binding order, pushed by the push command of the checked
    /// functions it records with (`Vk::push_descriptor`).
    Pushed(Vec<Vec<SetWrite>>),
    /// One set per dispatch.
    Sets(OwnedSets),
}

impl OwnedBinds {
    /// The binds of `kernel`'s dispatches on `device`, one per entry of `sets`: each its entry's
    /// sampled images, storage images (each in the `GENERAL` layout) and storage buffers, in
    /// binding order.
    pub(super) fn new(
        device: &Device,
        kernel: &OwnedKernel,
        sets: &[OwnedSet],
    ) -> Result<OwnedBinds, Error> {
        if !kernel.pushed {
            let kernels = vec![*kernel; sets.len()];
            return OwnedSets::write_each(device, &kernels, sets).map(OwnedBinds::Sets);
        }
        let image = |view: &vk::ImageView| {
            Info::Image(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: *view,
                image_layout: vk::ImageLayout::GENERAL,
            })
        };
        let mut pushed = Vec::with_capacity(sets.len());
        for set in sets {
            if set.images.len() != kernel.images as usize
                || set.storage_images.len() != kernel.storage_images as usize
                || set.buffers.len() != kernel.buffers as usize
            {
                return Err(internal("owned binds that do not match their kernel"));
            }
            let images = set
                .images
                .iter()
                .map(|view| (vk::DescriptorType::SAMPLED_IMAGE, image(view)));
            let storage = set
                .storage_images
                .iter()
                .map(|view| (vk::DescriptorType::STORAGE_IMAGE, image(view)));
            let buffers = set.buffers.iter().map(|&(buffer, offset, range)| {
                let info = vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range,
                };
                (vk::DescriptorType::STORAGE_BUFFER, Info::Buffer(info))
            });
            let writes = (0..)
                .zip(images.chain(storage).chain(buffers))
                .map(|(binding, (ty, info))| SetWrite {
                    binding,
                    element: 0,
                    ty,
                    info,
                })
                .collect();
            pushed.push(writes);
        }
        Ok(OwnedBinds::Pushed(pushed))
    }

    /// Bind dispatch `index`'s set 0 of `kernel` into `command_buffer`, whose compute pipeline is
    /// `kernel`'s.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording on the device of `device`, which made `kernel`; every view
    /// and buffer the binds name is live until it completes, and `index` names one of them.
    pub(super) unsafe fn bind(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        kernel: &OwnedKernel,
        index: usize,
    ) {
        let point = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's contract.
        unsafe {
            match (self, device.push_descriptor()) {
                (OwnedBinds::Pushed(writes), Some(push)) => {
                    binding::push_set(push, command_buffer, point, kernel.layout, &writes[index]);
                }
                // A kernel made pushed on a device that pushes (`OwnedKernel::pushed`), so not
                // reached: a push binds nothing here.
                (OwnedBinds::Pushed(_), None) => {}
                (OwnedBinds::Sets(sets), _) => {
                    let set = std::slice::from_ref(&sets.sets[index]);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        kernel.layout,
                        0,
                        set,
                        &[],
                    );
                }
            }
        }
    }
}
