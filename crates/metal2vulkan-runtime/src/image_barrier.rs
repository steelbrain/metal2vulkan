//! The reads the runtime's barriers name in their destination, from one table of the reads a
//! queue family's commands make ([`READS`], and [`EXTENSION_READS`] for what an enabled
//! extension adds), never `MEMORY_READ` or `HOST_READ` where they can be named: an image memory
//! barrier's, all made by [`image_barrier`] for the queue family that records them, whose
//! destination accesses are an [`ImageDst`]; and a global memory barrier's
//! ([`MemoryReads`], which `tracker::Barrier::record` names in place of `MEMORY_READ`).
//!
//! - **The device's accesses alone.** A command reads an image as a shader resource, an input,
//!   color or depth/stencil attachment, or a transfer's source; that is everything `MEMORY_READ`
//!   names after an image barrier whose destination stage is `ALL_COMMANDS` ("all read accesses ...
//!   valid where it is used"), and [`ImageDst::Every`] names exactly that. The host reaches no
//!   image's memory: it moves texels by copies, and a heap buffer's bytes reach the host through
//!   the global barrier that ends every batch (`tracker::Barrier::to_host` or `to_all`) or an
//!   internal submission (`texture::transfer_barrier`), which no image barrier stands for.
//! - **The recording family's reads.** `ALL_COMMANDS` stands for the stages the recording queue
//!   family supports, and an access mask may name only accesses those stages make
//!   (VUID-vkCmdPipelineBarrier-pImageMemoryBarriers-02820). So [`ImageDst::Every`] is the reads
//!   of the family's commands: a transfer's read on a family of transfers alone (the staging
//!   queue's, `device/staging.rs`), and a shader's read too on one of compute, and the attachment
//!   reads too on one of graphics. That is exact: on a family of transfers alone the only reads of
//!   an image after the barrier are transfers. A later access on another queue is not this
//!   barrier's to order: a semaphore wait orders it, whose "second access scope includes all
//!   memory access performed by the device" (the Vulkan specification, Semaphore Waiting), or,
//!   after the staging queue's work, the host's wait for its fence before that access is
//!   submitted.
//! - **Why not `MEMORY_READ`.** MoltenVK 1.4.1 takes `MEMORY_READ` or `HOST_READ` in the
//!   destination of a barrier into `GENERAL` of an optimal image bound to mapped host-coherent
//!   memory as a host read to come. Once the command buffer completes, a completed handler copies
//!   the texture into that memory (`getBytes`, `MVKImagePlane::pullFromDeviceOnCompletion`). A
//!   texture placed in its memory's `MTLHeap` lies in that same memory, so the copy writes its
//!   texels, laid out in rows, over its own storage, which the GPU lays out otherwise. The copy
//!   runs on the CPU while the GPU runs the submission's later command buffers. On an M3 Max,
//!   textures of a `Shared` placement heap read back permuted texels, or the heap's earlier bytes,
//!   where the copy came after the copies that wrote them (`runtime_lane::heap_textures`), with a
//!   texture's first move (`texture/first_move.rs`) asking for host reads. Every texture's first
//!   move, and every transition of a caller's image (`texture/wrapped.rs`), is such a barrier.
//! - **Global memory barriers.** A global barrier orders every read of memory a later command
//!   makes at its destination stages: an image's reads above, and those of a buffer alone, an
//!   indirect command's, an index's, a vertex attribute's and a uniform's, at stages of the family
//!   that make them; and the reads an enabled extension makes valid there ([`EXTENSION_READS`]:
//!   each extension of the Vulkan registry, vk.xml of Vulkan-Headers `vulkan-sdk-1.4.341.0`, the
//!   registry `scripts/regen-vulkan-requirements` pins, that adds a read access type or a stage a
//!   read takes). Named one by one beside the barrier's destination stages ([`MemoryReads`]) they
//!   are `MEMORY_READ` exactly: it "is treated as equivalent to setting all READ access flags that
//!   are valid where it is used", where an access is valid beside a destination stage mask only
//!   if a stage in the mask supports it, as the table of supported access types specifies
//!   (`VUID-vkCmdPipelineBarrier-dstAccessMask-02816`), and `ALL_COMMANDS` "specifies all
//!   operations performed by all commands supported on the queue it is used with", while the host stage "is not invoked by any
//!   commands recorded in a command buffer" (the Vulkan specification, Access Types and Pipeline
//!   Stages), so `HOST_READ` is none of them. Where an enabled extension makes valid a read only `VkAccessFlags2` names, which no
//!   `VkAccessFlags` bit covers (the runtime's barriers are `vkCmdPipelineBarrier`'s), the barrier
//!   keeps `MEMORY_READ`, and so does one on a family of none of graphics, compute and transfers.
//!   The extensions are those the runtime knows the device enabled: its own, and an adopted
//!   device's `RawDevice::enabled` and `device_extensions`. They are every extension whose
//!   commands the runtime records, and its barriers order its own commands: a batch's work reaches
//!   other commands through its signals, whose semaphore's "second access scope includes all
//!   memory access performed by the device" (the Vulkan specification, Semaphore Waiting). The
//!   runtime's global barriers record on its queues for encoders and internal work, every one of
//!   a family of graphics and compute (`device/create.rs`'s `queue_plan`), so their reads are
//!   that family's.
//! - **Why not `MEMORY_READ` there.** ANV (Mesa 26.2.4) takes `MEMORY_READ` in a barrier's
//!   destination as every invalidate it has, the aux-table invalidate among them; on the Iris Xe of
//!   RPL-P (Gfx12) that adds the render engine's idle sequence (a data cache, L3 fabric, render
//!   target and depth flush with a command streamer stall), a write of the aux-table invalidate
//!   register and a wait polling it clear, and the state and instruction caches' invalidates
//!   (`genX_cmd_buffer.c`'s `emit_apply_pipe_flushes` and `invalidate_aux_map`), at every such
//!   barrier. The named reads take none of them.

use ash::vk;
use metal2vulkan::vk::PipelineStageFlags2 as Stage;
use std::ffi::CStr;

/// The queue flags of the stages a read takes (vk.xml's `syncstage` queues).
const GRAPHICS: vk::QueueFlags = vk::QueueFlags::GRAPHICS;
const SHADERS: vk::QueueFlags =
    vk::QueueFlags::from_raw(vk::QueueFlags::GRAPHICS.as_raw() | vk::QueueFlags::COMPUTE.as_raw());
const TRANSFERS: vk::QueueFlags =
    vk::QueueFlags::from_raw(SHADERS.as_raw() | vk::QueueFlags::TRANSFER.as_raw());

/// The bits of `stages`.
const fn stages(stages: &[Stage]) -> u64 {
    let mut bits = 0;
    let mut i = 0;
    while i < stages.len() {
        bits |= stages[i].as_raw();
        i += 1;
    }
    bits
}

/// Every shader stage: those `SHADER_READ` and `UNIFORM_READ` take, and the reads of descriptors
/// and acceleration structures in shaders.
const SHADER_STAGES: u64 = stages(&[
    Stage::VERTEX_SHADER,
    Stage::TESSELLATION_CONTROL_SHADER,
    Stage::TESSELLATION_EVALUATION_SHADER,
    Stage::GEOMETRY_SHADER,
    Stage::FRAGMENT_SHADER,
    Stage::COMPUTE_SHADER,
    Stage::RAY_TRACING_SHADER_KHR,
    Stage::TASK_SHADER_EXT,
    Stage::MESH_SHADER_EXT,
    Stage::SUBPASS_SHADER_HUAWEI,
    Stage::CLUSTER_CULLING_SHADER_HUAWEI,
]);

/// The stages each stage that stands for others is "equivalent to" (vk.xml's `syncequivalent`).
const EQUIVALENTS: [(Stage, u64); 4] = [
    (
        Stage::VERTEX_INPUT,
        stages(&[Stage::INDEX_INPUT, Stage::VERTEX_ATTRIBUTE_INPUT]),
    ),
    (
        Stage::ALL_TRANSFER,
        stages(&[
            Stage::COPY,
            Stage::BLIT,
            Stage::RESOLVE,
            Stage::CLEAR,
            Stage::ACCELERATION_STRUCTURE_COPY_KHR,
        ]),
    ),
    (
        Stage::ALL_GRAPHICS,
        stages(&[
            Stage::DRAW_INDIRECT,
            Stage::COPY_INDIRECT_KHR,
            Stage::TASK_SHADER_EXT,
            Stage::MESH_SHADER_EXT,
            Stage::VERTEX_INPUT,
            Stage::INDEX_INPUT,
            Stage::VERTEX_ATTRIBUTE_INPUT,
            Stage::VERTEX_SHADER,
            Stage::TESSELLATION_CONTROL_SHADER,
            Stage::TESSELLATION_EVALUATION_SHADER,
            Stage::GEOMETRY_SHADER,
            Stage::FRAGMENT_SHADER,
            Stage::EARLY_FRAGMENT_TESTS,
            Stage::LATE_FRAGMENT_TESTS,
            Stage::COLOR_ATTACHMENT_OUTPUT,
            Stage::CONDITIONAL_RENDERING_EXT,
            Stage::TRANSFORM_FEEDBACK_EXT,
            Stage::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
            Stage::FRAGMENT_DENSITY_PROCESS_EXT,
            Stage::SUBPASS_SHADER_HUAWEI,
            Stage::INVOCATION_MASK_HUAWEI,
            Stage::CLUSTER_CULLING_SHADER_HUAWEI,
        ]),
    ),
    (
        Stage::PRE_RASTERIZATION_SHADERS,
        stages(&[
            Stage::VERTEX_SHADER,
            Stage::TESSELLATION_CONTROL_SHADER,
            Stage::TESSELLATION_EVALUATION_SHADER,
            Stage::GEOMETRY_SHADER,
            Stage::TASK_SHADER_EXT,
            Stage::MESH_SHADER_EXT,
            Stage::CLUSTER_CULLING_SHADER_HUAWEI,
        ]),
    ),
];

/// `stages` with each stage that stands for others widened to them.
fn expanded(stages: u64) -> u64 {
    (EQUIVALENTS.iter())
        .filter(|(stage, _)| stages & stage.as_raw() != 0)
        .fold(stages, |all, (_, equivalent)| all | equivalent)
}

/// One read access type a command may make: its bit, where `VkAccessFlags` has one (`None` where
/// only `VkAccessFlags2` does); the queue flags of a family whose stages make it, any of which
/// the family has; the stages that make it (vk.xml's `syncsupport`), as `VkPipelineStageFlags2`
/// bits; and whether an image barrier names it ([`ImageDst::Every`]): a core read of an image.
/// An image barrier names no read an extension adds; of those, only `VK_QCOM_tile_shading`'s
/// reads an image (a tile attachment), which `SHADER_READ` covers.
#[derive(Clone, Copy)]
pub(crate) struct Read {
    pub(crate) access: Option<vk::AccessFlags>,
    pub(crate) queues: vk::QueueFlags,
    pub(crate) stages: u64,
    pub(crate) image: bool,
}

impl Read {
    /// Whether the commands of a family of `family`'s queue flags make it.
    pub(crate) fn held_by(self, family: vk::QueueFlags) -> bool {
        family.intersects(self.queues)
    }

    /// Whether it is valid in a destination access mask beside the stages `dst` on a family of
    /// `family`'s queue flags: `ALL_COMMANDS` holds every stage the family runs; any other mask,
    /// which holds only stages the family runs and the device has
    /// (`VUID-vkCmdPipelineBarrier-dstStageMask-04098`, `-04090` and the like), those it names
    /// and those they stand for.
    fn valid_beside(self, dst: vk::PipelineStageFlags, family: vk::QueueFlags) -> bool {
        if dst.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
            return self.held_by(family);
        }
        expanded(u64::from(dst.as_raw())) & self.stages != 0
    }
}

const fn read(access: vk::AccessFlags, queues: vk::QueueFlags, stages: u64, image: bool) -> Read {
    Read {
        access: Some(access),
        queues,
        stages,
        image,
    }
}

/// A read only `VkAccessFlags2` names, which no `VkAccessFlags` bit covers.
const fn unnamed(queues: vk::QueueFlags, stages: u64) -> Read {
    Read {
        access: None,
        queues,
        stages,
        image: false,
    }
}

/// Every core read access type, the families whose stages make it and the stages (vk.xml's
/// `sync` section, the Vulkan specification's "Supported access types" table): an extension's
/// stage among them is one a mask holds only where the device has it. `SHADER_READ` "is
/// equivalent to the logical OR of" the sampled, storage, shader binding table and tile
/// attachment reads `VkAccessFlags2` names apart (the specification's `VkAccessFlagBits2`).
pub(crate) const READS: [Read; 9] = [
    read(
        vk::AccessFlags::TRANSFER_READ,
        TRANSFERS,
        stages(&[
            Stage::ALL_TRANSFER,
            Stage::COPY,
            Stage::RESOLVE,
            Stage::BLIT,
            Stage::ACCELERATION_STRUCTURE_BUILD_KHR,
            Stage::ACCELERATION_STRUCTURE_COPY_KHR,
            Stage::MICROMAP_BUILD_EXT,
            Stage::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV,
        ]),
        true,
    ),
    read(
        vk::AccessFlags::SHADER_READ,
        SHADERS,
        SHADER_STAGES
            | stages(&[
                Stage::ACCELERATION_STRUCTURE_BUILD_KHR,
                Stage::MICROMAP_BUILD_EXT,
            ]),
        true,
    ),
    read(vk::AccessFlags::UNIFORM_READ, SHADERS, SHADER_STAGES, false),
    read(
        vk::AccessFlags::INDIRECT_COMMAND_READ,
        SHADERS,
        stages(&[
            Stage::DRAW_INDIRECT,
            Stage::ACCELERATION_STRUCTURE_BUILD_KHR,
        ]),
        false,
    ),
    read(
        vk::AccessFlags::INPUT_ATTACHMENT_READ,
        GRAPHICS,
        stages(&[Stage::FRAGMENT_SHADER, Stage::SUBPASS_SHADER_HUAWEI]),
        true,
    ),
    read(
        vk::AccessFlags::COLOR_ATTACHMENT_READ,
        GRAPHICS,
        stages(&[Stage::FRAGMENT_SHADER, Stage::COLOR_ATTACHMENT_OUTPUT]),
        true,
    ),
    read(
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        GRAPHICS,
        stages(&[
            Stage::FRAGMENT_SHADER,
            Stage::EARLY_FRAGMENT_TESTS,
            Stage::LATE_FRAGMENT_TESTS,
        ]),
        true,
    ),
    read(
        vk::AccessFlags::INDEX_READ,
        GRAPHICS,
        stages(&[Stage::VERTEX_INPUT, Stage::INDEX_INPUT]),
        false,
    ),
    read(
        vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        GRAPHICS,
        stages(&[Stage::VERTEX_INPUT, Stage::VERTEX_ATTRIBUTE_INPUT]),
        false,
    ),
];

/// An acceleration structure's read in shaders and builds.
const STRUCTURE_READ: Read = read(
    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
    SHADERS,
    SHADER_STAGES | Stage::ACCELERATION_STRUCTURE_BUILD_KHR.as_raw(),
    false,
);

/// Each extension of the Vulkan registry whose enabling makes a read access type valid with a
/// stage where the core's is not (vk.xml of Vulkan-Headers `vulkan-sdk-1.4.341.0`, its `sync`
/// section: a read it adds, or a stage it adds that a read takes), that read, and those stages.
/// A read the sampled, storage, shader binding table or tile attachment read stands for is
/// `SHADER_READ`'s, which [`READS`] names with every shader stage.
pub(crate) const EXTENSION_READS: [(&CStr, Read); 21] = [
    (c"VK_KHR_acceleration_structure", STRUCTURE_READ),
    (c"VK_NV_ray_tracing", STRUCTURE_READ),
    // Its acceleration structure copy stage runs on families of transfers alone too; it requires
    // `VK_KHR_acceleration_structure`, whose read that is.
    (
        c"VK_KHR_ray_tracing_maintenance1",
        read(
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            TRANSFERS,
            Stage::ACCELERATION_STRUCTURE_COPY_KHR.as_raw(),
            false,
        ),
    ),
    // Its indirect copy stage runs on families of transfers alone too.
    (
        c"VK_KHR_copy_memory_indirect",
        read(
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            TRANSFERS,
            Stage::COPY_INDIRECT_KHR.as_raw(),
            false,
        ),
    ),
    (
        c"VK_EXT_device_generated_commands",
        read(
            vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
            SHADERS,
            Stage::COMMAND_PREPROCESS_EXT.as_raw(),
            false,
        ),
    ),
    (
        c"VK_NV_device_generated_commands",
        read(
            vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
            SHADERS,
            Stage::COMMAND_PREPROCESS_EXT.as_raw(),
            false,
        ),
    ),
    (
        c"VK_EXT_conditional_rendering",
        read(
            vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT,
            SHADERS,
            Stage::CONDITIONAL_RENDERING_EXT.as_raw(),
            false,
        ),
    ),
    (
        c"VK_EXT_transform_feedback",
        read(
            vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
            SHADERS,
            stages(&[Stage::DRAW_INDIRECT, Stage::TRANSFORM_FEEDBACK_EXT]),
            false,
        ),
    ),
    (
        c"VK_EXT_blend_operation_advanced",
        read(
            vk::AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
            GRAPHICS,
            Stage::COLOR_ATTACHMENT_OUTPUT.as_raw(),
            false,
        ),
    ),
    (
        c"VK_EXT_fragment_density_map",
        read(
            vk::AccessFlags::FRAGMENT_DENSITY_MAP_READ_EXT,
            GRAPHICS,
            Stage::FRAGMENT_DENSITY_PROCESS_EXT.as_raw(),
            false,
        ),
    ),
    (
        c"VK_KHR_fragment_shading_rate",
        read(
            vk::AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR,
            GRAPHICS,
            Stage::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw(),
            false,
        ),
    ),
    (
        c"VK_NV_shading_rate_image",
        read(
            vk::AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR,
            GRAPHICS,
            Stage::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw(),
            false,
        ),
    ),
    (c"VK_EXT_descriptor_buffer", unnamed(SHADERS, SHADER_STAGES)),
    (c"VK_EXT_descriptor_heap", unnamed(SHADERS, SHADER_STAGES)),
    (
        c"VK_EXT_opacity_micromap",
        unnamed(
            vk::QueueFlags::COMPUTE,
            stages(&[
                Stage::MICROMAP_BUILD_EXT,
                Stage::ACCELERATION_STRUCTURE_BUILD_KHR,
            ]),
        ),
    ),
    (
        c"VK_HUAWEI_invocation_mask",
        unnamed(GRAPHICS, Stage::INVOCATION_MASK_HUAWEI.as_raw()),
    ),
    (
        c"VK_EXT_memory_decompression",
        unnamed(SHADERS, Stage::MEMORY_DECOMPRESSION_EXT.as_raw()),
    ),
    (
        c"VK_KHR_video_decode_queue",
        unnamed(
            vk::QueueFlags::VIDEO_DECODE_KHR,
            Stage::VIDEO_DECODE_KHR.as_raw(),
        ),
    ),
    (
        c"VK_KHR_video_encode_queue",
        unnamed(
            vk::QueueFlags::VIDEO_ENCODE_KHR,
            Stage::VIDEO_ENCODE_KHR.as_raw(),
        ),
    ),
    (
        c"VK_NV_optical_flow",
        unnamed(
            vk::QueueFlags::OPTICAL_FLOW_NV,
            Stage::OPTICAL_FLOW_NV.as_raw(),
        ),
    ),
    // `VK_QUEUE_DATA_GRAPH_BIT_ARM`, which `ash 0.38` predates.
    (
        c"VK_ARM_data_graph",
        unnamed(
            vk::QueueFlags::from_raw(0x400),
            Stage::DATA_GRAPH_ARM.as_raw(),
        ),
    ),
];

/// The reads of an image the commands of a family of `family`'s queue flags make.
fn image_reads(family: vk::QueueFlags) -> vk::AccessFlags {
    (READS.iter())
        .filter(|read| read.image && read.held_by(family))
        .filter_map(|read| read.access)
        .fold(vk::AccessFlags::empty(), |reads, access| reads | access)
}

/// What a global memory barrier's destination names in place of `MEMORY_READ` on a device's
/// queue family for encoders (the module docs): its family, and which of [`EXTENSION_READS`] the
/// device enabled.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryReads {
    family: vk::QueueFlags,
    /// Bit `i` for `EXTENSION_READS[i]`.
    enabled: u32,
}

impl MemoryReads {
    /// The reads of a family of `family`'s queue flags on a device that enabled the extensions
    /// `enabled`.
    pub(crate) fn of<'a>(
        family: vk::QueueFlags,
        enabled: impl IntoIterator<Item = &'a CStr>,
    ) -> MemoryReads {
        let mut rows = 0;
        for name in enabled {
            for (i, (extension, _)) in EXTENSION_READS.iter().enumerate() {
                if *extension == name {
                    rows |= 1 << i;
                }
            }
        }
        MemoryReads {
            family,
            enabled: rows,
        }
    }

    /// The reads of a family of `family`'s queue flags on the same device: a queue of another
    /// family's (`DeviceShared::memory_reads_of`).
    pub(crate) fn for_family(self, family: vk::QueueFlags) -> MemoryReads {
        MemoryReads { family, ..self }
    }

    /// The destination access mask `dst` beside the destination stages `stages` with
    /// `MEMORY_READ` in it replaced by the reads it stands for there: every read valid beside
    /// those stages on the family, which the device has. `dst` itself where one of them has no
    /// `VkAccessFlags` bit, and on a family of none of graphics, compute and transfers.
    pub(crate) fn named(
        self,
        dst: vk::AccessFlags,
        stages: vk::PipelineStageFlags,
    ) -> vk::AccessFlags {
        if !dst.contains(vk::AccessFlags::MEMORY_READ) || !self.family.intersects(TRANSFERS) {
            return dst;
        }
        let added = (EXTENSION_READS.iter().enumerate())
            .filter(|(i, _)| self.enabled & (1 << i) != 0)
            .map(|(_, (_, read))| read);
        let mut reads = vk::AccessFlags::empty();
        for read in READS.iter().chain(added) {
            if !read.valid_beside(stages, self.family) {
                continue;
            }
            match read.access {
                Some(access) => reads |= access,
                None => return dst,
            }
        }
        (dst & !vk::AccessFlags::MEMORY_READ) | reads
    }
}

/// The accesses after an image barrier that it makes the image's earlier writes visible to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageDst {
    /// Every access a later command of the recording family may make of the image, its reads and
    /// its writes.
    Every,
    /// A transfer's read of it.
    TransferRead,
    /// A transfer's write of it.
    TransferWrite,
}

impl ImageDst {
    /// Its access mask in a barrier recorded for a family of `family`'s queue flags.
    pub(crate) fn access(self, family: vk::QueueFlags) -> vk::AccessFlags {
        match self {
            ImageDst::Every => image_reads(family) | vk::AccessFlags::MEMORY_WRITE,
            ImageDst::TransferRead => vk::AccessFlags::TRANSFER_READ,
            ImageDst::TransferWrite => vk::AccessFlags::TRANSFER_WRITE,
        }
    }
}

/// The barrier of `range` of `image` from `old` to `new` (the same layout for no move), after
/// the `src` accesses before it and before the `dst` accesses after it, recorded for a queue
/// family of `family`'s queue flags, of no queue family transfer.
pub(crate) fn image_barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    (old, new): (vk::ImageLayout, vk::ImageLayout),
    src: vk::AccessFlags,
    dst: ImageDst,
    family: vk::QueueFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst.access(family))
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}
