//! The memory requirements of the images a heap texture's plan asks about and never binds
//! (`heap.rs`'s `planned_requirements`): each query makes an image, asks its requirements and
//! destroys it, on the thread that makes the texture. A window server that makes its heap textures
//! each frame asked the same ones each frame.
//!
//! Vulkan (Resource Memory Association) fixes the answer by the creation parameters:
//!
//! - "The size member is identical for all VkImage objects created with the same combination of
//!   creation parameters specified in VkImageCreateInfo and its pNext chain."
//! - "For images created with a color format, the memoryTypeBits member is identical for all
//!   VkImage objects created with the same combination of values for the tiling member, the
//!   VK_IMAGE_CREATE_SPARSE_BINDING_BIT bit and VK_IMAGE_CREATE_PROTECTED_BIT bit of the flags
//!   member, the VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT bit of the flags member, [...]
//!   handleTypes member of VkExternalMemoryImageCreateInfo, and the
//!   VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT of the usage member", and for a depth/stencil format
//!   the same with the format member too.
//! - "If the maintenance4 feature is enabled, then the alignment member is identical for all
//!   VkImage objects created with the same combination of values for the flags, imageType,
//!   format, extent, mipLevels, arrayLayers, samples, tiling and usage members in the
//!   VkImageCreateInfo structure passed to vkCreateImage."
//!
//! So on a device that enables `maintenance4`, an image of the same creation parameters has the
//! same requirements, and a query of parameters asked before takes the answer kept for them
//! ([`RequirementAnswers`]); elsewhere each query makes its image, as alignment is not fixed. A
//! plan and a [`Query`] fix every parameter of the image a query makes (`ImagePlan::create_info`,
//! the device's sharing, the tiling and its initial layout, and the handle type of a linear image
//! made for host allocations), so they are the key. A kept answer is the whole query's: it is kept
//! only once the device's limits for the image took it, and those limits
//! (`vkGetPhysicalDeviceImageFormatProperties2`) are the physical device's for the same
//! parameters, so a query answered from it asks nothing more (`WorkCounts::requirement_limits_asked`
//! counts the limits asked). An image made for host allocations whose limits took its handle type
//! without a dedicated allocation requires none ("requiresDedicatedAllocation will otherwise be
//! VK_FALSE", `VkMemoryDedicatedRequirements`), so its kept answer needs no second query of that.

use crate::image_plan::ImagePlan;
use ash::vk;
use std::collections::VecDeque;

/// The answers kept at most: the least recently asked goes first.
const KEPT: usize = 64;

/// What an answer kept for a plan is of: its image in a tiling, made for memory the device
/// allocates (`texture/heap.rs`'s `planned_requirements`), or its linear image made for host
/// allocations (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT`, made `UNDEFINED`:
/// `texture/heap.rs`'s `host_image_requirements`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Query {
    Tiling(vk::ImageTiling),
    HostLinear,
}

/// The memory requirements of planned images by their plan and [`Query`] (the module docs), kept
/// only where the device enables `maintenance4`.
pub(crate) struct RequirementAnswers {
    keeps: bool,
    /// The most recently asked first.
    answers: VecDeque<(ImagePlan, Query, vk::MemoryRequirements)>,
}

impl RequirementAnswers {
    /// The answers of a device that enables `maintenance4` where `maintenance4`.
    pub(crate) fn new(maintenance4: bool) -> RequirementAnswers {
        RequirementAnswers {
            keeps: maintenance4,
            answers: VecDeque::new(),
        }
    }

    /// The requirements kept for an image of `plan` and `query`, now the most recently asked.
    pub(crate) fn answer(
        &mut self,
        plan: &ImagePlan,
        query: Query,
    ) -> Option<vk::MemoryRequirements> {
        let at = (self.answers.iter())
            .position(|(kept, kept_query, _)| *kept_query == query && kept == plan)?;
        let entry = self.answers.remove(at)?;
        let requirements = entry.2;
        self.answers.push_front(entry);
        Some(requirements)
    }

    /// Keep `requirements`, the answer for an image of `plan` and `query`, where the device
    /// enables `maintenance4`.
    pub(crate) fn keep(
        &mut self,
        plan: &ImagePlan,
        query: Query,
        requirements: vk::MemoryRequirements,
    ) {
        if !self.keeps {
            return;
        }
        if self.answers.len() == KEPT {
            self.answers.pop_back();
        }
        self.answers.push_front((plan.clone(), query, requirements));
    }
}
