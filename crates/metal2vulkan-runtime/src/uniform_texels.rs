//! A buffer slot's uniform view in its texel form (`DescriptorView::UniformTexel`): the range its
//! uniform buffer would bind, viewed as a uniform texel buffer of `R32G32B32A32_UINT` texels, which
//! its function reads with `OpImageFetch`. The translation takes the form on a device whose
//! uniform buffer reads of external memory miss a cache
//! (`DriverWorkaround::ExternalUniformReadsSkipACache`), for a window it bounded to at most 4 KiB.
//!
//! - **One view a write.** Each write of such a slot's descriptor makes a view of the range the
//!   plan states (`binding::written`), in whatever buffer holds it: the bound buffer's space, the
//!   command buffer's upload arena for `set*Bytes` data, or the device's zero buffer for a slot
//!   bound to nothing, each made with the uniform texel buffer usage. Making and destroying one
//!   took 0.082 us on an Iris Xe under ANV (mean of 4000), and writing one into a set 0.007 us
//!   against a uniform buffer's 0.024. No view is kept for a later write: a set 0 that names one
//!   is bound again by no later command (`binding::SetCache`), and a draw rebound at its slot binds
//!   whole (`binding::rebind_into`).
//! - **Held until the command buffer completed**, with what it retained (`State::retain_own`). A
//!   view is what the command buffer alone holds, so a recording of an indirect command buffer's
//!   chunk that made one is not kept for another command buffer to run (`kept_commands.rs`).
//! - **Exact.** The range is the uniform view's own (`bind::instantiate::planned_range`): it starts
//!   at a multiple of `max(minTexelBufferOffsetAlignment, 16)` and holds a multiple of 16 bytes,
//!   at most `maxTexelBufferElements` texels, and a region of the upload arena starts at a multiple
//!   of the texel buffer offset alignment (`upload.rs`). A texel is 16 bytes of the range in order,
//!   so a fetch answers what a uniform buffer's load of the same words would.

use super::buffer_texture::view_usage;
use super::error::{vulkan, Error};
use super::exit_guard;
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::device::VkFeature;
use metal2vulkan::vk::Format;

/// The texel format of a uniform view's texel form: four 32-bit words a texel, its lanes the
/// words of 16 bytes in order.
const FORMAT: Format = Format::R32G32B32A32_UINT;

/// One bind's uniform texel buffer view, destroyed as its command buffer lets it go.
pub(super) struct UniformTexels {
    device: Device,
    view: vk::BufferView,
}

impl UniformTexels {
    /// The view of `info`'s range of its buffer, a uniform view's range in its texel form.
    pub(super) fn new(
        device: &Device,
        info: &vk::DescriptorBufferInfo,
    ) -> Result<UniformTexels, Error> {
        let profile = device.compile_device().profile();
        let usage = view_usage(
            |usage| profile.supports_format(FORMAT, usage),
            false,
            device.enabled().features.contains(&VkFeature::Maintenance5),
        )?;
        let mut only = vk::BufferUsageFlags2CreateInfoKHR::default();
        let mut create = vk::BufferViewCreateInfo::default()
            .buffer(info.buffer)
            .format(FORMAT.into())
            .offset(info.offset)
            .range(info.range);
        if let Some(named) = usage.named() {
            only = only.usage(named);
            create = create.push_next(&mut only);
        }
        // SAFETY: the buffer is live (its command holds it), made with the uniform texel buffer
        // usage (`buffer::usage`, the upload arena's chunks, the device's zero buffer); `FORMAT`
        // serves the usages the view has ([`view_usage`]); the plan aligned the range and bounded
        // its texels (the module docs).
        let view =
            unsafe { device.shared().vk().create_buffer_view(&create, None) }.map_err(vulkan)?;
        Ok(UniformTexels {
            device: device.clone(),
            view,
        })
    }

    /// The view.
    pub(super) fn raw(&self) -> vk::BufferView {
        self.view
    }
}

impl Drop for UniformTexels {
    fn drop(&mut self) {
        // Once the process is exiting, the view is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: the command buffer that bound the view let it go once it completed, or never
        // submitted it.
        unsafe {
            release::object(raw, self.view);
        }
    }
}
