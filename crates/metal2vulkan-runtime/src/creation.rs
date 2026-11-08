//! The runtime's one door to `vkCreateImage`, `vkCreateRenderPass`, `vkCreateRenderPass2` and
//! `vkCreateFramebuffer`, so their entry points stay in the exit guard's tables; and to the other
//! makings a driver charges for — `vkCreateBuffer`, `vkCreateImageView`, `vkAllocateMemory`,
//! `vkBindImageMemory` and `vkBindBufferMemory` — so each counts itself where it is called.
//!
//! Each is made where it is first needed: on the program's thread, in a section of the device's
//! lowering thread, which lowers render encoders' calls (`render/lowering_worker.rs`), or in a host
//! step (`exit_guard::host_step`) on the steps' thread (`scheduler/steps.rs`), an object no earlier
//! settle made (a
//! state made after its command buffer's commit, whose run needs an attachment's stand-in, a
//! shadow, or a pass and framebuffer of its own shape). So the four, and an image's memory
//! requirements and binding, are in the guard's table (`exit_guard::Call`) and among the calls a
//! host step makes (`exit_guard::HOST_STEP`). The device's priming (`device/prime.rs`) makes its
//! own, on the thread that makes the device. No other file calls these entry points: a source test
//! fails when one does (`tests/exit_guard.rs`).
//!
//! **Each making is counted at its caller's place** (`churn_counts.rs`, as the releases are), so
//! what one call of the API made the driver do is a count of that call's own work rather than an
//! inference from a separately timed driver floor: a 256x256 texture's `vkCreateImage` and its
//! binding, a buffer's `vkCreateBuffer` and whether its memory came from a pool or a fresh
//! `vkAllocateMemory`. A count is deterministic, so a test can hold a call to it where a time
//! cannot.

use crate::allocations;
use crate::churn_counts::{vk_made, VkMaking};
use crate::error::{vulkan, Error};
use ash::prelude::VkResult;
use ash::vk;
use std::panic::Location;

/// A `VkImage` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
#[track_caller]
pub(crate) unsafe fn image(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<vk::Image, Error> {
    vk_made(VkMaking::Image, Location::caller());
    // SAFETY: the caller's.
    unsafe { device.create_image(info, None) }.map_err(vulkan)
}

/// A `VkImage` of `info`, made by `device`, or the driver's refusal of it: an image whose making
/// the driver may refuse for its parameters alone (one of a DRM format modifier's explicit
/// layout, `dma_buf.rs`), whose caller takes another way then. A lost device is
/// `Error::DeviceLost`.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
#[track_caller]
pub(crate) unsafe fn image_or_refusal(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<Result<vk::Image, vk::Result>, Error> {
    vk_made(VkMaking::Image, Location::caller());
    // SAFETY: the caller's.
    match unsafe { device.create_image(info, None) } {
        Ok(raw) => Ok(Ok(raw)),
        Err(vk::Result::ERROR_DEVICE_LOST) => Err(Error::DeviceLost),
        Err(refusal) => Ok(Err(refusal)),
    }
}

/// A `VkBuffer` of `info`, made by `device`, with the driver's result as it gave it: the callers
/// read a refusal differently (a staging buffer's caller takes another way, a program's buffer
/// turns it into `Error`).
///
/// # Safety
///
/// As `vkCreateBuffer`'s: `device` is live, and `info` and its chain are valid for it.
#[track_caller]
pub(crate) unsafe fn buffer(
    device: &ash::Device,
    info: &vk::BufferCreateInfo<'_>,
) -> VkResult<vk::Buffer> {
    vk_made(VkMaking::Buffer, Location::caller());
    // SAFETY: the caller's.
    unsafe { device.create_buffer(info, None) }
}

/// A `VkImageView` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateImageView`'s: `device` is live, and `info` and its chain are valid for it.
#[track_caller]
pub(crate) unsafe fn image_view(
    device: &ash::Device,
    info: &vk::ImageViewCreateInfo<'_>,
) -> VkResult<vk::ImageView> {
    vk_made(VkMaking::ImageView, Location::caller());
    // SAFETY: the caller's.
    unsafe { device.create_image_view(info, None) }
}

/// `VkDeviceMemory` of `info`, allocated by `device`: one call of `vkAllocateMemory`, counted
/// whether or not the driver gave it, because a caller that tries memory types in turn pays for
/// each try. Refused (`ERROR_TOO_MANY_OBJECTS`) with no call where the device has
/// `maxMemoryAllocationCount` allocations live (`allocations.rs`). The call is the device's work
/// in progress for its checks to step aside for (`step_aside.rs`).
///
/// # Safety
///
/// As `vkAllocateMemory`'s: `device` is live, and `info` and its chain are valid for it and live
/// for the call.
#[track_caller]
pub(crate) unsafe fn allocate_memory(
    device: &ash::Device,
    info: &vk::MemoryAllocateInfo<'_>,
) -> VkResult<vk::DeviceMemory> {
    let Some(admitted) = allocations::before_allocation(device.handle()) else {
        return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
    };
    vk_made(VkMaking::Memory, Location::caller());
    let busy = allocations::Busy::of(admitted.as_deref());
    // SAFETY: the caller's.
    let memory = unsafe { device.allocate_memory(info, None) };
    drop(busy);
    allocations::allocated(admitted, memory.is_ok());
    memory
}

/// `image` bound to `memory` at `offset`.
///
/// # Safety
///
/// As `vkBindImageMemory`'s: `device` is live, `image` is its unbound image, and `memory` is its
/// memory of a type the image takes, long enough for it from `offset`.
#[track_caller]
pub(crate) unsafe fn bind_image(
    device: &ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) -> VkResult<()> {
    vk_made(VkMaking::ImageBinding, Location::caller());
    // SAFETY: the caller's.
    unsafe { device.bind_image_memory(image, memory, offset) }
}

/// `buffer` bound to `memory` at `offset`.
///
/// # Safety
///
/// As `vkBindBufferMemory`'s: `device` is live, `buffer` is its unbound buffer, and `memory` is
/// its memory of a type the buffer takes, long enough for it from `offset`.
#[track_caller]
pub(crate) unsafe fn bind_buffer(
    device: &ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) -> VkResult<()> {
    vk_made(VkMaking::BufferBinding, Location::caller());
    // SAFETY: the caller's.
    unsafe { device.bind_buffer_memory(buffer, memory, offset) }
}

/// A `VkRenderPass` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateRenderPass`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn render_pass(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo<'_>,
) -> Result<vk::RenderPass, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_render_pass(info, None) }.map_err(vulkan)
}

/// A `VkRenderPass` of `info` (`vkCreateRenderPass2`), made by `device`.
///
/// # Safety
///
/// As `vkCreateRenderPass2`'s: `device` is live and of Vulkan 1.2 or later, and `info` and its
/// chain are valid for it.
pub(crate) unsafe fn render_pass2(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo2<'_>,
) -> Result<vk::RenderPass, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_render_pass2(info, None) }.map_err(vulkan)
}

/// A `VkFramebuffer` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateFramebuffer`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn framebuffer(
    device: &ash::Device,
    info: &vk::FramebufferCreateInfo<'_>,
) -> Result<vk::Framebuffer, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_framebuffer(info, None) }.map_err(vulkan)
}
