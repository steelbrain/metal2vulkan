//! Releasing Vulkan objects: the runtime's one door to the `vkDestroy*` and `vkFree*` entry points.
//! [`object`] runs each inside a section of the exit guard ([`exit_guard`]) and marks its entry
//! point ([`Call`]), so the hook is registered again after the first one of its kind; once the
//! process is exiting, the object is left to the process's end. A release can run on any thread (a
//! runtime thread drops the last reference to what a command buffer held), so every one goes
//! through here.
//!
//! A new kind of Vulkan object implements [`Released`] below, with a [`Call`] of its own (the
//! instance, its messenger, and the objects of `VK_EXT_device_generated_commands` have functions
//! of their own). No other file of the runtime calls such an entry point: a source test fails
//! when one does (`tests/exit_guard.rs`).

use crate::allocations;
use crate::churn_counts::{self, VkRelease};
use crate::exit_guard::{self, Call};
use crate::generated_commands::{Commands, Handle};
use ash::vk;
use std::panic::Location;

/// A Vulkan object a device releases.
pub(crate) trait Released {
    /// Its entry point.
    const CALL: Call;

    /// Release it: only [`object`] can, holding the [`Door`].
    ///
    /// # Safety
    ///
    /// As the entry point's: the object is `device`'s, and nothing pending uses it.
    unsafe fn destroy(self, device: &ash::Device, door: Door);
}

/// The proof that [`object`] calls [`Released::destroy`]: nothing else can make one.
pub(crate) struct Door(());

/// The device itself, released with its own functions (`vkDestroyDevice`).
pub(crate) struct WholeDevice;

/// Release `object` with `device`'s functions, inside a section of the exit guard, marking its
/// entry point; once the process is exiting, leave it to the process's end. Whether it was
/// released. Inside a section already (a whole drop's), it is always released. An image, a
/// buffer or a memory released is counted at its caller's place (`churn_counts.rs`), and a memory
/// among its device's live allocations too (`allocations.rs`), its free the device's work in
/// progress for its checks to step aside for (`step_aside.rs`).
///
/// # Safety
///
/// As the entry point's: the object is `device`'s, and nothing pending uses it.
#[track_caller]
pub(crate) unsafe fn object<T: Released>(device: &ash::Device, object: T) -> bool {
    let Some(_busy) = exit_guard::enter() else {
        return false;
    };
    let frees = matches!(T::CALL, Call::FreeMemory);
    let allocations = frees.then(|| allocations::of(device.handle())).flatten();
    let in_progress = allocations::Busy::of(allocations.as_deref());
    // SAFETY: the caller's.
    unsafe { object.destroy(device, Door(())) };
    drop(in_progress);
    exit_guard::called(T::CALL);
    let counted = match T::CALL {
        Call::DestroyImage => Some(VkRelease::Image),
        Call::DestroyBuffer => Some(VkRelease::Buffer),
        Call::DestroyBufferView => Some(VkRelease::BufferView),
        Call::FreeMemory => Some(VkRelease::Memory),
        _ => None,
    };
    if let Some(what) = counted {
        churn_counts::vk_released(what, Location::caller());
    }
    if frees {
        allocations::freed(allocations.as_deref());
    }
    true
}

/// [`object`] for the instance itself (`vkDestroyInstance`).
///
/// # Safety
///
/// Every device and messenger made from the instance is released.
pub(crate) unsafe fn instance(instance: &ash::Instance) -> bool {
    let Some(_busy) = exit_guard::enter() else {
        return false;
    };
    // SAFETY: the caller's.
    unsafe { instance.destroy_instance(None) };
    exit_guard::called(Call::DestroyInstance);
    true
}

/// [`object`] for a debug-utils messenger of `loader`'s instance.
///
/// # Safety
///
/// The messenger is the instance's, released once.
pub(crate) unsafe fn messenger(
    loader: &ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
) -> bool {
    let Some(_busy) = exit_guard::enter() else {
        return false;
    };
    // SAFETY: the caller's.
    unsafe { loader.destroy_debug_utils_messenger(messenger, None) };
    exit_guard::called(Call::DestroyDebugUtilsMessenger);
    true
}

/// An object of `VK_EXT_device_generated_commands` (`generated_commands.rs`), whose commands
/// `ash`'s device functions do not hold.
pub(crate) enum Generated {
    /// A `VkIndirectExecutionSetEXT`.
    ExecutionSet(Handle),
    /// A `VkIndirectCommandsLayoutEXT`.
    CommandsLayout(Handle),
}

/// [`object`] for an object of `VK_EXT_device_generated_commands`, released with the extension's
/// `commands`.
///
/// # Safety
///
/// As the entry point's: the object is the commands' device's, and nothing pending uses it.
pub(crate) unsafe fn generated(commands: &Commands, object: Generated) -> bool {
    let Some(_busy) = exit_guard::enter() else {
        return false;
    };
    let (destroy, handle, call) = match object {
        Generated::ExecutionSet(set) => (
            commands.destroy_execution_set,
            set,
            Call::DestroyIndirectExecutionSet,
        ),
        Generated::CommandsLayout(layout) => (
            commands.destroy_commands_layout,
            layout,
            Call::DestroyIndirectCommandsLayout,
        ),
    };
    // SAFETY: the caller's.
    unsafe { destroy(commands.device, handle, std::ptr::null()) };
    exit_guard::called(call);
    true
}

/// Command buffers of one command pool, freed together (`vkFreeCommandBuffers`), the pool then
/// trimmed (`vkTrimCommandPool`) where `trim`, so that the driver returns what it keeps of them: a
/// spare pool's past its recent uses (`spares.rs`), whose pool is not trimmed where the driver
/// fails a command buffer submitted again once one that came next after it is trimmed
/// (`RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission`, `driver_defects.rs`).
pub(crate) struct PoolBuffers<'a> {
    pub(crate) pool: vk::CommandPool,
    pub(crate) buffers: &'a [vk::CommandBuffer],
    pub(crate) trim: bool,
}

impl Released for PoolBuffers<'_> {
    const CALL: Call = Call::FreeCommandBuffers;

    unsafe fn destroy(self, device: &ash::Device, _: Door) {
        let PoolBuffers {
            pool,
            buffers,
            trim,
        } = self;
        // SAFETY: the caller's: none of the command buffers is pending, and it holds the pool
        // alone.
        unsafe { device.free_command_buffers(pool, buffers) };
        if trim {
            // SAFETY: as above.
            unsafe { device.trim_command_pool(pool, vk::CommandPoolTrimFlags::empty()) };
            exit_guard::called(Call::TrimCommandPool);
        }
    }
}

/// `Released` for handles a device's `vkDestroy*` or `vkFree*` releases.
macro_rules! released {
    ($($handle:ty => $method:ident, $call:ident;)*) => {
        $(
            impl Released for $handle {
                const CALL: Call = Call::$call;

                unsafe fn destroy(self, device: &ash::Device, _: Door) {
                    // SAFETY: the caller's.
                    unsafe { device.$method(self, None) }
                }
            }
        )*
    };
}

released! {
    vk::Buffer => destroy_buffer, DestroyBuffer;
    vk::BufferView => destroy_buffer_view, DestroyBufferView;
    vk::Image => destroy_image, DestroyImage;
    vk::ImageView => destroy_image_view, DestroyImageView;
    vk::Sampler => destroy_sampler, DestroySampler;
    vk::SamplerYcbcrConversion => destroy_sampler_ycbcr_conversion, DestroySamplerYcbcrConversion;
    vk::DeviceMemory => free_memory, FreeMemory;
    vk::Semaphore => destroy_semaphore, DestroySemaphore;
    vk::Fence => destroy_fence, DestroyFence;
    vk::CommandPool => destroy_command_pool, DestroyCommandPool;
    vk::Pipeline => destroy_pipeline, DestroyPipeline;
    vk::PipelineLayout => destroy_pipeline_layout, DestroyPipelineLayout;
    vk::DescriptorSetLayout => destroy_descriptor_set_layout, DestroyDescriptorSetLayout;
    vk::DescriptorPool => destroy_descriptor_pool, DestroyDescriptorPool;
    vk::ShaderModule => destroy_shader_module, DestroyShaderModule;
    vk::RenderPass => destroy_render_pass, DestroyRenderPass;
    vk::Framebuffer => destroy_framebuffer, DestroyFramebuffer;
    vk::PipelineCache => destroy_pipeline_cache, DestroyPipelineCache;
    vk::QueryPool => destroy_query_pool, DestroyQueryPool;
}

impl Released for WholeDevice {
    const CALL: Call = Call::DestroyDevice;

    unsafe fn destroy(self, device: &ash::Device, _: Door) {
        // SAFETY: the caller's: every object made from the device is released.
        unsafe { device.destroy_device(None) }
    }
}
