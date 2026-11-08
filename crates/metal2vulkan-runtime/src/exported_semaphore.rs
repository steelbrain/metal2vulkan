//! Shared events whose timeline semaphore another API imports (a runtime extension: Metal has no
//! counterpart).
//!
//! A consumer whose other GPU work signals or waits for a shared event on the GPU (CUDA's
//! `cuSignalExternalSemaphoresAsync` and `cuWaitExternalSemaphoresAsync` after a decode, a second
//! `VkDevice`) asks for the event at its making ([`Device::new_exportable_shared_event`]) and
//! takes its timeline semaphore as an opaque file descriptor
//! ([`SharedEvent::export_semaphore_fd`], `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`). No
//! other event pays for it.
//!
//! - **What the device must have.** `VK_KHR_external_semaphore_fd`, which the runtime enables
//!   wherever the device lists it (an adopted device: wherever its consumer enabled it), and an
//!   answer of `vkGetPhysicalDeviceExternalSemaphoreProperties` for a timeline semaphore that the
//!   handle type is exportable. Both are asked once, as the device is made; a device without them
//!   refuses the making (`NotImplemented(RuntimeGap::ExportableSemaphore)`).
//! - **Its counter is a signal index, not the event's value**, as for the semaphore the interop
//!   hooks lend (`interop.rs`, `event_timeline.rs`). The importer waits for the counter value
//!   [`SharedEvent::vk_wait`] answers for a Metal value, and signals only the counter value
//!   [`SharedEvent::reserve_vk_signal`] reserved, after the counter it names: the reservation's
//!   contract holds for the importer's signal as for a submission of the caller's.
//! - **The descriptor.** Each call of [`SharedEvent::export_semaphore_fd`] is a new one
//!   (`vkGetSemaphoreFdKHR`), the caller's to close; an import that succeeds takes it.
//! - **Lifetime.** The event must outlive every import of its semaphore, and every signal and wait
//!   an importer submitted must complete before the event's last clone goes: its release destroys
//!   the semaphore.
//! - **Device loss.** On a device loss the semaphore's state is lost for every import too, and the
//!   event's calls answer as a lost device's. A device made again makes new events with new
//!   semaphores: each is exported anew, with a new descriptor, and the old imports are to be let
//!   go of.

use super::error::{invalid, vulkan, Error, Problem, RuntimeGap};
use super::event::EventRef;
use super::{Device, SharedEvent};
use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};

/// The handle type the events export their semaphore as.
const HANDLE: vk::ExternalSemaphoreHandleTypeFlags =
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

/// How the device exports a timeline semaphore as an opaque file descriptor, asked once as the
/// device is made.
pub(super) struct OpaqueFdSemaphores {
    functions: ash::khr::external_semaphore_fd::Device,
}

impl OpaqueFdSemaphores {
    /// The device's export of a timeline semaphore as an opaque file descriptor: where it enables
    /// `VK_KHR_external_semaphore_fd` (`functions`, loaded then) and
    /// `vkGetPhysicalDeviceExternalSemaphoreProperties` answers the handle type exportable from a
    /// timeline semaphore. `None` where it lacks either.
    ///
    /// # Safety
    ///
    /// `physical` is a physical device of `instance`, made at Vulkan 1.2 or later.
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        functions: Option<ash::khr::external_semaphore_fd::Device>,
    ) -> Option<OpaqueFdSemaphores> {
        let functions = functions?;
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(HANDLE)
            .push_next(&mut timeline);
        let mut properties = vk::ExternalSemaphoreProperties::default();
        // SAFETY: the caller's contract.
        unsafe {
            instance.get_physical_device_external_semaphore_properties(
                physical,
                &query,
                &mut properties,
            )
        };
        let exports = (properties.external_semaphore_features)
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
            && properties.compatible_handle_types.contains(HANDLE);
        exports.then_some(OpaqueFdSemaphores { functions })
    }
}

impl Device {
    /// No Metal counterpart (a runtime extension): [`Device::new_shared_event`] whose timeline
    /// semaphore another API imports ([`SharedEvent::export_semaphore_fd`]): made exportable as
    /// an opaque file descriptor, where any other event's is not (`exported_semaphore.rs`). It is
    /// a shared event as any other in every call.
    ///
    /// The event must outlive every import of its semaphore and every signal and wait an
    /// importer submitted. On a device loss the semaphore's state is lost, for every import too,
    /// and a device made again makes new events, each exported anew.
    ///
    /// Refused as [`Device::new_shared_event`] refuses it, and `NotImplemented`
    /// ([`RuntimeGap::ExportableSemaphore`]) on a device without
    /// `VK_KHR_external_semaphore_fd` or that exports no timeline semaphore as an opaque file
    /// descriptor.
    pub fn new_exportable_shared_event(&self) -> Result<SharedEvent, Error> {
        self.at_serving(|device| device.new_exportable_shared_event_now())
    }

    /// [`Device::new_exportable_shared_event`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_exportable_shared_event_now(&self) -> Result<SharedEvent, Error> {
        if self.shared().opaque_fd_semaphores.is_none() {
            return Err(Error::NotImplemented(RuntimeGap::ExportableSemaphore));
        }
        self.new_exported_shared_event(HANDLE)
    }
}

impl SharedEvent {
    /// No Metal counterpart (a runtime extension): the event's timeline semaphore as a new opaque
    /// file descriptor (`vkGetSemaphoreFdKHR`, `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`),
    /// the caller's to close; an import that succeeds takes it. Each call answers a new
    /// descriptor of the same semaphore.
    ///
    /// Its counter is a signal index, not the event's value: an importer waits for the counter
    /// value [`vk_wait`](Self::vk_wait) answers, and signals only the one
    /// [`reserve_vk_signal`](Self::reserve_vk_signal) reserved, under that call's contract. The
    /// event must outlive every import and every signal and wait an importer submitted. On a
    /// device loss the semaphore's state is lost for every import too.
    ///
    /// Refused: an event not made by [`Device::new_exportable_shared_event`] (`Invalid`,
    /// `Problem::NotExportable`), one made before its device's recovery
    /// (`NotImplemented(MadeBeforeRecovery)`) or on a lost device (`DeviceLost`), and the driver's
    /// failure.
    pub fn export_semaphore_fd(&self) -> Result<OwnedFd, Error> {
        const CALL: &str = "export_semaphore_fd";
        let event = EventRef::from(self);
        let event = event.shared();
        let device = event.device();
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let export = device.shared().opaque_fd_semaphores.as_ref();
        let Some(export) = export.filter(|_| event.exportable()) else {
            return Err(invalid(CALL, Problem::NotExportable));
        };
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(event.semaphore())
            .handle_type(HANDLE);
        // SAFETY: the semaphore is the event's, live while the event is, made exportable as the
        // handle type (`EventShared::new`); the functions are the device's.
        let fd = unsafe { export.functions.get_semaphore_fd(&info) }.map_err(vulkan)?;
        // SAFETY: a new descriptor Vulkan handed over, which the caller owns from here on.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}
