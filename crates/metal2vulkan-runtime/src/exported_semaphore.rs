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
//!
//! The device also exports a command buffer's completion as a sync file
//! (`CommandBuffer::add_completion_sync_file`, `command_buffer/sync_file.rs`): a binary semaphore
//! made exportable as `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`, which a `VkDevice` of any
//! driver imports, where the timeline semaphore's opaque descriptor crosses only to a device of the
//! same `deviceUUID` and `driverUUID`. It is asked once, as the device is made, as above, for a
//! binary semaphore ([`SyncFdSemaphores`]); the device keeps a few such semaphores idle between
//! command buffers.

use super::device::lock;
use super::error::{invalid, vulkan, Error, Problem, RuntimeGap};
use super::event::EventRef;
use super::exit_guard::{self, Call};
use super::{release, Device, SharedEvent};
use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Mutex;

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

/// The handle type a command buffer's completion is exported as.
const SYNC_FD: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD;

/// The most binary semaphores a device keeps idle for its command buffers' sync files.
const IDLE_SYNC_FD: usize = 8;

/// How the device exports a binary semaphore as a sync file, asked once as the device is made, and
/// the semaphores it keeps idle for its command buffers' sync files
/// (`command_buffer/sync_file.rs`): each unsignaled, in no pending submission.
pub(super) struct SyncFdSemaphores {
    functions: ash::khr::external_semaphore_fd::Device,
    idle: Mutex<Vec<vk::Semaphore>>,
}

impl SyncFdSemaphores {
    /// The device's export of a binary semaphore as a sync file: where it enables
    /// `VK_KHR_external_semaphore_fd` (`functions`, loaded then) and
    /// `vkGetPhysicalDeviceExternalSemaphoreProperties` answers the handle type exportable from a
    /// binary semaphore. `None` where it lacks either.
    ///
    /// # Safety
    ///
    /// `physical` is a physical device of `instance`, made at Vulkan 1.2 or later.
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        functions: Option<ash::khr::external_semaphore_fd::Device>,
    ) -> Option<SyncFdSemaphores> {
        let functions = functions?;
        let mut binary =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::BINARY);
        let query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(SYNC_FD)
            .push_next(&mut binary);
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
            && properties.compatible_handle_types.contains(SYNC_FD);
        exports.then(|| SyncFdSemaphores {
            functions,
            idle: Mutex::new(Vec::new()),
        })
    }

    /// An unsignaled binary semaphore of `device` in no pending submission, made exportable as a
    /// sync file: an idle one, else a new one.
    ///
    /// # Safety
    ///
    /// `device` is the live device these semaphores are of.
    pub(super) unsafe fn take(&self, device: &ash::Device) -> Result<vk::Semaphore, Error> {
        if let Some(semaphore) = lock(&self.idle).pop() {
            return Ok(semaphore);
        }
        let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(SYNC_FD);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        // SAFETY: the caller's; the handle type is one the device exports a binary semaphore as
        // (`new`).
        let made = unsafe { device.create_semaphore(&info, None) };
        exit_guard::called(Call::CreateSemaphore);
        made.map_err(vulkan)
    }

    /// Give back `semaphore` of `device`, unsignaled and in no pending submission: kept idle up to
    /// [`IDLE_SYNC_FD`], else released.
    ///
    /// # Safety
    ///
    /// As [`take`](Self::take)'s; `semaphore` came from it, and nothing pending uses it.
    pub(super) unsafe fn give(&self, device: &ash::Device, semaphore: vk::Semaphore) {
        {
            let mut idle = lock(&self.idle);
            if idle.len() < IDLE_SYNC_FD {
                idle.push(semaphore);
                return;
            }
        }
        // SAFETY: the caller's.
        unsafe { release::object(device, semaphore) };
    }

    /// The payload of `semaphore` as a new sync file (`vkGetSemaphoreFdKHR`), the caller's to
    /// close; `None` where the driver answered -1, a payload already signaled. The export resets
    /// the semaphore's payload as a wait for it would (copy transference).
    ///
    /// # Safety
    ///
    /// `semaphore` came from [`take`](Self::take), lives, and has a signal operation submitted, as
    /// has every signal operation it depends on; no queue waits for it.
    pub(super) unsafe fn export(&self, semaphore: vk::Semaphore) -> Result<Option<OwnedFd>, Error> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(SYNC_FD);
        // SAFETY: the caller's; the semaphore was made exportable as the handle type (`take`).
        let fd = unsafe { self.functions.get_semaphore_fd(&info) }.map_err(vulkan)?;
        // SAFETY: a new descriptor Vulkan handed over, which the caller owns from here on.
        Ok((fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    /// Release every idle semaphore: the device is being torn down.
    ///
    /// # Safety
    ///
    /// As [`give`](Self::give)'s.
    pub(super) unsafe fn release(&self, device: &ash::Device) {
        let idle: Vec<vk::Semaphore> = lock(&self.idle).drain(..).collect();
        for semaphore in idle {
            // SAFETY: the caller's.
            unsafe { release::object(device, semaphore) };
        }
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
