//! An exportable shared event's timeline semaphore (`exported_semaphore.rs`) imported into a
//! second `VkDevice` of the same physical device, as another API imports it: the second device
//! sees the counter value a command buffer's signal reaches, and its own signal of a reserved
//! counter raises the event's value and releases a command buffer's wait. A device the runtime
//! refuses the event on (`RuntimeGap::ExportableSemaphore`) must lack the extension or the export
//! (`EXPORTED-SEMAPHORE:` lines).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::{Error, Problem, RuntimeGap};
use crate::{Device, ResourceOptions, VkEventSignal, VkEventWait};
use ash::vk;
use std::os::fd::IntoRawFd;
use std::time::Duration;

/// Whether the physical device under `device` lists `VK_KHR_external_semaphore_fd` and exports a
/// timeline semaphore as an opaque file descriptor.
fn exports(device: &Device) -> bool {
    // SAFETY: the handles are read while the device lives.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let listed = unsafe {
        handles
            .instance
            .enumerate_device_extension_properties(handles.physical_device)
    }
    .unwrap();
    let named = listed.iter().any(|extension| {
        extension.extension_name_as_c_str() == Ok(ash::khr::external_semaphore_fd::NAME)
    });
    let mut timeline =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let query = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
        .push_next(&mut timeline);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: as above, at Vulkan 1.2.
    unsafe {
        handles
            .instance
            .get_physical_device_external_semaphore_properties(
                handles.physical_device,
                &query,
                &mut properties,
            )
    };
    named
        && (properties.external_semaphore_features)
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
}

#[test]
fn an_exportable_shared_event_s_semaphore_crosses_to_a_second_device() {
    let scratch = Scratch::new("exported-semaphore");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let event = match device.new_exportable_shared_event() {
        Ok(event) => event,
        Err(Error::NotImplemented(RuntimeGap::ExportableSemaphore)) => {
            assert!(
                !exports(&device),
                "{name} exports a timeline semaphore as an opaque fd, and the runtime refused it"
            );
            eprintln!("EXPORTED-SEMAPHORE on {name}: refused, the device exports no opaque fd");
            return;
        }
        Err(error) => panic!("{name}: {error:?}"),
    };
    // A plain shared event's semaphore is not the caller's to export.
    let plain = device.new_shared_event().unwrap();
    assert!(matches!(
        plain.export_semaphore_fd(),
        Err(Error::Invalid(crate::error::Invalid {
            problem: Problem::NotExportable,
            ..
        }))
    ));
    let fd = event.export_semaphore_fd().unwrap();
    // Each call a new descriptor.
    let again = event.export_semaphore_fd().unwrap();
    assert_ne!(
        std::os::fd::AsRawFd::as_raw_fd(&again),
        std::os::fd::AsRawFd::as_raw_fd(&fd)
    );
    drop(again);

    // SAFETY: the handles are used while the device lives; the second device is the test's own
    // and destroyed below.
    let handles = unsafe { device.raw() };
    let (instance, physical) = (handles.instance, handles.physical_device);
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(handles.queue_family)
        .queue_priorities(&priorities)];
    let extensions = [ash::khr::external_semaphore_fd::NAME.as_ptr()];
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .push_next(&mut features12);
    // SAFETY: the physical device is the instance's; the extension is listed (the export runs).
    let second = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    let functions = ash::khr::external_semaphore_fd::Device::new(instance, &second);
    let mut timeline = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
    // SAFETY: the second device is live and enables `timelineSemaphore`.
    let imported = unsafe { second.create_semaphore(&semaphore_info, None) }.unwrap();
    let import = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(imported)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
        .fd(fd.into_raw_fd());
    // SAFETY: the descriptor is the export's of the same physical device, of a timeline
    // semaphore; a successful import owns it.
    unsafe { functions.import_semaphore_fd(&import) }.unwrap();

    // The runtime's signal of 5 raises the counter, and the second device reads the runtime's own
    // counter value (`vk_wait` answers `Reached` for a value reached, so the counter is read).
    // SAFETY: the second device's semaphore, an import of the event's.
    let before = unsafe { second.get_semaphore_counter_value(imported) }.unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_signal_event((&event).into(), 5)
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(event.vk_wait(5), VkEventWait::Reached, "{name}");
    // SAFETY: the counter is only read; the event lives.
    let ours = unsafe {
        handles
            .device
            .get_semaphore_counter_value(event.vk_semaphore())
    }
    .unwrap();
    // SAFETY: as above.
    let theirs = unsafe { second.get_semaphore_counter_value(imported) }.unwrap();
    assert!(
        theirs > before,
        "{name}: the signal of 5 raised no counter there"
    );
    assert_eq!(theirs, ours, "{name}: the two devices' counters");
    let semaphores = [imported];
    eprintln!("EXPORTED-SEMAPHORE on {name}: the runtime's 5 raised the counter to {theirs} there");

    // A command buffer waits for 9, then fills a buffer; the second device signals the counter
    // reserved for 9, which releases the wait and raises the event's value.
    let out = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let waiting = queue.command_buffer().unwrap();
    // SAFETY: the reservation's signal is the second device's, submitted below.
    let reserved = unsafe { event.reserve_vk_signal(9) }.unwrap();
    let VkEventSignal::Counter { counter, after } = reserved else {
        panic!("{name}: 9 reserved no counter");
    };
    waiting.encode_wait_for_event((&event).into(), 9).unwrap();
    let mut blit = waiting.blit_command_encoder().unwrap();
    blit.fill_buffer(&out, 0..64, 0x3c).unwrap();
    blit.end_encoding().unwrap();
    waiting.commit().unwrap();
    assert!(!event.wait_until_signaled_value(9, Duration::from_millis(50)));
    if let Some(after) = after {
        let values = [after];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: as above.
        unsafe { second.wait_semaphores(&wait, WAIT.as_nanos() as u64) }.unwrap();
    }
    let signal = vk::SemaphoreSignalInfo::default()
        .semaphore(imported)
        .value(counter);
    // SAFETY: the reserved counter, above every signal before it, which the semaphore reached.
    unsafe { second.signal_semaphore(&signal) }.unwrap();
    assert!(
        event.wait_until_signaled_value(9, WAIT),
        "{name}: the second device's signal of counter {counter} raised no value"
    );
    waiting.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(waiting.error(), None);
    let filled = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of 64 bytes whose commands all completed.
    let filled = unsafe { std::slice::from_raw_parts(filled.as_ptr(), 64) };
    assert!(filled.iter().all(|&byte| byte == 0x3c), "{name}: the fill");
    eprintln!("EXPORTED-SEMAPHORE on {name}: its signal of counter {counter} released 9");
    // SAFETY: the second device's objects, unused now; the event outlives the import.
    unsafe {
        second.destroy_semaphore(imported, None);
        second.destroy_device(None);
    }
}
