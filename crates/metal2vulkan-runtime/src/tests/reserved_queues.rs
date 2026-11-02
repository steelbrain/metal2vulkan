//! Queues of a device's own families reserved for the caller (`DeviceOptions::reserved_queues`,
//! `DeviceHandles::reserved_queues`), on a device: a reserved queue is of one of the device's
//! queue families, the round robin gives it up (one `VkQueue` fewer than the same device's
//! without the reservation), no command queue takes it, presenting or not, and a raw copy there
//! moves the runtime's own buffers between command buffers that order it through a shared
//! event's semaphore. A device with no queue to spare (MoltenVK's four families of one queue,
//! lavapipe's one queue) reserves none, says so in an informational message
//! (`metal2vulkan-reserved-queues-absent`), and is made and runs work all the same.
//!
//! Each prints what the device reported (`RESERVED-QUEUE:` lines), which is the evidence of the
//! path that ran: on a device with no queue to spare, the first and third tests check only that
//! nothing was reserved and that the absence was reported.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use crate::device::create::{queue_plan, LEAST_QUEUES, ROUND_ROBIN_CEILING};
use crate::{
    CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, MessageSink,
    ReservedQueue, ResourceOptions, ShaderCacheOptions, VkEventSignal, VkEventWait,
};
use ash::vk;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The command queues each run makes: one past the round robin's ceiling (`device/create.rs`), so
/// a device of any shape shares a `VkQueue` by the last and every round-robin `VkQueue` is taken.
const MOST: usize = 17;

/// The id name of the message a device with fewer reserved queues than asked sends.
const ABSENT: &str = "metal2vulkan-reserved-queues-absent";

/// The device the cfg selects, its shader cache in `cache`, with `reserved` queues asked for and
/// its messages kept in `messages`.
fn device(cache: &Path, reserved: u32, messages: &Arc<Mutex<Vec<String>>>) -> Device {
    let kept = Arc::clone(messages);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        reserved_queues: reserved,
        messages: Some(MessageSink::new(move |message| {
            kept.lock().unwrap().push(message.text.clone());
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The device's queue family properties.
fn families(device: &Device) -> Vec<vk::QueueFamilyProperties> {
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    }
}

/// The `VkQueue` each of `queues` submits to, each once.
fn vk_queues(queues: &[CommandQueue]) -> Vec<vk::Queue> {
    let mut distinct: Vec<vk::Queue> = Vec::new();
    for queue in queues {
        // SAFETY: the handle is read and not kept; `f` calls no runtime and waits for nothing.
        let vk_queue = unsafe { queue.with_vk_queue(|vk_queue, _| vk_queue) };
        if !distinct.contains(&vk_queue) {
            distinct.push(vk_queue);
        }
    }
    distinct
}

/// [`MOST`] command queues of `device` and the distinct `VkQueue`s they took: the round robin's.
fn round_robin(device: &Device) -> (Vec<CommandQueue>, Vec<vk::Queue>) {
    let queues: Vec<CommandQueue> = (0..MOST)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let taken = vk_queues(&queues);
    (queues, taken)
}

/// One reserved queue: of a family of the device's own, given up by the round robin, which takes
/// one `VkQueue` fewer than the same device's without the reservation, and taken by no command
/// queue (more of them than the round robin has, and presenting ones). Whether the device has one
/// to spare is read from what the same device without the reservation does, not from the plan:
/// it spares one exactly where that round robin has more than its floor of four `VkQueue`s (the
/// presenting reservation keeps its own). Where it has none to spare, it reserves none, reports
/// that, and its round robin is the one it has without.
#[test]
fn a_reserved_queue_is_a_runtime_family_s_and_no_command_queue_takes_it() {
    let scratch = Scratch::new("reserved-queue");
    let unreserved_messages = Arc::new(Mutex::new(Vec::new()));
    let unreserved = device(&scratch.0, 0, &unreserved_messages);
    let (_queues, unreserved_round_robin) = round_robin(&unreserved);
    let families = families(&unreserved);
    let unreserved_count = unreserved_round_robin.len();
    // The device spares a queue exactly where its round robin without one is past its floor.
    let spares = unreserved_count > LEAST_QUEUES as usize;

    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, 1, &messages);
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    let reserved: Vec<ReservedQueue> = raw.reserved_queues.to_vec();
    let (_queues, taken) = round_robin(&device);
    assert_eq!(
        device.work_counts().queues_sharing_a_vk_queue,
        (MOST - taken.len()) as u64,
        "every command queue past the round robin's shares a `VkQueue`"
    );
    let presenting: Vec<CommandQueue> = (0..3)
        .map(|_| device.new_command_queue_for_presenting().unwrap())
        .collect();
    let presenting_taken = vk_queues(&presenting);
    eprintln!(
        "RESERVED-QUEUE: {}: families {:?}, the runtime's {:?}, reserved {reserved:?}; round \
         robin {} VkQueues with the reservation, {} without; presenting {} VkQueues",
        device.name(),
        (families.iter())
            .map(|f| (f.queue_flags.as_raw(), f.queue_count))
            .collect::<Vec<_>>(),
        raw.queue_families,
        taken.len(),
        unreserved_count,
        presenting_taken.len(),
    );
    let plan = queue_plan(&families, 1).unwrap();
    assert_eq!(
        (reserved.len(), taken.len()),
        (plan.reserved.len(), plan.queues.len()),
        "the device's queues are its plan's: {plan:?}"
    );
    assert_eq!(
        reserved.len(),
        usize::from(spares),
        "a round robin of {unreserved_count} without the reservation spares one exactly past \
         its floor of {LEAST_QUEUES}"
    );
    let Some(queue) = reserved.first() else {
        let text = messages.lock().unwrap().join("\n");
        assert!(text.contains(ABSENT), "no {ABSENT} message: {text}");
        assert_eq!(
            taken.len(),
            unreserved_count,
            "a device that reserved nothing lends the round robin what it lends without"
        );
        eprintln!(
            "RESERVED-QUEUE: {}: no queue to spare, reported absent",
            device.name()
        );
        return;
    };
    assert!(
        raw.queue_families.contains(&queue.family),
        "{queue:?} is of none of the device's families {:?}",
        raw.queue_families
    );
    assert_ne!(queue.queue, vk::Queue::null());
    assert!(
        !taken.contains(&queue.queue),
        "a command queue took the reserved queue {queue:?}"
    );
    assert!(
        !presenting_taken.contains(&queue.queue),
        "a presenting queue took the reserved queue {queue:?}"
    );
    // The round robin gives the reserved queue up, unless the families have a queue past its
    // ceiling and presenting's two (a family of more than 18), which the reservation takes first.
    let at_ceiling = unreserved_count == ROUND_ROBIN_CEILING as usize;
    assert!(
        taken.len() + 1 == unreserved_count || (at_ceiling && taken.len() == unreserved_count),
        "the round robin gives up the reserved queue: {} VkQueues with it, {unreserved_count} \
         without",
        taken.len()
    );
    assert!(
        !messages.lock().unwrap().join("\n").contains(ABSENT),
        "a reservation made in full is not reported absent"
    );
}

/// A device asked for one queue more than its families spare (one, on a device with none to
/// spare: MoltenVK's, lavapipe's) reserves what it spares, reports the rest absent in an
/// informational message, and is made: a command buffer of its runs and completes.
#[test]
fn a_device_with_no_queue_to_spare_reports_the_reserved_queue_absent() {
    let scratch = Scratch::new("reserved-queue-absent");
    let probe_messages = Arc::new(Mutex::new(Vec::new()));
    let probe = device(&scratch.0, 0, &probe_messages);
    let spare = queue_plan(&families(&probe), u32::MAX)
        .unwrap()
        .reserved
        .len();
    drop(probe);
    let asked = spare as u32 + 1;
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, asked, &messages);
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    let text = messages.lock().unwrap().join("\n");
    eprintln!(
        "RESERVED-QUEUE: {}: {asked} asked, {} reserved; {}",
        device.name(),
        raw.reserved_queues.len(),
        (text.lines())
            .find(|line| line.contains(ABSENT))
            .unwrap_or("no message")
    );
    assert_eq!(raw.reserved_queues.len(), spare);
    assert!(
        text.contains(&format!("reserved {spare} of the {asked} queue(s)"))
            && text.contains(ABSENT),
        "the absence is reported: {text}"
    );
    // The device runs work.
    const BYTES: u64 = 256;
    let buffer = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..BYTES, 0xa5).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), BYTES as usize) };
    assert!(bytes.iter().all(|&b| b == 0xa5));
}

/// A raw copy on a reserved queue moves the runtime's own buffers, with no ownership transfer:
/// it waits for a command buffer's `encodeSignalEvent:` (the fill of its source), and signals
/// what a later command buffer's `encodeWaitForEvent:` waits for (the copy of its destination
/// into a buffer the host reads), through the event's semaphore. The later command buffer has
/// not completed before the raw signal. Where the device reserved no queue, nothing is copied.
#[test]
fn a_raw_copy_on_a_reserved_queue_moves_the_runtime_s_own_buffers() {
    let scratch = Scratch::new("reserved-queue-copy");
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, 1, &messages);
    // SAFETY: the handles are read while the device lives.
    let Some(&reserved) = unsafe { device.raw() }.reserved_queues.first() else {
        eprintln!(
            "RESERVED-QUEUE: {}: no queue to spare, no raw copy",
            device.name()
        );
        return;
    };
    const BYTES: u64 = 4096;
    let private = ResourceOptions::private();
    let source = device.new_buffer(BYTES, private).unwrap();
    let middle = device.new_buffer(BYTES, private).unwrap();
    let read = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();

    let first = queue.command_buffer().unwrap();
    let mut blit = first.blit_command_encoder().unwrap();
    blit.fill_buffer(&source, 0..BYTES, 0x3c).unwrap();
    blit.fill_buffer(&middle, 0..BYTES, 0).unwrap();
    blit.end_encoding().unwrap();
    first.encode_signal_event((&event).into(), 10).unwrap();
    first.commit().unwrap();
    first.wait_until_scheduled().unwrap();
    let VkEventWait::Counter(wait) = event.vk_wait(10) else {
        panic!("a submitted signal's counter: {:?}", event.vk_wait(10));
    };
    // SAFETY: the signal is submitted below, once, after a wait for `after`.
    let reservation = unsafe { event.reserve_vk_signal(20) }.unwrap();
    let VkEventSignal::Counter { counter, after } = reservation else {
        panic!("a signal above every value is kept: {reservation:?}");
    };

    let second = queue.command_buffer().unwrap();
    second.encode_wait_for_event((&event).into(), 20).unwrap();
    let mut blit = second.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&middle, 0, &read, 0, BYTES)
        .unwrap();
    blit.end_encoding().unwrap();
    second.commit().unwrap();
    assert!(
        second
            .wait_until_completed_for(Duration::from_millis(300))
            .is_err(),
        "the second command buffer ran before the raw signal"
    );

    // SAFETY: the buffers and the event live past the copy's completion, which the fence waits
    // for; the semaphore is signalled only as reserved; every object made here is destroyed here
    // after its work completed, and the command buffer's allocation is made with the loader's
    // word in place (`Device::with_loader_word`).
    unsafe {
        let raw = device.raw();
        let vk_device = raw.device;
        let ((from, from_at), (to, to_at)) = (source.vk_buffer(), middle.vk_buffer());
        let semaphore = event.vk_semaphore();
        let pool = vk_device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(reserved.family),
                None,
            )
            .unwrap();
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = device
            .with_loader_word(|| vk_device.allocate_command_buffers(&info))
            .unwrap();
        vk_device
            .begin_command_buffer(buffers[0], &vk::CommandBufferBeginInfo::default())
            .unwrap();
        let region = vk::BufferCopy {
            src_offset: from_at,
            dst_offset: to_at,
            size: BYTES,
        };
        vk_device.cmd_copy_buffer(buffers[0], from, to, &[region]);
        vk_device.end_command_buffer(buffers[0]).unwrap();
        let waits = [wait.max(after.unwrap_or(0))];
        let signals = [counter];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&waits)
            .signal_semaphore_values(&signals);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&semaphore))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
            .command_buffers(&buffers)
            .signal_semaphores(std::slice::from_ref(&semaphore))
            .push_next(&mut timeline);
        let fence = vk_device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        vk_device
            .queue_submit(reserved.queue, &[submit], fence)
            .unwrap();
        vk_device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        vk_device.destroy_fence(fence, None);
        vk_device.destroy_command_pool(pool, None);
    }
    second.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(second.status(), CommandBufferStatus::Completed);
    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let bytes =
        unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), BYTES as usize) };
    assert!(
        bytes.iter().all(|&b| b == 0x3c),
        "the fill reached the read through the raw copy"
    );
    eprintln!(
        "RESERVED-QUEUE: {}: a raw copy on {reserved:?} moved the runtime's buffers \
         (value 10 counter {wait}, value 20 counter {counter})",
        device.name()
    );
    drop((first, second));
}
