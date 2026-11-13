//! The line a failed `vkQueueSubmit` logs (`submit_failure.rs`): each `VkSubmitInfo`'s lists read
//! back from it as it was handed to the call, and the line's words for what it holds, bounded.

use crate::submit_failure::{
    flag_names, lists, Carried, InfoLists, InfoReport, Report, Resources, Role, SemaphoreUse,
    Standing, NAMED, NAMED_SEMAPHORES,
};
use ash::vk::{self, Handle};

/// A timeline wait or signal of `semaphore` with `role`, to `value`, the timeline at `now`.
fn timeline(semaphore: u64, role: Role, value: u64, now: u64) -> SemaphoreUse {
    SemaphoreUse {
        handle: semaphore,
        role,
        value: Some(value),
        now: Some(now),
    }
}

#[test]
fn an_info_s_lists_and_its_timeline_values_are_read_from_its_chain() {
    let waits = [vk::Semaphore::from_raw(0x10), vk::Semaphore::from_raw(0x11)];
    let wait_values = [41, 0];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
    let buffers = [vk::CommandBuffer::from_raw(0xa0)];
    let signals = [vk::Semaphore::from_raw(0x20)];
    let signal_values = [42];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    // A structure before the timeline values in the chain is passed over.
    let mut group = vk::DeviceGroupSubmitInfo::default();
    let info = vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .command_buffers(&buffers)
        .signal_semaphores(&signals)
        .push_next(&mut timeline)
        .push_next(&mut group);
    // SAFETY: built above, alive here.
    let read = unsafe { lists(&info) };
    let expected = InfoLists {
        waits: &waits,
        wait_values: Some(&wait_values[..]),
        buffers: &buffers,
        signals: &signals,
        signal_values: Some(&signal_values[..]),
    };
    assert_eq!(read, expected);
    // An info of nothing, with no chain: empty lists, no values.
    let empty = vk::SubmitInfo::default();
    // SAFETY: as above.
    let read = unsafe { lists(&empty) };
    assert_eq!(
        read,
        InfoLists {
            waits: &[],
            wait_values: None,
            buffers: &[],
            signals: &[],
            signal_values: None,
        }
    );
}

#[test]
fn the_line_names_what_the_failed_submission_held_and_flags_invalid_values() {
    let mut resources = Resources::default();
    resources.state("buffer #7 of 4096 bytes, Shared".to_owned(), Standing::Held);
    resources.state("buffer #8 GONE".to_owned(), Standing::Gone);
    resources.state(
        "buffer #9, no-copy, DETACHED".to_owned(),
        Standing::Released,
    );
    resources.own = 3;
    let report = Report {
        what: "a batch of a command buffer".to_owned(),
        queue: Some(0),
        result: vk::Result::ERROR_UNKNOWN,
        fence: Some(0xfe),
        infos: (
            vec![
                InfoReport {
                    waits: (
                        vec![
                            timeline(0x10, Role::Queue(1), 41, 40),
                            SemaphoreUse {
                                handle: 0x11,
                                role: Role::Binary,
                                value: None,
                                now: None,
                            },
                            // Past the least difference Vulkan allows from the timeline's value.
                            timeline(0x12, Role::Event, 1 << 32, 1),
                        ],
                        3,
                    ),
                    buffers: (
                        vec![
                            (0xa0, Some("encoder 0 Blit of #12".to_owned())),
                            (0xa1, None),
                        ],
                        2,
                    ),
                    signals: (Vec::new(), 0),
                },
                InfoReport {
                    waits: (Vec::new(), 0),
                    buffers: (vec![(0xa2, Some("end".to_owned()))], 1),
                    // A signal not above the timeline's value.
                    signals: (vec![timeline(0x20, Role::Queue(0), 42, 42)], 1),
                },
            ],
            2,
        ),
        command_buffers: 3,
        carried: Carried {
            buffers: vec!["#12 command buffer `probe` (its 1 encoder: 0 blit)".to_owned()],
            resources,
            ..Carried::default()
        },
    };
    let line = report.text();
    eprintln!("SUBMIT-FAILED {line}");
    for expected in [
        "vkQueueSubmit failed with VkResult -13 (ERROR_UNKNOWN) on VkQueue 0",
        "submitting a batch of a command buffer: 2 VkSubmitInfos, 3 VkCommandBuffers, fence 0xfe",
        "info 0: waits [0x10 VkQueue 1's timeline to 41 (at 40), 0x11 binary, the caller's",
        "0x12 an event's timeline to 4294967296 (at 1) PAST 2^31 - 1 FROM ITS VALUE \
         (VUID-VkSubmitInfo-pWaitSemaphores-03243)]",
        "runs [0xa0 encoder 0 Blit of #12, 0xa1], signals none",
        "info 1: waits none, runs [0xa2 end], signals [0x20 VkQueue 0's timeline to 42 (at 42) \
         NOT ABOVE ITS VALUE (VUID-VkSubmitInfo-pSignalSemaphores-03242)]",
        "of 1 command buffer: #12 command buffer `probe`",
        "resources: 3 stated, 1 gone, 1 detached or released; 3 keys of the runtime's own: \
         buffer #7 of 4096 bytes, Shared; buffer #8 GONE; buffer #9, no-copy, DETACHED",
    ] {
        assert!(line.contains(expected), "{expected:?} missing from {line}");
    }
}

#[test]
fn memory_flags_are_named_and_an_unnamed_bit_is_given_by_its_value() {
    use vk::MemoryPropertyFlags as F;
    let flags = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT;
    assert_eq!(flag_names(flags), "DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT");
    assert_eq!(flag_names(F::empty()), "none");
    let unnamed = F::from_raw(0x100);
    assert_eq!(flag_names(F::HOST_CACHED | unnamed), "HOST_CACHED|0x100");
}

#[test]
fn the_line_names_at_most_its_bounds_and_counts_the_rest() {
    let mut resources = Resources::default();
    for at in 0..NAMED + 5 {
        resources.state(format!("buffer #{at}"), Standing::Held);
    }
    assert_eq!(resources.named.len(), NAMED);
    assert_eq!(resources.stated, NAMED + 5);
    let waits = (0..NAMED_SEMAPHORES as u64)
        .map(|at| timeline(at, Role::Timeline, at + 1, at))
        .collect();
    let buffers = (0..NAMED as u64).map(|at| (at, None)).collect();
    let report = Report {
        what: "a batch of a command buffer".to_owned(),
        queue: None,
        result: vk::Result::ERROR_DEVICE_LOST,
        fence: None,
        infos: (
            vec![InfoReport {
                waits: (waits, NAMED_SEMAPHORES + 2),
                buffers: (buffers, NAMED + 4),
                signals: (Vec::new(), 0),
            }],
            9,
        ),
        command_buffers: NAMED + 4,
        carried: Carried {
            resources,
            ..Carried::default()
        },
    };
    let line = report.text();
    for expected in [
        "on its own VkQueue",
        "no fence",
        ", and 2 more], runs [",
        ", and 4 more], signals none",
        "; 8 more VkSubmitInfos",
        "; 4 VkCommandBuffers not named",
        "; and 5 more",
    ] {
        assert!(line.contains(expected), "{expected:?} missing from {line}");
    }
    assert!(!line.contains(&format!("buffer #{NAMED}")), "{line}");
}
