//! A submission's segments in submit infos: consecutive segments share one where no event wait
//! or signal separates them.

use crate::submit::{infos, Staged};
use ash::vk::{self, Handle};

fn command(raw: u64) -> Option<vk::CommandBufferSubmitInfo<'static>> {
    Some(vk::CommandBufferSubmitInfo::default().command_buffer(vk::CommandBuffer::from_raw(raw)))
}

fn semaphore(raw: u64, value: u64) -> vk::SemaphoreSubmitInfo<'static> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(vk::Semaphore::from_raw(raw))
        .value(value)
}

/// Each info as (the raw semaphores it waits on, its command buffers, the semaphores it signals).
fn shape(segments: &[Staged]) -> Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> {
    let raw = |infos: &[vk::SemaphoreSubmitInfo<'static>]| -> Vec<u64> {
        infos.iter().map(|info| info.semaphore.as_raw()).collect()
    };
    (infos(segments).iter())
        .map(|info| {
            (
                raw(info.waits),
                (info.commands.iter())
                    .map(|c| c.command_buffer.as_raw())
                    .collect(),
                raw(info.signals),
            )
        })
        .collect()
}

#[test]
fn segments_share_a_submit_info_until_an_event_wait_or_signal() {
    let staged = |waits: &[u64], raw: Option<u64>, signals: &[u64]| Staged {
        waits: waits.iter().map(|&w| semaphore(w, 1)).collect(),
        command: raw.and_then(command),
        signals: signals.iter().map(|&s| semaphore(s, 1)).collect(),
    };
    // The upload, three command buffers, the third signalling event 7; then one waiting on event
    // 8, one more, and the last with the progress semaphore (9).
    let segments = [
        staged(&[], Some(1), &[]),
        staged(&[], Some(2), &[]),
        staged(&[], Some(3), &[]),
        staged(&[], Some(4), &[7]),
        staged(&[8], Some(5), &[]),
        staged(&[], Some(6), &[]),
        staged(&[], Some(10), &[9]),
    ];
    assert_eq!(
        shape(&segments),
        [
            (vec![], vec![1, 2, 3, 4], vec![7]),
            (vec![8], vec![5, 6, 10], vec![9]),
        ]
    );
    // A signal alone (no commands) after a signal starts its own info; a wait right after a
    // signal does too; one segment is one info.
    let segments = [
        staged(&[], Some(1), &[7]),
        staged(&[], None, &[9]),
        staged(&[8], Some(2), &[]),
    ];
    assert_eq!(
        shape(&segments),
        [
            (vec![], vec![1], vec![7]),
            (vec![], vec![], vec![9]),
            (vec![8], vec![2], vec![]),
        ]
    );
    assert_eq!(
        shape(&[staged(&[], Some(1), &[9])]),
        [(vec![], vec![1], vec![9])]
    );
    assert!(shape(&[]).is_empty());
}
