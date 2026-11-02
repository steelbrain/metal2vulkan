//! An encoder's holds on the buffers over host pages its commands name (`rebind::Holds`): while
//! the device has such a buffer, every key a command names first is held, and only the keys of
//! those buffers are looked up among them under the lock a submission judges its holds under; the
//! key of a buffer of the runtime's memory is told apart by the device's filter of hosted keys
//! (`rebind::HostedKeys`) without the lookup (`WorkCounts::host_holds_looked_up`). A compositor's
//! draws each name a texture of their own, and looked each up in turn.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use crate::{CommandBufferStatus, ResourceOptions};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The buffers of the runtime's memory the blit copies between.
const OWN: usize = 8;

/// The bytes each copy moves.
const COPIED: u64 = 256;

/// A blit that copies between [`OWN`] buffers of the runtime's memory, and once from the no-copy
/// buffer, looks up the no-copy buffer's key alone; and its bytes arrive.
#[test]
fn only_hosted_keys_are_looked_up() {
    let host = Host::new();
    let scratch = Scratch::new("host-hold-filter");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(hosted) = over_host(&device, &host, &runs) else {
        return;
    };
    host.fill(0x1234_5678);
    let own: Vec<_> = (0..OWN)
        .map(|_| {
            device
                .new_buffer(COPIED, ResourceOptions::private())
                .unwrap()
        })
        .collect();
    let read = device
        .new_buffer(COPIED, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts().host_holds_looked_up;
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&hosted, 0, &own[0], 0, COPIED)
        .unwrap();
    for pair in own.windows(2) {
        blit.copy_from_buffer_to_buffer(&pair[0], 0, &pair[1], 0, COPIED)
            .unwrap();
    }
    blit.copy_from_buffer_to_buffer(&own[OWN - 1], 0, &read, 0, COPIED)
        .unwrap();
    blit.end_encoding().unwrap();
    let looked_up = device.work_counts().host_holds_looked_up - before;
    assert_eq!(looked_up, 1, "the no-copy buffer's key alone");
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let contents = read.contents().unwrap();
    // SAFETY: a shared buffer of `COPIED` bytes, which no command writes any longer.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), COPIED as usize) };
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert!(words.iter().all(|&w| w == 0x1234_5678), "{words:x?}");
}
