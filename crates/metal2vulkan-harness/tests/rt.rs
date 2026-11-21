//! The runtime through the harness, on the device the cfg names (docs/VALIDATION.md): a range
//! given back is the next one taken, buffers alive at once lie apart, and a buffer longer than a
//! pooled block takes a block of its own; and what textures released under work in flight leave
//! waiting stays within what the work in flight accounts for (`releases_under_work`), and a
//! queue's command buffers beside another queue's backlog complete, each waiter after its
//! signaller (`two_queues`).
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::authored::OwnRow;
use metal2vulkan_harness::{rt, Buffer, Device, Storage};

#[test]
fn a_range_given_back_is_the_next_taken_and_live_buffers_lie_apart() {
    let device = rt::system_default(None).unwrap();
    let at = |buffer: &rt::Buffer| buffer.contents().unwrap().as_ptr() as usize;
    let first = device.new_buffer(4096, Storage::Shared).unwrap();
    let second = device.new_buffer(4096, Storage::Shared).unwrap();
    let (first_at, second_at) = (at(&first), at(&second));
    assert!(
        second_at >= first_at + 4096 || second_at + 4096 <= first_at,
        "{first_at:#x} and {second_at:#x} overlap"
    );
    drop(first);
    let third = device.new_buffer(4096, Storage::Shared).unwrap();
    assert_eq!(at(&third), first_at, "the range given back is taken again");
    let private = device.new_buffer(4096, Storage::Private).unwrap();
    assert!(private.contents().is_none());
    // Past a pooled block: a dedicated block, released when the buffer is dropped.
    let long = (device.new_buffer(256 << 20 | 4096, Storage::Shared)).unwrap();
    metal2vulkan_harness::run::zeroed(&long, 256 << 20 | 4096, 4096).unwrap();
    drop(long);
    drop((second, third, private));
}

/// The runtime's row `releases_under_work` passes its check: textures released while three
/// command buffers keep the GPU running leave, at the 100th and the 200th iteration, no more
/// waiting and live than the work in flight accounts for.
#[test]
fn releases_under_work_in_flight_stay_flat() {
    let device = rt::system_default(None).unwrap();
    let report = (device.own_row(OwnRow::ReleasesUnderWork))
        .expect("a row of the runtime's own")
        .unwrap_or_else(|why| panic!("{why}"));
    println!("releases_under_work: {}", report.lines.join("; "));
}

/// The runtime's row `two_queues` passes its check: every command buffer a queue commits beside
/// another queue's backlog completes, and each that waits for a signal of the other's ends after
/// its signaller.
#[test]
fn two_queues_complete_each_waiter_after_its_signaller() {
    let device = rt::system_default(None).unwrap();
    let report = (device.own_row(OwnRow::TwoQueues))
        .expect("a row of the runtime's own")
        .unwrap_or_else(|why| panic!("{why}"));
    println!("two_queues: {}", report.lines.join("; "));
}
