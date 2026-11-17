//! The new runtime's buffers through the harness, on the device the cfg names
//! (docs/VALIDATION.md): every creation workload's buffers start zero, a range given back is the
//! next one taken, buffers alive at once lie apart, and a buffer longer than a pooled block takes
//! a block of its own.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::run::{Buffers, Runs, Session};
use metal2vulkan_harness::{authored, rt, Buffer, Device, Storage};

#[test]
fn every_creation_workload_starts_zero_on_the_new_runtime() {
    let mut session = Buffers(rt::system_default().unwrap());
    let runs = Runs {
        warmup: 0,
        timed: 1,
    };
    for create in authored::CREATES {
        let name = create.name;
        let report =
            (session.create(create.length, runs)).unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(report.create_us.len(), 1, "{name}");
    }
}

#[test]
fn a_range_given_back_is_the_next_taken_and_live_buffers_lie_apart() {
    let device = rt::system_default().unwrap();
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
