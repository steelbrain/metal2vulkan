//! The device's check of the first move in place (`in_place_check.rs`) takes no range of the Metal
//! address space: its buffers, one per class of pages, lie outside it (`address.rs`, "Outside the
//! space"), so no pin or pick of the program's depends on whether or when the check runs, and the
//! space's state changes no verdict.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Each device's shader cache is a fresh
//! directory, so no record holds its verdicts and its check runs.

use super::blit_holds::{device, Scratch};
use crate::address::{PAGE, SPACE_BASE, SPACE_END};
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, Device, Error, GpuAddress,
    ResourceOptions,
};
use std::time::{Duration, Instant};

/// The longest a check may take.
const WAIT: Duration = Duration::from_secs(120);

/// The low part of the space the pins cover: its first GiB.
const LOW: u64 = 1 << 30;

/// A pin every this many bytes of [`LOW`], of [`PINNED`] bytes: every second one held through the
/// check leaves no free run of the space a check's buffer of 40 MiB fits in.
const STRIDE: u64 = 8 << 20;
const PINNED: u64 = 16 << 10;

/// A buffer of [`PINNED`] bytes pinned at `at`.
fn pin(device: &Device, at: u64) -> Result<Buffer, Error> {
    let descriptor = BufferDescriptor::new(PINNED, ResourceOptions::shared(), BufferContents::Zero)
        .with_placement(AddressPlacement::At(GpuAddress(at)));
    device.new_buffer_with_descriptor(descriptor)
}

/// The pin addresses of [`LOW`] at even multiples of [`STRIDE`], or at odd ones.
fn pins(odd: bool) -> impl Iterator<Item = u64> {
    (0..LOW / STRIDE)
        .filter(move |k| k % 2 == u64::from(odd))
        .map(|k| SPACE_BASE + k * STRIDE)
}

/// A device whose check runs with the Metal space held but for runs shorter than a check's
/// buffer (the space past its first GiB by one pool, a pin every 16 MiB of that GiB) answers
/// every class as a device whose check ran with the space empty does, and each pin between the
/// held ones, made and released in turn until the check settles, is made. A check whose buffer
/// took a range of the space would find none: that class would be unchecked, or a pin refused
/// (`PlacementConflict::Held`).
#[test]
fn the_in_place_check_takes_no_range_of_the_metal_space() {
    let verdicts = |device: &Device| -> Vec<String> {
        device.in_place_check_report().into_iter().take(3).collect()
    };
    let reference = {
        let scratch = Scratch::new("in-place-check-reference");
        let device = device(&scratch.0);
        assert!(
            device.settle_in_place_checks(WAIT),
            "the reference check answered"
        );
        verdicts(&device)
    };
    let scratch = Scratch::new("in-place-check-placement");
    let device = device(&scratch.0);
    let rest = SPACE_BASE + LOW;
    // The pool's span is its pages and one guard page, to the space's end.
    let pool = device
        .reserve_address_pool(
            AddressPlacement::At(GpuAddress(rest)),
            SPACE_END - rest - PAGE,
        )
        .unwrap();
    let held: Vec<Buffer> = pins(false)
        .map(|at| pin(&device, at).unwrap_or_else(|error| panic!("the pin at {at:#x}: {error}")))
        .collect();
    let started = Instant::now();
    let mut rounds = 0;
    while !device.settle_in_place_checks(Duration::ZERO) {
        assert!(started.elapsed() < WAIT, "the check answered");
        let made: Vec<Buffer> = pins(true)
            .map(|at| {
                pin(&device, at).unwrap_or_else(|error| {
                    panic!("the pin at {at:#x} while the check ran: {error}")
                })
            })
            .collect();
        drop(made);
        rounds += 1;
    }
    let answered = verdicts(&device);
    eprintln!(
        "IN-PLACE-CHECK-PLACEMENT on {}: {answered:?} with the space held, {rounds} rounds of \
         pins while it ran; {reference:?} with it empty",
        device.name()
    );
    assert_eq!(answered, reference, "the verdicts with the space held");
    drop((held, pool));
}
