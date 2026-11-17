//! Set 1's index grants, and on devices its fill.
//!
//! - A grant takes a range inside its class's (storage-capable textures 1 to 4,095, other textures
//!   4,096 to 65,535, samplers 1 to 3,999); index 0, an empty range and one past the class's are
//!   `Limit::IndexRange`. An index granted already is `Refusal::IndicesGranted`, of any range that
//!   holds it; a release of an index not granted is `Refusal::IndicesNotGranted`. A released range
//!   is granted again.
//! - Under a device cfg, on every device of the cfg's kind, the device writes every element of
//!   set 1 its default at creation, and holds two `VkSampler`s for it; grants work through the
//!   device; making and destroying a second set 1 is timed and reported, and its two samplers
//!   are gone after it.

use crate::arrays::Grants;
use crate::error::{Error, Limit, Refusal};
use crate::IndexClass;

/// The refusal of `result`.
fn refusal<T: std::fmt::Debug>(result: Result<T, Error>) -> Refusal {
    match result {
        Err(Error::Refused(refusal)) => refusal,
        other => panic!("not a refusal: {other:?}"),
    }
}

/// The limit of `result`.
fn limit<T: std::fmt::Debug>(result: Result<T, Error>) -> Limit {
    match result {
        Err(Error::Limit(limit)) => limit,
        other => panic!("not a limit: {other:?}"),
    }
}

/// Each class's range, and a grant at either end of it.
#[test]
fn a_grant_takes_a_range_inside_its_class() {
    for (class, first, end) in [
        (IndexClass::StorageTexture, 1, 4096),
        (IndexClass::Texture, 4096, 65_536),
        (IndexClass::Sampler, 1, 4000),
    ] {
        assert_eq!(class.range(), first..end, "{class:?}");
        let mut grants = Grants::new();
        let whole = grants
            .grant(class, first, end - first)
            .unwrap_or_else(|error| panic!("{class:?}: {error}"));
        assert_eq!(
            (whole.class, whole.first, whole.count),
            (class, first, end - first)
        );
        grants.release(whole).expect("the release");
        for (first, count) in [(0, 1), (first - 1, 2), (end - 1, 2), (end, 1), (first, 0)] {
            assert_eq!(
                limit(grants.grant(class, first, count)),
                Limit::IndexRange {
                    class,
                    first,
                    count
                },
                "{class:?} {first} {count}"
            );
        }
        assert_eq!(
            limit(grants.grant(class, u32::MAX, 2)),
            Limit::IndexRange {
                class,
                first: u32::MAX,
                count: 2
            }
        );
    }
}

/// An index has one grant at a time; a released range is granted again; the classes are apart.
#[test]
fn an_index_has_one_grant_at_a_time() {
    let mut grants = Grants::new();
    let class = IndexClass::Sampler;
    let held = grants.grant(class, 10, 5).expect("the grant");
    for (first, count, index) in [(10, 5, 10), (14, 3, 14), (8, 3, 10)] {
        assert_eq!(
            refusal(grants.grant(class, first, count)),
            Refusal::IndicesGranted { class, index },
            "{first} {count}"
        );
    }
    // The same numbers of another class are free.
    grants
        .grant(IndexClass::StorageTexture, 10, 5)
        .expect("another class's grant");
    let next = grants.grant(class, 15, 1).expect("the next index");
    grants.release(held).expect("the release");
    assert_eq!(
        refusal(grants.release(held)),
        Refusal::IndicesNotGranted { class, index: 10 }
    );
    // A range partly granted is not released.
    let mut partly = next;
    partly.first = 14;
    partly.count = 2;
    assert_eq!(
        refusal(grants.release(partly)),
        Refusal::IndicesNotGranted { class, index: 14 }
    );
    grants.grant(class, 10, 5).expect("the range again");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::refusal;
    use crate::arrays::Arrays;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{Device, DeviceOptions, IndexClass, Instance, Refusal};
    use std::time::Instant;

    /// Set 1 is made with the device: its two default samplers are live while it is, grants
    /// work through it, and making a second set 1 with every element written is timed.
    #[test]
    fn set_1_is_filled_at_creation() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            assert_eq!(counters.gauges().device_samplers, 2, "{name}");
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, 64)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                refusal(device.grant_indices(IndexClass::Texture, 4100, 1)),
                Refusal::IndicesGranted {
                    class: IndexClass::Texture,
                    index: 4100
                },
                "{name}"
            );
            device.release_indices(grant).expect("the release");
            let core = device.core();
            let start = Instant::now();
            // SAFETY: the device is live and made at the floor.
            let second = unsafe { Arrays::new(&core.device, &counters) }
                .unwrap_or_else(|error| panic!("{name}: a second set 1: {error}"));
            let micros = start.elapsed().as_micros();
            assert_eq!(counters.gauges().device_samplers, 4, "{name}");
            // SAFETY: nothing used the second set 1 or a layout made from it.
            unsafe { second.destroy(&core.device, &counters) };
            assert_eq!(counters.gauges().device_samplers, 2, "{name}");
            println!("M2V-MATRIX set1_make/{name} {micros} us");
            drop(device);
            assert_eq!(counters.gauges().device_samplers, 0, "{name}");
        }
    }
}
