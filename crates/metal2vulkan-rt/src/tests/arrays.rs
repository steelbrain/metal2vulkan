//! Set 1's index grants, and on devices its fill.
//!
//! - A grant takes a range inside its class's (textures 1 to 65,535, samplers 1 to 3,999); index
//!   0, an empty range and one past the class's are
//!   `Limit::IndexRange`. An index granted already is `Refusal::IndicesGranted`, of any range that
//!   holds it; a release of an index not granted is `Refusal::IndicesNotGranted`. A released range
//!   is granted again.
//! - Under a device cfg, on every device of the cfg's kind, the device writes every element of
//!   set 1 its default at creation, and holds two `VkSampler`s for it; grants work through the
//!   device; `Device::new`, and making and destroying a second set 1, are timed and reported,
//!   and the second set's two samplers are gone after it; every device supports both of the
//!   runtime's set layouts, which `Device::new` asks before it makes them; and the arrays fill
//!   past their capacity: every index of each class takes an object, the samplers' through one
//!   grant and the textures' through one for a texture of shader-write usage and one for views
//!   of it at every other index, and objects dropped one at a time give their index to the next,
//!   more than 4,000 sampler states and 65,536 textures in all, none refused as in use, the
//!   descriptor-slot gauge back to zero after them.

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
        (IndexClass::Texture, 1, 65_536),
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
        .grant(IndexClass::Texture, 10, 5)
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
    use crate::arrays::{supported, with_arrays_layout, Arrays, BINDINGS};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Device, DeviceOptions, IndexClass, Instance, PixelFormat, Refusal, SamplerDescriptor,
        StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;
    use std::ops::Range;
    use std::time::Instant;

    /// How many objects of each class are made again at an index another left, past the
    /// class's capacity.
    const AGAIN: u32 = 128;

    /// One object `make` makes at every index of `indices` of `class`, through one grant of
    /// them, then [`AGAIN`] of them each dropped and made again at the index it left, as a
    /// consumer reuses an index: the descriptor-slot gauge counts them on top of what it counted
    /// before while they live, and is back to that once they are gone and the grant is released.
    fn fill_past_capacity<T>(
        name: &str,
        device: &Device,
        (class, indices): (IndexClass, Range<u32>),
        make: impl Fn(u32) -> T,
    ) {
        let counters = device.counters();
        let before = counters.gauges().descriptor_slots;
        let Range { start, end } = indices;
        let grant = (device.grant_indices(class, start, end - start))
            .unwrap_or_else(|error| panic!("{name}: {class:?}: {error}"));
        let mut live: Vec<Option<T>> = (start..end).map(|index| Some(make(index))).collect();
        let full = before + u64::from(end - start);
        assert_eq!(
            counters.gauges().descriptor_slots,
            full,
            "{name}: {class:?}"
        );
        for (object, index) in live.iter_mut().zip(start..start + AGAIN) {
            drop(object.take());
            *object = Some(make(index));
        }
        assert_eq!(
            counters.gauges().descriptor_slots,
            full,
            "{name}: {class:?}"
        );
        drop(live);
        assert_eq!(
            counters.gauges().descriptor_slots,
            before,
            "{name}: {class:?}"
        );
        (device.release_indices(grant)).unwrap_or_else(|error| panic!("{name}: {error}"));
    }

    /// Set 1's arrays filled past their capacity: 3,999 sampler states and 128 more, through one
    /// grant; a texture of shader-write usage at the first texture index and 65,534 views of it
    /// at every other, both elements written at each, and 128 views more. Every make is
    /// unwrapped, so an index refused as in use (`Refusal::IndexInUse`) fails the test.
    #[test]
    fn every_device_fills_set_1_past_its_capacity() {
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
            let sampler = SamplerDescriptor::new();
            let class = IndexClass::Sampler;
            fill_past_capacity(&name, &device, (class, class.range()), |index| {
                (device.new_sampler_state(&sampler, index))
                    .unwrap_or_else(|error| panic!("{name}: sampler {index}: {error}"))
            });
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, 1);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Private;
            let Range { start, end } = IndexClass::Texture.range();
            let first = (device.grant_indices(IndexClass::Texture, start, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = (device.new_texture(&d, Some(start)))
                .unwrap_or_else(|error| panic!("{name}: texture {start}: {error}"));
            let rest = (IndexClass::Texture, start + 1..end);
            fill_past_capacity(&name, &device, rest, |index| {
                (texture.new_texture_view(d.pixel_format, d.texture_type, 0..1, 0..1, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: view {index}: {error}"))
            });
            drop(texture);
            (device.release_indices(first)).unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.samplers, gauges.textures, gauges.descriptor_slots),
                (0, 0, 0),
                "{name}"
            );
        }
    }

    /// The floor's last item, asked of the created device: every device of the cfg's kind
    /// supports both of the runtime's set layouts (`vkGetDescriptorSetLayoutSupport`), the empty
    /// set 0 and set 1 of the arrays, which `Device::new` asks before it makes them.
    #[test]
    fn every_device_supports_the_runtimes_set_layouts() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let raw = &device.core().device;
            let empty = vk::DescriptorSetLayoutCreateInfo::default();
            // SAFETY: the device is live, with the floor's features.
            unsafe { supported(raw, &empty, 0) }.unwrap_or_else(|error| panic!("{name}: {error}"));
            with_arrays_layout(BINDINGS.map(|(_, _, count)| count), |info| {
                // SAFETY: the device is live, with the floor's features.
                unsafe { supported(raw, info, 1) }
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("floor: {name}: set layouts 0 and 1 supported");
        }
    }

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
            let opened = Instant::now();
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let micros = opened.elapsed().as_micros();
            println!("M2V-MATRIX device_new/{name} {micros} us");
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
            let second = unsafe { Arrays::new(&core.device, &counters, false) }
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
