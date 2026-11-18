//! Sampler states: the device's samplers shared by create info, and on devices the states.
//!
//! - The table makes one sampler per create info and shares it among its holders; past its cap a
//!   new create info is `Limit::DeviceSamplers`, and once a sampler's last holder gives it up the
//!   sampler is the reaper's and its room is free again.
//! - Under a device cfg, on every device of the cfg's kind: a state's ID is its index and the
//!   translator's sampler word. A nearest state clamped to the edge takes one device sampler for
//!   its plain and integer-texture uses, and an equal state, or one differing only in a border
//!   colour no address mode reads, shares it; a linear state with a compare function takes three;
//!   an anisotropic state is made. A cubic filter is `Gap::CubicSampler`. A state made at an index
//!   another holds takes it, and the earlier state's going leaves it; the grant waits for the
//!   index's owner, and the index is free once that state is gone. The device's samplers come
//!   back to zero with the device. A state's constexpr samplers numbered up to one the
//!   translator refuses give back what the earlier ones took.

use crate::error::{Error, Limit};
use crate::sampler::SamplerTable;
use ash::vk::{self, Handle};
use std::cell::Cell;

#[test]
fn a_device_sampler_is_shared_by_create_info_up_to_the_cap() {
    let mut table = SamplerTable::<u32>::new(2);
    let made = Cell::new(0_u64);
    let make = || {
        made.set(made.get() + 1);
        Ok(vk::Sampler::from_raw(made.get()))
    };
    let raw = |result: Result<vk::Sampler, Error>| match result {
        Ok(sampler) => sampler.as_raw(),
        Err(error) => panic!("{error}"),
    };
    assert_eq!(raw(table.take(&1, make)), 1);
    assert_eq!(raw(table.take(&1, make)), 1, "an equal create info shares");
    assert_eq!(raw(table.take(&2, make)), 2);
    match table.take(&3, make) {
        Err(Error::Limit(limit)) => assert_eq!(limit, Limit::DeviceSamplers { max: 2 }),
        other => panic!("past the cap: {:?}", other.map(|s| s.as_raw())),
    }
    assert_eq!(made.get(), 2, "nothing is made past the cap");
    assert_eq!(table.give(&1), None, "a holder is left");
    assert_eq!(table.give(&1).map(|s| s.as_raw()), Some(1));
    assert_eq!(raw(table.take(&3, make)), 3, "the room is free again");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        CompareFunction, Device, DeviceOptions, Error, Gap, IndexClass, Refusal,
        SamplerBorderColor, SamplerDescriptor, SamplerMinMagFilter, SamplerState,
    };
    use metal2vulkan::metal::sampler::SamplerWord;

    /// The refusal of `result`.
    fn refused(result: Result<impl Sized, Error>) -> Refusal {
        match result {
            Err(Error::Refused(refusal)) => refusal,
            Err(other) => panic!("not a refusal: {other}"),
            Ok(_) => panic!("not a refusal: it was made"),
        }
    }

    #[test]
    fn every_device_shares_sampler_states_device_samplers() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let grant = device
                .grant_indices(IndexClass::Sampler, 1, 16)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = |descriptor: &SamplerDescriptor, index| -> SamplerState {
                device
                    .new_sampler_state(descriptor, index)
                    .unwrap_or_else(|error| panic!("{name}: {descriptor:?}: {error}"))
            };
            let defaults = counters.gauges().device_samplers;
            let nearest = SamplerDescriptor::new();
            let a = state(&nearest, 1);
            assert_eq!(a.gpu_resource_id().index, 1, "{name}");
            assert_eq!(
                a.gpu_resource_id().word,
                SamplerWord::of(&nearest).0,
                "{name}"
            );
            assert_eq!(counters.gauges().device_samplers, defaults + 1, "{name}");
            let b = state(&nearest, 2);
            let mut unread_border = nearest;
            unread_border.border_color = SamplerBorderColor::OpaqueWhite;
            let c = state(&unread_border, 3);
            assert_eq!(counters.gauges().device_samplers, defaults + 1, "{name}");
            let mut linear = nearest;
            linear.min_filter = SamplerMinMagFilter::Linear;
            linear.mag_filter = SamplerMinMagFilter::Linear;
            linear.compare_function = CompareFunction::Less;
            // Its plain and comparison samplers are new; its integer-texture use, nearest, reads
            // as the nearest state's plain sampler and shares it.
            let d = state(&linear, 4);
            assert_eq!(counters.gauges().device_samplers, defaults + 3, "{name}");
            let mut anisotropic = linear;
            anisotropic.compare_function = CompareFunction::Never;
            anisotropic.max_anisotropy = 8;
            let e = state(&anisotropic, 5);
            let mut cubic = nearest;
            cubic.min_filter = SamplerMinMagFilter::Cubic;
            assert!(
                matches!(
                    device.new_sampler_state(&cubic, 6),
                    Err(Error::Unsupported(Gap::CubicSampler))
                ),
                "{name}"
            );
            // A state at the index `a` holds takes it; `a`'s going leaves it.
            let later = state(&nearest, 1);
            assert_eq!(
                refused(device.release_indices(grant)),
                Refusal::IndicesInUse {
                    class: IndexClass::Sampler,
                    index: 1
                },
                "{name}"
            );
            assert_eq!(counters.gauges().samplers, 6, "{name}");
            assert_eq!(counters.gauges().descriptor_slots, 5, "{name}");
            drop(a);
            assert_eq!(counters.gauges().descriptor_slots, 5, "{name}");
            drop(later);
            assert_eq!(counters.gauges().descriptor_slots, 4, "{name}");
            state(&linear, 1);
            drop((b, c, d, e));
            assert_eq!(counters.gauges().descriptor_slots, 0, "{name}");
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            let gauges = counters.gauges();
            assert_eq!(
                (
                    gauges.samplers,
                    gauges.device_samplers,
                    gauges.descriptor_slots
                ),
                (0, 0, 0),
                "{name}"
            );
        }
    }

    /// A state's constexpr samplers, numbered up to one whose device sampler the translator does
    /// not state: the refusal gives back what the earlier one took, its arrays' elements at once
    /// and its device sampler through the reaper, and leaves its index free.
    #[test]
    fn a_refused_constexpr_sampler_gives_back_the_ones_numbered_before_it() {
        use metal2vulkan::pipeline::{CombinedSampler, NumberedSampler, SamplerRef};
        // Linear on both filters, clamped to the edge on every axis, levels 0 to 65504: no
        // default sampler shares its create info.
        let linear = [
            1 | (1 << 3) | (1 << 6) | (1 << 9) | (1 << 11) | (0x7bff << 40),
            0,
        ];
        // The same state with a NaN level bias, of which the translator makes no device sampler.
        let nan_bias = [linear[0], 0x7e00];
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in devices {
            let device = Device::new(&physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let grant = device
                .grant_indices(IndexClass::Sampler, 1, 2)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let before = counters.gauges();
            let samplers = vec![
                NumberedSampler::Constexpr(linear),
                NumberedSampler::Constexpr(nan_bias),
            ];
            let uses = [linear, nan_bias]
                .map(|words| CombinedSampler::new(SamplerRef::Constexpr(words), false, false));
            let numbered = crate::pipeline::number_samplers(
                device.core(),
                samplers,
                &[(0, uses[0]), (1, uses[1])],
                |_| Ok(vec![1, 2]),
                || Error::Unsupported(Gap::KernelInput),
            );
            assert!(
                matches!(numbered, Err(Error::Translation(_))),
                "{name}: the NaN bias is the translator's refusal"
            );
            assert_eq!(
                counters.gauges().descriptor_slots,
                before.descriptor_slots,
                "{name}: the linear sampler's elements are given back"
            );
            crate::tests::command::on_device::within("the reaper frees the device sampler", || {
                counters.gauges().device_samplers == before.device_samplers
            });
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: index 1 is free: {error}"));
        }
    }
}
