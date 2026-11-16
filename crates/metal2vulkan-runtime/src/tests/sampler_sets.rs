//! The device's immutable sampler sets (`pipeline/objects.rs`): a layout's baked samplers are one
//! set per device for every variant instance whose layout bakes the same ones, counted against the
//! device's limit once, from the set's making until a program's call settles the device's map once
//! nothing else holds it; a host step takes the device's set and counts nothing, and on a miss
//! makes a set of its own that no lookup finds, gone at the map's next sweep or settle once nothing
//! else holds it (only the map destroys).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::device::lock;
use crate::pipeline::objects::sampler_set;
use crate::{Device, DeviceKind, DeviceOptions};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::sampler::{vk_sampler, SamplerUse};
use metal2vulkan::metal::SamplerAddressMode;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Two takings of the same samplers are one set, counted once; a host step takes the device's
/// set and counts nothing; a set goes, and its count with it, only where the device's map is
/// settled once nothing else holds it, in a program's call; other samplers make another set; a
/// host step's miss makes a set of its own, which no lookup finds and which goes at the next
/// settle once its holder let it go.
#[test]
fn equal_baked_samplers_are_one_set_counted_once() {
    let device = device();
    let shared = device.shared();
    let profile = device.compile_device().profile();
    let desc = |mode| {
        let mut descriptor = SamplerDescriptor::default();
        descriptor.s_address_mode = mode;
        vk_sampler(&descriptor, SamplerUse::default(), profile).unwrap()
    };
    let settle = || {
        let gone = lock(&shared.sampler_sets).settle();
        gone.destroy(shared);
    };
    let descs = vec![
        desc(SamplerAddressMode::Repeat),
        desc(SamplerAddressMode::ClampToEdge),
    ];
    let live = shared.sampler_count.live();
    let (first, made) = sampler_set(&device, descs.clone()).unwrap();
    assert!(made);
    assert_eq!(
        shared.sampler_count.live(),
        live + 2,
        "the set's two samplers"
    );
    let (again, made) = sampler_set(&device, descs.clone()).unwrap();
    assert!(!made && std::ptr::eq(&*first, &*again), "the device's set");
    let (stepped, made) =
        crate::exit_guard::host_step(|| sampler_set(&device, descs.clone())).unwrap();
    assert!(
        !made && std::ptr::eq(&*first, &*stepped),
        "a host step takes the device's set"
    );
    assert_eq!(shared.sampler_count.live(), live + 2, "counted once");
    let (other, made) = sampler_set(&device, vec![desc(SamplerAddressMode::MirrorRepeat)]).unwrap();
    assert!(
        made && !std::ptr::eq(&*first, &*other),
        "other samplers, another set"
    );
    assert_eq!(shared.sampler_count.live(), live + 3);
    drop((first, again, other));
    assert_eq!(
        shared.sampler_count.live(),
        live + 3,
        "the map holds them until it is settled"
    );
    settle();
    assert_eq!(
        shared.sampler_count.live(),
        live + 2,
        "the host step's holder keeps its set"
    );
    drop(stepped);
    settle();
    assert_eq!(
        shared.sampler_count.live(),
        live,
        "settled once nothing holds it"
    );
    // A host step's miss: its own set, counted while it holds it, never the device's. (The same
    // samplers in another order are another layout's.)
    let mirrored = vec![
        desc(SamplerAddressMode::ClampToEdge),
        desc(SamplerAddressMode::Repeat),
    ];
    let (own, made) =
        crate::exit_guard::host_step(|| sampler_set(&device, mirrored.clone())).unwrap();
    assert!(made);
    assert_eq!(shared.sampler_count.live(), live + 2);
    let (program, made) = sampler_set(&device, mirrored).unwrap();
    assert!(
        made && !std::ptr::eq(&*own, &*program),
        "the step's set is not the device's"
    );
    assert_eq!(shared.sampler_count.live(), live + 4);
    drop(own);
    assert_eq!(
        shared.sampler_count.live(),
        live + 4,
        "only the map destroys: the step's set waits for a sweep"
    );
    settle();
    assert_eq!(
        shared.sampler_count.live(),
        live + 2,
        "the step's own set went at the settle, its holder gone"
    );
    drop(program);
    settle();
    assert_eq!(shared.sampler_count.live(), live);
    // A layout that bakes none: an empty set, counted nowhere.
    let (empty, made) = sampler_set(&device, Vec::new()).unwrap();
    assert!(!made);
    drop(empty);
    assert_eq!(shared.sampler_count.live(), live);
}
