//! `maxArgumentBufferSamplerCount` on the device the cfg selects
//! (`DeviceProperties::max_argument_buffer_sampler_count`): one fewer than the smaller of the
//! elements of a set-1 sampler array and the limit the device's live samplers are held to, each
//! read where the runtime uses it. The array is the one link gives a kernel reading a sampler from
//! an argument buffer on the device's profile, which is the device's own count
//! (`Device::sampler_array`, which each sampler state is checked against as it is made); the limit
//! is the driver's `maxSamplerAllocationCount`, which `SamplerCount` refuses past
//! (`Error::SamplerLimit`). The states that count as they are made: as many as the array holds
//! beside its neutral are made with no notice, and the next is announced
//! (`Degradation::UnlistedArgumentSampler`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `argument_sampler_sample`
//! of the owned fixture `validation/fixtures/public/engine/tail_argument_samplers.ll`
//! (`llvm-as`).

use super::blit_holds::Scratch;
use crate::{
    Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions, IdPlacement, MessageSink,
    ResourceId, SamplerState, SamplerStateOptions, ShaderCacheOptions,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The notices of a set-1 array leaving a sampler state out that `heard` holds, emptied.
fn unlisted(heard: &Heard) -> Vec<DegradeNotice> {
    (heard.lock().unwrap().drain(..))
        .filter(|notice| notice.degradation == Degradation::UnlistedArgumentSampler)
        .collect()
}

/// A device of the kind the cfg selects, its notices into `heard`.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The element count of the set-1 sampler array of the kernel's pipeline on `device`'s profile.
fn kernel_sampler_array(device: &Device) -> u32 {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/tail_argument_samplers.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let function = metal2vulkan::library::Library::from_air(&bitcode)
        .unwrap()
        .function("argument_sampler_sample")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    let pipeline = device
        .compile_device()
        .compile_compute_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    let arrays: Vec<u32> = (pipeline.layout.sets.iter())
        .filter(|set| set.set == 1)
        .flat_map(|set| &set.bindings)
        .filter(|binding| binding.ty == vk::DescriptorType::SAMPLER)
        .map(|binding| binding.count)
        .collect();
    assert_eq!(arrays.len(), 1, "one set-1 sampler array: {arrays:?}");
    arrays[0]
}

/// The answer is one fewer than the smaller of the kernel's set-1 sampler array, which is the
/// device's own count of its elements, and the driver's `maxSamplerAllocationCount`, which the
/// device's live samplers are held to. Sampler states made one after another take elements 1, 2,
/// ...: those up to one fewer than the array's elements are made with no notice, and the one at
/// the array's count is announced as left out of it. Every state is placed at one ID of the
/// sampler tables' low window, so no state is left out of them for its ID, and the device makes no
/// set-1 array, so none takes a Vulkan sampler.
#[test]
fn the_argument_buffer_sampler_count_is_the_set_1_sampler_array_and_the_sampler_limit() {
    let scratch = Scratch::new("argument-sampler-count");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let shared = device.shared();
    let p = device.properties();
    let array = kernel_sampler_array(&device);
    let elements = device.sampler_array();
    let limit = shared.sampler_count.limit();
    // SAFETY: the physical device is the live instance's.
    let driver = unsafe { (shared.instance.raw).get_physical_device_properties(shared.physical) };
    println!(
        "ARGUMENT SAMPLERS {}: maxArgumentBufferSamplerCount {}, set-1 sampler array {array} \
         (the device's {elements}), maxSamplerAllocationCount {limit} (the driver's {})",
        p.name, p.max_argument_buffer_sampler_count, driver.limits.max_sampler_allocation_count
    );
    assert_eq!(array, elements, "the kernel's array and the device's count");
    assert_eq!(limit, driver.limits.max_sampler_allocation_count);
    assert_eq!(
        p.max_argument_buffer_sampler_count,
        u64::from(elements.min(limit)) - 1
    );
    assert_eq!(unlisted(&heard), []);
    let options = SamplerStateOptions {
        support_argument_buffers: true,
        placement: IdPlacement::At(ResourceId(5)),
    };
    let state = |n: u32| -> SamplerState {
        let mut descriptor = SamplerDescriptor::new();
        descriptor.lod_min_clamp = n as f32 / 65536.0;
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        assert_eq!(state.heap_element(), u64::from(n), "state {n}'s element");
        state
    };
    let mut states: Vec<SamplerState> = (1..elements).map(state).collect();
    assert_eq!(unlisted(&heard), [], "{} states", states.len());
    states.push(state(elements));
    let notices = unlisted(&heard);
    assert_eq!(notices.len(), 1, "{notices:?}");
    assert!(
        notices[0].reason.contains(&format!("{elements} elements")),
        "{}",
        notices[0].reason
    );
    println!(
        "ARGUMENT SAMPLERS {}: {} states made with no notice, the next announced: {}",
        p.name,
        elements - 1,
        notices[0].reason
    );
    drop(states);
}
