//! A sampler state a kind of sampler table leaves out (`heap_sets::SamplerTableKind`: a cubic
//! state, where the kind's commands' stages do not test the states they reach for a cubic filter)
//! is announced (`Degradation::UnlistedArgumentSampler`) while a variant whose commands may bind a
//! table of the kind lives (`heap_sets::TableBinder`, from its first command), not because the
//! device keeps the table after every such variant went. A whole-set conformance replay made a
//! kernel's table of such a kind in one case and announced a cubic state another case made
//! after the kernel was gone.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The variant's first command is
//! `heap_sets::bind_kind_for_tests`, which makes the device's table of the kind and takes its
//! binder, as a command of a variant whose fragment stage reads an argument buffer's sampler does.

use super::blit_holds::Scratch;
use crate::heap_sets::bind_kind_for_tests;
use crate::{
    Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions, MessageSink, SamplerState,
    ShaderCacheOptions,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::SamplerMinMagFilter;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The notices of a sampler state left out that `heard` holds, emptied.
fn unlisted(heard: &Heard) -> Vec<DegradeNotice> {
    (heard.lock().unwrap().drain(..))
        .filter(|notice| notice.degradation == Degradation::UnlistedArgumentSampler)
        .collect()
}

/// A device of the kind the cfg selects that announces a sampler state left out at every
/// occurrence, its notices into `heard`.
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
        announce_each: [Degradation::UnlistedArgumentSampler].into_iter().collect(),
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A sampler state of `device` whose min and mag filters are `filter`, told apart from the
/// others by `n` (its minimum level of detail).
fn state(device: &Device, filter: SamplerMinMagFilter, n: u32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.lod_min_clamp = n as f32 / 64.0;
    device.new_sampler_state(&descriptor).unwrap()
}

/// Whether `notice` names a cubic state left out.
fn names_cubic(notice: &DegradeNotice) -> bool {
    notice.reason.contains("a cubic state")
}

/// The device makes the table of a kind that leaves cubic states out at a variant's first
/// command, and keeps it after the variant's binder went: a cubic state made then is announced by
/// no notice, which was announced before (every table the device kept counted). A variant of the
/// kind binding it again announces the live cubic state at its first command, and a cubic state
/// made while it lives is announced as it is made; a state that is not cubic is announced by
/// none. A kind that maps cubic states announces none of them.
#[test]
fn a_cubic_state_is_announced_while_a_variant_binds_a_kind_that_leaves_it_out() {
    let scratch = Scratch::new("sampler-table-binders");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let capacity = device.sampler_array();
    if capacity == 0 {
        println!("SKIPPED: the device has no set-1 sampler array");
        return;
    }
    // A variant of the kind binds the table, which the device makes, and goes.
    let binder = bind_kind_for_tests(&device, capacity, false).unwrap();
    assert_eq!(unlisted(&heard), [], "no state lives yet");
    drop(binder);
    let first = state(&device, SamplerMinMagFilter::Cubic, 1);
    assert_eq!(
        unlisted(&heard),
        [],
        "a cubic state made with no variant of the kind live"
    );
    // A variant of the kind binds it again: the live cubic state is announced at its first
    // command.
    let binder = bind_kind_for_tests(&device, capacity, false).unwrap();
    let notices = unlisted(&heard);
    assert_eq!(notices.len(), 1, "{notices:?}");
    assert!(names_cubic(&notices[0]), "{notices:?}");
    assert!(
        notices[0]
            .object
            .contains(&format!("resource ID {}", first.gpu_resource_id().0)),
        "{notices:?}"
    );
    // Another made while it lives is announced as it is made; a linear one is not.
    let second = state(&device, SamplerMinMagFilter::Cubic, 2);
    let notices = unlisted(&heard);
    assert_eq!(notices.len(), 1, "{notices:?}");
    assert!(names_cubic(&notices[0]), "{notices:?}");
    let linear = state(&device, SamplerMinMagFilter::Linear, 3);
    assert_eq!(unlisted(&heard), [], "a linear state");
    drop(binder);
    // A kind that maps cubic states leaves none out.
    let cubic_kind = bind_kind_for_tests(&device, capacity, true).unwrap();
    let third = state(&device, SamplerMinMagFilter::Cubic, 4);
    assert_eq!(unlisted(&heard), [], "a kind that maps cubic states");
    drop((cubic_kind, first, second, third, linear));
}
