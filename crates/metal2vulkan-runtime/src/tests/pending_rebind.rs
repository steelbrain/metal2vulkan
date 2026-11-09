//! A dispatch whose `VkPipeline`s are still being made (`pending_dispatch.rs`) and a rebind of a
//! no-copy buffer it named, made after its command buffer's commit and before the command
//! buffer's first submission (`rebind.rs`).
//!
//! Such a dispatch's commands are recorded later -- by the scheduler's host step once the makings
//! are done -- and they are the commands its encoding resolved: its set 0 names the `VkBuffer`
//! the buffer lay in as the dispatch was encoded. A rebind meanwhile frees that `VkBuffer`, since
//! a command buffer committed and not yet submitted holds the pages its commands took only weakly
//! (`first_submission.rs`), and the submission that holds the dispatch then runs nothing:
//! `CommandBufferError::EncodedUseRebound`, the encoder having kept no log a rebind encodes it
//! again from (`Unlogged::NoCalls`: the encoder opened while the device had no buffer over host
//! pages). So the host step records none of its commands.
//!
//! Recording them named a freed `VkBuffer` in a bound descriptor, which the validation layer
//! reports (`VUID-vkCmdDispatch-None-08114`) and a driver that reads a descriptor as it encodes
//! reads freed memory for: the Vulkan specification's Descriptor Set Binding has the contents a
//! binding without `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` is bound with consumed "during
//! host execution of the command, or during shader execution of the resulting draws and
//! dispatches, or any time in between", so a descriptor must be valid as its command is recorded.
//!
//! The device's pool's test hold (`pool::held`) keeps the makings waiting until the rebind is
//! made, so the host step records after it in every run, not only under load.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`, as
//! `tests/dispatch_pending.rs` runs it.

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use super::dispatch_pending::state;
use super::layer_claim::Watched;
use crate::pipeline::pool::held;
use crate::{
    CommandBufferError, CommandBufferStatus, Device, DeviceKind, DeviceOptions, ResourceOptions,
    ShaderCacheOptions, Size,
};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The threads the dispatch runs, in threadgroups of this many: a `dispatchThreads` of a size the
/// state's creation predicts none of, so its making goes to the device's pool.
const COUNT: u64 = 1000;
const PER_GROUP: u64 = 64;

/// The scale the kernel multiplies by.
const SCALE: f32 = 2.0;

/// The device the cfg selects, its shader cache in `cache`, asking for the validation layer and
/// reporting to `watched`.
pub(super) fn device(watched: &Watched, cache: &Path) -> Device {
    let options = DeviceOptions {
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        layers: watched.layers(),
        messages: Some(watched.sink(|_| {})),
        ..options()
    };
    Device::system_default_with(&options).unwrap()
}

/// The options of a device of the kind the cfg selects: the CPU's under `m2v_cpu_icd`, else a
/// GPU's.
pub(super) fn options() -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    }
}

/// A dispatch of a kernel writing into a no-copy buffer, its `VkPipeline` held from being made,
/// and the buffer rebound after the commit: the command buffer fails `EncodedUseRebound`, writing
/// nothing into either range of pages, and the host step that records the dispatch once its
/// making is done records nothing of it -- the validation layer reports no descriptor naming the
/// `VkBuffer` the rebind freed.
#[test]
fn a_pending_dispatch_records_nothing_where_a_rebind_freed_what_its_descriptors_name() {
    // Declared before the device, so dropped after it.
    let hosts = [Host::new(), Host::new()];
    let watched = Watched::default();
    let scratch = Scratch::new("pending-rebind");
    let device = device(&watched, &scratch.0);
    let state = state(&device);
    let queue = device.new_command_queue().unwrap();
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device
        .new_buffer_with_bytes(&input, ResourceOptions::shared())
        .unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    // Every making the pool starts waits from here: only a host step records the dispatch, and
    // only once this hold is gone.
    let hold = held::hold();
    let command_buffer = queue.command_buffer().unwrap();
    // Opened while the device has no buffer over host pages, so the encoder logs no call and no
    // rebind encodes it again (`Unlogged::NoCalls`).
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(output) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder.set_buffer(Some(&scale), 0, 2).unwrap();
    let pooled = device.work_counts().dispatch_pipelines_pooled;
    encoder
        .dispatch_threads(Size::new(COUNT, 1, 1), Size::new(PER_GROUP, 1, 1))
        .unwrap();
    assert!(
        device.work_counts().dispatch_pipelines_pooled > pooled,
        "the dispatch handed a making to the pool, so a host step records it"
    );
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    // The buffer lies elsewhere from here, and the pages it lay in go: nothing submitted holds
    // them, and the committed command buffer holds them weakly.
    //
    // SAFETY: the pages outlive every buffer over them, and only its commands use them while it
    // lives.
    unsafe { output.rebind_bytes_no_copy(hosts[1].memory(), None, None) }.unwrap();
    // The makings may finish now, so the host step records the dispatch after the rebind.
    drop(hold);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::EncodedUseRebound),
        "the submission holding the dispatch ran nothing"
    );
    assert!(hosts[0].holds(0), "the pages the dispatch took");
    assert!(hosts[1].holds(0), "the pages the buffer lies in now");
    watched.layer_reported_nothing(
        "a dispatch recorded after a rebind of a buffer it named bound no freed VkBuffer",
    );
}
