//! A dispatch that took its key's residual form while its own key was made on the device's pool
//! (`pipeline/dispatch_making.rs`, `pipeline/render/residual.rs`), and a later dispatch of the
//! same inputs in its encoder, which takes what the first settled (`compute::DispatchMemo`): once
//! the own key's making landed, the later dispatch runs the own key's pipeline. The residual form
//! serves only while the own key is made, so the memo's entry never keeps it.
//!
//! A device test on the CPU ICD alone (`m2v_cpu_icd`): lavapipe, under a device whose view
//! check's facts the test fixes (`DeviceOptions::view_facts`) to
//! `DriverWorkaround::PushReadsLengthenThreadPayload`, stands in for a profile whose storage
//! views declare `SpecKey::RangeAtBindOffset`, where a key and its residual form differ. The
//! kernel is `engine/buffer_end_byte_load.ll`'s (`out[0] = in[5]`).

use super::blit_holds::{options, Scratch, WAIT};
use super::uniform_bind_offsets::idle;
use crate::{Buffer, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};
use metal2vulkan::device::DriverWorkaround;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The threadgroup the state requires, so its creation makes its predicted region and that
/// region's residual form (`pipeline/dispatch_making.rs`).
const THREADS: [u32; 3] = [32, 1, 1];

/// Where the source is bound: past offset 0, so its slot reads at its residual from then on
/// (`argument_table::AtBindViews`), a key no creation made.
const MOVED: u64 = 64;

/// A device whose profile's storage views declare `SpecKey::RangeAtBindOffset`.
fn device(cache: &std::path::Path) -> Device {
    let mut options = options(cache);
    let fact = DriverWorkaround::PushReadsLengthenThreadPayload;
    options.view_facts = Some([fact].into());
    let device = Device::system_default_with(&options).unwrap();
    let workarounds = device.compile_device().profile().workarounds();
    assert!(
        workarounds.contains(&fact),
        "the compile device's profile keys its storage views: {workarounds:?}"
    );
    device
}

/// The compute state of `buffer_end_byte_load`, its descriptor requiring [`THREADS`].
fn state(device: &Device) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("buffer_end_byte_load.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = (device.new_library_with_compile_library(library))
        .new_function_with_name("buffer_end_byte_load")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = Some(THREADS);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The first byte of `buffer`, which no command uses now.
fn first_byte(buffer: &Buffer) -> u8 {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's first byte, which no command uses now.
    unsafe { *contents.as_ptr() }
}

/// With the device's pool held, the first dispatch, its source at [`MOVED`], hands its own key's
/// making to the pool and takes the residual form the creation made. The pool let go and its
/// making landed, a second dispatch of the same inputs in the same encoder takes the first's
/// settle (`dispatches_settled_as_before`) and binds its own key's pipeline: one pipeline bound
/// and no residual-form dispatch more. Both read the source's byte `MOVED + 5`.
#[test]
fn a_settle_taken_again_runs_its_own_key_once_made_not_the_residual_form() {
    let scratch = Scratch::new("residual-memo");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let created = idle(&device);
    let data: Vec<u8> = (0..=255).collect();
    let source = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let output = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let [width, height, depth] = THREADS;
    let threads = Size::new(u64::from(width), u64::from(height), u64::from(depth));
    let hold = crate::pipeline::pool::held::hold();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&source), MOVED, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), threads)
        .unwrap();
    let first = device.work_counts();
    assert_eq!(
        (
            first.residual_form_draws - created.residual_form_draws,
            first.residual_form_makings - created.residual_form_makings,
            first.dispatch_pipelines_pooled - created.dispatch_pipelines_pooled,
        ),
        (1, 1, 1),
        "the first dispatch: residual-form dispatches, their makings and regions pooled"
    );
    drop(hold);
    let landed = idle(&device);
    assert_eq!(
        landed.variant_pipelines_made - first.variant_pipelines_made,
        1,
        "the own key's pipeline, made once the pool ran"
    );
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), threads)
        .unwrap();
    let second = device.work_counts();
    assert_eq!(
        second.dispatches_settled_as_before - landed.dispatches_settled_as_before,
        1,
        "the second dispatch took the first's settle"
    );
    assert_eq!(
        (
            second.residual_form_draws - landed.residual_form_draws,
            second.commands_recorded.cmd_bind_pipeline - landed.commands_recorded.cmd_bind_pipeline,
        ),
        (0, 1),
        "the second dispatch: residual-form dispatches, and pipelines bound (its own key's)"
    );
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(first_byte(&output), data[MOVED as usize + 5]);
}
