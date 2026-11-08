//! A degrade notice names the command buffer whose work announced it
//! (`DegradeNotice::command_buffer`, `CommandBuffer::object_id`), on whichever thread the work
//! runs: an indirect command buffer's execution recorded at a host step on the scheduler's steps
//! thread announces `released-command-state` there naming the executing command buffer, its commit
//! announces `host-replayed-indirect-commands` on the program's thread naming it too, and a notice
//! of the device's (`private-resource-option`, a buffer of commands made while another command
//! buffer's encoder is open) names none.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The kernel is `scale_and_offset` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use crate::write_routes::GpuWriter;
use crate::{
    CommandBufferStatus, ComputePipelineState, Degradation, DegradeNotice, Device, DeviceKind,
    DeviceOptions, IndirectCommandBufferDescriptor, MessageSink, PipelineOptions, ResourceOptions,
    ShaderCacheOptions, Size,
};
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

/// What the sink heard: each notice, with the thread that announced it and whether that thread
/// was recording a host step then.
type Heard = Arc<Mutex<Vec<(DegradeNotice, ThreadId, bool)>>>;

/// The threads each dispatch runs.
const COUNT: u32 = 64;

/// A device of the kind the cfg selects, its shader cache in `cache`, its notices into `heard`.
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
                let step = crate::exit_guard::in_host_step();
                let thread = std::thread::current().id();
                heard.lock().unwrap().push((notice.clone(), thread, step));
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A state of `scale_and_offset` that an indirect command buffer's commands may set.
fn state(device: &Device) -> ComputePipelineState {
    let function = library(device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A command buffer executes a buffer of commands whose records the GPU may write, its one command
/// set to a state the program released after another state took its ID: the commit leaves the
/// execution to a host step, whose recording cannot tell whether the GPU wrote the released
/// state's ID while the other held it (`Unmatched::Unsure`) and announces
/// `released-command-state`. Both that notice, heard at the host step on a thread other than the
/// program's, and the commit's `host-replayed-indirect-commands`, heard on the program's thread
/// outside any host step, name the executing command buffer, not the one made before it; the
/// device's `private-resource-option`, of a buffer of commands made while the executing command
/// buffer's encoder is open, names none.
#[test]
fn notices_name_the_command_buffer_whose_work_announced_them() {
    let scratch = Scratch::new("notice-command-buffers");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let shared = ResourceOptions::shared();
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device.new_buffer_with_bytes(&input, shared).unwrap();
    let output = device.new_buffer(u64::from(COUNT) * 4, shared).unwrap();
    let scale = device
        .new_buffer_with_bytes(&2.0f32.to_le_bytes(), shared)
        .unwrap();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device
        .new_indirect_command_buffer(&descriptor, 1, shared)
        .unwrap();
    let released = state(&device);
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&released).unwrap();
    for (index, buffer) in (0..).zip([&input, &output, &scale]) {
        command.set_kernel_buffer(Some(buffer), 0, index).unwrap();
    }
    let (grid, group) = (
        Size::new(u64::from(COUNT), 1, 1),
        Size::new(COUNT.into(), 1, 1),
    );
    command.concurrent_dispatch_threads(grid, group).unwrap();
    // The program releases the state, another takes its ID, and then the GPU may write the
    // records: one may name the ID as the other held it.
    let id = released.gpu_resource_id();
    drop(released);
    let holder = state(&device);
    assert_eq!(holder.gpu_resource_id(), id);
    icb.written_by_gpu(GpuWriter::Reset);
    drop(icb.gpu_write());

    let queue = device.new_command_queue().unwrap();
    let before = queue.command_buffer().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let id = command_buffer.object_id();
    assert_ne!(before.object_id(), id);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    // The device's notice, made between the encoder's calls.
    let private = ResourceOptions::from_raw(shared.raw() | ResourceOptions::PRIVATE_BIT).unwrap();
    let _private = device
        .new_indirect_command_buffer(&descriptor, 1, private)
        .unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);

    let program = std::thread::current().id();
    let heard = heard.lock().unwrap();
    let find = |kind: Degradation| {
        (heard.iter())
            .find(|(notice, ..)| notice.degradation == kind)
            .unwrap_or_else(|| panic!("no {}: {heard:?}", kind.code()))
    };
    let (notice, thread, step) = find(Degradation::ReleasedCommandState);
    assert!(*step, "released-command-state outside a host step");
    assert_ne!(*thread, program, "a host step on the program's thread");
    assert_eq!(notice.command_buffer, Some(id), "{notice:?}");
    let (notice, thread, step) = find(Degradation::HostReplayedIndirectCommands);
    assert!(
        !*step && *thread == program,
        "the commit's notice elsewhere"
    );
    assert_eq!(notice.command_buffer, Some(id), "{notice:?}");
    let (notice, ..) = find(Degradation::PrivateResourceOption);
    assert_eq!(notice.command_buffer, None, "{notice:?}");
    drop(before);
}
