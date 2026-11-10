//! Device tests of a predicated loop that runs again while Metal's GPU recovers from a fault or a
//! hang of primitive type 9 elsewhere on the device, on the device the cfg selects, under the
//! validation layer.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1: a `While` loop of 160,000 passes on
//! a queue, and 50 ms on two command buffers drawing type 9 on another queue or the same one):
//! a fault failed the loop's command buffer as a victim of the GPU's recovery
//! (`MTLCommandBufferErrorInternal`) as the faulting one completed, and a hang ended the loop
//! with no error as the hanging one failed; either way nothing after the loop ran. A loop
//! committed behind faults that the GPU recovered from before it started ran all its passes.
//!
//! Each check commits a loop without bound (`while (log[1] != 1) mark 0`, then mark 2) on a queue
//! of its own, waits until the runtime runs its laying again, then commits the draw of type 9 on
//! another queue: without the stop, the loop would run until the process ends. A command buffer a
//! program commits as failed with Metal's code of a page fault or a hang (3 or 2) stands for that
//! error, and stops the loop as it does; one of another code does not.

use super::{
    buffer_with, layer_reported_nothing_at_all, one, read_buffer, Draw, Lane, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::runtime_lane::at_plan;
use crate::runtime_lane::validated_device_with;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommitFailure,
    ComputeCommandEncoder, ComputePipelineState, Device, PipelineOptions, Size,
};
use std::time::Instant;

/// The owned synthetic kernel `lp_mark`: adds 1 to word `t[0]` of `log`.
const KERNEL: &str = r#"; Owned synthetic kernel: a loop's marks.
source_filename = "loops.metal"

define void @lp_mark(ptr addrspace(2) %t, ptr addrspace(1) %log) {
entry:
  %i = load i32, ptr addrspace(2) %t, align 4
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %i64
  %cur = load i32, ptr addrspace(1) %o, align 4
  %n = add i32 %cur, 1
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @lp_mark, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"log"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Metal's comparison code `!=`.
const NOT_EQUAL: u64 = 1;

/// The marking kernel's state on `device`.
fn mark_state(device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNEL).unwrap())
        .unwrap();
    let function = library.new_function_with_name("lp_mark").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Mark `tag` into `log` from `encoder`.
fn mark(encoder: &mut ComputeCommandEncoder, state: &ComputePipelineState, log: &Buffer, tag: u32) {
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_bytes(&tag.to_le_bytes(), 0).unwrap();
    encoder.set_buffer(Some(log), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threads(one, one).unwrap();
}

/// A loop's command buffer on a queue of its own: `while (log[word] != until) mark 0`, then mark
/// 2, committed.
fn looping(
    device: &Device,
    state: &ComputePipelineState,
    log: &Buffer,
    (word, until): (u64, u32),
) -> CommandBuffer {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .encode_start_while(Some(log), 4 * word, NOT_EQUAL, until)
        .unwrap();
    mark(&mut encoder, state, log, 0);
    assert_eq!(encoder.encode_end_while(), Ok(true));
    mark(&mut encoder, state, log, 2);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// The log's words 0 to 2.
fn words(log: &Buffer) -> [u32; 3] {
    let bytes = read_buffer(log, 0, 12).unwrap();
    std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
}

/// Wait until `device` ran a loop's laying again `more` times past `before`.
fn ran_again(device: &Device, before: u64, more: u64) {
    let start = Instant::now();
    while device.work_counts().predicated_loop_runs_again < before + more {
        assert!(
            start.elapsed() < WAIT,
            "the loop's laying ran again too few times"
        );
        std::thread::yield_now();
    }
}

/// A running loop without bound stops at the GPU's recovery from a type-9 draw on another queue:
/// a fault (an indexed draw) fails its command buffer as a victim, a hang (a direct draw) ends it
/// with no error, and nothing after the loop runs either way. A loop committed after a fault the
/// GPU recovered from runs every pass.
#[test]
fn a_running_loop_stops_at_the_gpu_s_recovery_as_metal_s_does() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipeline = lane.pipeline("pt_plain");
        let state = mark_state(&lane.device);
        let cases = [
            (
                Draw::Indexed {
                    indices: 3,
                    instances: 1,
                },
                CommandBufferError::PageFault,
                Some(CommandBufferError::GpuRecoveryVictim),
            ),
            (
                Draw::Direct {
                    vertices: 3,
                    instances: 1,
                },
                CommandBufferError::GpuHang,
                None,
            ),
        ];
        for (draw, error, loop_error) in cases {
            let label = format!("{draw:?} on {name}");
            let log = buffer_with(&lane.device, &[0; 16]).unwrap();
            let before = lane.device.work_counts().predicated_loop_runs_again;
            let looped = looping(&lane.device, &state, &log, (1, 1));
            ran_again(&lane.device, before, 2);
            let other = lane.device.new_command_queue().unwrap();
            let (drawn, _, _) = one(&lane, &other, &pipeline, Some(draw));
            assert_eq!(
                drawn.error,
                Some(error),
                "{label}: the draw's command buffer"
            );
            looped.wait_until_completed_for(WAIT).unwrap();
            let status = match loop_error {
                Some(_) => CommandBufferStatus::Error,
                None => CommandBufferStatus::Completed,
            };
            assert_eq!(
                looped.status(),
                status,
                "{label}: the loop's command buffer"
            );
            assert_eq!(
                looped.error(),
                loop_error,
                "{label}: the loop's command buffer"
            );
            let [passes, _, after] = words(&log);
            assert!(passes > 0, "{label}: the loop ran no pass");
            assert_eq!(after, 0, "{label}: the mark after the loop ran");
            println!(
                "{label}: the loop stopped after {passes} passes, {:?} {:?}",
                looped.status(),
                looped.error()
            );
        }
        // A fault the GPU recovered from before the loop's first submission leaves it running:
        // a loop of 200 passes, past the runtime's copies, runs them all.
        let other = lane.device.new_command_queue().unwrap();
        let fault = Draw::Indexed {
            indices: 3,
            instances: 1,
        };
        let (drawn, _, _) = one(&lane, &other, &pipeline, Some(fault));
        assert_eq!(drawn.error, Some(CommandBufferError::PageFault));
        let log = buffer_with(&lane.device, &[0; 16]).unwrap();
        let before = lane.device.work_counts().predicated_loop_runs_again;
        let bounded = looping(&lane.device, &state, &log, (0, 200));
        bounded.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(bounded.status(), CommandBufferStatus::Completed, "{name}");
        assert_eq!(bounded.error(), None, "{name}");
        assert_eq!(
            words(&log),
            [200, 0, 1],
            "{name}: the passes and the mark after"
        );
        let again = lane.device.work_counts().predicated_loop_runs_again - before;
        assert!(again > 0, "{name}: the loop of 200 ran its laying again");
    }
    layer_reported_nothing_at_all(&messages);
}

/// A command buffer committed as failed on another queue with Metal's code of a page fault (3)
/// stops a running loop without bound as a fault does (a victim), and with a hang's (2) as a hang
/// does (no error), nothing after the loop running; with a victim's code (1) the loop runs on.
#[test]
fn an_injected_fault_or_hang_stops_a_running_loop_as_the_error_it_stands_for() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let state = mark_state(&device);
        let inject = |code: u64| {
            let queue = device.new_command_queue().unwrap();
            let failed = queue.command_buffer().unwrap();
            failed.commit_failed(CommitFailure::new(code)).unwrap();
            failed.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(failed.error(), Some(CommandBufferError::Injected(code)));
        };
        let again = || device.work_counts().predicated_loop_runs_again;
        // A victim's code leaves the loop running; a page fault's then fails it as a victim.
        let log = buffer_with(&device, &[0; 16]).unwrap();
        let before = again();
        let looped = looping(&device, &state, &log, (1, 1));
        ran_again(&device, before, 2);
        inject(1);
        ran_again(&device, again(), 2);
        let running = !matches!(
            looped.status(),
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        );
        assert!(running, "{name}: after code 1: {:?}", looped.status());
        inject(3);
        looped.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(looped.status(), CommandBufferStatus::Error, "{name}");
        assert_eq!(
            looped.error(),
            Some(CommandBufferError::GpuRecoveryVictim),
            "{name}: after code 3"
        );
        assert_eq!(words(&log)[2], 0, "{name}: the mark after the loop ran");
        // A hang's code ends it with no error.
        let log = buffer_with(&device, &[0; 16]).unwrap();
        let before = again();
        let looped = looping(&device, &state, &log, (1, 1));
        ran_again(&device, before, 2);
        inject(2);
        looped.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(looped.status(), CommandBufferStatus::Completed, "{name}");
        assert_eq!(looped.error(), None, "{name}: after code 2");
        assert_eq!(words(&log)[2], 0, "{name}: the mark after the loop ran");
        println!("{name}: injected codes 3 and 2 stopped the loop; code 1 left it running");
    }
    layer_reported_nothing_at_all(&messages);
}
