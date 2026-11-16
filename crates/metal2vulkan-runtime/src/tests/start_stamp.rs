//! A command buffer's GPU start (`gpu_time.rs`) against the GPU work its submission carries.
//!
//! - **Its own long work.** A compute encoder of [`DISPATCHES`] dispatches over [`FLOATS`] floats,
//!   its pipeline made and its buffers' first work done by a command buffer before it, committed
//!   once that one completed: its GPU start lies in the first half of its life from its kernel end
//!   time to its GPU end, as the GPU starts the work once the submission reaches it, not at its
//!   end.
//! - **The batch's first work.** The next command buffer after a new `Private` buffer of
//!   [`FIRST`] bytes (its fill with zeros, `texture/first_move.rs`) and, on a device without
//!   unified memory, a `Managed` buffer of [`FIRST`] bytes whose whole range `did_modify_range`
//!   named (its copy in, `managed_copy.rs`), the command buffer one blit fill of a word of the
//!   `Managed` buffer (else of a buffer made before): its GPU span holds at least half the time
//!   that work adds to its life against the same fill without it, as the work runs in its own
//!   submission, before its commands. Where the device's fill queue took the work
//!   (`device/staging.rs`: it runs before the batch, outside it), the leg says so and checks
//!   nothing.
//!
//! Each leg prints its times (`START-STAMP`). Device tests: compiled only under the GPU safety
//! cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, CpuCacheMode,
    HazardTrackingMode, PipelineOptions, ResourceOptions, Size, StorageMode,
};

/// The floats each long dispatch scales.
const FLOATS: u64 = 1 << 24;

/// The long encoder's dispatches; a CPU device runs fewer.
const DISPATCHES: usize = match cfg!(m2v_cpu_icd) {
    true => 4,
    false => 64,
};

/// The bytes of each buffer of the first work.
const FIRST: u64 = match cfg!(m2v_cpu_icd) {
    true => 64 << 20,
    false => 256 << 20,
};

/// The kernel, and the fixture it is in.
const FIXTURE: &str = "examples/runtime_examples.ll";
const KERNEL: &str = "scale_and_offset";

/// A completed command buffer's kernel end time and GPU times, in seconds of the host's clock.
#[derive(Clone, Copy, Debug)]
struct Times {
    kernel_end: f64,
    start: f64,
    end: f64,
}

impl Times {
    /// From its kernel end time to its GPU end.
    fn life(self) -> f64 {
        self.end - self.kernel_end
    }

    /// From its GPU start to its GPU end.
    fn span(self) -> f64 {
        self.end - self.start
    }

    /// From its kernel end time to its GPU start.
    fn lead(self) -> f64 {
        self.start - self.kernel_end
    }

    fn line(self) -> String {
        format!(
            "life {:.3} ms, kernel end to GPU start {:.3} ms, GPU span {:.3} ms",
            self.life() * 1e3,
            self.lead() * 1e3,
            self.span() * 1e3
        )
    }
}

/// Commit `command_buffer`, hold that it completed, and read its times.
fn run(command_buffer: &CommandBuffer) -> Times {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Times {
        kernel_end: command_buffer.kernel_end_time(),
        start: command_buffer.gpu_start_time().unwrap(),
        end: command_buffer.gpu_end_time().unwrap(),
    }
}

/// A command buffer of `queue`'s of one compute encoder that scales [`FLOATS`] floats of `input`
/// into `output` with `state`, `times` over.
fn scale(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (input, output): (&Buffer, &Buffer),
    times: usize,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    for _ in 0..times {
        (encoder.dispatch_threads(Size::new(FLOATS, 1, 1), Size::new(256, 1, 1))).unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer
}

/// A command buffer of `queue`'s of one blit fill of `target`'s first word.
fn fill(queue: &CommandQueue, target: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..4, 7).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

#[test]
fn a_command_buffer_s_gpu_start_comes_before_its_own_long_work() {
    let scratch = Scratch::new("start-stamp-long");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let function = library(&device, FIXTURE)
        .new_function_with_name(KERNEL)
        .unwrap();
    let state = (device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
    .unwrap();
    let bytes = FLOATS * 4;
    let input = device
        .new_buffer(bytes, ResourceOptions::private())
        .unwrap();
    let output = device
        .new_buffer(bytes, ResourceOptions::private())
        .unwrap();
    // Its pipeline made, its buffers' first work done.
    run(&scale(&queue, &state, (&input, &output), 1));
    let long = run(&scale(&queue, &state, (&input, &output), DISPATCHES));
    eprintln!(
        "START-STAMP long {}: {DISPATCHES} dispatches: {}",
        device.name(),
        long.line()
    );
    assert!(
        long.lead() <= long.life() / 2.0,
        "the GPU start lies in the second half of the command buffer's life: {}",
        long.line()
    );
}

#[test]
fn a_command_buffer_s_gpu_span_holds_its_batch_s_first_work() {
    let scratch = Scratch::new("start-stamp-first");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let two_copies = !device.properties().has_unified_memory;
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    // Its first work done, then the fill alone.
    run(&fill(&queue, &small));
    let alone = run(&fill(&queue, &small));
    let before = device.work_counts();
    let fresh = device
        .new_buffer(FIRST, ResourceOptions::private())
        .unwrap();
    let managed = two_copies.then(|| {
        let options = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let buffer = device.new_buffer(FIRST, options).unwrap();
        let contents = buffer.contents().expect("a Managed buffer's host copy");
        // SAFETY: the buffer's host copy of `FIRST` bytes, which no command buffer uses yet.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), 0x5a, FIRST as usize) };
        buffer.did_modify_range(0..FIRST).unwrap();
        buffer
    });
    let with = run(&fill(&queue, managed.as_ref().unwrap_or(&small)));
    let after = device.work_counts();
    let copied_in = after.managed_bytes_copied_in - before.managed_bytes_copied_in;
    let zero_fills = after.new_buffer_zero_fills - before.new_buffer_zero_fills;
    let on_fill_queue = after.fill_queue_submissions - before.fill_queue_submissions;
    let added = with.life() - alone.life();
    eprintln!(
        "START-STAMP first work {}: {zero_fills} zero fills, {copied_in} bytes copied in, \
         {on_fill_queue} fill-queue submissions; alone: {}; with it: {}; it adds {:.3} ms",
        device.name(),
        alone.line(),
        with.line(),
        added * 1e3
    );
    drop(fresh);
    if on_fill_queue > 0 {
        eprintln!("START-STAMP first work: the fill queue took it, before the batch: no check");
        return;
    }
    assert!(zero_fills > 0, "the new buffer took no fill with zeros");
    if two_copies {
        assert!(
            copied_in >= FIRST,
            "{copied_in} bytes copied in, of {FIRST}"
        );
    }
    assert!(
        with.span() >= added / 2.0,
        "the GPU span leaves out the batch's first work: {:.3} ms added, span {:.3} ms",
        added * 1e3,
        with.span() * 1e3
    );
}
