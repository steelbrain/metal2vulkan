//! The fill queue (`device/staging.rs`): on a device with a family of transfers alone that has a
//! second queue (NVIDIA's copy engine), a batch's new buffers' fills go there, in a submission of
//! their own that the batch waits for, after the batches before it on its `VkQueue`.
//!
//! - **Where.** The fill queue is of a family whose flags hold transfers and no graphics or
//!   compute work, and each command buffer below that took fills made one submission there
//!   (`WorkCounts::fill_queue_submissions`); a device without one (lavapipe, RADV) made none.
//! - **Bytes.** A `Managed` buffer's copy in after `did_modify_range`, a new `Private` buffer's
//!   zeros and a `Private` buffer made with bytes read back exactly, through a blit into a
//!   `Shared` buffer.
//! - **Order.** A command buffer whose blit reads the `Managed` buffer behind a long run of
//!   kernels, submitted and still in flight as the CPU writes the buffer again and a second
//!   command buffer of the same queue takes the copy in: the first reads the first bytes, the second the new ones.
//!   Where the device has unified memory (lavapipe), whose `Managed` buffer is one copy, the first
//!   completes before the CPU writes again.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, CpuCacheMode,
    HazardTrackingMode, PipelineOptions, ResourceOptions, Size, StorageMode,
};
use ash::vk;
use std::time::Duration;

/// The `Managed` buffer's bytes: 4 MiB and a word, so the copy in is no whole number of pages.
const BYTES: u64 = (4 << 20) + 4;

/// The kernels that keep the first command buffer in flight: dispatches over a `Private` buffer
/// of [`LOAD_BYTES`]; a CPU device runs a few over a small one.
const LOAD_DISPATCHES: usize = match cfg!(m2v_cpu_icd) {
    true => 2,
    false => 96,
};
const LOAD_BYTES: u64 = match cfg!(m2v_cpu_icd) {
    true => 1 << 20,
    false => 64 << 20,
};

/// The bytes word `index` of a pattern `seed` holds.
fn pattern(seed: u32, bytes: u64) -> Vec<u8> {
    (0..bytes / 4)
        .flat_map(|index| {
            (index as u32)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(seed)
                .to_le_bytes()
        })
        .collect()
}

/// The bytes of `buffer`, a `Shared` one a completed command buffer wrote.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: `length` bytes of a `Shared` buffer no command buffer in flight writes.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A blit of `source`, whole, into `target` at its start.
fn copy(command_buffer: &CommandBuffer, source: &Buffer, target: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_buffer(source, 0, target, 0, source.length())).unwrap();
    blit.end_encoding().unwrap();
}

/// [`LOAD_DISPATCHES`] dispatches of `scale_and_offset` over `load`, into `scratch`.
fn load(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    load: &Buffer,
    scratch: &Buffer,
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(load), 0, 0).unwrap();
    encoder.set_buffer(Some(scratch), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    for _ in 0..LOAD_DISPATCHES {
        (encoder.dispatch_threads(Size::new(LOAD_BYTES / 4, 1, 1), Size::new(256, 1, 1))).unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// The module docs: where the fills run, their bytes, and their order behind a command buffer in
/// flight on the same queue.
#[test]
fn a_batch_s_fills_run_on_the_fill_queue_in_order_with_exact_bytes() {
    let scratch = Scratch::new("fill-queue");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let place = device.fill_queue_family();
    if let Some((family, flags)) = place {
        let work = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        assert!(
            flags.contains(vk::QueueFlags::TRANSFER) && !flags.intersects(work),
            "{name}: the fill queue's family {family} has flags {:#x}",
            flags.as_raw()
        );
    }
    let state = {
        let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
            .new_function_with_name("scale_and_offset")
            .unwrap();
        (device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
            .unwrap()
    };
    let queue = device.new_command_queue().unwrap();
    let shared = |bytes| device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let (load_buffer, load_scratch) = (
        device
            .new_buffer(LOAD_BYTES, ResourceOptions::private())
            .unwrap(),
        device
            .new_buffer(LOAD_BYTES, ResourceOptions::private())
            .unwrap(),
    );
    let managed = device
        .new_buffer(
            BYTES,
            ResourceOptions::new(
                StorageMode::Managed,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            ),
        )
        .unwrap();
    let write = |bytes: &[u8]| {
        let contents = managed.contents().unwrap();
        // SAFETY: the buffer's host copy, `BYTES` long; no command buffer reads it (the GPU
        // reads the device copy, which `did_modify_range` copies into).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        managed.did_modify_range(0..BYTES).unwrap();
    };
    let (first, second) = (pattern(0x1234_5678, BYTES), pattern(0x0bad_cafe, BYTES));
    // The device's own checks' submissions are done, so the two command buffers below are what
    // takes the fills.
    assert!(
        device.settle_in_place_checks(Duration::from_secs(120)),
        "{name}"
    );
    assert!(
        device.settle_queue_switch_check(Duration::from_secs(120)),
        "{name}"
    );
    let before = device.work_counts();
    write(&first);
    let first_read = shared(BYTES);
    let behind = queue.command_buffer().unwrap();
    load(&behind, &state, &load_buffer, &load_scratch);
    copy(&behind, &managed, &first_read);
    behind.commit().unwrap();
    // Once it was submitted, so its submission took the first copy in, the CPU writes the buffer
    // again while it may still run, and a second command buffer of the queue takes that copy in,
    // with a new buffer's zeros and another's bytes. Where the device has unified memory the
    // buffer is one copy, which the CPU's write would race, as on Metal: there the first command
    // buffer completes first.
    behind.wait_until_scheduled().unwrap();
    if device.properties().has_unified_memory {
        behind.wait_until_completed_for(WAIT).unwrap();
    }
    write(&second);
    let staged = pattern(0x5eed_0001, 1 << 20);
    let zeroed = device
        .new_buffer(1 << 20, ResourceOptions::private())
        .unwrap();
    let with_bytes = (device.new_buffer_with_bytes(&staged, ResourceOptions::private())).unwrap();
    let (second_read, zeros_read, bytes_read) = (shared(BYTES), shared(1 << 20), shared(1 << 20));
    let after = queue.command_buffer().unwrap();
    copy(&after, &managed, &second_read);
    copy(&after, &zeroed, &zeros_read);
    copy(&after, &with_bytes, &bytes_read);
    after.commit().unwrap();
    let still_running = behind.status() != CommandBufferStatus::Completed;
    for command_buffer in [&behind, &after] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{name}"
        );
    }
    let counts = device.work_counts();
    let submissions = counts.fill_queue_submissions - before.fill_queue_submissions;
    let shown = (place
        .map(|(family, flags)| format!("family {family}, flags {:#x}", flags.as_raw())))
    .unwrap_or_else(|| "none".to_owned());
    eprintln!(
        "FILL-QUEUE {name}: fill queue {shown}, submissions {submissions}, first command buffer \
         still running as the second committed: {still_running}"
    );
    assert!(
        read(&first_read) == first,
        "{name}: the first command buffer read other bytes"
    );
    assert!(
        read(&second_read) == second,
        "{name}: the second command buffer read other bytes"
    );
    assert!(
        read(&zeros_read).iter().all(|&byte| byte == 0),
        "{name}: a new buffer not zero"
    );
    assert!(
        read(&bytes_read) == staged,
        "{name}: a buffer made with bytes read other bytes"
    );
    // One submission a command buffer that took fills; none without a fill queue.
    let expected = match place {
        Some(_) => 2,
        None => 0,
    };
    assert_eq!(submissions, expected, "{name}");
}
