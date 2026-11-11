//! The cost of a kernel's pass over a copied host buffer (`copied.rs`), one command buffer a
//! pass, as an OpenCL enqueue loop runs it: `scale_and_offset` of the owned fixture
//! `examples/runtime_examples.ll` reads each float of the buffer and writes it back changed
//! (`output[i] = input[i] * scale + i`, the buffer bound as both), so every pass's completion
//! merges back most of the buffer's bytes. Each pass is timed apart: its commit (the copy in of
//! the CPU's writes, none here), its wait (the GPU's work across the bus, the completion's merge
//! back, and the latency of both), and the GPU's own time (`gpu_start_time` to `gpu_end_time`).
//! The same pass over a buffer of the device's own `Shared` memory (host memory the GPU reaches
//! across the bus, nothing copied) and of its `Private` memory are the controls. Printed as the
//! median of the passes after the first two (`COPIED-PASS`), with the bytes read and written a
//! second, counted twice (read and written) over the whole pass and over the GPU's time alone.
//! Ignored: run it in release, under the GPU safety cfg.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use crate::{Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, HostMemory};
use crate::{ImportGate, PipelineOptions, ResourceOptions, Size};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The passes of each buffer, the first two of which are not counted.
const PASSES: usize = 12;

/// A device of the cfg's kind; `None` where it imports no host memory.
fn device() -> Option<Device> {
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
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("COPIED-PASS: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// `PASSES` passes of the kernel over `buffer`'s `length` bytes, printed under `label`.
fn passes(device: &Device, state: &crate::ComputePipelineState, buffer: &Buffer, label: &str) {
    let length = buffer.length();
    let floats = length / 4;
    let queue = device.new_command_queue().unwrap();
    let (mut commits, mut waits, mut gpus, mut walls) = (vec![], vec![], vec![], vec![]);
    let before = device.work_counts();
    for pass in 0..PASSES {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_buffer(Some(buffer), 0, 0).unwrap();
        encoder.set_buffer(Some(buffer), 0, 1).unwrap();
        encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
        let threads = Size::new(floats, 1, 1);
        encoder
            .dispatch_threads(threads, Size::new(256, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        let started = Instant::now();
        command_buffer.commit().unwrap();
        let committed = Instant::now();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let ended = Instant::now();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        if pass < 2 {
            continue;
        }
        let gpu = (command_buffer.gpu_end_time().unwrap_or(0.0))
            - command_buffer.gpu_start_time().unwrap_or(0.0);
        commits.push((committed - started).as_secs_f64() * 1e3);
        waits.push((ended - committed).as_secs_f64() * 1e3);
        gpus.push(gpu * 1e3);
        walls.push((ended - started).as_secs_f64() * 1e3);
    }
    let after = device.work_counts();
    let (wall, gpu) = (median(walls), median(gpus));
    let moved = 2.0 * length as f64 / 1e9;
    eprintln!(
        "COPIED-PASS {label}, {} MiB: a pass {wall:.2} ms ({:.2} GB/s), its commit {:.2} ms, its \
         wait {:.2} ms, the GPU's time {gpu:.2} ms ({:.2} GB/s); a pass copied in {} KiB and \
         merged back {} KiB",
        length >> 20,
        moved / (wall / 1e3),
        median(commits),
        median(waits),
        moved / (gpu / 1e3),
        (after.copied_copy_in_bytes - before.copied_copy_in_bytes) / PASSES as u64 >> 10,
        (after.copied_merge_back_bytes - before.copied_merge_back_bytes) / PASSES as u64 >> 10,
    );
}

/// The kernel's passes over a copied buffer of 64 and of 256 MiB (the gate refuses its import),
/// and over the device's own `Shared` and `Private` buffers of the same lengths.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_pass_bench() {
    let Some(device) = device() else {
        return;
    };
    let named = device.name().to_string();
    let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    for length in [64usize << 20, 256 << 20] {
        let layout = Layout::from_size_align(length, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        let refuse: ImportGate = Arc::new(|_| false);
        // SAFETY: the memory outlives the buffer (freed below, after it), and only the buffer's
        // commands use it.
        let memory = unsafe { HostMemory::new(pointer) }.with_import_gate(refuse);
        // SAFETY: as above.
        let copied = unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        assert!(copied.copied(), "the buffer is a copy");
        passes(
            &device,
            &state,
            &copied,
            &format!("{named}, a copied buffer"),
        );
        drop(copied);
        // SAFETY: allocated above with this layout; nothing uses it now.
        unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
        for (options, kind) in [
            (ResourceOptions::shared(), "Shared"),
            (ResourceOptions::private(), "Private"),
        ] {
            let own = device.new_buffer(length as u64, options).unwrap();
            passes(
                &device,
                &state,
                &own,
                &format!("{named}, its own {kind} buffer"),
            );
        }
    }
}
