//! `Managed` no-copy buffers over a guest RAM (`host_ram.rs`) on a device without unified memory
//! (`managed_copy.rs`): two copies, the caller's pages and the GPU's own, the buffer's pages
//! across the end of a chunk whose import, and the next chunk's, live.
//!
//! - **Copies in from the chunks.** The buffer's host copy is its pages alone
//!   (`WorkCounts::managed_pages_alone_over_imports`): an import of their own would map pages the
//!   chunks' imports hold, so nothing is imported and the gate is not asked. The chunks' imports
//!   hold them all (`managed_pages_in_ram_chunks`). Its copy in as it is made is the GPU's, from
//!   those imports, the CPU staging nothing (`managed_bytes_copied_in_from_ram`,
//!   `managed_bytes_staged_in`), and so is the copy in of a range `did_modify_range` names across
//!   the chunks' end (under a test's bound of 0 for ranges written inline); under the default
//!   bound that range, of four pages, is written inline instead (`managed_ranges_inlined`), and
//!   under `ManagedCopyInPolicy::AtCommit` the CPU stages it at the commit. The GPU reads the
//!   CPU's bytes under each.
//! - **The bench across a chunk's end** (ignored, `MANAGED-RAM-COPY-IN`; run in release): what a
//!   `did_modify_range` of 256 KiB, 4 MiB and 32 MiB across the end of the RAM's first chunk,
//!   written by the CPU before each commit of a command buffer that reads the range's first and
//!   last bytes, costs the commit call and the wall time to the completion, on the default device
//!   and on one of `DevicePolicy::exact_answers()`: the medians, and the bytes copied in.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory, or has unified memory (where
//! a `Managed` buffer is one copy), says so and passes.

use super::host_ram_device::{
    bytes, fill_pattern, gate, gpu_bytes, held_device, itself, new_ram, resolver, settles, MIB,
    WAIT,
};
use crate::{
    Buffer, CpuCacheMode, Device, HazardTrackingMode, HostBacking, HostMemory, ImportRequest,
    ResourceOptions, StorageMode,
};
use metal2vulkan::device::{DevicePolicy, ManagedCopyInPolicy};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The end of the RAM's first chunk, from its first byte (`host_ram.rs`'s `CHUNK_BYTES`).
const CHUNK_END: u64 = 64 * MIB;

/// The bench's buffer: 32 MiB, its middle the first chunk's end.
const BENCH_BYTES: u64 = 32 * MIB;

/// The ranges the bench names, each its middle the first chunk's end.
const BENCH_RANGES: [u64; 3] = [256 << 10, 4 * MIB, 32 * MIB];

/// The commits of each of the bench's ranges.
const BENCH_COMMITS: u32 = 20;

/// The options of a `Managed` buffer.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A device of `policy` whose RAM's chunks are each imported by the first object over it
/// (`host_ram_device.rs`'s `held_device`), on which a `Managed` no-copy buffer is two copies;
/// `None` where it imports no host memory or has unified memory.
fn two_copy_device(policy: DevicePolicy) -> Option<Device> {
    let device = held_device(policy)?;
    if device.properties().has_unified_memory {
        eprintln!(
            "HOST-RAM-MANAGED {}: unified memory; a Managed buffer is one copy",
            device.name()
        );
        return None;
    }
    Some(device)
}

/// The memory at host address `at` of the test's RAM, its pages their own run of it, under a gate
/// that admits every import.
fn memory_at(at: u64) -> HostMemory {
    memory_gated(at, &Arc::default())
}

/// [`memory_at`], the gate's requests into `gates`.
fn memory_gated(at: u64, gates: &Arc<Mutex<Vec<ImportRequest>>>) -> HostMemory {
    let runs: Arc<Mutex<_>> = Arc::default();
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(resolver(&runs, itself))
        .with_import_gate(gate(gates, true))
}

/// `Shared` buffers of four pages, one in each of the RAM's two whole chunks from `start`, far
/// from the pages between them, which import the chunks.
fn import_both_chunks(device: &Device, start: u64) -> (Buffer, Buffer) {
    let page = device.shared().host_import.as_ref().unwrap().page;
    let in_chunk = |at: u64| {
        // SAFETY: as in `memory_at`.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory_at(at),
                4 * page,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    };
    (in_chunk(start + 16 * page), in_chunk(start + 96 * MIB))
}

/// Copies in from the chunks (the module docs), under `copy_in`; the named range written inline
/// where `inline` (`managed::INLINE_BYTES`), else under a bound of 0, copied from the chunks.
fn copies_in_across_a_chunk_end(copy_in: ManagedCopyInPolicy, inline: bool) {
    const LENGTH: u64 = 4 * MIB;
    let mut policy = DevicePolicy::default();
    policy.managed_copy_in = copy_in;
    let Some(device) = two_copy_device(policy) else {
        return;
    };
    if !inline {
        (device.shared().managed_inline_bound).store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = import_both_chunks(&device, start);
    let at = start + CHUNK_END - LENGTH / 2;
    fill_pattern(at..at + LENGTH);
    let gates = Arc::default();
    let before = device.work_counts();
    // SAFETY: as in `memory_at`.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(memory_gated(at, &gates), LENGTH, managed(), None)
    }
    .unwrap();
    assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(
        buffer.contents().map(|pointer| pointer.as_ptr() as u64),
        Some(at)
    );
    let made = device.work_counts();
    assert_eq!(
        (
            made.managed_pages_alone_over_imports - before.managed_pages_alone_over_imports,
            made.managed_pages_in_ram_chunks - before.managed_pages_in_ram_chunks,
            made.host_pointer_imports_made - before.host_pointer_imports_made,
            made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        ),
        (1, 1, 0, 0),
        "its pages alone, which the chunks' imports hold; nothing imported"
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    assert_eq!(
        gpu_bytes(&device, &buffer, LENGTH),
        bytes(at..at + LENGTH),
        "the bytes copied in as the buffer was made"
    );
    let first = device.work_counts();
    assert_eq!(
        (
            first.managed_bytes_copied_in_from_ram - before.managed_bytes_copied_in_from_ram,
            first.managed_bytes_staged_in - before.managed_bytes_staged_in,
        ),
        (LENGTH, 0),
        "copied in from the chunks' imports as it was made, nothing staged"
    );
    // The CPU's write across the chunks' end, named.
    let named = LENGTH / 2 - 2 * page..LENGTH / 2 + 2 * page;
    let length = named.end - named.start;
    // SAFETY: the RAM's bytes, which no command buffer reads now.
    unsafe { std::ptr::write_bytes((at + named.start) as *mut u8, 0xa5, length as usize) };
    buffer.did_modify_range(named).unwrap();
    assert_eq!(
        gpu_bytes(&device, &buffer, LENGTH),
        bytes(at..at + LENGTH),
        "the CPU's write across the chunks' end"
    );
    let after = device.work_counts();
    let inlined = inline
        && copy_in == ManagedCopyInPolicy::AtSubmission
        && length <= crate::managed::INLINE_BYTES;
    let expected = match (copy_in == ManagedCopyInPolicy::AtCommit, inlined) {
        (true, _) => (0, length, 0),
        (false, true) => (0, 0, 1),
        (false, false) => (length, 0, 0),
    };
    assert_eq!(
        (
            after.managed_bytes_copied_in_from_ram - first.managed_bytes_copied_in_from_ram,
            after.managed_bytes_staged_in - first.managed_bytes_staged_in,
            after.managed_ranges_inlined - first.managed_ranges_inlined,
        ),
        expected,
        "the named range copied in from the chunks' imports, written inline, or staged at the \
         commit ({copy_in:?}, inline {inline})"
    );
    drop((buffer, chunks, used, ram, device));
    assert!(settles(&dropped), "the RAM's keep-alive did not go");
}

#[test]
fn a_managed_buffer_across_a_chunk_end_copies_in_from_the_chunks_imports() {
    copies_in_across_a_chunk_end(ManagedCopyInPolicy::AtSubmission, false);
}

#[test]
fn a_managed_buffer_across_a_chunk_end_writes_a_small_named_range_inline() {
    copies_in_across_a_chunk_end(ManagedCopyInPolicy::AtSubmission, true);
}

#[test]
fn a_managed_buffer_across_a_chunk_end_is_staged_at_the_commit_under_at_commit() {
    copies_in_across_a_chunk_end(ManagedCopyInPolicy::AtCommit, true);
}

/// The median of `values`, in milliseconds.
fn median_ms(mut values: Vec<Duration>) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e3
}

/// The bench across a chunk's end (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn managed_ram_copy_in_bench() {
    for exact in [false, true] {
        let policy = match exact {
            true => DevicePolicy::default().exact_answers(),
            false => DevicePolicy::default(),
        };
        let Some(device) = two_copy_device(policy) else {
            return;
        };
        let named = match exact {
            true => "exact answers",
            false => "default",
        };
        let (ram, start, dropped) = new_ram();
        let used = device.use_host_ram(&ram).unwrap();
        let chunks = import_both_chunks(&device, start);
        let at = start + CHUNK_END - BENCH_BYTES / 2;
        // SAFETY: as in `memory_at`.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(memory_at(at), BENCH_BYTES, managed(), None)
        }
        .unwrap();
        assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
        let queue = device.new_command_queue().unwrap();
        let out = device.new_buffer(8192, ResourceOptions::shared()).unwrap();
        for range in BENCH_RANGES {
            let named_range = BENCH_BYTES / 2 - range / 2..BENCH_BYTES / 2 + range / 2;
            let before = device.work_counts();
            let (mut commits, mut whole) = (Vec::new(), Vec::new());
            for commit in 0..BENCH_COMMITS {
                let byte = commit as u8 + 1;
                // SAFETY: the RAM's bytes, which no command buffer reads now.
                unsafe {
                    std::ptr::write_bytes((at + named_range.start) as *mut u8, byte, range as usize)
                };
                let began = Instant::now();
                buffer.did_modify_range(named_range.clone()).unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(&buffer, named_range.start, &out, 0, 4096)
                    .unwrap();
                blit.copy_from_buffer_to_buffer(&buffer, named_range.end - 4096, &out, 4096, 4096)
                    .unwrap();
                blit.end_encoding().unwrap();
                let committing = Instant::now();
                command_buffer.commit().unwrap();
                commits.push(committing.elapsed());
                assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
                whole.push(began.elapsed());
                assert_eq!(command_buffer.error(), None);
                let contents = out.contents().unwrap();
                // SAFETY: the out buffer's bytes, which no command writes now.
                let seen = unsafe { (*contents.as_ptr(), *contents.as_ptr().add(4096)) };
                assert_eq!(seen, (byte, byte), "{range}: commit {commit}");
            }
            let after = device.work_counts();
            let (commit, wall) = (median_ms(commits.clone()), median_ms(whole));
            let most = commits.iter().max().unwrap().as_secs_f64() * 1e3;
            eprintln!(
                "MANAGED-RAM-COPY-IN {} {named}, {} KiB a commit across a chunk's end: commit \
                 {commit:.3} ms (most {most:.3}), to completion {wall:.3} ms, {:.2} GB/s; copied \
                 in {} bytes",
                device.name(),
                range >> 10,
                range as f64 / (wall * 1e6),
                after.managed_bytes_copied_in - before.managed_bytes_copied_in,
            );
        }
        drop((buffer, out, queue, chunks, used, ram, device));
        assert!(settles(&dropped), "the RAM's keep-alive did not go");
    }
}
