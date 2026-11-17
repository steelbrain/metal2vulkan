//! A no-copy buffer over pages a live import of another no-copy buffer holds some of
//! (`host_aliases.rs`, "One page, one import"), its memory with a source of dma-bufs
//! (`HostMemory::with_dma_buf`) over a `memfd` mapped shared, as QEMU's guest RAM is: where the
//! live import is of dma-bufs, on a device whose barriers order blits through two imports of the
//! same pages (no `RuntimeDriverDefect::AliasedImportBlitsUnordered`), the buffer imports its own
//! dma-bufs beside it (`WorkCounts::object_churn.dma_buf_imports_beside_live`), never a copy
//! (`copied_over_live_imports`); where the live import is of the pages' pointer, it is a copy of
//! them. A buffer rebound onto pages two live imports hold some of takes the same rule. A buffer
//! whose pages a live import of dma-bufs dedicated to nothing holds all of (a no-copy buffer's own
//! import of one chunk) is bound inside it, as inside an import of the pointer, with a source of
//! dma-bufs or with none (`Device::bind_in_import`,
//! `WorkCounts::object_churn.buffers_bound_in_import`), never a copy or a second import; where
//! the driver took that import only dedicated to its buffer, it is a copy, or beside it where it
//! has a source (`WorkCounts::dma_buf_buffer_imports_dedicated`). Either
//! way the GPU reads the CPU's bytes through it, CPU writes made after a command is encoded and
//! before its commit reach that command, and the GPU's writes through it reach the pages and the
//! other buffer's next read. Where both lie in imports, a write through one and a read through
//! the other in one command buffer are ordered by the bytes they share, as the runtime orders
//! any two buffers over the same host bytes; a copy agrees with the pages at its commit and its
//! completion, and Metal orders no two no-copy buffers over one allocation.
//!
//! Each runs on the device the cfg selects as made (RADV refuses a `memfd`'s pointer and imports
//! its pages as dma-bufs; lavapipe, NVIDIA and ANV import the pointer), and made without
//! `VK_EXT_external_memory_host`, where the device lists the dma-buf route and imports dma-bufs
//! alone.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, each says so and passes without running.

use super::host_residency::{counting, host_page, read, udmabuf_openable, Ram, Scratch};
use super::no_pointer_import::lists_dma_buf_route;
use crate::{
    BlitCommandEncoder, Buffer, CommandBuffer, CommandQueue, Device, HostBacking, HostMemory,
    ResourceOptions, WorkCounts,
};
use metal2vulkan::error::RuntimeDriverDefect;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "DMA-BUF-BESIDE";

/// How long a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

#[test]
fn a_buffer_over_part_of_a_live_import_lies_beside_its_dma_bufs_or_in_a_copy() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    over_part("as made", &super::host_residency::device(&scratch.0));
}

#[test]
fn without_pointer_imports_a_buffer_over_part_of_a_live_import_lies_beside_its_dma_bufs() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = super::no_pointer_import::device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        eprintln!("{TEST}: on {}: no dma-buf route listed", device.name());
        return;
    }
    over_part("without pointer imports", &device);
}

/// The counts that tell where a no-copy buffer's pages went: copies over live imports, imports of
/// dma-bufs beside live ones, imports where a live one held some of the pages all the same,
/// imports by pointer, and buffers bound inside a live import.
fn placed(counts: &WorkCounts) -> [u64; 5] {
    [
        counts.copied_over_live_imports,
        counts.object_churn.dma_buf_imports_beside_live,
        counts.object_churn.imports_inside_live,
        counts.host_pointer_imports_made,
        counts.object_churn.buffers_bound_in_import,
    ]
}

/// `after` less `before`.
fn since(before: [u64; 5], after: [u64; 5]) -> [u64; 5] {
    std::array::from_fn(|at| after[at] - before[at])
}

/// Commit `command_buffer` and wait for it.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A `Shared` buffer of the runtime's own of `length` bytes.
fn scratch_buffer(device: &Device, length: usize) -> Buffer {
    device
        .new_buffer(length as u64, ResourceOptions::shared())
        .unwrap()
}

/// `buffer`'s bytes.
fn bytes_of(buffer: &Buffer, length: usize) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: the buffer's own bytes, which no command writes any more.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length).to_vec() }
}

/// One command buffer of one blit encoder whose commands `encode` records, then what `before`
/// does once they are encoded and before the commit, then the commit and the wait.
fn blits(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut BlitCommandEncoder),
    before_commit: impl FnOnce(),
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    before_commit();
    run(&command_buffer);
}

/// The test on `device`: a first buffer over pages 0 and 1, a second over pages 1 and 2 from 256
/// bytes into page 1, a third made over pages 4 and 5 and rebound onto pages 1 to 3 from 512 bytes
/// into page 1.
fn over_part(how: &str, device: &Device) {
    let page = host_page();
    let ram = Ram::new(6 * page, 0, 6 * page);
    let bytes = counting(6 * page, 0);
    ram.write_file(0, &bytes);
    let calls = Arc::new(AtomicUsize::new(0));
    let memory = |offset: usize| {
        // SAFETY: the view outlives every buffer over it (each dropped first), and the source
        // makes dma-bufs of its pages.
        unsafe { HostMemory::with_dma_buf(ram.pointer(offset), ram.source(&calls)) }
    };
    let made = |offset: usize, length: usize| {
        // SAFETY: as `memory`'s.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory(offset),
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    };
    let first = made(0, 2 * page);
    let first_backing = first.host_backing();
    let unordered = (device.properties().driver_defects)
        .contains(&RuntimeDriverDefect::AliasedImportBlitsUnordered);
    let beside = first_backing == Some(HostBacking::ImportedAsDmaBuf) && !unordered;
    let before = placed(&device.work_counts());
    let (offset, length) = (page + 256, 2 * page - 256);
    let second = made(offset, length);
    let made_second = since(before, placed(&device.work_counts()));
    println!(
        "{TEST}: on {} ({how}): the first buffer {first_backing:?}, blits through two imports of \
         the same pages unordered: {unordered}; the second {:?}, [copied over live, dma-bufs \
         beside live, imported inside live, pointer imports, bound inside] {made_second:?}",
        device.name(),
        second.host_backing(),
    );
    match beside {
        true => {
            assert_eq!(
                second.host_backing(),
                Some(HostBacking::ImportedAsDmaBuf),
                "{TEST} ({how})"
            );
            assert_eq!(made_second, [0, 1, 0, 0, 0], "{TEST} ({how})");
        }
        false => {
            assert_eq!(
                second.host_backing(),
                Some(HostBacking::Copied),
                "{TEST} ({how})"
            );
            assert_eq!(made_second, [1, 0, 0, 0, 0], "{TEST} ({how})");
        }
    }
    let queue = device.new_command_queue().unwrap();
    // The GPU reads the CPU's bytes through the second.
    assert!(
        read(&queue, &second, length) == bytes[offset..offset + length],
        "{TEST} ({how}): the GPU's read of the CPU's bytes through the second"
    );
    // CPU writes made after a copy is encoded and before its commit reach it, through each.
    let out = (
        scratch_buffer(device, length),
        scratch_buffer(device, 2 * page),
    );
    let later = counting(6 * page, 7);
    blits(
        &queue,
        |blit| {
            blit.copy_from_buffer_to_buffer(&second, 0, &out.0, 0, length as u64)
                .unwrap();
            blit.copy_from_buffer_to_buffer(&first, 0, &out.1, 0, 2 * page as u64)
                .unwrap();
        },
        || ram.write_file(0, &later[..3 * page]),
    );
    assert!(
        bytes_of(&out.0, length) == later[offset..offset + length],
        "{TEST} ({how}): the CPU's write before the commit, through the second"
    );
    assert!(
        bytes_of(&out.1, 2 * page) == later[..2 * page],
        "{TEST} ({how}): the CPU's write before the commit, through the first"
    );
    // The GPU's fill through the second reaches the pages, and the first's next read.
    let shared = page - 256;
    blits(
        &queue,
        |blit| blit.fill_buffer(&second, 0..length as u64, 0xc3).unwrap(),
        || {},
    );
    assert!(
        ram.read_file(offset, length).iter().all(|&b| b == 0xc3),
        "{TEST} ({how}): the GPU's fill through the second in the file's pages"
    );
    let through_first = read(&queue, &first, 2 * page);
    assert!(
        through_first[..offset] == later[..offset]
            && through_first[offset..].iter().all(|&b| b == 0xc3),
        "{TEST} ({how}): the second's fill read through the first"
    );
    // Both in imports: a fill through one and a copy through the other of the bytes they share,
    // in one encoder, either way round.
    if beside {
        let out = scratch_buffer(device, shared);
        blits(
            &queue,
            |blit| {
                blit.fill_buffer(&first, offset as u64..2 * page as u64, 0xa1)
                    .unwrap();
                blit.copy_from_buffer_to_buffer(&second, 0, &out, 0, shared as u64)
                    .unwrap();
            },
            || {},
        );
        assert!(
            bytes_of(&out, shared).iter().all(|&b| b == 0xa1),
            "{TEST} ({how}): the first's fill read through the second in one encoder"
        );
        blits(
            &queue,
            |blit| {
                blit.fill_buffer(&second, 0..shared as u64, 0xb2).unwrap();
                blit.copy_from_buffer_to_buffer(&first, offset as u64, &out, 0, shared as u64)
                    .unwrap();
            },
            || {},
        );
        assert!(
            bytes_of(&out, shared).iter().all(|&b| b == 0xb2),
            "{TEST} ({how}): the second's fill read through the first in one encoder"
        );
    }
    // A buffer rebound onto pages 1 to 3, which both live imports hold some of.
    let third = made(4 * page, 2 * page);
    let before = placed(&device.work_counts());
    let rebound_at = page + 512;
    ram.write_file(0, &bytes);
    // SAFETY: as `memory`'s; no command that uses the buffer is encoded or pending.
    unsafe { third.rebind_bytes_no_copy(memory(rebound_at), None, None) }.unwrap();
    let rebound = since(before, placed(&device.work_counts()));
    println!(
        "{TEST}: on {} ({how}): the rebound buffer {:?}, {rebound:?}",
        device.name(),
        third.host_backing(),
    );
    // Beside dma-bufs where both live imports are (the second's own is where the first's is),
    // else a copy beside the first's import.
    let expected = match beside {
        true => (HostBacking::ImportedAsDmaBuf, [0, 1, 0, 0, 0]),
        false => (HostBacking::Copied, [1, 0, 0, 0, 0]),
    };
    assert_eq!(third.host_backing(), Some(expected.0), "{TEST} ({how})");
    assert_eq!(rebound, expected.1, "{TEST} ({how}): the rebind");
    assert!(
        read(&queue, &third, 2 * page) == bytes[rebound_at..rebound_at + 2 * page],
        "{TEST} ({how}): the GPU's read of the CPU's bytes through the rebound buffer"
    );
    blits(
        &queue,
        |blit| blit.fill_buffer(&third, 0..2 * page as u64, 0x5e).unwrap(),
        || {},
    );
    assert!(
        ram.read_file(rebound_at, 2 * page)
            .iter()
            .all(|&b| b == 0x5e),
        "{TEST} ({how}): the GPU's fill through the rebound buffer in the file's pages"
    );
    let through_second = read(&queue, &second, length);
    let at = rebound_at - offset;
    assert!(
        through_second[..at] == bytes[offset..rebound_at]
            && through_second[at..].iter().all(|&b| b == 0x5e),
        "{TEST} ({how}): the rebound buffer's fill read through the second"
    );
    println!(
        "{TEST}: on {} ({how}): the source called {} time(s)",
        device.name(),
        calls.load(Ordering::Acquire)
    );
    drop((first, second, third));
}

#[test]
fn without_pointer_imports_a_buffer_inside_a_live_dma_buf_import_is_bound_in_it() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = super::no_pointer_import::device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        eprintln!("{TEST}: on {}: no dma-buf route listed", device.name());
        return;
    }
    inside("without pointer imports", &device);
}

/// The test on `device`: a first buffer over pages 0 to 3 imported as dma-bufs, then inside it a
/// second over pages 1 and 2 from 256 bytes into page 1 whose memory has no source of dma-bufs
/// (as a platform hands a buffer a heap's pages by their pointer alone), and a third over pages 2
/// and 3 whose memory has one.
fn inside(how: &str, device: &Device) {
    let page = host_page();
    let ram = Ram::new(4 * page, 0, 4 * page);
    let bytes = counting(4 * page, 0);
    ram.write_file(0, &bytes);
    let calls = Arc::new(AtomicUsize::new(0));
    let made = |offset: usize, length: usize, sourced: bool| {
        let pointer = ram.pointer(offset);
        // SAFETY: the view outlives every buffer over it (each dropped first), and the source
        // makes dma-bufs of its pages.
        let memory = unsafe {
            match sourced {
                true => HostMemory::with_dma_buf(pointer, ram.source(&calls)),
                false => HostMemory::new(pointer),
            }
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                length as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    };
    let dedicated_before = device.work_counts().dma_buf_buffer_imports_dedicated;
    let first = made(0, 4 * page, true);
    assert_eq!(
        first.host_backing(),
        Some(HostBacking::ImportedAsDmaBuf),
        "{TEST} ({how}): the first buffer"
    );
    let dedicated = device.work_counts().dma_buf_buffer_imports_dedicated > dedicated_before;
    let before = placed(&device.work_counts());
    let (offset, length) = (page + 256, 2 * page - 256);
    let second = made(offset, length, false);
    let made_second = since(before, placed(&device.work_counts()));
    let before = placed(&device.work_counts());
    let (third_at, third_length) = (2 * page, 2 * page);
    let third = made(third_at, third_length, true);
    let made_third = since(before, placed(&device.work_counts()));
    println!(
        "{TEST}: on {} ({how}): the first's import dedicated to it: {dedicated}; the second \
         {:?}, the third {:?}, [copied over live, dma-bufs beside live, imported inside live, \
         pointer imports, bound inside] {made_second:?} {made_third:?}",
        device.name(),
        second.host_backing(),
        third.host_backing(),
    );
    // Bound inside an import dedicated to nothing; else a copy, or beside it with a source where
    // the device's barriers order blits through two imports of the same pages.
    let unordered = (device.properties().driver_defects)
        .contains(&RuntimeDriverDefect::AliasedImportBlitsUnordered);
    let expected = match (dedicated, unordered) {
        (false, _) => [
            (HostBacking::ImportedAsDmaBuf, [0, 0, 0, 0, 1]),
            (HostBacking::ImportedAsDmaBuf, [0, 0, 0, 0, 1]),
        ],
        (true, false) => [
            (HostBacking::Copied, [1, 0, 0, 0, 0]),
            (HostBacking::ImportedAsDmaBuf, [0, 1, 0, 0, 0]),
        ],
        (true, true) => [
            (HostBacking::Copied, [1, 0, 0, 0, 0]),
            (HostBacking::Copied, [1, 0, 0, 0, 0]),
        ],
    };
    assert_eq!(
        (second.host_backing(), made_second),
        (Some(expected[0].0), expected[0].1),
        "{TEST} ({how}): the second, with no source"
    );
    assert_eq!(
        (third.host_backing(), made_third),
        (Some(expected[1].0), expected[1].1),
        "{TEST} ({how}): the third, with a source"
    );
    let queue = device.new_command_queue().unwrap();
    // The GPU reads the CPU's bytes through each.
    assert!(
        read(&queue, &second, length) == bytes[offset..offset + length],
        "{TEST} ({how}): the GPU's read of the CPU's bytes through the second"
    );
    assert!(
        read(&queue, &third, third_length) == bytes[third_at..],
        "{TEST} ({how}): the GPU's read of the CPU's bytes through the third"
    );
    // A CPU write made after a copy is encoded and before its commit reaches it.
    let out = scratch_buffer(device, length);
    let later = counting(4 * page, 7);
    blits(
        &queue,
        |blit| {
            blit.copy_from_buffer_to_buffer(&second, 0, &out, 0, length as u64)
                .unwrap();
        },
        || ram.write_file(0, &later),
    );
    assert!(
        bytes_of(&out, length) == later[offset..offset + length],
        "{TEST} ({how}): the CPU's write before the commit, through the second"
    );
    // A fill through the first and a copy through the second of the bytes they share, either way
    // round: in one encoder where the second lies in an import (ordered by the bytes they share),
    // in two command buffers where it is a copy, which agrees with the pages at its commit and its
    // completion only (Metal orders no two no-copy buffers over one allocation).
    let in_import = expected[0].0 == HostBacking::ImportedAsDmaBuf;
    let apart = match in_import {
        true => "in one encoder",
        false => "in two command buffers",
    };
    let fill_then_read = |fill: &dyn Fn(&mut BlitCommandEncoder),
                          copy: &dyn Fn(&mut BlitCommandEncoder)| {
        if in_import {
            blits(
                &queue,
                |blit| {
                    fill(blit);
                    copy(blit);
                },
                || {},
            );
        } else {
            blits(&queue, fill, || {});
            blits(&queue, copy, || {});
        }
    };
    let shared = 2 * page - 256;
    let out = scratch_buffer(device, shared);
    fill_then_read(
        &|blit| {
            blit.fill_buffer(&first, offset as u64..(offset + shared) as u64, 0xa1)
                .unwrap();
        },
        &|blit| {
            blit.copy_from_buffer_to_buffer(&second, 0, &out, 0, shared as u64)
                .unwrap();
        },
    );
    assert!(
        bytes_of(&out, shared).iter().all(|&b| b == 0xa1),
        "{TEST} ({how}): the first's fill read through the second {apart}"
    );
    fill_then_read(
        &|blit| blit.fill_buffer(&second, 0..shared as u64, 0xb2).unwrap(),
        &|blit| {
            blit.copy_from_buffer_to_buffer(&first, offset as u64, &out, 0, shared as u64)
                .unwrap();
        },
    );
    assert!(
        bytes_of(&out, shared).iter().all(|&b| b == 0xb2),
        "{TEST} ({how}): the second's fill read through the first {apart}"
    );
    // The first released, the second's fill still reaches the file's pages, and the third reads
    // it where they meet.
    drop(first);
    blits(
        &queue,
        |blit| blit.fill_buffer(&second, 0..length as u64, 0xc3).unwrap(),
        || {},
    );
    assert!(
        ram.read_file(offset, length).iter().all(|&b| b == 0xc3),
        "{TEST} ({how}): the second's fill in the file's pages, the first released"
    );
    let through_third = read(&queue, &third, third_length);
    let met = offset + length - third_at;
    assert!(
        through_third[..met].iter().all(|&b| b == 0xc3)
            && through_third[met..] == later[third_at + met..],
        "{TEST} ({how}): the second's fill read through the third"
    );
    println!(
        "{TEST}: on {} ({how}): the source called {} time(s)",
        device.name(),
        calls.load(Ordering::Acquire)
    );
    drop((second, third));
}
