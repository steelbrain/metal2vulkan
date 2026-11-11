//! A copied host buffer's base on a device (`copied.rs`, `copied_base.rs`): a copy the GPU only
//! reads holds nothing beyond itself; the first command buffer that writes it gives it a base,
//! whose bytes lie in the device's memory where it has device-local memory the host maps in a
//! heap its resizable BAR makes as large as its memory (its tags alone, a 64th of the copy's
//! length, in host memory), else in the host's; the GPU's writes reach the pages through it and
//! the CPU's reach the GPU; the base goes with the buffer. And the 256 MiB bench of
//! `tests/copied_bench.rs` with the device's base (`COPIED-256`), ignored.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::copied::add_reading;
use crate::copied::{CopiedBuffers, CopiesReached, READ_BACK_FROM, TAG_BLOCK};
use crate::dirty_pages::DirtyTally;
use crate::work_counts::Counters;
use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, HostBacking, HostMemory,
    ImportGate, ResourceOptions,
};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The copied buffer's bytes.
const LENGTH: usize = 256 << 20;

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
        eprintln!("COPIED-BASE: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

/// Commit one command buffer whose blit encoder `encode` fills, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first page of `buffer` as the GPU reads it: copied into one of the device's own.
fn gpu_page(device: &Device, buffer: &Buffer) -> Vec<u8> {
    let own = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    blit(device, |encoder| {
        (encoder.copy_from_buffer_to_buffer(buffer, 0, &own, 0, 4096)).unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4096) }.to_vec()
}

/// A copied 256 MiB buffer: no base while the GPU only reads it, then a base whose bytes lie in
/// the device's memory where `Device::copied_base` gives it some (the test says which), its tags
/// alone in host memory, else in the host's; a fill and a CPU write reach the other side; the
/// base goes with the buffer.
#[test]
fn a_written_copy_takes_its_base_where_the_device_has_room() {
    let Some(device) = device() else {
        return;
    };
    let on_device = device.copied_base(LENGTH).is_some();
    let named = device.name().to_string();
    eprintln!("COPIED-BASE {named}: the base's bytes in the device's memory: {on_device}");
    let layout = Layout::from_size_align(LENGTH, 1 << 16).unwrap();
    // SAFETY: the layout's size is not zero.
    let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
    let refuse: ImportGate = Arc::new(|_| false);
    // SAFETY: the memory outlives the buffer (freed below, after it), and only the buffer's
    // commands and the test's reads and writes between command buffers use it.
    let memory = unsafe { HostMemory::new(pointer) }.with_import_gate(refuse);
    // SAFETY: as above.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            memory,
            LENGTH as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    assert_eq!(buffer.host_backing(), Some(HostBacking::Copied));
    let held = |device: &Device| {
        let counts = device.work_counts();
        (
            counts.copied_base_host_bytes,
            counts.copied_base_device_bytes,
        )
    };
    // SAFETY: the allocation's bytes, which no command buffer uses between command buffers.
    let pages = unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), LENGTH) };
    pages[7] = 0x77;
    assert_eq!(gpu_page(&device, &buffer)[7], 0x77, "the CPU's write read");
    assert_eq!(held(&device), (0, 0), "no base while the GPU only reads it");
    blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 4096..8192, 0x5A).unwrap();
    });
    // SAFETY: as above.
    let pages = unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), LENGTH) };
    assert!(
        pages[4096..8192].iter().all(|&b| b == 0x5A),
        "the fill merged back"
    );
    let tags = (LENGTH / 64) as u64;
    let want = match on_device {
        true => (tags, LENGTH as u64),
        false => (tags + LENGTH as u64, 0),
    };
    assert_eq!(held(&device), want, "the base's bytes");
    assert!(device.work_counts().copied_bases_made >= 1);
    pages[9] = 0x99;
    assert_eq!(
        gpu_page(&device, &buffer)[9],
        0x99,
        "the CPU's write read again"
    );
    drop(buffer);
    assert_eq!(held(&device), (0, 0), "the base went with the buffer");
    drop(device);
    // SAFETY: allocated above with this layout; nothing uses it now.
    unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
}

/// The 256 MiB bench of `tests/copied_bench.rs`, the base's bytes in the device's memory where
/// it has room for them (`COPIED-256`). Ignored: run it in release with `--ignored --nocapture`.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_256_device_bench() {
    let Some(device) = device() else {
        return;
    };
    let named = device.name().to_string();
    let label = match device.copied_base(1 << 20).is_some() {
        true => format!("{named}, device base"),
        false => format!("{named}, host base (no room on the device)"),
    };
    super::copied_bench::bench_256(&label, &mut |length| device.copied_base(length));
}

/// The rate of a read of a 1 KiB block of a base in the device's memory, the read a comparison
/// makes of a block both sides changed: with the host's streaming loads where it has them
/// (`BaseBytes::read`) and with plain loads, over 1 MiB, the median of five passes
/// (`COPIED-BASE-READ`). Ignored: run it in release with `--ignored --nocapture`.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_base_read_bench() {
    const BYTES: usize = 1 << 20;
    let Some(device) = device() else {
        return;
    };
    let Some(base) = device.copied_base(BYTES) else {
        eprintln!("COPIED-BASE-READ: no base in the device's memory here");
        return;
    };
    let named = device.name().to_string();
    let block = [0x5A_u8; 1024];
    for at in (0..BYTES).step_by(1024) {
        // SAFETY: a block of the base's bytes.
        unsafe { base.write(at, &block) };
    }
    let mut into = [0u8; 1024];
    for streaming in [true, false, true, false] {
        let mut passes: Vec<f64> = (0..5)
            .map(|_| {
                let started = Instant::now();
                for at in (0..BYTES).step_by(1024) {
                    // SAFETY: as above.
                    unsafe {
                        match streaming {
                            true => base.read(at, &mut into),
                            false => base.read_plain(at, &mut into),
                        }
                    }
                }
                started.elapsed().as_secs_f64()
            })
            .collect();
        passes.sort_by(f64::total_cmp);
        let seconds = passes[2];
        assert_eq!(into, block);
        eprintln!(
            "COPIED-BASE-READ {named}, {}: {:.2} us a 1 KiB block ({:.1} MB/s)",
            match streaming {
                true => "read (streaming loads where the host has SSE4.1)",
                false => "plain loads",
            },
            seconds * 1e6 / (BYTES / 1024) as f64,
            BYTES as f64 / 1e6 / seconds,
        );
    }
}

/// A device base's block reads at scale, as the first merge back of a copy every block of which
/// both sides changed makes them (`COPIED-BASE-SCALE`): 16 MiB of 1 KiB blocks read in order on
/// one thread, twice; the same split across four threads, as a comparison splits its blocks, both
/// over bytes one thread read and over bytes none did; one thread again after three seconds of
/// the device idle; and the whole 256 MiB on one thread. Ignored: run it in release with
/// `--ignored --nocapture`.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_base_read_scale_bench() {
    const SPAN: usize = 16 << 20;
    /// The base's bytes, read from the threads of one comparison, each its own blocks.
    struct Reads<'a>(&'a crate::copied::BaseBytes);
    // SAFETY: each thread reads its own blocks, and nothing writes them meanwhile.
    unsafe impl Sync for Reads<'_> {}
    let Some(device) = device() else {
        return;
    };
    let Some(base) = device.copied_base(LENGTH) else {
        eprintln!("COPIED-BASE-SCALE: no base in the device's memory here");
        return;
    };
    let named = device.name().to_string();
    let block = [0x5A_u8; 1024];
    for at in (0..LENGTH).step_by(1024) {
        // SAFETY: a block of the base's bytes.
        unsafe { base.write(at, &block) };
    }
    let reads = Reads(&base);
    let read = |from: usize, to: usize| {
        // The whole `Reads`, which the threads share, not its field.
        let reads = &reads;
        let mut into = [0u8; 1024];
        for at in (from..to).step_by(1024) {
            // SAFETY: a block of the base's bytes, written above.
            unsafe { reads.0.read(at, &mut into) };
        }
        assert_eq!(into, block);
    };
    let report = |what: &str, bytes: usize, started: Instant| {
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "COPIED-BASE-SCALE {named}, {what}: {:.2} us a 1 KiB block ({:.1} MB/s)",
            seconds * 1e6 / (bytes / 1024) as f64,
            bytes as f64 / 1e6 / seconds,
        );
    };
    let split = |from: usize| {
        std::thread::scope(|scope| {
            for quarter in 0..4 {
                let read = &read;
                let at = from + quarter * SPAN / 4;
                scope.spawn(move || read(at, at + SPAN / 4));
            }
        });
    };
    for pass in 1..=2 {
        let started = Instant::now();
        read(0, SPAN);
        report(&format!("16 MiB, one thread, pass {pass}"), SPAN, started);
    }
    let started = Instant::now();
    split(0);
    report("the same 16 MiB, four threads", SPAN, started);
    for pass in 1..=2 {
        let started = Instant::now();
        split(SPAN);
        report(
            &format!("16 MiB more, four threads, pass {pass}"),
            SPAN,
            started,
        );
    }
    std::thread::sleep(Duration::from_secs(3));
    let started = Instant::now();
    read(2 * SPAN, 3 * SPAN);
    report("16 MiB more, one thread, after 3 s idle", SPAN, started);
    let started = Instant::now();
    read(0, LENGTH);
    report("256 MiB, one thread", LENGTH, started);
}

/// A plain read of a test's pages, a copy's first.
fn plain(from: NonNull<u8>, into: &mut [u8]) -> bool {
    // SAFETY: the tests' pages are `into.len()` bytes of plain memory.
    unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// The `n`th byte of the pages [`a_whole_written_device_base_is_read_back_by_the_gpu`] makes.
fn page_byte(n: usize) -> u8 {
    (n ^ (n >> 11)) as u8
}

/// A first merge back of a 256 MiB copy every block of which both sides changed, its base in the
/// device's memory, after the device sat idle three seconds (where the host's reads of that
/// memory are slowest, about 0.2 ms a 1 KiB block on an RTX 5080 Laptop GPU's host): the
/// comparison reads every block back through one copy on the staging queue
/// (`WorkCounts::copied_base_blocks_read_back`), in under a second, and merges exactly, the
/// CPU's byte of each block kept and every other byte the GPU's.
#[test]
fn a_whole_written_device_base_is_read_back_by_the_gpu() {
    let Some(device) = device() else {
        return;
    };
    if device.copied_base(1 << 20).is_none() {
        eprintln!("COPIED-BASE: no base in the device's memory here; nothing to test");
        return;
    }
    let named = device.name().to_string();
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let mut pages: Vec<u8> = (0..LENGTH).map(page_byte).collect();
    let mut copy = vec![0_u8; LENGTH];
    let at = |bytes: &mut Vec<u8>| NonNull::new(bytes.as_mut_ptr()).unwrap();
    // SAFETY: the pages and the copy are the test's, of `LENGTH` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            LENGTH,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    let all = CopiesReached::all();
    let made = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut |length| {
        device.copied_base(length)
    });
    assert_eq!(made, Ok(()));
    assert_eq!(
        buffers.base_bytes().1,
        LENGTH as u64,
        "the base in the device's memory"
    );
    // The GPU rewrites every float as `x + i`; the CPU a byte of every block.
    for (index, float) in copy.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *float = (f32::from_le_bytes(*float) + index as f32).to_le_bytes();
    }
    for block in pages.chunks_mut(TAG_BLOCK) {
        block[1] = block[1].wrapping_add(1);
    }
    std::thread::sleep(Duration::from_secs(3));
    let started = Instant::now();
    buffers.merge_back(&counts, &all);
    let took = started.elapsed();
    let counted = counts.copied_tallies();
    eprintln!(
        "COPIED-BASE {named}: a first merge back of 256 MiB both sides wrote whole, after 3 s \
         idle: {:.1} ms, {} blocks read back",
        took.as_secs_f64() * 1e3,
        counted.base_blocks_read_back,
    );
    let blocks = (LENGTH / TAG_BLOCK) as u64;
    assert_eq!(counted.base_blocks_read, blocks);
    assert_eq!(
        counted.base_blocks_read_back, blocks,
        "every block read back"
    );
    assert!(took < Duration::from_secs(1), "{took:?}");
    for (n, (&got, &gpu)) in pages.iter().zip(&copy).enumerate() {
        let want = match n % TAG_BLOCK {
            1 => page_byte(n).wrapping_add(1),
            _ => gpu,
        };
        assert_eq!(got, want, "byte {n}");
    }
    drop(copied);
}

/// The cost of a comparison's reads of `n` blocks of a base in the device's memory that both
/// sides changed, each read by the host against all read back through one copy on the staging
/// queue, for `n` from 8 to 4096, each the median of three first merges back of fresh blocks of
/// a 128 MiB copy, every other block (`COPIED-READ-BACK`); and 16, 64 and 651 after three seconds
/// of the device idle, once each: what sets `copied.rs`'s `READ_BACK_BLOCKS`. Ignored: run it in
/// release with `--ignored --nocapture`.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_read_back_bench() {
    const BYTES: usize = 128 << 20;
    let Some(device) = device() else {
        return;
    };
    if device.copied_base(1 << 20).is_none() {
        eprintln!("COPIED-READ-BACK: no base in the device's memory here");
        return;
    }
    let named = device.name().to_string();
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let mut pages: Vec<u8> = (0..BYTES).map(page_byte).collect();
    let mut copy = vec![0_u8; BYTES];
    let at = |bytes: &mut Vec<u8>| NonNull::new(bytes.as_mut_ptr()).unwrap();
    // SAFETY: the pages and the copy are the bench's, of `BYTES` bytes, live past the answer.
    let copied =
        unsafe { add_reading(&buffers, at(&mut pages), at(&mut copy), BYTES, plain, false) };
    let copied = copied.unwrap();
    let all = CopiesReached::all();
    let made = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut |length| {
        device.copied_base(length)
    });
    assert_eq!(made, Ok(()));
    // Fresh blocks, every other one, so no two read back are adjacent.
    let mut next = 0usize;
    let mut merge = |n: usize, read_back: bool, pages: &mut Vec<u8>, copy: &mut Vec<u8>| {
        READ_BACK_FROM.store(if read_back { 1 } else { usize::MAX }, Ordering::Relaxed);
        for _ in 0..n {
            let at = next * 2 * TAG_BLOCK;
            next += 1;
            copy[at] = copy[at].wrapping_add(1);
            pages[at + 1] = pages[at + 1].wrapping_add(1);
        }
        let before = counts.copied_tallies().base_blocks_read_back;
        let started = Instant::now();
        buffers.merge_back(&counts, &all);
        let took = started.elapsed().as_secs_f64() * 1e3;
        let back = counts.copied_tallies().base_blocks_read_back - before;
        assert_eq!(back, if read_back { n as u64 } else { 0 });
        took
    };
    for n in [8, 16, 32, 64, 128, 256, 1024, 4096] {
        let (mut host, mut back) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            host.push(merge(n, false, &mut pages, &mut copy));
            back.push(merge(n, true, &mut pages, &mut copy));
        }
        host.sort_by(f64::total_cmp);
        back.sort_by(f64::total_cmp);
        eprintln!(
            "COPIED-READ-BACK {named}, {n} blocks: host reads {:.2} ms, read back {:.2} ms",
            host[1], back[1],
        );
    }
    for n in [16, 64, 651] {
        std::thread::sleep(Duration::from_secs(3));
        let host = merge(n, false, &mut pages, &mut copy);
        std::thread::sleep(Duration::from_secs(3));
        let back = merge(n, true, &mut pages, &mut copy);
        eprintln!(
            "COPIED-READ-BACK {named}, {n} blocks after 3 s idle: host reads {host:.2} ms, read \
             back {back:.2} ms",
        );
    }
    READ_BACK_FROM.store(0, Ordering::Relaxed);
    drop(copied);
}
