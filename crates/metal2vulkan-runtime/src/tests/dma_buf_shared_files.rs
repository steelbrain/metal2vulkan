//! No-copy buffers over pieces of shared dma-bufs (`dma_buf.rs`'s `DmaBufFiles`), on the device
//! the cfg selects made without `VK_EXT_external_memory_host` (`tests/no_pointer_import.rs`), so
//! that every device that lists the dma-buf route takes it. The memory is guest RAM as QEMU makes
//! it: a sealed `memfd` mapped shared, whose view maps the file from its first byte; the shared
//! file is one `udmabuf` of the file's pages, whose pieces a source hands each buffer (the same
//! `Arc`, the source holding its own clone), as a consumer groups its guest's buffers.
//!
//! - **One import.** Four buffers over pieces of one file: one import of the file
//!   (`WorkCounts::dma_buf_imports`), the three others joined to it (`dma_buf_pieces_joined`),
//!   one shared import live (`dma_buf_shared_imports_live`, within `dma_buf_memories_live`). Each
//!   buffer's GPU reads are its own piece's bytes, its GPU fill lands in its piece's pages and no
//!   other's, and the CPU's later writes to the file reach its next read. The import lives while
//!   any of the buffers does, and goes with the last.
//! - **Its negative control.** The same buffers whose source answers a dma-buf of its own for
//!   each: one import each, none joined, no shared import.
//! - **Several pieces.** A buffer longer than the device's sparse granule, answered with two
//!   pieces of one file laid in it in the other order: each bound at its offset in the one
//!   import (its second piece joined), the buffer's bytes in order. Where the device binds no
//!   buffer in several pieces (the request's granule is its length) the line says so and one
//!   piece is answered.
//! - **Released and imported again.** Two buffers over pieces of one file, read by blits, their
//!   imports released while idle (`Device::release_idle_host_imports`): the file's import goes
//!   with the second release; the next blit of each makes the import again and joins it, its
//!   bytes the file's.
//! - **A file counts once.** With the device keeping at most one listed import of its releasable
//!   ones (`Device::keep_resident_host_imports`): four buffers over pieces of one file hold one,
//!   so rounds of blits read them with no release and no import again
//!   (`WorkCounts::host_imports_released_over_most`, `host_imports_reimported`); its negative
//!   control, four buffers of dma-bufs of their own, has the three least recently read released
//!   past the most and imported again at their next blits. Two files of two buffers each: the
//!   least recently read file's buffers are released together, its import freed once
//!   (`dma_buf_shared_imports_live`), the other's read with no import again; once one of the
//!   other file's buffers is kept imported for good (its Metal address handed out), the other
//!   buffer over that file is released no more, even idle.
//! - **Misuse.** A piece whose offset in its file misses the request's alignment is refused
//!   (`Invalid`, `Problem::DmaBuf`), nothing imported.
//!
//! Each prints one line (`DMA-BUF-SHARED`) with the counts it checks. A device test: compiled only
//! under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux. Where `/dev/udmabuf` cannot be
//! opened, or the device lists no dma-buf route, it says so and passes without running.

use super::host_residency::{
    counting, counts_once, gpu_fill, host_page, read, release, udmabuf, udmabuf_list,
    udmabuf_openable, Ram, Scratch,
};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::{
    Buffer, Device, DmaBufChunk, DmaBufRequest, DmaBufSource, Error, HostMemory, Invalid, Problem,
    ResourceOptions, WorkCounts,
};
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const TEST: &str = "DMA-BUF-SHARED";

/// The buffers over pieces of one file.
const BUFFERS: usize = 4;

/// The device of these tests, with its shader cache's scratch directory; `None`, said, where
/// `/dev/udmabuf` cannot be opened or the device lists no dma-buf route.
fn on_device(test: &str) -> Option<(Scratch, Device)> {
    if !udmabuf_openable(test) {
        return None;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        eprintln!("{test}: {} lists no dma-buf route: not run", device.name());
        return None;
    }
    Some((scratch, device))
}

/// A source that answers each request with one piece of `file`, a dma-buf of `ram`'s file from
/// its first byte, at the request's offset in the view (which maps the file from its first byte)
/// plus `skew`, keeping the largest alignment asked in `alignment`.
fn shared_source(
    ram: &Ram,
    file: &Arc<OwnedFd>,
    skew: u64,
    alignment: &Arc<AtomicU64>,
) -> DmaBufSource {
    let base = ram.pointer(0).as_ptr() as u64;
    let file = Arc::clone(file);
    let alignment = Arc::clone(alignment);
    Arc::new(move |request: DmaBufRequest| {
        alignment.fetch_max(request.alignment, Ordering::AcqRel);
        let at = (request.start.as_ptr() as u64)
            .checked_sub(base)
            .ok_or_else(|| std::io::Error::other("pages before the view"))?;
        Ok(vec![DmaBufChunk {
            file: Arc::clone(&file),
            offset: at + skew,
            length: request.length,
        }])
    })
}

/// A no-copy buffer of `length` bytes over `ram`'s view from `offset`, of `source`'s dma-bufs.
fn no_copy(
    device: &Device,
    ram: &Ram,
    (offset, length): (usize, usize),
    source: DmaBufSource,
) -> Result<Buffer, Error> {
    // SAFETY: each test's view outlives its buffers, and the source makes dma-bufs of its pages.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(offset), source),
            length as u64,
            ResourceOptions::default(),
            None,
        )
    }
}

/// The dma-buf counts of `after` since `before`: imports made, pieces joined, the change of the
/// shared imports live and of the dma-buf memories live.
fn moved(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, i64, i64) {
    let live = |counts: &WorkCounts| {
        (
            counts.dma_buf_shared_imports_live as i64,
            counts.dma_buf_memories_live as i64,
        )
    };
    let (shared_before, memories_before) = live(before);
    let (shared_after, memories_after) = live(after);
    (
        after.dma_buf_imports - before.dma_buf_imports,
        after.dma_buf_pieces_joined - before.dma_buf_pieces_joined,
        shared_after - shared_before,
        memories_after - memories_before,
    )
}

/// Four buffers over pieces of one shared file, 16 pages apart: one import, three joined; each
/// buffer's reads, fills and the CPU's later writes in its own piece's pages; the import kept
/// while one buffer lives, and freed with the last.
#[test]
fn pieces_of_one_shared_file_lie_in_one_import() {
    const NAME: &str = "pieces_of_one_shared_file_lie_in_one_import";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = BUFFERS * stride;
    let ram = Ram::new(len, 0, len);
    let mut expected = counting(len, 3);
    ram.write_file(0, &expected);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let before = device.work_counts();
    let buffers: Vec<Buffer> = (0..BUFFERS)
        .map(|index| {
            no_copy(&device, &ram, (index * stride, length), Arc::clone(&source)).unwrap_or_else(
                |error| {
                    panic!(
                        "{NAME}: buffer {index}: {error:?} (alignment asked {})",
                        alignment.load(Ordering::Acquire)
                    )
                },
            )
        })
        .collect();
    let made = device.work_counts();
    let (imports, joined, shared, memories) = moved(&before, &made);
    eprintln!(
        "{TEST}: {NAME} on {}: {BUFFERS} buffers over one file: imports {imports}, joined \
         {joined}, shared imports live +{shared}, dma-buf memories live +{memories}, alignment \
         asked {}",
        device.name(),
        alignment.load(Ordering::Acquire)
    );
    assert_eq!(
        (imports, joined, shared, memories),
        (1, BUFFERS as u64 - 1, 1, 1),
        "{NAME}: one import of the file, every other piece joined"
    );
    let queue = device.new_command_queue().unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        let at = index * stride;
        assert_eq!(
            read(&queue, buffer, length),
            expected[at..at + length],
            "{NAME}: buffer {index} reads its own piece"
        );
        let value = 0xa0 + index as u8;
        gpu_fill(&queue, buffer, page as u64..page as u64 + 256, value);
        expected[at + page..at + page + 256].fill(value);
    }
    assert_eq!(
        ram.read_file(0, len),
        expected,
        "{NAME}: each fill in its own piece's pages, the rest as it was"
    );
    for (index, buffer) in buffers.iter().enumerate() {
        let bytes = vec![0x5a ^ index as u8; 64];
        ram.write_file(index * stride + 64, &bytes);
        assert_eq!(
            read(&queue, buffer, 128)[64..],
            bytes[..],
            "{NAME}: buffer {index} reads the CPU's later writes"
        );
    }
    drop(buffers);
    let freed = |counts: &WorkCounts| {
        counts.dma_buf_shared_imports_live == before.dma_buf_shared_imports_live
            && counts.dma_buf_memories_live == before.dma_buf_memories_live
    };
    counts_once(NAME, &device, "the import freed with the buffers", freed);
    // Two buffers no command used, whose releases free at once: the file imported anew, the
    // import kept while the second lies in it, freed with it.
    let first = no_copy(&device, &ram, (0, length), Arc::clone(&source)).unwrap();
    let second = no_copy(&device, &ram, (stride, length), Arc::clone(&source)).unwrap();
    drop(source);
    drop(file);
    let fresh = device.work_counts();
    drop(first);
    let kept = device.work_counts();
    assert_eq!(
        (moved(&made, &fresh).0, moved(&made, &fresh).1),
        (1, 1),
        "{NAME}: the file imported anew, the second joined"
    );
    assert_eq!(
        (moved(&before, &kept).2, moved(&before, &kept).3),
        (1, 1),
        "{NAME}: the file's import kept while one buffer lies in it"
    );
    assert_eq!(
        read(&queue, &second, length)[..page],
        ram.read_file(stride, page)[..],
        "{NAME}: the buffer left reads its piece"
    );
    drop(second);
    counts_once(NAME, &device, "the import freed with the last", freed);
}

/// The negative control: the same four buffers, each answered with a dma-buf of its own: an
/// import each, none joined, no shared import live.
#[test]
fn chunks_of_their_own_are_an_import_each() {
    const NAME: &str = "chunks_of_their_own_are_an_import_each";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = BUFFERS * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 5);
    ram.write_file(0, &expected);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let before = device.work_counts();
    let buffers: Vec<Buffer> = (0..BUFFERS)
        .map(|index| no_copy(&device, &ram, (index * stride, length), ram.source(&calls)).unwrap())
        .collect();
    let made = device.work_counts();
    let (imports, joined, shared, memories) = moved(&before, &made);
    eprintln!(
        "{TEST}: {NAME} on {}: {BUFFERS} buffers of their own dma-bufs: imports {imports}, joined \
         {joined}, shared imports live +{shared}, dma-buf memories live +{memories}",
        device.name()
    );
    assert_eq!(
        (imports, joined, shared, memories),
        (BUFFERS as u64, 0, 0, BUFFERS as i64),
        "{NAME}: an import each"
    );
    let queue = device.new_command_queue().unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        let at = index * stride;
        assert_eq!(read(&queue, buffer, length), expected[at..at + length]);
    }
    drop(buffers);
    counts_once(NAME, &device, "the imports freed", |counts| {
        counts.dma_buf_memories_live == before.dma_buf_memories_live
    });
}

/// A buffer longer than the device's sparse granule over two pieces of one file, laid in the file
/// in the other order: each bound at its offset in the file's one import, the second joined, the
/// buffer's bytes in order and a fill across the pieces' boundary in the file's pages there.
/// Where the request's granule is its length the device binds one piece, which the line says.
#[test]
fn a_buffer_over_two_pieces_of_one_file_binds_each_at_its_offset() {
    const NAME: &str = "a_buffer_over_two_pieces_of_one_file_binds_each_at_its_offset";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let length = 256 << 10;
    let len = 2 * length;
    let ram = Ram::new(len, 0, len);
    let mut expected = counting(len, 11);
    ram.write_file(0, &expected);
    let asked = Arc::new(std::sync::Mutex::new(None));
    // The file: the request's pages past the granule, then those before it.
    let made_file = {
        let asked = Arc::clone(&asked);
        let ram_file = ram.file().try_clone().unwrap();
        move |request: DmaBufRequest| -> std::io::Result<Vec<DmaBufChunk>> {
            *asked.lock().unwrap() = Some((request.granule, request.length, request.alignment));
            let (granule, length) = (request.granule, request.length);
            if granule >= length {
                let file = udmabuf(&ram_file, 0, length)?;
                return Ok(vec![DmaBufChunk {
                    file: Arc::new(file),
                    offset: 0,
                    length,
                }]);
            }
            let rounded = length.next_multiple_of(granule);
            let file = Arc::new(udmabuf_list(
                &ram_file,
                &[(granule, rounded - granule), (0, granule)],
            )?);
            Ok(vec![
                DmaBufChunk {
                    file: Arc::clone(&file),
                    offset: rounded - granule,
                    length: granule,
                },
                DmaBufChunk {
                    file,
                    offset: 0,
                    length: rounded - granule,
                },
            ])
        }
    };
    let before = device.work_counts();
    let buffer = no_copy(&device, &ram, (0, length), Arc::new(made_file)).unwrap();
    let made = device.work_counts();
    let answered = *asked.lock().unwrap();
    let (granule, _, alignment) = answered.expect("the source was asked");
    let (imports, joined, shared, _) = moved(&before, &made);
    let sparse = made.dma_buf_sparse_buffers - before.dma_buf_sparse_buffers;
    let split = granule < length as u64;
    eprintln!(
        "{TEST}: {NAME} on {}: granule {granule}, alignment {alignment}: {}: imports {imports}, \
         joined {joined}, shared imports live +{shared}, sparse buffers {sparse}",
        device.name(),
        match split {
            true => "two pieces of one file",
            false => "the device binds one piece: one dma-buf of its own",
        }
    );
    match split {
        true => assert_eq!((imports, joined, shared, sparse), (1, 1, 1, 1), "{NAME}"),
        false => assert_eq!((imports, joined, shared, sparse), (1, 0, 0, 0), "{NAME}"),
    }
    let queue = device.new_command_queue().unwrap();
    assert_eq!(
        read(&queue, &buffer, length),
        expected[..length],
        "{NAME}: the buffer's bytes in order"
    );
    let boundary = match split {
        true => granule as usize,
        false => length / 2,
    };
    let across = (boundary - 128) as u64..(boundary + 128) as u64;
    gpu_fill(&queue, &buffer, across, 0xc3);
    expected[boundary - 128..boundary + 128].fill(0xc3);
    assert_eq!(
        ram.read_file(0, len)[..length],
        expected[..length],
        "{NAME}: a fill across the pieces' boundary in the file's pages there ({page}-byte pages)"
    );
}

/// Two buffers over pieces of one file, read by blits, then released while idle: the file's
/// import goes with the second release; each buffer's next blit makes the import again or joins
/// it, reading its piece.
#[test]
fn released_pieces_import_their_file_again() {
    const NAME: &str = "released_pieces_import_their_file_again";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = 2 * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 13);
    ram.write_file(0, &expected);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let before = device.work_counts();
    let buffers: Vec<Buffer> = (0..2)
        .map(|index| no_copy(&device, &ram, (index * stride, length), Arc::clone(&source)).unwrap())
        .collect();
    let queue = device.new_command_queue().unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        let at = index * stride;
        assert_eq!(read(&queue, buffer, length), expected[at..at + length]);
    }
    release(NAME, &device, 2);
    let released = device.work_counts();
    let (imports, joined, shared, memories) = moved(&before, &released);
    assert_eq!(
        (imports, joined, shared, memories),
        (1, 1, 0, 0),
        "{NAME}: the file's import goes with both buffers released"
    );
    let changed = vec![0x77; 64];
    ram.write_file(stride, &changed);
    let mut now = expected.clone();
    now[stride..stride + 64].copy_from_slice(&changed);
    for (index, buffer) in buffers.iter().enumerate() {
        let at = index * stride;
        assert_eq!(
            read(&queue, buffer, length),
            now[at..at + length],
            "{NAME}: buffer {index} imported again reads its piece"
        );
    }
    let again = device.work_counts();
    let (imports, joined, shared, _) = moved(&released, &again);
    eprintln!(
        "{TEST}: {NAME} on {}: released 2 (shared imports live back to +0), imported again: \
         imports {imports}, joined {joined}, shared imports live +{shared}, reimported {}",
        device.name(),
        again.host_imports_reimported - released.host_imports_reimported
    );
    assert_eq!(
        (imports, joined, shared),
        (0, 1, 1),
        "{NAME}: one import of the file again (imports again are not counted as new imports), \
         the other joined"
    );
    assert_eq!(
        again.host_imports_reimported - released.host_imports_reimported,
        2
    );
}

/// Read each of `buffers` by a blit, `stride` bytes apart in `expected`, `length` bytes each.
fn read_each(
    name: &str,
    queue: &crate::CommandQueue,
    buffers: &[Buffer],
    expected: &[u8],
    (stride, length): (usize, usize),
) {
    for (index, buffer) in buffers.iter().enumerate() {
        let at = index * stride;
        assert_eq!(
            read(queue, buffer, length),
            expected[at..at + length],
            "{name}: buffer {index} reads its pages"
        );
    }
}

/// The device's counts once `count` imports more than in `before` were released past the most,
/// by the hook or by the device's residency thread, whichever comes first.
fn released_past_most(name: &str, device: &Device, before: &WorkCounts, count: u64) -> WorkCounts {
    let start = std::time::Instant::now();
    loop {
        let counts = device.work_counts();
        let released =
            counts.host_imports_released_over_most - before.host_imports_released_over_most;
        if released >= count {
            return counts;
        }
        assert!(
            start.elapsed() < crate::tests::SETTLE,
            "{name}: {released} of {count} released past the most"
        );
        device.release_host_imports_over_most();
        std::thread::yield_now();
    }
}

/// The residency counts of `after` since `before`: released, released past the most, imported
/// again; and the releasable imports holding their memory now.
fn residency_moved(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, u64, u64) {
    (
        after.host_imports_released - before.host_imports_released,
        after.host_imports_released_over_most - before.host_imports_released_over_most,
        after.host_imports_reimported - before.host_imports_reimported,
        after.host_imports_resident,
    )
}

/// Four buffers over pieces of one file, read by blits, the device keeping at most one listed
/// import of its releasable ones: four releasable imports holding one listed import, so none is
/// past the most. Rounds of blits read each buffer's piece with no release and no import again,
/// and the releases past the most release none.
#[test]
fn pieces_of_one_file_past_the_most_stay_while_read() {
    const NAME: &str = "pieces_of_one_file_past_the_most_stay_while_read";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = BUFFERS * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 29);
    ram.write_file(0, &expected);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let buffers: Vec<Buffer> = (0..BUFFERS)
        .map(|index| no_copy(&device, &ram, (index * stride, length), Arc::clone(&source)).unwrap())
        .collect();
    let queue = device.new_command_queue().unwrap();
    read_each(NAME, &queue, &buffers, &expected, (stride, length));
    let before = device.work_counts();
    device.keep_resident_host_imports(1);
    for _ in 0..3 {
        assert_eq!(
            device.release_host_imports_over_most(),
            0,
            "{NAME}: one listed import is not past the most"
        );
        read_each(NAME, &queue, &buffers, &expected, (stride, length));
    }
    let after = device.work_counts();
    let moved = residency_moved(&before, &after);
    eprintln!(
        "{TEST}: {NAME} on {}: {BUFFERS} buffers over one file, the most 1: released {}, past \
         the most {}, imported again {}, releasable imports resident {} (shared imports live {})",
        device.name(),
        moved.0,
        moved.1,
        moved.2,
        moved.3,
        after.dma_buf_shared_imports_live
    );
    assert_eq!(
        moved,
        (0, 0, 0, BUFFERS as u64),
        "{NAME}: no release, no import again"
    );
}

/// The negative control: four buffers of dma-bufs of their own, read by blits, the device keeping
/// at most one listed import: the three least recently read are released past the most, and each
/// is imported again at its next blit.
#[test]
fn chunks_of_their_own_past_the_most_are_released() {
    const NAME: &str = "chunks_of_their_own_past_the_most_are_released";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = BUFFERS * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 31);
    ram.write_file(0, &expected);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let buffers: Vec<Buffer> = (0..BUFFERS)
        .map(|index| no_copy(&device, &ram, (index * stride, length), ram.source(&calls)).unwrap())
        .collect();
    let queue = device.new_command_queue().unwrap();
    read_each(NAME, &queue, &buffers, &expected, (stride, length));
    let before = device.work_counts();
    device.keep_resident_host_imports(1);
    let past = BUFFERS as u64 - 1;
    let released = released_past_most(NAME, &device, &before, past);
    assert_eq!(
        residency_moved(&before, &released),
        (past, past, 0, 1),
        "{NAME}: every import but the most recently read released past the most"
    );
    // The released ones, read again: each imported again.
    let others = &buffers[..BUFFERS - 1];
    read_each(NAME, &queue, others, &expected, (stride, length));
    let again = device.work_counts();
    let moved = residency_moved(&before, &again);
    eprintln!(
        "{TEST}: {NAME} on {}: {BUFFERS} buffers of their own dma-bufs, the most 1: released \
         past the most {}, then {} imported again reading {}",
        device.name(),
        moved.1,
        moved.2,
        others.len()
    );
    assert_eq!(moved.2, past, "{NAME}: each released one imported again");
}

/// Two files of two buffers each, read by blits (the first file's a few milliseconds before the
/// second's), the device keeping at most one listed import: the first file's two buffers are
/// released together past the most and its import freed once; the second's are read with no
/// import again. Once the second file's first buffer is kept imported for good (its Metal address
/// handed out), its second is released no more, even idle and forced. The first file's buffers,
/// read again, import it once and join it.
#[test]
fn a_file_s_buffers_are_released_together_and_only_whole() {
    const NAME: &str = "a_file_s_buffers_are_released_together_and_only_whole";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let half = 2 * stride;
    let len = 2 * half;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 37);
    ram.write_file(0, &expected);
    let files: Vec<Arc<OwnedFd>> = (0..2)
        .map(|index| Arc::new(udmabuf(ram.file(), (index * half) as u64, half as u64).unwrap()))
        .collect();
    let base = ram.pointer(0).as_ptr() as u64;
    let source: DmaBufSource = Arc::new(move |request: DmaBufRequest| {
        let at = (request.start.as_ptr() as u64)
            .checked_sub(base)
            .ok_or_else(|| std::io::Error::other("pages before the view"))?;
        let (index, offset) = (at / half as u64, at % half as u64);
        let file = (files.get(index as usize))
            .ok_or_else(|| std::io::Error::other("pages past the files"))?;
        Ok(vec![DmaBufChunk {
            file: Arc::clone(file),
            offset,
            length: request.length,
        }])
    });
    let made = device.work_counts();
    let buffers: Vec<Buffer> = (0..4)
        .map(|index| no_copy(&device, &ram, (index * stride, length), Arc::clone(&source)).unwrap())
        .collect();
    let queue = device.new_command_queue().unwrap();
    read_each(NAME, &queue, &buffers[..2], &expected, (stride, length));
    std::thread::sleep(std::time::Duration::from_millis(5));
    read_each(
        NAME,
        &queue,
        &buffers[2..],
        &expected[half..],
        (stride, length),
    );
    let before = device.work_counts();
    assert_eq!(
        (moved(&made, &before).0, moved(&made, &before).2),
        (2, 2),
        "{NAME}: one import of each file"
    );
    device.keep_resident_host_imports(1);
    let released = released_past_most(NAME, &device, &before, 2);
    let (freed_imports, _, freed_shared, freed_memories) = moved(&before, &released);
    assert_eq!(
        (
            residency_moved(&before, &released),
            freed_imports,
            freed_shared,
            freed_memories,
        ),
        ((2, 2, 0, 2), 0, -1, -1),
        "{NAME}: the first file's buffers released together, its import freed once"
    );
    read_each(
        NAME,
        &queue,
        &buffers[2..],
        &expected[half..],
        (stride, length),
    );
    assert_eq!(
        device.work_counts().host_imports_reimported,
        before.host_imports_reimported,
        "{NAME}: the second file's buffers kept"
    );
    // The second file's first buffer kept for good: its second goes no more.
    buffers[2].gpu_address();
    assert_eq!(
        device.release_idle_host_imports(),
        0,
        "{NAME}: a file a buffer kept for good holds keeps its other pieces"
    );
    read_each(
        NAME,
        &queue,
        &buffers[3..],
        &expected[half + stride..],
        (stride, length),
    );
    // The first file's buffers read again: its import made once more, the second joining it.
    let kept = device.work_counts();
    read_each(NAME, &queue, &buffers[..2], &expected, (stride, length));
    let again = device.work_counts();
    let (imports, joined, _, _) = moved(&kept, &again);
    eprintln!(
        "{TEST}: {NAME} on {}: two files of two buffers, the most 1: released together {} past \
         the most (shared imports live {freed_shared:+} then), the other file's kept; read again: \
         imports {imports}, joined {joined}, shared imports live +{}, imported again {}",
        device.name(),
        released.host_imports_released_over_most - before.host_imports_released_over_most,
        moved(&released, &again).2,
        again.host_imports_reimported - kept.host_imports_reimported,
    );
    assert_eq!(
        (
            kept.host_imports_reimported - before.host_imports_reimported,
            again.host_imports_reimported - kept.host_imports_reimported,
            joined,
            moved(&released, &again).2,
        ),
        (0, 2, 1, 1),
        "{NAME}: the second file's buffers never imported again; the first file's import made \
         again once, the other buffer joining it"
    );
}

/// A piece whose offset in its file misses the request's alignment (half a page in) is misuse:
/// refused (`Invalid`, `Problem::DmaBuf`), nothing imported.
#[test]
fn a_piece_off_the_request_s_alignment_is_refused() {
    const NAME: &str = "a_piece_off_the_request_s_alignment_is_refused";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let len = 4 * page;
    let ram = Ram::new(len, 0, len);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, page as u64 / 2, &alignment);
    let before = device.work_counts();
    let made = no_copy(&device, &ram, (0, 2 * page), source);
    let after = device.work_counts();
    eprintln!(
        "{TEST}: {NAME} on {}: a piece {} bytes in, alignment {}: {:?}",
        device.name(),
        page / 2,
        alignment.load(Ordering::Acquire),
        made.as_ref().err()
    );
    assert!(
        matches!(
            made,
            Err(Error::Invalid(Invalid {
                problem: Problem::DmaBuf,
                ..
            }))
        ),
        "{NAME}: {:?}",
        made.err()
    );
    assert_eq!(
        moved(&before, &after),
        (0, 0, 0, 0),
        "{NAME}: nothing imported"
    );
}
