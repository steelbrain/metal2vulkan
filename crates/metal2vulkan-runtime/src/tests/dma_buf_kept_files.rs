//! A shared dma-buf imported ahead and kept (`Device::import_shared_dma_buf`, `dma_buf/kept.rs`),
//! on the device of `dma_buf_shared_files.rs` (made without `VK_EXT_external_memory_host`, so that
//! every device that lists the dma-buf route takes it), over guest RAM as QEMU makes it: a sealed
//! `memfd` mapped shared, the shared file one `udmabuf` of its pages.
//!
//! - **Kept across buffers.** With a handle, three buffers over pieces of the file, each made,
//!   read and dropped in turn: one import of the file (the handle's), three pieces joined, the
//!   import live and kept throughout, and freed as the handle drops. Its negative control, the
//!   same buffers with no handle: an import each (`WorkCounts::dma_buf_imports` +3 against +1).
//! - **A live import kept, and left to its buffer.** A handle made while a buffer's import lives
//!   keeps that one (no import more); the buffer dropped, the import stays; a second buffer joins
//!   it; the handle dropped under that buffer leaves the import to it, freed with it.
//! - **Recovery.** A handle kept across the device's recovery with no buffer: nothing imported on
//!   the new device as it recovers (its `Debug` says so), the next buffer imports the file there
//!   and the handle keeps that import past the buffer. With a buffer live at the recovery: the
//!   carried buffer imports the file on the new device, kept from then on.
//! - **Misuse.** `Private` storage is refused (`Invalid`, `Problem::StorageMode`), nothing
//!   imported.
//!
//! Each prints one line (`DMA-BUF-KEPT`) with the counts it checks. A device test: compiled only
//! under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux. Where `/dev/udmabuf` cannot be
//! opened, or the device lists no dma-buf route, it says so and passes without running.

use super::dma_buf_shared_files::{importing, moved, no_copy, on_device, shared_source};
use super::host_residency::{counting, counts_once, host_page, read, udmabuf, Ram};
use crate::{Buffer, Error, Invalid, Problem, ResourceOptions, WorkCounts};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

const TEST: &str = "DMA-BUF-KEPT";

/// The buffers made and dropped in turn.
const BUFFERS: usize = 3;

/// The kept imports live in `counts`.
fn kept(counts: &WorkCounts) -> u64 {
    counts.dma_buf_shared_imports_kept
}

/// Three buffers over pieces of one file, made, read and dropped in turn, with a handle and
/// without: one import against three.
#[test]
fn a_kept_file_is_imported_once_across_buffers_that_come_and_go() {
    const NAME: &str = "a_kept_file_is_imported_once_across_buffers_that_come_and_go";
    let Some((_scratch, device)) = importing(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = BUFFERS * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 7);
    ram.write_file(0, &expected);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let queue = device.new_command_queue().unwrap();
    // Each buffer made, read and dropped, its pages' release settled before the next.
    let in_turn = |base: &WorkCounts| {
        for index in 0..BUFFERS {
            let at = index * stride;
            let buffer = no_copy(&device, &ram, (at, length), Arc::clone(&source)).unwrap();
            assert_eq!(
                read(&queue, &buffer, length),
                expected[at..at + length],
                "{NAME}: buffer {index} reads its piece"
            );
            drop(buffer);
            let settled = |counts: &WorkCounts| {
                counts.dma_buf_memories_live == base.dma_buf_memories_live
                    && counts.dma_buf_shared_imports_live == base.dma_buf_shared_imports_live
            };
            counts_once(NAME, &device, "the buffer's release settled", settled);
        }
    };
    let before = device.work_counts();
    let handle = device
        .import_shared_dma_buf(&file, ResourceOptions::default())
        .unwrap();
    let made = device.work_counts();
    assert!(
        format!("{handle:?}").contains("imported: true"),
        "{NAME}: {handle:?}"
    );
    in_turn(&made);
    let kept_through = device.work_counts();
    drop(handle);
    let dropped = device.work_counts();
    let (imports, joined, shared, memories) = moved(&before, &kept_through);
    // The negative control: no handle.
    in_turn(&dropped);
    let unkept = device.work_counts();
    let (control_imports, control_joined, ..) = moved(&dropped, &unkept);
    eprintln!(
        "{TEST}: {NAME} on {}: kept: imports {imports}, joined {joined}, shared imports live \
         +{shared}, dma-buf memories live +{memories}, kept +{}; dropped: shared imports live \
         {:+}; without a handle: imports {control_imports}, joined {control_joined}",
        device.name(),
        kept(&kept_through) - kept(&before),
        moved(&before, &dropped).2,
    );
    assert_eq!(
        (imports, joined, shared, memories),
        (1, BUFFERS as u64, 1, 1),
        "{NAME}: the handle's one import, every buffer joined, the import live past each"
    );
    assert_eq!(kept(&kept_through), kept(&before) + 1, "{NAME}: kept");
    assert_eq!(
        (moved(&before, &dropped).2, moved(&before, &dropped).3),
        (0, 0),
        "{NAME}: freed as the handle dropped"
    );
    assert_eq!(kept(&dropped), kept(&before), "{NAME}: kept no more");
    assert_eq!(
        (control_imports, control_joined),
        (BUFFERS as u64, 0),
        "{NAME}: without a handle, an import each"
    );
}

/// A handle made over a buffer's live import keeps it past the buffer; dropped under a second
/// buffer, it leaves the import to that one, freed with it.
#[test]
fn a_handle_keeps_a_live_import_and_leaves_it_to_its_buffer() {
    const NAME: &str = "a_handle_keeps_a_live_import_and_leaves_it_to_its_buffer";
    let Some((_scratch, device)) = importing(NAME) else {
        return;
    };
    let page = host_page();
    let (stride, length) = (16 * page, 2 * page);
    let len = 2 * stride;
    let ram = Ram::new(len, 0, len);
    let expected = counting(len, 11);
    ram.write_file(0, &expected);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let first = no_copy(&device, &ram, (0, length), Arc::clone(&source)).unwrap();
    let handle = device
        .import_shared_dma_buf(&file, ResourceOptions::default())
        .unwrap();
    let made = device.work_counts();
    drop(first);
    let second: Buffer = no_copy(&device, &ram, (stride, length), Arc::clone(&source)).unwrap();
    let joined = device.work_counts();
    drop(handle);
    let left = device.work_counts();
    eprintln!(
        "{TEST}: {NAME} on {}: a buffer and a handle: imports {}, kept +{}; the buffer gone and \
         another made: imports {}, joined {}; the handle gone: shared imports live {:+}",
        device.name(),
        moved(&before, &made).0,
        kept(&made) - kept(&before),
        moved(&before, &joined).0,
        moved(&made, &joined).1,
        moved(&before, &left).2,
    );
    assert_eq!(
        (moved(&before, &made).0, kept(&made) - kept(&before)),
        (1, 1),
        "{NAME}: the handle keeps the buffer's import, none more"
    );
    assert_eq!(
        (moved(&before, &joined).0, moved(&made, &joined).1),
        (1, 1),
        "{NAME}: the import outlives its buffer; the next joins it"
    );
    assert_eq!(
        (moved(&before, &left).2, kept(&left)),
        (1, kept(&before)),
        "{NAME}: the handle gone, the import left to the buffer"
    );
    assert_eq!(
        read(&queue, &second, length),
        expected[stride..stride + length],
        "{NAME}: the buffer left reads its piece"
    );
    drop(second);
    let freed = |counts: &WorkCounts| {
        counts.dma_buf_shared_imports_live == before.dma_buf_shared_imports_live
            && counts.dma_buf_memories_live == before.dma_buf_memories_live
    };
    counts_once(NAME, &device, "the import freed with the buffer", freed);
}

/// A handle across the device's recovery: with no buffer, nothing imported as the device
/// recovers, the next buffer's import kept past it; with a buffer live, the carried buffer's
/// import kept.
#[test]
fn a_recovery_keeps_the_file_and_imports_it_at_the_next_buffer() {
    const NAME: &str = "a_recovery_keeps_the_file_and_imports_it_at_the_next_buffer";
    let Some((_scratch, device)) = importing(NAME) else {
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
    let base = device.work_counts();
    let handle = device
        .import_shared_dma_buf(&file, ResourceOptions::default())
        .unwrap();
    device.recover_for_test().unwrap();
    // The lost device's import goes once nothing records there; nothing is imported anew.
    let settled = |counts: &WorkCounts| {
        counts.dma_buf_shared_imports_live == base.dma_buf_shared_imports_live
            && counts.dma_buf_shared_imports_kept == base.dma_buf_shared_imports_kept
    };
    let recovered = counts_once(NAME, &device, "the lost device's import freed", settled);
    assert!(
        format!("{handle:?}").contains("imported: false"),
        "{NAME}: nothing imported as the device recovers: {handle:?}"
    );
    let queue = device.new_command_queue().unwrap();
    let buffer = no_copy(&device, &ram, (0, length), Arc::clone(&source)).unwrap();
    assert_eq!(
        read(&queue, &buffer, length),
        expected[..length],
        "{NAME}: the buffer after the recovery reads its piece"
    );
    drop(buffer);
    let lazily = device.work_counts();
    assert!(
        format!("{handle:?}").contains("imported: true"),
        "{NAME}: {handle:?}"
    );
    // A buffer live at a second recovery: carried, its import on the new device kept.
    let live = no_copy(&device, &ram, (stride, length), Arc::clone(&source)).unwrap();
    device.recover_for_test().unwrap();
    let carried = |counts: &WorkCounts| {
        counts.dma_buf_shared_imports_live == base.dma_buf_shared_imports_live + 1
            && counts.dma_buf_shared_imports_kept == base.dma_buf_shared_imports_kept + 1
    };
    let again = counts_once(NAME, &device, "the carried buffer's import kept", carried);
    drop(live);
    let past = device.work_counts();
    eprintln!(
        "{TEST}: {NAME} on {}: recovered with no buffer: shared imports live {:+}, kept {:+}; \
         the next buffer gone: imports {}, shared live {:+}, kept {:+}; a buffer carried and \
         gone: shared live {:+}, kept {:+}",
        device.name(),
        moved(&base, &recovered).2,
        kept(&recovered) as i64 - kept(&base) as i64,
        moved(&recovered, &lazily).0,
        moved(&base, &lazily).2,
        kept(&lazily) as i64 - kept(&base) as i64,
        moved(&base, &past).2,
        kept(&past) as i64 - kept(&base) as i64,
    );
    assert_eq!(
        (
            moved(&recovered, &lazily).0,
            moved(&base, &lazily).2,
            kept(&lazily) - kept(&base)
        ),
        (1, 1, 1),
        "{NAME}: the next buffer imports the file, kept past it"
    );
    assert_eq!(
        (moved(&base, &again).2, moved(&base, &past).2),
        (1, 1),
        "{NAME}: the carried buffer's import kept past it"
    );
    drop(handle);
    let freed = |counts: &WorkCounts| {
        counts.dma_buf_shared_imports_live == base.dma_buf_shared_imports_live
            && counts.dma_buf_shared_imports_kept == base.dma_buf_shared_imports_kept
    };
    counts_once(NAME, &device, "freed as the handle dropped", freed);
}

/// `Private` storage is refused, as a no-copy buffer's is, nothing imported.
#[test]
fn private_storage_is_refused() {
    const NAME: &str = "private_storage_is_refused";
    let Some((_scratch, device)) = on_device(NAME) else {
        return;
    };
    let page = host_page();
    let ram = Ram::new(4 * page, 0, 4 * page);
    let file = Arc::new(udmabuf(ram.file(), 0, 4 * page as u64).unwrap());
    let before = device.work_counts();
    let made = device.import_shared_dma_buf(&file, ResourceOptions::private());
    let after = device.work_counts();
    eprintln!(
        "{TEST}: {NAME} on {}: {:?}",
        device.name(),
        made.as_ref().err()
    );
    assert!(
        matches!(
            made,
            Err(Error::Invalid(Invalid {
                problem: Problem::StorageMode,
                ..
            }))
        ),
        "{NAME}: {made:?}"
    );
    assert_eq!(
        moved(&before, &after),
        (0, 0, 0, 0),
        "{NAME}: nothing imported"
    );
}
