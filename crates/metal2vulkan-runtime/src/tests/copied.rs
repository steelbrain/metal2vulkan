//! The comparisons that keep a copied host buffer coherent with the caller's pages (`copied.rs`),
//! on plain memory: a commit's copy in writes the bytes the CPU changed alone, and a completion's
//! merge back writes the bytes the GPU changed alone, keeping a byte the CPU changed since, as the
//! byte-at-a-time three-way merge does, with no base until a command buffer may write the copy
//! and with one after, in host memory or in memory standing in for the device's. And the copy's
//! first read of the pages: where the kernel's read fails, pages mapped readable are copied all
//! the same and others refused; on Linux, a mapping of a dma-buf (`VM_PFNMAP`), which the
//! kernel's read refuses, is copied (where `/dev/udmabuf` opens).

use crate::copied::{
    merge_block, read_caller, runs_of, tag, BaseBytes, Copied, CopiedBuffers, CopiesReached,
    FirstRead, Held, NotCopied, JOIN_GAP, READ_BACK_FROM, TAG_BLOCK,
};
use crate::dirty_pages::DirtyTally;
use crate::resource::ResourceKey;
use crate::work_counts::Counters;
use std::cell::Cell;
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::sync::Arc;

/// `bytes` as a pointer the comparison reads or writes.
fn at(bytes: &mut [u8]) -> NonNull<u8> {
    NonNull::new(bytes.as_mut_ptr()).unwrap()
}

/// The copy at `copy` of the `length` bytes at `pages` that `read` first reads, registered in
/// `buffers`: the GPU's writes to it dropped where `drops_writes`.
///
/// # Safety
///
/// As [`FirstRead::of`]'s and [`CopiedBuffers::add`]'s.
pub(super) unsafe fn add_reading(
    buffers: &Arc<CopiedBuffers>,
    pages: NonNull<u8>,
    copy: NonNull<u8>,
    length: usize,
    read: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
    drops_writes: bool,
) -> Result<Copied, NotCopied> {
    // SAFETY: the caller's.
    let first = unsafe { FirstRead::of(pages, length, read, drops_writes) }?;
    // SAFETY: the caller's.
    Ok(unsafe { buffers.add(first, pages, copy) })
}

/// A plain read of the pages, the copy's first.
fn plain(from: NonNull<u8>, into: &mut [u8]) -> bool {
    // SAFETY: the tests' pages are `into.len()` bytes of plain memory.
    unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// A commit whose command buffer uses the copies `used` names and may write those `written`
/// names, their bases' bytes in host memory.
pub(super) fn commit(
    buffers: &CopiedBuffers,
    counts: &Counters,
    used: &CopiesReached,
    written: &CopiesReached,
) {
    let tally = DirtyTally::default();
    let copied_in = buffers.copy_in(counts, &tally, (used, written), Vec::new, &mut |_| None);
    assert_eq!(copied_in, Ok(()));
}

/// A commit whose command buffer may write every copy: each takes its base.
pub(super) fn commit_writing(buffers: &CopiedBuffers, counts: &Counters) {
    commit(
        buffers,
        counts,
        &CopiesReached::all(),
        &CopiesReached::all(),
    );
}

/// A commit whose command buffer reads every copy and writes none.
pub(super) fn commit_reading(buffers: &CopiedBuffers, counts: &Counters) {
    commit(
        buffers,
        counts,
        &CopiesReached::all(),
        &CopiesReached::default(),
    );
}

/// Memory standing in for the device's that a base's bytes lie in ([`BaseBytes::device`]): the
/// test's own, freed as it goes.
struct StandIn(NonNull<[u8]>);

// SAFETY: the memory is this value's own, reached only through the base's pointer.
unsafe impl Send for StandIn {}

/// Its reads back copy the runs on the host, as the device's copy on its staging queue would,
/// where the test lets it ([`READS_BACK`]).
impl Held for StandIn {
    fn read_back(&self, runs: &[Range<usize>]) -> Option<Box<dyn Deref<Target = [u8]>>> {
        if !READS_BACK.with(Cell::get) {
            return None;
        }
        // SAFETY: the stand-in's memory, which the runs lie in.
        let bytes = unsafe { self.0.as_ref() };
        let read: Vec<u8> = (runs.iter())
            .flat_map(|run| bytes[run.clone()].iter().copied())
            .collect();
        Some(Box::new(read))
    }
}

thread_local! {
    /// Whether the stand-ins' reads back are made on this thread ([`StandIn`]'s [`Held`]).
    static READS_BACK: Cell<bool> = const { Cell::new(false) };
}

impl Drop for StandIn {
    fn drop(&mut self) {
        // SAFETY: the box `stand_in` leaked, freed once, here.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// `length` bytes standing in for the device's memory of a base.
fn stand_in(length: usize) -> Option<BaseBytes> {
    let raw = NonNull::from(Box::leak(vec![0u8; length].into_boxed_slice()));
    // SAFETY: `length` bytes the stand-in alone holds, mapped until it goes.
    Some(unsafe { BaseBytes::device(raw.cast(), length, Box::new(StandIn(raw))) })
}

/// A copy in before any command buffer may write the copy compares the pages with the copy
/// itself and writes the bytes that differ, and makes no base; the first commit whose command
/// buffer may write it makes one. Then a GPU write not yet merged back and a CPU write in the
/// same block: the copy in writes the CPU's bytes alone, the GPU's byte in the same word stays,
/// and the merge back writes the GPU's byte alone, keeping the CPU's.
#[test]
fn copied_copy_in_writes_the_bytes_the_cpu_changed_alone() {
    let length = 37;
    let mut pages: Vec<u8> = (0..length as u8).collect();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    pages[3] = 0xA3;
    commit_reading(&buffers, &counts);
    assert_eq!(copy[3], 0xA3, "copied in with no base");
    assert_eq!(
        buffers.base_bytes(),
        (0, 0),
        "no base while the GPU only reads"
    );
    commit_writing(&buffers, &counts);
    let tags = (length.div_ceil(TAG_BLOCK) * 16) as u64;
    assert_eq!(
        buffers.base_bytes(),
        (length as u64 + tags, 0),
        "a base in host memory"
    );
    let agreed = pages.clone();
    // The GPU changed byte 1 of the copy, not yet merged back; the CPU changed byte 2, in the
    // same word, and byte 36, in the last, short span.
    copy[1] = 0xEE;
    pages[2] = 0xAA;
    pages[36] = 0xBB;
    commit_reading(&buffers, &counts);
    assert_eq!(copy[1], 0xEE, "the GPU's byte in the same word stays");
    assert_eq!((copy[2], copy[36]), (0xAA, 0xBB), "the CPU's bytes are in");
    let counted = counts.copied_tallies();
    assert_eq!(counted.copy_in_bytes, 1 + 2, "byte 3, then bytes 2 and 36");
    assert_eq!(
        counted.base_blocks_read, 1,
        "both changed the block: its base read"
    );
    buffers.merge_back(&counts, &CopiesReached::all());
    assert_eq!(pages[1], 0xEE, "the GPU's byte is back");
    assert_eq!((pages[2], pages[36]), (0xAA, 0xBB), "the CPU's bytes stay");
    assert_eq!(copy, pages, "they agree");
    assert_ne!(agreed, pages);
    drop(copied);
    assert_eq!(buffers.base_bytes(), (0, 0), "the base went with the copy");
}

/// A merge back writes the bytes the GPU changed alone, keeping a byte the CPU changed since (one
/// both changed keeps the CPU's), and the next commit carries the CPU's bytes into the copy.
#[test]
fn copied_merge_back_keeps_a_byte_the_cpu_changed_since() {
    let length = 16;
    let mut pages: Vec<u8> = (0..length as u8).collect();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    commit_writing(&buffers, &counts);
    // The GPU wrote bytes 3, 4 and 9; the CPU wrote byte 4 too, and byte 5, since the commit.
    copy[3] = 0x33;
    copy[4] = 0x44;
    copy[9] = 0x99;
    pages[4] = 0xC4;
    pages[5] = 0xC5;
    buffers.merge_back(&counts, &CopiesReached::all());
    assert_eq!(counts.copied_tallies().merge_back_bytes, 2, "bytes 3 and 9");
    assert_eq!(
        (pages[3], pages[9]),
        (0x33, 0x99),
        "the GPU's bytes are back"
    );
    assert_eq!((pages[4], pages[5]), (0xC4, 0xC5), "the CPU's bytes stay");
    // The next copy in carries the CPU's byte 4 (and 5) into the copy.
    commit_reading(&buffers, &counts);
    assert_eq!((copy[4], copy[5]), (0xC4, 0xC5));
    assert_eq!(copy, pages);
    drop(copied);
}

/// A GPU that rewrote every byte of a block: where the CPU did not change it since the commit,
/// the merge back writes every byte, the pages' block its agreed bytes (no base read); where the
/// CPU wrote bytes of it before the completion (one alone, one inside a run of 16 and a run of 16
/// whole), the merge keeps each of them and writes every other, the count leaving them out, the
/// base's bytes read; the next commit carries them into the copy.
#[test]
fn a_cpu_write_made_before_the_completion_survives_a_block_the_gpu_rewrote() {
    let length = 3 * TAG_BLOCK + 5;
    let mut pages: Vec<u8> = (0..length as u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    commit_writing(&buffers, &counts);
    for byte in copy.iter_mut() {
        *byte = !*byte;
    }
    let gpu = copy.clone();
    // The CPU writes block 1 alone: byte 1030, byte 1060 and bytes 1088 to 1103.
    let mut cpu = vec![];
    for at in [1030, 1060].into_iter().chain(1088..1104) {
        pages[at] = gpu[at].wrapping_add(1);
        cpu.push((at, pages[at]));
    }
    buffers.merge_back(&counts, &CopiesReached::all());
    for &(at, byte) in &cpu {
        assert_eq!(pages[at], byte, "the CPU's byte {at} kept");
    }
    for at in (0..length).filter(|at| cpu.iter().all(|(cpu, _)| cpu != at)) {
        assert_eq!(pages[at], gpu[at], "the GPU's byte {at} merged");
    }
    let counted = counts.copied_tallies();
    assert_eq!(counted.merge_back_bytes, (length - cpu.len()) as u64);
    assert_eq!(
        counted.base_blocks_read, 1,
        "block 1 alone merged byte by byte"
    );
    commit_reading(&buffers, &counts);
    assert_eq!(copy, pages, "the CPU's bytes copied in");
    drop(copied);
}

/// A merge of 16 bytes writes each byte one side changed and the other did not, and no other byte:
/// for each of the 65,536 sets of bytes changed, every byte outside the set holds a value of the
/// other side's, which a wider store would have written over. On an x86-64 host both with the
/// host's masked stores (AVX-512BW, where it has them) and without.
#[test]
fn a_merge_stores_each_byte_it_writes_and_no_other() {
    for masked in [true, false] {
        #[cfg(target_arch = "x86_64")]
        crate::copied::x86::MASKED_STORES_OFF.store(!masked, std::sync::atomic::Ordering::Relaxed);
        for changed in 0..=0xFFFF_u32 {
            let set = |byte: usize| (changed >> byte) & 1 == 1;
            let now: Vec<u8> = (0..16).map(|byte| u8::from(set(byte))).collect();
            let mut was = [0u8; 16];
            let mut to: Vec<u8> = (0..16)
                .map(|byte| if set(byte) { 0 } else { 0x77 })
                .collect();
            // SAFETY: `to` is the test's 16 bytes.
            let written = unsafe { merge_block(&now, to.as_mut_ptr(), &mut was, true) };
            let named = format!("masked {masked}: {changed:016b}");
            assert_eq!(written, u64::from(changed.count_ones()), "{named}");
            for (byte, &got) in to.iter().enumerate() {
                let want = if set(byte) { 1 } else { 0x77 };
                assert_eq!(got, want, "{named}: byte {byte}");
            }
            assert_eq!(&was[..], &now[..], "{named}: `was` takes `now`");
        }
    }
}

/// A block of a base in the device's memory that both sides changed is read there once, and kept
/// in host memory from then on: the comparisons after it read and write the host's block, so they
/// answer exactly though the device's bytes of the block are overwritten; it counts a host block
/// and its bytes, which go with the copy.
#[test]
fn a_device_base_s_block_both_sides_changed_is_kept_on_the_host() {
    let length = 3 * TAG_BLOCK + 5;
    let mut pages: Vec<u8> = (0..length as u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    let device = Cell::new(None);
    let all = CopiesReached::all();
    let tally = DirtyTally::default();
    let made = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut |length| {
        let base = stand_in(length)?;
        device.set(Some(base.first()));
        Some(base)
    });
    assert_eq!(made, Ok(()));
    let device = device.get().expect("a base in the stand-in's memory");
    let tags = (length.div_ceil(TAG_BLOCK) * 16) as u64;
    assert_eq!(buffers.base_bytes(), (tags, length as u64));
    // Both sides change block 1.
    copy[1030] ^= 0xFF;
    pages[1031] ^= 0xFF;
    buffers.merge_back(&counts, &all);
    assert_eq!(pages[1030], copy[1030], "the GPU's byte merged");
    assert_eq!(counts.copied_tallies().base_blocks_read, 1);
    assert_eq!(buffers.base_host_blocks(), 1, "block 1 kept on the host");
    assert_eq!(
        buffers.base_bytes(),
        (tags + TAG_BLOCK as u64, length as u64)
    );
    // The device's bytes of block 1 overwritten: nothing reads them now.
    // SAFETY: block 1 of the stand-in's bytes, which the base holds.
    unsafe { std::ptr::write_bytes(device.as_ptr().add(TAG_BLOCK), 0xEE, TAG_BLOCK) };
    commit_reading(&buffers, &counts);
    assert_eq!(copy, pages, "the CPU's byte copied in");
    // Both sides change block 1 again: the merge reads the host's block.
    copy[1040] ^= 0xFF;
    pages[1041] ^= 0xFF;
    let cpu = pages[1041];
    buffers.merge_back(&counts, &all);
    assert_eq!(pages[1040], copy[1040], "the GPU's byte merged again");
    assert_eq!(pages[1041], cpu, "the CPU's byte kept");
    assert_eq!(counts.copied_tallies().base_blocks_read, 2);
    assert_eq!(buffers.base_host_blocks(), 1, "no block kept again");
    commit_reading(&buffers, &counts);
    assert_eq!(copy, pages);
    drop(copied);
    assert_eq!(buffers.base_bytes(), (0, 0));
    assert_eq!(buffers.base_host_blocks(), 0);
}

/// A read back copies runs of blocks, each joined with the next where at most [`JOIN_GAP`] blocks
/// lie between them, and finds each block at its bytes' place among the runs copied one after
/// another; a copy's last block is as long as it is.
#[test]
fn a_read_back_joins_runs_a_few_blocks_apart() {
    let far = 4 + JOIN_GAP;
    let last = far + 1 + JOIN_GAP + 1;
    let length = last * TAG_BLOCK + 5;
    let (runs, at) = runs_of(&[0, 1, 3, far, last], length);
    assert_eq!(runs, [0..(far + 1) * TAG_BLOCK, last * TAG_BLOCK..length]);
    let starts = [0, 1, 3, far].map(|block| block * TAG_BLOCK);
    assert_eq!(at[..4], starts);
    assert_eq!(at[4], (far + 1) * TAG_BLOCK);
}

/// A comparison that would read [`READ_BACK_FROM`] blocks or more of a base in the device's
/// memory leaves them for a second pass, which takes them from one read back (the stand-in's,
/// made on the host as the device's staging copy would be), and fewer each from the device's
/// bytes, as it does where the read back is not made; either way a merge back (one that keeps
/// the GPU's writes, and one that drops them) and the copy in after it answer exactly, each such
/// block read once and kept on the host.
#[test]
fn a_comparison_of_many_device_blocks_reads_them_back_together() {
    let blocks = [1, 2, 4, 8];
    // (Its switch, whether reads back are made, the blocks read back.)
    let cases = [(3, true, 4), (5, true, 0), (3, false, 0)];
    for drops in [false, true] {
        for (from, reads_back, read_back) in cases {
            READ_BACK_FROM.store(from, std::sync::atomic::Ordering::Relaxed);
            READS_BACK.with(|on| on.set(reads_back));
            let named = format!("drops {drops}, from {from}, reads back {reads_back}");
            let length = 8 * TAG_BLOCK + 5;
            let mut pages: Vec<u8> = (0..length as u32).map(|i| (i * 13 + 1) as u8).collect();
            let original = pages.clone();
            let mut copy = vec![0_u8; length];
            let buffers = Arc::new(CopiedBuffers::default());
            let counts = Counters::default();
            // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the
            // answer.
            let copied = unsafe {
                add_reading(
                    &buffers,
                    at(&mut pages),
                    at(&mut copy),
                    length,
                    plain,
                    drops,
                )
            };
            let copied = copied.unwrap();
            let all = CopiesReached::all();
            let tally = DirtyTally::default();
            let made = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut stand_in);
            assert_eq!(made, Ok(()), "{named}");
            // Both sides change each of `blocks` (8 the last, 5 bytes long), and the GPU block 6.
            for block in blocks {
                copy[block * TAG_BLOCK + 3] ^= 0xFF;
                pages[block * TAG_BLOCK + 4] ^= 0xFF;
            }
            copy[6 * TAG_BLOCK] ^= 0xFF;
            let (gpu, cpu) = (copy.clone(), pages.clone());
            buffers.merge_back(&counts, &all);
            let counted = counts.copied_tallies();
            assert_eq!(counted.base_blocks_read, 4, "{named}");
            assert_eq!(counted.base_blocks_read_back, read_back, "{named}");
            assert_eq!(
                buffers.base_host_blocks(),
                4,
                "{named}: each kept on the host"
            );
            match drops {
                false => {
                    let mut merged = gpu.clone();
                    for block in blocks {
                        let at = block * TAG_BLOCK + 4;
                        merged[at] = cpu[at];
                    }
                    assert_eq!(
                        pages, merged,
                        "{named}: the GPU's bytes merged, the CPU's kept"
                    );
                }
                true => {
                    assert_eq!(pages, cpu, "{named}: nothing written to the pages");
                    assert_eq!(
                        copy, original,
                        "{named}: the GPU's bytes taken out of the copy"
                    );
                }
            }
            commit_reading(&buffers, &counts);
            assert_eq!(copy, pages, "{named}: the CPU's bytes copied in");
            drop(copied);
            assert_eq!(buffers.base_host_blocks(), 0, "{named}");
        }
    }
    READ_BACK_FROM.store(0, std::sync::atomic::Ordering::Relaxed);
    READS_BACK.with(|on| on.set(false));
}

/// A copy whose GPU writes are dropped takes the GPU's bytes out of the copy alone, where the CPU
/// changed a byte of the same block (its base's bytes read), and the CPU's byte reaches the copy
/// at the next commit.
#[test]
fn copied_restore_takes_the_gpu_s_bytes_out_of_the_copy_alone() {
    let length = 21;
    let mut pages: Vec<u8> = (0..length as u8).collect();
    let original = pages.clone();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied =
        unsafe { add_reading(&buffers, at(&mut pages), at(&mut copy), length, plain, true) };
    let copied = copied.unwrap();
    commit_writing(&buffers, &counts);
    // The GPU wrote bytes 2, 3 and 20 (the last, short span); the CPU wrote byte 5.
    copy[2] = 0x22;
    copy[3] = 0x33;
    copy[20] = 0x44;
    pages[5] = 0x55;
    buffers.merge_back(&counts, &CopiesReached::all());
    let counted = counts.copied_tallies();
    assert_eq!(counted.dropped_bytes, 3);
    assert_eq!(counted.base_blocks_read, 1, "both changed the block");
    assert_eq!(copy, original, "the copy holds the agreed bytes again");
    assert_eq!(pages[2], 2, "nothing written to the pages");
    commit_reading(&buffers, &counts);
    assert_eq!(copy[5], 0x55, "the CPU's byte copied in");
    drop(copied);
}

/// A copy of pages the process cannot write drops the GPU's writes: a completion takes them out
/// of the copy and writes nothing to the pages; a CPU write to the pages is copied in at the next
/// commit as for any copy. A copy of writable pages merges them back.
#[test]
fn copied_merge_back_of_unwritable_pages_drops_the_gpu_s_writes() {
    let length = 4099;
    let mut pages: Vec<u8> = (0..length as u32).map(|i| (i * 5 + 1) as u8).collect();
    let original = pages.clone();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied =
        unsafe { add_reading(&buffers, at(&mut pages), at(&mut copy), length, plain, true) };
    let copied = copied.unwrap();
    assert!(copied.drops_writes());
    commit_writing(&buffers, &counts);
    // The GPU writes 100 bytes of the copy; the completion drops them.
    copy[1000..1100].fill(0xEE);
    buffers.merge_back(&counts, &CopiesReached::all());
    assert_eq!(pages, original, "nothing written to the pages");
    assert_eq!(copy, original, "the copy holds the pages' bytes again");
    // A CPU write to the pages reaches the copy at the next commit.
    pages[7] = !pages[7];
    commit_reading(&buffers, &counts);
    assert_eq!(copy[7], pages[7]);
    drop(copied);
    // A copy of writable pages merges the GPU's writes back.
    let mut writable = original.clone();
    let mut other = vec![0_u8; length];
    // SAFETY: as above.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut writable),
            at(&mut other),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    assert!(!copied.drops_writes());
    commit_writing(&buffers, &counts);
    other[0] = !other[0];
    buffers.merge_back(&counts, &CopiesReached::all());
    assert_eq!(writable[0], other[0], "merged back");
    drop(copied);
}

/// The byte-at-a-time three-way merge: each byte of `from` that differs from `snapshot` written
/// into `to` (where `keep_to`, unless `to`'s byte differs from `snapshot` already) and taken
/// into `snapshot`. The copies' comparisons must answer exactly as it does. The bytes written.
fn reference_sync(from: &[u8], to: &mut [u8], snapshot: &mut [u8], keep_to: bool) -> u64 {
    let mut written = 0;
    for byte in 0..snapshot.len() {
        if from[byte] == snapshot[byte] {
            continue;
        }
        if !keep_to || to[byte] == snapshot[byte] {
            to[byte] = from[byte];
            written += 1;
        }
        snapshot[byte] = from[byte];
    }
    written
}

/// The byte-at-a-time drop of the GPU's writes: each byte of `copy` that differs from
/// `snapshot` takes the snapshot's back. The bytes written.
fn reference_restore(copy: &mut [u8], snapshot: &[u8]) -> u64 {
    let mut written = 0;
    for (copy, agreed) in copy.iter_mut().zip(snapshot) {
        if copy != agreed {
            *copy = *agreed;
            written += 1;
        }
    }
    written
}

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Random writes to `side`: single bytes, at times a run that may cross blocks; and where `both`,
/// at times a byte `other` writes too, to the same value or another.
fn change(rng: &mut Rng, side: &mut [u8], other: &mut [u8], both: bool) {
    let length = side.len();
    for _ in 0..rng.below(12) {
        let at = rng.below(length);
        side[at] = side[at].wrapping_add(1 + rng.below(255) as u8);
    }
    if rng.below(3) == 0 {
        let start = rng.below(length);
        let end = (start + rng.below(9000)).min(length);
        for byte in &mut side[start..end] {
            *byte = rng.next() as u8;
        }
    }
    if both {
        for _ in 0..rng.below(3) {
            let at = rng.below(length);
            let value = side[at].wrapping_add(1);
            side[at] = value;
            other[at] = match rng.below(2) {
                0 => value,
                _ => value.wrapping_add(1),
            };
        }
    }
}

/// The copies' comparisons answer as the byte-at-a-time three-way merge does, over lengths that
/// end inside a block, inside an 8-byte span and on either edge: first while the GPU only reads
/// (no base: CPU writes and commits), then with a base in host memory, or in memory standing in
/// for the device's, through random steps of CPU writes, GPU writes, commits and completions in
/// any order (a commit while the GPU's writes are not merged back, as with a command buffer in
/// flight; two completions in a row). Bytes changed on one side, on the other, and on both (the
/// same byte to the same and to different values), alone, in runs crossing a block's edge and
/// over whole blocks; copies that keep the GPU's writes and copies that drop them.
#[test]
fn copied_comparisons_answer_as_byte_by_byte() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let lengths = [
        1,
        7,
        8,
        9,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        8191,
        3 * 4096 + 5,
        64 * 1024 + 3,
    ];
    let (all, none) = (CopiesReached::all(), CopiesReached::default());
    let tally = DirtyTally::default();
    let mut read = 0;
    for round in 0..156 {
        let length = lengths[round % lengths.len()];
        let on_device = round % 2 == 1;
        let drops = round % 3 == 2;
        let mut pages: Vec<u8> = (0..length).map(|_| rng.next() as u8).collect();
        let mut copy = vec![0_u8; length];
        let buffers = Arc::new(CopiedBuffers::default());
        let counts = Counters::default();
        // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
        let copied = unsafe {
            add_reading(
                &buffers,
                at(&mut pages),
                at(&mut copy),
                length,
                plain,
                drops,
            )
        };
        let copied = copied.unwrap();
        let mut snapshot = pages.clone();
        let mut want_copy = pages.clone();
        let mut base = |length| match on_device {
            true => stand_in(length),
            false => None,
        };
        // While the GPU only reads: CPU writes and commits, no base.
        for step in 0..3 {
            change(&mut rng, &mut pages, &mut copy, false);
            let wrote = reference_sync(&pages, &mut want_copy, &mut snapshot, false);
            let before = counts.copied_tallies().copy_in_bytes;
            let copied_in = buffers.copy_in(&counts, &tally, (&all, &none), Vec::new, &mut base);
            assert_eq!(copied_in, Ok(()));
            let got = counts.copied_tallies().copy_in_bytes - before;
            assert_eq!(
                got, wrote,
                "round {round} step {step}: copied in with no base"
            );
            assert!(
                copy == want_copy,
                "round {round} step {step}: the copy with no base"
            );
        }
        assert_eq!(buffers.base_bytes(), (0, 0), "round {round}: no base yet");
        // A command buffer that may write it: its base.
        let wrote = reference_sync(&pages, &mut want_copy, &mut snapshot, false);
        let copied_in = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut base);
        assert_eq!(copied_in, Ok(()));
        assert_eq!(wrote, 0);
        let tags = (length.div_ceil(TAG_BLOCK) * 16) as u64;
        let want = match on_device {
            true => (tags, length as u64),
            false => (tags + length as u64, 0),
        };
        assert_eq!(
            buffers.base_bytes(),
            want,
            "round {round}: the base's bytes"
        );
        assert_eq!(counts.copied_tallies().bases_made, 1);
        let mut want_pages = pages.clone();
        for step in 0..24 {
            match rng.below(4) {
                // The CPU's writes, and the GPU's: both sides see them.
                0 | 1 => {
                    match step % 2 {
                        0 => change(&mut rng, &mut pages, &mut copy, true),
                        _ => change(&mut rng, &mut copy, &mut pages, true),
                    }
                    want_pages.copy_from_slice(&pages);
                    want_copy.copy_from_slice(&copy);
                }
                2 => {
                    let wrote = reference_sync(&want_pages, &mut want_copy, &mut snapshot, false);
                    let before = counts.copied_tallies().copy_in_bytes;
                    let copied_in =
                        buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut base);
                    assert_eq!(copied_in, Ok(()));
                    let got = counts.copied_tallies().copy_in_bytes - before;
                    assert_eq!(got, wrote, "round {round} step {step}: bytes copied in");
                }
                _ => {
                    let before = counts.copied_tallies();
                    buffers.merge_back(&counts, &all);
                    let after = counts.copied_tallies();
                    let (merged, dropped) = (
                        after.merge_back_bytes - before.merge_back_bytes,
                        after.dropped_bytes - before.dropped_bytes,
                    );
                    match drops {
                        true => {
                            let wrote = reference_restore(&mut want_copy, &snapshot);
                            assert_eq!((merged, dropped), (0, wrote), "round {round} step {step}");
                        }
                        false => {
                            let gpu = want_copy.clone();
                            let wrote = reference_sync(&gpu, &mut want_pages, &mut snapshot, true);
                            assert_eq!((merged, dropped), (wrote, 0), "round {round} step {step}");
                        }
                    }
                }
            }
            assert!(pages == want_pages, "round {round} step {step}: the pages");
            assert!(copy == want_copy, "round {round} step {step}: the copy");
        }
        read += counts.copied_tallies().base_blocks_read;
        drop(copied);
        assert_eq!(buffers.base_bytes(), (0, 0), "round {round}: the base went");
        assert_eq!(
            buffers.base_host_blocks(),
            0,
            "round {round}: its host blocks too"
        );
    }
    assert!(read > 0, "some blocks both sides changed");
}

/// A block's tag changes with any one bit of the block flipped (8192 of them, under the
/// process's key), with the block's last word alone changed, and with a short block's bytes
/// (zero past its end) changed; equal blocks have equal tags.
#[test]
fn a_tag_changes_with_any_bit_of_its_block() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut block = [0u8; TAG_BLOCK];
    for byte in &mut block {
        *byte = rng.next() as u8;
    }
    let tagged = tag(&block);
    assert_eq!(tag(&block.clone()), tagged);
    for bit in 0..TAG_BLOCK * 8 {
        let mut flipped = block;
        flipped[bit / 8] ^= 1 << (bit % 8);
        assert_ne!(tag(&flipped), tagged, "bit {bit}");
    }
    let mut zero = [0u8; TAG_BLOCK];
    let zero_tag = tag(&zero);
    zero[TAG_BLOCK - 4..].copy_from_slice(&1u32.to_le_bytes());
    assert_ne!(tag(&zero), zero_tag, "the last word counts");
}

/// The SSE2 form of a block's tag (on x86-64) answers as the portable form does, over random
/// blocks, a block of all ones (the most every sum can carry) and one of zeros.
#[test]
fn a_tag_is_the_same_on_every_path() {
    use crate::copied::tag_portable;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut blocks = vec![[0u8; TAG_BLOCK], [0xFF; TAG_BLOCK]];
    for _ in 0..256 {
        let mut block = [0u8; TAG_BLOCK];
        for byte in &mut block {
            *byte = rng.next() as u8;
        }
        blocks.push(block);
    }
    for block in &blocks {
        assert_eq!(tag(block), tag_portable(block));
    }
}

/// A copied 256 MiB buffer holds nothing beyond its copy while the GPU only reads it; once a
/// command buffer may write it, its base's bytes in memory standing in for the device's leave
/// its tags alone in host memory: a 64th of its length, within `length / 16` and far under the
/// copy's length a base in host memory takes (as before, every copy's length again).
#[test]
fn a_copied_256_mib_buffer_holds_its_tags_alone_beyond_its_copy() {
    const LENGTH: usize = 256 << 20;
    let mut pages = vec![0_u8; LENGTH];
    let mut copy = vec![0_u8; LENGTH];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `LENGTH` bytes, live past the answer.
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
    pages[12345] = 7;
    commit_reading(&buffers, &counts);
    assert_eq!(copy[12345], 7);
    assert_eq!(
        buffers.base_bytes(),
        (0, 0),
        "nothing while the GPU only reads"
    );
    let (all, tally) = (CopiesReached::all(), DirtyTally::default());
    let copied_in = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut stand_in);
    assert_eq!(copied_in, Ok(()));
    let (host, device) = buffers.base_bytes();
    assert_eq!(host, (LENGTH / 64) as u64, "its tags: 4 MiB");
    assert!(host <= (LENGTH / 16) as u64);
    assert_eq!(device, LENGTH as u64, "its bytes in the device's memory");
    // A GPU write merges back through it.
    copy[LENGTH - 1] = 9;
    buffers.merge_back(&counts, &all);
    assert_eq!(pages[LENGTH - 1], 9);
    drop(copied);
    assert_eq!(buffers.base_bytes(), (0, 0));
}

/// A commit that finds no memory for a copy's base fails, the copy left with none and still
/// holding the bytes its pages and it agree on (the CPU's write copied in); the next commit that
/// may write it makes the base, and the GPU's writes merge back through it.
#[test]
fn a_copy_with_no_memory_for_its_base_keeps_none() {
    let length = 3 * TAG_BLOCK + 5;
    let mut pages: Vec<u8> = (0..length).map(|i| (i * 7) as u8).collect();
    let mut copy = vec![0_u8; length];
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            plain,
            false,
        )
    };
    let copied = copied.unwrap();
    pages[100] ^= 0xFF;
    buffers.refuse_bases(true);
    let (all, tally) = (CopiesReached::all(), DirtyTally::default());
    let copied_in = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut |_| None);
    assert_eq!(copied_in, Err(NotCopied::OutOfHostMemory));
    assert_eq!(counts.copied_tallies().bases_refused, 1);
    assert_eq!(buffers.base_bytes(), (0, 0), "no base");
    assert_eq!(copy, pages, "the copy holds the agreed bytes");
    buffers.refuse_bases(false);
    pages[200] ^= 0xFF;
    commit_writing(&buffers, &counts);
    assert_eq!(counts.copied_tallies().bases_made, 1);
    assert_eq!(copy, pages);
    copy[300] ^= 0xFF;
    buffers.merge_back(&counts, &all);
    assert_eq!(copy, pages, "merged back");
    drop(copied);
}

/// Where the kernel's read of the pages fails (a mapping it cannot pin), pages the process maps
/// readable are read with plain loads and copied; where a page is not mapped readable, the copy
/// is refused and nothing written; where the kernel reads them, no mapping is asked about.
#[test]
fn copied_pages_the_kernel_cannot_read_are_copied_where_mapped_readable() {
    let mut pages: Vec<u8> = (0..4099_u32).map(|i| (i * 7 + 3) as u8).collect();
    let length = pages.len();
    let start = pages.as_ptr() as u64;
    let buffers = Arc::new(CopiedBuffers::default());
    let asked = Cell::new(None);
    let mut copy = vec![0_u8; length];
    let read_readable = |from: NonNull<u8>, into: &mut [u8]| {
        let readable = |start, end| {
            asked.set(Some((start, end)));
            true
        };
        // SAFETY: plain memory, readable.
        unsafe { read_caller(from, into, |_, _| false, readable) }
    };
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut copy),
            length,
            read_readable,
            false,
        )
    };
    assert!(copied.is_ok(), "copied");
    assert_eq!(copy, pages, "the copy holds the pages' bytes");
    assert_eq!(asked.get(), Some((start, start + length as u64)), "asked");
    let mut refused_copy = vec![0_u8; length];
    let read_unreadable = |from: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: nothing is read.
        unsafe { read_caller(from, into, |_, _| false, |_, _| false) }
    };
    // SAFETY: as above; nothing reads the pages.
    let refused = unsafe {
        add_reading(
            &buffers,
            at(&mut pages),
            at(&mut refused_copy),
            length,
            read_unreadable,
            false,
        )
    };
    assert_eq!(refused.err(), Some(NotCopied::Unreadable));
    assert!(refused_copy.iter().all(|&byte| byte == 0), "nothing copied");
    let mut into = vec![0_u8; length];
    asked.set(None);
    let kernel = |from: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the pages are `into.len()` bytes of plain memory.
        unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    // SAFETY: as above.
    let read = unsafe {
        read_caller(at(&mut pages), &mut into, kernel, |start, end| {
            asked.set(Some((start, end)));
            false
        })
    };
    assert!(read);
    assert_eq!(
        (into, asked.get()),
        (pages, None),
        "the kernel's read alone"
    );
}

/// A copy of pages of a dma-buf mapping (udmabuf's, `VM_PFNMAP`), which the kernel's read cannot
/// pin and fails: the copy is made, and holds their bytes. The test says which read took them
/// (`--no-capture`). Not run where `/dev/udmabuf` does not open, which the test says.
#[cfg(target_os = "linux")]
#[test]
fn copied_pages_of_a_dma_buf_mapping_are_copied() {
    use crate::copied::read_pages;
    // SAFETY: `sysconf` reads a system constant.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
    let length = 16 * page;
    let pattern: Vec<u8> = (0..length).map(|i| (i % 251) as u8 ^ 0x5A).collect();
    let Some(mapping) = DmaBufMapping::new(&pattern) else {
        return;
    };
    // The path the test is for: the kernel cannot read the mapping's pages.
    let mut probe = vec![0_u8; page];
    match read_pages(mapping.pages, &mut probe) {
        true => println!("the kernel reads this udmabuf mapping: copied through the kernel"),
        false => println!("the kernel cannot read this udmabuf mapping: copied with plain loads"),
    }
    let buffers = Arc::new(CopiedBuffers::default());
    let mut copy = vec![0_u8; length];
    // SAFETY: the copy is the test's; the mapping is `length` bytes, readable, live past the
    // answer.
    let first = match unsafe { CopiedBuffers::read(mapping.pages, length, false) } {
        Ok(first) => first,
        Err(not) => panic!("not copied: {not:?}"),
    };
    // SAFETY: as above.
    let _copied = unsafe { buffers.add(first, mapping.pages, at(&mut copy)) };
    assert!(copy == pattern, "the copy holds the dma-buf's bytes");
}

/// A shared mapping of a dma-buf (`/dev/udmabuf`) of a `memfd`'s pages, as a consumer maps guest
/// RAM: `VM_PFNMAP`, which the kernel cannot pin. Unmapped when dropped.
#[cfg(target_os = "linux")]
pub(super) struct DmaBufMapping {
    /// The mapping's first byte.
    pub(super) pages: NonNull<u8>,
    length: usize,
    _dma_buf: std::os::fd::OwnedFd,
    _memfd: std::os::fd::OwnedFd,
}

#[cfg(target_os = "linux")]
impl DmaBufMapping {
    /// A mapping of a dma-buf of a new `memfd` holding `bytes` (whole pages), or `None` where
    /// `/dev/udmabuf` does not open, which it says.
    pub(super) fn new(bytes: &[u8]) -> Option<DmaBufMapping> {
        use std::io::Write;
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        /// `struct udmabuf_create` of Linux's `include/uapi/linux/udmabuf.h`.
        #[repr(C)]
        struct Create {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        /// `UDMABUF_CREATE`, `_IOW('u', 0x42, struct udmabuf_create)` in the generic ioctl
        /// encoding.
        const UDMABUF_CREATE: u64 =
            (1 << 30) | ((size_of::<Create>() as u64) << 16) | ((b'u' as u64) << 8) | 0x42;
        /// `UDMABUF_FLAGS_CLOEXEC`.
        const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;
        let device = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf");
        let device = match device {
            Ok(device) => device,
            Err(error) => {
                println!("/dev/udmabuf cannot be opened ({error}): not run");
                return None;
            }
        };
        let length = bytes.len();
        // A memfd sealed against shrinking, as udmabuf requires, holding the bytes.
        // SAFETY: a NUL-terminated name.
        let memfd = unsafe {
            libc::memfd_create(
                c"m2v-copied-udmabuf".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(memfd >= 0, "memfd: {}", std::io::Error::last_os_error());
        // SAFETY: a new descriptor, owned at once.
        let memfd = unsafe { OwnedFd::from_raw_fd(memfd) };
        std::fs::File::from(memfd.try_clone().unwrap())
            .write_all(bytes)
            .unwrap();
        // SAFETY: seals the memfd made above.
        let sealed =
            unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) };
        assert_eq!(sealed, 0, "seal: {}", std::io::Error::last_os_error());
        let create = Create {
            memfd: memfd.as_raw_fd() as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: length as u64,
        };
        // SAFETY: the ioctl reads `create`, a valid `udmabuf_create`.
        let dma_buf = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE as _, &create) };
        assert!(dma_buf >= 0, "udmabuf: {}", std::io::Error::last_os_error());
        // SAFETY: a new descriptor the ioctl returned, owned at once.
        let dma_buf = unsafe { OwnedFd::from_raw_fd(dma_buf) };
        // SAFETY: a new shared mapping of the dma-buf, unmapped by the answer's drop.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dma_buf.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            mapped,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        Some(DmaBufMapping {
            pages: NonNull::new(mapped.cast()).unwrap(),
            length,
            _dma_buf: dma_buf,
            _memfd: memfd,
        })
    }
}

#[cfg(all(target_os = "linux", any(m2v_gpu, m2v_cpu_icd)))]
impl DmaBufMapping {
    /// A source of the mapping's own dma-buf, for a request of all of its pages: a clone of the
    /// descriptor, which the runtime takes. A request of other pages is refused.
    pub(super) fn source(&self) -> crate::DmaBufSource {
        let dma_buf = std::sync::Arc::new(self._dma_buf.try_clone().unwrap());
        let (start, length) = (self.pages.as_ptr() as usize, self.length as u64);
        std::sync::Arc::new(move |request: crate::DmaBufRequest| {
            if request.start.as_ptr() as usize != start || request.length != length {
                return Err(std::io::Error::other("not the mapping's pages"));
            }
            Ok(vec![crate::DmaBufChunk {
                fd: dma_buf.try_clone()?,
                length,
            }])
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for DmaBufMapping {
    fn drop(&mut self) {
        // SAFETY: the mapping made by `new`, which nothing uses now.
        let unmapped = unsafe { libc::munmap(self.pages.as_ptr().cast(), self.length) };
        assert_eq!(unmapped, 0);
    }
}

/// A merge back compares only the copies its batches may have written: those whose keys a use
/// wrote, or every one where a command may have written memory no use names (`CopiesReached::all`),
/// and every one after a batch finished with no merge (`CopiedBuffers::owe`). A copy no batch
/// wrote keeps the GPU's change in the copy, unmerged, until a merge that takes it as written.
#[test]
fn a_merge_back_compares_only_the_copies_its_batches_wrote() {
    let length = 4 * 4096;
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    let read = |pages: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the test's pages, `into.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    let (mut pages_x, mut pages_y) = (vec![1_u8; length], vec![2_u8; length]);
    let (mut copy_x, mut copy_y) = (vec![0_u8; length], vec![0_u8; length]);
    // SAFETY: the copies and the pages are the test's, of `length` bytes, live past the answers.
    let x = unsafe {
        add_reading(
            &buffers,
            at(&mut pages_x),
            at(&mut copy_x),
            length,
            read,
            false,
        )
    };
    // SAFETY: as above.
    let y = unsafe {
        add_reading(
            &buffers,
            at(&mut pages_y),
            at(&mut copy_y),
            length,
            read,
            false,
        )
    };
    let (x, y) = (x.unwrap(), y.unwrap());
    let (key_x, key_y) = (ResourceKey::new(), ResourceKey::new());
    x.name(key_x);
    y.name(key_y);
    let named = |keys: &[ResourceKey]| {
        let mut written = CopiesReached::default();
        for key in keys {
            written.reached(*key);
        }
        written
    };
    // Committed command buffers may write both: each takes its base.
    commit(
        &buffers,
        &counts,
        &named(&[key_x, key_y]),
        &named(&[key_x, key_y]),
    );
    // The GPU writes both copies; the batch's uses wrote X alone.
    copy_x[100] = 0xAA;
    copy_y[100] = 0xBB;
    buffers.merge_back(&counts, &named(&[key_x]));
    assert_eq!(pages_x[100], 0xAA, "X merged back");
    assert_eq!(pages_y[100], 2, "Y not compared");
    buffers.merge_back(&counts, &named(&[key_y]));
    assert_eq!(pages_y[100], 0xBB, "Y merged back once a batch wrote it");
    // A command that may write memory no use names: every copy.
    copy_y[200] = 0xCC;
    buffers.merge_back(&counts, &CopiesReached::all());
    assert_eq!(pages_y[200], 0xCC, "every copy compared");
    // A batch finished with no merge: the next compares every copy, once.
    copy_y[300] = 0xDD;
    buffers.owe();
    buffers.merge_back(&counts, &CopiesReached::default());
    assert_eq!(
        pages_y[300], 0xDD,
        "every copy compared after a batch finished unmerged"
    );
    copy_y[400] = 0xEE;
    buffers.merge_back(&counts, &CopiesReached::default());
    assert_eq!(pages_y[400], 2, "and then none again");
    drop((x, y));
}

/// A commit's copy in takes the CPU's writes into the copies its command buffer's commands use
/// alone (`CopiesReached`), every copy where one may reach memory no use names: a copy left alone
/// keeps what it agreed on, and the first commit that uses it takes every write since.
#[test]
fn a_commit_copies_in_only_the_copies_its_commands_use() {
    let length = 4 * 4096;
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    let read = |pages: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the test's pages, `into.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    let (mut pages_x, mut pages_y) = (vec![1_u8; length], vec![2_u8; length]);
    let (mut copy_x, mut copy_y) = (vec![0_u8; length], vec![0_u8; length]);
    // SAFETY: the copies and the pages are the test's, of `length` bytes, live past the answers.
    let x = unsafe {
        add_reading(
            &buffers,
            at(&mut pages_x),
            at(&mut copy_x),
            length,
            read,
            false,
        )
    };
    // SAFETY: as above.
    let y = unsafe {
        add_reading(
            &buffers,
            at(&mut pages_y),
            at(&mut copy_y),
            length,
            read,
            false,
        )
    };
    let (x, y) = (x.unwrap(), y.unwrap());
    let (key_x, key_y) = (ResourceKey::new(), ResourceKey::new());
    x.name(key_x);
    y.name(key_y);
    let used = |keys: &[ResourceKey]| {
        let mut used = CopiesReached::default();
        for key in keys {
            used.reached(*key);
        }
        used
    };
    let none = CopiesReached::default();
    // The CPU writes both pages; the command buffer uses X alone.
    pages_x[100] = 0xAA;
    pages_y[100] = 0xBB;
    commit(&buffers, &counts, &used(&[key_x]), &none);
    assert_eq!(copy_x[100], 0xAA, "X copied in");
    assert_eq!(copy_y[100], 2, "Y left alone");
    // Another write to Y, then a command buffer that uses it: both writes are in.
    pages_y[200] = 0xCC;
    commit(&buffers, &counts, &used(&[key_y]), &none);
    assert_eq!(
        (copy_y[100], copy_y[200]),
        (0xBB, 0xCC),
        "Y takes every write since"
    );
    // A command that may reach memory no use names: every copy.
    pages_x[300] = 0xDD;
    commit(&buffers, &counts, &CopiesReached::all(), &none);
    assert_eq!(copy_x[300], 0xDD, "every copy copied in");
    drop((x, y));
}

/// A copy's keys are those of the pages lying in it now: each pages' naming of a key counts, and
/// goes as those pages go (`Copied::unname`), so a copy that buffers come and go in (a new buffer
/// over a surface's pages at each present, `host_aliases.rs`) holds no more keys than lie in it,
/// and a merge back compares it for a write through a key named still, not one gone.
#[test]
fn a_copy_s_keys_are_those_of_the_pages_lying_in_it() {
    let length = 4096;
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    let read = |pages: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the test's pages, `into.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    let (mut pages, mut copy) = (vec![1_u8; length], vec![0_u8; length]);
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied =
        unsafe { add_reading(&buffers, at(&mut pages), at(&mut copy), length, read, false) };
    let copied = copied.unwrap();
    let first = ResourceKey::new();
    copied.name(first);
    commit_writing(&buffers, &counts);
    // A thousand presents, each a buffer lying in the copy and gone.
    for _ in 0..1000 {
        let frame = ResourceKey::new();
        copied.name(frame);
        copied.unname(frame);
    }
    assert_eq!(copied.keys(), 1, "only the first buffer's key is left");
    // A key two pages named stays until both went.
    let rebound = ResourceKey::new();
    copied.name(rebound);
    copied.name(rebound);
    copied.unname(rebound);
    let mut written = CopiesReached::default();
    written.reached(rebound);
    copy[10] = 0xAB;
    buffers.merge_back(&counts, &written);
    assert_eq!(pages[10], 0xAB, "a key named still is compared");
    copied.unname(rebound);
    copy[20] = 0xCD;
    buffers.merge_back(&counts, &written);
    assert_eq!(pages[20], 1, "a key gone is not");
    assert_eq!(copied.keys(), 1);
    drop(copied);
}
