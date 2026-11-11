//! Copied host buffers: a no-copy buffer whose import of the caller's pages the caller's gate
//! refused ([`HostMemory::with_import_gate`](crate::HostMemory)), whose pointer the device refused
//! with no dma-buf of the pages serving it (no source of them given, or one that failed or whose
//! dma-bufs the device refused; RADV refuses shared memory, a `memfd` mapping among it, and a
//! device without `VK_EXT_external_memory_host` every pointer), or whose import the gate and the
//! device admitted and that then failed (out of memory, another Vulkan error, or the runtime's
//! refusal of pages the kernel cannot pin on a driver whose failed import of them breaks it,
//! `RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`, that no dma-buf of them served), is made
//! all the same over a copy of its bytes (`Device::buffer_over_host_memory`), announced
//! (`copied-host-pages`). No import maps the pages, so they take no kernel mapping.
//!
//! - **The copy.** A `VkBuffer` of the buffer's length in host-visible, coherent memory of the
//!   runtime's own ([`MemoryClass::Shared`](crate::memory::MemoryClass)), which the GPU uses; the
//!   buffer's `contents` stays the caller's pointer, as Metal's is. Made over pages the process
//!   reads ([`read_caller`]): the first copy reads them through the kernel ([`read_pages`]), and
//!   where the kernel fails the read, with plain loads where the process maps every page with
//!   read permission ([`mapped_readable`]): on Linux the kernel reads no page it cannot pin, such
//!   as a mapping of a dma-buf (`VM_PFNMAP`). Pages not mapped, or mapped without read permission
//!   (memory Metal makes no buffer over), keep the import's own refusal.
//! - **One memory, one copy.** A no-copy buffer made over exactly the bytes (the same first byte
//!   and length) a live copied buffer was made over lies in its copy, of either storage mode, as
//!   one made over a live import's bytes lies in that import (`host_aliases.rs`): one `VkBuffer`,
//!   one base, one hazard key, the gate not asked and no page read. So two textures over one
//!   surface's pages are ordered as one resource's accesses are, with no trip through the pages.
//!   A rebind to such bytes lies in the copy too where its textures were planned at the copy's
//!   offset (its first byte) and no image in place rides on it. Counted:
//!   `WorkCounts::copied_host_buffers_shared`. A heap's own buffer and a buffer whose GPU writes
//!   are dropped keep their own (`host_aliases.rs`, "Not joined").
//! - **Coherence.** The runtime keeps, for each copy, the bytes the pages and the copy last agreed
//!   on. At each commit, before any of the command buffer's work is submitted, every byte the
//!   CPU changed in the pages since then is written into each copy its commands use, or into
//!   every copy where one may reach memory no use names (a function that reaches buffers through
//!   addresses, work on the GPU alone; [`CopiedBuffers::copy_in`]): a copy the command buffer does
//!   not use takes the CPU's writes at the first commit that uses it. At each batch's completion,
//!   before its events are seen signaled, the scheduler is told, its handlers run or a waiter
//!   returns, every byte the GPU changed in the copy is written back into the pages, where the CPU
//!   did not change that byte since ([`CopiedBuffers::merge_back`]): a byte both changed keeps the
//!   CPU's, which the next commit copies in. The completion compares only the copies its batches
//!   may have written: one whose buffer's (or heap's) key a use of their command buffers'
//!   commands wrote, and every copy where one of those may have written memory no use names (the
//!   loaded addresses of a function that may store or update memory atomically, an indirect
//!   command buffer's executions; `CopiesReached`), or kept its writes not all apart. Counted:
//!   `WorkCounts::copied_merges_compared`, `copied_merges_skipped`. Inside a block that differs,
//!   bytes are compared 8 at a time and then one at a time, so a merge writes no byte the GPU did
//!   not change and a copy in none the CPU did not (`merge_block`). Each batch ends with a barrier
//!   that makes its writes visible to the host (`tracker::Barrier::to_host`). Where the pages'
//!   memory came with a log of the writes to them (`HostMemory::with_dirty_pages`, or its guest
//!   RAM's, `HostRam::with_dirty_pages`), each merge back reports the blocks of 4 KiB it wrote to
//!   the log as the copy's own writes, in one call (`DirtyPages::wrote_runs_as`, a writer of the
//!   copy's own) before it returns: the log sees no write the runtime makes, and every other copy
//!   and shadow of the pages takes them.
//! - **The copy is the base until the GPU may write it.** Only copies in write a copy until a
//!   command buffer that may write it is committed, so until then it holds the bytes its pages
//!   and it agree on, and a copy in compares the pages with the copy itself, a page (4 KiB) at a
//!   time, and writes in the bytes that differ. A copy the GPU only reads (no use of a committed
//!   command buffer wrote one of its keys, and none may have written memory no use names) holds
//!   nothing beyond itself, and no completion compares it.
//! - **A base once it may.** At the first commit whose command buffer may write the copy (its
//!   commands wrote one of its keys, or may write memory no use names), after its copy in and
//!   before any of its work is submitted, the copy takes a base of the bytes it then holds
//!   ([`Base`]): a 128-bit tag of each 1 KiB block ([`tag`]: two NH sums keyed by a random key of
//!   the process's, `TAG_BLOCK`), kept in host memory (a 64th of the copy's length), and the
//!   bytes themselves, kept in the device's memory where it has room for them, else in the
//!   host's (`Device::copied_base`): device-local memory the host maps, in a heap the device's
//!   resizable BAR makes as large as its own memory (not the 256 MiB window of a GPU without it,
//!   nor any device with unified memory, where that memory is the host's), while the runtime's
//!   blocks in that heap and the base stay within half of it. From then on each comparison reads
//!   one side and its tags: a commit's copy in reads the pages' blocks, and only a block whose
//!   tag changed reads the copy's block; a completion's merge back reads the copy's blocks, and
//!   only a block whose tag changed reads the pages' block. The base's bytes are the other
//!   side's block where its tag still agrees, and are read from the base only for a block both
//!   sides changed since they agreed (a read across the bus the host does not cache, where they
//!   are in the device's memory). The base then takes each block that changed. A block whose tag
//!   agrees is taken as unchanged: for a block that changed, that happens with a chance of at
//!   most 2^-64 at each comparison ([`tag`]), the only place the runtime takes a tag for the
//!   bytes; a copy without a base compares bytes. Where the host has no memory for a base, the
//!   commit fails with `Error::OutOfMemory` and runs nothing, as Metal fails a command buffer
//!   for memory, and the copy keeps none (it holds what its pages and it agree on). Counted:
//!   `WorkCounts::copied_bases_made`, `copied_bases_refused`, `copied_base_blocks_read`, and the
//!   gauges `copied_base_host_bytes` and `copied_base_device_bytes`.
//! - **Copies in the log gates.** Where the pages' memory came with such a log (a copy that drops
//!   the GPU's writes included: its guest RAM's runs are asked as a writable copy's are), a
//!   commit's copy in compares only the pages the log reports written since the copy's last copy
//!   in, and the
//!   bytes no log keeps (a guest RAM's runs that miss some of them): a stamp of each log is asked
//!   at the commit, once for every copy, its publication waited for (bounded) with no lock of
//!   the copies' or the command buffer's held, and the copy synced to it
//!   ([`CopiedBuffers::copy_in`]); every write made before the commit is in that publication,
//!   the GPU's in place through other imports of the pages and the runtime's own CPU writes among
//!   them, which the runtime reports (`dirty_pages.rs`). A copy compares every byte at its first
//!   copy in after its watch began, and where a log cannot say. Counted:
//!   `WorkCounts::copied_copy_in_bytes_skipped`.
//! - **Pages the process cannot write** (read-only pages, as a consumer's alias of a read-only
//!   replace, `replaceBackingWithRanges:readOnly:`, maps them), asked of the system as the copy is
//!   made ([`mapped_writable`]), and memory of Metal's private option bit 24 where the system maps
//!   no read-only view of the pages (`read_only_view.rs`, copied whatever the gate would answer):
//!   the GPU's writes to the copy are dropped, as Metal drops the GPU's writes to read-only
//!   memory. At each batch's completion every byte the
//!   GPU changed in the copy takes back the byte the pages and the copy last agreed on
//!   ([`CopiedBuffers::merge_back`]), and nothing is written to the pages: later commands of the
//!   same batch read the write back, and a later command buffer the pages' bytes. The CPU's writes
//!   to the pages through another mapping are copied in at each commit, as above. Counted:
//!   `WorkCounts::copied_dropped_bytes`.
//! - **Between copies, GPU to GPU** (`docs/FAST-PATHS.md`, known gaps). A copy and another copy, or
//!   an import, of pages that overlap without being the same bytes (a rebind's bytes inside a
//!   surface's, a guest RAM chunk's import over a copy's pages, a buffer whose texture moved in
//!   its page, the bytes again after a reported remap), and a copy beside a live import of the same
//!   bytes for the other storage mode's class (a `Managed` buffer's import, then a `Shared`
//!   buffer's copy where the gate refused its own), are two memories, which agree only through
//!   the pages: at the writer's completion (its merge back, or its writes in place) and at the
//!   reader's commit (its copy in). A read committed before the write completed reads the bytes
//!   as of its commit, wherever the tracker orders it after the write: in one command buffer, in
//!   one queue's command buffers in flight together, and across queues an event orders, as the
//!   held reader is not copied in again when the event releases it. Of two such copies written in
//!   flight, the second's merge back takes a byte the first merged for the CPU's write, which a
//!   merge keeps, so the first write stays where Metal keeps the later.
//! - **Event signals.** A batch's event signals, submitted while a copy is live, are seen at its
//!   completion turn alone, after the merge: no read of the value observes them earlier, and a
//!   wait for them is held on the host until that turn (`event_timeline.rs`), so the GPU's writes
//!   before a signal are in the pages once its value is seen, as Metal's are. No submission is
//!   added; a reader sees the value a completion turn later than the semaphore's counter shows it.
//! - **Cost.** A copy is compared whole at each batch's completion that may have written it, and
//!   at each commit where its pages carry no log: O(bytes) per such batch, and per commit without
//!   a log, a fallback's cost (`docs/FAST-PATHS.md`, known gaps). A command buffer whose
//!   functions read addresses may write every copy. A page that agrees costs one read of each
//!   side and a slice comparison: 680 MiB of copies that agree took 28-30 ms each way on an M3
//!   Max, 52-53 ms on a Core Ultra 9 275HX; their logged copy in took 0.15 ms (0.3) where no page
//!   changed, and 4.6 ms (3.5) where 1% of the pages changed on each side; a merge back of no
//!   copy written, none (`tests/copied_bench.rs`). Counted:
//!   `WorkCounts::copied_copy_in_bytes`, `WorkCounts::copied_merge_back_bytes`,
//!   `WorkCounts::copied_dropped_bytes`.
//! - **Heaps and rebinds.** A no-copy heap's own buffer is copied as a buffer is, where the heap's
//!   textures could lie, with a heap's hazard identity; its buffers lie in the copy at their
//!   offsets, and its textures are shadows of its bytes (`AliasRoute::MemoryShadow` or Metal's
//!   tiles). A buffer's or heap's rebind gives it a copy of its new pages laid out as their import
//!   would be (`Device::copied_pages`), so texel views planned at the bytes' offset lie where they
//!   lay, and at the old offset where the bytes moved in their page under a texture; the heap's
//!   textures bound to its bytes and the images in place over the buffer are made again over the
//!   copy: in fresh pages the runtime imported where an image in place rides on it, made as over
//!   the caller's pages; else linear images of its memory that start `PREINITIALIZED`.
//! - **What is not copied.** On macOS, memory whose GPU writes Metal's private option bit 24 drops
//!   imports a read-only view of its pages (`read_only_view.rs`).
//! - **Textures over a copy.** A texture made over a copied buffer is a linear image over the copy,
//!   bound in its memory at the texels' offset, where the device's in-place check passed for the
//!   caller's pages (a rebind may import them; and the device's store path, which renders and
//!   writes into the copy take too) and the copy's memory, for the texture's accesses and how its
//!   texels lie in lines there (`in_place.rs`, `texture/route.rs`; counted:
//!   `WorkCounts::copied_textures_in_place`): in fresh pages the runtime imported, the image made
//!   for host allocations, as over the caller's pages, under the fresh pages' check; in memory the
//!   device allocates, the driver's own linear image that starts `PREINITIALIZED`, so its first
//!   move keeps the copy's bytes, else a `DRM_FORMAT_MOD_LINEAR` one at Metal's pitch, made
//!   `UNDEFINED`, under the check of the runtime's pooled memory too (`PagesClass::Pooled`). Its
//!   hazard identity is the buffer's, so its uses name the copy as the buffer's do: each commit
//!   copies the CPU's writes in, and each completion that may have written it merges the GPU's
//!   back. Elsewhere it is a shadow of the copy's bytes (`shadowed-texture`).

use crate::device::lock;
use crate::dirty_pages::{new_writer, DirtySource, DirtyTally, PagesLog};
use crate::host_memory::{mapped_readable, mapped_writable};
use crate::key_hash::KeyMap;
use crate::resource::ResourceKey;
use crate::work_counts::Counters;
use crate::DirtyStamp;
use std::hash::BuildHasher;
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A device's live copied host buffers, compared at each commit and each batch's completion.
#[derive(Default)]
pub(crate) struct CopiedBuffers {
    entries: Mutex<Vec<Entry>>,
    /// How many are live: a commit or completion looks no further where none is.
    live: AtomicUsize,
    /// The next copy's id.
    next: AtomicU64,
    /// A batch finished with no merge back (a failed wait): the next merge back compares every
    /// copy, whatever its batches wrote.
    owed: AtomicBool,
    /// The host bytes the live copies' bases hold ([`Base`]): their tags, and the agreed bytes
    /// of those kept in host memory. A gauge (`WorkCounts::copied_base_host_bytes`).
    base_host_bytes: AtomicU64,
    /// The device bytes the live copies' bases hold: the agreed bytes of those kept in the
    /// device's memory. A gauge (`WorkCounts::copied_base_device_bytes`).
    base_device_bytes: AtomicU64,
    /// The tests' stand-in for a host with no memory for a base ([`CopiedBuffers::refuse_bases`]).
    #[cfg(test)]
    bases_refused: AtomicBool,
}

/// The copies commands reached, by the keys of their uses, or every copy: what the batches a
/// completion turn finishes may have written ([`CopiedBuffers::merge_back`], the keys their
/// command buffers' commands wrote, `DirtyCommit::copies_written`), or what a committed command
/// buffer's commands may read or write ([`CopiedBuffers::copy_in`], the keys they used,
/// `DirtyCommit::copies_used`).
#[derive(Default)]
pub(crate) struct CopiesReached {
    keys: KeyMap<ResourceKey, ()>,
    all: bool,
}

impl CopiesReached {
    /// Every copy may have been reached.
    pub(crate) fn all() -> CopiesReached {
        CopiesReached {
            keys: KeyMap::default(),
            all: true,
        }
    }

    /// A command reached the bytes of `key`.
    pub(crate) fn reached(&mut self, key: ResourceKey) {
        if !self.all {
            self.keys.insert(key, ());
        }
    }

    /// Some command may have reached a copy no use of it names.
    pub(crate) fn reached_all(&mut self) {
        self.all = true;
        self.keys.clear();
    }

    /// Whether a command may have reached the copy whose uses take `keys`.
    fn meets(&self, keys: &[(ResourceKey, usize)]) -> bool {
        self.all || keys.iter().any(|(key, _)| self.keys.contains_key(key))
    }
}

/// The caller's bytes as a copy's first read of them took them ([`CopiedBuffers::read`]), before
/// the copy's memory was made: the bytes the copy and the pages first agree on, written into the
/// copy and let go of as it is registered ([`CopiedBuffers::add`]), and whether the GPU's writes
/// to the copy are dropped.
pub(crate) struct FirstRead {
    bytes: Box<[u8]>,
    drops_writes: bool,
}

impl FirstRead {
    /// `length` bytes at `pages`, read by `read`, which answers whether it read every one.
    ///
    /// # Safety
    ///
    /// `read` reads `pages` as [`CopiedBuffers::read`]'s contract lets it.
    pub(crate) unsafe fn of(
        pages: NonNull<u8>,
        length: usize,
        read: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
        drops_writes: bool,
    ) -> Result<FirstRead, NotCopied> {
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(length).is_err() {
            return Err(NotCopied::OutOfHostMemory);
        }
        bytes.resize(length, 0u8);
        let mut bytes = bytes.into_boxed_slice();
        if !read(pages, &mut bytes) {
            return Err(NotCopied::Unreadable);
        }
        Ok(FirstRead {
            bytes,
            drops_writes,
        })
    }
}

/// Why a copy was not made ([`CopiedBuffers::read`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NotCopied {
    /// A page is not mapped, or mapped without read permission: memory Metal makes no buffer over.
    Unreadable,
    /// The host has no memory for the copy's first read of the pages, or for its base
    /// ([`CopiedBuffers::copy_in`]).
    OutOfHostMemory,
}

/// A copy and the caller's pages it mirrors.
struct Entry {
    id: u64,
    /// The caller's bytes (the buffer's `contents`).
    pages: NonNull<u8>,
    /// The copy's bytes, mapped.
    copy: NonNull<u8>,
    /// How many bytes the pages and the copy are.
    length: usize,
    /// The bytes the pages and the copy last agreed on, once a commit whose command buffer may
    /// write the copy made it ([`CopiedBuffers::copy_in`]). Until then the copy holds them: only
    /// copies in write it.
    base: Option<Base>,
    /// The process cannot write the pages: the GPU's writes to the copy are dropped, never merged
    /// back.
    drops_writes: bool,
    /// The logs of the writes to the pages, where their memory came with one, and the writer its
    /// merges back are reported as ([`Copied::set_log`]).
    log: Option<CopyLog>,
    /// The keys whose uses' writes reach the copy, each with how many pages lying in it named it
    /// ([`Copied::name`]): the buffer's, or the heap's bytes' (whose buffers' and textures' uses
    /// carry it), each buffer's that lies in it besides (`host_aliases.rs`), a buffer's again after
    /// a rebind. A key goes with the last pages that named it ([`Copied::unname`]), so the keys are
    /// those of the buffers lying in the copy now, however many came and went.
    keys: Vec<(ResourceKey, usize)>,
}

/// A copy's logs of its pages, and its writer in them.
struct CopyLog {
    pages: Arc<PagesLog>,
    writer: NonZeroU64,
    /// Its watches of the pages in each log, held while the copy lives.
    _watches: Vec<Box<dyn Send + Sync>>,
    /// The copy's bytes no log keeps the writes to: every copy in compares them.
    unlogged: Arc<[Range<u64>]>,
    /// The stamps, one of each log in [`PagesLog::sources`]' order, as of which the copy holds
    /// every byte the CPU wrote in the pages: a copy in compares only the pages the logs report
    /// written since. `None` until a copy in compared every byte after the watches began, and
    /// after one some log could not answer for.
    since: Option<Arc<[DirtyStamp]>>,
    /// The bytes its merges back wrote into the pages since `since`: at [`ASKS_APART`] or more,
    /// its copy in asks the logs for the pages any writer but it wrote
    /// ([`DirtyPages::changed_except`](crate::DirtyPages)), where a copy otherwise takes the
    /// pages every writer wrote, asked once per log and stamp for every copy, and compares again
    /// the pages it merged.
    merged: u64,
}

/// The bytes a copy's merges back wrote since its last copy in from which its copy in asks the
/// logs apart, for every writer's pages but its own (one more question of each log, which a
/// consumer answers by a pass over its watches), rather than compare its own writes again
/// (about 40 µs a MiB on an M3 Max). Both are exact.
const ASKS_APART: u64 = 1 << 20;

/// The bytes of a block a copy's base tags and compares at once ([`Base`]).
pub(crate) const TAG_BLOCK: usize = 1024;

/// A block's tag ([`tag`]).
type Tag = [u64; 2];

/// The words of the key of [`tag`]: one for each word of a block, and four more for its second
/// sum's key, the first's shifted by four words.
const KEY_WORDS: usize = TAG_BLOCK / 4 + 4;

/// The key of every [`tag`] in the process: random words, made once, which no caller sees.
fn key() -> &'static [u32; KEY_WORDS] {
    static KEY: OnceLock<[u32; KEY_WORDS]> = OnceLock::new();
    KEY.get_or_init(|| {
        // The standard library's hasher keyed from the system's randomness, as a keyed function
        // of each pair of words' place.
        let random = std::collections::hash_map::RandomState::new();
        std::array::from_fn(|at| (random.hash_one(at / 2) >> (32 * (at % 2))) as u32)
    })
}

/// A block's tag: two NH sums of its bytes (RFC 4418's NH: each 32-byte stretch's little-endian
/// words `m0` to `m7`, `(m_i + k_i) * (m_{i+4} + k_{i+4})` for `i` from 0 to 3, the words and
/// their key words added modulo 2^32, the products summed modulo 2^64), the second's key the
/// first's shifted by four words (RFC 4418's Toeplitz iteration), keyed by the process's random
/// [`key`]. NH of 32-bit words is 2^-32-almost-universal, and its Toeplitz extension to two sums
/// 2^-64 (the UMAC paper, Black, Halevi, Krawczyk, Krovetz and Rogaway, 1999): for any two
/// different blocks, which no key decides, their tags agree with a chance of at most 2^-64. A
/// block a copy's base takes as unchanged may have changed with that chance.
///
///
/// The sums run as SIMD on an x86-64 host ([`x86`]: AVX2 where the host has it, SSE2 otherwise),
/// and one word pair at a time elsewhere ([`tag_portable`]): the same tag either way.
pub(crate) fn tag(block: &[u8; TAG_BLOCK]) -> Tag {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the host has AVX2.
            return unsafe { x86::tag_avx2(block, key()) };
        }
        x86::tag_sse2(block, key())
    }
    #[cfg(not(target_arch = "x86_64"))]
    tag_portable(block)
}

/// [`tag`] one word pair at a time: the host's answer where it has no SIMD form, and the
/// reference the SIMD forms are tested against.
#[cfg(any(test, not(target_arch = "x86_64")))]
pub(crate) fn tag_portable(block: &[u8; TAG_BLOCK]) -> Tag {
    let key = key();
    let (mut first, mut second) = ([0u64; 4], [0u64; 4]);
    for (stretch, bytes) in block.chunks_exact(32).enumerate() {
        let word = |at: usize| {
            u32::from_le_bytes([
                bytes[4 * at],
                bytes[4 * at + 1],
                bytes[4 * at + 2],
                bytes[4 * at + 3],
            ])
        };
        let keys = &key[8 * stretch..8 * stretch + 12];
        let lanes = first.iter_mut().zip(second.iter_mut()).enumerate();
        for (lane, (first, second)) in lanes {
            let (low, high) = (word(lane), word(lane + 4));
            let one = u64::from(low.wrapping_add(keys[lane]))
                * u64::from(high.wrapping_add(keys[lane + 4]));
            let two = u64::from(low.wrapping_add(keys[lane + 4]))
                * u64::from(high.wrapping_add(keys[lane + 8]));
            *first = first.wrapping_add(one);
            *second = second.wrapping_add(two);
        }
    }
    let sum = |lanes: [u64; 4]| lanes.iter().fold(0u64, |sum, &lane| sum.wrapping_add(lane));
    [sum(first), sum(second)]
}

/// [`tag`]'s sums as x86-64 SIMD. A stretch's words `m0` to `m3` sit in one register and `m4` to
/// `m7` in another, each added to its key words; `_mm_mul_epu32` multiplies the even words of
/// the two (`m0` by `m4`, `m2` by `m6`) into 64-bit lanes, and again after a shift by 32 bits
/// the odd ones. Each sum keeps its products in two registers' lanes, added together at the end:
/// the same sums modulo 2^64 as [`tag_portable`]'s, in another order.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::{Tag, KEY_WORDS, TAG_BLOCK};
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_loadu2_m128i,
        _mm256_mul_epu32, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256,
        _mm_add_epi32, _mm_add_epi64, _mm_loadu_si128, _mm_mul_epu32, _mm_setzero_si128,
        _mm_srli_epi64, _mm_storeu_si128,
    };

    /// SSE2, which every x86-64 host has: one 32-byte stretch to a step.
    pub(crate) fn tag_sse2(block: &[u8; TAG_BLOCK], key: &[u32; KEY_WORDS]) -> Tag {
        let words = block.as_ptr().cast::<__m128i>();
        let keys = key.as_ptr().cast::<__m128i>();
        let (mut first, mut second) = ([_mm_setzero_si128(); 2], [_mm_setzero_si128(); 2]);
        for stretch in 0..TAG_BLOCK / 32 {
            // SAFETY: unaligned loads of the block's 16-byte pieces `2 * stretch` and the next
            // (at most its 63rd of 64), and of the key's 4-word pieces `2 * stretch` to two
            // after it (at most its 64th of 65: words 256 to 259).
            let (low, high, low_key, high_key, next_key) = unsafe {
                (
                    _mm_loadu_si128(words.add(2 * stretch)),
                    _mm_loadu_si128(words.add(2 * stretch + 1)),
                    _mm_loadu_si128(keys.add(2 * stretch)),
                    _mm_loadu_si128(keys.add(2 * stretch + 1)),
                    _mm_loadu_si128(keys.add(2 * stretch + 2)),
                )
            };
            let (low_first, high_first) =
                (_mm_add_epi32(low, low_key), _mm_add_epi32(high, high_key));
            let (low_second, high_second) =
                (_mm_add_epi32(low, high_key), _mm_add_epi32(high, next_key));
            first[0] = _mm_add_epi64(first[0], _mm_mul_epu32(low_first, high_first));
            first[1] = _mm_add_epi64(
                first[1],
                _mm_mul_epu32(
                    _mm_srli_epi64::<32>(low_first),
                    _mm_srli_epi64::<32>(high_first),
                ),
            );
            second[0] = _mm_add_epi64(second[0], _mm_mul_epu32(low_second, high_second));
            second[1] = _mm_add_epi64(
                second[1],
                _mm_mul_epu32(
                    _mm_srli_epi64::<32>(low_second),
                    _mm_srli_epi64::<32>(high_second),
                ),
            );
        }
        let sum = |lanes: [__m128i; 2]| {
            let mut words = [0u64; 2];
            // SAFETY: a 16-byte store into two u64s.
            unsafe {
                _mm_storeu_si128(words.as_mut_ptr().cast(), _mm_add_epi64(lanes[0], lanes[1]))
            };
            words[0].wrapping_add(words[1])
        };
        [sum(first), sum(second)]
    }

    /// AVX2: two 32-byte stretches to a step, the words `m0` to `m3` of both in one register and
    /// `m4` to `m7` of both in another.
    ///
    /// # Safety
    ///
    /// The host has AVX2.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn tag_avx2(block: &[u8; TAG_BLOCK], key: &[u32; KEY_WORDS]) -> Tag {
        let words = block.as_ptr().cast::<__m128i>();
        let keys = key.as_ptr().cast::<__m128i>();
        let (mut first, mut second) = ([_mm256_setzero_si256(); 2], [_mm256_setzero_si256(); 2]);
        for stretch in (0..TAG_BLOCK / 32).step_by(2) {
            let at = 2 * stretch;
            // SAFETY: unaligned loads of the block's 16-byte pieces `at` to three after it (at
            // most its 63rd of 64), and of the key's 4-word pieces `at` to four after it (at
            // most its 64th of 65: words 256 to 259); the host has AVX2.
            let (low, high, low_key, high_key, next_key) = unsafe {
                (
                    _mm256_loadu2_m128i(words.add(at + 2), words.add(at)),
                    _mm256_loadu2_m128i(words.add(at + 3), words.add(at + 1)),
                    _mm256_loadu2_m128i(keys.add(at + 2), keys.add(at)),
                    _mm256_loadu2_m128i(keys.add(at + 3), keys.add(at + 1)),
                    _mm256_loadu2_m128i(keys.add(at + 4), keys.add(at + 2)),
                )
            };
            let (low_first, high_first) = (
                _mm256_add_epi32(low, low_key),
                _mm256_add_epi32(high, high_key),
            );
            let (low_second, high_second) = (
                _mm256_add_epi32(low, high_key),
                _mm256_add_epi32(high, next_key),
            );
            first[0] = _mm256_add_epi64(first[0], _mm256_mul_epu32(low_first, high_first));
            first[1] = _mm256_add_epi64(
                first[1],
                _mm256_mul_epu32(
                    _mm256_srli_epi64::<32>(low_first),
                    _mm256_srli_epi64::<32>(high_first),
                ),
            );
            second[0] = _mm256_add_epi64(second[0], _mm256_mul_epu32(low_second, high_second));
            second[1] = _mm256_add_epi64(
                second[1],
                _mm256_mul_epu32(
                    _mm256_srli_epi64::<32>(low_second),
                    _mm256_srli_epi64::<32>(high_second),
                ),
            );
        }
        let mut sums = [[0u64; 4]; 2];
        for (sum, lanes) in sums.iter_mut().zip([first, second]) {
            // SAFETY: a 32-byte store into four u64s; the host has AVX2.
            unsafe {
                _mm256_storeu_si256(
                    sum.as_mut_ptr().cast(),
                    _mm256_add_epi64(lanes[0], lanes[1]),
                )
            };
        }
        sums.map(|lanes| lanes.iter().fold(0u64, |sum, &lane| sum.wrapping_add(lane)))
    }
}

/// The bytes the pages and a copy last agreed on, made at the first commit whose command buffer
/// may write the copy ([`CopiedBuffers::copy_in`]), from the copy, which held them until then:
/// a tag of each [`TAG_BLOCK`] block, which the comparisons read in place of the bytes, and the
/// bytes themselves, which a comparison reads only for a block both sides changed since they
/// agreed, in the device's memory where it has room for them ([`BaseBytes::device`]), else in
/// the host's.
struct Base {
    tags: Box<[Tag]>,
    bytes: BaseBytes,
}

impl Base {
    /// The base of the `length` bytes of the copy at `copy`, which holds the bytes the pages
    /// and it agree on: its bytes kept in `device`, where the device had room for them, else in
    /// the host's memory. `None` where the host has no memory for its tags or bytes.
    ///
    /// # Safety
    ///
    /// `copy` is `length` bytes of mapped memory, and `device` holds as many.
    unsafe fn of_copy(copy: NonNull<u8>, length: usize, device: Option<BaseBytes>) -> Option<Base> {
        let blocks = length.div_ceil(TAG_BLOCK);
        let mut tags = Vec::new();
        tags.try_reserve_exact(blocks).ok()?;
        let mut host = Vec::new();
        if device.is_none() {
            host.try_reserve_exact(length).ok()?;
        }
        let mut block = [0u8; TAG_BLOCK];
        for index in 0..blocks {
            let at = index * TAG_BLOCK;
            let span = (length - at).min(TAG_BLOCK);
            // SAFETY: `span` bytes at `at` lie in the copy (the caller's).
            unsafe { read_block(copy.as_ptr().add(at), span, &mut block) };
            tags.push(tag(&block));
            match &device {
                // SAFETY: as above, in the device's bytes.
                Some(bytes) => unsafe { bytes.write(at, &block[..span]) },
                None => host.extend_from_slice(&block[..span]),
            }
        }
        let bytes = match device {
            Some(bytes) => bytes,
            None => BaseBytes::host(host.into_boxed_slice()),
        };
        Some(Base {
            tags: tags.into_boxed_slice(),
            bytes,
        })
    }

    /// The host's bytes it holds: its tags, and its bytes where they are in the host's memory.
    fn host_bytes(&self) -> u64 {
        let tags = (self.tags.len() * size_of::<Tag>()) as u64;
        match self.bytes.on_device {
            true => tags,
            false => tags + self.bytes.length as u64,
        }
    }

    /// The device's bytes it holds: its bytes where they are in the device's memory.
    fn device_bytes(&self) -> u64 {
        match self.bytes.on_device {
            true => self.bytes.length as u64,
            false => 0,
        }
    }
}

/// Where a copy's base keeps the agreed bytes ([`Base`]): memory the host maps, the host's own,
/// or the device's (`Device::copied_base`), held while the base lives.
pub(crate) struct BaseBytes {
    at: NonNull<u8>,
    length: usize,
    on_device: bool,
    _held: Box<dyn Send>,
}

// SAFETY: `at` names the memory `_held` keeps mapped and nothing else uses, reached only under
// the copies' lock ([`CopiedBuffers`]).
unsafe impl Send for BaseBytes {}

impl BaseBytes {
    /// The host's memory of `bytes`.
    fn host(bytes: Box<[u8]>) -> BaseBytes {
        let length = bytes.len();
        let raw = NonNull::from(Box::leak(bytes));
        BaseBytes {
            at: raw.cast(),
            length,
            on_device: false,
            _held: Box::new(HostBaseBytes(raw)),
        }
    }

    /// `length` bytes of the device's memory at `at`, which the host maps, held by `held`.
    ///
    /// # Safety
    ///
    /// `at` is `length` bytes of memory the host reads and writes (coherent, needing no flush),
    /// which nothing else uses and which stays mapped until `held` goes.
    pub(crate) unsafe fn device(at: NonNull<u8>, length: usize, held: Box<dyn Send>) -> BaseBytes {
        BaseBytes {
            at,
            length,
            on_device: true,
            _held: held,
        }
    }

    /// Read `into.len()` of the bytes from `offset` into `into`. Of the device's memory, a read
    /// across the bus that the host does not cache: slow (`memory::host_cached`), and made only
    /// for a block both sides changed.
    ///
    /// # Safety
    ///
    /// `offset..offset + into.len()` lies in the bytes.
    unsafe fn read(&self, offset: usize, into: &mut [u8]) {
        // SAFETY: the caller's; `into` is the caller's own, apart.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.at.as_ptr().add(offset),
                into.as_mut_ptr(),
                into.len(),
            )
        };
    }

    /// Write `from` into the bytes from `offset`.
    ///
    /// # Safety
    ///
    /// `offset..offset + from.len()` lies in the bytes.
    unsafe fn write(&self, offset: usize, from: &[u8]) {
        // SAFETY: the caller's; `from` is the caller's own, apart.
        unsafe {
            std::ptr::copy_nonoverlapping(from.as_ptr(), self.at.as_ptr().add(offset), from.len())
        };
    }
}

/// Host memory of a base's bytes, freed as it goes: reached through [`BaseBytes`]' pointer alone.
struct HostBaseBytes(NonNull<[u8]>);

// SAFETY: the memory is this value's own, reached only through its base's pointer.
unsafe impl Send for HostBaseBytes {}

impl Drop for HostBaseBytes {
    fn drop(&mut self) {
        // SAFETY: the box `BaseBytes::host` leaked, freed once, here.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// Read `span` bytes at `from` into `into`, the rest of it zero: a block the tags read whole.
///
/// # Safety
///
/// `from` is `span` bytes of mapped memory, `span` at most [`TAG_BLOCK`].
unsafe fn read_block(from: *const u8, span: usize, into: &mut [u8; TAG_BLOCK]) {
    // SAFETY: the caller's; `into` is the caller's own, apart.
    unsafe { std::ptr::copy_nonoverlapping(from, into.as_mut_ptr(), span) };
    into[span..].fill(0);
}

// SAFETY: the pointers name the caller's pages, which stay mapped while the buffer's import lives
// (`HostMemory::new`), and the copy's mapped memory, which the import holds; both are reached only
// under the registry's lock, which the import's drop takes before either goes ([`Copied`]).
unsafe impl Send for Entry {}

/// A live copy's place in its device's [`CopiedBuffers`]: dropped (unregistered) before the copy's
/// memory is freed, so no comparison reads it after (`HostImport`'s drop).
pub(crate) struct Copied {
    buffers: Arc<CopiedBuffers>,
    id: u64,
    drops_writes: bool,
}

impl Copied {
    /// Whether the GPU's writes to the copy are dropped: the process cannot write the pages.
    pub(crate) fn drops_writes(&self) -> bool {
        self.drops_writes
    }

    /// Keep `log`, the logs of the writes to the copy's pages (`dirty_pages::PagesLog`), with a
    /// writer of its own: each merge back reports the bytes it wrote to them as that writer's
    /// ([`CopiedBuffers::merge_back`]). Set once.
    ///
    /// Its pages are watched in each log from here on ([`DirtyPages::watch`](crate::DirtyPages)),
    /// asked with no lock of the copies' held: its copies in compare every byte until one did
    /// after the watches began ([`CopiedBuffers::copy_in`]).
    pub(crate) fn set_log(&self, log: &Arc<PagesLog>) {
        let Some(writer) = new_writer() else {
            return;
        };
        let watches = log.watch();
        let unlogged: Arc<[Range<u64>]> = log.unlogged().into();
        let mut entries = lock(&self.buffers.entries);
        let entry = entries.iter_mut().find(|entry| entry.id == self.id);
        let unkept = match entry {
            Some(entry) if entry.log.is_none() => {
                entry.log = Some(CopyLog {
                    pages: Arc::clone(log),
                    writer,
                    _watches: watches,
                    unlogged,
                    since: None,
                    merged: 0,
                });
                None
            }
            _ => Some(watches),
        };
        // A watch's end is the consumer's call: made with no lock of the copies' held.
        drop(entries);
        drop(unkept);
    }

    /// The uses of `key` (a buffer's or a heap's bytes' lying in the copy) write the copy: a
    /// completion turn merges it back only where a batch it finishes wrote one of its keys, or
    /// may have written memory its uses do not name ([`CopiedBuffers::merge_back`]). Named once
    /// by each pages lying in the copy, until they go ([`Copied::unname`]).
    pub(crate) fn name(&self, key: ResourceKey) {
        let mut entries = lock(&self.buffers.entries);
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == self.id) {
            match entry.keys.iter_mut().find(|(named, _)| *named == key) {
                Some((_, count)) => *count += 1,
                None => entry.keys.push((key, 1)),
            }
        }
    }

    /// Pages that named `key` ([`Copied::name`]) went: no command writes the copy through them
    /// from now on, as every command buffer that used them completed, its batches' writes merged
    /// back first (`command_buffer/completion.rs`). The key goes with the last such pages.
    pub(crate) fn unname(&self, key: ResourceKey) {
        let mut entries = lock(&self.buffers.entries);
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == self.id) {
            if let Some(at) = entry.keys.iter().position(|(named, _)| *named == key) {
                entry.keys[at].1 -= 1;
                if entry.keys[at].1 == 0 {
                    entry.keys.swap_remove(at);
                }
            }
        }
    }

    /// How many keys the copy holds ([`Copied::name`]).
    #[cfg(test)]
    pub(crate) fn keys(&self) -> usize {
        let entries = lock(&self.buffers.entries);
        let entry = entries.iter().find(|entry| entry.id == self.id);
        entry.map_or(0, |entry| entry.keys.len())
    }
}

impl Drop for Copied {
    fn drop(&mut self) {
        let mut entries = lock(&self.buffers.entries);
        let removed = match entries.iter().position(|entry| entry.id == self.id) {
            Some(index) => {
                self.buffers.live.fetch_sub(1, Ordering::Relaxed);
                Some(entries.swap_remove(index))
            }
            None => None,
        };
        if let Some(base) = removed.as_ref().and_then(|entry| entry.base.as_ref()) {
            (self.buffers.base_host_bytes).fetch_sub(base.host_bytes(), Ordering::Relaxed);
            (self.buffers.base_device_bytes).fetch_sub(base.device_bytes(), Ordering::Relaxed);
        }
        // Its watches end, and its base's memory goes, with no lock of the copies' held: a
        // watch's end is the consumer's call, and the device's memory takes its own lock.
        drop(entries);
        drop(removed);
    }
}

impl CopiedBuffers {
    /// Whether a copy is live: a batch submitted now may write one, so its event signals are seen
    /// at its completion turn alone (`event_timeline.rs`).
    pub(crate) fn any(&self) -> bool {
        self.live.load(Ordering::Relaxed) > 0
    }

    /// The first read of the `length` bytes at `pages` for a copy of them ([`read_caller`]:
    /// through the kernel, or where it fails, with plain loads of pages mapped readable), made
    /// before the copy's memory is: a mapping the runtime makes for the copy may land where a
    /// page of the caller's is not mapped, and a read after it would find every byte mapped and
    /// copy memory Metal makes no buffer over. Where the process cannot write every page
    /// ([`mapped_writable`]), or `read_only` (memory of Metal's private option bit 24), the GPU's
    /// writes to the copy are dropped (the module docs). Refused where the process cannot read
    /// every page, or the host has no memory for the bytes the copy and the pages last agreed on.
    ///
    /// # Safety
    ///
    /// `pages` are the caller's bytes, which stay mapped while their copy lives
    /// (`HostMemory::new`), and which the CPU reads without a fault wherever the process maps
    /// them with read permission.
    pub(crate) unsafe fn read(
        pages: NonNull<u8>,
        length: usize,
        read_only: bool,
    ) -> Result<FirstRead, NotCopied> {
        let read = |pages: NonNull<u8>, into: &mut [u8]| {
            // SAFETY: the caller's bytes, read with plain loads only where the process maps them
            // readable (the caller's contract).
            unsafe { read_caller(pages, into, read_pages, mapped_readable) }
        };
        let start = pages.as_ptr() as u64;
        let drops_writes =
            read_only || !mapped_writable(start, start.saturating_add(length as u64));
        // SAFETY: the caller's.
        unsafe { FirstRead::of(pages, length, read, drops_writes) }
    }

    /// Register the copy at `copy` of the caller's bytes at `pages` that `first` read, its bytes
    /// those `first` took, which it holds as the bytes the pages and it agree on: it has no base
    /// until a commit whose command buffer may write it ([`CopiedBuffers::copy_in`]).
    ///
    /// # Safety
    ///
    /// `copy` is as many bytes as `first` read of mapped memory nothing else uses yet, which stays
    /// mapped until the answer is dropped; `pages` are the bytes `first` read, which stay mapped
    /// as long (`HostMemory::new`).
    pub(crate) unsafe fn add(
        self: &Arc<Self>,
        first: FirstRead,
        pages: NonNull<u8>,
        copy: NonNull<u8>,
    ) -> Copied {
        let FirstRead {
            bytes,
            drops_writes,
        } = first;
        let length = bytes.len();
        // SAFETY: the copy is `length` bytes nothing else uses yet (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy.as_ptr(), length) };
        // The copy holds the bytes from here on: no base until a command buffer may write it.
        drop(bytes);
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.entries).push(Entry {
            id,
            pages,
            copy,
            length,
            base: None,
            drops_writes,
            log: None,
            keys: Vec::new(),
        });
        self.live.fetch_add(1, Ordering::Relaxed);
        Copied {
            buffers: Arc::clone(self),
            id,
            drops_writes,
        }
    }

    /// At a commit: write into each copy the command buffer's commands may reach (`used`: the
    /// keys they used, `DirtyCommit::copies_used`, or every copy where one may reach memory no
    /// use names) the bytes the CPU changed in its pages since they last agreed, before any of
    /// the command buffer's work is submitted. A copy none of them reaches is neither compared nor
    /// asked of its logs, and keeps the stamp it last synced to, so the first commit whose
    /// commands reach it takes every write since: a command buffer reads only the resources its
    /// commands use, as Metal's GPU reaches only those (counted:
    /// `WorkCounts::copied_copy_ins_unused`).
    ///
    /// A copy with no base yet (no command buffer that may write it was committed) holds the
    /// bytes its pages and it agree on: its pages are compared with it, and the bytes that differ
    /// written in. Each copy the command buffer's commands may write (`written`: the keys they
    /// wrote, `DirtyCommit::copies_written`, or every copy where one may write memory no use
    /// names), and that has none, then takes its base ([`Base`]), before any of its work is
    /// submitted: its bytes in the device's memory where `device` gives them (`Device::copied_base`),
    /// else in the host's. A copy with a base compares its pages' blocks' tags with its base's,
    /// and only a block whose tag changed reads the copy's, and the base's bytes only where the
    /// copy's changed too: the bytes the CPU changed are written in, and the base takes the
    /// block. Counted: `WorkCounts::copied_bases_made`, `copied_base_blocks_read`. Where the
    /// host has no memory for a base, the copy keeps none and the answer is
    /// `Err(NotCopied::OutOfHostMemory)`: the command buffer runs nothing
    /// (`WorkCounts::copied_bases_refused`), and the copy still holds what its pages and it agree
    /// on, for the next commit.
    ///
    /// A copy whose pages' memory came with a log of the writes to them ([`Copied::set_log`])
    /// compares only the pages the log reports written since the stamp its last copy in synced
    /// to, and the bytes no log keeps: a stamp of each log is asked now, once for every copy
    /// (`DirtyPages::request`), and its publication waited for (`DirtyPages::wait`, bounded),
    /// with no lock of the copies' held; the pages written since are asked once per log and
    /// stamp for every copy (`DirtyPages::changed`), but by a copy whose merges back wrote
    /// [`ASKS_APART`] bytes or more of the pages since, which asks for every writer's but its
    /// own (`DirtyPages::changed_except`).
    /// Every write made before this call is in a publication the wait found, so the copy takes
    /// each. Every byte is compared where the copy has no earlier copy in to count on, or a log
    /// answered `false`. `tally` counts the stamps and waits (`WorkCounts::dirty_stamps_asked`
    /// and kin). A logged copy also compares the bytes at the host addresses `in_flight` gives,
    /// asked before the stamps: those committed command buffers wrote in place whose writes were
    /// not reported as run yet (`DirtyShadows::in_flight`), which the program may know ran (a
    /// host wait for an event they signal) before the logs hear it.
    pub(crate) fn copy_in(
        &self,
        counts: &Counters,
        tally: &DirtyTally,
        (used, written): (&CopiesReached, &CopiesReached),
        in_flight: impl FnOnce() -> Vec<Range<u64>>,
        device: &mut dyn FnMut(usize) -> Option<BaseBytes>,
    ) -> Result<(), NotCopied> {
        if self.live.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let reaches = |entry: &Entry| used.meets(&entry.keys) || written.meets(&entry.keys);
        let in_flight = in_flight();
        let asked: Vec<Asked> = (lock(&self.entries).iter())
            .filter(|entry| reaches(entry))
            .filter_map(|entry| {
                let log = entry.log.as_ref()?;
                Some(Asked {
                    id: entry.id,
                    length: entry.length as u64,
                    pages: Arc::clone(&log.pages),
                    writer: log.writer,
                    unlogged: Arc::clone(&log.unlogged),
                    since: log.since.clone(),
                    merged: log.merged,
                })
            })
            .collect();
        let plans = planned(asked, tally, &in_flight);
        let (mut moved, mut skipped, mut unused, mut read) = (0u64, 0u64, 0u64, 0u64);
        let (mut made, mut refused) = (0u64, 0u64);
        for entry in lock(&self.entries).iter_mut() {
            if !reaches(entry) {
                unused += 1;
                continue;
            }
            let plan = match &entry.log {
                Some(_) => (plans.binary_search_by_key(&entry.id, |(id, _)| *id).ok())
                    .map(|at| &plans[at].1),
                None => None,
            };
            let (pages, copy, length) = (entry.pages, entry.copy, entry.length);
            let whole = [0..length as u64];
            let ranges = match plan.and_then(|plan| plan.bytes.as_ref()) {
                Some(bytes) => &bytes[..],
                None => &whole[..],
            };
            let mut compared = 0u64;
            for range in ranges {
                let bytes = range.start as usize..range.end as usize;
                // SAFETY: the entry's pointers are live while it is registered (`Entry`), and
                // `bytes` lies in its bytes (`blocks`), from a multiple of `SYNC_BLOCK`.
                moved += unsafe {
                    match &mut entry.base {
                        Some(base) => copy_in_tagged(pages, copy, base, length, bytes, &mut read),
                        None => copy_in_untagged(pages, copy, bytes),
                    }
                };
                compared += range.end - range.start;
            }
            skipped += length as u64 - compared;
            if let (Some(log), Some(plan)) = (&mut entry.log, plan) {
                log.since = plan.stamps.clone();
                log.merged = 0;
            }
            if entry.base.is_none() && written.meets(&entry.keys) {
                // SAFETY: the entry's copy is live while it is registered (`Entry`), and the
                // device's bytes are as many as its own.
                let base = unsafe { Base::of_copy(copy, length, device(length)) };
                #[cfg(test)]
                let base = base.filter(|_| !self.bases_refused.load(Ordering::Relaxed));
                match base {
                    Some(base) => {
                        (self.base_host_bytes).fetch_add(base.host_bytes(), Ordering::Relaxed);
                        (self.base_device_bytes).fetch_add(base.device_bytes(), Ordering::Relaxed);
                        entry.base = Some(base);
                        made += 1;
                    }
                    None => refused += 1,
                }
            }
        }
        counts.copied_copied_in(moved);
        counts.copied_copy_in_skipped(skipped);
        counts.copied_copy_ins_unused(unused);
        counts.copied_bases(made, refused, read);
        match refused {
            0 => Ok(()),
            _ => Err(NotCopied::OutOfHostMemory),
        }
    }

    /// At a batch's completion: write back into each copy's pages the bytes the GPU changed in the
    /// copy since they last agreed, where the CPU did not change them since, before anything sees
    /// the batch complete. A copy of pages the process cannot write takes back instead the bytes
    /// they last agreed on wherever the GPU changed it: the GPU's writes are dropped, and nothing
    /// is written to the pages.
    ///
    /// Only the copies the finished batches may have written are compared (`written`): one a
    /// use of their command buffers' commands wrote by one of its keys ([`Copied::name`]), and
    /// every copy where a command may have written memory no use names (the loaded addresses of
    /// a function that may store or update memory atomically, `StageInterface::writes_memory`; an
    /// indirect command buffer's executions), or where its command buffer's writes were not all
    /// kept apart. Each batch's writes are done by its own completion turn, which
    /// compares what they reach; a copy no batch wrote since its last comparison holds what its
    /// pages do, but for the CPU's writes, which the next commit copies in. After a batch finished
    /// with no merge (a failed wait, [`CopiedBuffers::owe`]), the next compares every copy. A copy
    /// with no base was written by no command buffer, as its commit would have made one
    /// ([`CopiedBuffers::copy_in`]), and is not compared. Counted:
    /// `WorkCounts::copied_merges_compared`, `copied_merges_skipped`.
    ///
    /// A copy's blocks' tags are compared with its base's, and only a block whose tag changed
    /// reads its pages' block, and the base's bytes only where the pages' changed too
    /// (`WorkCounts::copied_base_blocks_read`); the base takes the copy's block.
    ///
    /// The bytes written into pages whose memory came with a log are reported to it, as the
    /// copy's own writes: the blocks of [`SYNC_BLOCK`] bytes it wrote a byte of, adjacent ones
    /// joined, in one call to each log for each copy
    /// ([`DirtyPages::wrote_runs_as`](crate::DirtyPages::wrote_runs_as)), before this returns:
    /// the log sees no write the runtime makes, and every other copy and shadow of those pages
    /// must take them, while the copy's own copies in need not.
    pub(crate) fn merge_back(&self, counts: &Counters, written: &CopiesReached) {
        if self.live.load(Ordering::Relaxed) == 0 {
            return;
        }
        let owed = self.owed.swap(false, Ordering::AcqRel);
        let (mut moved, mut dropped, mut read) = (0u64, 0u64, 0u64);
        let (mut compared, mut skipped) = (0u64, 0u64);
        let mut reports = Vec::new();
        for entry in lock(&self.entries).iter_mut() {
            let Some(base) = entry.base.as_mut() else {
                skipped += 1;
                continue;
            };
            if !owed && !written.meets(&entry.keys) {
                skipped += 1;
                continue;
            }
            compared += 1;
            let (copy, pages, length) = (entry.copy, entry.pages, entry.length);
            if entry.drops_writes {
                // SAFETY: the entry's pointers are live while it is registered (`Entry`).
                dropped += unsafe { restore_tagged(copy, pages, base, length, &mut read) };
                continue;
            }
            let mut blocks: Vec<Range<u64>> = Vec::new();
            let mut note = |wrote: Range<usize>| {
                // The block of `SYNC_BLOCK` bytes the write lies in, cut to the copy.
                let start = (wrote.start / SYNC_BLOCK * SYNC_BLOCK) as u64;
                let end = (start + SYNC_BLOCK as u64).min(length as u64);
                match blocks.last_mut() {
                    Some(last) if last.end >= start => last.end = last.end.max(end),
                    _ => blocks.push(start..end),
                }
            };
            // SAFETY: as above.
            moved += unsafe { merge_tagged(copy, pages, base, length, &mut read, &mut note) };
            if let (Some(log), false) = (&mut entry.log, blocks.is_empty()) {
                let wrote: u64 = blocks.iter().map(|bytes| bytes.end - bytes.start).sum();
                log.merged = log.merged.saturating_add(wrote);
                reports.push((Arc::clone(&log.pages), log.writer, blocks));
            }
        }
        // Reported with no lock of the copies' held.
        for (log, writer, blocks) in reports {
            log.wrote_as(&blocks, writer);
        }
        counts.copied_merged_back(moved);
        counts.copied_dropped(dropped);
        counts.copied_merges(compared, skipped);
        counts.copied_bases(0, 0, read);
    }

    /// Find no memory for a base from now on where `refused`: a test's stand-in for a host out of
    /// memory ([`CopiedBuffers::copy_in`]).
    #[cfg(test)]
    pub(crate) fn refuse_bases(&self, refused: bool) {
        self.bases_refused.store(refused, Ordering::Relaxed);
    }

    /// The host's and the device's bytes the live copies' bases hold
    /// (`WorkCounts::copied_base_host_bytes`, `copied_base_device_bytes`).
    pub(crate) fn base_bytes(&self) -> (u64, u64) {
        (
            self.base_host_bytes.load(Ordering::Relaxed),
            self.base_device_bytes.load(Ordering::Relaxed),
        )
    }

    /// A batch finished with no merge back (its queue's wait failed): the next merge back
    /// compares every copy.
    pub(crate) fn owe(&self) {
        self.owed.store(true, Ordering::Release);
    }
}

/// What a logged copy asks its logs at a copy in ([`CopiedBuffers::copy_in`]).
struct Asked {
    id: u64,
    length: u64,
    pages: Arc<PagesLog>,
    writer: NonZeroU64,
    unlogged: Arc<[Range<u64>]>,
    since: Option<Arc<[DirtyStamp]>>,
    merged: u64,
}

/// A logged copy's copy in, as its logs answered: the bytes it compares (`None`: every byte),
/// and the stamps it syncs to (`None`: a log could not say).
struct Plan {
    bytes: Option<Vec<Range<u64>>>,
    stamps: Option<Arc<[DirtyStamp]>>,
}

/// One stamp of a log at a copy in, and whether its publication was found.
struct Stamped {
    source: Arc<DirtySource>,
    stamp: DirtyStamp,
    published: bool,
}

/// The copies in of the copies `asked`, by their ids, in order: a stamp of each log asked, then
/// waited for (counted in `tally`); the pages each log reports written since a copy's stamp
/// asked once per log and stamp (or for one copy, of a copy whose merge back wrote its pages),
/// with no lock of the copies' held; and the bytes at the host addresses `in_flight`
/// ([`CopiedBuffers::copy_in`]).
fn planned(asked: Vec<Asked>, tally: &DirtyTally, in_flight: &[Range<u64>]) -> Vec<(u64, Plan)> {
    let mut stamped: Vec<Stamped> = Vec::new();
    for source in asked.iter().flat_map(|asked| asked.pages.sources()) {
        if !stamped.iter().any(|s| Arc::ptr_eq(&s.source, source)) {
            stamped.push(Stamped {
                source: Arc::clone(source),
                stamp: source.request(tally),
                published: false,
            });
        }
    }
    for stamp in &mut stamped {
        stamp.published = stamp.source.wait(stamp.stamp, tally);
    }
    // The pages each log reports written since a stamp, by every writer: asked once per log and
    // stamp.
    let mut written: Vec<(*const DirtySource, DirtyStamp, Vec<Range<u64>>)> = Vec::new();
    let mut plans: Vec<(u64, Plan)> = Vec::with_capacity(asked.len());
    for asked in asked {
        let of: Vec<&Stamped> = (asked.pages.sources())
            .filter_map(|source| stamped.iter().find(|s| Arc::ptr_eq(&s.source, source)))
            .collect();
        let now: Option<Vec<DirtyStamp>> = (of.iter())
            .map(|stamped| stamped.published.then_some(stamped.stamp))
            .collect();
        // Copies over the same logs sync to one set of stamps.
        let stamps: Option<Arc<[DirtyStamp]>> =
            now.map(
                |now| match plans.last().and_then(|(_, plan)| plan.stamps.as_ref()) {
                    Some(last) if last[..] == now[..] => Arc::clone(last),
                    _ => now.into(),
                },
            );
        let bytes = match (&asked.since, &stamps) {
            (Some(since), Some(_)) if since.len() == of.len() => {
                let mut bytes: Vec<Range<u64>> = asked.unlogged.to_vec();
                let mut own = Vec::new();
                for (index, (stamped, since)) in of.iter().zip(since.iter()).enumerate() {
                    let changed: &[Range<u64>] = match asked.merged >= ASKS_APART {
                        true => {
                            own.clear();
                            let writer = Some(asked.writer);
                            stamped.source.changed(*since, writer, &mut own);
                            joined(&mut own);
                            &own
                        }
                        false => {
                            let key = (Arc::as_ptr(&stamped.source), *since);
                            let at = match (written.iter()).position(|w| (w.0, w.1) == key) {
                                Some(at) => at,
                                None => {
                                    let mut runs = Vec::new();
                                    stamped.source.changed(*since, None, &mut runs);
                                    joined(&mut runs);
                                    written.push((key.0, key.1, runs));
                                    written.len() - 1
                                }
                            };
                            &written[at].2
                        }
                    };
                    asked.pages.bytes_of(index, changed, |run| bytes.push(run));
                }
                bytes.extend(in_flight.iter().filter_map(|at| asked.pages.bytes_at(at)));
                Some(blocks(bytes, asked.length))
            }
            _ => None,
        };
        plans.push((asked.id, Plan { bytes, stamps }));
    }
    plans.sort_by_key(|(id, _)| *id);
    plans
}

/// `runs` sorted by start, those that meet or touch joined.
fn joined(runs: &mut Vec<Range<u64>>) {
    runs.retain(|run| run.start < run.end);
    runs.sort_by_key(|run| run.start);
    let mut kept: Vec<Range<u64>> = Vec::with_capacity(runs.len());
    for run in runs.drain(..) {
        match kept.last_mut() {
            Some(last) if run.start <= last.end => last.end = last.end.max(run.end),
            _ => kept.push(run),
        }
    }
    *runs = kept;
}

/// `bytes` of a copy of `length` bytes, each widened to the [`SYNC_BLOCK`]s it meets (counted
/// from the copy's first byte) and cut to the copy, sorted and joined.
fn blocks(mut bytes: Vec<Range<u64>>, length: u64) -> Vec<Range<u64>> {
    let block = SYNC_BLOCK as u64;
    for run in &mut bytes {
        let start = run.start / block * block;
        let end = run.end.div_ceil(block).saturating_mul(block).min(length);
        *run = start.min(end)..end;
    }
    joined(&mut bytes);
    bytes
}

/// The bytes a copy with no base compares of its pages and itself at once ([`copy_in_untagged`]),
/// and the unit a merge back reports its writes to a log in: a host page.
pub(crate) const SYNC_BLOCK: usize = 4096;

/// A copy in of a copy with no base, which holds the bytes its pages and it agree on: write into
/// `copy` each byte of `pages` that differs from it, in `bytes`. The bytes written.
///
/// Read a block of [`SYNC_BLOCK`] bytes of each at a time, compared whole (slice equality, which
/// the compiler vectorizes); only a block that differs is looked into ([`merge_block`]), so no
/// byte the comparison found equal is written.
///
/// # Safety
///
/// `pages` and `copy` are mapped memory over `bytes`.
pub(crate) unsafe fn copy_in_untagged(
    pages: NonNull<u8>,
    copy: NonNull<u8>,
    bytes: Range<usize>,
) -> u64 {
    let (pages, copy) = (pages.as_ptr(), copy.as_ptr());
    let mut written = 0u64;
    let (mut now, mut was) = ([0u8; SYNC_BLOCK], [0u8; SYNC_BLOCK]);
    let mut at = bytes.start;
    while at < bytes.end {
        let span = (bytes.end - at).min(SYNC_BLOCK);
        // SAFETY: `span` bytes at `at` lie in both (the caller's); the blocks are this call's own.
        unsafe {
            std::ptr::copy_nonoverlapping(pages.add(at), now.as_mut_ptr(), span);
            std::ptr::copy_nonoverlapping(copy.add(at), was.as_mut_ptr(), span);
        }
        if now[..span] != was[..span] {
            // SAFETY: as above.
            written += unsafe { merge_block(&now[..span], copy.add(at), &mut was[..span], false) };
        }
        at += span;
    }
    written
}

/// A copy in of a copy with `base`, over `bytes` (from a multiple of [`TAG_BLOCK`]), of a copy
/// of `length` bytes: each block of the pages whose tag differs from the base's is compared with
/// the base's bytes, and each byte that differs written into `copy`. The base's bytes are the
/// copy's block where its tag still agrees (the GPU did not change it since), else read from the
/// base (counted in `read`). The base then holds the block as written, the pages' bytes over the
/// base's. The bytes written.
///
/// # Safety
///
/// `pages` and `copy` are `length` bytes of mapped memory, and `bytes` lies in them.
unsafe fn copy_in_tagged(
    pages: NonNull<u8>,
    copy: NonNull<u8>,
    base: &mut Base,
    length: usize,
    bytes: Range<usize>,
    read: &mut u64,
) -> u64 {
    let (pages, copy) = (pages.as_ptr(), copy.as_ptr());
    let mut written = 0u64;
    let (mut now, mut was) = ([0u8; TAG_BLOCK], [0u8; TAG_BLOCK]);
    let (first, end) = (bytes.start / TAG_BLOCK, bytes.end.div_ceil(TAG_BLOCK));
    let blocks = base.tags.iter_mut().enumerate().take(end).skip(first);
    for (index, agreed) in blocks {
        let at = index * TAG_BLOCK;
        let span = (length - at).min(TAG_BLOCK);
        // SAFETY: `span` bytes at `at` lie in the pages (the caller's).
        unsafe { read_block(pages.add(at), span, &mut now) };
        let changed = tag(&now);
        if changed == *agreed {
            continue;
        }
        // SAFETY: as above, in the copy.
        unsafe { read_block(copy.add(at), span, &mut was) };
        let copy_agrees = tag(&was) == *agreed;
        if !copy_agrees {
            // SAFETY: as above, in the base's bytes, which are as many.
            unsafe { base.bytes.read(at, &mut was[..span]) };
            *read += 1;
        }
        // SAFETY: as above, in the copy.
        written += unsafe { merge_block(&now[..span], copy.add(at), &mut was[..span], false) };
        // The pages' block over the base's: the pages' own where the copy's agreed.
        *agreed = match copy_agrees {
            true => changed,
            false => tag(&was),
        };
        // SAFETY: as above, in the base's bytes.
        unsafe { base.bytes.write(at, &was[..span]) };
    }
    written
}

/// A merge back of a copy with `base` (that keeps the GPU's writes) of `length` bytes: each block
/// of the copy whose tag differs from the base's is compared with the base's bytes, and each byte
/// that differs written into `pages`, unless the pages' byte differs from the base's already (the
/// CPU wrote it since: [`merge_block`]), each block it wrote a byte of told to `note`. The base's
/// bytes are the pages' block where its tag still agrees (the CPU did not change it since), else
/// read from the base (counted in `read`). The base then holds the copy's block. The bytes
/// written.
///
/// # Safety
///
/// `copy` and `pages` are `length` bytes of mapped memory.
unsafe fn merge_tagged(
    copy: NonNull<u8>,
    pages: NonNull<u8>,
    base: &mut Base,
    length: usize,
    read: &mut u64,
    note: &mut impl FnMut(Range<usize>),
) -> u64 {
    let (copy, pages) = (copy.as_ptr(), pages.as_ptr());
    let mut written = 0u64;
    let (mut now, mut was) = ([0u8; TAG_BLOCK], [0u8; TAG_BLOCK]);
    for (index, agreed) in base.tags.iter_mut().enumerate() {
        let at = index * TAG_BLOCK;
        let span = (length - at).min(TAG_BLOCK);
        // SAFETY: `span` bytes at `at` lie in the copy (the caller's).
        unsafe { read_block(copy.add(at), span, &mut now) };
        let changed = tag(&now);
        if changed == *agreed {
            continue;
        }
        // SAFETY: as above, in the pages.
        unsafe { read_block(pages.add(at), span, &mut was) };
        if tag(&was) != *agreed {
            // SAFETY: as above, in the base's bytes, which are as many.
            unsafe { base.bytes.read(at, &mut was[..span]) };
            *read += 1;
        }
        // SAFETY: as above, in the pages.
        let merged = unsafe { merge_block(&now[..span], pages.add(at), &mut was[..span], true) };
        if merged > 0 {
            note(at..at + span);
        }
        written += merged;
        *agreed = changed;
        // SAFETY: as above, in the base's bytes.
        unsafe { base.bytes.write(at, &now[..span]) };
    }
    written
}

/// A merge back of a copy with `base` whose GPU writes are dropped, of `length` bytes: each block
/// of the copy whose tag differs from the base's takes back the base's bytes wherever it differs
/// from them, and nothing is written to `pages` or the base. The base's bytes are the pages'
/// block where its tag still agrees, else read from the base (counted in `read`). The bytes
/// written into the copy.
///
/// # Safety
///
/// `copy` and `pages` are `length` bytes of mapped memory.
unsafe fn restore_tagged(
    copy: NonNull<u8>,
    pages: NonNull<u8>,
    base: &Base,
    length: usize,
    read: &mut u64,
) -> u64 {
    let (copy, pages) = (copy.as_ptr(), pages.as_ptr());
    let mut written = 0u64;
    let (mut now, mut agreed) = ([0u8; TAG_BLOCK], [0u8; TAG_BLOCK]);
    for (index, tagged) in base.tags.iter().enumerate() {
        let at = index * TAG_BLOCK;
        let span = (length - at).min(TAG_BLOCK);
        // SAFETY: `span` bytes at `at` lie in the copy (the caller's).
        unsafe { read_block(copy.add(at), span, &mut now) };
        if tag(&now) == *tagged {
            continue;
        }
        // SAFETY: as above, in the pages.
        unsafe { read_block(pages.add(at), span, &mut agreed) };
        if tag(&agreed) != *tagged {
            // SAFETY: as above, in the base's bytes, which are as many.
            unsafe { base.bytes.read(at, &mut agreed[..span]) };
            *read += 1;
        }
        // SAFETY: as above, in the copy.
        written += unsafe { merge_block(&agreed[..span], copy.add(at), &mut now[..span], false) };
    }
    written
}

/// One block whose bytes `now` (read of one side) differ from the agreed bytes `was`: 8 bytes at
/// a time, and in each 8 that differ, byte by byte, each byte of `now` that differs from `was`
/// written into `to` (where `keep_to`, unless `to`'s byte differs from `was` already: the other
/// side wrote it since) and taken into `was`: eight in one store where all eight differ (and,
/// where `keep_to`, none of `to`'s does), so no byte that did not change is written. The bytes
/// written.
///
/// # Safety
///
/// `to` is `now.len()` bytes of mapped memory, `was` as long.
unsafe fn merge_block(now: &[u8], to: *mut u8, was: &mut [u8], keep_to: bool) -> u64 {
    let mut written = 0u64;
    let mut at = 0;
    while at < now.len() {
        let span = (now.len() - at).min(8);
        let (now, was) = (&now[at..at + span], &mut was[at..at + span]);
        if now != was {
            let mut there = [0u8; 8];
            if keep_to {
                // SAFETY: `span` bytes at `at` lie in `to` (the caller's).
                unsafe { std::ptr::copy_nonoverlapping(to.add(at), there.as_mut_ptr(), span) };
            }
            // Eight bytes that all changed, none of them written in `to` since: the same bytes
            // the loop below writes, in one store.
            let every = now.iter().zip(was.iter()).all(|(now, was)| now != was);
            if let (Ok(word), true) = (<[u8; 8]>::try_from(now), every) {
                if !keep_to || there[..] == was[..] {
                    // SAFETY: as above; `[u8; 8]` is aligned to 1.
                    unsafe { to.add(at).cast::<[u8; 8]>().write_volatile(word) };
                    written += 8;
                    was.copy_from_slice(now);
                    at += span;
                    continue;
                }
            }
            for byte in 0..span {
                if now[byte] == was[byte] {
                    continue;
                }
                if !keep_to || there[byte] == was[byte] {
                    // SAFETY: as above.
                    unsafe { to.add(at + byte).write_volatile(now[byte]) };
                    written += 1;
                }
                was[byte] = now[byte];
            }
        }
        at += span;
    }
    written
}

/// Read the caller's `into.len()` bytes at `pages` into `into`: through the kernel (`kernel`,
/// [`read_pages`]), and where it fails, with plain loads where every page of them is mapped with
/// read permission (`readable`, asked for the bytes' addresses, [`mapped_readable`]). Whether
/// they were read: `false` where the kernel failed and a page is not mapped readable.
///
/// The kernel's read is a test that faults on nothing, but on Linux it pins each page, and no
/// page of a `VM_PFNMAP` or `VM_IO` mapping can be pinned: a mapping of a dma-buf (udmabuf's on
/// Linux 7.2, measured) is mapped readable and fails it all the same. The plain loads are the
/// access a copy's comparisons make at each commit ([`CopiedBuffers::copy_in`]).
///
/// # Safety
///
/// `pages` is `into.len()` bytes the CPU reads without a fault wherever `readable` answers `true`
/// for them.
pub(crate) unsafe fn read_caller(
    pages: NonNull<u8>,
    into: &mut [u8],
    kernel: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
    readable: impl FnOnce(u64, u64) -> bool,
) -> bool {
    if kernel(pages, into) {
        return true;
    }
    let start = pages.as_ptr() as u64;
    if !readable(start, start + into.len() as u64) {
        return false;
    }
    // SAFETY: the bytes are mapped readable (the caller's); `into` is the runtime's own, apart.
    unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// Read the caller's `into.len()` bytes at `pages` into `into` through the kernel, which answers a
/// page the process cannot read (unmapped, or mapped without read permission) with a failure, not a
/// fault: `false` then. On Linux `process_vm_readv` of the process's own pages; on macOS
/// `mach_vm_read_overwrite` of its own task's; elsewhere nothing is read and the answer is `false`.
#[cfg(target_os = "linux")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    let mut done = 0;
    while done < into.len() {
        let local = libc::iovec {
            iov_base: into[done..].as_mut_ptr().cast(),
            iov_len: into.len() - done,
        };
        let remote = libc::iovec {
            iov_base: pages.as_ptr().wrapping_add(done).cast(),
            iov_len: into.len() - done,
        };
        // SAFETY: the local vector names `into`'s bytes, which the call writes; the remote one is
        // read through the kernel, which answers an unreadable page with an error.
        let read = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if read <= 0 {
            return false;
        }
        done += read as usize;
    }
    true
}

/// [`read_pages`] on macOS.
#[cfg(target_os = "macos")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    unsafe extern "C" {
        /// `<mach/mach_init.h>`: the port of this process's task.
        static mach_task_self_: libc::mach_port_t;
        fn mach_vm_read_overwrite(
            target_task: libc::vm_map_t,
            address: libc::mach_vm_address_t,
            size: libc::mach_vm_size_t,
            data: libc::mach_vm_address_t,
            outsize: *mut libc::mach_vm_size_t,
        ) -> libc::kern_return_t;
    }
    let mut read = 0;
    // SAFETY: the call writes `into`'s bytes alone, and answers an unreadable page of the source
    // with an error.
    let answer = unsafe {
        mach_vm_read_overwrite(
            mach_task_self_,
            pages.as_ptr() as u64,
            into.len() as u64,
            into.as_mut_ptr() as u64,
            &mut read,
        )
    };
    answer == libc::KERN_SUCCESS && read == into.len() as u64
}

/// [`read_pages`] where the system is not asked: nothing is copied.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_pages(_pages: NonNull<u8>, _into: &mut [u8]) -> bool {
    false
}
