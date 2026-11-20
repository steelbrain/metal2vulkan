//! Per-draw and per-dispatch instantiation outputs shared by draws and dispatches, and the
//! arithmetic both share: descriptor ranges and residual bind offsets (`descriptor_range`), and the
//! resident root table ([`RootTable`]), and the buffer-slot, push-field, and texel-buffer
//! instantiation both share (`bind_buffers`, `slot_field`, `put`, `texel_ranges`). A push
//! field's bytes are held in place ([`FieldBytes`]), so a draw's fields allocate nothing.

use crate::bind::plan::compute::internal;
use crate::bind::plan::{BoundKind, PlannedBuffer, ResourceTemplate};
pub(crate) use crate::bind::resident::RootEntry;
use crate::bind::state::{BindState, BoundBuffer, BoundTexture};
use crate::error::{BindError, Feature, Unsupported};
use crate::memory::roots::{
    BIND_OFFSET, ENTRY_BYTES, GUEST, HOST, RESIDENT_COUNT, RESIDENT_FIRST, SINK_ROOT, SIZE,
};
use crate::pipeline::interface::{BufferForm, DescriptorView, PushLoc, Window};
use crate::pipeline::sym::{PushField, SlotRef};
use crate::vk;

/// Push-constant bytes for one stage range (`vkCmdPushConstants`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushConstantWrite {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub bytes: Vec<u8>,
}

/// Bytes the consumer uploads into the buffer behind a plan binding (argument table, handle
/// table) before recording the draw or dispatch.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorBytes {
    pub set: u32,
    pub binding: u32,
    pub bytes: Vec<u8>,
}

/// The byte window of one bound buffer slot that this draw or dispatch can touch: the staging and
/// writeback window. `window` is relative to the bind offset, as in the interface.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferExtent {
    pub slot: SlotRef,
    pub window: Window,
}

/// A push-constant field the consumer fills after uploading `descriptor_bytes`: a declared field,
/// not a patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PushHole {
    /// The device address of the uploaded resident root table ([`RootTable`]), as a `u64` at
    /// `offset` (absolute, in the push-constant block) of every region's push bytes.
    RootTableAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// The device address of the command buffer's `ExecutionStatus` buffer (a function that counts
    /// resident-set misses, reaches it by address), as a `u64` at `offset` of every region's push
    /// bytes.
    ExecutionStatusAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// The device address of the replayed indirect command buffer's record storage (its header), as
    /// a `u64` at `offset` of the device replay's preprocess push bytes.
    CommandRecordsAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// The device address of the region record a device-replay variant's dispatch reads:
    /// `ReplayLayout::region_offset` of the dispatch's command and region in the replay's scratch,
    /// as a `u64` at `offset` of the variant's push bytes. A flat variant's dispatch outside a
    /// replay (`BindingPlan::instantiate_dispatch`) reads the record its preprocess wrote beside
    /// the group counts. A mesh draw of a folded object function (`TASK_EXT`) reads its scratch's
    /// region record: the arguments pass's (`DrawCommands::mesh_arguments`) or the uploaded fold's
    /// (`DrawCommands::object_fold`), at [`MeshArguments::OBJECT_RECORD`](crate::bind::MeshArguments::OBJECT_RECORD), a multiple of 16.
    ReplayRegionAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// The device address of a de-indexed draw's first index (`VariantKey::primitive_id_indices`),
    /// as a `u64` at `offset` of the variant's vertex push bytes: the draw's index buffer's
    /// address plus its index offset, which the binder requires to be a multiple of the index
    /// size, in a buffer of `SHADER_DEVICE_ADDRESS` usage whose address is a multiple of 4; or a
    /// private list form draw's list (`VariantKey::private_list`). The draw reads its indices
    /// there in the vertex stage (`DrawCommands::uses`, as storage); a `u16` index is read in the
    /// aligned 4-byte word that holds it, which lies inside the `VkBuffer` an `IndexBufferRef`
    /// describes.
    CaptureIndexAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// The device address of a merged draw replay's header (`MergedLayout::header_offset` in the
    /// replay's scratch), as a `u64` at `offset` of a merged variant's vertex push bytes
    /// (`VariantKey::merged_draws`, `PushField::MergedDrawAddress`).
    MergedDrawAddress {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
    /// Which draw of a snapshot variant (`VariantKey::snapshot_textures`,
    /// `PushField::SnapshotPhase`) runs, as a `u32` at `offset` of the variant's fragment push
    /// bytes: 1 in the draw that takes the snapshot (every color write mask 0), 0 in the draw that
    /// reads it (the pipeline's masks).
    SnapshotPhase {
        stages: vk::ShaderStageFlags,
        offset: u32,
    },
}

/// A consumer-owned indirect-argument buffer and the byte offset of the arguments.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndirectRef {
    /// The consumer's id for the buffer.
    pub token: u32,
    pub offset: u64,
}

impl IndirectRef {
    /// The arguments at `offset` in the consumer's buffer `token`.
    pub fn new(token: u32, offset: u64) -> IndirectRef {
        IndirectRef { token, offset }
    }
}

/// A consumer-owned index buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexBufferRef {
    /// The consumer's id for the buffer.
    pub token: u32,
    /// The end of the buffer's bytes (`MTLBuffer.length`), against which a draw's indices are
    /// bounded ([`IndexBufferRef::fetch_end`]), in the offsets the draw binds the `VkBuffer` at
    /// (the length itself where the consumer binds the buffer from its start). The buffer's byte 0
    /// lies at a multiple of 4 bytes of those offsets, and the `VkBuffer` behind it holds at least
    /// the length rounded up to 16 bytes (the padding).
    pub length: u64,
}

impl IndexBufferRef {
    /// The consumer's index buffer `token`, `length` bytes long.
    pub fn new(token: u32, length: u64) -> IndexBufferRef {
        IndexBufferRef { token, length }
    }

    /// Where Metal's index fetch from `index_offset` stops reading bytes at their position: the
    /// length rounded up to 4 bytes where the offset lies before the length, so a partial last
    /// index reads the bytes past the length up to there (the padding, or a heap's bytes after a
    /// placed buffer); the length otherwise, where no byte lies inside from the offset. A position
    /// from there on reads index 0. Measured on an Apple M4 (macOS 27.0.1) from offsets that are
    /// multiples of 4; from an offset at or past the length Metal reads the memory past the buffer,
    /// which the bound replaces with index 0, a contained near-equivalent.
    pub fn fetch_end(&self, index_offset: u64) -> u64 {
        if index_offset < self.length {
            self.length
                .checked_next_multiple_of(4)
                .unwrap_or(self.length)
        } else {
            self.length
        }
    }
}

/// A byte range of a buffer: `range` bytes from `offset`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferRange {
    pub offset: u64,
    pub range: u64,
}

/// The range a buffer descriptor of the plan binds for this draw or dispatch. For a
/// `WriteSource::Buffer`, `range.offset` is from the start of the bound allocation; for a
/// `WriteSource::InlineBytes`, from the start of the consumer's upload of the bytes, which the
/// consumer places at an offset aligned to the descriptor type's `min*BufferOffsetAlignment`.
/// The consumer binds exactly this range, never more. It is at least 16 bytes, inside the
/// allocation padded to 16 bytes, and its offset is a multiple of the alignment. For a
/// `WriteSource::TexelBuffer`, the bytes the texel buffer view of the binding's array element
/// `element` views in the bound texture buffer's `VkBuffer` (`texel_ranges`): element 0 for one
/// view, and element `k` for part `k` of a texture buffer bound through its parts
/// (`bind::texel_parts`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorRange {
    pub set: u32,
    pub binding: u32,
    /// The binding's array element the range is bound at: 0 but for a texture buffer's parts.
    #[cfg_attr(feature = "serde", serde(default))]
    pub element: u32,
    pub range: BufferRange,
}

/// The resident root table of a draw or dispatch: 32-byte entries `{host, guest, size,
/// bind_offset}: u64`, little-endian.
///
/// Entry 0 is the null root (zero). Entry 1 is the sink: its `host` field (at
/// [`RootTable::SINK_HOST_OFFSET`]) is left 0 for the consumer to fill with the device address of
/// its `NeutralResource::Sink`; its `guest` and `size` fields hold the resident set's first root
/// `R0` and entry count `N`. Entry `2 + n` is the `BufferForm::Psb` slot `[[buffer(n)]]` (zero
/// when unbound, so every access through it is contained). Entries `R0 .. R0 + N` are the
/// resident set, sorted by guest address and disjoint, each with `bind_offset = 0`. The consumer
/// writes the sink address, uploads the bytes to memory with a 16-byte-aligned device address,
/// and fills `PushHole::RootTableAddress` with that address.
///
/// `R0` counts 32-byte entries from the table's own address, modulo `2^64`, so a consumer may
/// point the table at a resident set it uploaded elsewhere ([`ResidentSet`]): the table and the set
/// 32-byte-aligned, and the set holding at least the table's own ranges
/// ([`RootTable::resident_ranges`]), it writes [`RootTable::resident_fields`] over the sink
/// entry's `guest` and `size` fields, at [`RootTable::RESIDENT_FIELDS_OFFSET`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RootTable {
    pub bytes: Vec<u8>,
}

impl RootTable {
    /// Byte offset of the sink entry's `host` field, which the consumer fills.
    pub const SINK_HOST_OFFSET: u32 = (SINK_ROOT * ENTRY_BYTES + HOST) as u32;

    /// The table's resident set, `R0 .. R0 + N` as the binder wrote it, each at its padded size.
    pub fn resident_ranges(&self) -> Vec<ResidentRange> {
        let field = |root: u64, field: u64| {
            let at = (root * ENTRY_BYTES + field) as usize;
            self.bytes
                .get(at..at + 8)
                .and_then(|bytes| bytes.try_into().ok())
                .map_or(0, u64::from_le_bytes)
        };
        let first = field(SINK_ROOT, RESIDENT_FIRST);
        let count = field(SINK_ROOT, RESIDENT_COUNT);
        (first..first.saturating_add(count))
            .map(|root| ResidentRange {
                host: field(root, HOST),
                guest: field(root, GUEST),
                length: field(root, SIZE),
            })
            .collect()
    }

    /// The table with `statics` (entry `2 + n` for each `(n, entry)`, every `n < first - 2`) and
    /// the resident set `resident` (already coalesced,
    /// [`resident_set`](crate::bind::resident::resident_set)) from root `first`.
    pub(crate) fn build(statics: &[(u32, RootEntry)], first: u64, resident: &[RootEntry]) -> Self {
        let count = first as usize + resident.len();
        let mut bytes = vec![0u8; count * ENTRY_BYTES as usize];
        let mut put = |root: u64, field: u64, value: u64| {
            let at = (root * ENTRY_BYTES + field) as usize;
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        };
        put(SINK_ROOT, RESIDENT_FIRST, first);
        put(SINK_ROOT, RESIDENT_COUNT, resident.len() as u64);
        let statics = statics.iter().map(|(n, e)| (2 + u64::from(*n), e));
        let resident = (first..).zip(resident);
        for (root, e) in statics.chain(resident) {
            put(root, HOST, e.host);
            put(root, GUEST, e.guest);
            put(root, SIZE, e.size);
            put(root, BIND_OFFSET, e.bind_offset);
        }
        RootTable { bytes }
    }
}

/// One allocation of a resident set.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResidentRange {
    /// Its Vulkan device address.
    pub host: u64,
    /// Its Metal address.
    pub guest: u64,
    /// Its length in bytes, which a set pads to a multiple of 16.
    pub length: u64,
}

impl ResidentRange {
    /// The allocation of `length` bytes at device address `host` and Metal address `guest`.
    pub fn new(host: u64, guest: u64, length: u64) -> ResidentRange {
        ResidentRange {
            host,
            guest,
            length,
        }
    }
}

/// A resident set a consumer uploads apart from the root tables that search it: its allocations
/// coalesced as a root table's are (sorted by Metal address, one inside another dropped, a partial
/// overlap trimmed of its head, an empty one dropped; indirect command buffers' records, from
/// `RECORDS_ADDRESS_BASE`, kept whole, one for each start), each at its padded size, as entries in
/// a root table's layout. The consumer uploads [`bytes`]( Self::bytes) at a 32-byte-aligned device
/// address and points root tables there ([`RootTable::resident_fields`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSet {
    entries: Vec<RootEntry>,
}

/// A part of a resident set's ranges, sorted and coalesced once, for a consumer whose ranges change
/// a few at a time: it keeps its ranges in parts, makes again only the parts that changed, and
/// joins them ([`ResidentSet::of_parts`]) into the set [`ResidentSet::new`] makes of every part's
/// ranges, without sorting or coalescing the unchanged parts again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentPart {
    /// Its ranges' entries as a set takes them: sorted, the empty ones dropped.
    sorted: Vec<RootEntry>,
    /// Their resident set.
    set: Vec<RootEntry>,
}

impl ResidentPart {
    /// The part of `ranges`, of any count.
    pub fn new(ranges: impl IntoIterator<Item = ResidentRange>) -> ResidentPart {
        let sorted = crate::bind::resident::sorted(entries(ranges));
        let set = crate::bind::resident::coalesced_of_sorted(&sorted);
        ResidentPart { sorted, set }
    }
}

/// `ranges` as a set's entries, at their padded sizes.
fn entries(ranges: impl IntoIterator<Item = ResidentRange>) -> Vec<RootEntry> {
    ranges
        .into_iter()
        .map(|range| RootEntry {
            host: range.host,
            guest: range.guest,
            size: padded(range.length),
            bind_offset: 0,
        })
        .collect()
}

impl ResidentSet {
    /// The coalesced set of `ranges`, of any count (Refused with `BindError::ResidentSetTooLarge`
    /// past 2^16 entries before).
    pub fn new(ranges: impl IntoIterator<Item = ResidentRange>) -> ResidentSet {
        ResidentSet {
            entries: crate::bind::resident::resident_set(entries(ranges)),
        }
    }

    /// Its entries' count, `N`.
    pub fn len(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Whether it has no entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Its entries, 32 bytes each, in a root table's layout.
    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.entries.len() * ENTRY_BYTES as usize);
        for e in &self.entries {
            for value in [e.host, e.guest, e.size, 0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    /// Whether a search of the set resolves every address of `range` (at its padded length) as
    /// a root table holding `range` alone does: one entry holds the whole range, at the same
    /// device address for its first byte.
    pub fn holds(&self, range: &ResidentRange) -> bool {
        let below = self.entries.partition_point(|e| e.guest <= range.guest);
        let Some(e) = below.checked_sub(1).and_then(|at| self.entries.get(at)) else {
            return range.length == 0;
        };
        let into = range.guest - e.guest;
        let end = u128::from(into) + u128::from(padded(range.length));
        end <= u128::from(e.size) && e.host.wrapping_add(into) == range.host
    }

    /// The set of its entries and `ranges`, coalesced.
    pub fn with(&self, ranges: impl IntoIterator<Item = ResidentRange>) -> ResidentSet {
        let own = self
            .entries
            .iter()
            .map(|e| ResidentRange::new(e.host, e.guest, e.size));
        ResidentSet::new(own.chain(ranges))
    }
}

/// `n` rounded up to a multiple of 16 (the allocation padding contract).
pub(crate) fn padded(n: u64) -> u64 {
    n.saturating_add(15) & !15
}

/// Why [`descriptor_range`] has no range to bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangeError {
    /// The allocation is empty: nothing can back a descriptor.
    Empty,
    /// The range exceeds the descriptor type's maximum (`needed` bytes).
    TooLarge { needed: u64 },
}

/// Whether [`descriptor_range`] starts the range of a slot with `window` at `bind_offset` for the
/// descriptor base alignment `alignment` (`A`, at least 4 used), so its residual is 0, wherever
/// the bind offset lies before the padded end of the allocation: the bind offset is a multiple of
/// `A`, and the window is `Whole` or starts in `[0, A)` (`align_down(bind_offset + start, A)` is
/// then the bind offset, and a range from an aligned bind offset inside the allocation is at least
/// 16 bytes, so it is never widened down). At or past the end, where every byte the slot reaches
/// lies outside the bound buffer, the range is widened down into the allocation and the residual
/// is not 0. The plan specializes `SpecKey::RangeAtBindOffset` on it.
pub(crate) fn starts_at_bind(bind_offset: u64, window: Window, alignment: u64) -> bool {
    let a = alignment.max(4);
    bind_offset.is_multiple_of(a)
        && match window {
            Window::Whole => true,
            Window::Bounded { start, .. } => u64::try_from(start).is_ok_and(|s| s < a),
        }
}

/// The descriptor range and the `SlotBindOffset` residual of a descriptor-form slot, in checked
/// `i128` arithmetic, for an allocation of `length` bytes (padded to 16) bound at `bind_offset`,
/// with the slot's proven `window`, the descriptor base `alignment` `A` (at least 4 is used; a
/// Metal buffer slot's plan gives `max(min*BufferOffsetAlignment, 16)`, so a descriptor byte is
/// congruent to its Metal address modulo 16), and the view's maximum range `max_range`.
///
/// - `Bounded{start, len}`: from `align_down(max(0, bind_offset + start), A)` to
///   `round_up(bind_offset + start + len, 16)`; `Whole`: from `align_down(bind_offset, A)` to the
///   padded end. The end is clamped to the padded allocation.
/// - A range shorter than 16 bytes (a window at or past the end of the allocation) is widened to
///   16 bytes inside the allocation (its start moves down to an aligned offset), so a bound range
///   is never empty and a contained load of word 0 stays inside it.
/// - The residual is `bind_offset` minus the range's start, modulo `2^32`: a leaf reads byte
///   `residual + m` of the range for the `u32` Metal offset `m` from the bind offset.
pub(crate) fn descriptor_range(
    bind_offset: u64,
    length: u64,
    window: Window,
    alignment: u64,
    max_range: u64,
) -> Result<(BufferRange, u32), RangeError> {
    let end = i128::from(padded(length));
    if end == 0 {
        return Err(RangeError::Empty);
    }
    let a = i128::from(alignment.max(4));
    let down = |x: i128| x - x.rem_euclid(a);
    let b = i128::from(bind_offset);
    let (mut lo, hi) = match window {
        Window::Bounded { start, len } => {
            let s = b + i128::from(start);
            (down(s.max(0)), (s + i128::from(len) + 15) & !15)
        }
        Window::Whole => (down(b), end),
    };
    let mut hi = hi.min(end);
    if hi - lo < 16 {
        lo = lo.min(down(end - 16));
        hi = hi.max(lo + 16);
    }
    // `0 <= lo < hi <= end <= 2^64`: both fit `u64`.
    let range = (hi - lo) as u64;
    if range > max_range {
        return Err(RangeError::TooLarge { needed: range });
    }
    let residual = (b - lo).rem_euclid(1 << 32) as u32;
    Ok((
        BufferRange {
            offset: lo as u64,
            range,
        },
        residual,
    ))
}

/// The descriptor range and the `SlotBindOffset` residual of a slot of `window` read at its
/// residual, bound at `bind_offset` of an allocation of `length` bytes (padded to 16), for the
/// descriptor base alignment `alignment` (at least 4 used) and the view's maximum range
/// `max_range`: a range that a run of bind offsets of the allocation shares, so a draw or dispatch
/// that moves only the slot's offset binds the same descriptor (and keeps its set 0) and moves
/// only its residual. For a `Bounded` window, `M` the smaller of `max_range` and `2^32` (a `u32`
/// residual and offset reach no further): the padded allocation whole where it fits `M`, from 0,
/// its residual the bind offset; else the block of `h` bytes the window's first byte lies in (`h`
/// the largest power of two at most half of `M`, a multiple of the alignment), to `M` past the
/// block's start or the padded end. A window of at most `M - h` bytes (at least half of `M`) that
/// starts in a block ends before `M` past the block's start, so it never straddles its block's
/// range; a longer one takes no block. Like [`descriptor_range`]'s, the range starts at a multiple of the
/// alignment, holds at least 16 bytes inside the padded allocation, and holds each byte `m` of
/// the window inside the allocation at `residual + m` (modulo `2^32`). `None` where no such range
/// holds the window (a `Whole` one, one at or past the padded end, one longer than its block
/// allows): the caller takes [`descriptor_range`]'s.
pub(crate) fn shared_range(
    bind_offset: u64,
    length: u64,
    window: Window,
    alignment: u64,
    max_range: u64,
) -> Option<(BufferRange, u32)> {
    let Window::Bounded { start, len } = window else {
        return None;
    };
    let end = i128::from(padded(length));
    let b = i128::from(bind_offset);
    if b >= end {
        return None;
    }
    let first = (b + i128::from(start)).max(0);
    let last = (b + i128::from(start) + i128::from(len)).min(end);
    // No range past `2^32` bytes: a `u32` residual and offset reach no further.
    let max_range = max_range.min(1 << 32);
    let (lo, hi) = if end <= i128::from(max_range) {
        (0, end)
    } else {
        let a = alignment.max(4);
        let block = 1u64 << (max_range / 2).checked_ilog2()?;
        // The window's size limit: starting in the block, it ends inside the block's range.
        if block < a || !block.is_multiple_of(a) || len > max_range - block {
            return None;
        }
        let block = i128::from(block);
        let lo = first - first.rem_euclid(block);
        (lo, (lo + i128::from(max_range)).min(end))
    };
    debug_assert!(last <= hi, "a window past its shared range");
    let residual = b - lo;
    if first >= hi || hi - lo < 16 || residual.abs() >= 1 << 32 {
        return None;
    }
    // `0 <= lo < hi <= end <= 2^64`: both fit `u64`.
    let range = BufferRange {
        offset: lo as u64,
        range: (hi - lo) as u64,
    };
    Some((range, residual.rem_euclid(1 << 32) as u32))
}

/// The descriptor range and the `SlotBindOffset` residual of planned buffer slot `pb` bound at
/// `bind_offset` of an allocation of `length` bytes, where `views_at_residual` says the consumer
/// reads every uniform view at its residual (`BindState::uniform_views_at_residual`). Its function
/// reads byte `residual + m` of the range for the Metal offset `m`, the byte it read of the range
/// from the bind offset, so a slot read at its residual binds a range its bind offsets share:
///
/// - a storage view of a `Whole` window takes its range from the start of the bound buffer's
///   space (offset 0) to the padded end, its residual the bind offset, so every bind offset of one
///   buffer binds one descriptor and a draw or dispatch that moves only the offset keeps its set 0
///   (a dynamic one's offset moves with the space alone); the bytes before the bind offset, which
///   the window does not state, are reachable as Metal's own memory is. Where that range would
///   exceed the view's `max_range` or reach past `2^32` (a `u32` residual and offset), or the bind
///   offset lies at or past the padded end, it takes [`descriptor_range`]'s range;
/// - a storage view of a `Bounded` window, which a function reads at its residual but at bind
///   offset 0 (`SpecKey::RangeAtBindOffset`, where its profile declares the key, set only where
///   [`starts_at_bind`] holds there, which leaves that range's residual 0), and a uniform view of
///   one where `views_at_residual` (no function reads it at its bind offset alone), take
///   [`shared_range`]'s, where it holds the window;
/// - every other slot takes [`descriptor_range`]'s, which starts at the bind offset aligned down
///   to the descriptor base alignment: a uniform view's key may read it at its bind offset alone,
///   where its residual is 0.
pub(crate) fn planned_range(
    pb: &PlannedBuffer,
    bind_offset: u64,
    length: u64,
    views_at_residual: bool,
) -> Result<(BufferRange, u32), RangeError> {
    let end = padded(length);
    let storage = pb.form == BufferForm::Descriptor(DescriptorView::Storage);
    let from_start = pb.window == Window::Whole
        && storage
        && bind_offset < end
        && end <= pb.max_range
        && end <= 1 << 32;
    if !from_start {
        let shared = (storage || (views_at_residual && pb.form.uniform_view()))
            .then(|| shared_range(bind_offset, length, pb.window, pb.alignment, pb.max_range))
            .flatten();
        return match shared {
            Some(shared) => Ok(shared),
            None => descriptor_range(bind_offset, length, pb.window, pb.alignment, pb.max_range),
        };
    }
    // `bind_offset < end <= 2^32`: the residual is the bind offset.
    let range = BufferRange {
        offset: 0,
        range: end,
    };
    Ok((range, bind_offset as u32))
}

/// The bound buffer at `slot` of `state`, checked against the plan's shape.
pub(crate) fn bound_buffer(
    state: &BindState,
    slot: SlotRef,
    expected: BoundKind,
) -> Result<Option<&BoundBuffer>, BindError> {
    let table = state.stages.get(&slot.stage);
    let b = table.and_then(|t| t.buffers.get(&slot.index));
    // The kind alone is compared, which no offset class changes, its length classed as the
    // table's buffer ranges class it in the shape.
    let ranges = table.map(|t| t.buffer_ranges).unwrap_or_default();
    let kind = BoundKind::of_bound(b, ranges);
    debug_assert_eq!(kind, BoundKind::of(b.map(|b| b.shape(0, ranges)).as_ref()));
    if kind != expected {
        return Err(internal(format!(
            "the bind state's shape differs from the plan's at {slot:?}"
        )));
    }
    Ok(b)
}

/// What a draw or dispatch binds for the plan's buffer slots beside their descriptor ranges and
/// extents ([`bind_buffers`]).
pub(crate) struct BoundBuffers {
    /// The `SlotBindOffset` residual of every bound descriptor-form slot.
    pub residuals: Residuals,
    /// The root entry of every bound device-address slot.
    pub statics: Vec<(SlotRef, RootEntry)>,
}

/// The `SlotBindOffset` residuals a [`BoundBuffers`] holds that are kept in place, so a draw or
/// dispatch of no more descriptor-form slots allocates none.
const INLINE_RESIDUALS: usize = 8;

/// The `SlotBindOffset` residual of every bound descriptor-form slot, in the order they were
/// added: the first [`INLINE_RESIDUALS`] in place, the rest after them.
#[derive(Default)]
pub(crate) struct Residuals {
    inline: [Option<(SlotRef, u32)>; INLINE_RESIDUALS],
    rest: Vec<(SlotRef, u32)>,
}

impl Residuals {
    /// Add `slot`'s residual last.
    pub(crate) fn push(&mut self, slot: SlotRef, residual: u32) {
        match self.inline.iter_mut().find(|r| r.is_none()) {
            Some(free) => *free = Some((slot, residual)),
            None => self.rest.push((slot, residual)),
        }
    }

    /// The first residual added for `slot`.
    fn of(&self, slot: SlotRef) -> Option<u32> {
        (self.inline.iter().flatten())
            .chain(&self.rest)
            .find(|(s, _)| *s == slot)
            .map(|r| r.1)
    }
}

/// The Metal addresses `[lo, hi)` a buffer bound at `gpu_address + offset` (an allocation of
/// `length` bytes) can reach through `window`: a `Bounded` window from the bind offset, a `Whole`
/// one from the bind offset to the end, clamped to the padded allocation, since an access outside
/// it is not Metal-defined (containment keeps it in the bound range).
fn metal_extent(window: Window, gpu_address: u64, offset: u64, length: u64) -> (i128, i128) {
    let base = i128::from(gpu_address);
    let end = base + i128::from(padded(length));
    let bind = base + i128::from(offset);
    let (lo, hi) = match window {
        Window::Bounded { start, len } => {
            let lo = bind + i128::from(start);
            (lo, lo + i128::from(len))
        }
        _ => (bind, end),
    };
    (lo.max(base), hi.min(end))
}

/// Refuse a binding where a slot read through a `device` uniform view (`PlannedBuffer::disjoint`)
/// can reach Metal addresses a written slot of the pipeline can reach: a uniform view need not
/// observe the dispatch's or draw's own writes. Inline bytes are copies and alias nothing.
fn check_uniform_aliasing(buffers: &[PlannedBuffer], state: &BindState) -> Result<(), BindError> {
    let extent = |pb: &PlannedBuffer| -> Result<Option<(i128, i128)>, BindError> {
        Ok(match bound_buffer(state, pb.slot, pb.bound)? {
            Some(&BoundBuffer::Buffer {
                length,
                offset,
                gpu_address,
                ..
            }) => Some(metal_extent(pb.window, gpu_address, offset, length)),
            _ => None,
        })
    };
    for pb in buffers.iter().filter(|b| b.disjoint) {
        let Some((lo, hi)) = extent(pb)? else {
            continue;
        };
        for w in buffers
            .iter()
            .filter(|b| b.access.write && b.slot != pb.slot)
        {
            if let Some((wlo, whi)) = extent(w)? {
                if lo < whi && wlo < hi {
                    return Err(BindError::UniformViewAliased {
                        slot: pb.slot,
                        written: w.slot,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Instantiate `buffers` for `state`: each descriptor-form slot's range, pushed onto
/// `descriptor_ranges`, and residual (checked `i128`, `DescriptorRangeExceeded` above the view's
/// limit, an empty allocation `Unsupported{BufferRange}`), each bound device-address slot's root
/// entry, and each bound buffer's extent, pushed onto `extents`. A `device` slot's uniform view
/// must not alias a written slot (`UniformViewAliased`).
pub(crate) fn bind_buffers(
    buffers: &[PlannedBuffer],
    state: &BindState,
    descriptor_ranges: &mut Vec<DescriptorRange>,
    extents: &mut Vec<BufferExtent>,
) -> Result<BoundBuffers, BindError> {
    check_uniform_aliasing(buffers, state)?;
    let mut out = BoundBuffers {
        residuals: Residuals::default(),
        statics: Vec::new(),
    };
    for pb in buffers {
        let bound = bound_buffer(state, pb.slot, pb.bound)?;
        let (bind_offset, length) = match bound {
            Some(&BoundBuffer::Buffer { length, offset, .. }) => {
                extents.push(BufferExtent {
                    slot: pb.slot,
                    window: pb.window,
                });
                (offset, length)
            }
            // `set*Bytes` data: the consumer uploads it at an aligned offset of its own.
            Some(&BoundBuffer::Bytes { len, .. }) => (0, u64::from(len.max(1))),
            None => continue,
        };
        match (pb.form, pb.binding) {
            (BufferForm::Descriptor(_), Some((set, binding))) => {
                let views = state.uniform_views_at_residual;
                let (range, residual) =
                    planned_range(pb, bind_offset, length, views).map_err(|e| match e {
                        RangeError::TooLarge { needed } => BindError::DescriptorRangeExceeded {
                            slot: pb.slot,
                            needed,
                            limit: pb.max_range,
                        },
                        RangeError::Empty => BindError::Unsupported(
                            Unsupported::not_representable(Feature::BufferRange),
                        ),
                    })?;
                descriptor_ranges.push(DescriptorRange {
                    set,
                    binding,
                    element: 0,
                    range,
                });
                out.residuals.push(pb.slot, residual);
            }
            (BufferForm::Psb, _) => {
                if let Some(&BoundBuffer::Buffer {
                    length,
                    offset,
                    gpu_address,
                    device_address: Some(host),
                    ..
                }) = bound
                {
                    out.statics.push((
                        pb.slot,
                        RootEntry {
                            host,
                            guest: gpu_address,
                            size: padded(length),
                            bind_offset: offset,
                        },
                    ));
                }
            }
            _ => return Err(internal(format!("{:?} without its descriptor", pb.slot))),
        }
    }
    Ok(out)
}

/// The bytes of one push field ([`put`]): held in place up to [`FieldBytes::INLINE`] bytes, as
/// every field of a few words is, so writing a draw's or a dispatch's fields allocates nothing;
/// a longer one on the heap.
pub(crate) enum FieldBytes {
    /// The first `len` bytes of `bytes`.
    Inline {
        len: u8,
        bytes: [u8; FieldBytes::INLINE],
    },
    Heap(Vec<u8>),
}

impl FieldBytes {
    /// The most bytes a field holds in place: eight words.
    pub(crate) const INLINE: usize = 32;

    /// A field of `bytes`.
    pub(crate) fn of(bytes: &[u8]) -> FieldBytes {
        match bytes.len() <= Self::INLINE {
            true => {
                let mut inline = [0; Self::INLINE];
                inline[..bytes.len()].copy_from_slice(bytes);
                FieldBytes::Inline {
                    len: bytes.len() as u8,
                    bytes: inline,
                }
            }
            false => FieldBytes::Heap(bytes.to_vec()),
        }
    }

    /// A field of `words`, each little-endian.
    pub(crate) fn words(words: impl IntoIterator<Item = u32>) -> FieldBytes {
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }
}

impl std::ops::Deref for FieldBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            FieldBytes::Inline { len, bytes } => &bytes[..usize::from(*len)],
            FieldBytes::Heap(bytes) => bytes,
        }
    }
}

/// Bytes collected in place while they fit, else on the heap.
impl FromIterator<u8> for FieldBytes {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> FieldBytes {
        let mut inline = [0; Self::INLINE];
        let mut len = 0;
        let mut iter = iter.into_iter();
        for byte in iter.by_ref() {
            if len == Self::INLINE {
                let mut heap = inline.to_vec();
                heap.push(byte);
                heap.extend(iter);
                return FieldBytes::Heap(heap);
            }
            inline[len] = byte;
            len += 1;
        }
        FieldBytes::Inline {
            len: len as u8,
            bytes: inline,
        }
    }
}

/// The bytes of a slot push field (`SlotBindOffset`: the residual, 0 unbound; `SlotGuestBase`: the
/// slot's word of its `gpu_address + offset`, of 0 (Metal's `nil`) unbound ([`guest_word`]);
/// `TextureLevels`: the level count the plan
/// bound; `InstanceCount`: the bound structure's instance count; `Specialized`: the value the plan
/// recorded for its key), or `None` for another field. Planning refuses a `SlotGuestBase` of a
/// `set*Bytes` slot (`MissingGuestAddress`), so no inline-bytes slot is written as null here.
pub(crate) fn slot_field(
    field: PushField,
    residuals: &Residuals,
    resources: &crate::bind::plan::ResourceTemplate,
    state: &BindState,
) -> Option<FieldBytes> {
    match field {
        // The bound structure's instance count, 0 for an unbound slot.
        PushField::InstanceCount(slot) => {
            let count = state
                .stages
                .get(&slot.stage)
                .and_then(|t| t.acceleration_structures.get(&slot.index))
                .map_or(0, |s| s.instance_count);
            Some(FieldBytes::of(&count.to_le_bytes()))
        }
        // The in-place planes of the texture bound at the slot, zeros where none is (the plan
        // names the slot's variant only where one is).
        PushField::YCbCrInPlace(slot) => {
            let words = match state
                .stages
                .get(&slot.stage)
                .and_then(|t| t.textures.get(&slot.index))
            {
                Some(BoundTexture::Texture { facts, .. }) => facts
                    .ycbcr_in_place
                    .map_or([0; crate::bind::YCbCrInPlace::WORDS], |p| p.words()),
                _ => [0; crate::bind::YCbCrInPlace::WORDS],
            };
            let mut bytes = [0u8; 4 * crate::bind::YCbCrInPlace::WORDS];
            for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            Some(FieldBytes::of(&bytes))
        }
        // The rows of the texture bound at the slot in its texel buffer view, zeros where none
        // is (the plan names the slot's variant only where one is).
        PushField::TexelRows(slot) => {
            let words = match state
                .stages
                .get(&slot.stage)
                .and_then(|t| t.textures.get(&slot.index))
            {
                Some(BoundTexture::Texture { facts, .. }) => facts
                    .texel_rows
                    .map_or([0; crate::bind::TexelRows::WORDS], |r| r.words()),
                _ => [0; crate::bind::TexelRows::WORDS],
            };
            let mut bytes = [0u8; 4 * crate::bind::TexelRows::WORDS];
            for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            Some(FieldBytes::of(&bytes))
        }
        // The level count the plan bound.
        PushField::TextureLevels(slot) => resources
            .texture_levels
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, levels)| FieldBytes::of(&levels.to_le_bytes())),
        // The value the bound resources decide, as the plan recorded it.
        PushField::Specialized(key) => resources
            .specialized
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, value)| FieldBytes::of(&value.to_le_bytes())),
        // The integer channel widths of the texture the plan bound.
        PushField::TextureIntegerChannels(slot) => resources
            .texture_integer_channels
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, word)| FieldBytes::of(&word.to_le_bytes())),
        PushField::SlotBindOffset(slot) => Some(FieldBytes::of(&slot_bind_offset(residuals, slot))),
        PushField::SlotGuestBase(slot) => Some(FieldBytes::of(&slot_guest_base(
            state,
            slot,
            resources.biased_slot_words,
        ))),
        PushField::SlotBounds(slot) => Some(FieldBytes::of(&slot_bounds(state, slot))),
        _ => None,
    }
}

/// The bytes of `PushField::SlotBindOffset(slot)` ([`slot_field`]): the slot's residual, 0
/// unbound.
fn slot_bind_offset(residuals: &Residuals, slot: SlotRef) -> [u8; 4] {
    residuals.of(slot).unwrap_or(0).to_le_bytes()
}

/// The `SlotGuestBase` word of Metal address `address`: the address, or where `biased` (a plan of a
/// device compiling under the identity domains, `ResourceTemplate::biased_slot_words`) the address
/// less [`SLOT_ADDRESS_BIAS`](crate::lower::access::SLOT_ADDRESS_BIAS), modulo 2^64, which the
/// pipeline's accesses add back.
pub(crate) fn guest_word(address: u64, biased: bool) -> u64 {
    match biased {
        true => address.wrapping_sub(crate::lower::access::SLOT_ADDRESS_BIAS),
        false => address,
    }
}

/// The bytes of `PushField::SlotGuestBase(slot)` ([`slot_field`]): the word ([`guest_word`]) of the
/// slot's `gpu_address + offset`, of 0 (Metal's `nil`) where it binds no buffer.
pub(crate) fn slot_guest_base(state: &BindState, slot: SlotRef, biased: bool) -> [u8; 8] {
    let base = match state
        .stages
        .get(&slot.stage)
        .and_then(|t| t.buffers.get(&slot.index))
    {
        Some(&BoundBuffer::Buffer {
            gpu_address,
            offset,
            ..
        }) => gpu_address.wrapping_add(offset),
        _ => 0,
    };
    guest_word(base, biased).to_le_bytes()
}

/// The bytes of `PushField::SlotBounds(slot)` ([`slot_field`]): the bound buffer's bytes before
/// its bind offset, then its length, each at most `u32::MAX`; 0 and 0 where the slot binds no
/// buffer.
fn slot_bounds(state: &BindState, slot: SlotRef) -> [u8; 8] {
    let (before, length) = match state
        .stages
        .get(&slot.stage)
        .and_then(|t| t.buffers.get(&slot.index))
    {
        Some(&BoundBuffer::Buffer { length, offset, .. }) => (offset, length),
        _ => (0, 0),
    };
    let word = |v: u64| u32::try_from(v).unwrap_or(u32::MAX).to_le_bytes();
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&word(before));
    bytes[4..].copy_from_slice(&word(length));
    bytes
}

/// A field's bytes at its placed location: the push bytes (relative to the push range's start
/// `push_start`) or the argument table.
#[inline]
pub(crate) fn put(
    push: &mut [u8],
    push_start: u32,
    table: &mut [u8],
    loc: PushLoc,
    bytes: &[u8],
) -> Result<(), BindError> {
    let (buf, at) = match loc {
        PushLoc::Push { offset } => (push, offset.checked_sub(push_start)),
        PushLoc::ArgumentTable { offset } => (table, Some(offset)),
    };
    let at = at.ok_or_else(|| internal("a push field before its range"))? as usize;
    buf.get_mut(at..at + bytes.len())
        .ok_or_else(|| internal("a push field outside its range or table"))?
        .copy_from_slice(bytes);
    Ok(())
}

/// The texel-buffer view ranges of every bound texture buffer the plan writes: the view starts at
/// the texture buffer's offset, which must be a multiple of `minTexelBufferOffsetAlignment` (the
/// shader indexes from element 0 of the view; an unaligned offset is `Unsupported{TexelBuffer,
/// NotRepresentable}`), and spans the whole texel blocks of its length
/// (`VUID-VkBufferViewCreateInfo-range-00929`: a multiple of the view format's block size; bytes
/// past the last whole block hold no texel). Its element count, `⌊length / block size⌋ × texels per
/// block`, must be at most `maxTexelBufferElements` (`VUID-VkBufferViewCreateInfo-range-00930`),
/// else `TexelBufferTooLarge`; a view without a whole block is `Unsupported{TexelBuffer,
/// NotRepresentable}`. A texture buffer the plan binds through its parts (`PlannedTexel::parts`,
/// `bind::texel_parts`) takes one range per part, at its array element: part `k` from the
/// offset plus `k · P` texels, `P` texels long but the last, which holds the rest; one of a
/// format of blocks of several texels is `TexelBufferTooLarge` too.
pub(crate) fn texel_ranges(
    t: &ResourceTemplate,
    state: &BindState,
) -> Result<Vec<DescriptorRange>, BindError> {
    let mut out = Vec::new();
    for pt in &t.texel_buffers {
        let bound = state
            .stages
            .get(&pt.slot.stage)
            .and_then(|s| s.textures.get(&pt.slot.index));
        let Some(&BoundTexture::Buffer { offset, length, .. }) = bound else {
            return Err(internal(format!(
                "the bind state's shape differs from the plan's at {:?}",
                pt.slot
            )));
        };
        if offset % pt.alignment != 0 {
            return Err(BindError::Unsupported(Unsupported::not_representable(
                Feature::TexelBuffer,
            )));
        }
        let block = pt.block_size.max(1);
        let blocks = length / block;
        if blocks == 0 {
            return Err(BindError::Unsupported(Unsupported::not_representable(
                Feature::TexelBuffer,
            )));
        }
        let elements = blocks.saturating_mul(pt.texels_per_block);
        let too_large = BindError::TexelBufferTooLarge {
            slot: pt.slot,
            elements,
            limit: pt.max_elements,
        };
        let range = |element: u32, offset: u64, range: u64| DescriptorRange {
            set: pt.set,
            binding: pt.binding,
            element,
            range: BufferRange { offset, range },
        };
        if pt.parts <= 1 {
            if elements > pt.max_elements {
                return Err(too_large);
            }
            out.push(range(0, offset, blocks * block));
            continue;
        }
        // Parts of `P` texels of one block each: every part but the last full, the last holding
        // at least one texel.
        let (part, parts) = (pt.part_texels, u64::from(pt.parts));
        if pt.texels_per_block != 1 || part == 0 {
            return Err(too_large);
        }
        if elements > part.saturating_mul(parts) || elements <= part.saturating_mul(parts - 1) {
            return Err(internal(format!(
                "{elements} texels of a texture buffer at {:?} in {parts} parts of {part}",
                pt.slot
            )));
        }
        for k in 0..pt.parts {
            let first = part * u64::from(k);
            let texels = part.min(elements - first);
            out.push(range(k, offset + first * block, texels * block));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
