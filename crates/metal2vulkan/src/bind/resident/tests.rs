//! Resident-set coalescing properties over deterministic pseudo-random sets.

use super::*;

/// A small deterministic generator (xorshift64*), so the properties need no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Allocations that view one mapping (`host = guest + delta`), as heap sub-allocations do, placed
/// so that nesting, partial overlap, adjacency, and gaps all occur.
fn one_mapping(rng: &mut Rng, delta: u64) -> Vec<RootEntry> {
    let n = 1 + rng.below(24) as usize;
    (0..n)
        .map(|_| {
            let guest = 0x1_0000 + 16 * rng.below(256);
            let size = 16 * rng.below(40);
            RootEntry {
                host: guest.wrapping_add(delta),
                guest,
                size,
                bind_offset: rng.below(64),
            }
        })
        .collect()
}

fn covers(set: &[RootEntry], byte: u64) -> Option<&RootEntry> {
    set.iter()
        .find(|e| e.guest <= byte && u128::from(byte) < u128::from(e.guest) + u128::from(e.size))
}

/// The coalesced set is sorted, disjoint, and non-empty per entry; it covers exactly the bytes of
/// the input (every input byte in one entry, no other byte); every entry translates as the
/// mapping does; bind offsets are cleared; and coalescing is idempotent and independent of the
/// input order.
#[test]
fn coalescing_is_sorted_disjoint_exact_and_idempotent() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..500 {
        let delta = rng.next() & !15;
        let input = one_mapping(&mut rng, delta);
        let out = resident_set(input.clone());
        for pair in out.windows(2) {
            let end = u128::from(pair[0].guest) + u128::from(pair[0].size);
            assert!(end <= u128::from(pair[1].guest), "case {case}: {pair:?}");
        }
        for e in &out {
            assert!(e.size > 0, "case {case}");
            assert_eq!(e.bind_offset, 0, "case {case}");
            assert_eq!(e.host, e.guest.wrapping_add(delta), "case {case}");
        }
        let lo = input.iter().map(|e| e.guest).min().unwrap_or(0);
        let hi = input.iter().map(|e| e.guest + e.size).max().unwrap_or(0);
        for byte in (lo.saturating_sub(16)..hi + 16).step_by(8) {
            let inside = input
                .iter()
                .any(|e| e.guest <= byte && byte < e.guest + e.size);
            assert_eq!(
                covers(&out, byte).is_some(),
                inside,
                "case {case}: byte {byte:#x}"
            );
        }
        assert_eq!(resident_set(out.clone()), out.clone(), "case {case}");
        let mut reversed = input;
        reversed.reverse();
        assert_eq!(resident_set(reversed), out, "case {case}");
    }
}

/// A heap and the buffers sub-allocated inside it coalesce to the heap alone (the outermost
/// wins); a partial overlap keeps the earlier allocation whole and trims the later one's head.
#[test]
fn the_outermost_allocation_wins_and_overlaps_are_trimmed() {
    let e = |guest: u64, size: u64| RootEntry {
        host: guest + 0x1000_0000,
        guest,
        size,
        bind_offset: 3,
    };
    let heap = e(0x1000, 0x1000);
    let out = resident_set(vec![
        e(0x1100, 0x40),
        heap,
        e(0x1000, 0x10),
        e(0x1FF0, 0x10),
    ]);
    assert_eq!(
        out,
        vec![RootEntry {
            bind_offset: 0,
            ..heap
        }]
    );
    let out = resident_set(vec![e(0x2000, 0x100), e(0x1F00, 0x180)]);
    assert_eq!(out.len(), 2);
    assert_eq!((out[0].guest, out[0].size), (0x1F00, 0x180));
    assert_eq!((out[1].guest, out[1].size), (0x2080, 0x80));
    assert_eq!(out[1].host, 0x2080 + 0x1000_0000);
    assert_eq!(resident_set(vec![e(0x10, 0)]), Vec::new());
}

/// Coalescing counts entries after merging, duplicates of one allocation once; a set of more
/// than 2¹⁶ disjoint allocations, the search's unrolled reach, is made whole (2¹⁶ was its cap
/// before).
#[test]
fn a_set_past_the_unrolled_search_is_made_whole() {
    let past = (1u64 << crate::memory::roots::SEARCH_STEPS) * 2 + 3;
    let e = |i: u64| RootEntry {
        host: i * 32,
        guest: i * 32,
        size: 16,
        bind_offset: 0,
    };
    let all: Vec<RootEntry> = (0..past).map(e).collect();
    assert_eq!(resident_set(all.clone()).len() as u64, past);
    let mut dup = all.clone();
    dup.extend(all.iter().copied());
    assert_eq!(resident_set(dup), all);
}

/// A state's resident set lists bound buffers of every stage and used buffers and heaps with both
/// addresses; textures, bytes, and buffers without a device address stay out.
#[test]
fn a_states_resident_set_comes_from_every_stage_and_the_used_resources() {
    use crate::bind::state::{StageTable, TextureFacts, UsedResource};
    use crate::metal;
    use crate::pipeline::sym::Stage;
    let buffer = |gpu: u64, host: Option<u64>| BoundBuffer::Buffer {
        length: 100,
        offset: 8,
        gpu_address: gpu,
        device_address: host,
        attribute_stride: None,
    };
    let mut state = BindState::default();
    let mut v = StageTable::default();
    v.buffers.insert(0, buffer(0x1000, Some(0x9000)));
    v.buffers.insert(1, buffer(0x2000, None));
    v.buffers.insert(
        2,
        BoundBuffer::Bytes {
            len: 64,
            attribute_stride: None,
        },
    );
    let mut f = StageTable::default();
    f.buffers.insert(0, buffer(0x3000, Some(0xA000)));
    state.stages.insert(Stage::Vertex, v);
    state.stages.insert(Stage::Fragment, f);
    let mut heap = UsedResource::new(ResourceClass::Heap, 1);
    heap.gpu_address = Some(0x4000);
    heap.device_address = Some(0xB000);
    heap.length = 0x100;
    let mut tex = UsedResource::new(
        ResourceClass::Texture(TextureFacts::new(
            metal::TextureType::Type2D,
            metal::PixelFormat::RGBA8Unorm,
        )),
        2,
    );
    tex.gpu_address = Some(0x5000);
    tex.device_address = Some(0xC000);
    tex.length = 0x100;
    state.used_resources = vec![heap, tex];
    let got: Vec<(u64, u64, u64)> = resident_entries(&state)
        .iter()
        .map(|e| (e.guest, e.host, e.size))
        .collect();
    assert_eq!(
        got,
        vec![
            (0x1000, 0x9000, 112),
            (0x3000, 0xA000, 112),
            (0x4000, 0xB000, 0x100)
        ]
    );
}

fn entry(host: u64, guest: u64, size: u64) -> RootEntry {
    RootEntry {
        host,
        guest,
        size,
        bind_offset: 0,
    }
}

#[test]
fn the_resident_set_is_sorted_disjoint_and_outermost() {
    let heap = entry(0x9000, 0x1000, 0x1000);
    let inner = entry(0x9100, 0x1100, 0x100);
    let same_start = entry(0x9000, 0x1000, 0x10);
    let after = entry(0x5000, 0x4000, 0x40);
    let partial = entry(0x9f00, 0x1f00, 0x200);
    let empty = entry(0x7000, 0x3000, 0);
    let set = resident_set(vec![after, partial, inner, empty, same_start, heap]);
    assert_eq!(
        set,
        vec![
            heap,
            // The partially overlapping allocation loses its head, host and guest together.
            entry(0xa000, 0x2000, 0x100),
            after
        ]
    );
    for w in set.windows(2) {
        assert!(w[0].guest + w[0].size <= w[1].guest, "disjoint and sorted");
    }
    assert_eq!(resident_set(Vec::new()), Vec::new());
}

/// Indirect command buffers' records are kept whole however they overlap, one entry a start (the
/// largest of those at one start), past the Metal allocations, which coalesce as before: the
/// records of ID 5, 3 GiB long, run past ID 6's start, and ID 6's own stay an entry, so a search
/// at each ID's start finds that ID's records.
#[test]
fn records_are_kept_whole_one_entry_a_start() {
    use crate::indirect_commands::records_address;
    let gib = 1u64 << 30;
    let metal = entry(0x9000, 0x1000, 0x1000);
    let inner = entry(0x9100, 0x1100, 0x100);
    let five = entry(0x10_0000_0000, records_address(5), 3 * gib);
    let six = entry(0x20_0000_0000, records_address(6), 0x1000);
    let six_again = entry(0x30_0000_0000, records_address(6), 0x100);
    let seven = entry(0x40_0000_0000, records_address(7), 0x1000);
    let set = resident_set(vec![seven, six_again, inner, six, five, metal]);
    assert_eq!(set, vec![metal, five, six, seven]);
    for id in [5, 6, 7] {
        let start = records_address(id);
        let found = set.iter().rposition(|e| e.guest <= start).unwrap();
        assert_eq!(
            set[found].guest, start,
            "ID {id}'s start is its own entry's"
        );
    }
}

/// Each resident entry's extent is the unpadded run its allocations hold from its guest address. A
/// 60-byte allocation's entry is 64 bytes and its extent 60; an allocation whose padded window a
/// partially overlapping one continues holds its whole window; the trimmed one's entry holds up to
/// the overlapping allocation's own end; a gap inside a padded window ends the extent (allocations
/// not 16-aligned), so every address below an extent is one an allocation holds.
#[test]
fn each_entrys_extent_is_the_unpadded_run_its_allocations_hold() {
    use crate::bind::state::UsedResource;
    let used = |token: u32, guest: u64, length: u64| {
        let mut u = UsedResource::new(ResourceClass::Buffer, token);
        u.gpu_address = Some(guest);
        u.device_address = Some(guest + 0x10_0000);
        u.length = length;
        u
    };
    let state = BindState {
        used_resources: vec![
            used(0, 0x1000, 60),
            used(1, 0x2000, 100),
            // Overlaps the second's window [0x2000, 0x2070) and runs past it: trimmed to 0x2070.
            used(2, 0x2010, 200),
            // A 4-byte allocation one byte past a 60-byte one, inside its padded window.
            used(3, 0x3000, 60),
            used(4, 0x303D, 4),
        ],
        ..BindState::default()
    };
    let resident = resident_entries(&state);
    let got: Vec<(u64, u64)> = resident.iter().map(|e| (e.guest, e.size)).collect();
    assert_eq!(
        got,
        vec![
            (0x1000, 64),
            (0x2000, 112),
            (0x2070, 0x70),
            (0x3000, 64),
            (0x3040, 13),
        ]
    );
    let extents = resident_extents(&state, &resident);
    assert_eq!(extents, vec![60, 112, 0x68, 60, 1]);
    // Every address below an extent is held by an allocation at its own length.
    let held = |a: u64| {
        state
            .used_resources
            .iter()
            .any(|u| (u.gpu_address.unwrap()..u.gpu_address.unwrap() + u.length).contains(&a))
    };
    for (e, &x) in resident.iter().zip(&extents) {
        for a in e.guest..e.guest + x {
            assert!(held(a), "{a:#x}");
        }
    }
}

/// Ranges listed apart (`BindState::resident_ranges`) join the resident set as used buffers do,
/// their extents counted at their own lengths, and leave the shape as it was.
#[test]
fn ranges_listed_apart_join_the_resident_set_and_not_the_shape() {
    use crate::bind::state::UsedResource;
    use crate::bind::ResidentRange;
    let mut used = UsedResource::new(ResourceClass::Buffer, 0);
    used.gpu_address = Some(0x2000);
    used.device_address = Some(0x9_2000);
    used.length = 100;
    let state = BindState {
        used_resources: vec![used],
        ..BindState::default()
    };
    let listed = BindState {
        resident_ranges: vec![
            ResidentRange::new(0x9_1000, 0x1000, 60),
            // Inside the used buffer's window: dropped as one inside another is.
            ResidentRange::new(0x9_2010, 0x2010, 16),
            ResidentRange::new(0x9_3000, 0x3000, 0x20),
        ],
        ..state.clone()
    };
    let resident = resident_entries(&listed);
    let got: Vec<(u64, u64, u64)> = resident.iter().map(|e| (e.guest, e.host, e.size)).collect();
    assert_eq!(
        got,
        vec![
            (0x1000, 0x9_1000, 64),
            (0x2000, 0x9_2000, 112),
            (0x3000, 0x9_3000, 0x20)
        ]
    );
    assert_eq!(resident_extents(&listed, &resident), vec![60, 100, 0x20]);
    assert_eq!(listed.shape(), state.shape());
}

/// A root table's `R0` points at a resident set uploaded apart, below or above the table, as a
/// count of 32-byte entries modulo `2^64`; a set not a whole number of entries away is refused.
#[test]
fn a_root_table_points_at_a_resident_set_uploaded_apart() {
    use crate::bind::instantiate::ResidentRange;
    for (table, set) in [(0x1000u64, 0x2000u64), (0x2000, 0x1000), (0x40, 0x40)] {
        let fields = RootTable::resident_fields(table, set, 7).unwrap();
        let first = u64::from_le_bytes(fields[..8].try_into().unwrap());
        let count = u64::from_le_bytes(fields[8..].try_into().unwrap());
        assert_eq!(table.wrapping_add(first.wrapping_mul(32)), set);
        assert_eq!(count, 7);
    }
    assert_eq!(RootTable::resident_fields(0x1000, 0x1010, 1), None);
    // The fields lie where the search reads `R0` and `N`.
    let table = stage_root_table(
        &[],
        2,
        &[RootEntry {
            host: 0x9000,
            guest: 0x4000,
            size: 64,
            bind_offset: 0,
        }],
    );
    let at = RootTable::RESIDENT_FIELDS_OFFSET as usize;
    let first = u64::from_le_bytes(table.bytes[at..at + 8].try_into().unwrap());
    let count = u64::from_le_bytes(table.bytes[at + 8..at + 16].try_into().unwrap());
    assert_eq!((first, count), (2, 1));
    assert_eq!(
        table.resident_ranges(),
        vec![ResidentRange::new(0x9000, 0x4000, 64)]
    );
}

/// A resident set made apart coalesces as a root table's does, and holds a range exactly where
/// one of its entries resolves every address of it to the same device address.
#[test]
fn a_resident_set_holds_what_its_entries_resolve_alike() {
    use crate::bind::instantiate::{ResidentRange, ResidentSet};
    // A heap over 0x1_0000..0x1_1000 and a buffer placed in it, and a buffer of its own.
    let heap = ResidentRange::new(0xa_0000, 0x1_0000, 0x1000);
    let placed = ResidentRange::new(0xa_0100, 0x1_0100, 0x40);
    let own = ResidentRange::new(0xb_0000, 0x2_0000, 13);
    let set = ResidentSet::new([placed, own, heap]);
    assert_eq!(set.len(), 2);
    assert_eq!(set.bytes().len(), 64);
    assert!(set.holds(&placed));
    assert!(set.holds(&heap));
    assert!(set.holds(&own));
    // Another device address for the same Metal bytes, a range past the entry, and one in no
    // entry are not held.
    assert!(!set.holds(&ResidentRange::new(0xc_0100, 0x1_0100, 0x40)));
    assert!(!set.holds(&ResidentRange::new(0xa_0f00, 0x1_0f00, 0x200)));
    assert!(!set.holds(&ResidentRange::new(0xd_0000, 0x3_0000, 16)));
    let other = ResidentRange::new(0xd_0000, 0x3_0000, 16);
    let with = set.with([other]);
    assert_eq!(with.len(), 3);
    assert!(with.holds(&other) && with.holds(&own) && with.holds(&placed));
}

/// What a consumer points a replay preprocess's resident root table at holds the set of its
/// ranges in their entries' layout, then each entry's unpadded extent, then a zero: the same
/// set, and the same extents, as the binder lays out after the table it builds of those
/// allocations, so a table pointed there resolves as the binder's own does.
#[test]
fn replay_bytes_are_the_binders_set_then_its_extents() {
    use crate::bind::{ResidentRange, ResidentSet};
    // Three allocations, one inside another and one of a length the padding rounds up.
    let ranges = [
        ResidentRange::new(0x9000, 0x1_0000_0000, 64),
        ResidentRange::new(0x9010, 0x1_0000_0010, 16),
        ResidentRange::new(0x5000, 0x2_0000_0000, 60),
    ];
    let (bytes, count) = ResidentSet::replay_bytes(ranges);
    let set = ResidentSet::new(ranges);
    assert_eq!(count, set.len());
    assert_eq!(count, 2);
    let entries = set.bytes();
    assert_eq!(&bytes[..entries.len()], &entries[..]);
    let extents: Vec<u64> = bytes[entries.len()..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    assert_eq!(extents, [64, 60, 0]);
    let allocations = ranges.iter().map(|r| (r.guest, r.length));
    let entries: Vec<RootEntry> = (set.bytes().as_chunks::<32>().0.iter())
        .map(|e| {
            let field = |at: usize| u64::from_le_bytes(e[at..at + 8].try_into().unwrap());
            RootEntry {
                host: field(0),
                guest: field(8),
                size: field(16),
                bind_offset: field(24),
            }
        })
        .collect();
    assert_eq!(
        extents_bytes(extents_of(allocations, &entries)),
        bytes[set.bytes().len()..]
    );
}

/// Under `BindState::resident_apart` a root table's own resident set is the bound buffers' alone,
/// and without it the whole set; and a resident set made of a table's ranges and of the used
/// resources' and listed ranges (as a consumer that points its tables at one makes it) is the same
/// either way, resolving every address alike, over pseudo-random states whose bound buffers, used buffers and heaps and listed
/// ranges nest, overlap, touch and repeat.
#[test]
fn a_state_resident_apart_keeps_the_used_resources_out_of_its_tables() {
    use crate::bind::state::{StageTable, UsedResource};
    use crate::bind::{ResidentRange, ResidentSet};
    use crate::pipeline::sym::Stage;
    let mut rng = Rng(0x5eed_0fa9_a7c3);
    for _ in 0..300 {
        let delta = 0x10_0000;
        let mut state = BindState::default();
        let (mut vertex, mut fragment) = (StageTable::default(), StageTable::default());
        for (i, e) in one_mapping(&mut rng, delta).into_iter().enumerate() {
            let length = e.size.saturating_sub(rng.below(16));
            match rng.below(4) {
                0 | 1 => {
                    let table = match rng.below(2) {
                        0 => &mut vertex,
                        _ => &mut fragment,
                    };
                    let buffer = BoundBuffer::Buffer {
                        length,
                        offset: 0,
                        gpu_address: e.guest,
                        device_address: Some(e.host),
                        attribute_stride: None,
                    };
                    table.buffers.insert(i as u32, buffer);
                }
                2 => {
                    let class = match rng.below(2) {
                        0 => ResourceClass::Buffer,
                        _ => ResourceClass::Heap,
                    };
                    let mut used = UsedResource::new(class, i as u32);
                    used.gpu_address = Some(e.guest);
                    used.device_address = Some(e.host);
                    used.length = length;
                    state.used_resources.push(used);
                }
                _ => {
                    let range = ResidentRange::new(e.host, e.guest, length);
                    state.resident_ranges.push(range);
                }
            }
        }
        state.stages.insert(Stage::Vertex, vertex);
        state.stages.insert(Stage::Fragment, fragment);
        let mut apart = state.clone();
        apart.resident_apart = true;
        assert_eq!(table_entries(&state), resident_entries(&state));
        let bound_only = BindState {
            used_resources: Vec::new(),
            resident_ranges: Vec::new(),
            ..state.clone()
        };
        assert_eq!(table_entries(&apart), resident_entries(&bound_only));
        let ranges = |entries: Vec<RootEntry>| {
            let table = stage_root_table(&[], 2, &entries);
            let used = state.used_resources.iter().filter_map(|u| {
                Some(ResidentRange::new(
                    u.device_address?,
                    u.gpu_address?,
                    u.length,
                ))
            });
            let listed = state.resident_ranges.iter().copied();
            ResidentSet::new(
                table
                    .resident_ranges()
                    .into_iter()
                    .chain(used)
                    .chain(listed),
            )
        };
        let (pointed, whole) = (ranges(table_entries(&apart)), ranges(table_entries(&state)));
        let entries = |set: ResidentSet| -> Vec<RootEntry> {
            let word =
                |e: &[u8], k: usize| u64::from_le_bytes(e[8 * k..8 * k + 8].try_into().unwrap());
            (set.bytes().chunks(32))
                .map(|e| entry(word(e, 0), word(e, 1), word(e, 2)))
                .collect()
        };
        let (pointed, whole) = (&entries(pointed)[..], &entries(whole)[..]);
        for byte in 0x1_0000..0x1_0000 + 16 * (256 + 40) {
            let resolved =
                |set: &[RootEntry]| covers(set, byte).map(|e| e.host.wrapping_add(byte - e.guest));
            assert_eq!(resolved(pointed), resolved(whole), "{byte:#x}");
        }
        assert_eq!(apart.shape(), state.shape());
    }
}
