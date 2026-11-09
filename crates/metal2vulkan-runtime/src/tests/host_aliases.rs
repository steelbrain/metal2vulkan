//! No-copy buffers over the same host memory (`host_aliases.rs`).

use super::super::host_aliases::{Alias, HostAliases, Kind, Memories, Quiet};
use super::super::memory::MemoryClass;
use super::super::resource::ResourceKey;
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The registry, its imports stood in for by numbers.
type Aliases = HostAliases<u32>;

/// Note `key` over `memory`, lying in a new import of `Shared` memory, which is dropped at once.
fn join(aliases: &Aliases, key: ResourceKey, memory: (u64, u64)) -> Option<Alias> {
    aliases.join(key, memory, Kind::Import(MemoryClass::Shared), &Arc::new(0))
}

fn copy() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COPY,
        vk::AccessFlags2::TRANSFER_WRITE,
    )
}

/// A write of `bytes` of the buffer `key`.
fn write(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    Use::writes(key, true, UseRange::Bytes(bytes), copy())
}

/// A buffer made over exactly the bytes a live one covers joins it, and keeps the key it joined
/// while the first lives or not; one over other bytes, overlapping or inside, does not.
#[test]
fn a_buffer_over_the_bytes_another_covers_joins_it() {
    let aliases = Aliases::default();
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    assert_eq!(join(&aliases, first, (0x10000, 0x8000)), None);
    assert_eq!(aliases.of(first), None);
    // Inside, overlapping, and of another length from the same byte: none joins.
    for (start, length) in [(0x11000, 0x1000), (0x14000, 0x8000), (0x10000, 0x4000)] {
        let other = ResourceKey::new();
        assert_eq!(
            join(&aliases, other, (start, length)),
            None,
            "{start:#x}+{length:#x}"
        );
        aliases.forget(other);
    }
    assert_eq!(
        join(&aliases, second, (0x10000, 0x8000)),
        Some(Alias { key: first })
    );
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    // The first gone, a third over the memory joins its key still.
    aliases.forget(first);
    assert_eq!(
        join(&aliases, third, (0x10000, 0x8000)),
        Some(Alias { key: first })
    );
    // The second rebound: it keeps its key, and the memory is the third's to share.
    aliases.leave(second);
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    aliases.forget(second);
    aliases.forget(third);
    // Every buffer over the memory gone: the next one over it has its own key.
    let fourth = ResourceKey::new();
    assert_eq!(join(&aliases, fourth, (0x10000, 0x8000)), None);
    assert_eq!(aliases.augment(&[write(fourth, 0..16)]), None);
}

/// A command's uses of a buffer that joined carry a use of the memory's key, of the same bytes,
/// access and tracking; others are as they are, and with no buffer joined nothing is added.
#[test]
fn a_joined_buffers_uses_carry_its_memorys_key() {
    let aliases = Aliases::default();
    let (first, second, apart) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    join(&aliases, first, (0x40000, 0x1000));
    join(&aliases, apart, (0x80000, 0x1000));
    let uses = [write(first, 0..64), write(apart, 0..64)];
    assert_eq!(aliases.augment(&uses), None);
    join(&aliases, second, (0x40000, 0x1000));
    assert_eq!(aliases.augment(&uses), None);
    let untracked = Use::reads(second, false, UseRange::Bytes(128..256), copy());
    let uses = [write(second, 0..64), untracked.clone(), write(apart, 0..64)];
    let augmented = aliases.augment(&uses).unwrap();
    let mut want = uses.to_vec();
    want.push(write(first, 0..64));
    want.push(Use {
        key: first,
        ..untracked
    });
    assert_eq!(augmented, want);
    aliases.forget(second);
    assert_eq!(aliases.augment(&uses), None);
}

/// A buffer over the bytes a live buffer was made over lies in its import while it lives, of its
/// memory class only; the first import noted stays the memory's while it lives, and once every
/// buffer over the memory is gone none is shared.
#[test]
fn a_buffer_over_the_bytes_another_covers_lies_in_its_import() {
    let aliases = Aliases::default();
    let memory = (0x20000, 0x4000);
    let (first, second, managed) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    let import = Arc::new(1);
    aliases.join(first, memory, Kind::Import(MemoryClass::Shared), &import);
    let shared = aliases.import(memory, MemoryClass::Shared).unwrap();
    assert!(Arc::ptr_eq(&shared, &import));
    // Other bytes, or another class, share nothing.
    assert!(aliases
        .import((0x20000, 0x2000), MemoryClass::Shared)
        .is_none());
    assert!(aliases.import(memory, MemoryClass::Managed).is_none());
    // A second over the memory lies in it, and the first import stays the memory's.
    let again = Arc::new(2);
    aliases.join(second, memory, Kind::Import(MemoryClass::Shared), &again);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &import
    ));
    // A managed buffer's import is the memory's for its class.
    let own = Arc::new(3);
    aliases.join(managed, memory, Kind::Import(MemoryClass::Managed), &own);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Managed).unwrap(),
        &own
    ));
    // The first import gone (its buffers rebound), the next noted is shared.
    drop((import, shared));
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    let third = ResourceKey::new();
    aliases.join(third, memory, Kind::Import(MemoryClass::Shared), &again);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &again
    ));
    // Every buffer over the memory gone: its imports are not shared, though they live.
    for key in [first, second, managed, third] {
        aliases.forget(key);
    }
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    assert!(aliases.import(memory, MemoryClass::Managed).is_none());
    drop((again, own));
}

/// A buffer over the bytes a live copy was made over (`copied.rs`) lies in the copy, of either
/// memory class, while no import of its class lives; a copy is not an import a buffer over part
/// of the bytes counts as inside, and once every buffer over the memory is gone it is not shared.
#[test]
fn a_buffer_over_the_bytes_a_copy_covers_lies_in_the_copy() {
    let aliases = Aliases::default();
    let memory = (0x40000, 0x4000);
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let copy = Arc::new(1);
    assert_eq!(aliases.join(first, memory, Kind::Copy, &copy), None);
    for class in [MemoryClass::Shared, MemoryClass::Managed] {
        assert!(Arc::ptr_eq(&aliases.import(memory, class).unwrap(), &copy));
    }
    // Other bytes share nothing, and the copy is no live import around part of them.
    assert!(aliases
        .import((0x40000, 0x2000), MemoryClass::Shared)
        .is_none());
    assert!(!aliases.inside_live((0x40000, 0x2000), MemoryClass::Shared));
    // A buffer that lies in it joins the memory's key.
    assert_eq!(
        aliases.join(second, memory, Kind::Copy, &copy),
        Some(Alias { key: first })
    );
    // An import of a class noted over the same bytes (two buffers made at once) is its class's.
    let import = Arc::new(2);
    aliases.join(third, memory, Kind::Import(MemoryClass::Managed), &import);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Managed).unwrap(),
        &import
    ));
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &copy
    ));
    for key in [first, second, third] {
        aliases.forget(key);
    }
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    drop((copy, import));
}

/// The host page of the tests that name none of their own: an import's unit.
const PAGE: u64 = 0x1000;

/// A reported remap counts the ranges placed buffers lie over that have a page among its bytes
/// rounded out to the import's unit, with or without memories noted: none for bytes beside
/// them, one for a byte of a page one lies on, and none at all with nothing placed.
#[test]
fn a_reported_remap_counts_the_placed_ranges_it_touches() {
    let aliases = Aliases::default();
    assert_eq!(aliases.remapped(0, u64::MAX, PAGE), 0, "nothing placed");
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    aliases.place(first, (0x40000, 0x8000));
    // A second key over the same bytes: one range, one import.
    aliases.place(second, (0x40000, 0x8000));
    // Bytes in the middle of a page past the first range's.
    aliases.place(third, (0x50010, 0x20));
    assert_eq!(
        aliases.remapped(0x30000, 0x40000, PAGE),
        0,
        "the pages before"
    );
    assert_eq!(
        aliases.remapped(0x48000, 0x50000, PAGE),
        0,
        "the pages between"
    );
    assert_eq!(aliases.remapped(0x44000, 0x44001, PAGE), 1, "a byte inside");
    // A byte of the third's page that is not one of its bytes: its page is replaced all the same.
    assert_eq!(
        aliases.remapped(0x50800, 0x50900, PAGE),
        1,
        "a byte of its page"
    );
    assert_eq!(aliases.remapped(0x50800, 0x50900, 1), 0, "not its bytes");
    assert_eq!(aliases.remapped(0x3f000, 0x51000, PAGE), 2, "both");
    // A remap keeps the placements: the same bytes count again.
    assert_eq!(aliases.remapped(0x44000, 0x45000, PAGE), 1, "again");
    aliases.unplace(first);
    assert_eq!(
        aliases.remapped(0x44000, 0x45000, PAGE),
        1,
        "one key left there"
    );
    aliases.forget(second);
    aliases.forget(third);
    assert_eq!(aliases.remapped(0, u64::MAX, PAGE), 0, "all gone");
}

/// After the caller reports a remap of any byte of a memory, a later buffer over it lies in no
/// import made before and joins no earlier key; the buffers made before keep the key they
/// joined, and forgetting them later leaves the new buffer's memory alone. A remap of other
/// bytes changes nothing.
#[test]
fn a_reported_remap_ends_the_sharing_of_imports_made_before() {
    let aliases = Aliases::default();
    let memory = (0x40000, 0x8000);
    let (first, second, after) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let import = Arc::new(1);
    assert_eq!(
        aliases.join(first, memory, Kind::Import(MemoryClass::Shared), &import),
        None
    );
    assert_eq!(
        aliases.join(second, memory, Kind::Import(MemoryClass::Shared), &import),
        Some(Alias { key: first })
    );
    // Bytes next to the memory, on either side: nothing changes.
    aliases.remapped(0x30000, 0x40000, PAGE);
    aliases.remapped(0x48000, 0x50000, PAGE);
    assert!(aliases.import(memory, MemoryClass::Shared).is_some());
    // One page inside the memory: its import is shared no more.
    aliases.remapped(0x44000, 0x45000, PAGE);
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    // A buffer made after imports anew and joins nothing; the second keeps the key it joined.
    let fresh = Arc::new(2);
    assert_eq!(
        aliases.join(after, memory, Kind::Import(MemoryClass::Shared), &fresh),
        None
    );
    assert_eq!(aliases.of(second), Some(Alias { key: first }));
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &fresh
    ));
    // The buffers made before go: the memory stays the new buffer's.
    aliases.forget(first);
    aliases.forget(second);
    assert!(Arc::ptr_eq(
        &aliases.import(memory, MemoryClass::Shared).unwrap(),
        &fresh
    ));
    aliases.forget(after);
    assert!(aliases.import(memory, MemoryClass::Shared).is_none());
    drop((import, fresh));
}

/// Note a value over `memory`.
fn note(memories: &mut Memories<()>, memory: (u64, u64)) {
    memories.get_or_insert_with(memory, || ());
}

/// How many memories the lookup over `start..end` finds, and how many it reads.
fn looked_up(memories: &Memories<()>, start: u64, end: u64) -> (usize, usize) {
    (
        memories.overlapping(start, end).count(),
        memories.candidates(start, end).count(),
    )
}

/// A remap's lookup by address reads the memories it finds and, of each class of length, at
/// most those over one address before the range, never all of them: among 51,065 memories, a
/// free page between noted ones and a free stretch of the address space read none; a whole
/// reservation reads just the 50,000 it covers; each page report of a lazily mapped 32-MiB
/// buffer reads that buffer's memory alone; and a page just past a three-page memory reads that
/// one (its class's longest is four pages), not its neighbours.
#[test]
fn a_remap_s_lookup_reads_only_the_memories_near_its_range() {
    const PAGE: u64 = 0x4000;
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    let (pages, lazy, reserved, threes) = (16 * GIB, 64 * GIB, 128 * GIB, 112 * GIB);
    let mut memories = Memories::<()>::default();
    for page in 0..50_000 {
        note(&mut memories, (pages + 2 * page * PAGE, PAGE));
    }
    for buffer in 0..64 {
        note(&mut memories, (lazy + buffer * 64 * MIB, 32 * MIB));
    }
    note(&mut memories, (reserved, 4 * GIB));
    for three in 0..1000 {
        note(&mut memories, (threes + 4 * three * PAGE, 3 * PAGE));
    }
    assert_eq!(memories.len(), 51_065);
    // A free page between two noted ones, and a free stretch of 4 GiB.
    assert_eq!(looked_up(&memories, pages + PAGE, pages + 2 * PAGE), (0, 0));
    assert_eq!(looked_up(&memories, 200 * GIB, 204 * GIB), (0, 0));
    // A reservation over every one-page memory, and the whole address space.
    let end = pages + 100_000 * PAGE;
    assert_eq!(looked_up(&memories, pages - GIB, end), (50_000, 50_000));
    assert_eq!(looked_up(&memories, 0, u64::MAX), (51_065, 51_065));
    // Each page of the sixth lazily mapped buffer, and a page in the gap after it.
    let sixth = lazy + 5 * 64 * MIB;
    for page in 0..(32 * MIB / PAGE) {
        let at = sixth + page * PAGE;
        assert_eq!(looked_up(&memories, at, at + PAGE), (1, 1), "page {page}");
    }
    let gap = sixth + 40 * MIB;
    assert_eq!(looked_up(&memories, gap, gap + PAGE), (0, 0));
    // The page past each three-page memory.
    for three in 0..1000 {
        let past = threes + (4 * three + 3) * PAGE;
        assert_eq!(
            looked_up(&memories, past, past + PAGE),
            (0, 1),
            "three {three}"
        );
    }
}

/// The lookup by address finds exactly the memories with a byte in the range, among memories
/// of many lengths that overlap each other, and the index keeps its values and count through
/// removals.
#[test]
fn a_remap_s_lookup_finds_exactly_the_memories_with_a_byte_in_its_range() {
    const PAGE: u64 = 0x1000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |below: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % below
    };
    let mut memories = Memories::<()>::default();
    let mut noted = Vec::new();
    for _ in 0..2000 {
        let class = next(13);
        let length = (1 + next(1 << class)) * PAGE;
        let memory = (next(1 << 20) * PAGE, length);
        note(&mut memories, memory);
        noted.push(memory);
    }
    noted.sort_unstable();
    noted.dedup();
    assert_eq!(memories.len(), noted.len());
    for round in 0..2000 {
        if round % 4 == 0 {
            let gone = noted.swap_remove(next(noted.len() as u64) as usize);
            assert!(memories.remove(gone).is_some());
            assert!(memories.remove(gone).is_none());
            assert_eq!(memories.len(), noted.len());
        }
        let start = next(1 << 21) * PAGE;
        let class = next(16);
        let end = start + (1 + next(1 << class)) * PAGE;
        let mut found: Vec<(u64, u64)> = memories.overlapping(start, end).collect();
        found.sort_unstable();
        let mut expected: Vec<(u64, u64)> = noted
            .iter()
            .copied()
            .filter(|&(at, length)| at < end && start < at + length)
            .collect();
        expected.sort_unstable();
        assert_eq!(found, expected, "{start:#x}..{end:#x}");
    }
}

/// Remaps reported from two threads while four others note and forget buffers over eight
/// memories (the second of each pair lying in the first's import, or its own where a remap
/// ended the memory between them) leave the notes whole: once every buffer is gone, a buffer
/// over each memory joins none, as no memory, or an earlier key, lingered.
#[test]
fn remaps_from_any_thread_keep_the_notes_whole() {
    const BASE: u64 = 0x1_0000_0000;
    const PAGE: u64 = 0x4000;
    let aliases = Aliases::default();
    let memory = |round: u64| (BASE + round % 8 * PAGE, PAGE);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let aliases = &aliases;
                scope.spawn(move || {
                    for round in 0..4000 {
                        let memory = memory(round + worker);
                        let import = Arc::new(worker as u32);
                        let (first, second) = (ResourceKey::new(), ResourceKey::new());
                        aliases.join(first, memory, Kind::Import(MemoryClass::Shared), &import);
                        let shared = aliases
                            .import(memory, MemoryClass::Shared)
                            .unwrap_or_else(|| Arc::clone(&import));
                        aliases.join(second, memory, Kind::Import(MemoryClass::Shared), &shared);
                        aliases.forget(first);
                        aliases.forget(second);
                        assert_eq!(aliases.of(second), None);
                    }
                })
            })
            .collect();
        for _ in 0..2 {
            let (aliases, done) = (&aliases, &done);
            scope.spawn(move || {
                let mut round = 0;
                while !done.load(std::sync::atomic::Ordering::SeqCst) {
                    let (start, length) = memory(round);
                    aliases.remapped(start + PAGE / 2, start + length, PAGE);
                    aliases.remapped(0, u64::MAX, PAGE);
                    round += 1;
                }
            });
        }
        for worker in workers {
            worker.join().unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    for round in 0..8 {
        let key = ResourceKey::new();
        assert_eq!(join(&aliases, key, memory(round)), None);
        assert!(aliases.import(memory(round), MemoryClass::Shared).is_none());
        aliases.forget(key);
    }
}

/// A read of `bytes` of the buffer `key` by a fragment shader.
fn sample(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    Use::reads(key, true, UseRange::Bytes(bytes), access)
}

/// Keys over one range of host bytes that joined nothing (one rebound there, or made after a
/// remap of them) share the first placed there while any lies there: a use of another's bytes
/// carries a use of its key; the first's uses take none, the others carrying it. Once the first
/// went, the others carry its key still; once all went, the next over the bytes is on its own.
#[test]
fn keys_over_one_range_share_the_first_placed_there() {
    let aliases = Aliases::default();
    let (first, second, third) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let surface = (0x7f00_0000_0000, 0x7e_9000);
    aliases.place(first, surface);
    assert_eq!(aliases.augment(&[write(first, 0..0x7e_9000)]), None);
    aliases.place(second, surface);
    assert_eq!(aliases.of(second), None, "joined nothing");
    let uses = [sample(second, 0..0x7e_9000)];
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![uses[0].clone(), sample(first, 0..0x7e_9000)])
    );
    assert_eq!(aliases.augment(&[write(first, 0..0x7e_9000)]), None);
    aliases.forget(first);
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![uses[0].clone(), sample(first, 0..0x7e_9000)])
    );
    aliases.forget(second);
    aliases.place(third, surface);
    assert_eq!(aliases.augment(&[sample(third, 0..64)]), None);
}

/// A use of a key's bytes carries a use of the key over each other range they meet, of the bytes
/// they meet counted from that range's first, with the use's access and tracking; bytes that
/// meet none take nothing, and a rebind moves the key to its new range's company.
#[test]
fn a_use_takes_the_keys_of_the_ranges_its_bytes_meet_clipped() {
    let aliases = Aliases::default();
    let (a, b, c) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    aliases.place(a, (0x10000, 0x8000));
    aliases.place(b, (0x14000, 0x8000));
    aliases.place(c, (0x40000, 0x1000));
    // a's bytes 0x5000..0x7000 lie at 0x15000..0x17000: b's 0x1000..0x3000.
    let uses = [write(a, 0x5000..0x7000), write(c, 0..64)];
    assert_eq!(
        aliases.augment(&uses),
        Some(vec![
            uses[0].clone(),
            uses[1].clone(),
            write(b, 0x1000..0x3000)
        ])
    );
    // b's untracked read of its first page meets a's last.
    let untracked = Use::reads(b, false, UseRange::Bytes(0..0x1000), copy());
    assert_eq!(
        aliases.augment(std::slice::from_ref(&untracked)),
        Some(vec![
            untracked.clone(),
            Use {
                key: a,
                range: UseRange::Bytes(0x4000..0x5000),
                ..untracked.clone()
            }
        ])
    );
    assert_eq!(aliases.augment(&[write(a, 0..0x1000)]), None);
    // b rebound beside c: a is on its own, b and c meet.
    aliases.place(b, (0x40800, 0x1000));
    assert_eq!(aliases.augment(&[write(a, 0x5000..0x7000)]), None);
    assert_eq!(
        aliases.augment(&[write(c, 0..0x1000)]),
        Some(vec![write(c, 0..0x1000), write(b, 0..0x800)])
    );
    aliases.unplace(b);
    assert_eq!(aliases.augment(&[write(c, 0..0x1000)]), None);
}

/// The consumer's two keys over one surface's pages, neither joined: a render's write through
/// the first on one queue, then a sample through the second on another, is a wait for the
/// render's batch once the second's uses carry the first's key; as they were, nothing ordered
/// them.
#[test]
fn a_sample_through_a_second_key_waits_for_a_render_through_the_first() {
    use super::super::tracker::{EncoderUses, Point, Reach, Tracker, Wait};
    let aliases = Aliases::default();
    let (rendered, sampled) = (ResourceKey::new(), ResourceKey::new());
    let surface = (0x7f00_0000_0000, 0x7e_9000);
    aliases.place(rendered, surface);
    aliases.place(sampled, surface);
    let summary = |uses: &[Use<ResourceKey>]| {
        let mut encoder = EncoderUses::new(true);
        let uses = aliases.augment(uses).unwrap_or_else(|| uses.to_vec());
        encoder.order(vk::PipelineStageFlags2::NONE, &uses, Reach::Unnamed);
        encoder.finish()
    };
    let render = Use::writes(
        rendered,
        true,
        UseRange::Bytes(0..0x7e_9000),
        Access::new(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
    );
    let mut tracker = Tracker::default();
    let first = Point { queue: 0, value: 1 };
    let plan = tracker.plan(&[summary(&[render])], first);
    tracker.apply(plan);
    let plan = tracker.plan(
        &[summary(&[sample(sampled, 0..0x7e_9000)])],
        Point { queue: 1, value: 1 },
    );
    assert_eq!(
        plan.waits(),
        [Wait {
            at: first,
            stages: vk::PipelineStageFlags2::FRAGMENT_SHADER
        }]
    );
}

/// A use of a range a few others meet reads them from what its range keeps, looking none up by
/// address (`WorkCounts::host_range_lookups`), as a draw's uses of the consumer's buffers over
/// one stretch of its pages do draw after draw; a use of a range met by more than the classes of
/// length the lookup visits looks them up, and takes the same uses.
#[test]
fn a_use_reads_the_ranges_its_range_keeps_while_few() {
    let aliases = Aliases::default();
    let (whole, indices, rects) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    // Three classes of length: a page, two pages and three (up to four).
    aliases.place(whole, (0x10_0000, 0x3000));
    aliases.place(indices, (0x10_0000, 0x1000));
    aliases.place(rects, (0x10_1000, 0x2000));
    let uses = [sample(indices, 0x100..0x200), sample(rects, 0..0x2000)];
    let carried = vec![
        uses[0].clone(),
        uses[1].clone(),
        sample(whole, 0x100..0x200),
        sample(whole, 0x1000..0x3000),
    ];
    for _ in 0..100 {
        assert_eq!(aliases.augment(&uses), Some(carried.clone()));
    }
    // The whole range, met by two ranges, reads them too, in the lookup's order.
    let all = [write(whole, 0..0x3000)];
    let met = vec![
        all[0].clone(),
        write(indices, 0..0x1000),
        write(rects, 0..0x2000),
    ];
    assert_eq!(aliases.augment(&all), Some(met));
    assert_eq!(aliases.range_lookups(), 0);
    // Four quarter pages more inside it, a fourth class: met by six ranges, more than the four
    // classes, it looks them up; each quarter page, met by two, reads them.
    let quarters: Vec<ResourceKey> = (0..4).map(|_| ResourceKey::new()).collect();
    for (quarter, &key) in (0..4u64).zip(&quarters) {
        aliases.place(key, (0x10_0000 + quarter * 0x800, 0x400));
    }
    assert_eq!(
        aliases.augment(&[write(whole, 0x1800..0x1c00)]),
        Some(vec![
            write(whole, 0x1800..0x1c00),
            write(quarters[3], 0..0x400),
            write(rects, 0x800..0xc00),
        ])
    );
    assert_eq!(aliases.range_lookups(), 1);
    assert_eq!(
        aliases.augment(&[sample(quarters[1], 0..0x400)]),
        Some(vec![
            sample(quarters[1], 0..0x400),
            sample(indices, 0x800..0xc00),
            sample(whole, 0x800..0xc00),
        ])
    );
    assert_eq!(aliases.range_lookups(), 1);
}

/// The class of length `length` as the lookup by address orders ranges (`host_aliases::class`):
/// the least `b` with `length <= 2^b`.
fn class_of(length: u64) -> u32 {
    u64::BITS - length.saturating_sub(1).leading_zeros()
}

/// A model of the placements: each range of bytes by its first byte and length, with its key,
/// how many keys lie over it, and whether its key does.
type Model = BTreeMap<(u64, u64), (ResourceKey, usize, bool)>;

/// `key`, which lay over `was`, lies over no bytes in `model` from now on.
fn model_unplace(model: &mut Model, key: ResourceKey, was: Option<(u64, u64)>) {
    let Some(was) = was else {
        return;
    };
    let range = model.get_mut(&was).expect("a placed key's range");
    range.1 -= 1;
    if range.0 == key {
        range.2 = false;
    }
    if range.1 == 0 {
        model.remove(&was);
    }
}

/// The uses a use of `bytes` of `key`, over `own`, takes beside its own in `model`: its range's
/// key where another, and the key of each other range its bytes meet, clipped, in the lookup's
/// order; none where its range is not crowded.
fn model_sharing(
    model: &Model,
    key: ResourceKey,
    own: (u64, u64),
    bytes: std::ops::Range<u64>,
) -> Vec<Use<ResourceKey>> {
    let (range_key, count, first) = model[&own];
    let end = own.0 + own.1;
    let meets = |other: &(u64, u64)| *other != own && other.0 < end && own.0 < other.0 + other.1;
    let mut taken = Vec::new();
    if count == 1 && first && !model.keys().any(meets) {
        return taken;
    }
    if range_key != key {
        taken.push(write(range_key, bytes.clone()));
    }
    let (from, to) = (own.0 + bytes.start.min(own.1), own.0 + bytes.end.min(own.1));
    let mut others: Vec<(&(u64, u64), &(ResourceKey, usize, bool))> =
        model.iter().filter(|(other, _)| meets(other)).collect();
    others.sort_by_key(|(other, _)| (class_of(other.1), other.0, other.1));
    for (other, (other_key, _, _)) in others {
        let (lo, hi) = (from.max(other.0), to.min(other.0 + other.1));
        if lo < hi {
            taken.push(write(*other_key, lo - other.0..hi - other.0));
        }
    }
    taken
}

/// Keys placed, rebound and removed at random over overlapping ranges of many lengths: each use
/// takes exactly the uses a reading of every range gives ([`model_sharing`]), whether its
/// range's company was read from what the range keeps or looked up by address, and both ran.
#[test]
fn a_use_takes_the_ranges_its_bytes_meet_kept_or_looked_up() {
    const PAGE: u64 = 0x1000;
    const BASE: u64 = 0x7000_0000_0000;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move |below: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % below
    };
    let aliases = Aliases::default();
    let keys: Vec<ResourceKey> = (0..48).map(|_| ResourceKey::new()).collect();
    let mut at: Vec<Option<(u64, u64)>> = vec![None; keys.len()];
    let mut model = Model::new();
    let (mut kept, mut looked_up) = (0, 0);
    for round in 0..4000 {
        let index = next(keys.len() as u64) as usize;
        let key = keys[index];
        match next(8) {
            0 => {
                model_unplace(&mut model, key, at[index].take());
                aliases.unplace(key);
            }
            1..=2 => {
                let class = next(7);
                let length = (1 + next(1 << class)) * PAGE;
                let place = (BASE + next(192) * PAGE, length);
                model_unplace(&mut model, key, at[index].take());
                model.entry(place).or_insert((key, 0, true)).1 += 1;
                at[index] = Some(place);
                aliases.place(key, place);
            }
            _ => {}
        }
        let Some(own) = at[index] else {
            continue;
        };
        let start = next(own.1 + PAGE);
        let bytes = start..start + next(own.1 + PAGE);
        let uses = [write(key, bytes.clone())];
        let taken = model_sharing(&model, key, own, bytes);
        let met = model
            .keys()
            .filter(|other| **other != own && other.0 < own.0 + own.1 && own.0 < other.0 + other.1)
            .count();
        let expected: Option<Vec<Use<ResourceKey>>> =
            (!taken.is_empty()).then(|| uses.iter().cloned().chain(taken).collect());
        let before = aliases.range_lookups();
        assert_eq!(aliases.augment(&uses), expected, "round {round}");
        match aliases.range_lookups() - before {
            0 if met > 0 => kept += 1,
            0 => {}
            _ => looked_up += 1,
        }
    }
    assert!(
        kept > 100 && looked_up > 100,
        "kept {kept}, looked up {looked_up}"
    );
}

/// The version counts each change of what `augment` reads: a buffer joining another's memory,
/// a key placed, rebound or unplaced, a buffer forgotten. A buffer over a memory of its own, a
/// rebound buffer leaving its memory, a reported remap, and `augment` itself change nothing it
/// reads, and leave it standing.
#[test]
fn the_version_counts_each_change_augment_reads() {
    let aliases = Aliases::default();
    let (first, second, placed) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let version = aliases.version();
    join(&aliases, first, (0x40000, 0x1000));
    assert_eq!(aliases.version(), version, "a memory of its own");
    join(&aliases, second, (0x40000, 0x1000));
    let joined = aliases.version();
    assert!(joined > version, "joined");
    aliases.augment(&[write(second, 0..64)]);
    aliases.leave(second);
    aliases.remapped(0x40000, 0x41000, PAGE);
    assert_eq!(aliases.version(), joined, "augment, leave and a remap");
    let mut last = joined;
    let mut changed = |what: &str| {
        let now = aliases.version();
        assert!(now > last, "{what}");
        last = now;
    };
    aliases.place(placed, (0x80000, 0x2000));
    changed("placed");
    aliases.place(placed, (0x90000, 0x2000));
    changed("rebound");
    aliases.unplace(placed);
    changed("unplaced");
    aliases.forget(second);
    changed("forgotten");
}

/// An encoder's commands whose uses of bytes name only keys an earlier one found look nothing up
/// (`host_aliases::Quiet`, `WorkCounts::host_alias_lookups`), a key that takes no other use and
/// one that takes some alike; a buffer placed over bytes a quiet key lies over, or a quiet key
/// joining another's memory, between two commands makes the second carry the use it takes now.
#[test]
fn a_quiet_key_is_looked_up_again_once_a_buffer_moves_or_joins() {
    let aliases = Aliases::default();
    let mut quiet = Quiet::default();
    let mut all = Vec::new();
    let mut augment =
        |uses: &[Use<ResourceKey>]| match aliases.augment_into(uses, &mut all, Some(&mut quiet)) {
            true => Some(all.clone()),
            false => None,
        };
    let (first, second, alone, moved) = (
        ResourceKey::new(),
        ResourceKey::new(),
        ResourceKey::new(),
        ResourceKey::new(),
    );
    // Two keys over one range crowd the device's placements; a third lies alone.
    aliases.place(first, (0x10_0000, 0x1000));
    aliases.place(second, (0x10_0000, 0x1000));
    aliases.place(alone, (0x40_0000, 0x1000));
    let used = [write(alone, 0x800..0x840)];
    for _ in 0..3 {
        assert_eq!(augment(&used), None);
    }
    assert_eq!(aliases.lookups(), 1, "a quiet key is looked up once");
    // A key that takes a use is looked up once too, and its uses at other bytes take the same
    // key's, at their own bytes, from what the encoder kept of it.
    for bytes in [0..0x40, 0x40..0x80, 0..0x40] {
        let crowded = [write(second, bytes.clone())];
        assert_eq!(
            augment(&crowded),
            Some(vec![crowded[0].clone(), write(first, bytes)])
        );
    }
    assert_eq!(
        aliases.lookups(),
        2,
        "a key that takes a use is looked up once"
    );
    // A buffer placed over the second half of the lone key's bytes.
    aliases.place(moved, (0x40_0800, 0x1000));
    assert_eq!(
        augment(&used),
        Some(vec![used[0].clone(), write(moved, 0..0x40)])
    );
    // A key found quiet, then joining a live buffer's memory.
    let (owner, joining) = (ResourceKey::new(), ResourceKey::new());
    assert_eq!(join(&aliases, owner, (0x80_0000, 0x1000)), None);
    let joined = [write(joining, 0..0x40)];
    assert_eq!(augment(&joined), None);
    assert_eq!(
        join(&aliases, joining, (0x80_0000, 0x1000)),
        Some(Alias { key: owner })
    );
    assert_eq!(
        augment(&joined),
        Some(vec![joined[0].clone(), write(owner, 0..0x40)])
    );
}

/// The keys an ended encoder found go to the device's next encoder (`HostAliases::quiet`): one
/// that opens while nothing changed looks none of them up again, and takes what they took; one
/// that opens after a change looks its keys up again, and takes what they take now.
#[test]
fn an_ended_encoder_s_keys_go_to_the_next_while_nothing_changes() {
    let aliases = Aliases::default();
    let mut all = Vec::new();
    let (first, second, alone) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    // Two keys over one range crowd the device's placements; a third lies alone.
    aliases.place(first, (0x10_0000, 0x1000));
    aliases.place(second, (0x10_0000, 0x1000));
    aliases.place(alone, (0x40_0000, 0x1000));
    let (quiet_use, crowded_use) = ([write(alone, 0x800..0x840)], [write(second, 0x80..0xc0)]);
    let crowded_takes = vec![crowded_use[0].clone(), write(first, 0x80..0xc0)];
    // An encoder with nothing given back looks its two keys up, once each.
    let mut encoder = aliases.quiet();
    assert!(!aliases.augment_into(&quiet_use, &mut all, Some(&mut encoder)));
    assert!(aliases.augment_into(&crowded_use, &mut all, Some(&mut encoder)));
    assert_eq!(all, crowded_takes);
    assert_eq!(aliases.lookups(), 2);
    aliases.give_quiet(encoder);
    // The next looks up neither, and its crowded key's use takes what it took.
    for _ in 0..3 {
        let mut encoder = aliases.quiet();
        assert!(!aliases.augment_into(&quiet_use, &mut all, Some(&mut encoder)));
        assert!(aliases.augment_into(&crowded_use, &mut all, Some(&mut encoder)));
        assert_eq!(all, crowded_takes);
        aliases.give_quiet(encoder);
    }
    assert_eq!(
        aliases.lookups(),
        2,
        "keys given back are not looked up again"
    );
    // A buffer placed over the second half of the lone key's bytes: the next encoder looks the
    // key up again, and its use takes the new key's.
    let moved = ResourceKey::new();
    aliases.place(moved, (0x40_0800, 0x1000));
    let mut encoder = aliases.quiet();
    assert!(aliases.augment_into(&quiet_use, &mut all, Some(&mut encoder)));
    assert_eq!(all, vec![quiet_use[0].clone(), write(moved, 0..0x40)]);
    assert_eq!(
        aliases.lookups(),
        3,
        "keys given back are looked up again after a change"
    );
}

/// Keys placed, rebound, removed and joined at random, and uses of random bytes of them: an
/// encoder that keeps what it found of each key (`host_aliases::Quiet`) carries exactly the uses
/// a lookup of every use under the lock gives (`HostAliases::augment`), and looks a command up
/// again only once something changed since or a key is new to it, or its range is met by more
/// ranges than it keeps.
#[test]
fn what_an_encoder_keeps_of_a_key_takes_what_a_lookup_takes() {
    const PAGE: u64 = 0x1000;
    const BASE: u64 = 0x7000_0000_0000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |below: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % below
    };
    let aliases = Aliases::default();
    let keys: Vec<ResourceKey> = (0..24).map(|_| ResourceKey::new()).collect();
    let mut at: Vec<Option<(u64, u64)>> = vec![None; keys.len()];
    let mut quiet = Quiet::default();
    let mut all = Vec::new();
    let (mut kept, mut found) = (0, 0);
    for round in 0..6000 {
        let index = next(keys.len() as u64) as usize;
        let key = keys[index];
        match next(16) {
            0 => {
                at[index] = None;
                aliases.unplace(key);
            }
            1 => {
                let class = next(6);
                let length = (1 + next(1 << class)) * PAGE;
                let place = (BASE + next(96) * PAGE, length);
                at[index] = Some(place);
                aliases.place(key, place);
            }
            2 => {
                // Over the same bytes as another live key's: it joins that one's memory.
                let other = next(keys.len() as u64) as usize;
                if let Some(memory) = at[other] {
                    join(&aliases, key, memory);
                }
            }
            _ => {}
        }
        // A command of one or two uses, of keys placed or not.
        let mut uses = Vec::new();
        for _ in 0..1 + next(2) {
            let index = next(keys.len() as u64) as usize;
            let length = at[index].map_or(PAGE, |at| at.1);
            let start = next(length + PAGE);
            let bytes = start..start + 1 + next(length + PAGE);
            uses.push(match next(2) {
                0 => write(keys[index], bytes),
                _ => sample(keys[index], bytes),
            });
        }
        let looked = aliases.augment(&uses);
        let before = aliases.lookups();
        let took = aliases
            .augment_into(&uses, &mut all, Some(&mut quiet))
            .then(|| all.clone());
        assert_eq!(took, looked, "round {round}: {uses:?}");
        match aliases.lookups() - before {
            0 => kept += 1,
            _ => found += 1,
        }
    }
    assert!(kept > 1000 && found > 1000, "kept {kept}, found {found}");
}

/// A staged copy's question (`host_ram/copy.rs`): a key over a byte of the range answers yes,
/// unless it lies alone over its range and the caller calls it quiet; a range only meeting the
/// bytes' end, or one gone, answers nothing.
#[test]
fn others_over_finds_every_key_but_a_quiet_one_alone() {
    let aliases = Aliases::default();
    assert!(
        !aliases.others_over(0x10000, 0x20000, |_| false),
        "none placed"
    );
    let (chunk, buffer, second) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    aliases.place(chunk, (0x10000, 0x10000));
    let quiet = |key: ResourceKey| key == chunk;
    assert!(aliases.others_over(0x18000, 0x19000, |_| false));
    assert!(
        !aliases.others_over(0x18000, 0x19000, quiet),
        "the quiet key alone"
    );
    assert!(
        !aliases.others_over(0x20000, 0x30000, |_| false),
        "past the range's end"
    );
    // A buffer over some of the chunk's bytes: found where the bytes meet it, not elsewhere.
    aliases.place(buffer, (0x14000, 0x1000));
    assert!(aliases.others_over(0x14800, 0x16000, quiet));
    assert!(!aliases.others_over(0x16000, 0x17000, quiet));
    // A second key over the quiet key's very range: it is not alone there.
    aliases.place(second, (0x10000, 0x10000));
    assert!(aliases.others_over(0x16000, 0x17000, quiet));
    aliases.forget(second);
    aliases.forget(buffer);
    assert!(
        !aliases.others_over(0x10000, 0x20000, quiet),
        "the others gone"
    );
}
