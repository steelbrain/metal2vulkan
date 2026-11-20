//! The index bounds pass: both modules valid on every committed profile (L3, `G-val`), and, on
//! the CPU interpreter, the plan and the fill together draw Metal's vertices (each position
//! inside the buffer's whole indices at its index, each past them at index 0, plus the base
//! vertex) through exactly one of the two draws, every fetch inside its bound range; the pool's
//! counter advances by whole words, and a draw the pool cannot hold draws nothing and sets
//! `INDEX_POOL_OVERFLOW`.

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The byte of the bounds words at which the first draw's `VkDrawIndexedIndirectCommand` lies
/// (words 8 to 12 of the module's table).
pub(super) const FIRST_DRAW: u64 = 32;
/// The byte of the second draw's (words 13 to 17).
const SECOND_DRAW: u64 = 52;

/// The pool's index bytes in these tests.
const POOL_BYTES: u32 = 64;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Index buffer bytes of `indices`, 4 bytes each where `wide`, else 2.
fn index_bytes(indices: &[u32], wide: bool) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|&i| {
            if wide {
                i.to_le_bytes().to_vec()
            } else {
                (i as u16).to_le_bytes().to_vec()
            }
        })
        .collect()
}

/// The two modules' run over one draw: what both draws read, and the pool and status after.
struct Ran {
    words: Vec<u32>,
    pool: Vec<u32>,
    report: StatusReport,
}

/// Run the plan of `pass` over `arguments` (bound whole) with a pool whose counter is `counter`,
/// then the fill over `indices` (bound as `pass.indices_range` says, from offset 0 of `indices`),
/// each in both invocation orders.
fn ran(pass: &IndexBoundsPass, arguments: &[u32], indices: &[u8], counter: u32) -> Ran {
    let mut bounds = pass.parameters.to_vec();
    bounds.resize((IndexBoundsPass::BYTES / 4) as usize, 0xEEEE_EEEE);
    let header = (IndexBoundsPass::POOL_HEADER_BYTES / 4) as usize;
    let mut pool = vec![0xDDDD_DDDD; header + (POOL_BYTES / 4) as usize];
    pool[0] = counter;
    let plan = plan_typed_module().expect("plan");
    let mut planned = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(WORDS),
                    address: None,
                    bytes: bytes(&bounds),
                },
                Buffer {
                    slot: Some(SOURCE),
                    address: None,
                    bytes: bytes(arguments),
                },
                Buffer {
                    slot: Some(POOL),
                    address: None,
                    bytes: bytes(&pool),
                },
                Buffer {
                    slot: Some(STATUS),
                    address: None,
                    bytes: ExecutionStatus::initial().to_vec(),
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            ..Config::default()
        };
        planned.push(run(&plan, &cfg, res).expect("plan run").buffers);
    }
    assert_eq!(planned[0], planned[1], "order independent");
    let after = &planned[0];
    let bounds = words(&after[0]);
    let mut pool = words(&after[2]);
    let status: [u8; ExecutionStatus::BYTES] = after[3].as_slice().try_into().expect("status");
    let report = ExecutionStatus::decode(&status);
    let groups = bounds[Output::FillDispatch as usize];
    if groups > 0 {
        // The indices as bound: the range `pass` names, or 4 bytes nothing reads.
        let bound = match pass.indices_range {
            Some(range) => {
                indices[range.offset as usize..(range.offset + range.range) as usize].to_vec()
            }
            None => vec![0xCC; 4],
        };
        let fill = fill_typed_module().expect("fill");
        let mut filled = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: Some(WORDS),
                        address: None,
                        bytes: bytes(&bounds),
                    },
                    Buffer {
                        slot: Some(SOURCE),
                        address: None,
                        bytes: bound.clone(),
                    },
                    Buffer {
                        slot: Some(POOL),
                        address: None,
                        bytes: bytes(&pool),
                    },
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups, 1, 1],
                ..Config::default()
            };
            filled.push(run(&fill, &cfg, res).expect("fill run").buffers);
        }
        assert_eq!(filled[0], filled[1], "order independent");
        pool = words(&filled[0][2]);
    }
    Ran {
        words: bounds,
        pool,
        report,
    }
}

/// A `VkDrawIndexedIndirectCommand` at `offset` of the bounds words.
fn command(words: &[u32], offset: u64) -> [u32; 5] {
    let at = (offset / 4) as usize;
    words[at..at + 5].try_into().expect("command")
}

/// The vertices the two draws draw, as Vulkan fetches them without `robustBufferAccess2`: the
/// first from `indices` bound at `index_offset` (whose bytes run to `vk_end`, the `VkBuffer`'s
/// end), the second from the pool's index bytes; every fetch must lie inside its bound range.
fn drawn(r: &Ran, indices: &[u8], index_offset: u64, vk_end: u64, wide: bool) -> Vec<i64> {
    let size = if wide { 4 } else { 2 };
    let read = |bytes: &[u8], at: usize| -> u32 {
        if wide {
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("index"))
        } else {
            u32::from(u16::from_le_bytes(
                bytes[at..at + 2].try_into().expect("index"),
            ))
        }
    };
    let first = command(&r.words, FIRST_DRAW);
    let second = command(&r.words, SECOND_DRAW);
    assert!(
        first[0] == 0 || second[0] == 0,
        "one draw carries the primitives: {first:?} {second:?}"
    );
    let mut out = Vec::new();
    for i in 0..u64::from(first[0]) {
        let at = index_offset + (u64::from(first[2]) + i) * size;
        assert!(at + size <= vk_end, "the first draw fetches past its range");
        out.push(i64::from(read(indices, at as usize)) + i64::from(first[3] as i32));
    }
    let header = IndexBoundsPass::POOL_HEADER_BYTES as usize;
    let pool = bytes(&r.pool);
    for i in 0..u64::from(second[0]) {
        let at = (u64::from(second[2]) + i) * size;
        assert!(
            at + size <= u64::from(POOL_BYTES),
            "the second draw fetches past the pool"
        );
        out.push(i64::from(read(&pool[header..], at as usize)) + i64::from(second[3] as i32));
    }
    for draw in [first, second] {
        assert_eq!(draw[1], 1, "instances");
        assert_eq!(draw[4], 0, "base instance");
    }
    out
}

/// Metal's vertices: position `start + i` at its index where it lies among the `inside` whole
/// indices, else index 0, plus the base vertex.
fn metal(indices: &[u32], inside: u64, start: u32, count: u32, base: i32) -> Vec<i64> {
    (0..u64::from(count))
        .map(|i| {
            let p = u64::from(start) + i;
            let index = if p < inside { indices[p as usize] } else { 0 };
            i64::from(index) + i64::from(base)
        })
        .collect()
}

/// Metal's arguments: indexCount, instanceCount, indexStart, baseVertex, baseInstance.
fn arguments(count: u32, start: u32, base: i32) -> [u32; 5] {
    [count, 1, start, base as u32, 0]
}

#[test]
fn both_modules_are_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(plan_typed_module().expect("plan"), &p);
        assert_valid(fill_typed_module().expect("fill"), &p);
    }
}

/// The pass binds the arguments and the whole indices from aligned offsets, and states the
/// residuals, the whole indices, the index size and the pool; it refuses what Metal and Vulkan
/// do not take.
#[test]
fn the_pass_binds_aligned_ranges_and_states_its_parameters() {
    let pass = IndexBoundsPass::new(IndirectRef::new(0, 68), 4, 8, 30, 64, 1 << 27, POOL_BYTES)
        .expect("pass");
    assert_eq!(
        pass.arguments_range,
        BufferRange {
            offset: 64,
            range: 24
        }
    );
    // Whole indices from offset 8 of 30 bytes: 5 (bytes 8 to 28).
    assert_eq!(
        pass.indices_range,
        Some(BufferRange {
            offset: 0,
            range: 28
        })
    );
    assert_eq!(pass.parameters, [1, 8, 5, 1, POOL_BYTES, 0, 0, 0]);
    let at_end = IndexBoundsPass::new(IndirectRef::new(0, 0), 2, 30, 30, 64, 1 << 27, POOL_BYTES)
        .expect("at the end");
    assert_eq!(at_end.indices_range, None);
    assert_eq!(at_end.parameters[2], 0);
    let refused = |arguments: u64, size: u64, offset: u64, pool: u32, max_range: u32| {
        IndexBoundsPass::new(
            IndirectRef::new(0, arguments),
            size,
            offset,
            64,
            64,
            max_range,
            pool,
        )
        .is_none()
    };
    assert!(refused(2, 4, 0, POOL_BYTES, 1 << 27), "arguments at 2");
    assert!(refused(0, 8, 0, POOL_BYTES, 1 << 27), "8-byte indices");
    assert!(
        refused(0, 4, 2, POOL_BYTES, 1 << 27),
        "a misaligned index offset"
    );
    assert!(refused(0, 4, 0, 6, 1 << 27), "a pool of part of a word");
    assert!(
        refused(0, 4, 0, POOL_BYTES, 32),
        "indices past maxStorageBufferRange"
    );
}

/// Every position inside the whole indices draws through the first draw, from the Metal index
/// buffer as it is bound; the second draws nothing and the pool is untouched.
#[test]
fn a_draw_inside_the_buffer_draws_from_it() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, true);
    let pass =
        IndexBoundsPass::new(IndirectRef::new(0, 0), 4, 0, 16, 64, 1 << 27, POOL_BYTES).unwrap();
    for (count, start) in [(4, 0), (3, 1), (1, 3), (0, 4), (0, 0)] {
        let r = ran(&pass, &arguments(count, start, 100), &buffer, 0);
        assert_eq!(
            drawn(&r, &buffer, 0, 16, true),
            metal(&indices, 4, start, count, 100),
            "{count} from {start}"
        );
        assert_eq!(command(&r.words, FIRST_DRAW)[0], count);
        assert_eq!(r.pool[0], 0, "no reservation");
        assert!(!r.report.index_pool_overflow);
    }
}

/// A draw whose positions run past the buffer draws through the second draw: the positions
/// inside at their indices, those past at index 0, plus the base vertex, as Metal draws them;
/// the counter advances by whole words.
#[test]
fn a_draw_past_the_buffer_draws_metals_vertices_from_the_pool() {
    for wide in [true, false] {
        let indices = [5, 6, 7];
        let size = if wide { 4 } else { 2 };
        // From offset `size`: one whole index (6), then part of one (7's first bytes).
        let mut buffer = index_bytes(&indices, wide);
        buffer.truncate((3 * size - 1) as usize);
        let end = buffer.len() as u64;
        let inside = (end - size) / size;
        let vk_end = end.next_multiple_of(16);
        buffer.resize(vk_end as usize, 0x5A);
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, 0),
            size,
            size,
            end,
            64,
            1 << 27,
            POOL_BYTES,
        )
        .unwrap();
        let after = &indices[1..];
        for (count, start, base) in [(4, 0, 100), (3, 1, -3), (5, 2, 0), (1, 2, 7), (3, 0, 9)] {
            let r = ran(&pass, &arguments(count, start, base), &buffer, 8);
            let want = metal(after, inside, start, count, base);
            assert_eq!(
                drawn(&r, &buffer, size, vk_end, wide),
                want,
                "{size}-byte indices: {count} from {start}"
            );
            let first = command(&r.words, FIRST_DRAW);
            let second = command(&r.words, SECOND_DRAW);
            let past = u64::from(start) + u64::from(count) > inside;
            assert_eq!((first[0] != 0, second[0] != 0), (!past, past));
            if past {
                // The reservation starts at the counter (8 bytes) and takes whole words.
                assert_eq!(second[2], 8 / size as u32);
                let taken = (u64::from(count) * size).next_multiple_of(4) as u32;
                assert_eq!(r.pool[0], 8 + taken);
            }
            assert!(!r.report.index_pool_overflow);
        }
    }
}

/// A draw past the buffer reads the last whole index of a word the buffer's indices end inside,
/// not 0: an odd count of 16-bit indices from a multiple of 4 bytes, or an even count from 2
/// past one. The indices' range ends at a whole word, and the bytes past their end in it (the
/// `VkBuffer`'s padding) are no position's index.
#[test]
fn a_draw_past_the_buffer_reads_the_last_index_of_a_partial_word() {
    let indices = [1, 2, 3, 4, 5, 6];
    for (offset, whole) in [(0, 5), (2, 4), (2, 5), (4, 1)] {
        let mut buffer = vec![0xAB; offset as usize];
        buffer.extend(index_bytes(&indices[..whole], false));
        let end = buffer.len() as u64;
        let vk_end = end.next_multiple_of(16);
        buffer.resize(vk_end as usize, 0x5A);
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, 0),
            2,
            offset,
            end,
            64,
            1 << 27,
            POOL_BYTES,
        )
        .unwrap();
        assert_eq!(
            pass.indices_range.map(|r| r.range),
            Some(end.next_multiple_of(4)),
            "from {offset}: whole words"
        );
        let inside = whole as u64;
        for (count, start) in [(whole as u32 + 2, 0), (3, whole as u32 - 1)] {
            let r = ran(&pass, &arguments(count, start, 100), &buffer, 0);
            assert_eq!(
                drawn(&r, &buffer, offset, vk_end, false),
                metal(&indices[..whole], inside, start, count, 100),
                "{whole} from byte {offset}: {count} from {start}"
            );
        }
    }
}

/// From an index offset at or past the buffer's length every position is past it: index 0 plus
/// the base vertex at each, from the pool.
#[test]
fn an_offset_past_the_buffer_draws_index_zero_at_every_position() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, true);
    for offset in [16, 32] {
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, 0),
            4,
            offset,
            16,
            64,
            1 << 27,
            POOL_BYTES,
        )
        .unwrap();
        let r = ran(&pass, &arguments(3, 1, 100), &buffer, 0);
        assert_eq!(drawn(&r, &buffer, 0, 16, true), [100, 100, 100], "{offset}");
        let r = ran(&pass, &arguments(0, 0, 100), &buffer, 0);
        assert_eq!(drawn(&r, &buffer, 0, 16, true), Vec::<i64>::new());
        assert_eq!(r.pool[0], 0, "a draw of no index reserves nothing");
    }
}

/// A start whose positions wrap past `2^32` reads no index at the wrapped positions: they lie
/// past the buffer, as Metal's 64-bit positions do.
#[test]
fn positions_past_two_to_the_32_read_index_zero() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, true);
    let pass =
        IndexBoundsPass::new(IndirectRef::new(0, 0), 4, 0, 16, 64, 1 << 27, POOL_BYTES).unwrap();
    let start = u32::MAX - 1;
    let r = ran(&pass, &arguments(4, start, 100), &buffer, 0);
    assert_eq!(drawn(&r, &buffer, 0, 16, true), [100; 4]);
}

/// The arguments at a residual inside their bound range are read from there.
#[test]
fn the_arguments_are_read_at_their_residual() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, true);
    let pass =
        IndexBoundsPass::new(IndirectRef::new(0, 12), 4, 0, 16, 16, 1 << 27, POOL_BYTES).unwrap();
    assert_eq!(pass.parameters[0], 3);
    let mut bound = vec![0xBBBB_BBBB; 3];
    bound.extend(arguments(2, 3, 10));
    let r = ran(&pass, &bound, &buffer, 0);
    assert_eq!(drawn(&r, &buffer, 0, 16, true), [14, 10]);
}

/// A draw the pool cannot hold draws nothing, leaves the counter, and sets
/// `INDEX_POOL_OVERFLOW`; one that fits exactly is drawn.
#[test]
fn a_draw_the_pool_cannot_hold_draws_nothing_and_says_so() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, true);
    let pass =
        IndexBoundsPass::new(IndirectRef::new(0, 0), 4, 0, 16, 64, 1 << 27, POOL_BYTES).unwrap();
    // 48 of 64 bytes taken: 4 32-bit indices fit, 5 do not.
    let r = ran(&pass, &arguments(4, 2, 0), &buffer, 48);
    assert_eq!(drawn(&r, &buffer, 0, 16, true), [3, 4, 0, 0]);
    assert_eq!(r.pool[0], 64);
    assert!(!r.report.index_pool_overflow);
    let r = ran(&pass, &arguments(5, 2, 0), &buffer, 48);
    assert_eq!(drawn(&r, &buffer, 0, 16, true), Vec::<i64>::new());
    assert_eq!(r.pool[0], 48, "the counter stays");
    assert!(r.report.index_pool_overflow);
    assert!(!r.report.replay_fault);
    let r = ran(&pass, &arguments(u32::MAX, 2, 0), &buffer, 0);
    assert_eq!(drawn(&r, &buffer, 0, 16, true), Vec::<i64>::new());
    assert!(r.report.index_pool_overflow);
}

/// The fill writes only its reservation's words: the pool's other bytes keep theirs.
#[test]
fn the_fill_writes_only_its_reservation() {
    let indices = [1, 2, 3, 4];
    let buffer = index_bytes(&indices, false);
    let pass =
        IndexBoundsPass::new(IndirectRef::new(0, 0), 2, 0, 8, 64, 1 << 27, POOL_BYTES).unwrap();
    // Three 16-bit positions: two words, the second's high half 0.
    let r = ran(&pass, &arguments(3, 3, 0), &buffer, 4);
    let header = (IndexBoundsPass::POOL_HEADER_BYTES / 4) as usize;
    assert_eq!(r.pool[header], 0xDDDD_DDDD, "before the reservation");
    assert_eq!(r.pool[header + 1], 4, "position 3, then position 4 (past)");
    assert_eq!(
        r.pool[header + 2],
        0,
        "position 5 (past), then a half past the count"
    );
    assert_eq!(r.pool[header + 3], 0xDDDD_DDDD, "after it");
    assert_eq!(r.pool[0], 12);
}
