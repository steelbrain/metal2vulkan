//! A memory pool derives its longest free range **once**, however many creations ask whether it
//! has room (`memory.rs`'s `Pool::room`).
//!
//! # Why a creation asks at all
//!
//! Every creation call the runtime makes that may wait -- a queue's, a pipeline state's, a
//! buffer's, a texture's, a heap's -- keeps room in the pools the encodings and commits place
//! their own resources in (`memory::keep_room`), so that no encoding makes a memory block. It
//! does that by asking each of those pools whether it has room for half a block, which is one
//! query per kept pool per creation.
//!
//! # Why this is a count, and what each end of it catches
//!
//! The answer is a fact of the pool's free ranges, not of the host: between the changes that can
//! move it -- a placement, a block added or freed, the pool's alignment growing -- every query
//! answers the same, so the pool holds the answer and derives it once. A query that derives
//! instead reads every free range of every block, work that grows with how many blocks the pool
//! has made and how fragmented they are, which is to say with the pool's history and not with the
//! creation that asked.
//!
//! So the proof is a **count** of the free ranges the derivations visited and the block slots they
//! walked, which holds on any host under any load, where a time is a measurement of this machine
//! under this load. Each row asserts both exactly, which holds the structure at both ends:
//!
//! - **Above**, a query that derives per call turns `visited` from one pool's worth of ranges into
//!   that many per query, so a memo removed or an invalidation added where nothing changed fails.
//! - **Below**, an answer held past a change never derives again, so `visited` stops moving and
//!   the row fails on the low side. [`the_held_room_is_the_room_derived_now`] is the other half of
//!   that end: it holds the answers themselves to what deriving now gives, over a run of
//!   placements, frees, blocks and alignments, so an invalidation that is merely missing is caught
//!   as a wrong answer rather than as a number.
//!
//! The counts are of one pool the test builds, so they depend on nothing that ran before them in
//! the process and need no quiet host and no ordering between tests.

use crate::memory::Pool;
use std::time::Instant;

/// The bytes of one unit the rows fill a block with: two of them a hole, so a block of
/// `2 * UNIT * holes` bytes filled by units and emptied of every other one holds `holes` free
/// ranges, no two of which touch.
const UNIT: u64 = 64;

/// What a row places between its queries: first-fit takes it from the front of the first hole of
/// the first block, and a row's whole run of them is less than one hole ([`UNIT`]), so that hole
/// never closes. The block keeps the free ranges it had and every step of a row costs the same.
const STEP: u64 = 4;

/// A pool of `pooled` pooled blocks, each holding `holes` free ranges, and `dedicated` dedicated
/// blocks of one placement each. The blocks are filled before any is emptied, so each one's units
/// are its own.
fn pool(pooled: usize, holes: u64, dedicated: usize) -> Pool<usize> {
    let size = 2 * UNIT * holes;
    let mut pool: Pool<usize> = Pool::default();
    let blocks: Vec<usize> = (0..pooled)
        .map(|index| pool.add(size, false, index, 0))
        .collect();
    for _ in 0..pooled {
        for _ in 0..2 * holes {
            assert!(pool.place(UNIT, 1).is_some(), "a unit of a filled block");
        }
    }
    let now = Instant::now();
    for &block in &blocks {
        for hole in 0..holes {
            assert_eq!(pool.release(block, 2 * UNIT * hole, UNIT, now), None);
        }
    }
    for index in 0..dedicated {
        pool.add(UNIT, true, pooled + index, UNIT);
    }
    pool
}

/// One row: a pool, how many creations ask it for room, whether each places first, and the block
/// slots and free ranges that must cost.
struct Row {
    /// The shape of the pool: pooled blocks, free ranges each, dedicated blocks.
    shape: (usize, u64, usize),
    /// How many creations ask whether the pool has room.
    queries: u64,
    /// Whether each creation places [`STEP`] bytes in the pool before it asks, as a buffer's or a
    /// texture's creation does and a pipeline state's does not.
    places: bool,
    /// The block slots the derivations must walk, and the free ranges they must visit.
    walked: u64,
    visited: u64,
}

/// Every pool shape and reader. The first three ask without placing, which is the pipeline
/// state's shape: one derivation however many ask, so the counts do not move with `queries`. The
/// last three place first, which is the buffer's and the texture's: the pool's own answer must be
/// derived again, but only the one block whose ranges moved derives its longest free range, so
/// each creation costs that block's free ranges and never the pool's.
const ROWS: &[Row] = &[
    // One block of 8 free ranges, asked once and asked a hundred times over: the same.
    Row {
        shape: (1, 8, 0),
        queries: 1,
        places: false,
        walked: 1,
        visited: 8,
    },
    Row {
        shape: (1, 8, 0),
        queries: 100,
        places: false,
        walked: 1,
        visited: 8,
    },
    // Sixteen blocks of 32 free ranges each and 8 dedicated blocks, asked a hundred times over:
    // 512 ranges once, not 51200. The dedicated blocks are walked and hold no free range of the
    // pool's.
    Row {
        shape: (16, 32, 8),
        queries: 100,
        places: false,
        walked: 24,
        visited: 512,
    },
    // Placing first: the first query derives all 8 ranges, and each of the next 9 derives the 8
    // of the one block that was placed in.
    Row {
        shape: (1, 8, 0),
        queries: 10,
        places: true,
        walked: 10,
        visited: 80,
    },
    // The same over sixteen blocks of 32: each creation derives one block's 32 ranges, not the
    // pool's 512.
    Row {
        shape: (16, 32, 0),
        queries: 10,
        places: true,
        walked: 160,
        visited: 512 + 9 * 32,
    },
    Row {
        shape: (16, 32, 8),
        queries: 10,
        places: true,
        walked: 240,
        visited: 512 + 9 * 32,
    },
];

/// However many creations ask a pool for room, it derives its longest free range once, and a
/// creation that places in it derives the one block's that moved.
#[test]
fn room_is_derived_once_however_many_creations_ask_for_it() {
    for (index, row) in ROWS.iter().enumerate() {
        let (pooled, holes, dedicated) = row.shape;
        let mut pool = pool(pooled, holes, dedicated);
        let asked = UNIT;
        for _ in 0..row.queries {
            if row.places {
                assert!(pool.place(STEP, 1).is_some(), "a step of a hole");
            }
            assert!(
                pool.has_room(asked),
                "an untouched hole holds {asked} bytes"
            );
        }
        let what = format!(
            "row {index}: {pooled} blocks of {holes} ranges, {dedicated} dedicated, \
             {} queries{}",
            row.queries,
            if row.places {
                " each placing first"
            } else {
                ""
            },
        );
        assert_eq!(pool.room_work(), (row.walked, row.visited), "{what}");
    }
}

/// The room a pool holds is the room derived now, through every change that can move it: a
/// placement, a placement that fails, a range freed, a block added, a block freed, and the pool's
/// alignment growing. An invalidation that is missing shows here as a wrong answer.
#[test]
fn the_held_room_is_the_room_derived_now() {
    let now = Instant::now();
    let mut pool: Pool<usize> = Pool::default();
    let check = |pool: &mut Pool<usize>, what: &str| {
        let derived = pool.room_derived_now();
        for size in [1, UNIT / 2, UNIT, 2 * UNIT, 4 * UNIT, 1 << 20] {
            assert_eq!(
                pool.has_room(size),
                derived >= size,
                "{what}: room for {size} bytes of {derived}",
            );
        }
    };
    check(&mut pool, "an empty pool");
    let first = pool.add(4 * UNIT, false, 0, 0);
    check(&mut pool, "a block added");
    assert_eq!(pool.place(UNIT, 1), Some((first, 0)));
    check(&mut pool, "a unit placed");
    assert_eq!(pool.place(UNIT, 2 * UNIT), Some((first, 2 * UNIT)));
    check(&mut pool, "a unit placed at a larger alignment");
    assert_eq!(pool.place(4 * UNIT, 1), None, "the block holds no more");
    check(&mut pool, "a placement that fails");
    assert_eq!(pool.release(first, 0, UNIT, now), None);
    check(&mut pool, "a unit returned");
    let second = pool.add(8 * UNIT, false, 1, 0);
    check(&mut pool, "a second block added");
    let dedicated = pool.add(1 << 20, true, 2, 1 << 20);
    check(&mut pool, "a dedicated block added");
    assert_eq!(pool.release(dedicated, 0, 1 << 20, now), Some(2));
    check(&mut pool, "the dedicated block freed");
    assert_eq!(pool.place(8 * UNIT, 1), Some((second, 0)));
    check(&mut pool, "the second block filled");
    assert_eq!(pool.release(second, 0, 8 * UNIT, now), None);
    check(&mut pool, "the second block emptied and kept");
    assert_eq!(pool.release(first, 2 * UNIT, UNIT, now), None);
    check(&mut pool, "the first block emptied and kept");
    assert_eq!(pool.take_kept().len(), 2, "both kept blocks go");
    check(&mut pool, "every block freed");
}
