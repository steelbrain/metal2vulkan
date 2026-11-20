//! A run of the index bounds pass: both run modules valid on every committed profile (L3,
//! `G-val`), and, on the CPU interpreter, a run's one plan and one fill leave each draw the two
//! draws, the pool and the status its own pass leaves when the passes record one after another:
//! draws inside the buffer, past it, from an offset past it, of no position, and past the pool's
//! room, of either index size, in either invocation order; and a run refuses what it cannot take.

use super::super::{fill_typed_module, plan_typed_module, POOL, SOURCE, STATUS, WORDS};
use super::*;
use crate::bind::status::ExecutionStatus;
use crate::spirv::interp::{run as interpret, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The storage buffer offset alignment of these tests.
const ALIGNMENT: u64 = 16;
/// The pool's index bytes in these tests.
const POOL_BYTES: u32 = 96;

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

/// `module` over `buffers` (at `WORDS`, `SOURCE`, `POOL` and, for the plan, `STATUS`) with
/// `workgroups` along x and y, in both invocation orders, which must agree: the buffers after.
fn dispatch(module: &TypedModule, buffers: Vec<Vec<u8>>, workgroups: [u32; 2]) -> Vec<Vec<u8>> {
    let mut after = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: [WORDS, SOURCE, POOL, STATUS]
                .into_iter()
                .zip(buffers.clone())
                .map(|(slot, bytes)| Buffer {
                    slot: Some(slot),
                    address: None,
                    bytes,
                })
                .collect(),
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [workgroups[0], workgroups[1], 1],
            ..Config::default()
        };
        after.push(interpret(module, &cfg, res).expect("run").buffers);
    }
    assert_eq!(after[0], after[1], "order independent");
    after.swap_remove(0)
}

/// `range` of `buffer`, as Vulkan binds it.
fn bound(buffer: &[u8], range: Option<BufferRange>) -> Vec<u8> {
    match range {
        Some(range) => {
            buffer[range.offset as usize..(range.offset + range.range) as usize].to_vec()
        }
        None => vec![0xCC; 4],
    }
}

/// What the draws left: each draw's two commands, the pool's words and the status bytes.
type Left = (Vec<[u32; 10]>, Vec<u32>, Vec<u8>);

/// The draws' own passes, one after another over one pool: what they left.
fn passes(draws: &[BoundsRunDraw], arguments: &[u8], indices: &[u8]) -> Left {
    let header = (IndexBoundsPass::POOL_HEADER_BYTES / 4) as usize;
    let mut pool = bytes(&vec![0; header + (POOL_BYTES / 4) as usize]);
    let mut status = ExecutionStatus::initial().to_vec();
    let (plan, fill) = (
        plan_typed_module().expect("plan"),
        fill_typed_module().expect("fill"),
    );
    let mut commands = Vec::new();
    for draw in draws {
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, draw.arguments_offset),
            draw.index_size,
            draw.index_offset,
            draw.index_end,
            ALIGNMENT,
            u32::MAX,
            POOL_BYTES,
        )
        .expect("pass");
        let mut words_of = pass.parameters.to_vec();
        words_of.resize((IndexBoundsPass::BYTES / 4) as usize, 0);
        let after = dispatch(
            &plan,
            vec![
                bytes(&words_of),
                bound(arguments, Some(pass.arguments_range)),
                pool,
                status,
            ],
            [1, 1],
        );
        let planned = words(&after[0]);
        (pool, status) = (after[2].clone(), after[3].clone());
        let groups = planned[Output::FillDispatch as usize];
        if groups > 0 {
            let filled = dispatch(
                &fill,
                vec![after[0].clone(), bound(indices, pass.indices_range), pool],
                [groups, 1],
            );
            pool = filled[2].clone();
        }
        let at = (super::super::tests::FIRST_DRAW / 4) as usize;
        commands.push(planned[at..at + 10].try_into().expect("commands"));
    }
    (commands, words(&pool), status)
}

/// The draws as one run whose fill takes at most `widest` workgroups along x: what its plan and
/// fill left, and its fill's workgroups along x and y and in all.
fn run_of(
    draws: &[BoundsRunDraw],
    (arguments, indices): (&[u8], &[u8]),
    widest: u32,
) -> (Left, [u32; 3]) {
    let mut run = IndexBoundsRun::new(
        (0, arguments.len() as u64),
        Some((0, indices.len() as u64)),
        (ALIGNMENT, u32::MAX, widest),
        POOL_BYTES,
    )
    .expect("run");
    for (i, draw) in draws.iter().enumerate() {
        assert_eq!(run.push(*draw), Some(i as u32));
    }
    let header = (IndexBoundsPass::POOL_HEADER_BYTES / 4) as usize;
    let pool = bytes(&vec![0; header + (POOL_BYTES / 4) as usize]);
    let after = dispatch(
        &plan_module_of(true).expect("plan"),
        vec![
            run.words(),
            bound(arguments, Some(run.arguments_range())),
            pool,
            ExecutionStatus::initial().to_vec(),
        ],
        [1, 1],
    );
    let planned = words(&after[0]);
    let at = (IndexBoundsRun::DISPATCH_OFFSET / 4) as usize;
    let (width, rows) = (planned[at], planned[at + 1]);
    assert_eq!(planned[at + 2], 1);
    let mut pool = after[2].clone();
    if width * rows > 0 {
        let filled = dispatch(
            &fill_module_of(true).expect("fill"),
            vec![after[0].clone(), bound(indices, run.indices_range()), pool],
            [width, rows],
        );
        pool = filled[2].clone();
    }
    // The draws' fill workgroups in all: the last draw's first, and its own.
    let last = (IndexBoundsRun::HEADER_WORDS
        + (draws.len() as u32 - 1) * IndexBoundsRun::SLOT_WORDS) as usize;
    let total =
        planned[last + FIRST_GROUP as usize] + planned[last + Output::FillDispatch as usize];
    let commands = (0..draws.len() as u32)
        .map(|slot| {
            let at = (IndexBoundsRun::HEADER_WORDS + slot * IndexBoundsRun::SLOT_WORDS) as usize
                + (super::super::tests::FIRST_DRAW / 4) as usize;
            planned[at..at + 10].try_into().expect("commands")
        })
        .collect();
    (
        (commands, words(&pool), after[3].clone()),
        [width, rows, total],
    )
}

#[test]
fn both_run_modules_are_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(plan_module_of(true).expect("plan"), &p);
        assert_valid(fill_module_of(true).expect("fill"), &p);
    }
}

/// A run's draws are their own passes' draws, reservations and status, in turn, its fill
/// dispatched along x alone or spread over y.
#[test]
fn a_run_leaves_what_its_draws_passes_leave() {
    for size in [2u64, 4] {
        // 37 indices after 4 bytes of 0xAB, padded to whole words of 0xCC.
        let mut indices = vec![0xAB; 4];
        for i in 0..37u32 {
            let value = (i * 7 + 3) % 50 + 100 * (size as u32 - 2);
            match size {
                4 => indices.extend(value.to_le_bytes()),
                _ => indices.extend((value as u16).to_le_bytes()),
            }
        }
        let end = indices.len() as u64;
        indices.resize(indices.len().next_multiple_of(16), 0xCC);
        // Each draw's arguments: count, instances, first index, base vertex, base instance.
        let calls: [(u64, [u32; 5]); 8] = [
            // Inside the buffer.
            (4, [6, 1, 0, 3, 0]),
            // Past it, reserved and filled.
            (4, [10, 2, 30, 0, 1]),
            // From an offset past the buffer.
            (end.next_multiple_of(size) + 4 * size, [5, 1, 0, 0, 0]),
            // No position.
            (4, [0, 1, 3, 0, 0]),
            // Inside, from an offset.
            (4 + 10 * size, [9, 1, 2, 0, 0]),
            // Past the pool's room by now.
            (4, [40, 1, 0, 0, 0]),
            // Past the buffer again, after the overflow.
            (4 + 2 * size, [7, 1, 31, 5, 0]),
            // Inside, last.
            (4, [3, 1, 33, 0, 0]),
        ];
        let mut arguments = vec![0xEE; 12];
        let mut draws = Vec::new();
        for (index_offset, call) in calls {
            draws.push(BoundsRunDraw {
                arguments_offset: arguments.len() as u64,
                index_size: size,
                index_offset,
                index_end: end,
            });
            arguments.extend(bytes(&call));
        }
        let want = passes(&draws, &arguments, &indices);
        let mut spread = false;
        for widest in [u32::MAX, 2, 1] {
            let (got, [width, rows, total]) = run_of(&draws, (&arguments, &indices), widest);
            let case = format!("{size}-byte indices, at most {widest} along x");
            assert_eq!(got.0, want.0, "{case}: the draws");
            assert_eq!(got.1, want.1, "{case}: the pool");
            assert_eq!(got.2, want.2, "{case}: the status");
            // The dispatch holds every workgroup within the widest, and fewer than a row more.
            assert!(width <= widest && width * rows >= total && width * (rows - 1) < total);
            spread |= rows > 1 && width * rows > total;
            // The overflow happened, and both kinds of draw are among them.
            let report = ExecutionStatus::decode(&got.2[..32].try_into().expect("status"));
            assert!(report.index_pool_overflow, "{case}: no overflow");
            assert!(got.0.iter().any(|c| c[0] > 0) && got.0.iter().any(|c| c[5] > 0));
        }
        // Some dispatch spread over y, with a workgroup past the last draw's.
        assert!(spread, "{size}-byte indices: no spread dispatch");
    }
}

/// A run takes no draw whose own pass would not be made, nor one whose arguments or indices lie
/// outside its bound bytes.
#[test]
fn a_run_refuses_what_it_cannot_take() {
    let draw = |arguments_offset, index_offset| BoundsRunDraw {
        arguments_offset,
        index_size: 2,
        index_offset,
        index_end: 128,
    };
    let mut run = IndexBoundsRun::new((64, 128), Some((64, 128)), (ALIGNMENT, u32::MAX, 65535), 64)
        .expect("run");
    assert_eq!(run.push(draw(64, 64)), Some(0));
    assert_eq!(run.push(draw(66, 64)), None, "arguments off a word");
    assert_eq!(
        run.push(draw(32, 64)),
        None,
        "arguments before the bound bytes"
    );
    assert_eq!(
        run.push(draw(112, 64)),
        None,
        "arguments past the bound bytes"
    );
    assert_eq!(run.push(draw(64, 65)), None, "an odd index offset");
    assert_eq!(
        run.push(draw(64, 32)),
        None,
        "indices before the bound bytes"
    );
    assert_eq!(
        run.push(BoundsRunDraw {
            index_end: 256,
            ..draw(64, 64)
        }),
        None,
        "indices past the bound bytes"
    );
    assert_eq!(run.len(), 1);
    assert_eq!(
        IndexBoundsRun::new((8, 8), None, (ALIGNMENT, u32::MAX, 65535), 64),
        None
    );
    assert_eq!(
        IndexBoundsRun::new((8, 64), None, (ALIGNMENT, u32::MAX, 0), 64),
        None,
        "no workgroup along x"
    );
}
