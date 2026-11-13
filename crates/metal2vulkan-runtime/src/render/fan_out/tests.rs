use super::{parts, Flagged, Part, Probed, MOST_RUNS};
use metal2vulkan::bind::fan_out::{
    BLOCK_TRIANGLES, PROBE_BLOCKS, PROBE_COUNT_WORDS, PROBE_DRAW_WORDS,
};

/// A part of `count` triangles from `first`.
fn part(first: u32, count: u32, fanned: bool) -> Part {
    Part {
        first,
        count,
        fanned,
    }
}

/// The probe's words of a draw whose triangles `flagged` it flagged: the first, the last and their
/// blocks' bits.
fn flagged_of(flagged: &[u32]) -> Flagged {
    let mut blocks = [0u32; PROBE_COUNT_WORDS as usize];
    for &t in flagged {
        let block = (t / BLOCK_TRIANGLES).min(PROBE_BLOCKS - 1);
        blocks[(block / 32) as usize] |= 1 << (block % 32);
    }
    Flagged {
        first: flagged.iter().copied().min().unwrap_or(1),
        last: flagged.iter().copied().max().unwrap_or(0),
        blocks,
    }
}

/// A flagged draw's parts take each of its triangles once, in order: its runs of blocks that hold
/// a flagged triangle region by region, cut to the first and the last flagged one, runs fewer than
/// `RUN_GAP` triangles apart joined and at most [`MOST_RUNS`] of them, and the others fanned out;
/// a 16-bit list's parts begin at even triangles; a flag past the triangles draws them all region
/// by region, and so does a flag without a block's bit.
#[test]
fn a_flagged_draw_s_parts_take_each_triangle_once_in_order() {
    assert_eq!(
        parts(&flagged_of(&[5, 7]), 20, false),
        [part(0, 5, true), part(5, 3, false), part(8, 12, true)]
    );
    assert_eq!(
        parts(&flagged_of(&[5, 7]), 20, true),
        [part(0, 4, true), part(4, 4, false), part(8, 12, true)]
    );
    assert_eq!(
        parts(&flagged_of(&[0, 19]), 20, false),
        [part(0, 20, false)]
    );
    assert_eq!(parts(&flagged_of(&[0]), 1, true), [part(0, 1, false)]);
    assert_eq!(
        parts(&flagged_of(&[18]), 19, true),
        [part(0, 18, true), part(18, 1, false)]
    );
    assert_eq!(parts(&flagged_of(&[]), 20, false), [part(0, 20, false)]);
    assert_eq!(
        parts(&flagged_of(&[3, 20]), 20, false),
        [part(0, 20, false)]
    );
    let no_bits = Flagged {
        blocks: [0; PROBE_COUNT_WORDS as usize],
        ..flagged_of(&[3, 5])
    };
    assert_eq!(parts(&no_bits, 20, false), [part(0, 20, false)]);
    // Far apart, two runs of their blocks with the triangles between fanned out; near, one.
    assert_eq!(
        parts(&flagged_of(&[100, 1000]), 2000, false),
        [
            part(0, 100, true),
            part(100, 28, false),
            part(128, 832, true),
            part(960, 41, false),
            part(1001, 999, true),
        ]
    );
    assert_eq!(
        parts(&flagged_of(&[100, 200]), 2000, false),
        [
            part(0, 100, true),
            part(100, 101, false),
            part(201, 1799, true)
        ]
    );
    // Past the blocks, the last block's.
    assert_eq!(
        parts(&flagged_of(&[70_000]), 80_000, false),
        [
            part(0, 70_000, true),
            part(70_000, 1, false),
            part(70_001, 9_999, true)
        ]
    );
    let mut random = 7u64;
    let mut next = |below: u32| {
        random = random
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((random >> 33) % u64::from(below)) as u32
    };
    for _ in 0..3000 {
        let triangles = 1 + next(4000);
        let flagged: Vec<u32> = (0..1 + next(12)).map(|_| next(triangles)).collect();
        let (first, last) = (flagged.iter().min().unwrap(), flagged.iter().max().unwrap());
        for even in [false, true] {
            let drawn = parts(&flagged_of(&flagged), triangles, even);
            let mut at = 0;
            for p in &drawn {
                assert_eq!(p.first, at, "{drawn:?}");
                assert!(p.count > 0, "{drawn:?}");
                assert!(!even || p.first.is_multiple_of(2), "{drawn:?}");
                let holds = |t: &u32| (p.first..p.first + p.count).contains(t);
                assert!(!p.fanned || !flagged.iter().any(holds), "{drawn:?}");
                // No part region by region past the first and the last flagged triangle.
                assert!(
                    p.fanned
                        || (p.first >= first & !1
                            && p.first + p.count <= (last + 2).min(triangles)),
                    "{flagged:?} {drawn:?}"
                );
                at += p.count;
            }
            assert_eq!(at, triangles, "{drawn:?}");
            let runs = drawn.iter().filter(|p| !p.fanned).count();
            assert!((1..=MOST_RUNS).contains(&runs), "{drawn:?}");
        }
    }
}

/// A probed draw's words: the counts summed, a wide mark, the first and last flagged triangle
/// from the least and the greatest of their raised words, the flagged counts summed, and the
/// blocks' bits.
#[test]
fn a_probed_draw_s_words_name_its_first_and_last_flagged_triangle() {
    let lanes = PROBE_COUNT_WORDS as usize;
    let mut words = vec![0u32; PROBE_DRAW_WORDS as usize];
    words[1] = 3;
    words[1 + lanes - 1] = 4;
    assert_eq!(
        Probed::of(&words),
        Probed {
            copies: 7,
            wide: false,
            flagged: None,
            flagged_triangles: 0,
        }
    );
    words[0] = 1;
    words[1 + lanes + 2] = 1;
    // Triangles 9 and 40 flagged, in their lanes.
    words[1 + 2 * lanes + 9] = u32::MAX - 9;
    words[1 + 2 * lanes + 8] = u32::MAX - 40;
    words[1 + 3 * lanes + 9] = 10;
    words[1 + 3 * lanes + 8] = 41;
    // Their counts, and a third flagged triangle's in triangle 9's lane.
    words[1 + 4 * lanes + 9] = 2;
    words[1 + 4 * lanes + 8] = 1;
    // Their blocks' bits: 9's and 40's block 0, and a third's block 33.
    words[1 + 5 * lanes] = 1;
    words[1 + 5 * lanes + 1] = 2;
    let mut blocks = [0u32; PROBE_COUNT_WORDS as usize];
    (blocks[0], blocks[1]) = (1, 2);
    assert_eq!(
        Probed::of(&words),
        Probed {
            copies: 7,
            wide: true,
            flagged: Some(Flagged {
                first: 9,
                last: 40,
                blocks,
            }),
            flagged_triangles: 3,
        }
    );
}
