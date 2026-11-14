use super::{parts, Flagged, Part, Probed, MOST_RUNS};
use metal2vulkan::bind::fan_out::{
    BLOCK_TRIANGLES, PROBE_BLOCKS, PROBE_BOUNDS_AT, PROBE_COUNT_WORDS, PROBE_DRAW_WORDS,
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
/// from the least and the greatest of their raised words, the flagged counts summed, the
/// blocks' bounds together, and the blocks' bits.
#[test]
fn a_probed_draw_s_words_name_its_first_and_last_flagged_triangle() {
    let lanes = PROBE_COUNT_WORDS as usize;
    let mut words = vec![0u32; PROBE_DRAW_WORDS as usize];
    words[1] = 3;
    words[1 + lanes - 1] = 4;
    // Two blocks' bounds: x's buckets 1 and 0, y's 0 and 2; together x's 0 and 1, y's 0 and 2.
    words[PROBE_BOUNDS_AT as usize + 2] = 0x0001_0002;
    words[PROBE_BOUNDS_AT as usize + 5] = 0x0004_0001;
    assert_eq!(
        Probed::of(&words),
        Probed {
            copies: 7,
            wide: false,
            flagged: None,
            flagged_triangles: 0,
            straddling_triangles: 0,
            reach: 0x0005_0003,
            bounds: None,
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
    // A straddling one in triangle 40's lane, and the bounds of blocks 0 and 33.
    words[1 + 6 * lanes + 8] = 1;
    words[PROBE_BOUNDS_AT as usize] = 0x0003_0006;
    words[PROBE_BOUNDS_AT as usize + 33] = u32::MAX;
    let mut blocks = [0u32; PROBE_COUNT_WORDS as usize];
    (blocks[0], blocks[1]) = (1, 2);
    let mut bounds = vec![0u32; PROBE_BLOCKS as usize];
    (bounds[0], bounds[33]) = (0x0003_0006, u32::MAX);
    (bounds[2], bounds[5]) = (0x0001_0002, 0x0004_0001);
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
            straddling_triangles: 1,
            reach: u32::MAX,
            bounds: Some(bounds.into()),
        }
    );
}

/// A run's bounds are its blocks' bounds together, and the regions they reach the table's regions
/// whose segments' buckets they hold, as the boxes of each run of consecutive such segments of
/// each axis.
#[test]
fn a_run_reaches_the_regions_its_blocks_bounds_hold() {
    use super::{run_bounds, run_boxes, RegionBox};
    let mut bounds = vec![0u32; PROBE_BLOCKS as usize];
    // Block 0 reaches x bucket 1 and y bucket 0; block 1 x buckets 2 and 3, y bucket 1.
    (bounds[0], bounds[1]) = (0x0001_0002, 0x0002_000c);
    let run = part(10, 100, false);
    assert_eq!(run_bounds(&bounds, &run), 0x0003_000e);
    assert_eq!(run_bounds(&bounds, &part(10, 20, false)), 0x0001_0002);
    let boxed = |first, columns, rows| RegionBox {
        first,
        columns,
        rows,
    };
    // 32 segments a side, two to a bucket: x segments 2 to 7 of y segments 0 to 3.
    assert_eq!(run_boxes([32, 32], 0x0003_000e), [boxed(2, 6, 4)]);
    // Every bucket, every region, in one box.
    assert_eq!(run_boxes([5, 3], u32::MAX), [RegionBox::whole([5, 3])]);
    assert!(run_boxes([5, 3], 0).is_empty());
    // Buckets 1 and 3 of x, 0 and 2 of y: a box of each pair of runs, y's then x's.
    assert_eq!(
        run_boxes([32, 32], 0x0005_000a),
        [
            boxed(2, 2, 2),
            boxed(6, 2, 2),
            boxed(4 * 32 + 2, 2, 2),
            boxed(4 * 32 + 6, 2, 2)
        ]
    );
}

/// A box is recorded at most as many regions at a time as a command draws, its rows in order,
/// as wide as that allows and as many rows as then fit; one column at a time where its columns'
/// borders may hold a sample, one row where its rows' may; and one region at a time with one
/// region a command.
#[test]
fn a_box_is_recorded_as_many_regions_a_time_as_a_command_draws() {
    use super::{box_recordings, RegionBox};
    let boxed = |first, columns, rows| RegionBox {
        first,
        columns,
        rows,
    };
    let recorded = |whole, most, folds| box_recordings(whole, 10, most, folds).collect::<Vec<_>>();
    let whole = boxed(12, 4, 5);
    assert_eq!(recorded(whole, 2047, [true; 2]), [whole]);
    // Two rows of four at a time, the last row alone.
    assert_eq!(
        recorded(whole, 9, [true; 2]),
        [boxed(12, 4, 2), boxed(32, 4, 2), boxed(52, 4, 1)]
    );
    // Wider than a command: each row in parts.
    assert_eq!(
        recorded(boxed(0, 4, 2), 3, [true; 2]),
        [
            boxed(0, 3, 1),
            boxed(3, 1, 1),
            boxed(10, 3, 1),
            boxed(13, 1, 1)
        ]
    );
    // One column at a time, its rows together; one row at a time, its columns together.
    let columns: Vec<_> = (12..16).map(|first| boxed(first, 1, 5)).collect();
    assert_eq!(recorded(whole, 2047, [false, true]), columns);
    let rows: Vec<_> = (0..5).map(|r| boxed(12 + 10 * r, 4, 1)).collect();
    assert_eq!(recorded(whole, 2047, [true, false]), rows);
    // One region a command, or neither axis folded: each region alone, in the table's order.
    let regions: Vec<_> = (0..5)
        .flat_map(|r| (0..4).map(move |c| boxed(12 + 10 * r + c, 1, 1)))
        .collect();
    assert_eq!(recorded(whole, 1, [true; 2]), regions);
    assert_eq!(recorded(whole, 2047, [false; 2]), regions);
}

/// A box's scissor is the union of its regions', whichever way the table's order runs over the
/// pixels.
#[test]
fn a_box_s_scissor_holds_its_regions_pixels() {
    use super::RegionBox;
    let rect = |x: i32, y: i32, width, height| ash::vk::Rect2D {
        offset: ash::vk::Offset2D { x, y },
        extent: ash::vk::Extent2D { width, height },
    };
    // Three segments of x of 10, 20 and 30 pixels; two of y, the first 8 pixels below the
    // second's 4 (a flipped y).
    let xs = [(0, 10), (10, 20), (30, 30)];
    let ys = [(4, 8), (0, 4)];
    let scissors: Vec<_> = (ys.iter())
        .flat_map(|&(y, h)| xs.iter().map(move |&(x, w)| rect(x, y, w, h)))
        .collect();
    // A box's scissor as `[x, y, width, height]`, of `first`, `columns` and `rows`.
    let scissor = |first, columns, rows| {
        let boxed = RegionBox {
            first,
            columns,
            rows,
        };
        (boxed.scissor(&scissors, 3)).map(|r| {
            [
                r.offset.x,
                r.offset.y,
                r.extent.width as i32,
                r.extent.height as i32,
            ]
        })
    };
    assert_eq!(scissor(1, 2, 2), Some([10, 0, 50, 12]));
    assert_eq!(scissor(0, 1, 1), Some([0, 4, 10, 8]));
    assert_eq!(scissor(3, 3, 1), Some([0, 0, 60, 4]));
    assert_eq!(scissor(4, 2, 2), None);
}

/// A box folds its columns, and its rows, together where no sample of the pass lies on a pixel's
/// left edge, and none on its top edge: at the device's standard locations up to 8 samples, and
/// at custom positions none of which is 0 on that axis (samples past them lie at the corner).
#[test]
fn a_box_folds_where_no_sample_lies_on_its_borders() {
    use super::folds;
    for samples in [1, 2, 4, 8] {
        assert_eq!(folds(true, samples, &[]), [true; 2], "{samples}");
        assert_eq!(folds(false, samples, &[]), [false; 2], "{samples}");
    }
    assert_eq!(folds(true, 16, &[]), [false; 2]);
    assert_eq!(
        folds(true, 4, &[[8, 8], [4, 12], [12, 4], [2, 14]]),
        [true; 2]
    );
    assert_eq!(folds(true, 2, &[[0, 8], [8, 8]]), [false, true]);
    assert_eq!(folds(false, 2, &[[8, 0], [8, 8]]), [true, false]);
    // Two positions of a pass of four: the samples past them at the pixel's corner.
    assert_eq!(
        folds(true, 4, &[[8, 8], [4, 4], [0, 0], [0, 0]]),
        [false; 2]
    );
}
