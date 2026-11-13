use super::{parts, Part, Probed};
use metal2vulkan::bind::fan_out::{PROBE_COUNT_WORDS, PROBE_DRAW_WORDS};

/// A part of `count` triangles from `first`.
fn part(first: u32, count: u32, fanned: bool) -> Part {
    Part {
        first,
        count,
        fanned,
    }
}

/// A flagged draw's parts take each of its triangles once, in order, those from the first flagged
/// one to the last region by region and the others fanned out; a 16-bit list's parts begin at
/// even triangles; a flag past the triangles draws them all region by region.
#[test]
fn a_flagged_draw_s_parts_take_each_triangle_once_in_order() {
    assert_eq!(
        parts((5, 7), 20, false),
        [part(0, 5, true), part(5, 3, false), part(8, 12, true)]
    );
    assert_eq!(
        parts((5, 7), 20, true),
        [part(0, 4, true), part(4, 4, false), part(8, 12, true)]
    );
    assert_eq!(parts((0, 19), 20, false), [part(0, 20, false)]);
    assert_eq!(parts((0, 0), 1, true), [part(0, 1, false)]);
    assert_eq!(
        parts((18, 18), 19, true),
        [part(0, 18, true), part(18, 1, false)]
    );
    assert_eq!(parts((3, 2), 20, false), [part(0, 20, false)]);
    assert_eq!(parts((3, 20), 20, false), [part(0, 20, false)]);
    for triangles in 1..40 {
        for first in 0..triangles {
            for last in first..triangles {
                for even in [false, true] {
                    let drawn = parts((first, last), triangles, even);
                    let mut next = 0;
                    for p in &drawn {
                        assert_eq!(p.first, next, "{drawn:?}");
                        assert!(p.count > 0, "{drawn:?}");
                        assert!(!even || p.first.is_multiple_of(2), "{drawn:?}");
                        let flagged =
                            (p.first..p.first + p.count).any(|t| (first..=last).contains(&t));
                        assert!(!p.fanned || !flagged, "{drawn:?}");
                        next += p.count;
                    }
                    assert_eq!(next, triangles, "{drawn:?}");
                }
            }
        }
    }
}

/// A probed draw's words: the counts summed, a wide mark, the first and last flagged triangle
/// from the least and the greatest of their raised words, and the flagged counts summed.
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
    assert_eq!(
        Probed::of(&words),
        Probed {
            copies: 7,
            wide: true,
            flagged: Some((9, 40)),
            flagged_triangles: 3,
        }
    );
}
