//! A fill by transfers (`word_fill.rs`): exact, and every copy reads only words the fill wrote
//! before a barrier and writes words nothing in the fill wrote yet.

use super::super::word_fill::{steps, Copy, Step};

/// The fill of `words` by its steps, on a buffer of `len` bytes: every byte of `words` ends as the
/// fill's, every other byte untouched; each update writes at most 65536 bytes, a multiple of 4 at
/// a multiple of 4; each copy's source lies in words written before the barrier ahead of it, and
/// its destination in words not yet written, not even by another region of its command, so one
/// command's sources and destinations never overlap.
fn check(words: std::ops::Range<u64>, len: u64) {
    const FILL: u8 = 0xa5;
    const UNTOUCHED: u8 = 0x5a;
    let mut bytes = vec![UNTOUCHED; len as usize];
    // The bytes written, and of them those a barrier has made readable.
    let mut written = vec![false; len as usize];
    let mut visible = vec![false; len as usize];
    for step in steps(words.clone()) {
        match step {
            Step::Update(range) => {
                assert!(range.end - range.start <= 65536, "{words:?}: {range:?}");
                assert_eq!(range.start % 4, 0, "{words:?}: {range:?}");
                assert_eq!(range.end % 4, 0, "{words:?}: {range:?}");
                for at in range {
                    assert!(!written[at as usize], "{words:?}: {at} written twice");
                    bytes[at as usize] = FILL;
                    written[at as usize] = true;
                }
            }
            Step::Barrier => visible.clone_from(&written),
            Step::Copies(copies) => {
                assert!(!copies.is_empty(), "{words:?}: a command of no region");
                for Copy { from, to, len: n } in copies {
                    assert!(
                        from + n <= to || to + n <= from,
                        "{words:?}: overlapping copy"
                    );
                    for i in 0..n {
                        let (src, dst) = ((from + i) as usize, (to + i) as usize);
                        assert!(visible[src], "{words:?}: copy reads {src}, not yet visible");
                        assert!(!written[dst], "{words:?}: copy writes {dst} twice");
                        bytes[dst] = bytes[src];
                        written[dst] = true;
                    }
                }
            }
        }
    }
    for (at, &byte) in bytes.iter().enumerate() {
        let inside = words.contains(&(at as u64));
        let want = if inside { FILL } else { UNTOUCHED };
        assert_eq!(byte, want, "{words:?}: byte {at}");
    }
}

#[test]
fn a_fill_by_transfers_writes_exactly_its_words() {
    for (start, words) in [
        (0, 0),
        (0, 4),
        (8, 256),
        (4, 65532),
        (0, 65536),
        (12, 65540),
        (0, 131072),
        (16, 131076),
        (4, 3 * 65536 + 8),
        (0, 1 << 20),
        (64, (1 << 20) + 65536 * 5 + 12),
    ] {
        check(start..start + words, start + words + 16);
    }
}

#[test]
fn a_fill_by_transfers_takes_a_barrier_per_sixteenfold() {
    assert!(steps(0..0).is_empty());
    assert_eq!(steps(4..260), [Step::Update(4..260)]);
    // 1 MiB: 64 KiB updated, then fifteen copies of it in one command.
    let s = steps(0..1 << 20);
    assert_eq!(s.iter().filter(|s| **s == Step::Barrier).count(), 1);
    let copies: Vec<Copy> = (1..16)
        .map(|at| Copy {
            from: 0,
            to: at << 16,
            len: 1 << 16,
        })
        .collect();
    assert_eq!(s.last(), Some(&Step::Copies(copies)));
    // 256 MiB: to 1, 16 and 256 MiB, three barriers where doubling took twelve.
    let s = steps(0..256 << 20);
    assert_eq!(s.iter().filter(|s| **s == Step::Barrier).count(), 3);
    // A tail shorter than what is filled is the last region's.
    let s = steps(0..(1 << 20) + 12);
    assert_eq!(
        s.last(),
        Some(&Step::Copies(vec![Copy {
            from: 0,
            to: 1 << 20,
            len: 12
        }]))
    );
}
