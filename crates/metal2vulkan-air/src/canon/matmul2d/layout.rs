//! Metal's cooperative-tensor layouts, as measured.
//!
//! A cooperative tensor's elements are spread over the scope's threads in 8 x 8 blocks. Within a
//! block, lane `L` of a simdgroup holds two adjacent elements, at block row `row(L)` and block
//! columns `col(L)` and `col(L) + 1`:
//!
//! ```text
//! row(L) = ((L >> 1) & 3) | ((L >> 4) << 2)        col(L) = ((L >> 3) & 1) * 4 + (L & 1) * 2
//! ```
//!
//! Element `e` of a thread is the `e % 2`th of block `p = e / 2` of its simdgroup `s`. The blocks
//! of the operand form a grid of `R x C` blocks (the destination: `M / 8` rows of `N / 8`
//! columns; the right input: `K / 8` rows of `N / 8` columns; the left input: `M / 8` rows of
//! `K / 8` columns), split among the scope's
//! simdgroups as `SR x SC` interleaved: simdgroup `s` holds the blocks whose row is `s / SC`
//! modulo `SR` and whose column is `s % SC` modulo `SC`, in row-major order. So
//!
//! ```text
//! block row = SR * (p / (C / SC)) + s / SC        block column = SC * (p % (C / SC)) + s % SC
//! ```
//!
//! and a thread holds `2 * (R / SR) * (C / SC)` elements, contiguously, `e` at byte
//! `e * size(element)`. The destination's element at row `m` and column `n` has the coordinates
//! `(n, m)`; the right input's at row `k` and column `n` has `(n, k)`, or `(k, n)` when the
//! right input is transposed; the left input's at row `m` and column `k` has `(k, m)`, whichever
//! the right input's transpose (a transposed left input was not measured).
//!
//! The split is measured per shape, not derived: only the shapes below are modeled, and any other
//! is refused. Every layout probe found every element of every thread valid.
//!
//! The arithmetic is written once over [`Arith`], which the lowering emits as instructions and
//! the tests evaluate, so what the tests check against the readings is what the lowering emits.

use crate::error::Error;
use crate::ir::BinOp;
use crate::tensor_ops::{Descriptor, Inner};

/// `u32` arithmetic on values of type [`Arith::V`]: emitted as instructions, or evaluated.
pub(super) trait Arith {
    type V: Copy;
    fn k(&mut self, v: u32) -> Result<Self::V, Error>;
    fn op(&mut self, op: BinOp, a: Self::V, b: Self::V) -> Result<Self::V, Error>;

    fn opk(&mut self, op: BinOp, a: Self::V, k: u32) -> Result<Self::V, Error> {
        let k = self.k(k)?;
        self.op(op, a, k)
    }
}

/// The block row of lane `lane`'s elements and the first of its two block columns.
pub(super) fn lane_place<A: Arith>(a: &mut A, lane: A::V) -> Result<(A::V, A::V), Error> {
    // row = ((lane >> 1) & 3) | ((lane >> 4) << 2)
    let x = a.opk(BinOp::LShr, lane, 1)?;
    let x = a.opk(BinOp::And, x, 3)?;
    let y = a.opk(BinOp::LShr, lane, 4)?;
    let y = a.opk(BinOp::Shl, y, 2)?;
    let row = a.op(BinOp::Or, x, y)?;
    // col = ((lane >> 3) & 1) * 4 + (lane & 1) * 2
    let x = a.opk(BinOp::LShr, lane, 3)?;
    let x = a.opk(BinOp::And, x, 1)?;
    let x = a.opk(BinOp::Shl, x, 2)?;
    let y = a.opk(BinOp::And, lane, 1)?;
    let y = a.opk(BinOp::Shl, y, 1)?;
    let col = a.op(BinOp::Or, x, y)?;
    Ok((row, col))
}

/// Which operand a cooperative tensor holds, as the layouts distinguish them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Holds {
    Left,
    Destination,
    Right,
}

/// One cooperative tensor's measured layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Layout {
    /// The block grid's columns of 8 x 8 blocks.
    cols: u32,
    /// The block grid's rows.
    rows: u32,
    /// The simdgroup split: `SR x SC`.
    split: (u32, u32),
    /// Whether the coordinates are `(row, column)` rather than `(column, row)` (a transposed right
    /// input).
    pub row_first: bool,
}

/// The measured shapes: `(m, n, k, simdgroups)` and the destination's split.
const DESTINATIONS: &[((u32, u32, u32, u32), (u32, u32))] = &[
    ((16, 16, 32, 1), (1, 1)),
    ((32, 32, 32, 1), (1, 1)),
    ((16, 64, 16, 2), (1, 2)),
    ((64, 64, 32, 4), (2, 2)),
    ((128, 128, 32, 4), (2, 2)),
    ((8, 128, 32, 4), (1, 4)),
    ((16, 32, 32, 1), (1, 1)),
];

/// The measured right-input shapes: `(m, n, k, simdgroups)`, all one simdgroup.
const RIGHTS: &[(u32, u32, u32, u32)] = &[(16, 16, 32, 1), (32, 32, 32, 1), (16, 32, 32, 1)];

/// The measured left-input shapes: `(m, n, k, simdgroups)`, all one simdgroup, the left input
/// not transposed.
const LEFTS: &[(u32, u32, u32, u32)] = &[(16, 32, 32, 1), (32, 32, 32, 1)];

impl Layout {
    /// The layout of `holds` for descriptor `d` over `simdgroups`, when it is measured.
    pub fn of(holds: Holds, d: &Descriptor, simdgroups: u32) -> Option<Layout> {
        let Inner::Static(k) = d.k else {
            return None;
        };
        let shape = (d.m, d.n, k, simdgroups);
        let (rows, cols, split) = match holds {
            Holds::Destination => {
                let (_, split) = DESTINATIONS.iter().find(|(s, _)| *s == shape)?;
                (d.m / 8, d.n / 8, *split)
            }
            Holds::Right => {
                RIGHTS.iter().find(|s| **s == shape)?;
                (k / 8, d.n / 8, (1, 1))
            }
            Holds::Left => {
                if d.transpose_left {
                    return None;
                }
                LEFTS.iter().find(|s| **s == shape)?;
                (d.m / 8, k / 8, (1, 1))
            }
        };
        Some(Layout {
            cols,
            rows,
            split,
            row_first: holds == Holds::Right && d.transpose_right,
        })
    }

    /// Elements per thread.
    pub fn capacity(&self) -> u32 {
        2 * (self.rows / self.split.0) * (self.cols / self.split.1)
    }

    /// The row and column of element `e` of the lane at block place `(lane_row, lane_col)`
    /// ([`lane_place`]) in simdgroup `sg`.
    pub fn place<A: Arith>(
        &self,
        a: &mut A,
        e: A::V,
        (lane_row, lane_col): (A::V, A::V),
        sg: A::V,
    ) -> Result<(A::V, A::V), Error> {
        let (sr, sc) = self.split;
        let per_row = self.cols / sc;
        let p = a.opk(BinOp::LShr, e, 1)?;
        let j = a.opk(BinOp::And, e, 1)?;
        let q = a.opk(BinOp::UDiv, p, per_row)?;
        let q = a.opk(BinOp::Mul, q, sr)?;
        let s_row = a.opk(BinOp::UDiv, sg, sc)?;
        let block_row = a.op(BinOp::Add, q, s_row)?;
        let r = a.opk(BinOp::URem, p, per_row)?;
        let r = a.opk(BinOp::Mul, r, sc)?;
        let s_col = a.opk(BinOp::URem, sg, sc)?;
        let block_col = a.op(BinOp::Add, r, s_col)?;
        let row = a.opk(BinOp::Shl, block_row, 3)?;
        let row = a.op(BinOp::Add, row, lane_row)?;
        let col = a.opk(BinOp::Shl, block_col, 3)?;
        let col = a.op(BinOp::Add, col, lane_col)?;
        let col = a.op(BinOp::Add, col, j)?;
        Ok((row, col))
    }

    /// Where a one-simdgroup right layout ([`Layout::of`] of [`Holds::Right`]) holds row `k` and
    /// column `n`, where `n` is a column lane `lane` holds in its own blocks (`n % 8` is
    /// `col(lane)` or `col(lane) + 1`): the holding lane, which keeps `lane`'s bits 0 and 3 (its
    /// block column) and takes bits 1, 2, and 4 from `k % 8` (the block row), and the element,
    /// `2 * ((k / 8) * (N / 8) + n / 8) + n % 2`, the same in every lane.
    pub fn right_holder<A: Arith>(
        &self,
        a: &mut A,
        (k, n): (A::V, A::V),
        lane: A::V,
    ) -> Result<(A::V, A::V), Error> {
        let keep = a.opk(BinOp::And, lane, 9)?;
        let r = a.opk(BinOp::And, k, 3)?;
        let r = a.opk(BinOp::Shl, r, 1)?;
        let h = a.opk(BinOp::LShr, k, 2)?;
        let h = a.opk(BinOp::And, h, 1)?;
        let h = a.opk(BinOp::Shl, h, 4)?;
        let src = a.op(BinOp::Or, keep, r)?;
        let src = a.op(BinOp::Or, src, h)?;
        let bk = a.opk(BinOp::LShr, k, 3)?;
        let bk = a.opk(BinOp::Mul, bk, self.cols)?;
        let bn = a.opk(BinOp::LShr, n, 3)?;
        let p = a.op(BinOp::Add, bk, bn)?;
        let p = a.opk(BinOp::Shl, p, 1)?;
        let j = a.opk(BinOp::And, n, 1)?;
        let e = a.op(BinOp::Add, p, j)?;
        Ok((src, e))
    }

    /// Where a one-simdgroup left layout ([`Layout::of`] of [`Holds::Left`]) holds row `m` and
    /// column `k`, where `m` is a row lane `lane` holds in its own blocks of a one-simdgroup
    /// destination (`m % 8` is `row(lane)`): the holding lane, which keeps `lane`'s bits 1, 2, and
    /// 4 (its block row) and takes bits 0 and 3 from `k % 8` (the block column), and the element,
    /// `2 * ((m / 8) * (K / 8) + k / 8) + k % 2`, the same in every lane.
    pub fn left_holder<A: Arith>(
        &self,
        a: &mut A,
        (m, k): (A::V, A::V),
        lane: A::V,
    ) -> Result<(A::V, A::V), Error> {
        let keep = a.opk(BinOp::And, lane, 0b1_0110)?;
        let lo = a.opk(BinOp::LShr, k, 1)?;
        let lo = a.opk(BinOp::And, lo, 1)?;
        let hi = a.opk(BinOp::LShr, k, 2)?;
        let hi = a.opk(BinOp::And, hi, 1)?;
        let hi = a.opk(BinOp::Shl, hi, 3)?;
        let src = a.op(BinOp::Or, keep, lo)?;
        let src = a.op(BinOp::Or, src, hi)?;
        let bm = a.opk(BinOp::LShr, m, 3)?;
        let bm = a.opk(BinOp::Mul, bm, self.cols)?;
        let bk = a.opk(BinOp::LShr, k, 3)?;
        let p = a.op(BinOp::Add, bm, bk)?;
        let p = a.opk(BinOp::Shl, p, 1)?;
        let j = a.opk(BinOp::And, k, 1)?;
        let e = a.op(BinOp::Add, p, j)?;
        Ok((src, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::Mode;

    /// [`Arith`] evaluated.
    struct Eval;

    impl Arith for Eval {
        type V = u32;
        fn k(&mut self, v: u32) -> Result<u32, Error> {
            Ok(v)
        }
        fn op(&mut self, op: BinOp, a: u32, b: u32) -> Result<u32, Error> {
            Ok(match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::UDiv => a / b,
                BinOp::URem => a % b,
                BinOp::Shl => a << b,
                BinOp::LShr => a >> b,
                BinOp::And => a & b,
                BinOp::Or => a | b,
                _ => unreachable!("{op:?}"),
            })
        }
    }

    fn desc(m: u32, n: u32, k: u32, tr: bool) -> Descriptor {
        Descriptor {
            m,
            n,
            k: Inner::Static(k),
            transpose_left: false,
            transpose_right: tr,
            relaxed_precision: false,
            mode: Mode::MultiplyAccumulate,
        }
    }

    fn place(l: &Layout, e: u32, lane: u32, s: u32) -> (u32, u32) {
        let lp = lane_place(&mut Eval, lane).unwrap();
        l.place(&mut Eval, e, lp, s).unwrap()
    }

    /// Every element of every thread of every measured layout is a distinct element of the
    /// operand, and together they cover it.
    #[test]
    fn measured_layouts_partition_their_operands() {
        let shapes = DESTINATIONS
            .iter()
            .map(|&((m, n, k, s), _)| (Holds::Destination, m, n, k, s))
            .chain(
                RIGHTS
                    .iter()
                    .map(|&(m, n, k, s)| (Holds::Right, m, n, k, s)),
            )
            .chain(LEFTS.iter().map(|&(m, n, k, s)| (Holds::Left, m, n, k, s)));
        for (holds, m, n, k, sgs) in shapes {
            let l = Layout::of(holds, &desc(m, n, k, false), sgs).unwrap();
            let (rows, cols) = (8 * l.rows, 8 * l.cols);
            let mut seen = vec![false; (rows * cols) as usize];
            for s in 0..sgs {
                for lane in 0..32 {
                    for e in 0..l.capacity() {
                        let (r, c) = place(&l, e, lane, s);
                        assert!(r < rows && c < cols, "{holds:?} {m}x{n}x{k}/{sgs}");
                        let slot = &mut seen[(r * cols + c) as usize];
                        assert!(!*slot, "{holds:?} {m}x{n}x{k}/{sgs}: ({r}, {c}) twice");
                        *slot = true;
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "{holds:?} {m}x{n}x{k}/{sgs}");
        }
    }

    /// The probes' readings: the blocks of simdgroup 0 and 3 of the 64 x 64 x 32
    /// destination, simdgroup 1 of the 8 x 128 x 32 one, and lane 9's first elements.
    #[test]
    fn layouts_match_the_readings() {
        let d = Layout::of(Holds::Destination, &desc(64, 64, 32, false), 4).unwrap();
        assert_eq!(d.capacity(), 32);
        // sg0: blocks (col, row) (0,0) (2,0) (4,0) (6,0) (0,2) ...; sg3: (1,1) (3,1) ...
        assert_eq!(place(&d, 0, 0, 0), (0, 0));
        assert_eq!(place(&d, 2, 0, 0), (0, 16));
        assert_eq!(place(&d, 8, 0, 0), (16, 0));
        assert_eq!(place(&d, 0, 0, 3), (8, 8));
        let d = Layout::of(Holds::Destination, &desc(8, 128, 32, false), 4).unwrap();
        assert_eq!(d.capacity(), 8);
        assert_eq!(place(&d, 2, 0, 1), (0, 40));
        // Lane 9: row 0, columns 6 and 7, coordinates (column, row).
        let d = Layout::of(Holds::Destination, &desc(16, 16, 32, false), 1).unwrap();
        assert_eq!(place(&d, 0, 9, 0), (0, 6));
        assert_eq!(place(&d, 1, 9, 0), (0, 7));
        assert!(!d.row_first);
        // The right input is k-major: element 2 is the second block of n.
        let r = Layout::of(Holds::Right, &desc(16, 16, 32, true), 1).unwrap();
        assert_eq!(r.capacity(), 16);
        assert_eq!(place(&r, 2, 0, 0), (0, 8));
        assert_eq!(place(&r, 4, 0, 0), (8, 0));
        assert!(r.row_first);
        // The left input (`tns_coop_left`): lane 0 holds (k, m) = (0, 0), (1, 0), (8, 0), (9, 0),
        // ..., then (0, 8); lane 9 starts at (6, 0); whichever the right input's transpose.
        for tr in [false, true] {
            let l = Layout::of(Holds::Left, &desc(16, 32, 32, tr), 1).unwrap();
            assert_eq!(l.capacity(), 16);
            assert_eq!(place(&l, 1, 0, 0), (0, 1));
            assert_eq!(place(&l, 2, 0, 0), (0, 8));
            assert_eq!(place(&l, 8, 0, 0), (8, 0));
            assert_eq!(place(&l, 0, 9, 0), (0, 6));
            assert!(!l.row_first);
        }
        let l = Layout::of(Holds::Left, &desc(32, 32, 32, false), 1).unwrap();
        assert_eq!(l.capacity(), 32);
        // Unmeasured shapes are refused, and a transposed left input.
        assert!(Layout::of(Holds::Destination, &desc(16, 64, 32, false), 1).is_none());
        assert!(Layout::of(Holds::Right, &desc(64, 64, 32, false), 4).is_none());
        assert!(Layout::of(Holds::Left, &desc(16, 16, 32, false), 1).is_none());
        let mut transposed = desc(16, 32, 32, false);
        transposed.transpose_left = true;
        assert!(Layout::of(Holds::Left, &transposed, 1).is_none());
    }

    /// For every destination element a lane holds and every `k`, [`Layout::right_holder`] names
    /// the lane and element of the right input's row `k` at the destination element's column.
    #[test]
    fn the_right_holder_holds_the_element() {
        for &(m, n, k, sgs) in RIGHTS {
            let d = Layout::of(Holds::Destination, &desc(m, n, k, false), sgs).unwrap();
            let r = Layout::of(Holds::Right, &desc(m, n, k, false), sgs).unwrap();
            for lane in 0..32 {
                for e in 0..d.capacity() {
                    let (_, col) = place(&d, e, lane, 0);
                    for kk in 0..k {
                        let (src, e2) = r.right_holder(&mut Eval, (kk, col), lane).unwrap();
                        assert!(src < 32 && e2 < r.capacity());
                        assert_eq!(place(&r, e2, src, 0), (kk, col), "{m}x{n}x{k} lane {lane}");
                    }
                }
            }
        }
    }

    /// For every destination element a lane holds and every `k`, [`Layout::left_holder`] names
    /// the lane and element of the left input's row at the destination element's row, column `k`.
    #[test]
    fn the_left_holder_holds_the_element() {
        for &(m, n, k, sgs) in LEFTS {
            let d = Layout::of(Holds::Destination, &desc(m, n, k, false), sgs).unwrap();
            let l = Layout::of(Holds::Left, &desc(m, n, k, false), sgs).unwrap();
            for lane in 0..32 {
                for e in 0..d.capacity() {
                    let (row, _) = place(&d, e, lane, 0);
                    for kk in 0..k {
                        let (src, e2) = l.left_holder(&mut Eval, (row, kk), lane).unwrap();
                        assert!(src < 32 && e2 < l.capacity());
                        assert_eq!(place(&l, e2, src, 0), (row, kk), "{m}x{n}x{k} lane {lane}");
                    }
                }
            }
        }
    }
}
