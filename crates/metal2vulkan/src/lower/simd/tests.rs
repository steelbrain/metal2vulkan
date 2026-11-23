//! Unit tests of the simd family's tables: operation classes per tier, class raising, and the
//! operations each scope admits. The lowerings themselves run on the CPU interpreter in
//! `src/tests/engine_simd.rs`.

use super::group::check;
use super::*;
use crate::air::intrinsics::{GroupOp as G, GroupScope as S};
use crate::lower::arith::FloatFmt;

/// A Metal lane index or delta operand (`i16`, held in a `u32`).
const LANE_FORM: Form = Form {
    held: Held::Int {
        bits: 16,
        container: 32,
    },
    lanes: 1,
};
const BOOL: Form = Form {
    held: Held::Bool,
    lanes: 1,
};
const F32: Form = Form {
    held: Held::Float(FloatFmt::F32),
    lanes: 1,
};
const I64: Form = Form {
    held: Held::Int {
        bits: 64,
        container: 64,
    },
    lanes: 1,
};

fn plan(scope: GroupScope, op: GroupOp, form: Form, konst: Option<u32>) -> GroupPlan {
    GroupPlan {
        scope,
        op,
        form,
        arith: combine(op, form, Some(Sign::Signed)).expect("combine"),
        konst,
        quad_instructions: true,
        helpers: false,
    }
}

/// `plan` in a stage that takes no quad instructions.
fn without_quads(g: GroupPlan) -> GroupPlan {
    GroupPlan {
        quad_instructions: false,
        ..g
    }
}

/// `plan` in a fragment function, whose quads hold helpers.
fn in_fragment(g: GroupPlan) -> GroupPlan {
    GroupPlan {
        quad_instructions: false,
        helpers: true,
        ..g
    }
}

/// The classes each operation needs follow the SPIR-V the tier emits: a `Direct` vote is
/// `OpGroupNonUniformAll` (vote), a `Cluster32` one a ballot word; a `Direct` reduction or product
/// is arithmetic, a `Cluster32` one clustered; a `Cluster32` prefix sum or product is ballot and
/// shuffles; a rotate up is a shuffle; a quad broadcast or shuffle-xor by a constant is the quad
/// class where the stage takes quad instructions and a shuffle elsewhere; every fragment quad
/// operation is the quad class, but a fill by a constant modulo whose segment is wider than the
/// quad (8 lanes for a modulo of 8), a shuffle.
#[test]
fn operation_classes_follow_the_tier() {
    let cases: [(GroupPlan, OpBits, OpBits); 34] = [
        (
            plan(S::Simdgroup, G::Shuffle, F32, None),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            plan(S::Simdgroup, G::ShuffleUp, F32, None),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            plan(S::Simdgroup, G::All, BOOL, None),
            OpBits::VOTE,
            OpBits::BALLOT,
        ),
        (
            plan(S::Simdgroup, G::Sum, F32, None),
            OpBits::ARITHMETIC,
            OpBits::CLUSTERED,
        ),
        (
            plan(S::Simdgroup, G::PrefixInclusiveSum, F32, None),
            OpBits::ARITHMETIC,
            OpBits::BALLOT.union(OpBits::SHUFFLE),
        ),
        (
            plan(S::Simdgroup, G::Product, F32, None),
            OpBits::ARITHMETIC,
            OpBits::CLUSTERED,
        ),
        (
            plan(S::Simdgroup, G::PrefixExclusiveProduct, F32, None),
            OpBits::ARITHMETIC,
            OpBits::BALLOT.union(OpBits::SHUFFLE),
        ),
        (
            plan(S::Simdgroup, G::ShuffleRotateUp, F32, None),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            plan(S::Simdgroup, G::IsFirst, BOOL, None),
            OpBits::BASIC,
            OpBits::BALLOT,
        ),
        (
            plan(S::Simdgroup, G::BroadcastFirst, F32, None),
            OpBits::BALLOT,
            OpBits::BALLOT.union(OpBits::SHUFFLE),
        ),
        (
            plan(S::Simdgroup, G::Ballot, I64, None),
            OpBits::BALLOT,
            OpBits::BALLOT,
        ),
        (
            plan(S::Simdgroup, G::ActiveThreadsMask, I64, None),
            OpBits::BALLOT,
            OpBits::BALLOT,
        ),
        (
            plan(S::Quad, G::Ballot, LANE_FORM, None),
            OpBits::BALLOT,
            OpBits::BALLOT,
        ),
        (
            plan(S::Quad, G::Broadcast, F32, Some(2)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            plan(S::Quad, G::Broadcast, F32, None),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            without_quads(plan(S::Quad, G::Broadcast, F32, Some(2))),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            without_quads(plan(S::Quad, G::ShuffleXor, F32, Some(3))),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
        (
            plan(S::Quad, G::ShuffleXor, F32, Some(0)),
            OpBits::NONE,
            OpBits::NONE,
        ),
        (
            without_quads(plan(S::Quad, G::ShuffleXor, F32, Some(0))),
            OpBits::NONE,
            OpBits::NONE,
        ),
        (
            plan(S::Quad, G::Sum, F32, None),
            OpBits::BALLOT.union(OpBits::SHUFFLE),
            OpBits::BALLOT.union(OpBits::SHUFFLE),
        ),
        (
            plan(S::Quad, G::All, BOOL, None),
            OpBits::BALLOT,
            OpBits::BALLOT,
        ),
        (
            in_fragment(plan(S::Quad, G::Sum, F32, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::Xor, LANE_FORM, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::All, BOOL, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::Any, BOOL, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::PrefixInclusiveSum, F32, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::BroadcastFirst, F32, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::Ballot, LANE_FORM, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::ActiveThreadsMask, LANE_FORM, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::IsFirst, BOOL, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::Shuffle, F32, None)),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::Broadcast, F32, Some(1))),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::ShuffleAndFillDown, F32, Some(4))),
            OpBits::QUAD,
            OpBits::QUAD,
        ),
        (
            in_fragment(plan(S::Quad, G::ShuffleAndFillUp, F32, Some(8))),
            OpBits::SHUFFLE,
            OpBits::SHUFFLE,
        ),
    ];
    for (g, d, c) in cases {
        assert_eq!(direct_ops(&g), d, "{g:?}");
        assert_eq!(cluster_ops(&g), c, "{g:?}");
        assert_eq!(check(&g), Ok(()), "{g:?}");
    }
}

/// Raising adds classes, keeps `Quad` a quad user, and leaves a class alone for an operation that
/// needs none (a quad shuffle-xor by a constant 0 is the value itself).
#[test]
fn raising_adds_operation_classes() {
    assert_eq!(
        raise(SimdClass::None, OpBits::SHUFFLE),
        SimdClass::CrossLane(OpBits::SHUFFLE)
    );
    assert_eq!(
        raise(SimdClass::LaneArithmetic, OpBits::VOTE),
        SimdClass::CrossLane(OpBits::VOTE)
    );
    assert_eq!(
        raise(SimdClass::Quad, OpBits::SHUFFLE),
        SimdClass::CrossLane(OpBits::SHUFFLE.union(OpBits::QUAD))
    );
    assert_eq!(
        raise(SimdClass::CrossLane(OpBits::BASIC), OpBits::BALLOT),
        SimdClass::CrossLane(OpBits::BASIC.union(OpBits::BALLOT))
    );
    assert_eq!(raise(SimdClass::None, OpBits::NONE), SimdClass::None);
    assert_eq!(raise(SimdClass::Matrix, OpBits::SHUFFLE), SimdClass::Matrix);
}

/// Each scope admits its operations with the forms emission assumes: a simdgroup's ballot and
/// active mask are 64-bit, a quad's `ushort`s, a vote and `is_first` a `bool`. Every operation
/// exists in both scopes: a quad's prefix sums and first active lane among them.
#[test]
fn checks_admit_each_scope_and_form() {
    assert_eq!(
        check(&plan(S::Quad, G::ActiveThreadsMask, LANE_FORM, None)),
        Ok(())
    );
    assert_eq!(check(&plan(S::Quad, G::Ballot, LANE_FORM, None)), Ok(()));
    assert_eq!(check(&plan(S::Quad, G::Ballot, I64, None)), Err(()));
    assert_eq!(
        check(&plan(S::Simdgroup, G::ActiveThreadsMask, I64, None)),
        Ok(())
    );
    assert_eq!(
        check(&plan(S::Simdgroup, G::ActiveThreadsMask, LANE_FORM, None)),
        Err(())
    );
    assert_eq!(
        check(&plan(S::Quad, G::PrefixInclusiveSum, F32, None)),
        Ok(())
    );
    assert_eq!(check(&plan(S::Quad, G::BroadcastFirst, F32, None)), Ok(()));
    assert_eq!(check(&plan(S::Quad, G::All, F32, None)), Err(()));
    assert_eq!(check(&plan(S::Simdgroup, G::All, F32, None)), Err(()));
    assert_eq!(
        check(&plan(S::Simdgroup, G::Ballot, LANE_FORM, None)),
        Err(())
    );
    let no_combine = GroupPlan {
        arith: None,
        ..plan(S::Simdgroup, G::Sum, F32, None)
    };
    assert_eq!(check(&no_combine), Err(()));
}

/// The combining operation follows the value's kind and sign token; `bool` sums and extrema are
/// not lowered.
#[test]
fn reductions_combine_by_kind_and_sign() {
    let int = LANE_FORM;
    assert_eq!(
        combine(G::Min, int, Some(Sign::Signed)),
        Ok(Some(GroupArith::SMin))
    );
    assert_eq!(
        combine(G::Min, int, Some(Sign::Unsigned)),
        Ok(Some(GroupArith::UMin))
    );
    assert_eq!(combine(G::Max, F32, None), Ok(Some(GroupArith::FMax)));
    assert_eq!(
        combine(G::PrefixExclusiveSum, F32, None),
        Ok(Some(GroupArith::FAdd))
    );
    assert_eq!(combine(G::Or, BOOL, None), Ok(Some(GroupArith::LogicalOr)));
    assert_eq!(combine(G::Product, F32, None), Ok(Some(GroupArith::FMul)));
    assert_eq!(
        combine(G::PrefixInclusiveProduct, int, Some(Sign::Unsigned)),
        Ok(Some(GroupArith::IMul))
    );
    assert!(combine(G::PrefixExclusiveProduct, BOOL, None).is_err());
    assert_eq!(combine(G::Shuffle, F32, None), Ok(None));
    assert!(combine(G::Sum, BOOL, None).is_err());
}

/// The segment width of a fill whose modulo is a constant, as Metal answers it on the M3 Max and
/// the M4 (every constant from 0 to 1100 and the larger ones here, both scopes).
#[test]
fn constant_fill_moduli_take_metal_s_widths() {
    let cases: [(u32, u32); 24] = [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 2),
        (64, 2),
        (65, 2),
        (256, 2),
        (512, 2),
        (513, 4),
        (1000, 4),
        (1025, 8),
        (2048, 8),
        (4095, 16),
        (8192, 32),
        (16384, 2),
        (65535, 2),
    ];
    for (m, width) in cases {
        assert_eq!(super::group::constant_fill_width(m), width, "modulo {m}");
    }
}
