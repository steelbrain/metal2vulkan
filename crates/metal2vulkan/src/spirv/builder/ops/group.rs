//! Subgroup (`GroupNonUniform*`) constructors.
//!
//! The execution scope is always `Subgroup` (the quad vote, `OpGroupNonUniformQuadAnyKHR`, has no
//! scope operand); cluster sizes are constant powers of two. Which ops a
//! device supports (and in which stages) is the profile's concern, checked by requirement
//! derivation; every subgroup operation is tangled.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::{err, Num};
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    GroupArith, GroupInst, GroupOperation, Inst, QuadDirection, Scope, ShuffleOp,
};

/// The mode of a subgroup reduction or scan.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupMode {
    Reduce,
    InclusiveScan,
    ExclusiveScan,
    /// A reduction within clusters of this many lanes (a power of two).
    Clustered(u32),
}

impl FunctionBuilder<'_> {
    fn group(&mut self, op: GroupInst, ty: TypeId) -> BResult<Value> {
        let scope = self.scope_id(Scope::Subgroup)?;
        self.emit(Inst::Group { scope, op }, ty)
    }

    /// A value a subgroup can move between lanes: a scalar or vector of integers, floats, or
    /// `bool`s.
    fn lane_value(&self, op: crate::spv::Op, v: Value) -> BResult<Num> {
        self.want(op, 0, v, |_| true)
    }

    /// An unsigned 32-bit integer scalar operand (lane id, mask, delta, index).
    fn lane_index(&self, op: crate::spv::Op, i: u8, v: Value) -> BResult<()> {
        self.want(op, i, v, |n| n.is_uint() && n.is_scalar() && n.width == 32)
            .map(|_| ())
    }

    /// A ballot: `uvec4`.
    fn ballot_operand(&self, op: crate::spv::Op, v: Value) -> BResult<()> {
        self.want(op, 0, v, |n| n.is_uint() && n.width == 32 && n.n == 4)
            .map(|_| ())
    }

    fn uvec4(&mut self) -> BResult<TypeId> {
        let u = self.uint_ty(32)?;
        self.vec_of(u, 4)
    }

    fn cluster(
        &mut self,
        op: crate::spv::Op,
        i: u8,
        size: u32,
    ) -> BResult<(u32, crate::spirv::ids::ConstId)> {
        if !size.is_power_of_two() {
            return Err(err(op, OperandRule::Range(i)));
        }
        Ok((size, self.u32_const(size)?))
    }

    /// `OpGroupNonUniformElect`. Result: `bool`.
    pub fn group_elect(&mut self) -> BResult<Value> {
        let ty = self.bool_ty(1)?;
        self.group(GroupInst::Elect, ty)
    }

    /// `OpGroupNonUniformAll` (`all`) or `…Any` of a `bool` scalar. Result: `bool`.
    pub fn group_vote(&mut self, all: bool, pred: Value) -> BResult<Value> {
        let op = if all {
            crate::spv::Op::GroupNonUniformAll
        } else {
            crate::spv::Op::GroupNonUniformAny
        };
        self.want(op, 0, pred, |n| n.is_bool() && n.is_scalar())?;
        let ty = self.bool_ty(1)?;
        self.group(
            if all {
                GroupInst::All(pred)
            } else {
                GroupInst::Any(pred)
            },
            ty,
        )
    }

    /// `OpGroupNonUniformBroadcastFirst`. Result: the value's type.
    pub fn group_broadcast_first(&mut self, value: Value) -> BResult<Value> {
        let v = self.lane_value(crate::spv::Op::GroupNonUniformBroadcastFirst, value)?;
        self.group(GroupInst::BroadcastFirst(value), v.ty)
    }

    /// `OpGroupNonUniformBallot` of a `bool` scalar. Result: `uvec4`.
    pub fn group_ballot(&mut self, pred: Value) -> BResult<Value> {
        self.want(crate::spv::Op::GroupNonUniformBallot, 0, pred, |n| {
            n.is_bool() && n.is_scalar()
        })?;
        let ty = self.uvec4()?;
        self.group(GroupInst::Ballot(pred), ty)
    }

    /// `OpGroupNonUniformBallotBitExtract` of bit `index` (`u32`). Result: `bool`.
    pub fn group_ballot_bit_extract(&mut self, ballot: Value, index: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::GroupNonUniformBallotBitExtract;
        self.ballot_operand(OP, ballot)?;
        self.lane_index(OP, 1, index)?;
        let ty = self.bool_ty(1)?;
        self.group(GroupInst::BallotBitExtract { ballot, index }, ty)
    }

    /// A shuffle: `x` is the lane, mask, or delta (`u32`). Result: the value's type.
    pub fn group_shuffle(&mut self, op: ShuffleOp, value: Value, x: Value) -> BResult<Value> {
        let inst = GroupInst::Shuffle { op, value, x };
        let o = inst.to_spirv();
        let v = self.lane_value(o, value)?;
        self.lane_index(o, 1, x)?;
        self.group(inst, v.ty)
    }

    /// A reduction or scan: integer operations on integers, float operations on floats, logical
    /// operations on `bool`s. Result: the value's type.
    pub fn group_arith(&mut self, op: GroupArith, mode: GroupMode, value: Value) -> BResult<Value> {
        let o = op.to_spirv();
        let want: fn(&Num) -> bool = match op {
            GroupArith::FAdd | GroupArith::FMul | GroupArith::FMin | GroupArith::FMax => {
                Num::is_float
            }
            GroupArith::LogicalAnd | GroupArith::LogicalOr | GroupArith::LogicalXor => Num::is_bool,
            _ => Num::is_int,
        };
        let v = self.want(o, 0, value, want)?;
        let group = match mode {
            GroupMode::Reduce => GroupOperation::Reduce,
            GroupMode::InclusiveScan => GroupOperation::InclusiveScan,
            GroupMode::ExclusiveScan => GroupOperation::ExclusiveScan,
            GroupMode::Clustered(size) => {
                let (size, id) = self.cluster(o, 2, size)?;
                GroupOperation::ClusteredReduce { size, id }
            }
        };
        self.group(GroupInst::Arith { op, group, value }, v.ty)
    }

    /// `OpGroupNonUniformQuadBroadcast` from quad lane `index` (`u32`). Result: the value's type.
    pub fn group_quad_broadcast(&mut self, value: Value, index: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::GroupNonUniformQuadBroadcast;
        let v = self.lane_value(OP, value)?;
        self.lane_index(OP, 1, index)?;
        self.group(GroupInst::QuadBroadcast { value, index }, v.ty)
    }

    /// `OpGroupNonUniformQuadSwap`. Result: the value's type.
    pub fn group_quad_swap(&mut self, value: Value, dir: QuadDirection) -> BResult<Value> {
        let v = self.lane_value(crate::spv::Op::GroupNonUniformQuadSwap, value)?;
        let dir_id = self.u32_const(dir.bits())?;
        self.group(GroupInst::QuadSwap { value, dir, dir_id }, v.ty)
    }

    /// `OpGroupNonUniformQuadAnyKHR` of a `bool` scalar. Result: `bool`.
    pub fn group_quad_any(&mut self, pred: Value) -> BResult<Value> {
        self.want(crate::spv::Op::GroupNonUniformQuadAnyKHR, 0, pred, |n| {
            n.is_bool() && n.is_scalar()
        })?;
        let ty = self.bool_ty(1)?;
        self.group(GroupInst::QuadAny(pred), ty)
    }
}
