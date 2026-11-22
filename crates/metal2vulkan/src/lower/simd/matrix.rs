//! Simdgroup matrices `air.simdgroup_matrix_8x8_*`.
//!
//! **The value is a per-lane fragment.** An 8x8 matrix is a `<64 x T>` AIR value, held as the
//! arithmetic family's chunked vector (sixteen 4-lane chunks). It does not hold the logical matrix:
//! Metal distributes the matrix over the 32 lanes of the simdgroup, two cells per lane, and the
//! frontend spells `thread_elements()[i]` as element `i` of the value
//! (`extractelement`/`insertelement` at 0 and 1). Lane `L` holds, in elements 0 and 1, the
//! cells `(row(L), col(L))` and `(row(L), col(L) + 1)` with
//!
//! ```text
//! row(L) = ((L >> 1) & 3) | ((L >> 4) << 2)
//! col(L) = ((L >> 3) & 1) · 4 + (L & 1) · 2
//! ```
//!
//! the device-measured storage mapping of the `agx2` multiply-add the backend lowers the 8x8
//! product to. Elements 2 to 63 hold no cell (the hardware keeps two registers); every operation
//! here reads only elements 0 and 1 and produces zero in the others. Element access itself is the
//! arithmetic family's chunked `extractelement`/`insertelement`.
//!
//! **Operations.** `L` is the Metal lane of the calling lane (`L & 31` in `Cluster32`):
//!
//! - `init_diag(v)`: cell `(r, c)` is `v` when `r == c`, else 0; `init_filled(v)`: every cell
//!   is `v`.
//! - `load(p, extent, stride, origin)` / `store(m, p, extent, stride, origin)`: cell `(r, c)` is
//!   the element at `p[(r + origin.y) · stride.y + (c + origin.x) · stride.x]` (the old engine's
//!   device-measured address model; `transpose_matrix` swaps both components of `stride` and
//!   `origin`, and no address depends on `extent`). Each lane reads or writes its own two cells,
//!   one leaf each, through the access family's endpoints (`lower::access::copy`), at the
//!   element's ABI alignment (an access LLVM states no alignment for has its type's ABI
//!   alignment). Constant descriptors are displacements fixed at selection (the farthest cell
//!   within `u32`); a runtime descriptor (a leading dimension the kernel computes) is read lane by
//!   lane as its low word and the displacement computed in wrapping `u32` arithmetic, which is
//!   exact for every Metal-defined cell because pointer offsets are `u32`. [`block_touch`] states
//!   what a block reads or writes for memory provenance.
//! - `multiply_accumulate(a, b, c)` = `c + a·b`: per step `k`, a lane reads its `a` cell from the
//!   lane owning `(row, k)` and its `b` cells from the lane owning `(k, col)` with absolute
//!   shuffles inside its own simdgroup (in a simdgroup only partly taking part, which MSL leaves
//!   undefined, no read leaves the lanes that take part: `tile_mad`), and accumulates with `Fma`
//!   in increasing `k` (the `agx2` schedule, shared with [`super::agx`]). Every operand format is
//!   converted exactly to `f32` first (`half` and `bfloat` are held widened; 8-bit floats decode
//!   exactly). When every format is `half` each step rounds to `half` (the `f16matmad` form);
//!   otherwise the sum accumulates in `f32` and rounds once to the result format.
//!
//! **Classes.** The multiply-add is `CrossLane(SHUFFLE)` and tangled in both tiers; the other
//! operations read only the lane (`CrossLane(BASIC)`), so the fragment layout is the one the
//! shuffles and the dispatch builtins agree on. Only kernels are lowered (a fragment or vertex
//! simdgroup may be partially occupied, which the 32-lane layout cannot express), else
//! `Unsupported{SimdgroupMatrix, NotImplemented}`.
//!
//! **16x16x16 products** (`MultiplyAccumulate16`, `<8 x R>` fragments) are refused in every stage
//! as `Unsupported{SimdgroupMatrix16x16x16, NotImplemented}`: Metal makes them only on an Apple10
//! GPU and later (an M3 Max refuses each at pipeline creation,
//! `crates/metal2vulkan-conformance/probes/simd_matrix16.m`), so their fragment layout cannot be
//! measured on the Apple9 GPUs at hand.

use std::borrow::Cow;

use crate::air::intrinsics::{Fp8, MatElem, SimdMatrix};
use crate::air::ir::{Call, InstId, Value, ValueId};
use crate::air::Entry;
use crate::budget::Meter;
use crate::device::subgroup::OpBits;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::access::copy::{self, Delta, Endpoint};
use crate::lower::access::AccessPlan;
use crate::lower::arith::legal::{self, FloatFmt, Form, Held, LoweredConst};
use crate::lower::arith::narrow::{self, Fp8Layout};
use crate::lower::arith::{self, ops, wide};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::{k32, Leaf, Path};
use crate::memory::provenance::Reach;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, GlslOp};

use super::check_operands;
use super::group::{bin_k, lane, Lane};

/// The lanes of an 8x8 matrix value.
const LANES: u32 = 64;

/// The refusal of a simdgroup-matrix form this lowering does not cover.
pub fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::SimdgroupMatrix).into()
}

/// How one matrix cell's format is held and read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    /// A float held in an `f32` container (`half` and `bfloat` widened).
    Float(FloatFmt),
    /// An 8-bit float carried in an `i8` lane (its bits zero-extended in a `u32`), of its
    /// layout, decoded with integers on every device.
    Fp8(Fp8Layout),
}

impl Cell {
    fn of(e: MatElem) -> Option<Cell> {
        Some(match e {
            MatElem::F32 => Cell::Float(FloatFmt::F32),
            MatElem::F16 => Cell::Float(FloatFmt::Half),
            MatElem::BF16 => Cell::Float(FloatFmt::BFloat),
            MatElem::F8E4M3 => Cell::Fp8(Fp8Layout::of(Fp8::E4M3)),
            MatElem::F8E4M3Fn => Cell::Fp8(Fp8Layout::of(Fp8::E4M3Fn)),
            MatElem::F8E5M2 => Cell::Fp8(Fp8Layout::of(Fp8::E5M2)),
            MatElem::I8(_) | MatElem::I32 => return None,
        })
    }

    /// How the arithmetic family holds one cell.
    fn held(self) -> Held {
        match self {
            Cell::Float(f) => Held::Float(f),
            Cell::Fp8(_) => Held::Int {
                bits: 8,
                container: 32,
            },
        }
    }

    fn form(self) -> Form {
        Form {
            held: self.held(),
            lanes: 1,
        }
    }

    /// An upper estimate of the words converting one cell to `f32`.
    fn to_f32_words(self) -> u32 {
        match self {
            Cell::Float(_) => 0,
            Cell::Fp8(_) => 256,
        }
    }
}

/// `init_diag` or `init_filled`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fill {
    pub cell: FloatFmt,
    pub diag: bool,
    /// The scalar operand.
    pub value: ValueId,
}

/// `multiply_accumulate`: result and operand formats and the operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mac {
    pub result: FloatFmt,
    /// `a`, `b`, `c`.
    pub cells: [Cell; 3],
    pub operands: [ValueId; 3],
    /// Every format is `half`: each step rounds to `half`.
    pub half_steps: bool,
}

/// A block descriptor (`stride` or `origin`): a constant `(x, y)`, or a runtime `<2 x i64>`
/// held as a `u64` vector or, without `shaderInt64`, as two words per lane (`wide`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Desc {
    Const([u64; 2]),
    Value { id: ValueId, wide: bool },
}

/// How a block addresses its cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Addressing {
    /// Constant descriptors: bytes per column step (`stride.x`) and per row step (`stride.y`),
    /// and to cell `(0, 0)`, the farthest cell within `u32` ([`block_steps`]).
    Const { step: [u32; 2], origin: u32 },
    /// Some descriptor is a runtime value: cell `(r, c)` is at byte
    /// `((r + origin.y) · stride.y + (c + origin.x) · stride.x) · size` in wrapping `u32`
    /// arithmetic, each component its low word.
    Runtime { stride: Desc, origin: Desc },
}

/// A block `load` or `store`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Where the matrix lives (a store's endpoint is written).
    pub end: Endpoint,
    /// The stored matrix (stores only).
    pub value: Option<ValueId>,
    pub cell: FloatFmt,
    /// One cell's leaf.
    pub leaf: Leaf,
    pub addr: Addressing,
}

/// A selected simdgroup-matrix operation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixStrategy {
    Fill(Fill),
    Mac(Mac),
    Load(Block),
    Store(Block),
}

impl MatrixStrategy {
    /// The census name in the finished plan `slots`: a block load or store qualified by what its
    /// endpoint reaches (`access::SideReach`).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            MatrixStrategy::Fill(Fill { diag: true, .. }) => "simd.matrix_init_diag".into(),
            MatrixStrategy::Fill(_) => "simd.matrix_init_filled".into(),
            MatrixStrategy::Mac(_) => "simd.matrix_multiply_accumulate".into(),
            MatrixStrategy::Load(b) => b.end.reach(slots).qualify("simd.matrix_load"),
            MatrixStrategy::Store(b) => b.end.reach(slots).qualify("simd.matrix_store"),
        }
    }

    /// An upper estimate of the SPIR-V words one use emits: the lane and its row and column
    /// arithmetic, the fragment's construction (sixteen chunks), and per operation its steps (as
    /// the `agx2` multiply-add), conversions, or two contained leaf accesses.
    pub fn words(&self) -> u32 {
        let lane = 96;
        let build = 160;
        match self {
            MatrixStrategy::Fill(f) => {
                lane + build
                    + if f.diag { 96 } else { 0 }
                    + 2 * narrow::round_words(fmt_form(f.cell))
            }
            MatrixStrategy::Mac(m) => {
                let step_round = if m.half_steps {
                    narrow::round_words(fmt_form(FloatFmt::Half))
                } else {
                    0
                };
                let convert: u32 = m.cells.iter().map(|c| 2 * c.to_f32_words() + 24).sum();
                lane + build
                    + convert
                    + 2 * narrow::round_words(fmt_form(m.result))
                    + 8 * (72 + 2 * step_round)
            }
            MatrixStrategy::Load(b) | MatrixStrategy::Store(b) => {
                let per = b.leaf.cost().saturating_add(64);
                let side = b.end.access_words(per).saturating_add(b.end.base_words());
                // Runtime descriptors: four component words (an extract and a conversion
                // each) and the address arithmetic.
                let runtime = match b.addr {
                    Addressing::Const { .. } => 0,
                    Addressing::Runtime { .. } => 160,
                };
                (lane + build + 96 + runtime).saturating_add(side.saturating_mul(2))
            }
        }
    }
}

/// The scalar form of float format `f`.
fn fmt_form(f: FloatFmt) -> Form {
    Form {
        held: Held::Float(f),
        lanes: 1,
    }
}

fn value_id(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("a matrix operand out of the value range"))
}

fn arg(cx: &SelectCx<'_>, call: &Call, i: usize) -> Result<Value, Error> {
    call.args
        .get(i)
        .copied()
        .ok_or_else(|| cx.invariant("a matrix call without its operands"))
}

/// `v` is an 8x8 matrix value held as chunks of `cell`.
fn check_matrix(cx: &SelectCx<'_>, v: Value, cell: Cell) -> Result<(), Error> {
    let ty = cx
        .value_ty(v)
        .ok_or_else(|| cx.invariant("a matrix value without a type"))?;
    match legal::chunked(cx.entry.module.types(), ty) {
        Some((held, LANES)) if held == cell.held() => Ok(()),
        _ => Err(cx.invariant("a matrix value outside its signature")),
    }
}

/// The float format of a load, store, fill, or result element.
fn float_cell(cx: &SelectCx<'_>, e: MatElem) -> Result<FloatFmt, Error> {
    match Cell::of(e) {
        Some(Cell::Float(f)) => Ok(f),
        _ => Err(cx.invariant("a matrix element outside its signature")),
    }
}

/// Select the strategy of matrix call `inst` (`call`, operation `op`).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    call: &Call,
    op: &SimdMatrix,
    meter: &mut Meter,
) -> Result<MatrixStrategy, Error> {
    let (s, ops) = match *op {
        // Refused in every stage, ahead of the kernel-only check (the module docs).
        SimdMatrix::MultiplyAccumulate16 { .. } => {
            return Err(Unsupported::not_implemented(Feature::SimdgroupMatrix16x16x16).into())
        }
        _ if cx.stage != Stage::Kernel => return Err(not_implemented()),
        // Canonicalization rewrote every AIR 2.7 block operation whose transpose flag is a
        // constant into the AIR 2.8 form; one whose flag is not is not lowered.
        SimdMatrix::Load8x8Legacy { .. } | SimdMatrix::Store8x8Legacy { .. } => {
            return Err(not_implemented())
        }
        SimdMatrix::InitDiag8x8 { elem } | SimdMatrix::InitFilled8x8 { elem } => {
            let cell = float_cell(cx, elem)?;
            let value = arg(cx, call, 0)?;
            check_operands(cx, &[value])?;
            check_matrix(cx, Value::Inst(at), Cell::Float(cell))?;
            let fill = Fill {
                cell,
                diag: matches!(op, SimdMatrix::InitDiag8x8 { .. }),
                value: value_id(cx, value)?,
            };
            (MatrixStrategy::Fill(fill), OpBits::BASIC)
        }
        SimdMatrix::MultiplyAccumulate8x8 {
            result,
            lhs,
            rhs,
            acc,
        } => {
            let result = float_cell(cx, result)?;
            let cells = [lhs, rhs, acc].map(Cell::of);
            let [Some(a), Some(b), Some(c)] = cells else {
                return Err(cx.invariant("a matrix element outside its signature"));
            };
            let cells = [a, b, c];
            check_matrix(cx, Value::Inst(at), Cell::Float(result))?;
            let mut operands = Vec::with_capacity(3);
            for (i, cell) in cells.iter().enumerate() {
                let v = arg(cx, call, i)?;
                check_operands(cx, &[v])?;
                check_matrix(cx, v, *cell)?;
                operands.push(value_id(cx, v)?);
            }
            let operands: [ValueId; 3] = operands
                .try_into()
                .map_err(|_| cx.invariant("a matrix product without three operands"))?;
            let half_steps =
                result == FloatFmt::Half && cells.iter().all(|c| *c == Cell::Float(FloatFmt::Half));
            cx.precision.insert(PrecisionNote::ReductionOrder);
            cx.precision.insert(PrecisionNote::DeviceArithmetic);
            // Every step is an `f32` `Fma` (each format is held widened): without all of `nnan`,
            // `ninf`, and `nsz` on the call, the entry keeps their signed zeros.
            arith::fp_env(cx, call.fmf, fmt_form(FloatFmt::F32));
            if half_steps {
                cx.precision.insert(PrecisionNote::F16FmaDoubleRounding);
            }
            let id = cx
                .values
                .inst(at)
                .ok_or_else(|| cx.invariant("an instruction out of the value range"))?;
            cx.tangled.insert(id);
            let mac = Mac {
                result,
                cells,
                operands,
                half_steps,
            };
            (MatrixStrategy::Mac(mac), OpBits::SHUFFLE)
        }
        SimdMatrix::Load8x8 { elem, .. } => {
            let cell = float_cell(cx, elem)?;
            check_matrix(cx, Value::Inst(at), Cell::Float(cell))?;
            let b = select_block(cx, call, cell, None, meter)?;
            (MatrixStrategy::Load(b), OpBits::BASIC)
        }
        SimdMatrix::Store8x8 { elem, .. } => {
            let cell = float_cell(cx, elem)?;
            let m = arg(cx, call, 0)?;
            check_operands(cx, &[m])?;
            check_matrix(cx, m, Cell::Float(cell))?;
            let b = select_block(cx, call, cell, Some(m), meter)?;
            (MatrixStrategy::Store(b), OpBits::BASIC)
        }
    };
    let simd = &mut cx.families.simd;
    simd.direct_ops = simd.direct_ops.union(ops);
    simd.cluster_ops = simd.cluster_ops.union(ops);
    simd.groups = true;
    Ok(s)
}

/// Constant `<2 x i64>` descriptor `v` as its `(x, y)` components (undefined lanes are 0, as
/// the arithmetic family holds them); `None` when `v` is not a constant.
fn const_descriptor(e: &Entry, v: Value) -> Option<[u64; 2]> {
    match crate::lower::arith::constant(e, v)? {
        LoweredConst::Vector { lanes, .. } => match *lanes {
            [x, y] => Some([x, y]),
            _ => None,
        },
        LoweredConst::Undef(_) => Some([0, 0]),
        _ => None,
    }
}

/// Block descriptor `v`: a constant, or a runtime value the arithmetic family holds.
fn descriptor(cx: &SelectCx<'_>, v: Value) -> Result<Desc, Error> {
    if let Some(k) = const_descriptor(cx.entry, v) {
        return Ok(Desc::Const(k));
    }
    check_operands(cx, &[v])?;
    let ty = cx
        .value_ty(v)
        .ok_or_else(|| cx.invariant("a matrix descriptor without a type"))?;
    let wide = match legal::wide_form_on(cx.entry, cx.profile, ty)? {
        Form {
            held: Held::Int {
                bits: 64,
                container: 64,
            },
            lanes: 2,
        } => false,
        Form {
            held: Held::Wide { bits: 64 },
            lanes: 2,
        } => true,
        _ => return Err(cx.invariant("a matrix descriptor outside its signature")),
    };
    if matches!(v, Value::Const(_)) {
        return Err(cx.invariant("a matrix descriptor constant without lanes"));
    }
    Ok(Desc::Value {
        id: value_id(cx, v)?,
        wide,
    })
}

/// The byte displacements of a block access: per column and row step, and to cell `(0, 0)`,
/// each such that the farthest cell's displacement `origin + 7 · (step.x + step.y)` is a `u32`.
pub fn block_steps(elem_bytes: u32, stride: [u64; 2], origin: [u64; 2]) -> Option<([u32; 2], u32)> {
    let b = u64::from(elem_bytes);
    let sx = stride[0].checked_mul(b)?;
    let sy = stride[1].checked_mul(b)?;
    let o = origin[0]
        .checked_mul(sx)?
        .checked_add(origin[1].checked_mul(sy)?)?;
    let far = o.checked_add(sx.checked_add(sy)?.checked_mul(7)?)?;
    u32::try_from(far).ok()?;
    Some((
        [u32::try_from(sx).ok()?, u32::try_from(sy).ok()?],
        u32::try_from(o).ok()?,
    ))
}

/// The memory operand of a block `load` or `store`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockTouch {
    /// The pointer operand's position.
    pub ptr: usize,
    /// A store (else a load).
    pub write: bool,
    /// With constant descriptors the lowering accepts ([`block_steps`]), the bytes the block
    /// reaches from the pointer: the hull of its 64 cells, from cell `(0, 0)` at `origin` to
    /// the end of cell `(7, 7)` at `origin + 7 · (step.x + step.y) + size`. `None` when a
    /// descriptor is a runtime value.
    pub reach: Option<Reach>,
}

/// The positions of a block operation's pointer, stride, and origin, its element, and whether
/// it stores; `None` for the other operations.
fn block_operands(op: &SimdMatrix) -> Option<(usize, MatElem, bool)> {
    match *op {
        SimdMatrix::Load8x8 { elem, .. } => Some((0, elem, false)),
        SimdMatrix::Store8x8 { elem, .. } => Some((1, elem, true)),
        _ => None,
    }
}

/// What block operation `op` of `call` touches: memory provenance's contract of the matrix
/// load and store (`memory::provenance`), read from the same descriptors the lowering reads.
pub fn block_touch(e: &Entry, call: &Call, op: &SimdMatrix) -> Option<BlockTouch> {
    let (ptr, elem, write) = block_operands(op)?;
    let size = elem.carrier().bits().div_ceil(8);
    let desc = |i: usize| call.args.get(i).and_then(|v| const_descriptor(e, *v));
    let reach = desc(ptr + 2)
        .zip(desc(ptr + 3))
        .and_then(|(stride, origin)| block_steps(size, stride, origin))
        .map(|([sx, sy], origin)| {
            let start = u64::from(origin);
            Reach {
                start,
                end: start + 7 * (u64::from(sx) + u64::from(sy)) + u64::from(size),
            }
        });
    Some(BlockTouch { ptr, write, reach })
}

/// A block `load` (`stored` is `None`) or `store` of `stored`.
fn select_block(
    cx: &mut SelectCx<'_>,
    call: &Call,
    cell: FloatFmt,
    stored: Option<Value>,
    meter: &mut Meter,
) -> Result<Block, Error> {
    let first = usize::from(stored.is_some());
    let ptr = arg(cx, call, first)?;
    let stride = descriptor(cx, arg(cx, call, first + 2)?)?;
    let origin = descriptor(cx, arg(cx, call, first + 3)?)?;
    let bytes = u32::from(cell.bits() / 8);
    let addr = match (stride, origin) {
        (Desc::Const(stride), Desc::Const(origin)) => {
            let (step, origin) = block_steps(bytes, stride, origin).ok_or_else(not_implemented)?;
            Addressing::Const { step, origin }
        }
        (stride, origin) => Addressing::Runtime { stride, origin },
    };
    // The element's ABI alignment (`f32:32`, `f16:16` in the AIR datalayout): every cell's
    // displacement is a multiple of the element size.
    let leaf = Leaf {
        form: fmt_form(cell),
        bytes,
        path: if bytes >= 4 {
            Path::Aligned
        } else {
            Path::OneWord
        },
    };
    let end = copy::endpoint(cx, ptr, stored.is_some(), meter)?;
    Ok(Block {
        end,
        value: stored.map(|v| value_id(cx, v)).transpose()?,
        cell,
        leaf,
        addr,
    })
}

/// The row of the tile cells Metal lane `m` holds.
fn row(fb: &mut FunctionBuilder<'_>, m: SValue) -> BResult<SValue> {
    let lo = bin_k(fb, BinOp::ShiftRightLogical, m, 1)?;
    let lo = bin_k(fb, BinOp::BitwiseAnd, lo, 3)?;
    let hi = bin_k(fb, BinOp::ShiftRightLogical, m, 4)?;
    let hi = bin_k(fb, BinOp::ShiftLeftLogical, hi, 2)?;
    fb.binary(BinOp::BitwiseOr, lo, hi)
}

/// The first column of the tile cells Metal lane `m` holds.
fn col(fb: &mut FunctionBuilder<'_>, m: SValue) -> BResult<SValue> {
    let hi = bin_k(fb, BinOp::ShiftRightLogical, m, 1)?;
    let hi = bin_k(fb, BinOp::BitwiseAnd, hi, 4)?;
    let lo = bin_k(fb, BinOp::BitwiseAnd, m, 1)?;
    let lo = bin_k(fb, BinOp::ShiftLeftLogical, lo, 1)?;
    fb.binary(BinOp::BitwiseOr, hi, lo)
}

/// The Metal lane owning the `a` cell of step `k` (`0` for the lane's own bits) and the `b`
/// cell, relative to the calling lane's kept bits.
pub(super) fn owner_bits(dim: u32, k: u32) -> (u32, u32) {
    if dim == 4 {
        ((k >> 1) & 1, k << 1)
    } else {
        (
            ((k & 4) << 1) | ((k >> 1) & 1),
            ((k & 3) << 1) | ((k & 4) << 2),
        )
    }
}

/// The lane bits an `a` and a `b` owner keep from the calling lane: the row bits (1, 2, 4) and,
/// for 4x4, the column-quadrant bit 3 for `a`; the column bits (0, 3) and, for 4x4, the
/// row-quadrant bit 4 for `b`.
pub(super) fn kept_bits(dim: u32) -> (u32, u32) {
    if dim == 4 {
        (0x1e, 0x19)
    } else {
        (0x16, 0x09)
    }
}

/// How [`tile_mad`] multiplies.
#[derive(Clone, Copy, Debug)]
pub(super) struct TileMad {
    /// 8 (the tile), or 4 (its four quadrants, each multiplied independently).
    pub dim: u32,
    /// The form each step's result is rounded to.
    pub step: Form,
    /// The multiply-add's block is one the uniformity analysis cannot prove every lane of a
    /// whole subgroup executes (`SimdPlan::partly_live`).
    pub divergent: bool,
}

/// `c + a·b` over the distributed tile (dimension 8) or its four quadrants (dimension 4): the
/// calling lane's two cells, each accumulating `a[row][k] · b[k][col_j]` read from their owner
/// lanes with `Fma` in increasing `k`. The cells are `f32` values; `step` rounds each step's
/// result (a narrow cell form, or an `f32` one, which is exact). A lane that does not take part
/// is read through the calling lane ([`Lane::cell_source`]): in the emulated tier's exchange, and
/// on a subgroup, where the multiply-add is [`TileMad::divergent`], through the ballot of the
/// lanes executing it. MSL makes a multiply-add outside uniform control flow within the
/// simdgroup undefined (section 2.4), so the product is promised only where the whole simdgroup
/// takes part.
pub(super) fn tile_mad(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    how: &TileMad,
    a: [SValue; 2],
    b: [SValue; 2],
    mut acc: [SValue; 2],
) -> BResult<[SValue; 2]> {
    let TileMad {
        dim,
        step,
        divergent,
    } = *how;
    let own = ln.metal(fb)?;
    let (ka, kb) = kept_bits(dim);
    let a_key = bin_k(fb, BinOp::BitwiseAnd, own, ka)?;
    let b_key = bin_k(fb, BinOp::BitwiseAnd, own, kb)?;
    // Values 0 and 1 are the `a` cells, 2 and 3 the `b` cells.
    let ex = ln.exchange(fb, &[a[0], a[1], b[0], b[1]])?;
    let live = match (divergent, &ln.emu) {
        (true, None) => Some(ln.live_lanes(fb)?),
        _ => None,
    };
    for k in 0..dim {
        let (va, vb) = owner_bits(dim, k);
        let ao = bin_k(fb, BinOp::BitwiseOr, a_key, va)?;
        let ao = ln.absolute(fb, ao)?;
        let ao = ln.cell_source(fb, live, ao)?;
        let bo = bin_k(fb, BinOp::BitwiseOr, b_key, vb)?;
        let bo = ln.absolute(fb, bo)?;
        let bo = ln.cell_source(fb, live, bo)?;
        let av = ln.read(fb, &ex, (k & 1) as usize, ao)?;
        for (j, slot) in acc.iter_mut().enumerate() {
            let bv = ln.read(fb, &ex, 2 + j, bo)?;
            let r = fb.glsl(GlslOp::Fma, &[av, bv, *slot])?;
            *slot = narrow::round(fb, step, r)?;
        }
    }
    ln.close(fb, ex)?;
    Ok(acc)
}

/// Emit strategy `s` of call `at` and bind its result.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &MatrixStrategy,
) -> BResult<()> {
    let result = cx.plan.values.inst(at);
    let ln = lane(cx, fb)?;
    let x = match s {
        MatrixStrategy::Fill(f) => fill(cx, fb, &ln, f)?,
        MatrixStrategy::Mac(m) => mac(cx, fb, &ln, at, m)?,
        MatrixStrategy::Load(b) => load(cx, fb, &ln, b)?,
        MatrixStrategy::Store(b) => return store(cx, fb, &ln, b),
    };
    let Some(result) = result else {
        return Err(cx.fail("a matrix result out of the value range"));
    };
    cx.define(result, x);
    Ok(())
}

/// The fragment of a matrix held as chunks of `held` with cells `x` in elements 0 and 1 and zero
/// in the others.
fn fragment(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    held: Held,
    x: [SValue; 2],
) -> BResult<SValue> {
    let Some(ty) = legal::spirv_type(fb.module(), &legal::chunked_demote(held, LANES))? else {
        return Err(cx.fail("a matrix fragment arith does not type"));
    };
    let chunk = Form { held, lanes: 4 };
    let zero = ops::splat(fb, Form { held, lanes: 1 }, 0)?;
    let zeros = ops::splat(fb, chunk, 0)?;
    let chunk_ty = ops::ty(fb, chunk)?;
    let first = fb.composite_construct(chunk_ty, &[x[0], x[1], zero, zero])?;
    let members: Vec<SValue> = std::iter::once(first)
        .chain(std::iter::repeat_n(zeros, (LANES / 4 - 1) as usize))
        .collect();
    fb.composite_construct(ty, &members)
}

/// The two cells (elements 0 and 1) of a matrix value.
fn cells(fb: &mut FunctionBuilder<'_>, m: SValue) -> BResult<[SValue; 2]> {
    Ok([
        fb.composite_extract(m, &[0, 0])?,
        fb.composite_extract(m, &[0, 1])?,
    ])
}

fn fill(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, ln: &Lane, f: &Fill) -> BResult<SValue> {
    let v = cx.operand(fb, f.value)?;
    let x = if f.diag {
        let own = ln.metal(fb)?;
        let r = row(fb, own)?;
        let c = col(fb, own)?;
        let zero = ops::splat(fb, fmt_form(f.cell), 0)?;
        let on0 = fb.binary(BinOp::IEqual, r, c)?;
        let c1 = bin_k(fb, BinOp::IAdd, c, 1)?;
        let on1 = fb.binary(BinOp::IEqual, r, c1)?;
        [fb.select(on0, v, zero)?, fb.select(on1, v, zero)?]
    } else {
        [v, v]
    };
    fragment(cx, fb, Held::Float(f.cell), x)
}

/// A cell of format `c` (held as the arithmetic family holds it) as its exact `f32` value.
fn to_f32(fb: &mut FunctionBuilder<'_>, c: Cell, x: SValue) -> BResult<SValue> {
    match c {
        Cell::Float(_) => Ok(x),
        Cell::Fp8(layout) => narrow::fp8_to_f32(
            fb,
            layout,
            narrow::u32_form(c.form()),
            fmt_form(FloatFmt::F32),
            x,
        ),
    }
}

fn mac(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    at: InstId,
    m: &Mac,
) -> BResult<SValue> {
    let mut xs = Vec::with_capacity(3);
    for (v, c) in m.operands.iter().zip(m.cells) {
        let whole = cx.operand(fb, *v)?;
        let [x0, x1] = cells(fb, whole)?;
        xs.push([to_f32(fb, c, x0)?, to_f32(fb, c, x1)?]);
    }
    let (a, b, c) = (nth2(cx, &xs, 0)?, nth2(cx, &xs, 1)?, nth2(cx, &xs, 2)?);
    let step = fmt_form(if m.half_steps {
        FloatFmt::Half
    } else {
        FloatFmt::F32
    });
    let how = TileMad {
        dim: 8,
        step,
        divergent: super::partly_live_at(cx, at),
    };
    let r = tile_mad(fb, ln, &how, a, b, c)?;
    let out = fmt_form(m.result);
    let r = [narrow::round(fb, out, r[0])?, narrow::round(fb, out, r[1])?];
    fragment(cx, fb, Held::Float(m.result), r)
}

fn nth2(cx: &mut EmitCx<'_>, xs: &[[SValue; 2]], i: usize) -> BResult<[SValue; 2]> {
    match xs.get(i) {
        Some(x) => Ok(*x),
        None => Err(cx.fail("a matrix product without its operands")),
    }
}

/// The low word of lane `j` of runtime descriptor `d`, or a constant component modulo 2³².
fn component(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    d: Desc,
    j: u32,
) -> BResult<SValue> {
    match d {
        Desc::Const([x, y]) => k32(fb, if j == 0 { x } else { y } as u32),
        Desc::Value { id, wide: true } => {
            let x = cx.operand(fb, id)?;
            fb.composite_extract(x, &[0, j])
        }
        Desc::Value { id, wide: false } => {
            let x = cx.operand(fb, id)?;
            let l = fb.composite_extract(x, &[j])?;
            wide::low_word(fb, l)
        }
    }
}

/// The byte displacements of the calling lane's two cells.
fn displacements(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    b: &Block,
) -> BResult<[SValue; 2]> {
    let own = ln.metal(fb)?;
    let r = row(fb, own)?;
    let c = col(fb, own)?;
    match b.addr {
        Addressing::Const { step, origin } => {
            let dr = bin_k(fb, BinOp::IMul, r, step[1])?;
            let dc = bin_k(fb, BinOp::IMul, c, step[0])?;
            let d = fb.binary(BinOp::IAdd, dr, dc)?;
            let d0 = bin_k(fb, BinOp::IAdd, d, origin)?;
            let d1 = bin_k(fb, BinOp::IAdd, d0, step[0])?;
            Ok([d0, d1])
        }
        Addressing::Runtime { stride, origin } => {
            // Everything modulo 2³²: the fat pointer's and the arena's offsets are `u32`, so each
            // 64-bit component contributes its low word.
            let sx = component(cx, fb, stride, 0)?;
            let sy = component(cx, fb, stride, 1)?;
            let ox = component(cx, fb, origin, 0)?;
            let oy = component(cx, fb, origin, 1)?;
            let sx = bin_k(fb, BinOp::IMul, sx, b.leaf.bytes)?;
            let sy = bin_k(fb, BinOp::IMul, sy, b.leaf.bytes)?;
            let rr = fb.binary(BinOp::IAdd, r, oy)?;
            let cc = fb.binary(BinOp::IAdd, c, ox)?;
            let dr = fb.binary(BinOp::IMul, rr, sy)?;
            let dc = fb.binary(BinOp::IMul, cc, sx)?;
            let d0 = fb.binary(BinOp::IAdd, dr, dc)?;
            let d1 = fb.binary(BinOp::IAdd, d0, sx)?;
            Ok([d0, d1])
        }
    }
}

fn load(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    b: &Block,
) -> BResult<SValue> {
    let base = copy::base(cx, fb, &b.end)?;
    let [d0, d1] = displacements(cx, fb, ln, b)?;
    let x0 = copy::read(cx, fb, base, Delta::Value(d0), b.leaf)?;
    let x1 = copy::read(cx, fb, base, Delta::Value(d1), b.leaf)?;
    fragment(cx, fb, Held::Float(b.cell), [x0, x1])
}

fn store(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, ln: &Lane, b: &Block) -> BResult<()> {
    let Some(v) = b.value else {
        return Err(cx.fail("a matrix store without its value"));
    };
    let m = cx.operand(fb, v)?;
    let xs = cells(fb, m)?;
    let base = copy::base(cx, fb, &b.end)?;
    let ds = displacements(cx, fb, ln, b)?;
    for (x, d) in xs.into_iter().zip(ds) {
        copy::write(cx, fb, base, Delta::Value(d), b.leaf, x)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{block_steps, kept_bits, owner_bits};

    /// The storage mapping of the distributed 8x8 tile (device-measured): lane `L` owns row
    /// `r(L)` and the columns `c(L)`, `c(L) + 1`.
    fn row(l: u32) -> u32 {
        ((l >> 1) & 3) | ((l >> 4) << 2)
    }
    fn col(l: u32) -> u32 {
        ((l >> 3) & 1) * 4 + (l & 1) * 2
    }

    /// The owner formulas read exactly the cells the product needs: for 8x8, `a[r][k]` and
    /// `b[k][c]` of the whole tile; for 4x4, the same within the lane's quadrant.
    #[test]
    fn owners_hold_the_cells_each_step_reads() {
        for dim in [4u32, 8] {
            let (ka, kb) = kept_bits(dim);
            for l in 0..32u32 {
                for k in 0..dim {
                    let (va, vb) = owner_bits(dim, k);
                    let ao = (l & ka) | va;
                    let bo = (l & kb) | vb;
                    // The `a` cell of step `k`: component `k & 1` of lane `ao`.
                    let (ar, ac) = (row(ao), col(ao) + (k & 1));
                    // The `b` cells: row of lane `bo`, both components.
                    let (br, bc) = (row(bo), col(bo));
                    let quad_col = if dim == 4 { col(l) & 4 } else { 0 };
                    let quad_row = if dim == 4 { row(l) & 4 } else { 0 };
                    assert_eq!(ar, row(l), "a row, lane {l} step {k}");
                    assert_eq!(ac, quad_col + k, "a column, lane {l} step {k}");
                    assert_eq!(br, quad_row + k, "b row, lane {l} step {k}");
                    assert_eq!(bc, col(l), "b column, lane {l} step {k}");
                }
            }
        }
    }

    /// The 32 lanes times two cells cover the tile exactly once.
    #[test]
    fn the_tile_is_covered_once() {
        let mut seen = [[false; 8]; 8];
        for l in 0..32u32 {
            for c in 0..2 {
                let cell = &mut seen[row(l) as usize][(col(l) + c) as usize];
                assert!(!*cell, "lane {l} cell {c}");
                *cell = true;
            }
        }
    }

    /// Block displacements: bytes per step and to the origin, refused when the farthest cell
    /// is beyond a `u32` displacement.
    #[test]
    fn block_steps_scale_and_bound() {
        assert_eq!(block_steps(4, [1, 16], [3, 2]), Some(([4, 64], 140)));
        assert_eq!(block_steps(2, [16, 1], [2, 3]), Some(([32, 2], 70)));
        assert_eq!(block_steps(4, [1, 8], [0, 0]), Some(([4, 32], 0)));
        assert_eq!(block_steps(4, [1, 1 << 30], [0, 0]), None);
        assert_eq!(block_steps(4, [u64::MAX, 1], [0, 0]), None);
        assert_eq!(block_steps(4, [1, 8], [u64::MAX, 0]), None);
    }
}
