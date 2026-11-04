//! Metal runtime routines as plain AIR.
//!
//! Apple's compiler emits MSL's `frexp` and `ilogb` as calls to routines its backend defines at
//! pipeline creation, `___metal_frexp_<T>_pthread<I>(T x, thread I *e)` and
//! `___metal_ilogb_<T>(T x, i32 mode)`, which the math family decodes ([`MathFn::Frexp`],
//! [`MathFn::Ilogb`]). After SCCP, [`expand`] replaces each call by the routine's answer in plain
//! AIR: integer operations on the bits of `x` (a `half` widened exactly to `float` first), lane by
//! lane in `T`'s shape, with no fast-math flags (the answer is the routine's, measured alike with
//! and without fast math).
//!
//! **What Metal answers** (an M4, macOS 27.0.1; libraries compiled offline by `xcrun metal`, with
//! and without `-fno-fast-math`, and with `air.compile.denorms_disable` stripped from the module,
//! which changes nothing). For `x`'s `float` bits `u`, `a = u & 0x7fffffff`, and the biased
//! exponent `E = a >> 23`:
//!
//! | `x` | `frexp` (mantissa, exponent) | `ilogb` |
//! | --- | --- | --- |
//! | `E == 0`: `±0` or a `float` denormal | `(+0, 0)` | `INT_MIN` |
//! | `E == 255`, `a & 0x7fffff == 0`: `±inf` | `(x, 0)` | `INT_MAX` |
//! | `E == 255`, a NaN | `(x, 0)`, its payload kept | `INT_MIN` |
//! | otherwise | `(sign \| 0x3f000000 \| (u & 0x7fffff), E - 126)` | `E - 127` |
//!
//! A `half` is exact as a `float` and never a `float` denormal, so the same table answers it:
//! its denormals normalize (`0x0001` is `(0.5, -23)`, `ilogb` `-24`), its zeros are `(+0, 0)`, and
//! the mantissa narrows back to `half` exactly (it has at most 11 significant bits). Its infinities
//! and NaNs return `x` itself, so their bits are kept. `ilogb`'s mode (0 for `ilogb` and
//! `fast::ilogb`, 3 for `precise::ilogb`) changes no answer; the decoder takes only those two.
//! The vector forms answer lane by lane as the scalar one (every lane of the 2-, 3- and 4-lane
//! forms checked).
//!
//! Runtime-compiled source keeps `float` denormals (`frexp(0x1p-149) = (0.5, -148)`): what it
//! compiles to is not this routine, which AIR the translator receives calls by name.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, Elem, Math, MathFn};
use crate::ir::{
    BinOp, Block, Body, Call, Callee, CastOp, Const, FastMath, Idx, Inst, InstId, InstKind,
    IntFlags, IntPredicate, ModuleParts, Type, TypeId, Value,
};

use super::rewrite::invariant;

/// The routine a call is, and its value type's lanes and element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Routine {
    frexp: bool,
    half: bool,
    lanes: u32,
}

/// The call's routine, when it is one [`expand`] replaces.
fn routine(kind: &InstKind) -> Option<(Routine, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    let Callee::Air {
        op: AirIntrinsic::Math(Math { f, ty, .. }),
        ..
    } = call.callee
    else {
        return None;
    };
    let frexp = match f {
        MathFn::Frexp => true,
        MathFn::Ilogb => false,
        _ => return None,
    };
    let half = match ty.elem {
        Elem::F16 => true,
        Elem::F32 => false,
        _ => return None,
    };
    Some((
        Routine {
            frexp,
            half,
            lanes: ty.lanes,
        },
        call,
    ))
}

/// An instruction sink over the output body, interning the types and constants it uses.
struct Sink<'a> {
    out: &'a mut Body,
    parts: &'a mut ModuleParts,
    lanes: u32,
}

impl Sink<'_> {
    /// Scalar `scalar` in the routine's shape: itself, or a vector of `lanes` of it.
    fn shaped(&mut self, scalar: Type, m: &mut Meter) -> Result<TypeId, Error> {
        let elem = self.parts.types.intern(scalar, m)?;
        if self.lanes == 1 {
            return Ok(elem);
        }
        self.parts.types.intern(
            Type::Vector {
                elem,
                len: self.lanes,
            },
            m,
        )
    }

    /// The constant `value` of scalar type `scalar`, in the routine's shape (splat).
    fn splat(&mut self, scalar: Type, value: Const, m: &mut Meter) -> Result<Value, Error> {
        let elem_ty = self.parts.types.intern(scalar.clone(), m)?;
        let elem = self
            .parts
            .consts
            .intern(crate::ir::Constant { ty: elem_ty, value }, m)?;
        if self.lanes == 1 {
            return Ok(Value::Const(elem));
        }
        let ty = self.shaped(scalar, m)?;
        let lanes = vec![elem; self.lanes as usize].into_boxed_slice();
        let c = self.parts.consts.intern(
            crate::ir::Constant {
                ty,
                value: Const::Aggregate(lanes),
            },
            m,
        )?;
        Ok(Value::Const(c))
    }

    fn int(&mut self, v: u32, m: &mut Meter) -> Result<Value, Error> {
        self.splat(Type::Int(32), Const::Int(u64::from(v)), m)
    }

    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        let id = self.out.push_inst(Inst { ty, kind }, m)?;
        Ok(Value::Inst(id))
    }

    fn bin(&mut self, op: BinOp, a: Value, b: Value, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.shaped(Type::Int(32), m)?;
        let kind = InstKind::BinOp {
            op,
            flags: IntFlags::NONE,
            fmf: FastMath::NONE,
            a,
            b,
        };
        self.push(ty, kind, m)
    }

    fn bin_k(&mut self, op: BinOp, a: Value, k: u32, m: &mut Meter) -> Result<Value, Error> {
        let k = self.int(k, m)?;
        self.bin(op, a, k, m)
    }

    fn eq_k(&mut self, a: Value, k: u32, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.shaped(Type::Int(1), m)?;
        let b = self.int(k, m)?;
        let kind = InstKind::ICmp {
            pred: IntPredicate::Eq,
            a,
            b,
        };
        self.push(ty, kind, m)
    }

    fn select(
        &mut self,
        ty: TypeId,
        c: Value,
        t: Value,
        f: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::Select {
            c,
            t,
            f,
            fmf: FastMath::NONE,
        };
        self.push(ty, kind, m)
    }

    fn cast(&mut self, op: CastOp, v: Value, to: TypeId, m: &mut Meter) -> Result<Value, Error> {
        self.push(to, InstKind::Cast { op, v }, m)
    }

    /// `x`'s `float` bits and biased exponent: `(u, a, E)`.
    fn bits(&mut self, r: Routine, x: Value, m: &mut Meter) -> Result<[Value; 3], Error> {
        let f32t = self.shaped(Type::Float, m)?;
        let i32t = self.shaped(Type::Int(32), m)?;
        let f = if r.half {
            self.cast(CastOp::FPExt, x, f32t, m)?
        } else {
            x
        };
        let u = self.cast(CastOp::Bitcast, f, i32t, m)?;
        let a = self.bin_k(BinOp::And, u, 0x7fff_ffff, m)?;
        let e = self.bin_k(BinOp::LShr, a, 23, m)?;
        Ok([u, a, e])
    }

    /// `frexp(x)`, storing the exponent through `p`; the last instruction is the mantissa.
    fn frexp(&mut self, r: Routine, x: Value, p: Value, m: &mut Meter) -> Result<(), Error> {
        let (i32t, f32t, i1t) = (
            self.shaped(Type::Int(32), m)?,
            self.shaped(Type::Float, m)?,
            self.shaped(Type::Int(1), m)?,
        );
        let [u, _, e] = self.bits(r, x, m)?;
        let low = self.eq_k(e, 0, m)?;
        let high = self.eq_k(e, 255, m)?;
        let keep = self.push(
            i1t,
            InstKind::BinOp {
                op: BinOp::Or,
                flags: IntFlags::NONE,
                fmf: FastMath::NONE,
                a: low,
                b: high,
            },
            m,
        )?;
        // `E - 126`, as the addition of its two's complement.
        let unbiased = self.bin_k(BinOp::Add, e, 126u32.wrapping_neg(), m)?;
        let zero = self.int(0, m)?;
        let exponent = self.select(i32t, keep, zero, unbiased, m)?;
        let void = self.parts.types.intern(Type::Void, m)?;
        self.push(
            void,
            InstKind::Store {
                val: exponent,
                ptr: p,
                align: None,
                volatile: false,
                atomic: None,
            },
            m,
        )?;
        let kept = self.bin_k(BinOp::And, u, 0x807f_ffff, m)?;
        let mbits = self.bin_k(BinOp::Or, kept, 0x3f00_0000, m)?;
        let mantissa = self.cast(CastOp::Bitcast, mbits, f32t, m)?;
        let (ty, mantissa, plus_zero) = if r.half {
            let f16t = self.shaped(Type::Half, m)?;
            let narrow = self.cast(CastOp::FPTrunc, mantissa, f16t, m)?;
            (f16t, narrow, self.splat(Type::Half, Const::Float(0), m)?)
        } else {
            (f32t, mantissa, self.splat(Type::Float, Const::Float(0), m)?)
        };
        let finite = self.select(ty, low, plus_zero, mantissa, m)?;
        self.select(ty, high, x, finite, m)?;
        Ok(())
    }

    /// `ilogb(x)`; the last instruction is the result.
    fn ilogb(&mut self, r: Routine, x: Value, m: &mut Meter) -> Result<(), Error> {
        let i32t = self.shaped(Type::Int(32), m)?;
        let [_, a, e] = self.bits(r, x, m)?;
        let low = self.eq_k(e, 0, m)?;
        let high = self.eq_k(e, 255, m)?;
        let fraction = self.bin_k(BinOp::And, a, 0x007f_ffff, m)?;
        let infinite = self.eq_k(fraction, 0, m)?;
        let (int_max, int_min) = (self.int(0x7fff_ffff, m)?, self.int(0x8000_0000, m)?);
        let special = self.select(i32t, infinite, int_max, int_min, m)?;
        let unbiased = self.bin_k(BinOp::Add, e, 127u32.wrapping_neg(), m)?;
        let finite = self.select(i32t, low, int_min, unbiased, m)?;
        self.select(i32t, high, special, finite, m)?;
        Ok(())
    }

    /// The instructions of call `r` with operands `args`; the last is the call's result.
    fn emit(&mut self, r: Routine, args: &[Value], m: &mut Meter) -> Result<(), Error> {
        match (r.frexp, args) {
            (true, [x, p]) => self.frexp(r, *x, *p, m),
            (false, [x, _mode]) => self.ilogb(r, *x, m),
            _ => Err(invariant("a runtime routine's call without its operands")),
        }
    }
}

/// The number of instructions call `r` becomes.
fn routine_len(r: Routine, parts: &mut ModuleParts, m: &mut Meter) -> Result<u32, Error> {
    let mut scratch = Body::default();
    let mut s = Sink {
        out: &mut scratch,
        parts,
        lanes: r.lanes,
    };
    let any = Value::Inst(InstId::new(0));
    s.emit(r, &[any, any], m)?;
    let len = scratch.insts.len();
    scratch.release(m);
    u32::try_from(len).map_err(|_| invariant("routine length"))
}

/// Replace every `frexp` and `ilogb` routine call of `body` by its answer (the module docs).
pub(super) fn expand(body: Body, parts: &mut ModuleParts, m: &mut Meter) -> Result<Body, Error> {
    if !body.insts.iter().any(|(_, i)| routine(&i.kind).is_some()) {
        return Ok(body);
    }
    // Each old instruction's id in the output: a replaced call's is its expansion's last.
    let mut lens: Vec<(Routine, u32)> = Vec::new();
    let mut result = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u32;
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let n = match routine(&inst.kind) {
            Some((r, _)) => match lens.iter().find(|(k, _)| *k == r) {
                Some(&(_, n)) => n,
                None => {
                    let n = routine_len(r, parts, m)?;
                    m.reserve(&mut lens, 1)?;
                    lens.push((r, n));
                    n
                }
            },
            None => 1,
        };
        next = next
            .checked_add(n)
            .ok_or_else(|| invariant("instruction ids"))?;
        result.push(InstId::new(next - 1));
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .map(|&n| Value::Inst(n))
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let mut out = Body::default();
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            match routine(&inst.kind) {
                Some((r, call)) => {
                    let mut args = Vec::new();
                    m.reserve(&mut args, call.args.len())?;
                    for a in call.args.iter() {
                        args.push(map(*a)?);
                    }
                    let mut s = Sink {
                        out: &mut out,
                        parts,
                        lanes: r.lanes,
                    };
                    s.emit(r, &args, m)?;
                    m.release(args.len() * std::mem::size_of::<Value>());
                }
                None => {
                    let mut kind = inst.kind.clone();
                    let mut failed = None;
                    kind.for_each_value_mut(|val| match map(*val) {
                        Ok(n) => *val = n,
                        Err(e) => {
                            failed.get_or_insert(e);
                        }
                    });
                    if let Some(e) = failed {
                        return Err(e);
                    }
                    out.push_inst(Inst { ty: inst.ty, kind }, m)?;
                }
            }
            let want = result.get(i.index()).map(|r| r.index() + 1);
            if want != Some(out.insts.len()) {
                return Err(invariant("a routine's length differs from its measure"));
            }
        }
        let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(result.len().saturating_mul(std::mem::size_of::<InstId>()));
    m.release(
        lens.len()
            .saturating_mul(std::mem::size_of::<(Routine, u32)>()),
    );
    // The body the expansions replaced is freed.
    body.release(m);
    Ok(out)
}
