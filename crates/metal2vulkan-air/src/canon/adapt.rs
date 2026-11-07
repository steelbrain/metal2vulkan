//! A visible-function reference to a linked definition of another function type, its arguments
//! passed by position.
//!
//! Metal resolves a reference to the linked function of its name whatever the two types are, and
//! makes the pipeline: the probe `linked_function_signatures.m` (the conformance crate's
//! `probes/`) on an M3 Max under macOS 27.0.1, and the consumer's probes on a Radeon Pro 5300M and
//! a UHD Graphics 630 under macOS 26.7.1. The call passes what it has by position. [`adapter`]
//! makes a function of the reference's type that calls the definition so; every call to the
//! reference's stub then calls it, and inlining takes it like any other callee:
//!
//! - an argument past the definition's last parameter is dropped (Metal on all three);
//! - a parameter past the reference's last argument reads zero (Metal read zero on all three);
//! - an argument of another plain-value type (an integer or floating-point scalar, or a vector of
//!   them, a boolean as a 1-bit integer) for a plain-value parameter passes its bits: of the same
//!   width, all of them (Metal on all three); for a narrower parameter, its low bits (the 5300M's
//!   answer, where the M3 and the UHD 630 leave the parameter undefined); for a wider one, its
//!   bits with zeros above (the M3 leaves it undefined);
//! - the result passes back the same way, from the definition's return type to the reference's:
//!   of the same width, its bits (Metal on all three); otherwise its low bits or its bits with
//!   zeros above (the M3 leaves it undefined). A definition returning nothing returns zero (the
//!   M3 leaves it undefined), and a reference returning nothing drops the result (the M3 runs
//!   the call).
//!
//! No adapter passes a pointer for a value or a value for a pointer, a pointer of another
//! address space, or an aggregate of another type, and none adapts a variadic type: Metal makes
//! those pipelines too, and the reference is refused (`Feature::Linking`, not implemented).

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{
    Arena, ArgIdx, Block, BlockId, Body, Call, CallConv, Callee, CastOp, Const, ConstTable,
    Constant, FastMath, FuncId, Function, Inst, InstId, InstKind, Linkage, Param, Sym, TailKind,
    Type, TypeId, TypeTable, Value,
};

use super::rewrite::invariant;

/// How one value passes from a type to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// The types are one: the value itself.
    Same,
    /// Plain values of one width: its bits, `bitcast`.
    Bits,
    /// Plain values of other widths, both at most 64 bits: its bits as an integer of the source's
    /// width, truncated or zero-extended to the destination's, as the destination type.
    Resized {
        /// The source's width.
        from: u8,
        /// The destination's width.
        to: u8,
    },
}

/// How the definition's result passes to the reference's.
#[derive(Clone, Copy, Debug)]
enum Returned {
    /// As a value, from the definition's return type.
    Value(Pass),
    /// The reference returns nothing: the result is dropped.
    Dropped,
    /// The definition returns nothing: the zero of the reference's return type.
    Zero,
}

/// The bit width of `ty` when it is a plain value: an integer of 1, 8, 16, 32 or 64 bits, a
/// floating-point scalar, or a vector of them (of integers of 8 bits or more).
fn plain_bits(types: &TypeTable, ty: TypeId) -> Option<u64> {
    let scalar = |t: &Type, boolean: bool| match t {
        Type::Int(1) if boolean => Some(1u64),
        Type::Int(b @ (8 | 16 | 32 | 64)) => Some(u64::from(*b)),
        t => t.float_bits().map(u64::from),
    };
    match types.get(ty)? {
        Type::Vector { elem, len } => {
            scalar(types.get(*elem)?, false)?.checked_mul(u64::from(*len))
        }
        t => scalar(t, true),
    }
}

/// How a value of type `from` passes to type `to`, if it can.
fn pass(types: &TypeTable, from: TypeId, to: TypeId) -> Option<Pass> {
    if from == to {
        return Some(Pass::Same);
    }
    let (f, t) = (plain_bits(types, from)?, plain_bits(types, to)?);
    if f == t {
        return Some(Pass::Bits);
    }
    match (u8::try_from(f), u8::try_from(t)) {
        (Ok(f), Ok(t)) if f <= 64 && t <= 64 => Some(Pass::Resized { from: f, to: t }),
        _ => None,
    }
}

/// What the adapter does, decided before anything is interned.
struct Plan {
    want_ret: TypeId,
    got_ret: TypeId,
    /// The definition's parameters, each with the reference's argument type at its position and
    /// how it passes, or `None` past the reference's last argument.
    params: Vec<(TypeId, Option<(TypeId, Pass)>)>,
    /// The number of the reference's parameters.
    arity: usize,
    returned: Returned,
}

/// The plan of an adapter from the reference's type `want` to the definition's `got`, or `None`
/// when no adapter passes them.
fn plan(
    types: &TypeTable,
    want: TypeId,
    got: TypeId,
    m: &mut Meter,
) -> Result<Option<Plan>, Error> {
    let fn_parts = |ty: TypeId| match types.get(ty) {
        Some(Type::Function {
            ret,
            params,
            vararg,
        }) => Ok((*ret, params, *vararg)),
        _ => Err(invariant("a function's type is not a function type")),
    };
    let (want_ret, want_params, want_vararg) = fn_parts(want)?;
    let (got_ret, got_params, got_vararg) = fn_parts(got)?;
    if want_vararg || got_vararg {
        return Ok(None);
    }
    let void = |t: TypeId| matches!(types.get(t), Some(Type::Void));
    let returned = match (void(want_ret), void(got_ret)) {
        (true, _) => Returned::Dropped,
        (false, true) => Returned::Zero,
        (false, false) => match pass(types, got_ret, want_ret) {
            Some(p) => Returned::Value(p),
            None => return Ok(None),
        },
    };
    let mut params = Vec::new();
    m.reserve(&mut params, got_params.len())?;
    for (i, &to) in got_params.iter().enumerate() {
        m.tick(1)?;
        let arg = match want_params.get(i) {
            Some(&from) => match pass(types, from, to) {
                Some(p) => Some((from, p)),
                None => return Ok(None),
            },
            None => None,
        };
        params.push((to, arg));
    }
    Ok(Some(Plan {
        want_ret,
        got_ret,
        params,
        arity: want_params.len(),
        returned,
    }))
}

/// A function of the reference's type `want`, named `name` as its stub is, that calls `def`, of
/// type `got`, with its arguments by position and returns the result (the module docs), or
/// `None` when the types differ in a way no adapter passes. Its types and constants are interned
/// into `types` and `consts`; the caller places the function.
pub(super) fn adapter(
    (name, want): (Sym, TypeId),
    (def, got): (FuncId, TypeId),
    (types, consts): (&mut TypeTable, &mut ConstTable),
    m: &mut Meter,
) -> Result<Option<Function>, Error> {
    let Some(plan) = plan(types, want, got, m)? else {
        return Ok(None);
    };
    let mut body = Emit {
        insts: Arena::new(),
        types,
        consts,
    };
    let mut args = Vec::new();
    m.reserve(&mut args, plan.params.len())?;
    for (i, &(to, arg)) in plan.params.iter().enumerate() {
        let value = match arg {
            Some((from, p)) => {
                let index = u32::try_from(i).map_err(|_| invariant("a parameter's index"))?;
                body.pass(Value::Arg(ArgIdx::new(index)), (from, to), p, m)?
            }
            None => body.zero(to, m)?,
        };
        args.push(value);
    }
    let call = Call {
        callee: Callee::Direct(def),
        args: args.into_boxed_slice(),
        fmf: FastMath::NONE,
        tail: TailKind::None,
        attrs: None,
        range: None,
    };
    let result = body.push(plan.got_ret, InstKind::Call(Box::new(call)), m)?;
    let ret = match plan.returned {
        Returned::Dropped => None,
        Returned::Zero => Some(body.zero(plan.want_ret, m)?),
        Returned::Value(p) => Some(body.pass(result, (plan.got_ret, plan.want_ret), p, m)?),
    };
    let void = body.types.intern(Type::Void, m)?;
    body.push(void, InstKind::Ret(ret), m)?;
    let insts = body.insts;
    let end = u32::try_from(insts.len()).map_err(|_| invariant("an adapter's length"))?;
    let mut blocks = Arena::<BlockId, Block>::new();
    blocks.push(
        Block {
            start: InstId::new(0),
            end: InstId::new(end),
            name: None,
        },
        m,
    )?;
    let mut params = Vec::new();
    m.reserve(&mut params, plan.arity)?;
    params.resize(plan.arity, Param::default());
    Ok(Some(Function {
        name,
        ty: want,
        linkage: Linkage::Internal,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        cconv: CallConv::C,
        attrs: Default::default(),
        ret_attrs: Default::default(),
        params: params.into_boxed_slice(),
        body: Some(Body { blocks, insts }),
        section: None,
        align: None,
    }))
}

/// The adapter's instructions as they are written, with the tables its casts and constants are
/// interned into.
struct Emit<'t> {
    insts: Arena<InstId, Inst>,
    types: &'t mut TypeTable,
    consts: &'t mut ConstTable,
}

impl Emit<'_> {
    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        m.tick(1)?;
        self.insts.push(Inst { ty, kind }, m).map(Value::Inst)
    }

    fn cast(&mut self, op: CastOp, v: Value, to: TypeId, m: &mut Meter) -> Result<Value, Error> {
        self.push(to, InstKind::Cast { op, v }, m)
    }

    /// The zero of `ty`.
    fn zero(&mut self, ty: TypeId, m: &mut Meter) -> Result<Value, Error> {
        let zero = Constant {
            ty,
            value: Const::Null,
        };
        self.consts.intern(zero, m).map(Value::Const)
    }

    /// `v`, of type `from`, passed to type `to` as `p` says.
    fn pass(
        &mut self,
        v: Value,
        (from, to): (TypeId, TypeId),
        p: Pass,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        match p {
            Pass::Same => Ok(v),
            Pass::Bits => self.cast(CastOp::Bitcast, v, to, m),
            Pass::Resized { from: f, to: t } => {
                let source = self.types.intern(Type::Int(f), m)?;
                let dest = self.types.intern(Type::Int(t), m)?;
                let bits = if from == source {
                    v
                } else {
                    self.cast(CastOp::Bitcast, v, source, m)?
                };
                let op = if t < f { CastOp::Trunc } else { CastOp::ZExt };
                let resized = self.cast(op, bits, dest, m)?;
                if to == dest {
                    Ok(resized)
                } else {
                    self.cast(CastOp::Bitcast, resized, to, m)
                }
            }
        }
    }
}
