//! `half` values held as 16-bit floats, in a module that computes `half` in 16 bits under the
//! 16-bit float controls.
//!
//! The lowering holds a `half` widened in an `f32` (`lower::arith::narrow`): a 16-bit `+ - * /`
//! or `sqrt` narrows its operands, computes and widens its result, and every other operation on `half`
//! values (a minimum, a select, a dot product, a store to a demotion variable or a thread array)
//! reads the widened `f32`. Where the module computes in 16 bits under the float controls
//! (`narrow::half_controls`, `narrow::computes_in_16_bits`), the device's native form of `half`
//! arithmetic is its 16-bit instruction, so [`hold`] holds those values at 16 bits, from the
//! module's structure alone:
//!
//! - **Exact halves.** An `f32` scalar or vector is an exact half where it is the widening of a
//!   16-bit float (`OpFConvert`), a constant whose every lane is a `half` (not a NaN), an undefined
//!   value, a load of a held variable (below), the member a composite extract takes from the
//!   composite construct of it, or a select, a move (copy, composite construct, extract or insert,
//!   shuffle, dynamic extract or insert), a negation, an absolute value, a minimum, a maximum, a
//!   clamp, a sign or a rounding to an integral value (`Floor`, `Ceil`, `Trunc`, `Round`,
//!   `RoundEven`) of exact halves, each of which answers a `half` exactly. Each gets a 16-bit twin
//!   computed by the same instruction on its operands' twins; a use of it that needs the `f32`
//!   reads one widening of the twin.
//! - **Narrowings.** A narrowing (`OpFConvert` to 16 bits) of an exact half is its twin: exact, it
//!   leaves the module.
//! - **Held variables.** A `Function` variable, or a `Private` one, of an `f32` scalar or vector or
//!   an array of them, that only loads and stores of a scalar or vector reach (through access
//!   chains or not), every store of an exact half, holds the 16-bit type (an optimistic fixed
//!   point, for the loop-carried values the emitter demotes).
//! - **Rounded operations.** A sum, difference, product, product by a scalar, dot product or `mix`
//!   of exact halves whose every use narrows it to 16 bits (a `half` computed in `float` and
//!   rounded) is the device's 16-bit instruction. A sum, difference and product are correctly
//!   rounded under the float controls, the same `half` as the `float` one rounded once (`f32`
//!   carries at least `2p + 2` bits); a dot product and a `mix` are the device's 16-bit `OpDot`
//!   and `FMix`, within the bounds `PrecisionNote::NativeDot` and `PrecisionNote::NativeMix`
//!   record for them, which every `half` one records.
//! - A comparison, a NaN or infinity test, or a conversion to an integer of exact halves reads
//!   their twins.
//!
//! What no instruction reads afterwards leaves the module (`spirv::dead`). Passes over the module
//! for its variables and uses, fixed points over its values and variables (each round but the
//! last settles one more), and one rebuild of each block, metered. The byte ledger holds what the
//! pass holds: its tables over the module's values, blocks and variables and its maps of twins
//! while it runs; each rebuilt block's instructions, read one at a time from the block they
//! replace, beside it until they take its place; and what the module's values, constants, types
//! and declarations grow by, which stays charged with the module.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::{tree_bytes, vec_bytes, Growth, Meter};
use crate::error::{Error, InternalError, InternalKind};
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::spirv::ids::{BlockId, ConstId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{
    BinOp, ConstBits, ConstKind, Constant, Contraction, ConvOp, Decl, GlobalKind, GlslOp, Inst,
    InstNode, Phi, StorageClass, Type, TypedModule, UnOp, ValueDef, ValueOrigin,
};

/// What [`hold`] changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Held {
    /// The narrowings it removed.
    pub narrowings: usize,
    /// The operations computed in `float` and rounded that it computes at 16 bits.
    pub rounded: usize,
    /// The variables it holds at 16 bits.
    pub variables: usize,
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

fn units(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Charge `bytes` to the ledger for a table [`hold`] holds until it returns, counted in `ledger`.
fn charge(meter: &mut Meter, ledger: &mut usize, bytes: usize) -> Result<(), Error> {
    meter.charge(bytes)?;
    *ledger = ledger.saturating_add(bytes);
    Ok(())
}

/// Return `bytes` that [`charge`] counted in `ledger`, for a table dropped before [`hold`] returns.
fn discharge(meter: &mut Meter, ledger: &mut usize, bytes: usize) {
    meter.release(bytes);
    *ledger = ledger.saturating_sub(bytes);
}

/// The heap bytes of `insts`: their vector's, and their boxed operand lists'.
fn insts_bytes(insts: &Vec<InstNode>) -> usize {
    insts.iter().fold(vec_bytes(insts), |sum, n| {
        sum.saturating_add(n.inst.heap_bytes())
    })
}

/// The heap bytes of `phis`: their vector's, and their incoming lists'.
fn phis_bytes(phis: &Vec<Phi>) -> usize {
    phis.iter().fold(vec_bytes(phis), |sum, p| {
        sum.saturating_add(vec_bytes(&p.incoming))
    })
}

/// The heap bytes of the module's tables [`hold`] adds to besides its blocks: its values, its
/// constants and their values, its types and its declarations.
fn tables_bytes(m: &TypedModule) -> usize {
    vec_bytes(&m.values)
        .saturating_add(vec_bytes(&m.const_values))
        .saturating_add(vec_bytes(&m.decls))
        .saturating_add(m.types.footprint())
        .saturating_add(m.consts.footprint())
}

/// The bits of the `half` equal to the `f32` of bits `bits`, where one is (a NaN is not).
pub fn half_bits(bits: u32) -> Option<u16> {
    let x = f32::from_bits(bits);
    let h = f32_to_f16_bits(x);
    (!x.is_nan() && f16_bits_to_f32(h).to_bits() == bits).then_some(h)
}

fn flag(v: &[bool], x: Value) -> bool {
    v.get(x.index()).copied().unwrap_or(false)
}

fn raise(v: &mut [bool], x: Value) {
    if let Some(f) = v.get_mut(x.index()) {
        *f = true;
    }
}

fn count(v: &[u32], x: Value) -> u32 {
    v.get(x.index()).copied().unwrap_or(0)
}

fn bump(v: &mut [u32], x: Value) {
    if let Some(c) = v.get_mut(x.index()) {
        *c += 1;
    }
}

/// The width of float scalar or vector type `ty`.
fn float_width(m: &TypedModule, ty: TypeId) -> Option<u8> {
    match m.types.get(ty).ok()? {
        Type::Float { width } => Some(*width),
        Type::Vector { elem, .. } => match m.types.get(*elem).ok()? {
            Type::Float { width } => Some(*width),
            _ => None,
        },
        _ => None,
    }
}

fn width_of(m: &TypedModule, v: Value) -> Option<u8> {
    float_width(m, m.values.get(v.index())?.ty)
}

fn is_f32(m: &TypedModule, v: Value) -> bool {
    width_of(m, v) == Some(32)
}

fn is_f16(m: &TypedModule, v: Value) -> bool {
    width_of(m, v) == Some(16)
}

/// Whether `ty` is an `f32` scalar or vector, or an implicitly laid out array of them: the types
/// a held variable may hold.
fn holdable(m: &TypedModule, ty: TypeId) -> bool {
    match m.types.get(ty) {
        Ok(Type::Array {
            elem, stride: None, ..
        }) => holdable(m, *elem),
        _ => float_width(m, ty) == Some(32),
    }
}

/// Whether constant `c` is a plain one of a holdable type with every lane a `half`.
fn half_const(m: &TypedModule, c: ConstId) -> bool {
    let Ok(def) = m.consts.get(c) else {
        return false;
    };
    let ConstKind::Plain(constant) = &def.kind else {
        return false;
    };
    match constant {
        Constant::Scalar { ty, bits } => {
            matches!(m.types.get(*ty), Ok(Type::Float { width: 32 }))
                && u32::try_from(bits.0).ok().and_then(half_bits).is_some()
        }
        Constant::Composite { ty, parts } => {
            holdable(m, *ty) && parts.iter().all(|p| half_const(m, *p))
        }
        Constant::Null { ty } => holdable(m, *ty),
    }
}

/// Whether the GLSL operation `op` answers a `half` exactly of `half` operands.
fn exact_glsl(op: GlslOp) -> bool {
    matches!(
        op,
        GlslOp::FAbs
            | GlslOp::FSign
            | GlslOp::Floor
            | GlslOp::Ceil
            | GlslOp::Trunc
            | GlslOp::Round
            | GlslOp::RoundEven
            | GlslOp::FMin
            | GlslOp::FMax
            | GlslOp::FClamp
            | GlslOp::NMin
            | GlslOp::NMax
            | GlslOp::NClamp
    )
}

/// The operands a move or an exact operation reads as halves; `None` for any other instruction.
fn moved_operands(inst: &Inst) -> Option<Vec<Value>> {
    Some(match inst {
        Inst::Unary {
            op: UnOp::FNegate | UnOp::CopyObject,
            a,
        } => vec![*a],
        Inst::Select { t, f, .. } => vec![*t, *f],
        Inst::CompositeConstruct { parts } => parts.to_vec(),
        Inst::CompositeExtract { composite, .. } => vec![*composite],
        Inst::CompositeInsert {
            object, composite, ..
        } => vec![*object, *composite],
        Inst::VectorShuffle { a, b, .. } => vec![*a, *b],
        Inst::VectorExtractDynamic { vector, .. } => vec![*vector],
        Inst::VectorInsertDynamic {
            vector, component, ..
        } => vec![*vector, *component],
        Inst::Glsl { op, args, .. } if exact_glsl(*op) => args.to_vec(),
        _ => return None,
    })
}

/// The operands of an operation [`hold`] may compute at 16 bits in place of its `float` one
/// rounded; `None` for any other instruction.
fn rounded_operands(inst: &Inst) -> Option<Vec<Value>> {
    Some(match inst {
        Inst::Binary {
            op: BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::VectorTimesScalar | BinOp::Dot,
            a,
            b,
            ..
        } => vec![*a, *b],
        Inst::Glsl {
            op: GlslOp::FMix,
            args,
            ..
        } => args.to_vec(),
        _ => return None,
    })
}

/// Whether `inst`, whose float operands are all halves, answers the same of their twins: a float
/// comparison, a NaN or infinity test, or a conversion to an integer.
fn reads_twins(inst: &Inst) -> bool {
    match inst {
        Inst::Binary { op, .. } => matches!(
            op,
            BinOp::FOrdEqual
                | BinOp::FUnordEqual
                | BinOp::FOrdNotEqual
                | BinOp::FUnordNotEqual
                | BinOp::FOrdLessThan
                | BinOp::FUnordLessThan
                | BinOp::FOrdGreaterThan
                | BinOp::FUnordGreaterThan
                | BinOp::FOrdLessThanEqual
                | BinOp::FUnordLessThanEqual
                | BinOp::FOrdGreaterThanEqual
                | BinOp::FUnordGreaterThanEqual
        ),
        Inst::Unary {
            op: UnOp::IsNan | UnOp::IsInf,
            ..
        }
        | Inst::Convert {
            op: ConvOp::ConvertFToS | ConvOp::ConvertFToU,
            ..
        } => true,
        _ => false,
    }
}

/// The float operands of `inst`.
fn float_operands(m: &TypedModule, inst: &Inst) -> Vec<Value> {
    let mut out = Vec::new();
    inst.for_each_value(|v| {
        if width_of(m, v).is_some() {
            out.push(v);
        }
    });
    out
}

/// A variable [`hold`] may hold at 16 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Root {
    Local(VarId),
    Global(GlobalId),
}

/// What [`hold`] reads of the module before it rewrites it.
struct Facts {
    /// Per value: an exact half.
    half: Vec<bool>,
    /// Per value: an operation computed in `float` and rounded, computed at 16 bits.
    rounded: Vec<bool>,
    /// Per value: its uses that read the `f32`.
    wide: Vec<u32>,
    /// The pointers into held variables (each variable's own and every access chain into it).
    places: BTreeMap<Value, Root>,
    /// The held variables.
    held: BTreeSet<Root>,
    /// The members composite extracts take from composite constructs: the extract's result, the
    /// part it is.
    folds: BTreeMap<Value, Value>,
    /// Whether some narrowing leaves.
    narrowed: bool,
}

/// The pointer values of the variables `m` may hold, and of every access chain into them.
fn places(m: &TypedModule, meter: &mut Meter) -> Result<BTreeMap<Value, Root>, Error> {
    let init_half = |init: Option<ConstId>| init.is_none_or(|c| half_const(m, c));
    let mut places = BTreeMap::new();
    for (index, var) in m.vars.iter().enumerate() {
        if holdable(m, var.pointee) && init_half(var.init) {
            let id = u32::try_from(index).map_err(|_| invariant("variable ids"))?;
            places.insert(var.value, Root::Local(VarId::new(id)));
        }
    }
    for (index, g) in m.globals.iter().enumerate() {
        if g.class == StorageClass::Private
            && g.kind == GlobalKind::Plain
            && holdable(m, g.pointee)
            && init_half(g.init)
        {
            let id = u32::try_from(index).map_err(|_| invariant("global ids"))?;
            places.insert(g.value, Root::Global(GlobalId::new(id)));
        }
    }
    if places.is_empty() {
        return Ok(places);
    }
    // Access chains into them, in as many rounds as chains of chains out of layout order need.
    loop {
        let mut changed = false;
        for block in &m.blocks {
            meter.tick(units(block.insts.len()))?;
            for node in &block.insts {
                if let (Inst::AccessChain { base, .. }, Some(r)) = (&node.inst, node.result) {
                    if let Some(root) = places.get(base).copied() {
                        changed |= places.insert(r, root).is_none();
                    }
                }
            }
        }
        if !changed {
            return Ok(places);
        }
    }
}

/// The variables of `places` that a use other than a load or store of a scalar or vector, or an
/// access chain, reaches.
fn escaped(m: &TypedModule, places: &BTreeMap<Value, Root>) -> BTreeSet<Root> {
    let mut out = BTreeSet::new();
    let mut escape = |v: Value| {
        if let Some(root) = places.get(&v) {
            out.insert(*root);
        }
    };
    for block in &m.blocks {
        for node in &block.insts {
            let kept = match &node.inst {
                Inst::AccessChain { base, indices, .. } if places.contains_key(base) => {
                    indices.iter().for_each(|v| escape(*v));
                    true
                }
                Inst::Load { ptr, .. } => {
                    places.contains_key(ptr) && node.result.is_some_and(|r| is_f32(m, r))
                }
                Inst::Store { ptr, value, .. } => places.contains_key(ptr) && is_f32(m, *value),
                _ => false,
            };
            if !kept {
                node.inst.for_each_value(&mut escape);
            }
        }
        for phi in &block.phis {
            phi.incoming.iter().for_each(|(v, _)| escape(*v));
        }
        block.term.for_each_value(&mut escape);
    }
    out
}

/// The layout position of each block, by its id.
fn positions(m: &TypedModule) -> Vec<usize> {
    let mut pos = vec![usize::MAX; m.blocks.len()];
    for (i, b) in m.functions.iter().flat_map(|f| &f.blocks).enumerate() {
        if let Some(p) = pos.get_mut(b.index()) {
            *p = i;
        }
    }
    pos
}

/// The members composite extracts of one index take from composite constructs whose parts are
/// their members (a structure's or an array's, or a vector's of scalars): each extract's result,
/// with the `f32` part it is. The map stays charged in `ledger`; the constructs' while they are
/// read.
fn folds(
    m: &TypedModule,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<BTreeMap<Value, Value>, Error> {
    let ty_of = |v: Value| m.values.get(v.index()).and_then(|d| m.types.get(d.ty).ok());
    let mut constructs: BTreeMap<Value, &[Value]> = BTreeMap::new();
    for node in m.blocks.iter().flat_map(|b| &b.insts) {
        if let (Inst::CompositeConstruct { parts }, Some(r)) = (&node.inst, node.result) {
            let members = match ty_of(r) {
                Some(Type::Struct { .. } | Type::Array { .. }) => true,
                Some(Type::Vector { .. }) => parts
                    .iter()
                    .all(|p| matches!(ty_of(*p), Some(Type::Float { .. }))),
                _ => false,
            };
            if members {
                constructs.insert(r, parts);
            }
        }
    }
    let mut out = BTreeMap::new();
    if constructs.is_empty() {
        return Ok(out);
    }
    let read = tree_bytes(constructs.len(), std::mem::size_of::<(Value, &[Value])>());
    charge(meter, ledger, read)?;
    for node in m.blocks.iter().flat_map(|b| &b.insts) {
        if let (Inst::CompositeExtract { composite, indices }, Some(r)) = (&node.inst, node.result)
        {
            let part = match (constructs.get(composite), indices.as_ref()) {
                (Some(parts), [i]) => usize::try_from(*i).ok().and_then(|i| parts.get(i)),
                _ => None,
            };
            if let Some(p) = part.filter(|p| is_f32(m, **p)) {
                out.insert(r, *p);
            }
        }
    }
    discharge(meter, ledger, read);
    charge(
        meter,
        ledger,
        tree_bytes(out.len(), std::mem::size_of::<(Value, Value)>()),
    )?;
    Ok(out)
}

/// The exact halves of `m` with the variables of `places` held: a least fixed point over its
/// instructions and phis, in rounds until none changes.
fn exact_halves(
    m: &TypedModule,
    places: &BTreeMap<Value, Root>,
    folds: &BTreeMap<Value, Value>,
    pos: &[usize],
    meter: &mut Meter,
) -> Result<Vec<bool>, Error> {
    let mut half = vec![false; m.values.len()];
    for (index, def) in m.values.iter().enumerate() {
        if let ValueOrigin::Const(c) = def.origin {
            let v = Value::new(u32::try_from(index).map_err(|_| invariant("value ids"))?);
            if is_f32(m, v) && half_const(m, c) {
                raise(&mut half, v);
            }
        }
    }
    // A phi's twin reads its incoming values' twins, so each comes before it in the layout.
    let earlier = |v: Value, at: usize| match m.values.get(v.index()).map(|d| d.origin) {
        Some(ValueOrigin::Const(_)) => true,
        Some(ValueOrigin::Inst(b) | ValueOrigin::Phi(b)) => {
            pos.get(b.index()).copied().unwrap_or(usize::MAX) < at
        }
        _ => false,
    };
    loop {
        let mut changed = false;
        for (b, block) in m.blocks.iter().enumerate() {
            meter.tick(units(block.insts.len()))?;
            let at = pos.get(b).copied().unwrap_or(usize::MAX);
            for phi in &block.phis {
                let r = phi.result;
                if flag(&half, r) || !is_f32(m, r) {
                    continue;
                }
                let all = phi
                    .incoming
                    .iter()
                    .all(|(v, _)| flag(&half, *v) && earlier(*v, at));
                if all {
                    raise(&mut half, r);
                    changed = true;
                }
            }
            for node in &block.insts {
                let Some(r) = node.result.filter(|r| is_f32(m, *r) && !flag(&half, *r)) else {
                    continue;
                };
                let is = match (&node.inst, folds.get(&r)) {
                    (_, Some(part)) => flag(&half, *part),
                    (
                        Inst::Convert {
                            op: ConvOp::FConvert,
                            a,
                            ..
                        },
                        None,
                    ) => is_f16(m, *a),
                    (Inst::Load { ptr, .. }, None) => places.contains_key(ptr),
                    (Inst::Undef, None) => true,
                    (inst, None) => moved_operands(inst)
                        .is_some_and(|operands| operands.iter().all(|v| flag(&half, *v))),
                };
                if is {
                    raise(&mut half, r);
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(half);
        }
    }
}

/// What [`hold`] reads of `m`, its tables charged in `ledger` (those it drops before it returns
/// released as they go).
fn analyse(m: &TypedModule, meter: &mut Meter, ledger: &mut usize) -> Result<Facts, Error> {
    let roots = |len: usize| tree_bytes(len, std::mem::size_of::<Root>());
    let mut places = places(m, meter)?;
    charge(
        meter,
        ledger,
        tree_bytes(places.len(), std::mem::size_of::<(Value, Root)>()),
    )?;
    let gone = escaped(m, &places);
    charge(meter, ledger, roots(gone.len()))?;
    places.retain(|_, root| !gone.contains(root));
    discharge(meter, ledger, roots(gone.len()));
    let folds = folds(m, meter, ledger)?;
    let n = m.values.len();
    charge(
        meter,
        ledger,
        m.blocks.len().saturating_mul(std::mem::size_of::<usize>()),
    )?;
    let pos = positions(m);
    // The exact halves with every candidate held, then without each one a store of another value
    // refutes, until none does: each round holds fewer variables, and one table of exact halves.
    charge(meter, ledger, n.saturating_mul(std::mem::size_of::<bool>()))?;
    let half = loop {
        let half = exact_halves(m, &places, &folds, &pos, meter)?;
        let mut refuted = BTreeSet::new();
        for node in m.blocks.iter().flat_map(|b| &b.insts) {
            if let Inst::Store { ptr, value, .. } = &node.inst {
                if let Some(root) = places.get(ptr).filter(|_| !flag(&half, *value)) {
                    refuted.insert(*root);
                }
            }
        }
        if refuted.is_empty() {
            break half;
        }
        charge(meter, ledger, roots(refuted.len()))?;
        places.retain(|_, root| !refuted.contains(root));
        discharge(meter, ledger, roots(refuted.len()));
    };
    // Each value's uses that read the `f32`: not a narrowing's, nor one that reads its twin.
    let counts = n.saturating_mul(std::mem::size_of::<u32>());
    charge(meter, ledger, counts)?;
    let mut wide = vec![0u32; n];
    charge(meter, ledger, counts)?;
    let mut narrows = vec![0u32; n];
    for block in &m.blocks {
        meter.tick(units(block.insts.len()))?;
        for node in &block.insts {
            let twins = match &node.inst {
                Inst::Convert {
                    op: ConvOp::FConvert,
                    a,
                    ..
                } if is_f32(m, *a) && node.result.is_some_and(|r| is_f16(m, r)) => {
                    bump(&mut narrows, *a);
                    continue;
                }
                Inst::Load { ptr, .. } | Inst::Store { ptr, .. } if places.contains_key(ptr) => {
                    true
                }
                Inst::AccessChain { base, .. } if places.contains_key(base) => true,
                inst if reads_twins(inst) => {
                    float_operands(m, inst).iter().all(|v| flag(&half, *v))
                }
                // An exact half's own instruction reads its operands' twins, but a member taken
                // from its construct, which stays.
                _ => node
                    .result
                    .is_some_and(|r| flag(&half, r) && !folds.contains_key(&r)),
            };
            if !twins {
                node.inst.for_each_value(|v| bump(&mut wide, v));
            }
        }
        for phi in &block.phis {
            phi.incoming.iter().for_each(|(v, _)| bump(&mut wide, *v));
        }
        block.term.for_each_value(|v| bump(&mut wide, v));
    }
    // The operations computed in `float` of exact halves, every use a narrowing.
    charge(meter, ledger, n.saturating_mul(std::mem::size_of::<bool>()))?;
    let mut rounded = vec![false; n];
    for node in m.blocks.iter().flat_map(|b| &b.insts) {
        let Some(r) = node.result.filter(|r| is_f32(m, *r)) else {
            continue;
        };
        let of_halves = rounded_operands(&node.inst)
            .is_some_and(|operands| operands.iter().all(|v| flag(&half, *v)));
        if of_halves && count(&wide, r) == 0 && count(&narrows, r) > 0 {
            raise(&mut rounded, r);
        }
    }
    let narrowed = narrows
        .iter()
        .zip(half.iter().zip(&rounded))
        .any(|(n, (h, r))| *n > 0 && (*h || *r));
    discharge(meter, ledger, counts);
    let held: BTreeSet<Root> = places.values().copied().collect();
    charge(meter, ledger, roots(held.len()))?;
    Ok(Facts {
        half,
        rounded,
        wide,
        places,
        held,
        folds,
        narrowed,
    })
}

/// Intern type `t`, declaring it when new.
fn ty(m: &mut TypedModule, t: Type) -> Result<TypeId, Error> {
    let (id, fresh) = m.types.intern(t, &m.consts)?;
    if fresh {
        m.decls.push(Decl::Type(id));
    }
    Ok(id)
}

/// The 16-bit form of holdable type `t`, or of a pointer to one.
fn narrow_type(m: &mut TypedModule, t: TypeId) -> Result<TypeId, Error> {
    let narrowed = match m.types.get(t)?.clone() {
        Type::Float { width: 32 } => Type::Float { width: 16 },
        Type::Vector { elem, len } => Type::Vector {
            elem: narrow_type(m, elem)?,
            len,
        },
        Type::Array { elem, len, stride } => Type::Array {
            elem: narrow_type(m, elem)?,
            len,
            stride,
        },
        Type::Pointer { class, pointee } => Type::Pointer {
            class,
            pointee: narrow_type(m, pointee)?,
        },
        _ => return Err(invariant("a held value of a holdable type")),
    };
    ty(m, narrowed)
}

fn new_value(m: &mut TypedModule, ty: TypeId, origin: ValueOrigin) -> Result<Value, Error> {
    let v = Value::new(u32::try_from(m.values.len()).map_err(|_| invariant("value ids"))?);
    m.values.push(ValueDef { ty, origin });
    Ok(v)
}

/// Plain constant `c`, declared when new.
fn constant_id(m: &mut TypedModule, c: Constant) -> Result<ConstId, Error> {
    let ty = c.ty();
    let (id, fresh) = m.consts.intern(c, &m.types)?;
    if fresh {
        if id.index() != m.const_values.len() {
            return Err(invariant("constant ids"));
        }
        let v = new_value(m, ty, ValueOrigin::Const(id))?;
        m.const_values.push(v);
        m.decls.push(Decl::Const(id));
    }
    Ok(id)
}

/// The 16-bit twin of constant `c`, a plain one of a holdable type with every lane a `half`.
fn twin_const(m: &mut TypedModule, c: ConstId) -> Result<ConstId, Error> {
    let ConstKind::Plain(constant) = m.consts.get(c)?.kind.clone() else {
        return Err(invariant("a held constant is plain"));
    };
    let twin = match constant {
        Constant::Scalar { ty, bits } => {
            let h = u32::try_from(bits.0)
                .ok()
                .and_then(half_bits)
                .ok_or_else(|| invariant("a held constant is a half"))?;
            Constant::Scalar {
                ty: narrow_type(m, ty)?,
                bits: ConstBits(u64::from(h)),
            }
        }
        Constant::Composite { ty, parts } => {
            let mut twins = Vec::with_capacity(parts.len());
            for p in parts.iter() {
                twins.push(twin_const(m, *p)?);
            }
            Constant::Composite {
                ty: narrow_type(m, ty)?,
                parts: twins.into(),
            }
        }
        Constant::Null { ty } => Constant::Null {
            ty: narrow_type(m, ty)?,
        },
    };
    constant_id(m, twin)
}

/// The rewriting state: each exact half's and rounded value's 16-bit twin, and the twin that
/// replaces each narrowing removed.
struct Twins {
    twin: BTreeMap<Value, Value>,
    replaced: BTreeMap<Value, Value>,
}

impl Twins {
    /// The twin of exact half or rounded value `v`: its own, or a constant's.
    fn of(&mut self, m: &mut TypedModule, v: Value) -> Result<Value, Error> {
        if let Some(t) = self.twin.get(&v) {
            return Ok(*t);
        }
        match m.values.get(v.index()).map(|d| d.origin) {
            Some(ValueOrigin::Const(c)) => {
                let twin = twin_const(m, c)?;
                let t = m.const_value(twin)?;
                self.twin.insert(v, t);
                Ok(t)
            }
            _ => Err(invariant("an exact half has a twin before its use")),
        }
    }

    /// `inst` reading its `f32` operands' twins.
    fn twinned(&mut self, m: &mut TypedModule, inst: &Inst) -> Result<Inst, Error> {
        let mut map = BTreeMap::new();
        for v in float_operands(m, inst) {
            if is_f32(m, v) {
                map.insert(v, self.of(m, v)?);
            }
        }
        let mut inst = inst.clone();
        inst.for_each_value_mut(|v| {
            if let Some(t) = map.get(v) {
                *v = *t;
            }
        });
        Ok(inst)
    }
}

/// The widening of twin `t` to the `f32` value `r`: `NoContraction`, so no device folds it with a
/// rounding narrowing `t` may be.
fn widening(r: Value, t: Value) -> InstNode {
    InstNode {
        result: Some(r),
        inst: Inst::Convert {
            op: ConvOp::FConvert,
            a: t,
            contract: Contraction::Forbidden,
        },
    }
}

/// A new 16-bit value of `f32` value `r`'s shape, defined in `block`.
fn twin_value(m: &mut TypedModule, r: Value, block: BlockId) -> Result<Value, Error> {
    let wide = m.value(r)?.ty;
    let t = narrow_type(m, wide)?;
    new_value(m, t, ValueOrigin::Inst(block))
}

/// Hold variable `root` at 16 bits: its type, its pointer's and its initializer's.
fn hold_root(m: &mut TypedModule, root: Root) -> Result<(), Error> {
    let (pointee, init, value) = match root {
        Root::Local(v) => {
            let var = m
                .vars
                .get(v.index())
                .ok_or_else(|| invariant("a held variable"))?;
            (var.pointee, var.init, var.value)
        }
        Root::Global(g) => {
            let var = m
                .globals
                .get(g.index())
                .ok_or_else(|| invariant("a held global"))?;
            (var.pointee, var.init, var.value)
        }
    };
    let wide_ptr = m.value(value)?.ty;
    let pointee = narrow_type(m, pointee)?;
    let ptr_ty = narrow_type(m, wide_ptr)?;
    let init = match init {
        Some(c) => Some(twin_const(m, c)?),
        None => None,
    };
    match root {
        Root::Local(v) => {
            let var = m
                .vars
                .get_mut(v.index())
                .ok_or_else(|| invariant("a held variable"))?;
            (var.pointee, var.ptr_ty, var.init) = (pointee, ptr_ty, init);
        }
        Root::Global(g) => {
            let var = m
                .globals
                .get_mut(g.index())
                .ok_or_else(|| invariant("a held global"))?;
            (var.pointee, var.ptr_ty, var.init) = (pointee, ptr_ty, init);
            // Declared after its new types.
            m.decls.retain(|d| *d != Decl::Global(g));
            m.decls.push(Decl::Global(g));
        }
    }
    m.values
        .get_mut(value.index())
        .ok_or_else(|| invariant("a held variable's pointer"))?
        .ty = ptr_ty;
    Ok(())
}

/// Block `block` rebuilt with `facts`' values at 16 bits: its twin phis, and its instructions,
/// with the bytes they hold, which are charged. The rebuild changes the module's values, types,
/// constants and declarations, never a block, so it reads each phi and instruction from the block
/// as it goes, holding no copy of it.
fn rebuild(
    m: &mut TypedModule,
    facts: &Facts,
    tw: &mut Twins,
    held: &mut Held,
    block: BlockId,
    meter: &mut Meter,
) -> Result<(Vec<Phi>, Vec<InstNode>, usize), Error> {
    let (phis, len) = m
        .blocks
        .get(block.index())
        .map(|b| (b.phis.len(), b.insts.len()))
        .ok_or_else(|| invariant("a block"))?;
    // An exact half phi's twin beside it.
    let mut twin_phis = Vec::new();
    for at in 0..phis {
        let phi = m
            .blocks
            .get(block.index())
            .and_then(|b| b.phis.get(at))
            .filter(|p| flag(&facts.half, p.result))
            .cloned();
        let Some(phi) = phi else {
            continue;
        };
        let mut incoming = Vec::with_capacity(phi.incoming.len());
        for (v, from) in &phi.incoming {
            incoming.push((tw.of(m, *v)?, *from));
        }
        let wide = m.value(phi.result)?.ty;
        let t = narrow_type(m, wide)?;
        let t = new_value(m, t, ValueOrigin::Phi(block))?;
        tw.twin.insert(phi.result, t);
        twin_phis.push(Phi {
            result: t,
            incoming,
        });
    }
    let reserved = len.saturating_mul(std::mem::size_of::<InstNode>());
    meter.charge(reserved)?;
    let mut out: Vec<InstNode> = Vec::with_capacity(len);
    for at in 0..len {
        let node = m
            .blocks
            .get(block.index())
            .and_then(|b| b.insts.get(at))
            .cloned()
            .ok_or_else(|| invariant("an instruction"))?;
        let Some(r) = node.result else {
            match &node.inst {
                // A store into a held variable stores the twin.
                Inst::Store { ptr, value, access } if facts.places.contains_key(ptr) => {
                    let inst = Inst::Store {
                        ptr: *ptr,
                        value: tw.of(m, *value)?,
                        access: *access,
                    };
                    out.push(InstNode { result: None, inst });
                }
                _ => out.push(node),
            }
            continue;
        };
        let wide_uses = count(&facts.wide, r) > 0;
        match &node.inst {
            // A narrowing of an exact half or a rounded value: its twin.
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if is_f16(m, r) && (flag(&facts.half, *a) || flag(&facts.rounded, *a)) => {
                let t = tw.of(m, *a)?;
                tw.replaced.insert(r, t);
                held.narrowings += 1;
            }
            // A widening: its twin is what it widens.
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            } if flag(&facts.half, r) => {
                let t = tw.replaced.get(a).copied().unwrap_or(*a);
                tw.twin.insert(r, t);
                out.push(node);
            }
            // An access chain into a held variable points at 16 bits.
            Inst::AccessChain { base, .. } if facts.places.contains_key(base) => {
                let wide = m.value(r)?.ty;
                let t = narrow_type(m, wide)?;
                m.values
                    .get_mut(r.index())
                    .ok_or_else(|| invariant("an access chain"))?
                    .ty = t;
                out.push(node);
            }
            // A load of a held variable loads the twin.
            Inst::Load { ptr, access } if facts.places.contains_key(ptr) => {
                let t = twin_value(m, r, block)?;
                let inst = Inst::Load {
                    ptr: *ptr,
                    access: *access,
                };
                out.push(InstNode {
                    result: Some(t),
                    inst,
                });
                tw.twin.insert(r, t);
                if wide_uses {
                    out.push(widening(r, t));
                }
            }
            // A member taken from its composite construct: the member's twin.
            _ if flag(&facts.half, r) && facts.folds.contains_key(&r) => {
                let part = facts
                    .folds
                    .get(&r)
                    .copied()
                    .ok_or_else(|| invariant("a fold"))?;
                let t = tw.of(m, part)?;
                tw.twin.insert(r, t);
                out.push(node);
            }
            // An undefined value's twin is the 16-bit undefined value; its `f32` stays undefined,
            // where a use reads it, as no widening of the twin is any more defined.
            Inst::Undef if flag(&facts.half, r) => {
                let t = twin_value(m, r, block)?;
                out.push(InstNode {
                    result: Some(t),
                    inst: Inst::Undef,
                });
                tw.twin.insert(r, t);
                if wide_uses {
                    out.push(node);
                }
            }
            // An exact half's own instruction on its operands' twins, or a rounded value's at 16
            // bits; the `f32` is the twin's widening.
            inst if flag(&facts.half, r) || flag(&facts.rounded, r) => {
                let inst = tw.twinned(m, inst)?;
                let t = twin_value(m, r, block)?;
                out.push(InstNode {
                    result: Some(t),
                    inst,
                });
                tw.twin.insert(r, t);
                if flag(&facts.rounded, r) {
                    held.rounded += 1;
                } else if wide_uses {
                    out.push(widening(r, t));
                }
            }
            // A comparison, test or conversion of exact halves reads their twins.
            inst if reads_twins(inst)
                && float_operands(m, inst)
                    .iter()
                    .all(|v| flag(&facts.half, *v)) =>
            {
                let inst = tw.twinned(m, inst)?;
                out.push(InstNode {
                    result: Some(r),
                    inst,
                });
            }
            _ => out.push(node),
        }
    }
    // What the rebuilt block holds past the reservation: the instructions' growth, their boxed
    // operand lists and the twin phis.
    let bytes = insts_bytes(&out).saturating_add(phis_bytes(&twin_phis));
    meter.charge(bytes.saturating_sub(reserved))?;
    Ok((twin_phis, out, bytes.max(reserved)))
}

/// Hold `m`'s `half` values at 16 bits (the module docs), where it computes `half` in 16 bits;
/// the caller asks only under the 16-bit float controls (`narrow::half_controls`): on a driver
/// that folds the conversion's round trip (NVIDIA's) a narrowing a held value reads at 16 bits is
/// no round trip. What no
/// instruction reads afterwards leaves the module, and its interface and requirements are derived
/// again.
pub fn hold(m: &mut TypedModule, meter: &mut Meter) -> Result<Held, Error> {
    if !crate::lower::arith::narrow::computes_in_16_bits(m) {
        return Ok(Held::default());
    }
    // The pass's tables and maps, charged while it holds them (the module docs).
    let mut ledger = 0usize;
    let r = analyse(m, meter, &mut ledger).and_then(|facts| rewrite(m, &facts, meter, &mut ledger));
    meter.release(ledger);
    let Some(held) = r? else {
        return Ok(Held::default());
    };
    crate::spirv::dead::remove_dead_keeping(m, &[], meter)?;
    Ok(held)
}

/// `m` with `facts`' values held at 16 bits, or `None` where it holds none and no narrowing
/// leaves. Its maps of twins are charged in `ledger`; the rebuilt blocks and what the module's
/// tables grow by stay charged, as the module's.
fn rewrite(
    m: &mut TypedModule,
    facts: &Facts,
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Option<Held>, Error> {
    if facts.held.is_empty() && !facts.narrowed {
        return Ok(None);
    }
    let mut tables = Growth::from_charged(tables_bytes(m));
    let mut held = Held::default();
    for root in &facts.held {
        hold_root(m, *root)?;
        held.variables += 1;
    }
    tables.charge_to(tables_bytes(m), meter)?;
    let mut tw = Twins {
        twin: BTreeMap::new(),
        replaced: BTreeMap::new(),
    };
    let order: Vec<BlockId> = m
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter().copied())
        .collect();
    charge(meter, ledger, vec_bytes(&order))?;
    let mut maps = 0usize;
    for block in order {
        let replaced = m
            .blocks
            .get(block.index())
            .map_or(0, |b| insts_bytes(&b.insts));
        let (twin_phis, insts, bytes) = rebuild(m, facts, &mut tw, &mut held, block, meter)?;
        // What the rebuild added to the module's tables and the maps, beside both blocks.
        tables.charge_to(tables_bytes(m), meter)?;
        let now = tree_bytes(
            tw.twin.len().saturating_add(tw.replaced.len()),
            std::mem::size_of::<(Value, Value)>(),
        );
        charge(meter, ledger, now.saturating_sub(maps))?;
        maps = maps.max(now);
        let b = m
            .blocks
            .get_mut(block.index())
            .ok_or_else(|| invariant("a block"))?;
        meter.tick(units(insts.len()))?;
        b.phis.extend(twin_phis);
        b.insts = insts;
        // The instructions replaced are gone; what the module was charged for them now holds the
        // rebuilt block, its growth charged on top.
        meter.release(replaced.min(bytes));
    }
    // Every read of a narrowing removed reads its twin.
    let replaced = tw.replaced;
    let swap = |v: &mut Value| {
        if let Some(t) = replaced.get(v) {
            *v = *t;
        }
    };
    for block in &mut m.blocks {
        for node in &mut block.insts {
            node.inst.for_each_value_mut(swap);
        }
        for phi in &mut block.phis {
            for (v, _) in &mut phi.incoming {
                swap(v);
            }
        }
        block.term.for_each_value_mut(swap);
    }
    Ok(Some(held))
}

#[cfg(test)]
mod tests {
    use super::{half_bits, hold};
    use crate::budget::{Budget, Meter};
    use crate::pipeline::sym::{AirUserKey, VaryingKey};
    use crate::spirv::builder::ModuleBuilder;
    use crate::spirv::ir::{
        BinOp, ConvOp, ExecModel, GlslOp, IoDecor, IoDir, MemAccess, Type, TypedModule,
    };
    use crate::tests::constant_blob_footprint::held_during;

    /// A fragment function that computes `half` in 16 bits as the lowering writes it, in one
    /// block: `FragCoord`'s x narrowed, widened and its minimum with itself taken in `f32` (an
    /// exact half, whose narrowing the builder writes), then `steps` times narrowed, doubled at 16
    /// bits, widened, and its minimum taken so; the last is its output.
    fn halves(steps: usize) -> TypedModule {
        let mut b = ModuleBuilder::new();
        let f32_ = b.ty(Type::Float { width: 32 }).expect("float");
        let f16_ = b.ty(Type::Float { width: 16 }).expect("half");
        let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 }).expect("vec4");
        let frag_coord = b
            .builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)
            .expect("FragCoord");
        let key = VaryingKey(AirUserKey("out".as_bytes().into()));
        let out = b
            .io(key, IoDir::Output, f32_, IoDecor::default())
            .expect("output");
        let void = b.ty(Type::Void).expect("void");
        let mut f = b.function(void).expect("main");
        let fc = f.module().global_value(frag_coord).expect("FragCoord");
        let position = f.load(fc, MemAccess::default()).expect("position");
        let x = f.composite_extract(position, &[0]).expect("x");
        let h = f.convert(ConvOp::FConvert, x, f16_).expect("narrowed");
        let w = f.convert(ConvOp::FConvert, h, f32_).expect("widened");
        let mut a = f.glsl(GlslOp::FMin, &[w, w]).expect("minimum");
        for _ in 0..steps {
            let n = f.convert(ConvOp::FConvert, a, f16_).expect("narrowing");
            let sum = f.binary(BinOp::FAdd, n, n).expect("16-bit sum");
            let w = f.convert(ConvOp::FConvert, sum, f32_).expect("widening");
            a = f.glsl(GlslOp::FMin, &[w, w]).expect("minimum");
        }
        let o = f.module().global_value(out).expect("output");
        f.store(o, a, MemAccess::default()).expect("store");
        f.ret(None).expect("ret");
        let main = f.finish().expect("finish");
        b.entry(ExecModel::Fragment, main, Vec::new())
            .expect("entry");
        b.finalize().expect("finalize")
    }

    /// What the pass holds beyond what it charges the byte ledger does not grow with the module:
    /// its tables and maps of twins, each rebuilt block (read from the block it replaces, never
    /// copied whole) and what the module's tables grow by are charged, so the translation's
    /// allocator peak stays within 10/7 of its ledger's. Every step's narrowing leaves, so the
    /// pass did its work at each size.
    #[test]
    fn what_the_pass_holds_uncharged_does_not_grow_with_the_module() {
        let measure = |steps: usize| {
            let mut m = halves(steps);
            let mut meter = Meter::new(&Budget::default());
            let (r, peak) = held_during(|| hold(&mut m, &mut meter));
            assert_eq!(r.expect("hold").narrowings, steps, "{steps} steps");
            let ledger = usize::try_from(meter.ledger_peak()).unwrap_or(usize::MAX);
            let peak = usize::try_from(peak).unwrap_or(usize::MAX);
            (peak, peak.saturating_sub(ledger))
        };
        let (small, small_uncharged) = measure(16);
        let (large, large_uncharged) = measure(4096);
        let grown = large.saturating_sub(small);
        let uncharged = large_uncharged.saturating_sub(small_uncharged);
        assert!(
            uncharged.saturating_mul(10) <= grown,
            "{uncharged} of the {grown} bytes the pass holds over 4,080 more steps are uncharged"
        );
    }

    #[test]
    fn half_bits_are_exact_or_none() {
        assert_eq!(half_bits(0x3f80_0000), Some(0x3c00));
        assert_eq!(half_bits(0x8000_0000), Some(0x8000));
        assert_eq!(half_bits(0x7f80_0000), Some(0x7c00));
        assert_eq!(half_bits(0x7fc0_0000), None);
        assert_eq!(half_bits(0x477f_e000), Some(0x7bff));
        assert_eq!(half_bits(0x4780_0000), None);
        // The smallest denormal, the largest, one with a bit past them, and a tenth.
        assert_eq!(half_bits(0x3380_0000), Some(0x0001));
        assert_eq!(half_bits(0x387f_c000), Some(0x03ff));
        assert_eq!(half_bits(0x3380_0001), None);
        assert_eq!(half_bits(0x3dcc_cccd), None);
    }
}
