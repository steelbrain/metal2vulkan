//! `matmul2d` and tensor objects as plain AIR.
//!
//! MetalPerformancePrimitives' `matmul2d` reaches AIR as calls to the declarations
//! [`crate::tensor_ops`] reads, whose bodies Metal supplies at pipeline creation, and its
//! tensors as `air.*_private_tensor` intrinsics over objects in thread memory. After inlining and
//! before SCCP, [`expand`] gives each call it models a body of its own, built here from the
//! call's constant arguments, and inlines it:
//!
//! - **Tensor objects** follow this translation's layout ([`tensor`]).
//! - **Cooperative tensors** follow Metal's measured layouts ([`layout`]): a thread's storage holds
//!   its elements contiguously, construction zeroes them (`+0` over floats, a destination's and a
//!   right input's alike), the size helpers answer constants (so the storage allocas SCCP sees are
//!   static), and the coordinate helper computes an element's place from the lane's index in its
//!   simdgroup and the simdgroup's index in the threadgroup, which the expansion adds to the kernel
//!   as two builtin parameters ([`Lanes`]).
//! - **`run`** computes each destination element the thread holds as Metal was measured to:
//!   `acc = C` in `multiply_accumulate` mode and zero (`+0` over floats)
//!   in `multiply` mode, then for `k` ascending below the descriptor's `k`, over integers
//!   `acc += a * b`, wrapping, and over floats `acc = fma(a, b, acc)`, one rounding per step
//!   (`llvm.fma.f32`), with or without relaxed precision. An element of `A` or `B` past its
//!   tensor's extents reads zero. A cooperative right input's element is another lane's, read
//!   with a shuffle every lane of the simdgroup runs.
//!
//! A call this does not model stays a call. If one survives SCCP, [`refuse_left`] refuses the
//! kernel as `Unsupported{Tensors, NotImplemented}`, naming what is not modeled: a shape or
//! simdgroup count no probe measured, a cooperative left input, a tensor destination, a dynamic
//! `k`, a tensor other than `tensor_inline`, an element type outside the measured ones, the
//! forms `get_element_index`, the copies,
//! conversions, reductions, the `_v2` and single-thread runs, and any stage but a kernel.

mod layout;
mod synth;
mod tensor;

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};

use crate::abi::{ArgTypeInfo, Builtin, EntryPoint, ParamBinding, ParamRole, Stage};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{DiagName, Location};
use crate::intrinsics::{AirIntrinsic, LlvmIntrinsic, Tensor};
use crate::ir::{
    ArgIdx, BinOp, Body, Call, Callee, CastOp, Const, FastMath, FuncId, Idx, InstId, InstKind,
    ModuleParts, Section, TailKind, Type, TypeId, Value,
};
use crate::tensor_ops::{Descriptor, MatMul2dFn, DESCRIPTOR_BYTES, RESERVED_INDEX};

use super::inline;
use super::rewrite::{append_params, invariant};
use synth::{Helper, Lowered, Run};

/// The two builtin parameters [`expand`] added to the kernel: the lane's index in its simdgroup
/// and the simdgroup's index in the threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Lanes {
    lane: u32,
    simdgroup: u32,
}

impl Lanes {
    /// The parameters' bindings.
    pub fn bindings(self) -> [ParamBinding; 2] {
        let binding = |param, b, name: &str| ParamBinding {
            param,
            role: ParamRole::Builtin(b),
            unused: false,
            predicate: None,
            ty: ArgTypeInfo::default(),
            name: Some(DiagName::new(name)),
        };
        [
            binding(
                self.lane,
                Builtin::ThreadIndexInSimdgroup,
                "matmul2d thread_index_in_simdgroup",
            ),
            binding(
                self.simdgroup,
                Builtin::SimdgroupIndexInThreadgroup,
                "matmul2d simdgroup_index_in_threadgroup",
            ),
        ]
    }
}

/// A call [`expand`] reads: a `matmul2d` ABI function or a tensor intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    MatMul(MatMul2dFn),
    Tensor(Tensor),
}

/// Whether `parts` declares a function a [`site`] can call: a body-less `matmul2d` ABI function
/// or a tensor intrinsic. Every site calls one, so a module that declares neither has none, and
/// [`expand`] and [`refuse_left`] need not read its instructions: the check costs one step per
/// function, not one per instruction.
fn declares_sites(parts: &ModuleParts, m: &mut Meter) -> Result<bool, Error> {
    for f in parts.funcs.values() {
        m.tick(1)?;
        if f.body.is_some() {
            continue;
        }
        let Some(name) = parts.symbols.get(f.name) else {
            continue;
        };
        let section = match f.section {
            Some(Section::Other(s)) => parts.symbols.get(s),
            _ => None,
        };
        let tensor = std::str::from_utf8(name)
            .ok()
            .and_then(AirIntrinsic::parse)
            .is_some_and(|op| matches!(op, AirIntrinsic::Tensor(_)));
        if tensor || MatMul2dFn::of_declaration(name, section).is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The site `kind` calls, with the call.
fn site<'b>(parts: &ModuleParts, kind: &'b InstKind) -> Option<(Site, &'b Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    let call: &Call = call;
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::Tensor(t),
            ..
        } => Some((Site::Tensor(t), call)),
        Callee::Direct(f) => {
            let func = parts.funcs.get(f).filter(|f| f.body.is_none())?;
            let section = match func.section {
                Some(Section::Other(s)) => parts.symbols.get(s),
                _ => None,
            };
            let name = parts.symbols.get(func.name)?;
            MatMul2dFn::of_declaration(name, section).map(|f| (Site::MatMul(f), call))
        }
        _ => None,
    }
}

/// The deepest chain of instructions [`eval`] follows.
const EVAL_DEPTH: u32 = 64;

/// The most instructions one [`eval`] visits.
const EVAL_STEPS: u32 = 1024;

/// The integer `v` is, read structurally from constants, `air.get_simdgroup_size` (32), integer
/// casts, integer arithmetic, selects of a known condition, and PHIs whose incoming values agree;
/// zero-extended from its width. `None` when it is not known, or not within [`EVAL_DEPTH`] and
/// [`EVAL_STEPS`].
fn eval(body: &Body, parts: &ModuleParts, v: Value) -> Option<u64> {
    eval_at(body, parts, v, EVAL_DEPTH, &Cell::new(EVAL_STEPS))
}

fn width(parts: &ModuleParts, ty: TypeId) -> Option<u32> {
    match parts.types.get(ty)? {
        Type::Int(bits) => Some(u32::from(*bits)),
        _ => None,
    }
}

fn mask(v: u64, bits: u32) -> u64 {
    if bits >= 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}

fn eval_at(
    body: &Body,
    parts: &ModuleParts,
    v: Value,
    depth: u32,
    steps: &Cell<u32>,
) -> Option<u64> {
    let id = match v {
        Value::Const(c) => {
            return match parts.consts.get(c)?.value {
                Const::Int(x) => Some(x),
                _ => None,
            };
        }
        Value::Inst(id) => id,
        Value::Arg(_) => return None,
    };
    let depth = depth.checked_sub(1)?;
    steps.set(steps.get().checked_sub(1)?);
    let inst = body.insts.get(id)?;
    let bits = width(parts, inst.ty)?;
    let at = |v: Value| eval_at(body, parts, v, depth, steps);
    let value = match &inst.kind {
        InstKind::Call(call) => match call.callee {
            Callee::Air {
                op: AirIntrinsic::SimdgroupSize(_),
                ..
            } => 32,
            _ => return None,
        },
        InstKind::Cast { op, v } => {
            let x = at(*v)?;
            match op {
                CastOp::ZExt | CastOp::Trunc => x,
                CastOp::SExt => {
                    let from = width(parts, value_ty(body, parts, *v)?)?;
                    if from == 0 || from >= 64 {
                        x
                    } else if x >> (from - 1) & 1 == 1 {
                        x | !((1u64 << from) - 1)
                    } else {
                        x
                    }
                }
                _ => return None,
            }
        }
        InstKind::BinOp { op, a, b, .. } => {
            let (a, b) = (at(*a)?, at(*b)?);
            match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Shl if b < u64::from(bits) => a << b,
                BinOp::LShr if b < u64::from(bits) => a >> b,
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::UDiv if b != 0 => a / b,
                BinOp::URem if b != 0 => a % b,
                _ => return None,
            }
        }
        InstKind::Select { c, t, f, .. } => match at(*c)? {
            0 => at(*f)?,
            _ => at(*t)?,
        },
        InstKind::Phi { incoming, .. } => {
            let mut known = None;
            for &(x, _) in incoming.iter() {
                if x == v {
                    continue;
                }
                let x = at(x)?;
                if known.is_some_and(|k| k != x) {
                    return None;
                }
                known = Some(x);
            }
            known?
        }
        _ => return None,
    };
    Some(mask(value, bits))
}

/// The type of instruction or constant `v` in `body`.
fn value_ty(body: &Body, parts: &ModuleParts, v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Const(c) => parts.consts.ty(c),
        Value::Arg(_) => None,
    }
}

/// The descriptor a `run`'s descriptor argument `v` points at: a constant global's initializer,
/// directly or copied whole (`llvm.memcpy` of its 20 bytes) into an object nothing else writes or
/// reads. `writes` maps each such object to its one copy's source, `None` when anything else
/// uses it.
fn descriptor_at(parts: &ModuleParts, v: Value, copies: &Copies) -> Option<Descriptor> {
    let src = match v {
        Value::Inst(i) => copies.get(&i).copied().flatten()?.0,
        v => v,
    };
    let Value::Const(c) = src else {
        return None;
    };
    let Const::GlobalAddr(g) = parts.consts.get(c)?.value else {
        return None;
    };
    let global = parts.globals.get(g).filter(|g| g.constant)?;
    let init = parts.consts.get(global.init?)?;
    let Const::Aggregate(fields) = &init.value else {
        return None;
    };
    let mut values = [0u64; 7];
    if fields.len() != values.len() {
        return None;
    }
    for (slot, &f) in values.iter_mut().zip(fields.iter()) {
        *slot = match parts.consts.get(f)?.value {
            Const::Int(x) => x,
            _ => return None,
        };
    }
    Descriptor::from_fields(values)
}

/// For every object a `run` reads its descriptor from, its one whole copy (an `llvm.memcpy` of 20
/// bytes) and that copy's source, `None` when anything but that copy and `run`s uses it.
fn descriptor_copies(body: &Body, parts: &ModuleParts, m: &mut Meter) -> Result<Copies, Error> {
    let mut objects: HashMap<InstId, Object> = HashMap::new();
    for inst in body.insts.values() {
        m.tick(1)?;
        if let Some((Site::MatMul(MatMul2dFn::Run(_)), call)) = site(parts, &inst.kind) {
            if let Some(Value::Inst(i)) = call.args.first() {
                if !objects.contains_key(i) {
                    m.charge(OBJECT_BYTES)?;
                    objects.insert(*i, Object::default());
                }
            }
        }
    }
    if objects.is_empty() {
        return Ok(Copies::new());
    }
    // Any use but a run's descriptor argument or a whole copy into the object.
    fn other(objects: &mut HashMap<InstId, Object>, v: Value) {
        if let Value::Inst(i) = v {
            if let Some(o) = objects.get_mut(&i) {
                o.other = true;
            }
        }
    }
    for (at, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Call(call) = &inst.kind else {
            inst.kind.for_each_value(|v| other(&mut objects, v));
            continue;
        };
        let run = matches!(
            site(parts, &inst.kind),
            Some((Site::MatMul(MatMul2dFn::Run(_)), _))
        );
        let copy = matches!(
            call.callee,
            Callee::Llvm {
                op: LlvmIntrinsic::Memcpy {
                    volatile: false,
                    ..
                },
                ..
            }
        ) && call.args.get(2).and_then(|&len| eval(body, parts, len))
            == Some(DESCRIPTOR_BYTES as u64);
        if let Callee::Indirect { callee, .. } = call.callee {
            other(&mut objects, callee);
        }
        for (pos, &v) in call.args.iter().enumerate() {
            match v {
                Value::Inst(_) if pos == 0 && run => {}
                Value::Inst(i) if pos == 0 && copy && objects.contains_key(&i) => {
                    if let (Some(o), Some(&src)) = (objects.get_mut(&i), call.args.get(1)) {
                        o.copies += 1;
                        o.copy = Some((src, at));
                    }
                }
                v => other(&mut objects, v),
            }
        }
    }
    let mut copies = Copies::new();
    for (i, o) in objects {
        copies.insert(i, o.copy.filter(|_| o.copies == 1 && !o.other));
    }
    Ok(copies)
}

/// The bytes [`expand`] charges per parameter of a function's key.
const KEY_BYTES: usize = std::mem::size_of::<(TypeId, Option<Value>)>();

/// What [`descriptor_copies`] has seen of one object.
#[derive(Default)]
struct Object {
    copies: u32,
    other: bool,
    copy: Option<(Value, InstId)>,
}

/// The bytes [`descriptor_copies`] charges per object.
const OBJECT_BYTES: usize = std::mem::size_of::<(InstId, Object)>();

/// Per object a `run` reads its descriptor from: its one whole copy's source and the copy, `None`
/// when anything else uses the object ([`descriptor_copies`]). Ordered by the object, so
/// [`expand`] plans the copies it removes, and charges the ledger for them, in the body's order
/// rather than a hash seed's: a second compile of the same input reaches the same ledger peak
/// (`engine_metering`).
type Copies = BTreeMap<InstId, Option<(Value, InstId)>>;

/// How [`expand`] lowers one call.
enum Plan {
    /// Leave it a call ([`refuse_left`] refuses it if it survives SCCP).
    Leave,
    /// Call a function built for it.
    Lower(Lowered),
}

/// What a call is, read from `body`: its lowering, or why it has none.
fn classify(
    body: &Body,
    parts: &ModuleParts,
    s: Site,
    call: &Call,
    kernel: bool,
    copies: &Copies,
) -> Result<Lowered, Error> {
    let ev = |v: Value| eval(body, parts, v);
    let lowered = match s {
        Site::Tensor(t) => {
            let null = |v: Value| tensor::is_null(&parts.consts, v);
            let rank = tensor::form(t, &call.args, &ev, &null).ok_or_else(|| {
                synth::refuse(format!("a tensor operation {t:?} of an unmodeled form"))
            })?;
            match t {
                Tensor::DescriptorSize => Lowered::Const {
                    value: tensor::object_bytes(rank),
                    bits: 16,
                },
                op => Lowered::Tensor { op, rank },
            }
        }
        Site::MatMul(MatMul2dFn::Run(form)) => {
            if !kernel {
                return Err(synth::refuse("matmul2d outside a kernel"));
            }
            let storage = |v: Value| -> Option<Helper> {
                let Value::Inst(i) = v else {
                    return None;
                };
                let (Site::MatMul(f), call) = site(parts, &body.insts.get(i)?.kind)? else {
                    return None;
                };
                if f != MatMul2dFn::TensorGetElementPointer
                    || call.args.get(9).copied().and_then(ev) != Some(u64::from(RESERVED_INDEX))
                {
                    return None;
                }
                Helper::of_call(f, &call.args, &ev).ok()
            };
            let desc = call
                .args
                .first()
                .and_then(|&v| descriptor_at(parts, v, copies));
            Lowered::Run(Run::of_call(form, &call.args, desc, &ev, &storage)?)
        }
        Site::MatMul(f) => {
            let h = Helper::of_call(f, &call.args, &ev)?;
            match h.size()? {
                Some(value) => Lowered::Const {
                    value,
                    bits: if f == MatMul2dFn::TensorDataSize {
                        64
                    } else {
                        16
                    },
                },
                None => Lowered::Helper(h),
            }
        }
    };
    if synth::needs_lanes(&lowered) && !kernel {
        return Err(synth::refuse("matmul2d outside a kernel"));
    }
    Ok(lowered)
}

/// Lower every `matmul2d` and tensor call of the inlined entry `body` it models (module doc),
/// and inline the result. Returns the body and, when a lowering reads them, the lane parameters
/// added to the entry `ep.func`.
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    ep: &EntryPoint,
    m: &mut Meter,
) -> Result<(Body, Option<Lanes>), Error> {
    if !declares_sites(parts, m)? {
        return Ok((body, None));
    }
    let mut any = false;
    for inst in body.insts.values() {
        m.tick(1)?;
        any |= site(parts, &inst.kind).is_some();
    }
    if !any {
        return Ok((body, None));
    }
    let kernel = matches!(ep.stage, Stage::Kernel { .. });
    let copies = descriptor_copies(&body, parts, m)?;
    let mut plans = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((s, call)) = site(parts, &inst.kind) else {
            continue;
        };
        let plan = match classify(&body, parts, s, call, kernel, &copies) {
            Ok(l) => Plan::Lower(l),
            Err(Error::Unsupported(_)) => Plan::Leave,
            Err(e) => return Err(e),
        };
        m.reserve(&mut plans, 1)?;
        plans.push((id, plan));
    }
    // A descriptor's copy every `run` reading it was lowered for is read by nothing: it becomes
    // nothing, and its source global goes with it (a constant in thread memory, which memory has
    // no model of).
    let mut read_left = Vec::new();
    for (id, plan) in &plans {
        if let (Plan::Leave, Some(InstKind::Call(call))) =
            (plan, body.insts.get(*id).map(|i| &i.kind))
        {
            if let Some(Value::Inst(object)) = call.args.first() {
                m.reserve(&mut read_left, 1)?;
                read_left.push(*object);
            }
        }
    }
    for (object, copy) in &copies {
        if let Some((_, at)) = copy {
            if !read_left.contains(object) {
                m.reserve(&mut plans, 1)?;
                plans.push((*at, Plan::Lower(Lowered::Nothing)));
            }
        }
    }
    m.release(read_left.capacity() * std::mem::size_of::<InstId>());
    drop(read_left);
    m.release(copies.len() * OBJECT_BYTES);
    drop(copies);

    // The lane parameters, appended to the entry's.
    let func = ep.func;
    let lanes = if plans
        .iter()
        .any(|(_, p)| matches!(p, Plan::Lower(l) if synth::needs_lanes(l)))
    {
        Some(add_lanes(parts, func, m)?)
    } else {
        None
    };

    // One function per distinct lowering and the arguments it knows, which its body reads as the
    // constants they are (an integer [`eval`] knows, or any constant).
    let entry_params: Vec<TypeId> = match parts.funcs.get(func).and_then(|f| parts.types.get(f.ty))
    {
        Some(Type::Function { params, .. }) => params.to_vec(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    // Each function's key: its lowering, and per parameter its type and the constant the body
    // reads it as.
    let mut made: HashMap<(Lowered, Vec<(TypeId, Option<Value>)>), FuncId> = HashMap::new();
    let mut made_bytes = 0usize;
    let mut calls = Vec::new();
    m.reserve(&mut calls, plans.len())?;
    for (id, plan) in &plans {
        let Plan::Lower(lowered) = plan else {
            continue;
        };
        let inst = body
            .insts
            .get(*id)
            .ok_or_else(|| invariant("a planned call out of range"))?;
        let InstKind::Call(call) = &inst.kind else {
            return Err(invariant("a planned call is not a call"));
        };
        let lanes_too = synth::needs_lanes(lowered);
        let mut params = Vec::new();
        m.reserve(&mut params, call.args.len() + 2)?;
        for &v in call.args.iter() {
            let ty = match v {
                Value::Arg(a) => entry_params.get(a.index()).copied(),
                v => value_ty(&body, parts, v),
            }
            .ok_or_else(|| invariant("an untyped argument"))?;
            let known = match v {
                Value::Const(_) => Some(v),
                v => match (width(parts, ty), eval(&body, parts, v)) {
                    (Some(_), Some(x)) => Some(int_const(parts, ty, x, m)?),
                    _ => None,
                },
            };
            params.push((ty, known));
        }
        if lanes_too {
            let i32t = parts.types.intern(Type::Int(32), m)?;
            params.extend([(i32t, None), (i32t, None)]);
        }
        let bytes = params.capacity() * KEY_BYTES;
        let key = (*lowered, params);
        let f = match made.get(&key) {
            Some(&f) => {
                m.release(bytes);
                f
            }
            None => {
                let f = synth::function(parts, lowered, &key.1, inst.ty, m)?;
                made_bytes += bytes;
                made.insert(key, f);
                f
            }
        };
        calls.push((*id, f, lanes_too));
    }
    drop(made);
    m.release(made_bytes);
    m.release(plans.capacity() * std::mem::size_of::<(InstId, Plan)>());
    drop(plans);
    if calls.is_empty() {
        m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId, bool)>());
        return Ok((body, lanes));
    }

    // The calls rewritten in place to the functions built for them, then inlined.
    let mut body = body;
    for &(id, f, with_lanes) in &calls {
        m.tick(1)?;
        let inst = body
            .insts
            .get_mut(id)
            .ok_or_else(|| invariant("a planned call out of range"))?;
        let InstKind::Call(call) = &mut inst.kind else {
            return Err(invariant("a planned call is not a call"));
        };
        let mut args = call.args.to_vec();
        if with_lanes {
            let l = lanes.ok_or_else(|| invariant("lanes not added"))?;
            args.extend([
                Value::Arg(ArgIdx::new(l.lane)),
                Value::Arg(ArgIdx::new(l.simdgroup)),
            ]);
        }
        let lowered = Call {
            callee: Callee::Direct(f),
            args: args.into_boxed_slice(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        };
        m.charge(lowered.heap_bytes())?;
        m.release(call.heap_bytes());
        **call = lowered;
    }
    m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId, bool)>());
    drop(calls);
    if let Some(replaced) = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(body)
    {
        replaced.release(m);
    }
    let inlined = inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        func,
        m,
    )?
    .ok_or_else(|| invariant("a lowered matmul2d call was not inlined"))?;
    Ok((inlined, lanes))
}

/// The integer constant `x` of type `ty`.
fn int_const(parts: &mut ModuleParts, ty: TypeId, x: u64, m: &mut Meter) -> Result<Value, Error> {
    let c = parts.consts.intern(
        crate::ir::Constant {
            ty,
            value: Const::Int(x),
        },
        m,
    )?;
    Ok(Value::Const(c))
}

/// Append two `i32` parameters to entry `func`: the lane's index in its simdgroup and the
/// simdgroup's index in the threadgroup.
fn add_lanes(parts: &mut ModuleParts, func: FuncId, m: &mut Meter) -> Result<Lanes, Error> {
    let i32t = parts.types.intern(Type::Int(32), m)?;
    let n = append_params(parts, func, &[i32t, i32t], m)?;
    Ok(Lanes {
        lane: n,
        simdgroup: n + 1,
    })
}

/// Refuse the first `matmul2d` or tensor call left in the solved entry `body` (one [`expand`]
/// did not model) that is refused for itself, naming why, at the call. A call left only because
/// the `run` it feeds is refused is passed over for that `run`; a call that only SCCP's constants
/// make modeled is refused as such.
pub(super) fn refuse_left(
    body: &Body,
    parts: &ModuleParts,
    stage: &Stage,
    name: &DiagName,
    m: &mut Meter,
) -> Result<(), Error> {
    if !declares_sites(parts, m)? {
        return Ok(());
    }
    let kernel = matches!(stage, Stage::Kernel { .. });
    let mut copies = None;
    let mut first = None;
    for (b, block) in body.blocks.iter() {
        for i in block.inst_ids() {
            m.tick(1)?;
            let Some((s, call)) = body.insts.get(i).and_then(|inst| site(parts, &inst.kind)) else {
                continue;
            };
            if copies.is_none() {
                copies = Some(descriptor_copies(body, parts, m)?);
            }
            let copies = copies
                .as_ref()
                .ok_or_else(|| invariant("descriptor copies"))?;
            let at = Location::function(name.clone(), Some(b.raw()), Some(i.raw()));
            match classify(body, parts, s, call, kernel, copies) {
                Err(Error::Unsupported(u)) => return Err(Error::Unsupported(u.at(at))),
                Err(e) => return Err(e),
                Ok(_) => {
                    first.get_or_insert(at);
                }
            }
        }
    }
    match first {
        Some(at) => match synth::refuse("a matmul2d call whose arguments only SCCP makes known") {
            Error::Unsupported(u) => Err(Error::Unsupported(u.at(at))),
            e => Err(e),
        },
        None => Ok(()),
    }
}
