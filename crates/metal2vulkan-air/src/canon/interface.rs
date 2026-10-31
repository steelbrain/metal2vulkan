//! The entry's ABI after specialization.
//!
//! The metadata may state an integer (an argument-table index or count, a render-target
//! attachment) or a predicate as a global that the static initializer computes. Once the globals
//! hold their pipeline values, each such integer becomes [`AbiInt::Const`], each parameter or
//! output whose predicate is false is removed from the entry point, and each surviving predicate
//! is dropped. No global id survives in the result, so it stays valid after pruning renumbers the
//! globals.
//!
//! **Absent outputs**. An output gated off is not part of the specialized function: it leaves the
//! entry point's outputs, and the returned struct keeps its field, which no interface variable
//! receives. Each remaining output names its field. Two remaining color outputs at one color
//! attachment and one blend index are an invalid request (`RequestError::DuplicateColorOutput`):
//! Metal refuses such a specialization when it makes the function.
//!
//! **Absent arguments**. The LLVM parameter of an argument gated off stays, and the body may still
//! read it (SCCP deletes the reads its own constants guard, not every dead one).
//! [`absent_arguments`] gives such a read the argument's absent value: the null reference for a
//! buffer or argument buffer, which the access family reads as zero. An absent argument of any
//! other role (a texture or texture table, a sampler, a stage input, a builtin, a color input,
//! threadgroup memory, an object) is not declared in the specialized function (MSL §5.8.1.2), so
//! Metal defines no value for it: a read whose result nothing observes reads `undef`, an observed
//! read of an argument passed by value (its parameter is not a pointer: a vertex attribute, a stage
//! input, a varying, a color input, a builtin, imageblock data, an indirect constant, a per-patch
//! attribute) reads the all-zero value of its type, which refines Metal's undefined one. A texture
//! or sampler argument read only as the texture or sampler operand of texture calls has each such
//! call made [`FunctionConstant::AbsentRead`]: Metal gives the read no value, so lowering marks the
//! command buffer's execution status where the read runs and the command buffer fails, while a run
//! that never reaches the read is Metal's exactly. A texture or sampler argument (a texture table
//! included) read any other way stays in the body, and the entry lists it
//! ([`crate::canon::Entry::absent_handles`]): lowering follows its handle through memory,
//! aggregates, table elements and merges as it follows a present one's, and each texture call it
//! reaches is that absent read. Threadgroup memory read only through loads, stores, and atomics at
//! addresses derived from it has each load and atomic made [`FunctionConstant::AbsentMemory`], the
//! same absent read, and each store dropped. An observed read of any other (threadgroup memory read
//! another way, an object, a pointer-typed imageblock or control-point input) is refused as
//! `FunctionConstantArgument`, naming the argument. A write to, or fence of, an absent texture
//! itself is deleted first: Metal drops such a write ([`absent_texture_writes`]). A store into a
//! stack allocation is observed only once an observed read reaches its bytes ([`private`]). What
//! nothing observes then goes ([`unobserved_code`]), an absent read whose result nothing observes
//! included.
//!
//! **Argument tables**. A present resource parameter's resolved slots lie in Metal's table for its
//! kind, every GPU family's maximum: 31 buffer entries (a buffer table's `count` elements are slots
//! `index .. index + count`; an acceleration structure or function table binds at a buffer index),
//! 31 threadgroup memory entries, 128 texture entries, and 16 sampler entries. A binding past its
//! table is one no Metal compiler emits, so it is invalid input (`AbiError::SlotPastTable`) rather
//! than a shape the lowering has to bound.

use crate::abi::{AbiInt, AirAbi, EntryPoint, OutputRole, ParamRole, PatchInput, Predicate};
use crate::abi::{MemberKind, OutputBinding, ParamBinding, Slot, StructInfo};
use crate::budget::Meter;
use crate::error::{AbiError, CanonError, DiagName, InputError, InputErrorKind, Location};
use crate::intrinsics::{AbsentAccess, AtomicSpace, NumTy, Texture};
use crate::intrinsics::{AirIntrinsic, FunctionConstant, LlvmIntrinsic, Operand, TexOp, TexRole};
use crate::ir::{Arena, ArgIdx, Body, Callee, Const, ConstId, ConstTable, Constant, FuncId};
use crate::ir::{CastOp, FastMath, Inst, TailKind, TypeLayouts, Value};
use crate::ir::{Global, GlobalId, Idx, InstId, InstKind, Layout, ModuleParts, Type, TypeId};
use std::collections::HashMap;

use crate::error::{Error, Feature, RequestError, Unsupported};

use super::rewrite::{apply, invariant, scoped, BodyEdit};
use super::sccp::{is_pure, FcState};

mod private;

use private::Private;

/// Read-only view of the globals once they hold their pipeline values.
pub(super) struct Values<'a> {
    pub globals: &'a Arena<GlobalId, Global>,
    pub consts: &'a ConstTable,
    /// Function constants and the globals derived from them.
    pub fcs: &'a HashMap<GlobalId, FcState>,
}

/// A global the ABI reads as an integer does not hold one. Mapped through the single
/// `From<InputError>` conversion (`Unsupported{StaticInitializer, NotImplemented}`), never built by
/// hand.
pub(super) fn not_evaluable() -> Error {
    InputError::new(
        InputErrorKind::Canon(CanonError::StaticInitNotEvaluable),
        Location::module(),
    )
    .into()
}

impl Values<'_> {
    /// The integer value global `g` holds: an undefined one's (an unsupplied constant's, or one
    /// computed from it) is 0, as SCCP's is.
    pub(super) fn int(&self, g: GlobalId) -> Result<u64, Error> {
        if self.fcs.get(&g).is_some_and(|f| !f.supplied) {
            return Ok(0);
        }
        let init = self
            .globals
            .get(g)
            .and_then(|g| g.init)
            .ok_or_else(not_evaluable)?;
        match self.consts.get(init).map(|c| &c.value) {
            Some(Const::Int(v)) => Ok(*v),
            Some(Const::Null) => Ok(0),
            _ => Err(not_evaluable()),
        }
    }

    fn resolve(&self, v: &mut AbiInt) -> Result<(), Error> {
        if let AbiInt::Global(g) = *v {
            let x = u32::try_from(self.int(g)?).map_err(|_| not_evaluable())?;
            *v = AbiInt::Const(x);
        }
        Ok(())
    }

    fn slot(&self, s: &mut Slot) -> Result<(), Error> {
        self.resolve(&mut s.index)?;
        self.resolve(&mut s.count)
    }

    /// Whether the argument or output a predicate gates is present.
    fn present(&self, p: &Option<Predicate>) -> Result<bool, Error> {
        match p {
            Some(p) => Ok(self.int(p.global)? != 0),
            None => Ok(true),
        }
    }

    fn role(&self, r: &mut ParamRole) -> Result<(), Error> {
        match r {
            ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) | ParamRole::Threadgroup(b) => {
                self.slot(&mut b.location)
            }
            ParamRole::Texture(t) => self.slot(&mut t.location),
            ParamRole::Sampler { location }
            | ParamRole::StageIn { location }
            | ParamRole::VertexInput { location }
            | ParamRole::IndirectConstant { location }
            | ParamRole::PatchInput(PatchInput::Patch { location }) => self.slot(location),
            ParamRole::ColorInput { index, .. } => self.resolve(index),
            ParamRole::AccelerationStructure { arg, .. }
            | ParamRole::FunctionTable(arg)
            | ParamRole::IntersectionTable(arg)
            | ParamRole::CommandBuffer(arg)
            | ParamRole::ComputePipelineState(arg)
            | ParamRole::RenderPipelineState(arg)
            | ParamRole::DepthStencilState(arg) => self.slot(&mut arg.location),
            // Each control-point field's attribute, and whether its predicate gates it off: the
            // field stays a member of the fetched struct either way.
            ParamRole::PatchInput(PatchInput::ControlPoints { fields, .. }) => {
                for f in fields {
                    self.slot(&mut f.location)?;
                    f.gated_off = !self.present(&f.predicate)?;
                    f.predicate = None;
                }
                Ok(())
            }
            // Whether each mesh field's predicate gates it off: the field stays in the type, and a
            // write to it stores nothing.
            ParamRole::Mesh(m) => {
                for f in m.vertex.iter_mut().chain(m.primitive.iter_mut()) {
                    f.gated_off = !self.present(&f.predicate)?;
                    f.predicate = None;
                }
                Ok(())
            }
            ParamRole::Builtin(_)
            | ParamRole::FragmentInput(_)
            | ParamRole::Imageblock(_)
            | ParamRole::Payload { .. }
            | ParamRole::MeshGridProperties => Ok(()),
        }
    }

    /// The entry point `ep` with its integers and predicates resolved and its gated-off
    /// parameters and outputs removed, the ABI's struct layouts with their integers resolved, and
    /// the gated-off parameters' bindings (their predicates dropped, their integers unresolved:
    /// an absent argument binds no slot).
    pub fn entry(
        &self,
        abi: &AirAbi,
        ep: &EntryPoint,
    ) -> Result<(EntryPoint, Vec<StructInfo>, Vec<ParamBinding>), Error> {
        let mut out = ep.clone();
        let mut params = Vec::with_capacity(out.params.len());
        let mut absent = Vec::new();
        for mut p in out.params {
            let present = self.present(&p.predicate)?;
            p.predicate = None;
            if !present {
                absent.push(p);
                continue;
            }
            let arg = p.param;
            self.role(&mut p.role)
                .map_err(|e| located(e, &ep.name, Some(arg)))?;
            if past_table(&p.role) {
                return Err(InputError::new(
                    InputErrorKind::Abi(AbiError::SlotPastTable),
                    Location::metadata(
                        Some(ep.name.clone()),
                        Some(arg),
                        Some("air.location_index"),
                    ),
                )
                .into());
            }
            params.push(p);
        }
        out.params = params;
        let mut outputs = Vec::with_capacity(out.outputs.len());
        for mut o in out.outputs {
            // An output gated off is not part of the specialized function.
            if !self.present(&o.predicate)? {
                continue;
            }
            o.predicate = None;
            if let OutputRole::RenderTarget { attachment, .. } = &mut o.role {
                self.resolve(attachment)?;
            }
            outputs.push(o);
        }
        duplicate_color(&ep.name, &outputs)?;
        out.outputs = outputs;
        let mut structs = abi.structs.clone();
        for s in &mut structs {
            for member in &mut s.members {
                if let MemberKind::Indirect(arg) = &mut member.kind {
                    self.role(&mut arg.role)?;
                }
            }
        }
        Ok((out, structs, absent))
    }
}

/// Refuse two of the remaining `outputs` of entry `name` that write one color attachment at one
/// blend index. MSL refuses such a pair outright, so only function constants that leave both of a
/// gated pair present make one, and Metal then refuses the specialization when it makes the
/// function ("duplicated indices pair"): the request is invalid
/// (`RequestError::DuplicateColorOutput`). Writing both to one location would let one store
/// win, a result Metal never produces.
fn duplicate_color(name: &DiagName, outputs: &[OutputBinding]) -> Result<(), Error> {
    let mut seen = Vec::with_capacity(outputs.len());
    for o in outputs {
        let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index,
            ..
        } = o.role
        else {
            continue;
        };
        if seen.contains(&(color, index)) {
            return Err(InputError::new(
                InputErrorKind::Request(RequestError::DuplicateColorOutput { color, index }),
                Location::metadata(Some(name.clone()), None, Some("air.render_target")),
            )
            .into());
        }
        seen.push((color, index));
    }
    Ok(())
}

/// Metal's argument tables per function, every GPU family's maximum in Metal's feature set tables:
/// buffer, threadgroup memory, texture, and sampler entries.
const BUFFER_TABLE: u64 = 31;
const THREADGROUP_TABLE: u64 = 31;
const TEXTURE_TABLE: u64 = 128;
const SAMPLER_TABLE: u64 = 16;

/// Whether the resolved argument-table slots of entry parameter role `r` run past Metal's table
/// for its kind. A buffer or argument buffer, whose `count` elements are a buffer table's slots
/// `index .. index + count`, and an acceleration structure, visible function table, or intersection
/// function table, which the API binds at a buffer index, lie in the 31 buffer entries; threadgroup
/// memory in the 31 threadgroup memory entries; a texture or texture array in the 128 texture
/// entries; a sampler or sampler array in the 16 sampler entries. No Metal compiler emits such a
/// binding, so it is malformed AIR (`AbiError::SlotPastTable`). An argument buffer's members index
/// its own resource IDs, not these tables, and are not checked.
fn past_table(r: &ParamRole) -> bool {
    let (location, table) = match r {
        ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => (b.location, BUFFER_TABLE),
        ParamRole::AccelerationStructure { arg, .. }
        | ParamRole::FunctionTable(arg)
        | ParamRole::IntersectionTable(arg) => (arg.location, BUFFER_TABLE),
        ParamRole::Threadgroup(b) => (b.location, THREADGROUP_TABLE),
        ParamRole::Texture(t) => (t.location, TEXTURE_TABLE),
        ParamRole::Sampler { location } => (*location, SAMPLER_TABLE),
        _ => return false,
    };
    match (location.index, location.count) {
        (AbiInt::Const(index), AbiInt::Const(count)) => u64::from(index) + u64::from(count) > table,
        _ => false,
    }
}

/// How an absent argument of role `r` is read. A buffer or argument buffer reads as the null
/// reference ([`Absent::Null`]): a load through an undeclared buffer is undefined, so the access
/// family's null root (zero) refines it. A texture or sampler is read through texture calls
/// ([`Absent::Reads`]): the result of a read, query, or sample of an absent texture, or of a sample
/// through an absent sampler, is undefined, not a null texture's. Measured on device, Metal's
/// compiler drops a store of such a result and makes a fragment color computed from one all-zero,
/// where its validation layer reads zero instead; with a gated-off texture table beside it, it
/// dropped a kernel's whole body. No one translation is Metal's, so each such call becomes
/// [`FunctionConstant::AbsentRead`], which fails the command buffer where it runs; a handle that
/// reaches its calls through memory, an aggregate, a table element or a merge is lowering's absent
/// handle, whose calls are that read. Threadgroup memory has no contents Metal defines either:
/// measured on an M4, its compiler dropped stores of values loaded from it, dropped a kernel's
/// whole body where an access ran on every path, and an atomic on it changed another argument's
/// threadgroup memory under the validation layer and not without it. So each load or atomic through
/// it is the same absent read, and each store is dropped ([`Absent::Memory`]). Stage inputs,
/// builtins, color inputs, and objects have no absent value Metal defines (an observed read of one
/// passed by value reads zero instead, [`absent_values`]).
fn absent_null(r: &ParamRole) -> Absent {
    match r {
        ParamRole::Buffer(_) | ParamRole::ArgumentBuffer(_) => Absent::Null,
        ParamRole::Texture(_) | ParamRole::Sampler { .. } => Absent::Reads,
        ParamRole::Threadgroup(_) => Absent::Memory,
        ParamRole::Builtin(_)
        | ParamRole::StageIn { .. }
        | ParamRole::VertexInput { .. }
        | ParamRole::FragmentInput(_)
        | ParamRole::ColorInput { .. }
        | ParamRole::PatchInput(_)
        | ParamRole::Imageblock(_)
        | ParamRole::AccelerationStructure { .. }
        | ParamRole::FunctionTable(_)
        | ParamRole::IntersectionTable(_)
        | ParamRole::IndirectConstant { .. }
        | ParamRole::CommandBuffer(_)
        | ParamRole::ComputePipelineState(_)
        | ParamRole::RenderPipelineState(_)
        | ParamRole::DepthStencilState(_)
        | ParamRole::Payload { .. }
        | ParamRole::MeshGridProperties
        | ParamRole::Mesh(_) => Absent::Never,
    }
}

/// How an absent argument is read ([`absent_null`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Absent {
    /// As the null reference.
    Null,
    /// Through texture calls, each one an absent read: directly ([`absent_reads`]), or as
    /// lowering's absent handle.
    Reads,
    /// As threadgroup memory that is not there: each load or atomic through it an absent read,
    /// each store dropped ([`absent_memory`]).
    Memory,
    /// Not at all: an observed read of the pointer is refused.
    Never,
}

/// The texture calls of `body` through which it reads absent argument `a` of role `role`, each
/// with its decoded operation: `None` unless every instruction that reads `a` is a texture call
/// (other than a write or a fence, which [`absent_texture_writes`] deleted) whose only use of `a`
/// is its texture operand, for a texture, or its sampler operand, for a sampler.
fn absent_reads(
    body: &Body,
    a: ArgIdx,
    role: &ParamRole,
    m: &mut Meter,
) -> Result<Option<Vec<(InstId, Texture)>>, Error> {
    let operand = match role {
        ParamRole::Texture(_) => TexRole::Texture,
        ParamRole::Sampler { .. } => TexRole::Sampler,
        _ => return Ok(None),
    };
    let mut calls = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let mut reads = 0usize;
        inst.kind.for_each_value(|v| {
            if v == Value::Arg(a) {
                reads += 1;
            }
        });
        if reads == 0 {
            continue;
        }
        let InstKind::Call(call) = &inst.kind else {
            return Ok(None);
        };
        let Callee::Air {
            op: AirIntrinsic::Texture(t),
            ..
        } = &call.callee
        else {
            return Ok(None);
        };
        let direct = t
            .layout()
            .index(operand)
            .and_then(|i| call.args.get(i))
            .is_some_and(|v| *v == Value::Arg(a));
        if reads != 1 || !direct || matches!(t.op, TexOp::Write | TexOp::Fence) {
            return Ok(None);
        }
        calls.push((id, *t));
    }
    Ok(Some(calls))
}

/// What an access through absent threadgroup memory becomes ([`absent_memory`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryUse {
    /// A load or atomic whose result is a scalar or vector: the absent read of it.
    Read(InstId, AbsentAccess),
    /// A store, or an atomic without a result: dropped.
    Drop(InstId),
}

/// The accesses through which `body` reads or writes absent threadgroup argument `a`, each with
/// what it becomes: `None` unless every pointer derived from `a` (`a` itself, a `getelementptr`
/// whose base is one at indices derived from no such pointer, or a no-op pointer cast of one) is
/// used only to derive another, or as the address of a load of a scalar or vector, a store of a
/// value that is no such pointer, an `atomicrmw` of a value that is no such pointer, or a
/// threadgroup atomic with no other pointer operand (a compare-exchange's `expected` is thread
/// memory it writes) whose result is a scalar, a vector, or nothing. Metered: a work unit per
/// instruction per pass, and a pass per step of the longest derivation.
fn absent_memory(
    body: &Body,
    parts: &ModuleParts,
    a: ArgIdx,
    m: &mut Meter,
) -> Result<Option<Vec<MemoryUse>>, Error> {
    let mut derived: Vec<InstId> = Vec::new();
    let is = |derived: &[InstId], v: Value| match v {
        Value::Arg(x) => x == a,
        Value::Inst(i) => derived.contains(&i),
        Value::Const(_) => false,
    };
    // The derivations, to a fixpoint: each pass adds those whose operand an earlier one derived.
    loop {
        let mut grew = false;
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            if derived.contains(&id) {
                continue;
            }
            let derives = match &inst.kind {
                InstKind::Gep { base, indices, .. } => {
                    is(&derived, *base) && !indices.iter().any(|&x| is(&derived, x))
                }
                InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    v,
                } => is(&derived, *v),
                _ => false,
            };
            if derives {
                m.reserve(&mut derived, 1)?;
                derived.push(id);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    let read = |ty: TypeId, atomic: bool| {
        NumTy::of(&parts.types, ty).map(|ty| AbsentAccess { ty, atomic })
    };
    let mut uses = Vec::new();
    let mut ok = true;
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if derived.contains(&id) {
            continue;
        }
        let mut reads = 0usize;
        inst.kind.for_each_value(|v| {
            if is(&derived, v) {
                reads += 1;
            }
        });
        if reads == 0 {
            continue;
        }
        let u = match &inst.kind {
            InstKind::Load { ptr, atomic, .. } if is(&derived, *ptr) => {
                read(inst.ty, atomic.is_some()).map(|r| MemoryUse::Read(id, r))
            }
            InstKind::Store { val, ptr, .. } if is(&derived, *ptr) && !is(&derived, *val) => {
                Some(MemoryUse::Drop(id))
            }
            InstKind::AtomicRmw { ptr, val, .. } if is(&derived, *ptr) && !is(&derived, *val) => {
                read(inst.ty, true).map(|r| MemoryUse::Read(id, r))
            }
            InstKind::Call(call) => match &call.callee {
                Callee::Air {
                    op: op @ AirIntrinsic::Atomic(at),
                    ..
                } if at.space == AtomicSpace::Local => {
                    // The derived operand is the atomic's only pointer: a compare-exchange's
                    // `expected` is thread memory it writes, which dropping it would lose.
                    let sig = op.signature();
                    let pointers = sig
                        .params()
                        .iter()
                        .filter(|p| matches!(p, Operand::Ptr(_)))
                        .count();
                    let object = call
                        .args
                        .iter()
                        .zip(sig.params())
                        .all(|(&v, p)| !is(&derived, v) || matches!(p, Operand::Ptr(_)));
                    match (
                        pointers == 1 && object && reads == 1,
                        parts.types.get(inst.ty),
                    ) {
                        (true, Some(Type::Void)) => Some(MemoryUse::Drop(id)),
                        (true, _) => read(inst.ty, true).map(|r| MemoryUse::Read(id, r)),
                        (false, _) => None,
                    }
                }
                _ => None,
            },
            _ => None,
        };
        match u {
            Some(u) => {
                m.reserve(&mut uses, 1)?;
                uses.push(u);
            }
            None => {
                ok = false;
                break;
            }
        }
    }
    m.release(derived.capacity() * std::mem::size_of::<InstId>());
    if !ok {
        m.release(uses.capacity() * std::mem::size_of::<MemoryUse>());
        return Ok(None);
    }
    Ok(Some(uses))
}

/// Delete from the body of entry `func` each store into absent threadgroup memory in `dropped`
/// ([`absent_memory`]): no defined read observes what it would write.
fn drop_absent_stores(
    parts: &mut ModuleParts,
    func: FuncId,
    dropped: &[InstId],
    m: &mut Meter,
) -> Result<(), Error> {
    if dropped.is_empty() {
        return Ok(());
    }
    let body = parts
        .funcs
        .get(func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry has no body"))?;
    let rebuilt = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(body, m, held)?;
        for &id in dropped {
            m.tick(1)?;
            edit.delete(id);
        }
        apply(body, &edit, &|c| c, m)
    })?;
    let f = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    if let Some(old) = f.body.replace(rebuilt) {
        old.release(m);
    }
    Ok(())
}

/// Make each instruction of `reads` in the body of entry `func` the absent read it is paired with
/// ([`FunctionConstant::AbsentRead`] of a texture call, [`FunctionConstant::AbsentMemory`] of a
/// threadgroup load or atomic): a call with no operands and the same result, through a
/// declaration of type `T ()` named for no AIR intrinsic (the callee states the operation, as the
/// table word's declaration does in `devirt`).
fn rewrite_absent_reads(
    parts: &mut ModuleParts,
    func: FuncId,
    reads: &[(InstId, FunctionConstant)],
    m: &mut Meter,
) -> Result<(), Error> {
    const NAME: &[u8] = b"absent argument read";
    let mut decls: Vec<(TypeId, FuncId)> = Vec::new();
    for &(id, op) in reads {
        let ret = parts
            .funcs
            .get(func)
            .and_then(|f| f.body.as_ref())
            .and_then(|b| b.insts.get(id))
            .map(|i| i.ty)
            .ok_or_else(|| invariant("an absent read out of the entry's body"))?;
        let decl = match decls.iter().find(|(r, _)| *r == ret) {
            Some(&(_, d)) => d,
            None => {
                let ty = parts.types.intern(
                    Type::Function {
                        ret,
                        params: Box::new([]),
                        vararg: false,
                    },
                    m,
                )?;
                let d = super::devirt::declare(parts, (NAME, ty), m)?;
                decls.push((ret, d));
                d
            }
        };
        let inst = parts
            .funcs
            .get_mut(func)
            .and_then(|f| f.body.as_mut())
            .and_then(|b| b.insts.get_mut(id))
            .ok_or_else(|| invariant("an absent read out of the entry's body"))?;
        let before = inst.kind.heap_bytes();
        let callee = Callee::Air {
            decl,
            op: AirIntrinsic::FunctionConstant(op),
        };
        match &mut inst.kind {
            InstKind::Call(call) => {
                call.callee = callee;
                call.args = Box::new([]);
                call.attrs = None;
                call.range = None;
            }
            // A threadgroup load becomes the call.
            kind @ InstKind::Load { .. } => {
                *kind = InstKind::Call(Box::new(crate::ir::Call {
                    callee,
                    args: Box::new([]),
                    fmf: FastMath::NONE,
                    tail: TailKind::None,
                    attrs: None,
                    range: None,
                }));
            }
            _ => return Err(invariant("an absent read that is no call or load")),
        }
        let after = inst.kind.heap_bytes();
        if after <= before {
            m.release(before - after);
        } else {
            m.charge(after - before)?;
        }
    }
    Ok(())
}

/// Which part of the entry's returned value its declared outputs store, and so which part a
/// return observes.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Stored {
    /// All of it: the entry returns its one declared output itself, or nothing.
    Whole,
    /// These fields of the returned struct (none when every output is gated off; a value the
    /// entry returns itself whose output is gated off is stored nowhere either).
    Fields(Vec<u32>),
}

/// The [`Stored`] part of entry `ep`'s returned value, of type `ret`.
fn stored(parts: &ModuleParts, ret: TypeId, ep: &EntryPoint) -> Stored {
    match parts.types.get(ret) {
        Some(Type::Struct { .. }) => Stored::Fields(ep.outputs.iter().map(|o| o.field).collect()),
        Some(Type::Void) | None => Stored::Whole,
        Some(_) if ep.outputs.is_empty() => Stored::Fields(Vec::new()),
        Some(_) => Stored::Whole,
    }
}

/// Give every read of a gated-off argument of entry `func` (`absent`, from [`Values::entry`]) its
/// absent value, and return the texture and sampler arguments the body still reads
/// (`Entry::absent_handles`). An argument whose role [`absent_null`] names reads as the null
/// constant of its pointer type. Any other absent argument has no value Metal defines: a read of it
/// that no side effect, branch, or return observes ([`observed_args`]) reads as `undef` (the read's
/// result is never observed, so every value is exact); an observed read of an argument passed by
/// value (its parameter is no pointer) reads the all-zero constant of its type, one value Metal's
/// undefined read may take; a texture or sampler read only through texture calls ([`absent_reads`])
/// has each call made [`FunctionConstant::AbsentRead`]; a texture or sampler read any other way
/// stays, returned, for lowering to read as the absent handle; threadgroup memory read only through
/// loads, stores, and atomics ([`absent_memory`]) has each load and atomic made
/// [`FunctionConstant::AbsentMemory`] and each store dropped; and any other observed read of a
/// pointer is refused as `Unsupported{FunctionConstantArgument, NotImplemented}` at the argument's
/// metadata, naming it. An absent argument nothing reads needs nothing.
pub(super) fn absent_arguments(
    parts: &mut ModuleParts,
    func: FuncId,
    ep: &EntryPoint,
    absent: &[ParamBinding],
    m: &mut Meter,
) -> Result<Vec<ParamBinding>, Error> {
    if absent.is_empty() {
        return Ok(Vec::new());
    }
    absent_texture_writes(parts, func, absent, m)?;
    let (values, reads, handles, dropped) = absent_values(parts, func, ep, absent, m)?;
    rewrite_absent_reads(parts, func, &reads, m)?;
    drop_absent_stores(parts, func, &dropped, m)?;
    if values.is_empty() {
        return Ok(handles);
    }
    let body = parts
        .funcs
        .get_mut(func)
        .and_then(|f| f.body.as_mut())
        .ok_or_else(|| invariant("the entry has no body"))?;
    for inst in body.insts.values_mut() {
        m.tick(1)?;
        inst.kind.for_each_value_mut(|v| {
            if let Value::Arg(a) = *v {
                if let Some((_, c)) = values.iter().find(|(p, _)| *p == a) {
                    *v = Value::Const(*c);
                }
            }
        });
    }
    Ok(handles)
}

/// Delete from the body of entry `func` every write to, and every fence of, an absent texture
/// argument in `absent`: a call of `air.write_texture_*` or `air.fence_texture_*` whose texture
/// operand is the argument itself. Metal drops such a write: on an M3 Max (macOS 26.7) and an M4
/// (macOS 27.0.1), with its validation layers off and on, a kernel's writes to a texture its
/// function constant removes changed nothing, not even a texture the encoder bound at the
/// argument's index all the same. Every other use of the argument (a read, a query, a sample, a
/// gather, an atomic, a merge with another texture) is left for [`absent_values`]: Metal gives such
/// a read no value.
fn absent_texture_writes(
    parts: &mut ModuleParts,
    func: FuncId,
    absent: &[ParamBinding],
    m: &mut Meter,
) -> Result<(), Error> {
    let textures: Vec<ArgIdx> = absent
        .iter()
        .filter(|b| matches!(b.role, ParamRole::Texture(_)))
        .map(|b| ArgIdx::new(b.param))
        .collect();
    if textures.is_empty() {
        return Ok(());
    }
    let body = parts
        .funcs
        .get(func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry has no body"))?;
    let rebuilt = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(body, m, held)?;
        let mut any = false;
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            if writes_absent_texture(&inst.kind, &textures) {
                edit.delete(id);
                any = true;
            }
        }
        if !any {
            return Ok(None);
        }
        apply(body, &edit, &|c| c, m).map(Some)
    })?;
    if let Some(rebuilt) = rebuilt {
        let f = parts
            .funcs
            .get_mut(func)
            .ok_or_else(|| invariant("entry out of range"))?;
        if let Some(old) = f.body.replace(rebuilt) {
            old.release(m);
        }
    }
    Ok(())
}

/// Whether `kind` is a texture write or fence ([`absent_texture_writes`]) whose texture operand is
/// one of `textures`.
fn writes_absent_texture(kind: &InstKind, textures: &[ArgIdx]) -> bool {
    let InstKind::Call(call) = kind else {
        return false;
    };
    let Callee::Air {
        op: AirIntrinsic::Texture(t),
        ..
    } = &call.callee
    else {
        return false;
    };
    matches!(t.op, TexOp::Write | TexOp::Fence)
        && t.layout()
            .index(TexRole::Texture)
            .and_then(|i| call.args.get(i))
            .is_some_and(|v| matches!(v, Value::Arg(a) if textures.contains(a)))
}

/// The values the absent arguments the entry's body reads take (see [`absent_arguments`]): the
/// constant of each argument read as one, the calls that become absent reads (texture calls
/// through [`absent_reads`], threadgroup loads and atomics through [`absent_memory`]) with the
/// read each becomes, the texture and sampler arguments the body reads any other way, which stay
/// for lowering's absent handle, and the stores into absent threadgroup memory, which are dropped;
/// or the refusal of the first one none takes.
type AbsentValues = (
    Vec<(ArgIdx, ConstId)>,
    Vec<(InstId, FunctionConstant)>,
    Vec<ParamBinding>,
    Vec<InstId>,
);

/// The [`AbsentValues`] of the absent arguments the body of entry `func` reads, or the refusal
/// of the first one it cannot read.
fn absent_values(
    parts: &mut ModuleParts,
    func: FuncId,
    ep: &EntryPoint,
    absent: &[ParamBinding],
    m: &mut Meter,
) -> Result<AbsentValues, Error> {
    let f = parts
        .funcs
        .get(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    let body = f
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry has no body"))?;
    let Some(Type::Function { ret, params, .. }) = parts.types.get(f.ty) else {
        return Err(invariant("the entry has no function type"));
    };
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let reads = {
        let mut private = Private::new(body, &parts.types, &parts.consts, params, &layouts, m)?;
        let reads = observed_args(
            body,
            params.len(),
            &stored(parts, *ret, ep),
            &mut private,
            m,
        );
        private.release(m);
        reads?
    };
    m.release(parts.types.len() * std::mem::size_of::<Option<Layout>>());
    let mut values = Vec::with_capacity(absent.len());
    let mut calls = Vec::new();
    let mut handles = Vec::new();
    let mut dropped = Vec::new();
    for b in absent {
        let arg = ArgIdx::new(b.param);
        let Some(read) = reads.get(arg.index()).copied().flatten() else {
            continue;
        };
        let ty = params
            .get(arg.index())
            .copied()
            .ok_or_else(|| invariant("an absent argument out of the entry's parameters"))?;
        let pointer = matches!(parts.types.get(ty), Some(Type::Ptr { .. }));
        let absent = absent_null(&b.role);
        let value = if pointer && absent == Absent::Null {
            Const::Null
        } else if read == Read::Unobserved {
            Const::Undef
        } else if !pointer {
            // Not declared (MSL 5.8.1.2), so Metal gives the read no value: zero is one it may
            // take, and contains whatever observes it.
            Const::Null
        } else if absent == Absent::Reads {
            match absent_reads(body, arg, &b.role, m)? {
                // Every read is a texture call, which becomes an absent read: nothing reads the
                // argument after it.
                Some(reads) => calls.extend(
                    reads
                        .into_iter()
                        .map(|(id, t)| (id, FunctionConstant::AbsentRead(t))),
                ),
                // Read another way: lowering follows the handle to the texture calls it reaches.
                None => handles.push(b.clone()),
            }
            continue;
        } else if let Some(uses) = match absent {
            Absent::Memory => absent_memory(body, parts, arg, m)?,
            Absent::Null | Absent::Reads | Absent::Never => None,
        } {
            // Every access through the memory becomes an absent read or goes; what derives its
            // addresses then reads `undef`, which nothing observes.
            for u in uses {
                match u {
                    MemoryUse::Read(id, a) => calls.push((id, FunctionConstant::AbsentMemory(a))),
                    MemoryUse::Drop(id) => dropped.push(id),
                }
            }
            Const::Undef
        } else {
            let at = Location::metadata(Some(ep.name.clone()), Some(b.param), None);
            let mut refusal =
                Unsupported::not_implemented(Feature::FunctionConstantArgument).at(at);
            if let Some(name) = &b.name {
                refusal = refusal.naming(name.clone());
            }
            return Err(refusal.into());
        };
        let c = parts.consts.intern(Constant { ty, value }, m)?;
        values.push((arg, c));
    }
    Ok((values, calls, handles, dropped))
}

/// Delete from the body of entry `func` what nothing observes ([`observation`]): a write into a
/// stack allocation whose bytes no observed read reaches before the allocation escapes (a plain
/// `store`, a non-volatile `memset`, `memcpy`, or `memmove`), and then every [`value_only`]
/// instruction that no instruction that stays reads. Nothing else goes: a store into device,
/// threadgroup, or image memory, an atomic, a barrier, a call with an effect, a volatile access, an
/// allocation, and a terminator are never [`value_only`] or deferred, so the observation starts
/// from them and they stay. A [`value_only`] instruction the observation observes is read by an
/// instruction that stays, or only by the address of a write it deletes; one it does not observe
/// stays only where an instruction that stays reads it (the part of a returned struct no output
/// stores). The body stays valid, and nothing any output, side effect, or branch sees changes.
///
/// Specialization leaves such code where it deletes uses: SCCP removes the uses its constants
/// guard, and an absent argument's unobserved reads take `undef`, so an absent texture table's
/// element addresses, an `array_ref` built from it, a phi of a texture with an absent one, or a
/// handle table copied into a stack struct that nothing reads can outlive every use; no lowering of
/// them is then ever needed.
pub(super) fn unobserved_code(
    parts: &mut ModuleParts,
    func: FuncId,
    ep: &EntryPoint,
    m: &mut Meter,
) -> Result<(), Error> {
    let rebuilt = {
        let f = parts
            .funcs
            .get(func)
            .ok_or_else(|| invariant("entry out of range"))?;
        let body = f
            .body
            .as_ref()
            .ok_or_else(|| invariant("the entry has no body"))?;
        let Some(Type::Function { ret, params, .. }) = parts.types.get(f.ty) else {
            return Err(invariant("the entry has no function type"));
        };
        let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
        let n = body.insts.len();
        let gone_bytes = n * std::mem::size_of::<bool>();
        m.charge(gone_bytes)?;
        let gone = {
            let mut private = Private::new(body, &parts.types, &parts.consts, params, &layouts, m)?;
            let observed = observation(
                body,
                params.len(),
                &stored(parts, *ret, ep),
                &mut private,
                m,
            );
            let gone = observed.and_then(|(o, bytes)| {
                let mut gone = vec![false; n];
                for (id, inst) in body.insts.iter() {
                    m.tick(1)?;
                    let seen = o.observed.get(id.index()).copied().unwrap_or(true);
                    if let Some(g) = gone.get_mut(id.index()) {
                        *g = value_only(&inst.kind) || (!seen && private.deferred(id));
                    }
                }
                m.release(bytes);
                Ok(gone)
            });
            private.release(m);
            gone
        };
        m.release(parts.types.len() * std::mem::size_of::<Option<Layout>>());
        let rebuilt = gone.and_then(|mut gone| {
            if !kept_reads(body, &mut gone, m)? {
                return Ok(None);
            }
            scoped(m, |m, held| {
                let mut edit = BodyEdit::keep_all(body, m, held)?;
                for (i, g) in gone.iter().enumerate() {
                    if *g {
                        edit.delete(InstId::new(u32::try_from(i).map_err(|_| invariant("ids"))?));
                    }
                }
                apply(body, &edit, &|c| c, m).map(Some)
            })
        });
        m.release(gone_bytes);
        rebuilt?
    };
    if let Some(body) = rebuilt {
        let f = parts
            .funcs
            .get_mut(func)
            .ok_or_else(|| invariant("entry out of range"))?;
        if let Some(old) = f.body.replace(body) {
            old.release(m);
        }
    }
    Ok(())
}

/// Keep, among the instructions `gone` marks for deletion, every one an instruction that stays
/// reads (transitively), so no surviving use names a deleted instruction. Whether any is still
/// marked.
fn kept_reads(body: &Body, gone: &mut [bool], m: &mut Meter) -> Result<bool, Error> {
    let bytes = gone.len() * std::mem::size_of::<InstId>();
    m.charge(bytes)?;
    let mut work: Vec<InstId> = body
        .insts
        .ids()
        .filter(|id| !gone.get(id.index()).copied().unwrap_or(false))
        .collect();
    while let Some(id) = work.pop() {
        m.tick(1)?;
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        inst.kind.for_each_value(|v| {
            if let Value::Inst(j) = v {
                if let Some(g) = gone.get_mut(j.index()) {
                    if *g {
                        *g = false;
                        work.push(j);
                    }
                }
            }
        });
    }
    m.release(bytes);
    Ok(gone.iter().any(|g| *g))
}

/// How the body reads a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// Only instructions whose results nothing observes read it.
    Unobserved,
    /// A side effect, a branch, or a return depends on it.
    Observed,
}

/// Whether `kind` only computes its result from its operands, so an instruction nothing observes
/// changes nothing: SCCP's pure operations (arithmetic, casts, compares, `select`, element and
/// member access, GEPs, phis, non-volatile loads), calls of the value-only intrinsic families
/// (math, conversions, packing, `bool` reductions; `llvm` float and integer operations) that take
/// no pointer, and an absent read other than a texture atomic ([`FunctionConstant::AbsentRead`]).
fn value_only(kind: &InstKind) -> bool {
    if is_pure(kind) {
        return true;
    }
    let InstKind::Call(call) = kind else {
        return false;
    };
    match &call.callee {
        // A pointer operand is memory the call may write (`sincos` stores its cosine through
        // one): such a call is a side effect, not a value.
        Callee::Air { op, .. } => {
            // An absent read only reports that something observes its result; an absent texture
            // atomic also writes, which Metal's answer for is not measured.
            if let AirIntrinsic::FunctionConstant(FunctionConstant::AbsentRead(t)) = op {
                return !matches!(t.op, TexOp::Atomic(_));
            }
            // So does an absent threadgroup load; an absent atomic is an access that marks
            // whether or not its result is observed, as the write it stands for would.
            if let AirIntrinsic::FunctionConstant(FunctionConstant::AbsentMemory(a)) = op {
                return !a.atomic;
            }
            matches!(
                op,
                AirIntrinsic::Math(_)
                    | AirIntrinsic::Convert(_)
                    | AirIntrinsic::Pack(_)
                    | AirIntrinsic::BoolReduce(_)
            ) && !op
                .signature()
                .params()
                .iter()
                .any(|p| matches!(p, Operand::Ptr(_)))
        }
        Callee::Llvm { op, .. } => matches!(
            op,
            LlvmIntrinsic::Float { .. }
                | LlvmIntrinsic::Int { .. }
                | LlvmIntrinsic::Abs { .. }
                | LlvmIntrinsic::Cttz { .. }
                | LlvmIntrinsic::Ctlz { .. }
        ),
        Callee::Direct(_) | Callee::Indirect { .. } | Callee::UnknownIntrinsic(_) => false,
    }
}

/// The observation's state: how each parameter is read, which instructions are observed, and
/// the worklist of those whose operands are not yet observed.
struct Observation {
    reads: Vec<Option<Read>>,
    observed: Vec<bool>,
    work: Vec<InstId>,
}

impl Observation {
    /// Mark `v` observed: a parameter is read observably; an instruction not yet observed joins
    /// the worklist.
    fn observe(&mut self, v: Value) {
        match v {
            Value::Arg(a) => {
                if let Some(r) = self.reads.get_mut(a.index()) {
                    *r = Some(Read::Observed);
                }
            }
            Value::Inst(j) => {
                if let Some(o) = self.observed.get_mut(j.index()) {
                    if !*o {
                        *o = true;
                        self.work.push(j);
                    }
                }
            }
            Value::Const(_) => {}
        }
    }

    /// Mark `v` observed as a value: a pointer into an allocation escapes it ([`private`]).
    fn value(&mut self, v: Value, private: &mut Private<'_>) {
        self.observe(v);
        if let Some(a) = private.alloca(v) {
            private.escape(a, |w| self.observe(Value::Inst(w)));
        }
    }

    /// Observe observed instruction `id`: the bytes a load or copy reads observe the writes they
    /// overlap, and its operands are observed ([`Observation::operands`]).
    fn visit(
        &mut self,
        id: InstId,
        body: &Body,
        private: &mut Private<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let Some(inst) = body.insts.get(id) else {
            return Ok(());
        };
        private.read(inst, m, |w| self.observe(Value::Inst(w)))?;
        self.operands(id, inst, private, Bytes::Observed);
        Ok(())
    }

    /// Observe the operands of instruction `id`. An address (of a load, a store, a memory
    /// intrinsic) and the base of a pointer derivation into an allocation are observed without
    /// escaping it; every other operand is a value. With [`Bytes::Deferred`] (a deferred write,
    /// seen before any read reaches it), the bytes it writes (a store's value, a `memset`'s fill
    /// byte) are left out: they wait for a read, while where and how much it writes does not.
    fn operands(&mut self, id: InstId, inst: &Inst, private: &mut Private<'_>, bytes: Bytes) {
        let deferred = bytes == Bytes::Deferred;
        match &inst.kind {
            InstKind::Load { ptr, .. } => self.observe(*ptr),
            InstKind::Store { val, ptr, .. } => {
                if !deferred {
                    self.value(*val, private);
                }
                self.observe(*ptr);
            }
            InstKind::Gep { base, indices, .. } => {
                if private.derives(id) {
                    self.observe(*base);
                } else {
                    self.value(*base, private);
                }
                for i in indices.iter() {
                    self.value(*i, private);
                }
            }
            InstKind::Cast { v, .. } if private.derives(id) => self.observe(*v),
            InstKind::Call(call) => {
                let addresses = Private::addresses(call);
                let written = Private::fill(call).filter(|_| deferred);
                if let Callee::Indirect { callee, .. } = call.callee {
                    self.value(callee, private);
                }
                for (k, v) in call.args.iter().enumerate() {
                    if Some(k) == written {
                        continue;
                    }
                    if k < addresses {
                        self.observe(*v);
                    } else {
                        self.value(*v, private);
                    }
                }
            }
            kind => kind.for_each_value(|v| self.value(v, private)),
        }
    }
}

/// Whether [`Observation::operands`] observes the bytes a write writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bytes {
    /// The instruction is observed: every operand is.
    Observed,
    /// A deferred write no read has reached yet: all but the bytes it writes.
    Deferred,
}

/// How the body reads each of the entry's `params` parameters (`None`: not at all): the reads of
/// [`observation`].
fn observed_args(
    body: &Body,
    params: usize,
    stored: &Stored,
    private: &mut Private<'_>,
    m: &mut Meter,
) -> Result<Vec<Option<Read>>, Error> {
    let (o, bytes) = observation(body, params, stored, private, m)?;
    m.release(bytes);
    Ok(o.reads)
}

/// The observation of `body`: how it reads each of the entry's `params` parameters (`None`: not
/// at all), and which instructions it observes. An instruction is observed when it is not
/// [`value_only`] (a side effect, a branch) or an observed instruction reads its result; a
/// parameter an observed instruction reads is [`Read::Observed`]. The bytes a write into a stack
/// allocation writes are observed only once an observed read reaches them or the allocation
/// escapes; its other operands are observed at once (`private`). A return observes only the part of
/// its value the declared outputs store (`stored`): walking the returned struct's `insertvalue`
/// chain, the values inserted into a stored field (the last insert of each), and the whole of any
/// other origin the chain reaches. One backward worklist over the instructions; the ledger bytes it
/// holds are returned with it, for the caller to release.
fn observation(
    body: &Body,
    params: usize,
    stored: &Stored,
    private: &mut Private<'_>,
    m: &mut Meter,
) -> Result<(Observation, usize), Error> {
    let n = body.insts.len();
    let bytes = n * (std::mem::size_of::<bool>() + std::mem::size_of::<InstId>())
        + params * std::mem::size_of::<Option<Read>>();
    m.charge(bytes)?;
    let mut o = Observation {
        reads: vec![None; params],
        observed: vec![false; n],
        work: Vec::with_capacity(n),
    };
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                if let Some(r) = o.reads.get_mut(a.index()) {
                    r.get_or_insert(Read::Unobserved);
                }
            }
        });
    }
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        match (&inst.kind, stored) {
            (InstKind::Ret(Some(v)), Stored::Fields(fields)) => {
                // The fields a later insert already set, whose earlier inserts are dead.
                let mut set: Vec<u32> = Vec::new();
                let mut cur = *v;
                // A chain longer than the body revisits an insert: a cycle only unreachable code
                // can form (an instruction there may read itself). Observe it whole.
                let mut steps = 0usize;
                loop {
                    m.tick(1)?;
                    if fields.is_empty() {
                        break;
                    }
                    steps += 1;
                    if steps > n {
                        o.value(cur, private);
                        break;
                    }
                    let Value::Inst(i) = cur else {
                        o.value(cur, private);
                        break;
                    };
                    let Some(InstKind::InsertValue { agg, val, indices }) =
                        body.insts.get(i).map(|x| &x.kind)
                    else {
                        o.value(cur, private);
                        break;
                    };
                    let Some(&field) = indices.first() else {
                        o.value(cur, private);
                        break;
                    };
                    if fields.contains(&field) && !set.contains(&field) {
                        o.value(*val, private);
                    }
                    if indices.len() == 1 && !set.contains(&field) {
                        set.push(field);
                    }
                    cur = *agg;
                }
            }
            // Where and how much a deferred write writes is observed now; its bytes wait.
            _ if private.deferred(id) => o.operands(id, inst, private, Bytes::Deferred),
            (kind, _) if !value_only(kind) => {
                if let Some(x) = o.observed.get_mut(id.index()) {
                    *x = true;
                }
                o.work.push(id);
            }
            _ => {}
        }
    }
    while let Some(id) = o.work.pop() {
        m.tick(1)?;
        o.visit(id, body, private, m)?;
    }
    Ok((o, bytes))
}

/// An interface refusal without a location, located at the ABI metadata of `entry` (argument
/// `arg`, when it is one parameter's).
fn located(e: Error, entry: &DiagName, arg: Option<u32>) -> Error {
    match e {
        Error::Unsupported(u) if u.location.is_none() => {
            Error::Unsupported(u.at(Location::metadata(Some(entry.clone()), arg, None)))
        }
        other => other,
    }
}
