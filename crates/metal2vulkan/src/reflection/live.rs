//! What Metal's compiler keeps of a specialized entry once it has linked its pipeline: aggressive
//! dead code elimination by LLVM ADCE's rules, which Metal's answers follow (measured on an M4
//! under macOS 27.0.1; the M3 Max's confirmation follows).
//!
//! - **Roots.** Every instruction with an effect outside the function: a store, an atomic, a
//!   fence, a volatile or ordered load, a memory copy or set into memory other than the
//!   function's own stack, and a call other than an `air.`/`llvm.` operation without an effect
//!   (a texture write, atomic or fence, a barrier, a discard, an imageblock store, an indirect
//!   command's encoding, an asynchronous copy, a call canonicalization left, among them). A mesh
//!   function's write of an output field the pipeline keeps ([`Roots::mesh`]). The `ret`, which
//!   demands the returned fields the pipeline keeps ([`Returned`]), and `unreachable`. Every
//!   loop: the terminator of each block with a retreating edge, LLVM ADCE's default, which
//!   removes no loop, so a loop's trip count is kept whatever its result feeds (a texel that sets a
//!   trip count is kept with the loop's result going only to a varying no
//!   fragment function reads).
//! - **Propagation.** A kept instruction keeps its operands. A struct value is kept field by
//!   field: an `insertvalue` keeps its member where that field is demanded and passes the other
//!   demanded fields to its aggregate, an `extractvalue` demands its field, and a `phi`, `select`
//!   or `freeze` of structs passes on the fields demanded of it. A kept instruction keeps its
//!   block, and a kept block keeps the branch or switch of every block it is control dependent on
//!   (its post-dominance frontier), so a texel that decides whether a kept store runs is kept. A
//!   kept `phi` keeps its incoming blocks, so the branch that chooses its value is kept; a `select`
//!   keeps its condition only where its result is kept (a texel choosing an unread varying's value
//!   is not). A load from the function's own stack (an `alloca`, through `getelementptr` and
//!   pointer casts) keeps every store, copy or set into that allocation; an allocation whose
//!   address goes anywhere else keeps them all.
//! - **Linear in the function.** Each instruction is processed once, and again only when the
//!   fields demanded of a struct value grow (at most once per field); each block's control
//!   dependences are its post-dominance frontier, computed once from the post-dominator tree.
//!   Everything is charged to the meter.
//! - **Kept beyond Metal, where unmeasured** (so a state may be refused that Metal makes), each
//!   rule on only where [`Roots::rules`] keeps its [`LinkedUseGap`]: a block that reaches no exit
//!   (an infinite loop) keeps every branch (`EndlessLoop`); a load from the function's own stack
//!   keeps every write into its allocation, and an allocation whose address goes elsewhere keeps
//!   them all (`StackMemory`: no measured row passed a value through memory); a mesh function's
//!   write of a built-in field its constants gate off is kept (`GatedOffMeshBuiltin`). With a rule
//!   off, what only it keeps is dropped: the answer is then at most Metal's, so a use kept with
//!   every rule off is one Metal's measured rule keeps (`linked.rs`).

use std::collections::BTreeSet;

use super::linked::{LinkedUseGap, Rules};
use crate::air;
use crate::air::abi::MeshFieldRole;
use crate::air::intrinsics::{
    AirIntrinsic, AsyncCopy, Fragment, FunctionTable, Imageblock, LlvmIntrinsic, Mesh, TexOp,
};
use crate::air::ir::{BlockId, CastOp, Const, InstKind, Type, TypeId, Value};
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::PostDomTree;

/// "No allocation, no slot" in the per-instruction tables.
const NONE: u32 = u32::MAX;

/// Work units between meter polls.
const BATCH: u32 = 1024;

/// What the function's `ret` demands of the value it returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Returned {
    /// Nothing: the function returns `void`, or the pipeline keeps none of its outputs.
    Nothing,
    /// The whole value.
    All,
    /// These fields of the returned struct (a value of another type: all of it, where any).
    Fields(BTreeSet<u32>),
}

/// What a function's linking keeps beyond its effects.
pub(crate) struct Roots {
    /// What its `ret` demands.
    pub returned: Returned,
    /// For a mesh function, the data slots whose writes the pipeline keeps (`None`: not a mesh
    /// function, or every write kept).
    pub mesh: Option<MeshSlots>,
    /// The rules kept beyond Metal that apply (the module docs).
    pub rules: Rules,
}

/// A mesh function's data slots whose writes its pipeline keeps: those whose user key a fragment
/// input reads.
pub(crate) struct MeshSlots {
    /// `air.set_vertex_data_mesh` slots kept.
    pub vertex: BTreeSet<u64>,
    /// `air.set_primitive_data_mesh` slots kept.
    pub primitive: BTreeSet<u64>,
    /// The built-in fields its function constants gate off.
    pub gated_off: BTreeSet<MeshBuiltin>,
}

/// A built-in field of a mesh function's output, which one `air.set_*_mesh` call writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MeshBuiltin {
    Position,
    PointSize,
    ClipDistance,
    PrimitiveCulled,
    PrimitiveId,
    RenderTargetArrayIndex,
    ViewportArrayIndex,
}

impl MeshBuiltin {
    /// The built-in field mesh output field `role` is, if any.
    pub(crate) fn of_field(role: &MeshFieldRole) -> Option<MeshBuiltin> {
        match role {
            MeshFieldRole::Position { .. } => Some(MeshBuiltin::Position),
            MeshFieldRole::PointSize => Some(MeshBuiltin::PointSize),
            MeshFieldRole::ClipDistance { .. } => Some(MeshBuiltin::ClipDistance),
            MeshFieldRole::PrimitiveCulled => Some(MeshBuiltin::PrimitiveCulled),
            MeshFieldRole::PrimitiveId => Some(MeshBuiltin::PrimitiveId),
            MeshFieldRole::RenderTargetArrayIndex => Some(MeshBuiltin::RenderTargetArrayIndex),
            MeshFieldRole::ViewportArrayIndex => Some(MeshBuiltin::ViewportArrayIndex),
            _ => None,
        }
    }

    /// The built-in field mesh write `op` writes, if any.
    fn of_write(op: Mesh) -> Option<MeshBuiltin> {
        match op {
            Mesh::Position => Some(MeshBuiltin::Position),
            Mesh::PointSize => Some(MeshBuiltin::PointSize),
            Mesh::ClipDistance => Some(MeshBuiltin::ClipDistance),
            Mesh::PrimitiveCulled => Some(MeshBuiltin::PrimitiveCulled),
            Mesh::PrimitiveId => Some(MeshBuiltin::PrimitiveId),
            Mesh::RenderTargetArrayIndex(_) => Some(MeshBuiltin::RenderTargetArrayIndex),
            Mesh::ViewportArrayIndex(_) => Some(MeshBuiltin::ViewportArrayIndex),
            Mesh::ThreadgroupsPerGrid
            | Mesh::VertexData(_)
            | Mesh::Index
            | Mesh::Indices(_)
            | Mesh::PrimitiveData(_)
            | Mesh::PrimitiveCount => None,
        }
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        format!("linked use: {detail}"),
    ))
}

/// Whether each parameter of `entry`'s function is read by an instruction its pipeline's linking
/// keeps under `roots` (the module docs), by parameter index.
pub(crate) fn kept_parameters(
    entry: &air::Entry,
    roots: &Roots,
    m: &mut Meter,
) -> Result<Vec<bool>, Error> {
    let function = entry
        .module
        .func(entry.func)
        .ok_or_else(|| invariant("the entry is not in its module"))?;
    let body = function
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry has no body"))?;
    let params = match entry.module.types().get(function.ty) {
        Some(Type::Function { params, .. }) => params.len(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let pdt = PostDomTree::compute(&entry.cfg, m)?;
    let result = Engine::new(entry, body, params, m).and_then(|mut engine| {
        let run = engine.run(roots, &pdt, m);
        m.release(engine.charged);
        run.map(|()| engine.params)
    });
    m.release(pdt.ledger_bytes());
    result
}

/// What the values of some parameters reach, followed forward from them ([`forward_reach`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    /// Nothing the linking keeps: only instructions without an effect, fields of the returned
    /// value it keeps none of, and branches that decide nothing it keeps.
    Nothing,
    /// Something it keeps: a root, a loop's latch, a field of the returned value it keeps, a
    /// branch that decides whether one of those runs, or a write into a stack allocation whose
    /// address goes elsewhere.
    Kept,
}

/// What parameters `params` of `entry`'s function reach under `roots` (the module docs' roots):
/// the backward analysis' dual, an independent check of it before a compile gives a parameter
/// it drops no value. An instruction that reads a reached value is reached, a struct value field
/// by field (`insertvalue`, `extractvalue`, `phi`, `select`, `freeze`); a branch or switch whose
/// condition is reached reaches every instruction of each block control dependent on it, and the
/// `phi`s that choose by an edge out of those blocks or its own; a write of a reached value into
/// the function's own stack, under `StackMemory`, reaches every load from that allocation and
/// every copy out of it, and is kept where the allocation's address goes elsewhere. Reaching a
/// root, a loop's latch, a field of the returned value `roots` keeps, or, where `EndlessLoop`
/// applies, a branch, is [`Reach::Kept`]. Linear in the function: each instruction is queued
/// again only when the fields it carries grow, each block's dependents and each allocation's
/// readers are visited once.
pub(crate) fn forward_reach(
    entry: &air::Entry,
    params: &BTreeSet<u32>,
    roots: &Roots,
    m: &mut Meter,
) -> Result<Reach, Error> {
    let function = entry
        .module
        .func(entry.func)
        .ok_or_else(|| invariant("the entry is not in its module"))?;
    let body = function
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry has no body"))?;
    let count = match entry.module.types().get(function.ty) {
        Some(Type::Function { params, .. }) => params.len(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let pdt = PostDomTree::compute(&entry.cfg, m)?;
    let result = Engine::new(entry, body, count, m).and_then(|mut engine| {
        let reach = engine.forward(params, roots, &pdt, m);
        m.release(engine.charged);
        reach
    });
    m.release(pdt.ledger_bytes());
    result
}

/// The analysis' state (the module docs), its tables indexed by instruction, block, parameter
/// or struct slot.
struct Engine<'a> {
    entry: &'a air::Entry,
    body: &'a air::ir::Body,
    /// The block of each instruction.
    block_of: Vec<u32>,
    /// Whether each instruction is kept (a struct value tracked by field: any field demanded).
    live: Vec<bool>,
    /// A struct value tracked by field: its first word in `words`, else [`NONE`].
    slot: Vec<u32>,
    /// The demanded fields of each tracked struct value, a bit per field.
    words: Vec<u64>,
    /// Whether each parameter is read by a kept instruction.
    params: Vec<bool>,
    /// Whether each block is kept, and whether its `phi`s' incoming blocks are.
    blocks: Vec<bool>,
    phis: Vec<bool>,
    /// The `alloca` a stack pointer points into, else [`NONE`].
    base: Vec<u32>,
    /// By `alloca`: whether a kept instruction reads it, and whether its address goes elsewhere.
    read: Vec<bool>,
    escaped: Vec<bool>,
    /// `(alloca, instruction)` for each store, copy or set into an allocation, sorted.
    writes: Vec<(u32, u32)>,
    /// `(block, controlling block)` for each control dependence, sorted.
    controls: Vec<(u32, u32)>,
    /// Instructions kept and not processed yet.
    work: Vec<u32>,
    /// Bytes charged for the tables above.
    charged: usize,
    units: u32,
}

/// A vector of `len` copies of `value`, charged to `m` and counted in `charged`.
fn filled<T: Clone>(
    len: usize,
    value: T,
    m: &mut Meter,
    charged: &mut usize,
) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    m.grow(&mut v, len)?;
    *charged = charged.saturating_add(v.capacity().saturating_mul(std::mem::size_of::<T>()));
    v.resize(len, value);
    Ok(v)
}

/// Push `value` onto `v`, charging growth to `m` and counting it in `charged`.
fn push<T>(v: &mut Vec<T>, value: T, m: &mut Meter, charged: &mut usize) -> Result<(), Error> {
    let before = v.capacity();
    m.grow(v, 1)?;
    *charged =
        charged.saturating_add((v.capacity() - before).saturating_mul(std::mem::size_of::<T>()));
    v.push(value);
    Ok(())
}

/// The field count of struct type `ty`, else `None`.
fn struct_fields(entry: &air::Entry, ty: TypeId) -> Option<usize> {
    match entry.module.types().get(ty) {
        Some(Type::Struct { fields, .. }) => Some(fields.len()),
        _ => None,
    }
}

fn words_for(fields: usize) -> usize {
    fields.div_ceil(64)
}

/// Whether `kind` passes the fields demanded of its struct result on to its operands field by
/// field.
fn field_wise(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::InsertValue { .. }
            | InstKind::Phi { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
    )
}

/// The destination and source pointers of a memory set, copy or move.
fn memory_operands(kind: &InstKind) -> Option<(Value, Option<Value>)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match &call.callee {
        air::ir::Callee::Llvm {
            op: LlvmIntrinsic::Memset { .. },
            ..
        } => Some((*call.args.first()?, None)),
        air::ir::Callee::Llvm {
            op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
            ..
        } => Some((*call.args.first()?, Some(*call.args.get(1)?))),
        _ => None,
    }
}

impl<'a> Engine<'a> {
    fn new(
        entry: &'a air::Entry,
        body: &'a air::ir::Body,
        params: usize,
        m: &mut Meter,
    ) -> Result<Engine<'a>, Error> {
        let insts = body.insts.len();
        let blocks = body.blocks.len();
        let mut charged = 0;
        let mut engine = Engine {
            entry,
            body,
            block_of: filled(insts, NONE, m, &mut charged)?,
            live: filled(insts, false, m, &mut charged)?,
            slot: filled(insts, NONE, m, &mut charged)?,
            words: Vec::new(),
            params: filled(params, false, m, &mut charged)?,
            blocks: filled(blocks, false, m, &mut charged)?,
            phis: filled(blocks, false, m, &mut charged)?,
            base: filled(insts, NONE, m, &mut charged)?,
            read: filled(insts, false, m, &mut charged)?,
            escaped: filled(insts, false, m, &mut charged)?,
            writes: Vec::new(),
            controls: Vec::new(),
            work: Vec::new(),
            charged,
            units: 0,
        };
        for (b, block) in body.blocks.iter() {
            for i in block.inst_ids() {
                if let Some(slot) = engine.block_of.get_mut(i.raw() as usize) {
                    *slot = b.raw();
                }
            }
        }
        Ok(engine)
    }

    /// Count `units` of work, polling the meter once a batch is full.
    fn step(&mut self, m: &mut Meter, units: usize) -> Result<(), Error> {
        self.units = self
            .units
            .saturating_add(u32::try_from(units).unwrap_or(u32::MAX));
        if self.units >= BATCH {
            let units = self.units;
            self.units = 0;
            m.tick(units)?;
        }
        Ok(())
    }

    fn inst(&self, i: u32) -> Option<&'a air::ir::Inst> {
        let body: &'a air::ir::Body = self.body;
        body.insts.get(air::ir::InstId::new(i))
    }

    /// The `alloca` value `v` points into, else [`NONE`].
    fn base_of(&self, v: Value) -> u32 {
        match v {
            Value::Inst(i) => self.base.get(i.raw() as usize).copied().unwrap_or(NONE),
            _ => NONE,
        }
    }

    /// Value `v` goes somewhere other than a load, a store's or a memory operation's address:
    /// the allocation it points into, if any, escapes.
    fn escape(&mut self, v: Value) {
        let a = self.base_of(v);
        if let Some(e) = self.escaped.get_mut(a as usize) {
            *e = true;
        }
    }

    fn run(
        &mut self,
        roots: &Roots,
        pdt: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        self.slots(m)?;
        self.stack(m)?;
        self.dependences(pdt, m)?;
        self.seed(roots, pdt, m)?;
        while let Some(i) = self.work.pop() {
            self.step(m, 1)?;
            self.process(i, roots, m)?;
        }
        m.tick(self.units)?;
        self.units = 0;
        Ok(())
    }

    /// Give each struct value merged field by field its words of demanded fields.
    fn slots(&mut self, m: &mut Meter) -> Result<(), Error> {
        let body: &'a air::ir::Body = self.body;
        let mut next = 0usize;
        for (id, inst) in body.insts.iter() {
            if !field_wise(&inst.kind) {
                continue;
            }
            let Some(fields) = struct_fields(self.entry, inst.ty) else {
                continue;
            };
            let start = u32::try_from(next)
                .ok()
                .filter(|&s| s != NONE)
                .ok_or_else(|| invariant("too many struct fields"))?;
            if let Some(s) = self.slot.get_mut(id.raw() as usize) {
                *s = start;
            }
            next = next.saturating_add(words_for(fields));
        }
        self.step(m, body.insts.len())?;
        self.words = filled(next, 0u64, m, &mut self.charged)?;
        Ok(())
    }

    /// The stack: which `alloca` each pointer points into, the writes into each, and the
    /// allocations whose address goes elsewhere.
    fn stack(&mut self, m: &mut Meter) -> Result<(), Error> {
        let entry: &'a air::Entry = self.entry;
        let body: &'a air::ir::Body = self.body;
        // Bases in reverse postorder, where a value's definition comes before its uses (but a
        // `phi`'s, whose stack operands escape below).
        for b in entry.cfg.rpo() {
            for (id, inst) in body.block_insts(b) {
                let base = match &inst.kind {
                    InstKind::Alloca { .. } => id.raw(),
                    InstKind::Gep { base, .. } => self.base_of(*base),
                    InstKind::Cast {
                        op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                        v,
                    } => self.base_of(*v),
                    _ => NONE,
                };
                if let Some(slot) = self.base.get_mut(id.raw() as usize) {
                    *slot = base;
                }
            }
            self.step(m, 1)?;
        }
        for b in entry.cfg.rpo() {
            for (id, inst) in body.block_insts(b) {
                self.step(m, 1)?;
                match &inst.kind {
                    // An address read, offset or recast stays on the stack.
                    InstKind::Load { .. }
                    | InstKind::Gep { .. }
                    | InstKind::Cast {
                        op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                        ..
                    } => {}
                    InstKind::Store { val, ptr, .. } => {
                        self.escape(*val);
                        let a = self.base_of(*ptr);
                        if a != NONE {
                            push(&mut self.writes, (a, id.raw()), m, &mut self.charged)?;
                        }
                    }
                    kind => match memory_operands(kind) {
                        // A set, copy or move: a write into its destination's allocation, and
                        // a read of its source's where it is kept (`process`).
                        Some((dst, _)) => {
                            let a = self.base_of(dst);
                            if a != NONE {
                                push(&mut self.writes, (a, id.raw()), m, &mut self.charged)?;
                            }
                        }
                        None => kind.for_each_value(|v| self.escape(v)),
                    },
                }
            }
        }
        self.writes.sort_unstable();
        Ok(())
    }

    /// The control dependences: block `b` depends on the branch of each block in its
    /// post-dominance frontier, found by walking up the post-dominator tree from each successor
    /// of a block with two or more to that block's immediate post-dominator.
    fn dependences(&mut self, pdt: &PostDomTree<BlockId>, m: &mut Meter) -> Result<(), Error> {
        let entry: &'a air::Entry = self.entry;
        let cfg = &entry.cfg;
        for a in cfg.rpo() {
            if cfg.successors(a).len() < 2 {
                continue;
            }
            let stop = pdt.ipdom(a);
            for s in cfg.successors(a) {
                let mut runner = Some(s);
                while runner != stop {
                    let Some(b) = runner else { break };
                    if !pdt.is_reachable(b) {
                        break;
                    }
                    push(&mut self.controls, (b.raw(), a.raw()), m, &mut self.charged)?;
                    self.step(m, 1)?;
                    runner = pdt.ipdom(b);
                }
            }
        }
        self.controls.sort_unstable();
        self.controls.dedup();
        Ok(())
    }

    /// The roots (the module docs).
    fn seed(
        &mut self,
        roots: &Roots,
        pdt: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let entry: &'a air::Entry = self.entry;
        let body: &'a air::ir::Body = self.body;
        let cfg = &entry.cfg;
        let endless = cfg.rpo().any(|b| !pdt.is_reachable(b));
        for b in cfg.rpo() {
            for (id, inst) in body.block_insts(b) {
                self.step(m, 1)?;
                if self.is_root(&inst.kind, roots) {
                    self.keep(id.raw(), m)?;
                }
            }
            let latch = cfg.successors(b).any(|s| cfg.is_retreating(b, s));
            let branch = body.terminator(b).is_some_and(|t| {
                matches!(t.kind, InstKind::CondBr { .. } | InstKind::Switch { .. })
            });
            // Every loop is kept (its latch's terminator), and where a block reaches no exit
            // every branch (the module docs).
            if latch || (endless && branch && roots.rules.keeps(LinkedUseGap::EndlessLoop)) {
                self.keep_terminator(b.raw(), m)?;
            }
        }
        // The writes into an allocation whose address goes elsewhere.
        if !roots.rules.keeps(LinkedUseGap::StackMemory) {
            return Ok(());
        }
        for k in 0..self.writes.len() {
            let Some(&(a, w)) = self.writes.get(k) else {
                break;
            };
            if self.escaped.get(a as usize).copied().unwrap_or(true) {
                self.keep(w, m)?;
            }
        }
        Ok(())
    }

    /// Whether instruction `kind` is a root (the module docs).
    fn is_root(&self, kind: &InstKind, roots: &Roots) -> bool {
        match kind {
            InstKind::Store {
                ptr,
                volatile,
                atomic,
                ..
            } => *volatile || atomic.is_some() || self.base_of(*ptr) == NONE,
            InstKind::Load {
                volatile, atomic, ..
            } => *volatile || atomic.is_some(),
            InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Fence(_)
            | InstKind::Ret(_)
            | InstKind::Unreachable => true,
            InstKind::Call(call) => self.call_is_root(call, roots),
            InstKind::BinOp { .. }
            | InstKind::FNeg { .. }
            | InstKind::Cast { .. }
            | InstKind::ICmp { .. }
            | InstKind::FCmp { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
            | InstKind::ExtractElement { .. }
            | InstKind::InsertElement { .. }
            | InstKind::ShuffleVector { .. }
            | InstKind::ExtractValue { .. }
            | InstKind::InsertValue { .. }
            | InstKind::Alloca { .. }
            | InstKind::Gep { .. }
            | InstKind::Phi { .. }
            | InstKind::Br { .. }
            | InstKind::CondBr { .. }
            | InstKind::Switch { .. } => false,
        }
    }

    /// Whether call `call` has an effect the linking keeps.
    fn call_is_root(&self, call: &air::ir::Call, roots: &Roots) -> bool {
        use air::ir::Callee;
        match &call.callee {
            Callee::Air { op, .. } => match op {
                AirIntrinsic::Math(_)
                | AirIntrinsic::Convert(_)
                | AirIntrinsic::Pack(_)
                | AirIntrinsic::Derivative(_)
                | AirIntrinsic::BoolReduce(_)
                | AirIntrinsic::Sampler(_)
                | AirIntrinsic::Group(_)
                | AirIntrinsic::IsUniform(_)
                | AirIntrinsic::SimdgroupSize(_)
                | AirIntrinsic::GlobalBuiltin(_)
                | AirIntrinsic::FunctionConstant(_)
                | AirIntrinsic::Interpolate(_)
                | AirIntrinsic::AccelerationStructure(_) => false,
                AirIntrinsic::Texture(t) => matches!(
                    t.op,
                    TexOp::Write | TexOp::Atomic(_) | TexOp::Fence | TexOp::WriteImageblockSlice
                ),
                AirIntrinsic::Fragment(f) => *f == Fragment::Discard,
                AirIntrinsic::Imageblock(i) => matches!(i, Imageblock::Store(_)),
                AirIntrinsic::Coherent(c) => {
                    c.kind == air::intrinsics::AccessKind::Store || c.volatile
                }
                AirIntrinsic::FunctionTable(f) => *f == FunctionTable::SetIntersectionBuffer,
                AirIntrinsic::AsyncCopy(a) => {
                    !matches!(a, AsyncCopy::NullEvent | AsyncCopy::IsNullEvent)
                }
                AirIntrinsic::Mesh(op) => self.mesh_write_kept(*op, call, roots),
                AirIntrinsic::Atomic(_)
                | AirIntrinsic::Sync(_)
                | AirIntrinsic::SimdMatrix(_)
                | AirIntrinsic::Intersect(_)
                | AirIntrinsic::Tensor(_)
                | AirIntrinsic::IndirectCommand(_) => true,
            },
            Callee::Llvm { op, .. } => match op {
                LlvmIntrinsic::Float { .. }
                | LlvmIntrinsic::Int { .. }
                | LlvmIntrinsic::NarrowInt { .. }
                | LlvmIntrinsic::Abs { .. }
                | LlvmIntrinsic::Cttz { .. }
                | LlvmIntrinsic::Ctlz { .. } => false,
                LlvmIntrinsic::Memset { volatile, .. }
                | LlvmIntrinsic::Memcpy { volatile, .. }
                | LlvmIntrinsic::Memmove { volatile, .. } => {
                    *volatile
                        || call
                            .args
                            .first()
                            .is_none_or(|&dst| self.base_of(dst) == NONE)
                }
                LlvmIntrinsic::Agx(_) => true,
            },
            Callee::Direct(_)
            | Callee::Indirect { .. }
            | Callee::UnknownIntrinsic(_)
            | Callee::Unbound { .. } => true,
        }
    }

    /// Whether a mesh function's write `op` (`call`) is kept: a data write to a slot whose key no
    /// fragment input reads is not (one to a slot that is not a constant is), nor, where
    /// `GatedOffMeshBuiltin` is off, a write of a built-in field its constants gate off.
    fn mesh_write_kept(&self, op: Mesh, call: &air::ir::Call, roots: &Roots) -> bool {
        let Some(slots) = &roots.mesh else {
            return true;
        };
        let kept = match op {
            Mesh::VertexData(_) => &slots.vertex,
            Mesh::PrimitiveData(_) => &slots.primitive,
            _ => {
                return roots.rules.keeps(LinkedUseGap::GatedOffMeshBuiltin)
                    || MeshBuiltin::of_write(op).is_none_or(|b| !slots.gated_off.contains(&b));
            }
        };
        match call.args.get(1) {
            Some(Value::Const(c)) => match self.entry.module.consts().get(*c).map(|k| &k.value) {
                Some(Const::Int(slot)) => kept.contains(slot),
                _ => true,
            },
            _ => true,
        }
    }

    /// The field count of instruction `i`'s struct result (0 for another type).
    fn field_count(&self, i: u32) -> usize {
        self.inst(i)
            .and_then(|inst| struct_fields(self.entry, inst.ty))
            .unwrap_or(0)
    }

    /// The words of instruction `i`'s demanded fields, where it is tracked by field.
    fn tracked(&self, i: u32) -> Option<std::ops::Range<usize>> {
        let s = self.slot.get(i as usize).copied().filter(|&s| s != NONE)? as usize;
        Some(s..s + words_for(self.field_count(i)))
    }

    /// Queue instruction `i`, charging the work list's growth.
    fn queue(&mut self, i: u32, m: &mut Meter) -> Result<(), Error> {
        push(&mut self.work, i, m, &mut self.charged)
    }

    /// Keep instruction `i` wholly: every field of a value tracked by field.
    fn keep(&mut self, i: u32, m: &mut Meter) -> Result<(), Error> {
        if let Some(words) = self.tracked(i) {
            let fields = self.field_count(i);
            let mask: Vec<u64> = (0..words.len())
                .map(|w| match fields - w * 64 {
                    n if n >= 64 => u64::MAX,
                    n => (1u64 << n) - 1,
                })
                .collect();
            return self.demand_fields(i, &mask, m);
        }
        match self.live.get_mut(i as usize) {
            Some(live) if !*live => {
                *live = true;
                self.queue(i, m)
            }
            _ => Ok(()),
        }
    }

    /// OR `mask` into the demanded fields of tracked value `i`, queueing it where they grew.
    fn demand_fields(&mut self, i: u32, mask: &[u64], m: &mut Meter) -> Result<(), Error> {
        let Some(words) = self.tracked(i) else {
            return self.keep(i, m);
        };
        let mut grew = false;
        for (w, &bits) in self.words.get_mut(words).into_iter().flatten().zip(mask) {
            if *w | bits != *w {
                *w |= bits;
                grew = true;
            }
        }
        if !grew {
            return Ok(());
        }
        if let Some(live) = self.live.get_mut(i as usize) {
            *live = true;
        }
        self.queue(i, m)
    }

    /// Demand all of value `v`.
    fn demand(&mut self, v: Value, m: &mut Meter) -> Result<(), Error> {
        match v {
            Value::Arg(a) => {
                if let Some(p) = self.params.get_mut(a.raw() as usize) {
                    *p = true;
                }
                Ok(())
            }
            Value::Inst(i) => self.keep(i.raw(), m),
            Value::Const(_) => Ok(()),
        }
    }

    /// Demand the fields `mask` of value `v` (all of it where it is not tracked by field).
    fn demand_mask(&mut self, v: Value, mask: &[u64], m: &mut Meter) -> Result<(), Error> {
        if mask.iter().all(|&w| w == 0) {
            return Ok(());
        }
        match v {
            Value::Inst(i) => self.demand_fields(i.raw(), mask, m),
            other => self.demand(other, m),
        }
    }

    /// Demand field `k` of value `v`.
    fn demand_field(&mut self, v: Value, k: u32, m: &mut Meter) -> Result<(), Error> {
        let w = k as usize / 64;
        let mask: Vec<u64> = (0..=w)
            .map(|x| if x == w { 1u64 << (k % 64) } else { 0 })
            .collect();
        self.demand_mask(v, &mask, m)
    }

    /// The demanded fields of tracked value `i`.
    fn fields_of(&self, i: u32) -> Vec<u64> {
        self.tracked(i)
            .and_then(|words| self.words.get(words))
            .map(<[u64]>::to_vec)
            .unwrap_or_default()
    }

    /// Keep the terminator of block `b`.
    fn keep_terminator(&mut self, b: u32, m: &mut Meter) -> Result<(), Error> {
        let body: &'a air::ir::Body = self.body;
        match body
            .blocks
            .get(BlockId::new(b))
            .and_then(|block| block.last())
        {
            Some(t) => self.keep(t.raw(), m),
            None => Ok(()),
        }
    }

    /// Keep block `b`: the branches it is control dependent on.
    fn keep_block(&mut self, b: u32, m: &mut Meter) -> Result<(), Error> {
        match self.blocks.get_mut(b as usize) {
            Some(kept) if !*kept => *kept = true,
            _ => return Ok(()),
        }
        let from = self.controls.partition_point(|&(x, _)| x < b);
        let to = self.controls.partition_point(|&(x, _)| x <= b);
        for k in from..to {
            if let Some(&(_, a)) = self.controls.get(k) {
                self.keep_terminator(a, m)?;
            }
        }
        Ok(())
    }

    /// A kept `phi` of block `b` keeps its incoming blocks.
    fn keep_incoming(&mut self, b: u32, m: &mut Meter) -> Result<(), Error> {
        match self.phis.get_mut(b as usize) {
            Some(done) if !*done => *done = true,
            _ => return Ok(()),
        }
        let entry: &'a air::Entry = self.entry;
        for p in entry.cfg.predecessors(BlockId::new(b)) {
            self.keep_block(p.raw(), m)?;
        }
        Ok(())
    }

    /// A kept instruction reads allocation `a`: every write into it is kept.
    fn read_stack(&mut self, a: u32, m: &mut Meter) -> Result<(), Error> {
        match self.read.get_mut(a as usize) {
            Some(read) if !*read => *read = true,
            _ => return Ok(()),
        }
        let from = self.writes.partition_point(|&(x, _)| x < a);
        let to = self.writes.partition_point(|&(x, _)| x <= a);
        for k in from..to {
            if let Some(&(_, w)) = self.writes.get(k) {
                self.keep(w, m)?;
            }
        }
        Ok(())
    }

    /// Process kept instruction `i`: keep its block and what it reads (the module docs).
    fn process(&mut self, i: u32, roots: &Roots, m: &mut Meter) -> Result<(), Error> {
        let inst = self
            .inst(i)
            .ok_or_else(|| invariant("an instruction out of range"))?;
        let b = self.block_of.get(i as usize).copied().unwrap_or(NONE);
        if b == NONE || !self.entry.cfg.is_reachable(BlockId::new(b)) {
            return Ok(());
        }
        self.keep_block(b, m)?;
        let tracked = self.tracked(i).is_some();
        match &inst.kind {
            InstKind::Ret(Some(v)) => match &roots.returned {
                Returned::Nothing => Ok(()),
                Returned::All => self.demand(*v, m),
                Returned::Fields(fields) => {
                    for &k in fields {
                        self.demand_field(*v, k, m)?;
                    }
                    Ok(())
                }
            },
            InstKind::InsertValue { agg, val, indices } if tracked => {
                let mut mask = self.fields_of(i);
                let k = *indices
                    .first()
                    .ok_or_else(|| invariant("an insertvalue without an index"))?;
                let (w, bit) = (k as usize / 64, 1u64 << (k % 64));
                if let Some(x) = mask.get_mut(w).filter(|x| **x & bit != 0) {
                    // The member is demanded; the aggregate's own field `k` is too where the
                    // member is only a part of it.
                    if indices.len() == 1 {
                        *x &= !bit;
                    }
                    self.demand(*val, m)?;
                }
                self.demand_mask(*agg, &mask, m)
            }
            InstKind::Phi { incoming, .. } => {
                self.keep_incoming(b, m)?;
                let mask = self.fields_of(i);
                for &(v, _) in incoming.iter() {
                    match tracked {
                        true => self.demand_mask(v, &mask, m)?,
                        false => self.demand(v, m)?,
                    }
                }
                Ok(())
            }
            InstKind::Select { c, t, f, .. } if tracked => {
                let mask = self.fields_of(i);
                self.demand(*c, m)?;
                self.demand_mask(*t, &mask, m)?;
                self.demand_mask(*f, &mask, m)
            }
            InstKind::Freeze(v) if tracked => {
                let mask = self.fields_of(i);
                self.demand_mask(*v, &mask, m)
            }
            InstKind::ExtractValue { agg, indices } => {
                let field_wise = match *agg {
                    Value::Inst(a) => self.tracked(a.raw()).is_some(),
                    _ => false,
                };
                match indices.first() {
                    Some(&k) if field_wise => self.demand_field(*agg, k, m),
                    _ => self.demand(*agg, m),
                }
            }
            InstKind::Load { ptr, .. } => {
                let a = self.base_of(*ptr);
                if a != NONE && roots.rules.keeps(LinkedUseGap::StackMemory) {
                    self.read_stack(a, m)?;
                }
                self.demand(*ptr, m)
            }
            kind => {
                if let Some((_, Some(src))) = memory_operands(kind) {
                    let a = self.base_of(src);
                    if a != NONE && roots.rules.keeps(LinkedUseGap::StackMemory) {
                        self.read_stack(a, m)?;
                    }
                }
                let mut result = Ok(());
                let mut operands = 0usize;
                kind.for_each_value(|v| {
                    operands += 1;
                    if result.is_ok() {
                        result = self.demand(v, m);
                    }
                });
                result?;
                self.step(m, operands)
            }
        }
    }
    /// [`forward_reach`] itself, the backward analysis' dual under the same roots: `live` marks a
    /// reached instruction, `words` the reached fields of one tracked by field, `blocks` a block
    /// whose branch a reached value decides, `phis` a block whose `phi`s are reached, and `read`
    /// an allocation a reached value is written into.
    fn forward(
        &mut self,
        params: &BTreeSet<u32>,
        roots: &Roots,
        pdt: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<Reach, Error> {
        self.slots(m)?;
        self.stack(m)?;
        self.dependences(pdt, m)?;
        let entry: &'a air::Entry = self.entry;
        let body: &'a air::ir::Body = self.body;
        let cfg = &entry.cfg;
        let insts =
            u32::try_from(body.insts.len()).map_err(|_| invariant("too many instructions"))?;
        // `(value, user)` for each operand: an instruction's value is its index, a parameter's
        // `insts` past it.
        let mut users: Vec<(u32, u32)> = Vec::new();
        // `(allocation, reader)` for each load from, and copy or move out of, the stack.
        let mut reads: Vec<(u32, u32)> = Vec::new();
        for (id, inst) in body.insts.iter() {
            let mut result = Ok(());
            let mut uses = Vec::new();
            inst.kind.for_each_value(|v| match v {
                Value::Inst(i) => uses.push(i.raw()),
                Value::Arg(a) => uses.push(insts.saturating_add(a.raw())),
                Value::Const(_) => {}
            });
            for u in uses {
                if result.is_ok() {
                    result = push(&mut users, (u, id.raw()), m, &mut self.charged);
                }
            }
            result?;
            let source = match &inst.kind {
                InstKind::Load { ptr, .. } => Some(*ptr),
                kind => memory_operands(kind).and_then(|(_, src)| src),
            };
            if let Some(a) = source.map(|p| self.base_of(p)).filter(|&a| a != NONE) {
                push(&mut reads, (a, id.raw()), m, &mut self.charged)?;
            }
            self.step(m, 1)?;
        }
        users.sort_unstable();
        reads.sort_unstable();
        // `(controlling block, block)` for each control dependence.
        let mut dependents: Vec<(u32, u32)> = Vec::new();
        for k in 0..self.controls.len() {
            if let Some(&(b, a)) = self.controls.get(k) {
                push(&mut dependents, (a, b), m, &mut self.charged)?;
            }
        }
        dependents.sort_unstable();
        let endless = cfg.rpo().any(|b| !pdt.is_reachable(b));
        let every_branch = endless && roots.rules.keeps(LinkedUseGap::EndlessLoop);
        let mut work: Vec<u32> = Vec::new();
        for &p in params {
            push(&mut work, insts.saturating_add(p), m, &mut self.charged)?;
        }
        // Blocks whose branch a reached value decides, and allocations a reached value is
        // written into, not followed yet.
        let mut decided: Vec<u32> = Vec::new();
        let mut written: Vec<u32> = Vec::new();
        loop {
            if let Some(a) = written.pop() {
                // Every load from the allocation, and every copy out of it, reads the value.
                let from = reads.partition_point(|&(x, _)| x < a);
                let to = reads.partition_point(|&(x, _)| x <= a);
                for k in from..to {
                    let Some(&(_, r)) = reads.get(k) else { break };
                    self.step(m, 1)?;
                    let kind = &self
                        .inst(r)
                        .ok_or_else(|| invariant("an instruction out of range"))?
                        .kind;
                    if self.is_root(kind, roots) {
                        return Ok(Reach::Kept);
                    }
                    match memory_operands(kind) {
                        Some((dst, _)) => {
                            if self.follow_write(dst, roots, &mut written, m)? {
                                return Ok(Reach::Kept);
                            }
                        }
                        None => {
                            if self.reach_whole(r) {
                                push(&mut work, r, m, &mut self.charged)?;
                            }
                        }
                    }
                }
                continue;
            }
            if let Some(a) = decided.pop() {
                // Every block the branch decides runs or not by it, and every `phi` that
                // chooses by an edge out of the branch's block or one of those.
                let from = dependents.partition_point(|&(x, _)| x < a);
                let to = dependents.partition_point(|&(x, _)| x <= a);
                let mut edges: Vec<u32> = Vec::new();
                push(&mut edges, a, m, &mut self.charged)?;
                for k in from..to {
                    let Some(&(_, b)) = dependents.get(k) else {
                        break;
                    };
                    push(&mut edges, b, m, &mut self.charged)?;
                    let latch = cfg
                        .successors(BlockId::new(b))
                        .any(|s| cfg.is_retreating(BlockId::new(b), s));
                    if latch {
                        return Ok(Reach::Kept);
                    }
                    for (id, inst) in body.block_insts(BlockId::new(b)) {
                        self.step(m, 1)?;
                        let reached = match &inst.kind {
                            InstKind::CondBr { .. } | InstKind::Switch { .. } => {
                                if every_branch {
                                    return Ok(Reach::Kept);
                                }
                                if let Some(done) = self.blocks.get_mut(b as usize) {
                                    if !*done {
                                        *done = true;
                                        push(&mut decided, b, m, &mut self.charged)?;
                                    }
                                }
                                false
                            }
                            // A `ret` or `unreachable` too: the linking keeps every one, and
                            // so every branch that decides whether it runs.
                            kind if self.is_root(kind, roots) => return Ok(Reach::Kept),
                            InstKind::Store { ptr, .. } => {
                                if self.follow_write(*ptr, roots, &mut written, m)? {
                                    return Ok(Reach::Kept);
                                }
                                false
                            }
                            kind => match memory_operands(kind) {
                                Some((dst, _)) => {
                                    if self.follow_write(dst, roots, &mut written, m)? {
                                        return Ok(Reach::Kept);
                                    }
                                    false
                                }
                                None => self.reach_whole(id.raw()),
                            },
                        };
                        if reached {
                            push(&mut work, id.raw(), m, &mut self.charged)?;
                        }
                    }
                }
                for p in edges {
                    for j in cfg.successors(BlockId::new(p)) {
                        match self.phis.get_mut(j.raw() as usize) {
                            Some(done) if !*done => *done = true,
                            _ => continue,
                        }
                        for (id, inst) in body.block_insts(j) {
                            self.step(m, 1)?;
                            if matches!(inst.kind, InstKind::Phi { .. })
                                && self.reach_whole(id.raw())
                            {
                                push(&mut work, id.raw(), m, &mut self.charged)?;
                            }
                        }
                    }
                }
                continue;
            }
            let Some(v) = work.pop() else { break };
            self.step(m, 1)?;
            // The fields `v` carries, where it is tracked by field (a parameter is whole).
            let carried = (v < insts)
                .then(|| self.fields_of(v))
                .filter(|w| !w.is_empty());
            let from = users.partition_point(|&(x, _)| x < v);
            let to = users.partition_point(|&(x, _)| x <= v);
            for k in from..to {
                let Some(&(_, u)) = users.get(k) else {
                    break;
                };
                self.step(m, 1)?;
                let b = self.block_of.get(u as usize).copied().unwrap_or(NONE);
                if b == NONE || !cfg.is_reachable(BlockId::new(b)) {
                    continue;
                }
                let inst = self
                    .inst(u)
                    .ok_or_else(|| invariant("an instruction out of range"))?;
                let operand = |x: &Value| match *x {
                    Value::Inst(i) => i.raw() == v,
                    Value::Arg(a) => insts.saturating_add(a.raw()) == v,
                    Value::Const(_) => false,
                };
                let latch = cfg
                    .successors(BlockId::new(b))
                    .any(|s| cfg.is_retreating(BlockId::new(b), s));
                let grew = match &inst.kind {
                    InstKind::Ret(Some(_)) => {
                        let kept = match &roots.returned {
                            Returned::Nothing => false,
                            Returned::All => true,
                            Returned::Fields(fields) => match &carried {
                                Some(words) => fields.iter().any(|&f| {
                                    words
                                        .get(f as usize / 64)
                                        .is_some_and(|w| w & (1u64 << (f % 64)) != 0)
                                }),
                                None => !fields.is_empty(),
                            },
                        };
                        if kept {
                            return Ok(Reach::Kept);
                        }
                        false
                    }
                    // A loop's latch is kept whatever it decides (the module docs).
                    InstKind::CondBr { .. } | InstKind::Switch { .. } if latch || every_branch => {
                        return Ok(Reach::Kept);
                    }
                    InstKind::CondBr { .. } | InstKind::Switch { .. } => {
                        if let Some(done) = self.blocks.get_mut(b as usize) {
                            if !*done {
                                *done = true;
                                push(&mut decided, b, m, &mut self.charged)?;
                            }
                        }
                        false
                    }
                    kind if self.is_root(kind, roots) => return Ok(Reach::Kept),
                    InstKind::Store { ptr, .. } => {
                        if self.follow_write(*ptr, roots, &mut written, m)? {
                            return Ok(Reach::Kept);
                        }
                        false
                    }
                    InstKind::InsertValue { agg, val, indices } if self.tracked(u).is_some() => {
                        let mut mask = vec![0u64; self.fields_of(u).len()];
                        let first = indices.first().copied();
                        if operand(agg) {
                            match &carried {
                                Some(words) => {
                                    for (x, w) in mask.iter_mut().zip(words) {
                                        *x |= w;
                                    }
                                }
                                None => mask.iter_mut().for_each(|x| *x = u64::MAX),
                            }
                            // A member that replaces its whole field replaces what it held.
                            if let (1, Some(f)) = (indices.len(), first) {
                                if let Some(x) = mask.get_mut(f as usize / 64) {
                                    *x &= !(1u64 << (f % 64));
                                }
                            }
                        }
                        if let (true, Some(f)) = (operand(val), first) {
                            if let Some(x) = mask.get_mut(f as usize / 64) {
                                *x |= 1u64 << (f % 64);
                            }
                        }
                        self.reach_fields(u, &mask)
                    }
                    InstKind::ExtractValue { agg, indices } => {
                        let reached = match (&carried, indices.first()) {
                            (Some(words), Some(&f)) if operand(agg) => words
                                .get(f as usize / 64)
                                .is_some_and(|w| w & (1u64 << (f % 64)) != 0),
                            _ => true,
                        };
                        reached && self.reach_whole(u)
                    }
                    InstKind::Phi { .. } | InstKind::Freeze(_) if self.tracked(u).is_some() => {
                        match &carried {
                            Some(words) => {
                                let words = words.clone();
                                self.reach_fields(u, &words)
                            }
                            None => self.reach_whole(u),
                        }
                    }
                    InstKind::Select { c, .. } if self.tracked(u).is_some() && !operand(c) => {
                        match &carried {
                            Some(words) => {
                                let words = words.clone();
                                self.reach_fields(u, &words)
                            }
                            None => self.reach_whole(u),
                        }
                    }
                    kind => match memory_operands(kind) {
                        // A set, copy or move into the stack, whichever operand is reached.
                        Some((dst, _)) => {
                            if self.follow_write(dst, roots, &mut written, m)? {
                                return Ok(Reach::Kept);
                            }
                            false
                        }
                        None => self.reach_whole(u),
                    },
                };
                if grew {
                    push(&mut work, u, m, &mut self.charged)?;
                }
            }
        }
        m.tick(self.units)?;
        self.units = 0;
        Ok(Reach::Nothing)
    }

    /// A reached value is written through stack pointer `ptr` by an instruction that is not a
    /// root: whether that write is kept (`StackMemory` keeps every write into an allocation whose
    /// address goes elsewhere); else the allocation is queued on `written` where it is new, so the
    /// loads from it, and the copies out of it, are followed. Without `StackMemory` no load keeps
    /// a write into the stack, and nothing is followed.
    fn follow_write(
        &mut self,
        ptr: Value,
        roots: &Roots,
        written: &mut Vec<u32>,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        if !roots.rules.keeps(LinkedUseGap::StackMemory) {
            return Ok(false);
        }
        let a = self.base_of(ptr);
        if a == NONE {
            return Err(invariant("a write that is not a root outside the stack"));
        }
        if self.escaped.get(a as usize).copied().unwrap_or(true) {
            return Ok(true);
        }
        match self.read.get_mut(a as usize) {
            Some(done) if !*done => *done = true,
            _ => return Ok(false),
        }
        push(written, a, m, &mut self.charged)?;
        Ok(false)
    }

    /// Mark instruction `i` reached wholly; whether that is new.
    fn reach_whole(&mut self, i: u32) -> bool {
        if let Some(words) = self.tracked(i) {
            let fields = self.field_count(i);
            let mask: Vec<u64> = (0..words.len())
                .map(|w| match fields - w * 64 {
                    n if n >= 64 => u64::MAX,
                    n => (1u64 << n) - 1,
                })
                .collect();
            return self.reach_fields(i, &mask);
        }
        match self.live.get_mut(i as usize) {
            Some(live) if !*live => {
                *live = true;
                true
            }
            _ => false,
        }
    }

    /// OR `mask` into the reached fields of tracked instruction `i`; whether they grew.
    fn reach_fields(&mut self, i: u32, mask: &[u64]) -> bool {
        let Some(words) = self.tracked(i) else {
            return self.reach_whole(i);
        };
        let mut grew = false;
        for (w, &bits) in self.words.get_mut(words).into_iter().flatten().zip(mask) {
            if *w | bits != *w {
                *w |= bits;
                grew = true;
            }
        }
        if grew {
            if let Some(live) = self.live.get_mut(i as usize) {
                *live = true;
            }
        }
        grew
    }
}
